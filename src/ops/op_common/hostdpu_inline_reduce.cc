/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <functional>
#include <mutex>
#include "hostdpu_inline_reduce.h"
#include "comm_negotiation.h" // IsNegotiationComm（C++本体，防递归门控判定）
#include "comm_negotiation_c.h"
#include "hcomm_primitives_dl.h"
#include "log.h"

namespace ops_hccl {
// IsStreamInCaptureMode 声明见 op_common.h（定义于 op_common.cc）；此处前向声明以避免
// 引入 op_common.h 的重依赖（executor/selector/ccu 等模块头），UT 由桩提供定义
bool IsStreamInCaptureMode(aclrtStream stream);

namespace {
    // inline reduce 能力缓存三态值（缓存值含义 + 缓存条目是否存在，组合即可溯源完整状态）：
    // 条目不存在 = 未协商过；条目存在 = 已协商，取值区分"协商结果"与"协商失败降级"。
    // 取值排布 fail-safe：HCOMM HcclEngineCtxCreate 底层为 calloc 清零块，Create 成功到首次写值
    // 之间存在清零窗口，0 必须是不支持语义（DEGRADED），防清零默认值被误读为全域支持（假启用）
    enum class InlineReduceCap : int8_t {
        NEGOTIATION_DEGRADED = 0,  // 协商失败，按不支持降级（自愈手段=重建通信域）
        SOME_RANK_UNSUPPORTED = 1, // 协商成功，存在不支持的rank，全域禁用
        ALL_RANKS_SUPPORTED = 2,   // 协商成功，全域支持，可启用 inline reduce
    };

    // per-comm能力缓存读写互斥（分片锁，按comm指针哈希取片）：读命中段与写段（含原地写/
    // 重建）整体纳入同一临界区，消除两类竞态：写者 Destroy 释放条目期间读者解引用已释放
    // 指针（use-after-free）；读者在写者 Create 成功到写值完成的窗口内读到清零默认值。
    // 配合写路径"合规条目原地写、不 Destroy"（合规块一旦存在永不被释放），读者持有的指针
    // 永远有效。分片与comm生命周期无绑定，无销毁清理问题；锁内仅本地EngineCtx操作，无跨rank
    // 阻塞（协商AllReduce在锁外）。残余风险：并发op的首轮miss判定与协商集合在锁外，其他rank
    // 缓存命中后不再进集合时，迟到的并发协商仍会等待至HCCL_EXEC_TIMEOUT超时后降级（一次性
    // 代价，非永久分叉；永久分叉路径已由缓存写失败上抛消除）
    constexpr uint32_t CAP_CACHE_LOCK_SHARDS = 16;
    std::mutex g_capCacheLockShards[CAP_CACHE_LOCK_SHARDS];
} // namespace

HcclResult NegotiateHostDpuInlineReduceCapability(
    HcclComm comm, HcclCMDType opType, aclrtStream opStream, TopoInfoWithNetLayerDetails* topoInfo)
{
    // 门控1：P2P算子不参与全rank集合协商——P2P语义仅要求配对rank调用（域内其余rank可不进入
    // Selector），全域协商AllReduce会因对端永不到达而挂死；且P2P路径不消费allDpuSupportInlineReduce，
    // 首个集合算子到达时缓存miss自然触发协商
    if (opType == HcclCMDType::HCCL_CMD_SEND || opType == HcclCMDType::HCCL_CMD_RECEIVE
        || opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        return HCCL_SUCCESS;
    }
    // 门控2/3/4：非hostDPU组网 / 非A5(950)及后续代际 / 协商子通信域（防递归主防线）
    if (!topoInfo->hostDpuOnly || !EqualToOrGreaterThan950(topoInfo->deviceType) || IsNegotiationComm(comm)) {
        return HCCL_SUCCESS;
    }

    constexpr char hostDpuInlineCapTag[] = "hostdpu_inline_reduce_cap";
    uint32_t lockShard = static_cast<uint32_t>(std::hash<void*>{}(comm)) % CAP_CACHE_LOCK_SHARDS;
    {
        // 读命中段与写段共持同一分片锁：防写者 Destroy 释放/重建期间解引用悬垂指针，
        // 防在写者 Create~写值窗口读到清零默认值（读命中零协商开销，锁内仅本地Get+解引用）
        std::lock_guard<std::mutex> cacheLock(g_capCacheLockShards[lockShard]);
        void* capPtr = nullptr;
        uint64_t capSize = 0;
        HcclResult getRet
            = HcclEngineCtxGet(comm, hostDpuInlineCapTag, CommEngine::COMM_ENGINE_CPU_TS, &capPtr, &capSize);
        if (getRet == HCCL_SUCCESS && capPtr != nullptr && capSize >= sizeof(InlineReduceCap)) {
            // per-comm能力缓存命中（能力为comm级静态属性，首op协商一次，命中后零开销）；
            // 三态值还原：仅 ALL_RANKS_SUPPORTED 视为支持，SOME_RANK_UNSUPPORTED/NEGOTIATION_DEGRADED
            // 一律不支持
            InlineReduceCap cachedCap = *(static_cast<InlineReduceCap*>(capPtr));
            topoInfo->allDpuSupportInlineReduce = (cachedCap == InlineReduceCap::ALL_RANKS_SUPPORTED);
            HCCL_INFO(
                "[NegotiateHostDpuInlineReduce] cache hit, cap state[%d], allDpuSupportInlineReduce[%d].",
                static_cast<int>(cachedCap), static_cast<int>(topoInfo->allDpuSupportInlineReduce));
            return HCCL_SUCCESS;
        }
    }

    // 本端能力查询（本端语义，无跨rank交换）：以CalcHostDPUOnly落存的host侧代表端点查询per-endpoint特性;
    // 符号缺失/查询失败一律按不支持降级，禁止上抛；HcommEndpointCheckFeature 为 9.1.0+ 符号，低版本CANN包按不支持处理
    bool localSupportInlineReduce = false;
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    HcclResult localRet = HcclCheckDpuInlineReduceLocal(topoInfo->hostDpuEndpoint, &localSupportInlineReduce);
    if (localRet != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[NegotiateHostDpuInlineReduce] local check failed, ret[%d], fallback to not supported.", localRet);
        localSupportInlineReduce = false;
    }
#endif

    // 全域聚合（公共协商机制，AllReduce-MIN=全域AND）：任一hostDPU不支持即不支持；
    // 协商失败按不支持降级并写终态缓存（避免逐op重试失败的子通信域建链+AllReduce；自愈手段=重建通信域）
    bool allRanksSupportInlineReduce = false;
    InlineReduceCap capState = InlineReduceCap::SOME_RANK_UNSUPPORTED;
    HcclResult negRet = NegotiateCommC(
        comm, "hostdpu_inline", NEGOTIATE_OP_ALL_REDUCE_MIN, &localSupportInlineReduce, sizeof(bool),
        &allRanksSupportInlineReduce, sizeof(bool));
    if (negRet != HCCL_SUCCESS) {
        HCCL_WARNING("[NegotiateHostDpuInlineReduce] negotiation failed, ret[%d], fallback to local reduce.", negRet);
        allRanksSupportInlineReduce = false;
        capState = InlineReduceCap::NEGOTIATION_DEGRADED;
    } else {
        capState = allRanksSupportInlineReduce ? InlineReduceCap::ALL_RANKS_SUPPORTED :
                                                 InlineReduceCap::SOME_RANK_UNSUPPORTED;
    }

    // 写per-comm终态缓存；topoInfo字段已直接设置。HCCL_E_PARA以外的写失败上抛快速失败：
    // 本rank缓存条目缺失而其它rank已缓存命中直接return时，后续每个算子都会重入协商集合，
    // 而其它rank永不再到达——集合成员永久分叉，逐算子等满HCCL_EXEC_TIMEOUT（默认1836s）
    // 超时；显式失败本次算子优于延迟到后续每个算子挂超时。
    // 图模式捕获态（opStream处于capture）跳过写入、协商照常参与（捕获态不是跳过协商的门控——
    // capture是per-rank本地状态，混跑时部分rank跳过会使全域AllReduce集合成员分叉导致对端挂死）：
    // 录制期各rank异步推进，CommWorker并发拒绝的假失败率高于稳态，NEGOTIATION_DEGRADED假失败
    // 终态会连累同comm的稳态算子（缓存per-comm共享、生命周期不可自愈）；捕获态每算子重复协商
    // 是可接受的录制期代价；缓存命中读取不受捕获态影响，replay后非捕获态op缓存miss时正常缓存。
    // 写前二次Get（首轮Get在协商前，其结果已过期）与写值规则：
    // 1) 本端NEGOTIATION_DEGRADED多为同comm并发算子命中缓存miss窗口被CommWorker拒绝的假象
    //    （同comm跨rank集合操作须全rank同序，拒绝是正确防御），他方已写入ALL_RANKS_SUPPORTED
    //    时直接采纳（能力为comm级静态属性且AllReduce-MIN全域一致，正确协商结果必然相同），
    //    避免假失败终态固化导致该comm生命周期内inline reduce永久降级；
    // 2) 命中合规条目（指针非空且尺寸足）原地写覆盖，不Destroy重建：合规块一旦存在永不被释放，
    //    锁外曾拿到该条目指针的读者永远安全（use-after-free结构性根除），且无Destroy~Create
    //    中间态窗口（条目不存在→并发首协商竞争）；
    // 3) 仅不合规残留条目（Get成功但指针空/尺寸不足）才Destroy后重建：HCOMM语义同名tag已存在
    //    即拒绝创建（HCCL_E_PARA），须先销毁再建。写段整体持分片锁，E_PARA分支仅作防御保留
    if (!IsStreamInCaptureMode(opStream)) {
        std::lock_guard<std::mutex> cacheLock(g_capCacheLockShards[lockShard]);
        void* curPtr = nullptr;
        uint64_t curSize = 0;
        HcclResult curGetRet
            = HcclEngineCtxGet(comm, hostDpuInlineCapTag, CommEngine::COMM_ENGINE_CPU_TS, &curPtr, &curSize);
        if (curGetRet == HCCL_SUCCESS && curPtr != nullptr && curSize >= sizeof(InlineReduceCap)) {
            if (*(static_cast<InlineReduceCap*>(curPtr)) == InlineReduceCap::ALL_RANKS_SUPPORTED
                && capState == InlineReduceCap::NEGOTIATION_DEGRADED) {
                HCCL_INFO("[NegotiateHostDpuInlineReduce] concurrent op already cached SUPPORTED, adopt it, "
                          "local negotiation may be concurrent-reject fallback.");
                topoInfo->allDpuSupportInlineReduce = true;
                return HCCL_SUCCESS;
            }
            // 原地写：迟到成功覆盖假失败终态（LateSuccess）/幂等重写同值，均不重建条目
            *(static_cast<InlineReduceCap*>(curPtr)) = capState;
        } else {
            // 无条目或残留不合规条目（指针空/尺寸不足）：后者须先销毁再建
            if (curGetRet == HCCL_SUCCESS) {
                (void)HcclEngineCtxDestroy(comm, hostDpuInlineCapTag, CommEngine::COMM_ENGINE_CPU_TS);
            }
            void* newCapPtr = nullptr;
            HcclResult createRet = HcclEngineCtxCreate(
                comm, hostDpuInlineCapTag, CommEngine::COMM_ENGINE_CPU_TS, sizeof(InlineReduceCap), &newCapPtr);
            if (createRet == HCCL_SUCCESS && newCapPtr != nullptr) {
                *(static_cast<InlineReduceCap*>(newCapPtr)) = capState;
            } else if (createRet == HCCL_E_PARA) {
                // 写段持锁后同comm写者已串行，本分支仅作跨分片哈希碰撞等异常时序的防御
                HCCL_INFO("[NegotiateHostDpuInlineReduce] cache already created by concurrent op, skip caching.");
            } else {
                // HCCL_SUCCESS+空指针为病态组合，统一按内部错误上抛（返回SUCCESS会静默跳过缓存写）
                HcclResult failRet = (createRet == HCCL_SUCCESS) ? HCCL_E_INTERNAL : createRet;
                HCCL_WARNING("[NegotiateHostDpuInlineReduce] cache create failed, ret[%d].", failRet);
                return failRet;
            }
        }
    }
    topoInfo->allDpuSupportInlineReduce = allRanksSupportInlineReduce;
    HCCL_INFO(
        "[NegotiateHostDpuInlineReduce] done, localSupportInlineReduce[%d], allRanksSupportInlineReduce[%d], "
        "cap state[%d].",
        static_cast<int>(localSupportInlineReduce), static_cast<int>(allRanksSupportInlineReduce),
        static_cast<int>(capState));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
