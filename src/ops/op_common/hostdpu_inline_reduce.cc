/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hostdpu_inline_reduce.h"
#include "comm_negotiation.h"
#include "hcomm_primitives_dl.h"
#include "log.h"

namespace ops_hccl {
// IsStreamInCaptureMode 声明见 op_common.h（定义于 op_common.cc）；此处前向声明以避免
// 引入 op_common.h 的重依赖（executor/selector/ccu 等模块头），UT 由桩提供定义
bool IsStreamInCaptureMode(aclrtStream stream);

namespace {
    // inline reduce 能力缓存三态值（缓存值含义 + 缓存条目是否存在，组合即可溯源完整状态）：
    // 条目不存在 = 未协商过；条目存在 = 已协商，取值区分"协商结果"与"协商失败降级"
    enum class InlineReduceCap : int8_t {
        ALL_RANKS_SUPPORTED = 0,   // 协商成功，全域支持，可启用 inline reduce
        SOME_RANK_UNSUPPORTED = 1, // 协商成功，存在不支持的rank，全域禁用
        NEGOTIATION_DEGRADED = 2,  // 协商失败，按不支持降级（自愈手段=重建通信域）
    };
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
    void* capPtr = nullptr;
    uint64_t capSize = 0;
    HcclResult getRet = HcclEngineCtxGet(comm, hostDpuInlineCapTag, CommEngine::COMM_ENGINE_CPU_TS, &capPtr, &capSize);
    if (getRet == HCCL_SUCCESS && capPtr != nullptr && capSize >= sizeof(InlineReduceCap)) {
        // per-comm能力缓存命中（能力为comm级静态属性，首op协商一次，命中后零开销）；
        // 三态值还原：仅 ALL_RANKS_SUPPORTED 视为支持，SOME_RANK_UNSUPPORTED/NEGOTIATION_DEGRADED 一律不支持
        InlineReduceCap cachedCap = *(static_cast<InlineReduceCap*>(capPtr));
        topoInfo->allDpuSupportInlineReduce = (cachedCap == InlineReduceCap::ALL_RANKS_SUPPORTED);
        HCCL_INFO(
            "[NegotiateHostDpuInlineReduce] cache hit, cap state[%d], allDpuSupportInlineReduce[%d].",
            static_cast<int>(cachedCap), static_cast<int>(topoInfo->allDpuSupportInlineReduce));
        return HCCL_SUCCESS;
    }

    // 门控5：图模式捕获态跳过协商且不写终态缓存——录制期执行全域集合通信+同步会把"录制可各rank
    // 异步推进"变成全域同步点，且协商失败会写NEGOTIATION_DEGRADED终态缓存（本comm生命周期不可
    // 自愈）；能力位保持默认false，replay后的非捕获态op缓存miss时正常协商
    if (IsStreamInCaptureMode(opStream)) {
        return HCCL_SUCCESS;
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
    HcclResult negRet
        = NegotiateAllReduceMin(comm, "hostdpu_inline", localSupportInlineReduce, allRanksSupportInlineReduce);
    if (negRet != HCCL_SUCCESS) {
        HCCL_WARNING("[NegotiateHostDpuInlineReduce] negotiation failed, ret[%d], fallback to local reduce.", negRet);
        allRanksSupportInlineReduce = false;
        capState = InlineReduceCap::NEGOTIATION_DEGRADED;
    } else {
        capState = allRanksSupportInlineReduce ? InlineReduceCap::ALL_RANKS_SUPPORTED :
                                                 InlineReduceCap::SOME_RANK_UNSUPPORTED;
    }

    // 写per-comm终态缓存；失败不影响本次op（topoInfo字段已直接设置，仅损失缓存收益）。
    // 写前二次Get（首轮Get在协商前，其结果已过期）：
    // 1) 本端NEGOTIATION_DEGRADED多为同comm并发算子命中缓存miss窗口被CommWorker拒绝的假象
    //    （同comm跨rank集合操作须全rank同序，拒绝是正确防御），他方已写入ALL_RANKS_SUPPORTED
    //    时直接采纳（能力为comm级静态属性且AllReduce-MIN全域一致，正确协商结果必然相同），
    //    避免假失败终态固化导致该comm生命周期内inline reduce永久降级；
    // 2) 存在旧条目（含首轮Get发现的不合规条目：指针空或尺寸不足）：HCOMM语义为同名tag已存在
    //    即拒绝创建，须先销毁再写
    void* curPtr = nullptr;
    uint64_t curSize = 0;
    HcclResult curGetRet
        = HcclEngineCtxGet(comm, hostDpuInlineCapTag, CommEngine::COMM_ENGINE_CPU_TS, &curPtr, &curSize);
    if (curGetRet == HCCL_SUCCESS) {
        if (curPtr != nullptr && curSize >= sizeof(InlineReduceCap)
            && *(static_cast<InlineReduceCap*>(curPtr)) == InlineReduceCap::ALL_RANKS_SUPPORTED
            && capState == InlineReduceCap::NEGOTIATION_DEGRADED) {
            HCCL_INFO("[NegotiateHostDpuInlineReduce] concurrent op already cached SUPPORTED, adopt it, "
                      "local negotiation may be concurrent-reject fallback.");
            topoInfo->allDpuSupportInlineReduce = true;
            return HCCL_SUCCESS;
        }
        (void)HcclEngineCtxDestroy(comm, hostDpuInlineCapTag, CommEngine::COMM_ENGINE_CPU_TS);
    }
    void* newCapPtr = nullptr;
    HcclResult createRet = HcclEngineCtxCreate(
        comm, hostDpuInlineCapTag, CommEngine::COMM_ENGINE_CPU_TS, sizeof(InlineReduceCap), &newCapPtr);
    if (createRet == HCCL_SUCCESS && newCapPtr != nullptr) {
        *(static_cast<InlineReduceCap*>(newCapPtr)) = capState;
    } else if (createRet == HCCL_E_PARA) {
        HCCL_INFO("[NegotiateHostDpuInlineReduce] cache already created by concurrent op, skip caching.");
    } else {
        HCCL_WARNING("[NegotiateHostDpuInlineReduce] cache create failed, ret[%d], skip caching.", createRet);
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
