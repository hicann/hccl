/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_custom_op.h"
#include "hccl_op_utils.h"

#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <string>

#include "hccl/hcom.h"
#include "hccl/hccl_res.h"
#include "acl/acl_rt.h"
#include "log.h"
#include "runtime/rt_external_model.h"
#include "runtime/rt_external_stream.h"

// ---- RTS 外部流注册接口弱声明 (旧 runtime 无此符号) ----
// RTS 头的普通声明在前, 此处弱重声明把本 TU 的调用引用降为 WEAK UND (GCC 允许
// 后置弱化); 旧运行时符号缺失时弱引用解析为 null。C 接口无 mangling, 弱重声明
// 即可; 在位性探测与注册门控共用此判据, 通过则 Execute 侧调用必命中真实实现
extern "C" rtError_t rtGetModelByStream(rtStream_t stm, rtModel_t* model) __attribute__((weak));
extern "C" rtError_t rtModelAddHcclStream(rtStream_t stm, rtModel_t mdl) __attribute__((weak));

namespace hccl {
// 辅流建流 flags (RequestAttachedStream 支持算子指定 RTS 流标志):
// 显式对齐旧 hcom 系流语义 (hccl_task_info.cc 同款 PERSISTENT|FORCE_COPY), 仅新建时生效
constexpr uint32_t HCCL_GE_ATTACHED_STREAM_FLAGS = RT_STREAM_PERSISTENT | RT_STREAM_FORCE_COPY;

bool IsCustomOpTargetSoc()
{
    // 判定结果缓存 (C++11 magic static): 12 个注册点各自调用, HAL 查询与日志只发生一次
    static const bool isTarget = []() {
        // HAL 层查询, 不依赖 ACL 运行时与设备上下文 (dlopen 静态初始化期可用):
        // halGetSocVersion(deviceId, buf, bufLen) 直接读驱动侧 SoC 版本串;
        // 句柄不 dlclose — 注册对象进程存续, hal 保持加载
        void* hal = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
        if (hal == nullptr) {
            HCCL_WARNING("[Hccl][GeCustomOp][Register] dlopen libascend_hal failed, not register ge custom op.");
            return false;
        }
        using HalGetSocVersionFunc = int32_t (*)(uint32_t, char*, uint32_t);
        auto func = reinterpret_cast<HalGetSocVersionFunc>(dlsym(hal, "halGetSocVersion"));
        char socVersion[64] = {0};
        if (func == nullptr || func(0, socVersion, sizeof(socVersion) - 1U) != 0) {
            HCCL_WARNING("[Hccl][GeCustomOp][Register] halGetSocVersion failed, not register ge custom op.");
            return false;
        }
        const std::string soc(socVersion);
        const bool matched = (soc.find("950") != std::string::npos) || (soc.find("960") != std::string::npos);
        // 匹配时不下结论 (接口在位判定随后收口); 不匹配时结论明确: 不注册, 回退原引擎
        HCCL_INFO(
            "[Hccl][GeCustomOp][Register] soc version: %s, match 950/960: %d%s.", soc.c_str(),
            static_cast<int>(matched), matched ? "" : ", skip registration, fallback to hccl_engine");
        return matched;
    }();
    return isTarget;
}

bool HcclGeRuntimeInterfaceAvailable()
{
    // 判定结果缓存 (C++11 magic static): 12 个注册点各自调用, 判空与日志只发生一次。
    // 三重判空堵双向版本错配: GetContextValue 代表 GE 同批新接口 (含五参构造器,
    // GE 头侧声明 weak 后为真实运行期判空, 未声明 weak 时编译期折叠为真 — 此时
    // 本 .so 仅能加载于接口齐备的运行时, 折叠结果仍正确); RTS 两个为 Execute 侧
    // 直接调用的接口 (本文件弱重声明)
    static const bool available = []() {
        const bool present = (&gert::EagerOpExecutionContext::GetContextValue != nullptr)
                             && (&rtGetModelByStream != nullptr) && (&rtModelAddHcclStream != nullptr);
        HCCL_INFO(
            "[Hccl][GeCustomOp][Register] ge/rts custom-op interfaces %s.",
            present ? "available, register ge custom op" : "unavailable, skip registration, fallback to hccl_engine");
        return present;
    }();
    return available;
}

ge::CustomOpCreatorRegister* CreateSocGatedRegister(const char* opType, ge::CustomOpCreateFunc creator)
{
    if (!IsCustomOpTargetSoc()) {
        return nullptr;
    }
    if (!HcclGeRuntimeInterfaceAvailable()) {
        return nullptr;
    }
    // 5 元接管注册 (kBottom 认领 kHccl 引擎槽位), 与 GE 认领链按优先级回退解析配套。
    // 在位性门控保证此处命中真实构造器
    return new ge::CustomOpCreatorRegister(
        ge::AscendString(opType), ge::OpBackend::kDevice, ge::OpRegistrationPriority::kBottom, ge::OpEngine::kHccl,
        creator);
}

// GetOrCreateIn/OutCCLbuf 公共实现:
// 获取→空则创建→重查→创建后仍空报错, In/Out 仅差 hcom 获取函数名
static ge::graphStatus GetOrCreateCCLbufImpl(
    const char* group, const char* role, HcclResult (*getCCLbuf)(const char*, void**, u64*), void*& ptr, uint64_t& size)
{
    u64 bufSize = 0;
    HcclResult ret = getCCLbuf(group, &ptr, &bufSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCustomOpBase][GetOrCreate%sCCLbuf] getCCLbuf failed for group '%s', ret=%d.", role, group,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    if (ptr == nullptr || bufSize == 0) {
        ret = HcomCreateCommCCLbuffer(group);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcclCustomOpBase][GetOrCreate%sCCLbuf] HcomCreateCommCCLbuffer failed for group '%s', ret=%d.", role,
                group, static_cast<int>(ret));
            return ge::GRAPH_FAILED;
        }
        ret = getCCLbuf(group, &ptr, &bufSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcclCustomOpBase][GetOrCreate%sCCLbuf] getCCLbuf retry failed for group '%s', ret=%d.", role, group,
                static_cast<int>(ret));
            return ge::GRAPH_FAILED;
        }
        // 创建成功后 buffer 仍为空属异常: 不带空缓冲返回成功, 让调用方在错误路径处理
        if (ptr == nullptr || bufSize == 0) {
            HCCL_ERROR(
                "[HcclCustomOpBase][GetOrCreate%sCCLbuf] buffer still empty after create, group '%s'.", role, group);
            return ge::GRAPH_FAILED;
        }
    }
    size = bufSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcclCustomOpBase::GetOrCreateInCCLbuf(const char* group, void*& ptr, uint64_t& size)
{
    return GetOrCreateCCLbufImpl(group, "In", HcomGetInCCLbuffer, ptr, size);
}

ge::graphStatus HcclCustomOpBase::GetOrCreateOutCCLbuf(const char* group, void*& ptr, uint64_t& size)
{
    return GetOrCreateCCLbufImpl(group, "Out", HcomGetOutCCLbuffer, ptr, size);
}

// aivCoreLimit 三级回退解析 (对照 InfoStore GetAivCoreLimit, hcom_op_utils.cc 的回退链):
//   ① _op_vectorcore_num — 经 GetContextValue(kVectorCoreNum) 逐节点可达
//     (V1 由 CustomTaskInfo 从 OpDesc 属性注入, 属性缺失时返回 false);
//   ② ge.hardwareInfo 的 vector_core_cnt (session option, ctx->GetOption);
//   ③ 缺省 56 (MAX_NUM_BLOCKS_DEFAULT)。
static uint32_t ResolveAivCoreLimitUncached(gert::EagerOpExecutionContext* ctx)
{
    uint64_t vectorCoreNum = 0;
    if (ctx->GetContextValue(gert::ContextParam::kVectorCoreNum, vectorCoreNum) && vectorCoreNum > 0
        && vectorCoreNum <= UINT32_MAX) {
        const uint32_t aivCoreLimit = static_cast<uint32_t>(vectorCoreNum);
        HCCL_INFO(
            "[CTXDBG][ResolveAivCoreLimit] aivCoreLimit: kVectorCoreNum (_op_vectorcore_num) -> %u.", aivCoreLimit);
        return aivCoreLimit;
    }
    ge::AscendString optValue;
    // GetOption 返回 graphStatus (成功=0), 不得按布尔使用 — 原写法在选项在位时恒 false,
    // hardwareInfo 回退分支永不进入
    if (ctx->GetOption(ge::AscendString("ge.hardwareInfo"), optValue) == ge::GRAPH_SUCCESS
        && optValue.GetLength() > 0) {
        const uint32_t parsed = ParseVectorCoreCntFromHardwareInfo(optValue.GetString());
        if (parsed > 0) {
            HCCL_INFO("[CTXDBG][ResolveAivCoreLimit] aivCoreLimit: hardwareInfo vector_core_cnt -> %u.", parsed);
            return parsed;
        }
    }
    HCCL_INFO(
        "[CTXDBG][ResolveAivCoreLimit] aivCoreLimit: no override found, using default %u.", MAX_NUM_BLOCKS_DEFAULT);
    return MAX_NUM_BLOCKS_DEFAULT;
}

// aivCoreLimit 解析入口: 结果按 (graphId, nodeId) 缓存 (节点属性是编译期常量, 缓存不会过期) —
// 动态图路径逐执行重复调 Execute, 命中即跳过解析; 查/写各持一次短锁,
// 解析本身 (GetOption + 字符串解析) 不持锁, 不阻塞其他节点的并发执行。
uint32_t HcclCustomOpBase::ResolveAivCoreLimit(gert::EagerOpExecutionContext* ctx)
{
    uint64_t nodeId = 0;
    uint64_t graphId = 0;
    const bool cacheable = ctx->GetContextValue(gert::ContextParam::kNodeId, nodeId)
                           && ctx->GetContextValue(gert::ContextParam::kGraphId, graphId);
    if (cacheable) {
        std::lock_guard<std::mutex> lock(aivCoreLimitCacheMutex_);
        const auto it = aivCoreLimitCache_.find(std::make_pair(graphId, nodeId));
        if (it != aivCoreLimitCache_.end()) {
            return it->second;
        }
    }
    const uint32_t aivCoreLimit = ResolveAivCoreLimitUncached(ctx);
    if (cacheable) {
        std::lock_guard<std::mutex> lock(aivCoreLimitCacheMutex_);
        aivCoreLimitCache_[std::make_pair(graphId, nodeId)] = aivCoreLimit;
        HCCL_INFO(
            "[CTXDBG][ResolveAivCoreLimit] aivCoreLimit: first resolve cached (graphId=%u nodeId=%u) -> %u.",
            static_cast<uint32_t>(graphId), static_cast<uint32_t>(nodeId), aivCoreLimit);
    } else {
        // kNodeId/kGraphId 任一取不到 (运行时未注入): 无法缓存, 每次全量解析 — 与 cache hit
        // 日志共同构成注入是否生效的判据 (两者皆无 = 注入失败)
        HCCL_INFO("[CTXDBG][ResolveAivCoreLimit] aivCoreLimit: kNodeId/kGraphId unavailable, cache disabled.");
    }
    return aivCoreLimit;
}

ge::graphStatus HcclCustomOpBase::RequestFollowStreams(HcclOpState& st)
{
    // ifAiv 或 streamNum==0 不申请 — followStreams 保持空, GraphMode streams 传空 (与旧路径行为一致)
    if (st.ifAiv || st.streamNum == 0) {
        st.followStreams.clear();
        return ge::GRAPH_SUCCESS;
    }

    // 主流标识 = rtStreamId (GE 同款用法 davinci_model.cc; [STREAMDBG] key 与 GE 加载日志
    // "Logical stream index: %u, rtstream: %d" 的 rtstream 字段可直接对账)
    int32_t rtStreamId = -1;
    aclError aclRet = aclrtStreamGetId(static_cast<aclrtStream>(st.stream), &rtStreamId);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCustomOpBase][RequestFollowStreams] aclrtStreamGetId failed, opType='%s', ret=%d.", st.opType,
            static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }

    // 建流 flags 显式指定 (不继承模型 0 号逻辑流); HEAD 绑定自动启动;
    // 主辅流同步/激活不在算子侧 (HCCL 消费侧负责)。
    st.followStreams.clear();
    st.followStreams.reserve(st.streamNum);
    for (uint32_t i = 0; i < st.streamNum; i++) {
        const std::string key = "hccl_follow#" + std::to_string(rtStreamId) + "#" + std::to_string(i);
        gert::rtStream stream
            = st.ctx->RequestAttachedStream(ge::AscendString(key.c_str()), HCCL_GE_ATTACHED_STREAM_FLAGS);
        if (stream == nullptr) {
            // GE 运行时未注入辅流 provider 时亦至此 (不支持 RequestAttachedStream 的运行时)
            HCCL_ERROR(
                "[HcclCustomOpBase][RequestFollowStreams] RequestAttachedStream failed, key='%s', opType='%s'.",
                key.c_str(), st.opType);
            st.followStreams.clear();
            return ge::GRAPH_FAILED;
        }
        st.followStreams.push_back(stream);
    }
    HCCL_INFO(
        "[STREAMDBG][RequestFollowStreams] main rtStreamId=%d streamNum=%u followStreams=%zu (key: hccl_follow#%d#i).",
        rtStreamId, st.streamNum, st.followStreams.size(), rtStreamId);
    return ge::GRAPH_SUCCESS;
}

// 获取 aicpu 提前展开流 (对照 InfoStore GetHcclUnfoldStream, info_store.cc):
// ctx 先查后建 — ctx = "<group>_unfold" (COMM_ENGINE_CPU_TS) 存 ThreadHandle, 同通信域后续
// 算子命中复用; HcclDedicatedThreadAcquire 自身按 useType 在通信域内缓存复用 (hccl_res.h
// @note), 双层去重。thread 经 HcclThreadResGetInfo(THREAD_RES_TYPE_STREAM) 转 stream。
static HcclResult GetHcclUnfoldStream(HcclComm comm, const char* group, const char* opType, void*& unfoldStream)
{
    const std::string ctxName = std::string(group != nullptr ? group : "") + "_unfold";
    void* ctx = nullptr;
    uint64_t ctxSize = sizeof(ThreadHandle);
    HcclResult ret = HcclEngineCtxGet(comm, ctxName.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &ctx, &ctxSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_INFO(
            "[STREAMDBG][GetHcclUnfoldStream] unfold ctx '%s' not found, acquire dedicated thread and create ctx.",
            ctxName.c_str());
        ThreadHandle unfoldThread = 0;
        ctxSize = sizeof(ThreadHandle);
        ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 1, &unfoldThread);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[GetHcclUnfoldStream] HcclDedicatedThreadAcquire failed, opType='%s', ret=%d.", opType,
                static_cast<int>(ret));
            return ret;
        }
        // 防御: SUCCESS 但句柄为 0 属接口契约异常 — 空句柄写入 ctx 后续线程资源查询必失败, 显式报错
        if (unfoldThread == 0) {
            HCCL_ERROR(
                "[GetHcclUnfoldStream] HcclDedicatedThreadAcquire returned null thread handle, opType='%s'.", opType);
            return HCCL_E_INTERNAL;
        }
        ret = HcclEngineCtxCreate(comm, ctxName.c_str(), CommEngine::COMM_ENGINE_CPU_TS, ctxSize, &ctx);
        if (ret != HCCL_SUCCESS) {
            // 竞态降级: 并发首建时后到者 Create 可能因 "already exists" 失败,
            // 回退重查 Get 复用先建者的 ctx 而非算子整体失败
            ctxSize = sizeof(ThreadHandle);
            HcclResult getRet = HcclEngineCtxGet(comm, ctxName.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &ctx, &ctxSize);
            if (getRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[GetHcclUnfoldStream] HcclEngineCtxCreate failed (ret=%d) and retry Get also failed (ret=%d), "
                    "opType='%s'.",
                    static_cast<int>(ret), static_cast<int>(getRet), opType);
                return ret;
            }
            HCCL_INFO(
                "[GetHcclUnfoldStream] Create raced with concurrent creation, reused existing ctx '%s', opType='%s'.",
                ctxName.c_str(), opType);
        }
        // 防御: API 异常返回 SUCCESS 但 ctx 输出为空 — 必须在任何解引用之前,
        // 否则下方的写入先于判空执行, 空指针场景实际走到 SIGSEGV 而非优雅报错
        if (ctx == nullptr) {
            HCCL_ERROR(
                "[GetHcclUnfoldStream] ctx is null after create, ctxName='%s', opType='%s'.", ctxName.c_str(), opType);
            return HCCL_E_INTERNAL;
        }
        ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
        *threadPtr = unfoldThread;
    }
    // 防御: API 异常返回 SUCCESS 但 ctx 输出为空 → 空指针崩溃
    if (ctx == nullptr) {
        HCCL_ERROR(
            "[GetHcclUnfoldStream] ctx is null after EngineCtxGet/Create, ctxName='%s', opType='%s'.", ctxName.c_str(),
            opType);
        return HCCL_E_INTERNAL;
    }
    // ctxSize 校验: 按 ThreadHandle 解引用要求 ctx 至少 sizeof(ThreadHandle)
    if (ctxSize < sizeof(ThreadHandle)) {
        HCCL_ERROR(
            "[GetHcclUnfoldStream] ctxSize(%lu) < sizeof(ThreadHandle)(%zu), ctxName='%s'.", ctxSize,
            sizeof(ThreadHandle), ctxName.c_str());
        return HCCL_E_INTERNAL;
    }
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
    ThreadHandle unfoldThread = *threadPtr;

    void* stream = nullptr;
    ret = HcclThreadResGetInfo(comm, unfoldThread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(void*), &stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[GetHcclUnfoldStream] HcclThreadResGetInfo failed, opType='%s', ret=%d.", opType, static_cast<int>(ret));
        return ret;
    }
    // 防御: 线程资源在位但 stream 输出为空属异常 (对照 InfoStore 调用侧的 aicpuStream 判空),
    // 空流注册进 rtModel 会在 RTS 侧解引用, 先显式报错
    if (stream == nullptr) {
        HCCL_ERROR(
            "[GetHcclUnfoldStream] unfold stream is null after HcclThreadResGetInfo, opType='%s', ctxName='%s'.",
            opType, ctxName.c_str());
        return HCCL_E_INTERNAL;
    }
    unfoldStream = stream;
    return HCCL_SUCCESS;
}

ge::graphStatus HcclCustomOpBase::RegisterUnfoldStreamToModel(HcclOpState& st)
{
    // ---- ① 获取 unfold stream (ctx 先查后建 + thread→stream, 详见 GetHcclUnfoldStream) ----
    void* unfoldStream = nullptr;
    HcclResult hRet = GetHcclUnfoldStream(st.comm, st.group, st.opType, unfoldStream);
    if (hRet != HCCL_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // ---- ② RTS 接口: 直接调用 (链接期符号未定义, dlopen 时由提供该接口的运行时解析) ----
    // 签名 (与同族 rtStreamAddToModel(rtStream_t, rtModel_t) 同参序 — stream 首参):
    //   rtError_t rtGetModelByStream(rtStream_t stream, rtModel_t* model) — 未绑定流成功输出空 model
    //   rtError_t rtModelAddHcclStream(rtStream_t stream, rtModel_t model) — 去重登记, 不改变执行绑定

    // ---- ③ 从主流反查 rtModel 并注册 (借用句柄: V1 加载期单线程, 与销毁天然串行) ----
    rtModel_t model = nullptr;
    rtError_t rtRet = rtGetModelByStream(static_cast<rtStream_t>(st.stream), &model);
    if (rtRet != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[STREAMDBG][RegisterUnfoldStreamToModel] rtGetModelByStream(%p) failed, opType='%s', ret=0x%x.", st.stream,
            st.opType, static_cast<int>(rtRet));
        return ge::GRAPH_FAILED;
    }
    if (model == nullptr) {
        // rtGetModelByStream 语义: 有效但未绑定模型的流成功输出空 model — 主流未绑模型属异常
        // (V1 下沉模型的主流必绑 rtModel), 报错拒绝而非静默跳过
        HCCL_ERROR(
            "[STREAMDBG][RegisterUnfoldStreamToModel] main stream %p not bound to any model "
            "(rtGetModelByStream succeeded but model output is null), opType='%s'.",
            st.stream, st.opType);
        return ge::GRAPH_FAILED;
    }
    rtRet = rtModelAddHcclStream(static_cast<rtStream_t>(unfoldStream), model);
    if (rtRet != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[STREAMDBG][RegisterUnfoldStreamToModel] rtModelAddHcclStream(unfoldStream=%p, "
            "model=%p) failed, opType='%s', ret=0x%x.",
            unfoldStream, model, st.opType, static_cast<int>(rtRet));
        return ge::GRAPH_FAILED;
    }
    HCCL_INFO(
        "[STREAMDBG][RegisterUnfoldStreamToModel] unfold stream registered to model: group='%s' unfoldStream=%p "
        "model=%p (main=%p).",
        st.group, unfoldStream, model, st.stream);
    return ge::GRAPH_SUCCESS;
}

// 双流申请 (图粒度 + 通信域粒度): key 沿用旧 reuse_key 命名便于与旧路径日志对账;
// 流 ID 一并取出 — 与 GE 加载日志 "rtstream: %d"、hcomm SetHcomStream 的 hcomStreamId[%u]
// 对账用 (指针无法跨日志对账), 获取失败打 -1
static ge::graphStatus RequestAttachStreamPair(HcclOpState& st, int32_t& graphStreamId, int32_t& groupStreamId)
{
    const std::string graphKey = "hccl_attached_graph_stream";
    const std::string groupKey = "hccl_attached_group_stream_" + std::string(st.group != nullptr ? st.group : "");
    st.attachGraphStream
        = st.ctx->RequestAttachedStream(ge::AscendString(graphKey.c_str()), HCCL_GE_ATTACHED_STREAM_FLAGS);
    st.attachGroupStream
        = st.ctx->RequestAttachedStream(ge::AscendString(groupKey.c_str()), HCCL_GE_ATTACHED_STREAM_FLAGS);
    if (st.attachGraphStream == nullptr || st.attachGroupStream == nullptr) {
        HCCL_ERROR(
            "[HcclCustomOpBase][RequestAttachStream] RequestAttachedStream failed, graphKey='%s', groupKey='%s', "
            "opType='%s'.",
            graphKey.c_str(), groupKey.c_str(), st.opType);
        return ge::GRAPH_FAILED;
    }
    graphStreamId = -1;
    groupStreamId = -1;
    (void)aclrtStreamGetId(static_cast<aclrtStream>(st.attachGraphStream), &graphStreamId);
    (void)aclrtStreamGetId(static_cast<aclrtStream>(st.attachGroupStream), &groupStreamId);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcclCustomOpBase::RequestAttachStream(HcclOpState& st)
{
    st.attachGraphStream = nullptr;
    st.attachGroupStream = nullptr;
    // 旧机制 AIV 模式同样规避附属流申请 — attach 流服务于 aicpu 按序下发, AIV 路径不需要
    if (st.ifAiv) {
        return ge::GRAPH_SUCCESS;
    }

    // 旧机制为两条 attach 流 (hcom_ops_kernel_builder.cc SetAttachedStreamInfoList, 均按
    // force_reuse 跨主流复用 — assign pass 的 CalcuAttachedStreamReuseKey 对 force_reuse
    // 不拼主流 id): [0] 图粒度 (全模型一条, 所有通信域共享) + [1] 通信域粒度 (每 group 一条)。
    // 交付顺序 [graph, group] 与旧 AssembleAttachedRtStream (按属性列表序) 一致, HCCL 闭源侧
    // 按位置消费 — 顺序不可交换。建流 flags 显式指定; task_num 对齐旧 STREAM_ATTACHED_TASK_NUM=2
    // 的缺口仍存 (RequestAttachedStream 无 task_num 参数)。
    int32_t graphStreamId = -1;
    int32_t groupStreamId = -1;
    HCCL_GE_CHK_RET(RequestAttachStreamPair(st, graphStreamId, groupStreamId));

    // 交付: HcomSetAttachedStream 注入通信域 (对照旧 InfoStore SetAttachedStream, 每 op 一次;
    // 同 key 的后续算子复用同两条流)。graphId 经 GetContextValue(kGraphId) 获取
    // (kGraphId 统一取根图: V1 用 GetRootGraphId, V2 动态路径由 lowering 注入) — hcomm 侧
    // "Streams configured" 日志的 graphAttachedStreamId/groupStreamId 应与交付日志的
    // rtStreamId 逐字一致
    uint64_t graphId = 0;
    if (!st.ctx->GetContextValue(gert::ContextParam::kGraphId, graphId)) {
        HCCL_INFO("[STREAMDBG][RequestAttachStream] GetContextValue(kGraphId) unavailable, using 0.");
    }
    void* attachStreams[2] = {st.attachGraphStream, st.attachGroupStream};
    HcclResult ret = HcomSetAttachedStream(
        st.group, static_cast<uint32_t>(graphId), reinterpret_cast<const rtStream_t*>(attachStreams), 2);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCustomOpBase][RequestAttachStream] HcomSetAttachedStream failed for group '%s', opType='%s', "
            "ret=%d.",
            st.group, st.opType, static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    HCCL_INFO(
        "[STREAMDBG][RequestAttachStream] attach streams delivered via HcomSetAttachedStream: group='%s' graphId=%u "
        "graphStream=%p "
        "(rtStreamId=%d) groupStream=%p (rtStreamId=%d).",
        st.group, static_cast<uint32_t>(graphId), st.attachGraphStream, graphStreamId, st.attachGroupStream,
        groupStreamId);
    return ge::GRAPH_SUCCESS;
}

// Execute 的路径分派信号 (基类职责, 各算子只消费 st. 字段, 不 override 判定逻辑):
// GetContextValue 为正式信号源 (注入侧 CustomTaskInfo::UpdateContextInfo):
// kStaticModel → isDynamicShape。
// 动态图每次 Execute 都会走到这里, 只查分派必需的 kStaticModel — needRefresh/needCleanSeparately
// 仅静态路径消费, 拆到 ResolveStaticSignals, 避免动态路径每执行白付两次查询
static ge::graphStatus ResolvePathSignals(HcclOpState& st)
{
    uint64_t ctxValue = 0;
    if (!st.ctx->GetContextValue(gert::ContextParam::kStaticModel, ctxValue)) {
        HCCL_ERROR("[ResolvePathSignals] GetContextValue(kStaticModel) unavailable, opType='%s'.", st.opType);
        return ge::GRAPH_FAILED;
    }
    st.isDynamicShape = (ctxValue == 0); // kStaticModel=1 → 静态路径
    return ge::GRAPH_SUCCESS;
}

// 静态路径专属信号: kAddressRefreshEnabled → needRefresh (Refresh/Direct 分派);
// ge.exec.atomicCleanPolicy → needCleanSeparately (crack 清零门控, 对照旧路径 builder/InfoStore
// 两处同款门控 — 融合本身不受门控, 裂纹可存在而清理按选项开关)。动态路径不消费
static ge::graphStatus ResolveStaticSignals(HcclOpState& st)
{
    uint64_t ctxValue = 0;
    if (!st.ctx->GetContextValue(gert::ContextParam::kAddressRefreshEnabled, ctxValue)) {
        HCCL_ERROR(
            "[ResolveStaticSignals] GetContextValue(kAddressRefreshEnabled) unavailable, opType='%s'.", st.opType);
        return ge::GRAPH_FAILED;
    }
    st.needRefresh = (ctxValue == 1);

    ge::AscendString cleanPolicy;
    // 同上: GetOption 返回 graphStatus, 按 == GRAPH_SUCCESS 判定 — 原布尔用法使
    // policySet 恒 false, crack 清零门控永不开启
    const bool policySet
        = st.ctx->GetOption(ge::AscendString("ge.exec.atomicCleanPolicy"), cleanPolicy) == ge::GRAPH_SUCCESS;
    st.needCleanSeparately = policySet && cleanPolicy.GetLength() == 1 && cleanPolicy.GetString()[0] == '1';
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcclCustomOpBase::Execute(gert::EagerOpExecutionContext* ctx)
{
    // 防御: ResolvePathSignals 首行即解引用 ctx (GetContextValue), 入口显式判空
    if (ctx == nullptr) {
        HCCL_ERROR("[HcclCustomOpBase][Execute] execution context is null.");
        return ge::GRAPH_FAILED;
    }
    HcclOpState st;
    st.ctx = ctx;
    st.opType = GetOpTypeName();

    HCCL_GE_CHK_RET(ResolvePathSignals(st));

    // aivCoreLimit 三级回退解析 + (graphId, nodeId) 缓存 — 独立函数 (回退链/缓存语义见其实现处)
    st.aivCoreLimit = ResolveAivCoreLimit(ctx);

    // [CTXDBG] 路径信号与 aivCoreLimit 来源一行打全, 可据此判定信号源是否正确注入
    HCCL_INFO(
        "[CTXDBG][Execute] opType='%s' isDynamicShape=%d needRefresh=%d aivCoreLimit=%u, enter %s path.", st.opType,
        static_cast<int>(st.isDynamicShape), static_cast<int>(st.needRefresh), st.aivCoreLimit,
        st.isDynamicShape ? "dynamic" : "static");

    if (st.isDynamicShape) {
        HCCL_GE_CHK_RET(ExtractParamsDynamic(st));
        HCCL_GE_CHK_RET(GetCommunicator(st));
        return LaunchHcclOpDynamic(st);
    }
    HCCL_GE_CHK_RET(ExtractParamsStatic(st));
    // 静态专属信号 (Refresh/Direct 分派 + crack 门控): 动态路径不消费, 见 ResolvePathSignals 注释
    HCCL_GE_CHK_RET(ResolveStaticSignals(st));
    // crack 门控: policy != 1 时丢弃裂纹参数 — Clean 见空 no-op, 对齐旧路径"不清零"行为
    if (!st.needCleanSeparately) {
        st.crackOffsets.clear();
        st.crackSizes.clear();
    }
    HCCL_GE_CHK_RET(GetCommunicator(st));
    HCCL_GE_CHK_RET(CalcResourcesStatic(st));
    return LaunchHcclOpStatic(st);
}

ge::graphStatus HcclCustomOpBase::GetCommunicator(HcclOpState& st)
{
    if (st.group == nullptr) {
        HCCL_ERROR("[HcclCustomOpBase][GetCommunicator] group is null, opType='%s'.", st.opType);
        return ge::GRAPH_FAILED;
    }
    HcclComm comm = nullptr;
    HcclResult ret = HcomGetCommHandleByGroup(st.group, &comm);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCustomOpBase][GetCommunicator] HcomGetCommHandleByGroup failed for group '%s', opType='%s', "
            "ret=%d.",
            st.group, st.opType, static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    st.comm = comm;
    return ge::GRAPH_SUCCESS;
}
} // namespace hccl
