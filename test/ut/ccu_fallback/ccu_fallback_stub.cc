/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * ccu_fallback UT 桩实现：
 * - comm_negotiation 接口桩：NegotiateAllReduceMin / NegotiateAllGather（AllGather 桩保真回填各 rank
 *   载荷，opExecuteConfig 字段按 gatherFillConfig 可配，驱动"全域一致"与"存在低配 rank"两类用例）；
 * - op_common 回退动作桩：ReSelector（保真设置 param.opExecuteConfig）/ SetCommEngine / SetOpParamAlgTag；
 * - 一致性检视桩：InconsistentCheckOpType / ReportOpExchangeInfoCheckFailed（仅计数，用例不触发报错断言）；
 * - DlHcommFunction 单例桩：不做真实 dlopen，dlHcclConfigGetInfo 由测试注入控制展开模式；
 * - HcclEngineCtx* 以 (comm, ctxTag) 内存 KV 保真实现，语义与 HCOMM 一致（同名已存在报错不覆盖）。
 */
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>

#include "ut_ccu_stub.h"
#include "op_common.h"
#include "comm_negotiation.h"
#include "inconsistent_check.h"
#include "dlhcomm_function.h"

namespace ut {
CcuStubConfig g_ccuStub;

namespace {
    // EngineCtx KV：key = (comm, ctxTag)，value = (calloc 内存, size)
    std::mutex g_kvMutex;
    std::map<std::pair<HcclComm, std::string>, std::pair<void*, uint64_t>> g_engineCtxKv;
} // namespace

void CcuStubReset()
{
    std::lock_guard<std::mutex> lock(g_kvMutex);
    for (auto& item : g_engineCtxKv) {
        free(item.second.first);
    }
    g_engineCtxKv.clear();
    g_ccuStub = CcuStubConfig();
    // 单例成员复位（空函数即 IsCommExpansionModeCcu 的"查询不支持"路径）
    ops_hccl::DlHcommFunction::GetInstance().dlHcclConfigGetInfo = nullptr;
}

bool CcuStubHasCtx(HcclComm comm, const std::string& tag)
{
    std::lock_guard<std::mutex> lock(g_kvMutex);
    return g_engineCtxKv.find({comm, tag}) != g_engineCtxKv.end();
}

bool CcuStubGetCtx(HcclComm comm, const std::string& tag, ops_hccl::FallbackCtxData& out)
{
    std::lock_guard<std::mutex> lock(g_kvMutex);
    auto it = g_engineCtxKv.find({comm, tag});
    if (it == g_engineCtxKv.end() || it->second.second < sizeof(ops_hccl::FallbackCtxData)) {
        return false;
    }
    out = *static_cast<ops_hccl::FallbackCtxData*>(it->second.first);
    return true;
}

HcclResult
CcuStubPresetCtx(HcclComm comm, const std::string& tag, const std::string& algName, ops_hccl::OpExecuteConfig cfg)
{
    void* ctx = nullptr;
    HcclResult ret
        = HcclEngineCtxCreate(comm, tag.c_str(), CommEngine::COMM_ENGINE_CCU, sizeof(ops_hccl::FallbackCtxData), &ctx);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    auto* ctxData = static_cast<ops_hccl::FallbackCtxData*>(ctx);
    int copyRet = sprintf_s(ctxData->algName, sizeof(ctxData->algName), "%s", algName.c_str());
    if (copyRet <= 0) {
        return HCCL_E_INTERNAL;
    }
    ctxData->opExecuteConfig = cfg;
    return HCCL_SUCCESS;
}
} // namespace ut

using ut::g_ccuStub;

/* ---------------- comm_negotiation 接口桩（公共协商模块，解耦不链接） ---------------- */
namespace ops_hccl {
HcclResult NegotiateAllReduceMin(HcclComm comm, const std::string& typeTag, bool localCapability, bool& allRanksCapable)
{
    (void)comm;
    (void)typeTag;
    g_ccuStub.negoReduceCnt++;
    g_ccuStub.lastLocalCapability = localCapability;
    allRanksCapable = g_ccuStub.negoAllRanksCapable;
    return g_ccuStub.negoReduceRet;
}

// 与 ccu_fallback.cc 内 static CheckParamInfo 逐字段锁定（UT 锚定协商载荷 ABI）
struct CheckParamInfoAbi {
    uint64_t count;
    uint32_t opExecuteConfig;
    uint32_t dataType;
    uint32_t opType;
    uint32_t magic;
};
static_assert(sizeof(CheckParamInfoAbi) == 24, "CheckParamInfo ABI locked");

HcclResult NegotiateAllGather(
    HcclComm comm, const std::string& typeTag, const void* localPayload, uint64_t payloadSize, void* recvBuf,
    uint64_t recvBufSize)
{
    (void)comm;
    (void)typeTag;
    g_ccuStub.negoGatherCnt++;
    if (g_ccuStub.negoGatherRet != HCCL_SUCCESS) {
        return g_ccuStub.negoGatherRet;
    }
    // 保真 AllGather：每个 rank 收到所有 rank 的载荷。UT 简化为全域同源（count/dataType/opType/magic 与
    // 本端一致），仅 opExecuteConfig/magic 按 gatherFillConfig/gatherFillMagic 覆盖（驱动"全域一致"、
    // "存在低配 rank"、"静默假成功"三类用例）
    if (localPayload == nullptr || recvBuf == nullptr || payloadSize != sizeof(CheckParamInfoAbi)
        || recvBufSize < payloadSize * g_ccuStub.rankSize) {
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < g_ccuStub.rankSize; i++) {
        auto* slot = reinterpret_cast<CheckParamInfoAbi*>(static_cast<uint8_t*>(recvBuf) + i * payloadSize);
        *slot = *static_cast<const CheckParamInfoAbi*>(localPayload);
        slot->opExecuteConfig = g_ccuStub.gatherFillConfig;
        slot->magic = g_ccuStub.gatherFillMagic;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

/* ---------------- op_common 回退动作桩 ---------------- */
namespace ops_hccl {
HcclResult ReSelector(
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName,
    OpExecuteConfig executeConfig)
{
    (void)comm;
    (void)topoInfo;
    g_ccuStub.reSelectorCnt++;
    g_ccuStub.lastReSelectorConfig = executeConfig;
    // 保真关键行为（op_common.cc 真实现首步）：设置回退目标配置后重跑 Selector，
    // algName 由 Selector 重选为回退引擎算法名；UT 以加后缀模拟重选结果
    param.opExecuteConfig = executeConfig;
    algName += "_fallback";
    return HCCL_SUCCESS;
}

HcclResult SetCommEngine(OpParam& param)
{
    (void)param;
    g_ccuStub.setCommEngineCnt++;
    return HCCL_SUCCESS;
}

HcclResult SetOpParamAlgTag(OpParam& param, const std::string& algName)
{
    (void)param;
    (void)algName;
    g_ccuStub.setAlgTagCnt++;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

/* ---------------- 一致性检视桩（仅计数） ---------------- */
namespace ops_hccl {
HcclResult
InconsistentCheckOpType(uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const HcclCMDType& rmtOpType)
{
    (void)exchangeInfo;
    (void)rmtOpType;
    g_ccuStub.inconsistentCheckCnt++;
    (void)remoteRank;
    return HCCL_SUCCESS;
}

HcclResult ReportOpExchangeInfoCheckFailed(
    uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const std::string& paraName, uint32_t expectVal,
    uint32_t remotePara)
{
    (void)exchangeInfo;
    (void)paraName;
    (void)expectVal;
    (void)remotePara;
    g_ccuStub.inconsistentCheckCnt++;
    (void)remoteRank;
    return HCCL_SUCCESS;
}

HcclResult ReportOpExchangeInfoCheckFailed(
    uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const std::string& paraName, const std::string& expectVal,
    const std::string& remotePara)
{
    (void)exchangeInfo;
    (void)paraName;
    (void)expectVal;
    (void)remotePara;
    g_ccuStub.inconsistentCheckCnt++;
    (void)remoteRank;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

/* ---------------- DlHcommFunction 单例桩（不做真实 dlopen） ---------------- */
namespace ops_hccl {
DlHcommFunction& DlHcommFunction::GetInstance()
{
    static DlHcommFunction hcclDlHcommFunction;
    return hcclDlHcommFunction;
}

DlHcommFunction::~DlHcommFunction() = default;
} // namespace ops_hccl

extern "C" {
/* ---------------- 通信域属性桩 ---------------- */
HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
{
    (void)comm;
    if (rankSize == nullptr) {
        return HCCL_E_PARA;
    }
    *rankSize = g_ccuStub.rankSize;
    return HCCL_SUCCESS;
}

/* ---------------- HcclEngineCtx KV 桩（语义与 HCOMM 一致） ---------------- */
HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    (void)engine;
    std::lock_guard<std::mutex> lock(ut::g_kvMutex);
    g_ccuStub.engineCtxCreateCnt++;
    if (ctxTag == nullptr || ctx == nullptr) {
        return HCCL_E_PARA;
    }
    // 同名 tag 已存在：与 HCOMM 真实语义一致（报错，不覆盖不返回旧块）
    std::pair<HcclComm, std::string> key = {comm, ctxTag};
    if (ut::g_engineCtxKv.find(key) != ut::g_engineCtxKv.end()) {
        return HCCL_E_INTERNAL;
    }
    void* mem = calloc(1, size);
    if (mem == nullptr) {
        return HCCL_E_MEMORY;
    }
    ut::g_engineCtxKv[key] = {mem, size};
    *ctx = mem;
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size)
{
    (void)engine;
    std::lock_guard<std::mutex> lock(ut::g_kvMutex);
    if (ctxTag == nullptr || ctx == nullptr || size == nullptr) {
        return HCCL_E_PARA;
    }
    auto it = ut::g_engineCtxKv.find({comm, ctxTag});
    if (it == ut::g_engineCtxKv.end()) {
        return HCCL_E_NOT_FOUND;
    }
    *ctx = it->second.first;
    *size = it->second.second;
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxDestroy(HcclComm comm, const char* ctxTag, CommEngine engine)
{
    (void)engine;
    std::lock_guard<std::mutex> lock(ut::g_kvMutex);
    if (ctxTag == nullptr) {
        return HCCL_E_PARA;
    }
    auto it = ut::g_engineCtxKv.find({comm, ctxTag});
    if (it == ut::g_engineCtxKv.end()) {
        return HCCL_E_NOT_FOUND;
    }
    free(it->second.first);
    ut::g_engineCtxKv.erase(it);
    return HCCL_SUCCESS;
}
} // extern "C"
