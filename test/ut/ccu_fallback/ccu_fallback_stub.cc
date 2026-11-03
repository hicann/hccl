/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <functional>
#include "acl/acl_rt.h"
#include "adapter_acl.h"
#include "ccu_fallback_stub.h"
#include "comm_worker_mgr.h"
#include "hccl.h"
#include "hccl_res.h"
#include "inconsistent_check.h"
#include "op_common.h"
#include "securec.h"

namespace ut_stub {
CcuFallbackStubCounters g_ccuFallbackStubCounters{};
CcuFallbackStubConfig g_ccuFallbackStubConfig{};

void ResetCcuFallbackStub()
{
    g_ccuFallbackStubCounters = CcuFallbackStubCounters{};
    g_ccuFallbackStubConfig = CcuFallbackStubConfig{};
}
} // namespace ut_stub

// ---- 公共通信API桩：以强定义覆盖CANN头文件中的弱声明，仅计数不做真实通信 ----
extern "C" {
HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
{
    (void)comm;
    if (rankSize == nullptr) {
        return HCCL_E_PARA;
    }
    ut_stub::g_ccuFallbackStubCounters.getRankSizeCalled++;
    *rankSize = ut_stub::g_ccuFallbackStubConfig.rankSize;
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t* rank)
{
    (void)comm;
    if (rank == nullptr) {
        return HCCL_E_PARA;
    }
    ut_stub::g_ccuFallbackStubCounters.getRankIdCalled++;
    *rank = 0;
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm comm, char* commName)
{
    (void)comm;
    if (commName == nullptr) {
        return HCCL_E_PARA;
    }
    ut_stub::g_ccuFallbackStubCounters.getCommNameCalled++;
    (void)strcpy_s(commName, ops_hccl::COMM_INDENTIFIER_MAX_LENGTH, "ut_comm");
    return HCCL_SUCCESS;
}

HcclResult HcclAllGather(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    (void)sendBuf;
    (void)recvBuf;
    (void)sendCount;
    (void)dataType;
    (void)comm;
    (void)stream;
    ut_stub::g_ccuFallbackStubCounters.allGatherCalled++;
    return HCCL_SUCCESS;
}

HcclResult HcclAllReduce(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    (void)sendBuf;
    (void)recvBuf;
    (void)count;
    (void)dataType;
    (void)op;
    (void)comm;
    (void)stream;
    ut_stub::g_ccuFallbackStubCounters.allReduceCalled++;
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    ut_stub::g_ccuFallbackStubCounters.engineCtxGetCalled++;
    if (ctx != nullptr) {
        *ctx = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return ut_stub::g_ccuFallbackStubConfig.engineCtxGetRet;
}

HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    (void)size;
    ut_stub::g_ccuFallbackStubCounters.engineCtxCreateCalled++;
    if (ctx != nullptr) {
        *ctx = nullptr;
    }
    return ut_stub::g_ccuFallbackStubConfig.engineCtxCreateRet;
}

HcclResult HcclEngineCtxDestroy(HcclComm comm, const char* ctxTag, CommEngine engine)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    ut_stub::g_ccuFallbackStubCounters.engineCtxDestroyCalled++;
    return HCCL_SUCCESS;
}

HcclResult HcclCreateSubCommConfig(
    HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId,
    HcclCommConfig* config, HcclComm* subComm)
{
    (void)comm;
    (void)rankNum;
    (void)rankIds;
    (void)subCommId;
    (void)subCommRankId;
    (void)config;
    ut_stub::g_ccuFallbackStubCounters.createSubCommCalled++;
    if (subComm != nullptr) {
        *subComm = reinterpret_cast<HcclComm>(0x1);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommDestroy(HcclComm comm)
{
    (void)comm;
    ut_stub::g_ccuFallbackStubCounters.commDestroyCalled++;
    return HCCL_SUCCESS;
}

HcclResult HcclCommResume(HcclComm comm)
{
    (void)comm;
    ut_stub::g_ccuFallbackStubCounters.commResumeCalled++;
    return HCCL_SUCCESS;
}

HcclResult HcclCommRegCommStateCallback(const char* regName, HcclCommStateCallback cb, void* args)
{
    (void)regName;
    (void)cb;
    (void)args;
    ut_stub::g_ccuFallbackStubCounters.regStateCbCalled++;
    return HCCL_SUCCESS;
}

bool HcommIsSupportHcclCommResume(void) { return false; }

bool HcommIsSupportHcclCommRegCommStateCallback(void) { return false; }
} // extern "C"

// ---- aclrt运行时桩：UT无NPU环境，仅提供可链接符号 ----
extern "C" {
aclError aclrtCreateStream(aclrtStream* stream)
{
    if (stream != nullptr) {
        *stream = reinterpret_cast<aclrtStream>(0x1);
    }
    return ACL_SUCCESS;
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtMallocHost(void** hostPtr, size_t size)
{
    (void)size;
    if (hostPtr != nullptr) {
        *hostPtr = nullptr;
    }
    return ACL_SUCCESS;
}

aclError aclrtFreeHost(void* hostPtr)
{
    (void)hostPtr;
    return ACL_SUCCESS;
}

aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    (void)size;
    (void)policy;
    if (devPtr != nullptr) {
        *devPtr = nullptr;
    }
    return ACL_SUCCESS;
}

aclError aclrtFree(void* devPtr)
{
    (void)devPtr;
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtGetCurrentContext(aclrtContext* context)
{
    if (context != nullptr) {
        *context = nullptr;
    }
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext context)
{
    (void)context;
    return ACL_SUCCESS;
}
} // extern "C"

namespace ops_hccl {
// ---- HCCL内部函数桩 ----
HcclResult haclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    (void)kind;
    if (dst == nullptr || src == nullptr || count > destMax) {
        return HCCL_E_PARA;
    }
    return memcpy_s(dst, destMax, src, count) == EOK ? HCCL_SUCCESS : HCCL_E_MEMORY;
}

HcclResult haclrtMemset(void* dst, size_t destMax, int32_t value, size_t count)
{
    if (dst == nullptr || count > destMax) {
        return HCCL_E_PARA;
    }
    return memset_s(dst, destMax, static_cast<int>(value), count) == EOK ? HCCL_SUCCESS : HCCL_E_MEMORY;
}

HcclResult SetCommEngine(OpParam& param)
{
    (void)param;
    ut_stub::g_ccuFallbackStubCounters.setCommEngineCalled++;
    return HCCL_SUCCESS;
}

HcclResult SetOpParamAlgTag(OpParam& param, const std::string& algName)
{
    (void)param;
    (void)algName;
    ut_stub::g_ccuFallbackStubCounters.setOpParamAlgTagCalled++;
    return HCCL_SUCCESS;
}

HcclResult ReSelector(
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName,
    OpExecuteConfig executeConfig)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)algName;
    (void)executeConfig;
    ut_stub::g_ccuFallbackStubCounters.reSelectorCalled++;
    return HCCL_SUCCESS;
}

HcclResult
InconsistentCheckOpType(uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const HcclCMDType& rmtOpType)
{
    (void)remoteRank;
    (void)exchangeInfo;
    (void)rmtOpType;
    ut_stub::g_ccuFallbackStubCounters.inconsistentCheckOpTypeCalled++;
    // 本端/对端opType一致场景返回成功，与真实InconsistentCheckOpType语义一致
    return HCCL_SUCCESS;
}

HcclResult ReportOpExchangeInfoCheckFailed(
    uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const std::string& paraName, uint32_t expectVal,
    uint32_t remotePara)
{
    (void)remoteRank;
    (void)exchangeInfo;
    (void)paraName;
    (void)expectVal;
    (void)remotePara;
    ut_stub::g_ccuFallbackStubCounters.reportCheckFailedCalled++;
    // 真实语义为参数不一致报错（CompareCcuParam经CHK_RET传播），据此观测误报路径
    return HCCL_E_PARA;
}

HcclResult ReportOpExchangeInfoCheckFailed(
    uint32_t remoteRank, const OpExchangeInfo& exchangeInfo, const std::string& paraName, const std::string& expectVal,
    const std::string& remotePara)
{
    (void)remoteRank;
    (void)exchangeInfo;
    (void)paraName;
    (void)expectVal;
    (void)remotePara;
    ut_stub::g_ccuFallbackStubCounters.reportCheckFailedCalled++;
    return HCCL_E_PARA;
}

// ---- CommWorkerMgr桩：不做真实worker线程，仅计数 ----
CommWorkerMgr& CommWorkerMgr::GetInstance()
{
    static CommWorkerMgr instance;
    return instance;
}

HcclResult CommWorkerMgr::Submit(HcclComm comm, const std::function<HcclResult()>& task)
{
    (void)comm;
    (void)task;
    ut_stub::g_ccuFallbackStubCounters.submitCalled++;
    return HCCL_SUCCESS;
}

void CommWorkerMgr::Remove(HcclComm comm)
{
    (void)comm;
    ut_stub::g_ccuFallbackStubCounters.removeWorkerCalled++;
}

CommWorkerMgr::~CommWorkerMgr() {}
} // namespace ops_hccl
