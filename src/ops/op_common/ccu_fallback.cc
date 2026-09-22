/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring> // 包含strncmp函数
#include <functional>
#include "ccu_fallback.h"
#include "ccu_fallback_c.h"
#include "comm_worker_mgr.h"
#include "op_common.h"
#include "inconsistent_check.h"
#include "log.h"
#include "adapter_acl.h"
#include "acl/acl_rt.h"
#include "dlhcomm_function.h"

namespace ops_hccl {
constexpr uint32_t NEGOTIATION_DATA_COUNT = 1;
constexpr uint32_t NEGOTIATION_CCL_BUFFER_SIZE = 1; // 单位MB
constexpr int32_t NEGOTIATION_SUCCESS_VAL = 0x5a5a5a5a;
constexpr int32_t NEGOTIATION_FAIL_VAL = 0x00000000;
// 通信域状态回调统一注册名：同名覆盖幂等（cb与args恒定），回调按comm动态查询协商tag判定归属
constexpr char CCU_NEGO_RESOURCE_CALLBACK[] = "ccu_nego_resource_callback";
static constexpr uint32_t opExpansionModeCcuMs = 4;
static constexpr uint32_t opExpansionModeCcuSched = 5;
// HcclCommConfig.hcclOpExpansionMode取值: 0:默认 1:host 2:aicpu 3:aiv，与HcclOpExpansionMode枚举值空间不同
static constexpr uint32_t commConfigOpExpansionAicpu = 2;

struct CheckParamInfo {
    uint64_t count;
    uint32_t opExecuteConfig;
    uint32_t dataType;
    uint32_t opType;
};
constexpr uint64_t SEND_BUF_SIZE = sizeof(CheckParamInfo);
static constexpr uint32_t EXEC_CONFIG_LEVEL_CCU_MS = 3;
static constexpr uint32_t EXEC_CONFIG_LEVEL_CCU_SCHED = 2;

static uint32_t GetOpExecuteConfigLevel(OpExecuteConfig config)
{
    switch (config) {
        case OpExecuteConfig::CCU_MS:
            return EXEC_CONFIG_LEVEL_CCU_MS;
        case OpExecuteConfig::CCU_SCHED:
            return EXEC_CONFIG_LEVEL_CCU_SCHED;
        case OpExecuteConfig::AICPU_TS:
            return 1;
        default:
            return 0;
    }
}

struct NegotiationResCtx {
    aclrtStream stream = nullptr;
    void* hostSendBuf = nullptr;
    void* hostRecvBuf = nullptr;
    void* deviceSendBuf = nullptr;
    void* deviceRecvBuf = nullptr;
    HcclComm subComm = nullptr;
};

static void CleanupNegotiationRes(NegotiationResCtx* ctx)
{
    if (ctx->deviceSendBuf != nullptr) {
        (void)aclrtFree(ctx->deviceSendBuf);
        ctx->deviceSendBuf = nullptr;
    }
    if (ctx->deviceRecvBuf != nullptr) {
        (void)aclrtFree(ctx->deviceRecvBuf);
        ctx->deviceRecvBuf = nullptr;
    }
    if (ctx->hostSendBuf != nullptr) {
        (void)aclrtFreeHost(ctx->hostSendBuf);
        ctx->hostSendBuf = nullptr;
    }
    if (ctx->hostRecvBuf != nullptr) {
        (void)aclrtFreeHost(ctx->hostRecvBuf);
        ctx->hostRecvBuf = nullptr;
    }
    if (ctx->stream != nullptr) {
        (void)aclrtDestroyStream(ctx->stream);
        ctx->stream = nullptr;
    }
    if (ctx->subComm != nullptr) {
        (void)HcclCommDestroy(ctx->subComm);
        ctx->subComm = nullptr;
    }
}

HcclResult NegotiationCleanupCb(HcclComm comm, HcclCommStatePhase state, void* args)
{
    (void)args; // 注册时不携带ownerComm，回调按comm动态查询协商tag判定归属
    if (state != HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_PRE
        && state != HcclCommStatePhase::HCCL_COMM_STATE_PHASE_RESUME_POST) {
        return HCCL_SUCCESS;
    }
    // 通过comm动态获取negCtx，避免持有EngineCtx内存指针导致野指针；存在协商tag才执行处理
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));
    std::string negTag = std::string(commName) + NEGOTIATION_COMM_SUFFIX;
    void* ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    HcclResult getRet = HcclEngineCtxGet(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
    if (getRet == HCCL_E_NOT_FOUND || (getRet == HCCL_SUCCESS && ctxPtr == nullptr)) {
        HCCL_INFO("[%s] negotiation tag not found, skip.", __func__);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(getRet != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclEngineCtxGet failed, ret[%d].", __func__, getRet), getRet);
    NegotiationResCtx* ctx = static_cast<NegotiationResCtx*>(ctxPtr);
    // 快恢场景：task abort时子通信域已被Suspend+Clean，owner恢复完成后级联恢复协商子通信域，
    // 复用stream/buffer/worker等进程级资源；失败或不支持时降级走下方清理路径，下次协商懒重建
    if (state == HcclCommStatePhase::HCCL_COMM_STATE_PHASE_RESUME_POST && ctx->subComm != nullptr
        && HcommIsSupportHcclCommResume()) {
        HcclResult resumeRet = HcclCommResume(ctx->subComm);
        if (resumeRet == HCCL_SUCCESS) {
            HCCL_INFO("[%s] negotiation subComm resumed, subComm[%p].", __func__, ctx->subComm);
            return HCCL_SUCCESS;
        }
        HCCL_ERROR("[%s] resume subComm failed, ret[%d], fallback to rebuild.", __func__, resumeRet);
    }
    // 先回收常驻worker（等待在途任务收尾并join），再清理其正在使用的stream/subComm；
    // 所有ctx销毁路径均同步回收worker，维持tag不存在即worker已回收的不变式
    CommWorkerMgr::GetInstance().Remove(comm);
    CleanupNegotiationRes(ctx);
    (void)HcclEngineCtxDestroy(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
    HCCL_INFO("[%s] negotiation resources released.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult ClearNegotiationBuf(NegotiationResCtx* negCtx, u32 rankSize)
{
    uint64_t recvBufSize = static_cast<uint64_t>(rankSize) * sizeof(CheckParamInfo);
    errno_t memRet = memset_s(negCtx->hostSendBuf, SEND_BUF_SIZE, 0, SEND_BUF_SIZE);
    CHK_PRT_RET(
        memRet != EOK, HCCL_ERROR("[%s] memset_s for hostSendBuf failed, ret[%d].", __func__, memRet), HCCL_E_MEMORY);
    memRet = memset_s(negCtx->hostRecvBuf, recvBufSize, 0, recvBufSize);
    CHK_PRT_RET(
        memRet != EOK, HCCL_ERROR("[%s] memset_s for hostRecvBuf failed, ret[%d].", __func__, memRet), HCCL_E_MEMORY);
    HcclResult setRet = haclrtMemset(negCtx->deviceSendBuf, SEND_BUF_SIZE, 0, SEND_BUF_SIZE);
    CHK_PRT_RET(
        setRet != HCCL_SUCCESS, HCCL_ERROR("[%s] haclrtMemset for deviceSendBuf failed, ret[%d].", __func__, setRet),
        HCCL_E_RUNTIME);
    setRet = haclrtMemset(negCtx->deviceRecvBuf, recvBufSize, 0, recvBufSize);
    CHK_PRT_RET(
        setRet != HCCL_SUCCESS, HCCL_ERROR("[%s] haclrtMemset for deviceRecvBuf failed, ret[%d].", __func__, setRet),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

static HcclResult AllocNegotiationStreamAndBuf(NegotiationResCtx* negCtx, u32 rankSize)
{
    aclError aclRet = aclrtCreateStream(&negCtx->stream);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtCreateStream failed, ret[%d].", __func__, aclRet), HCCL_E_RUNTIME);

    aclRet = aclrtMallocHost(&negCtx->hostSendBuf, SEND_BUF_SIZE);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtMallocHost for sendBuf failed, ret[%d].", __func__, aclRet),
        HCCL_E_RUNTIME);

    uint64_t recvBufSize = static_cast<uint64_t>(rankSize) * sizeof(CheckParamInfo);
    aclRet = aclrtMallocHost(&negCtx->hostRecvBuf, recvBufSize);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtMallocHost for recvBuf failed, ret[%d].", __func__, aclRet),
        HCCL_E_RUNTIME);

    aclRet = aclrtMalloc(&negCtx->deviceSendBuf, SEND_BUF_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtMalloc for deviceSendBuf failed, ret[%d].", __func__, aclRet),
        HCCL_E_RUNTIME);

    aclRet = aclrtMalloc(&negCtx->deviceRecvBuf, recvBufSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtMalloc for deviceRecvBuf failed, ret[%d].", __func__, aclRet),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

static HcclResult
CreateNegotiationSubCommAndRegCb(HcclComm comm, const std::string& negTag, u32 rankSize, NegotiationResCtx* negCtx)
{
    std::vector<uint32_t> rankIds(rankSize);
    for (u32 i = 0; i < rankSize; i++) {
        rankIds[i] = i;
    }
    // rankIds按序包含父通信域全部rank（恒等映射），本rank在子通信域中的rank id即为父通信域rank id
    u32 myRank = 0;
    CHK_RET(HcclGetRankId(comm, &myRank));

    HcclCommConfig subCommConfig;
    HcclCommConfigInit(&subCommConfig);
    subCommConfig.hcclBufferSize = NEGOTIATION_CCL_BUFFER_SIZE;
    subCommConfig.hcclOpExpansionMode = commConfigOpExpansionAicpu;
    // negTag超长时按尾部截断至hcclCommName上限，保留"_negotiation"后缀区分父子通信域名
    std::string subCommName = negTag;
    const uint32_t subCommNameMaxLen = sizeof(subCommConfig.hcclCommName) - 1; // 127字节
    if (subCommName.size() > subCommNameMaxLen) {
        subCommName = subCommName.substr(subCommName.size() - subCommNameMaxLen);
        HCCL_WARNING(
            "[%s] negTag exceeds max length, truncated, negTag[%s], subCommName[%s].", __func__, negTag.c_str(),
            subCommName.c_str());
    }
    auto nameRet = sprintf_s(subCommConfig.hcclCommName, sizeof(subCommConfig.hcclCommName), "%s", subCommName.c_str());
    CHK_PRT_RET(nameRet <= 0, HCCL_ERROR("[%s] sprintf_s for hcclCommName failed.", __func__), HCCL_E_INTERNAL);

    // HcclCreateSubCommConfig内部调用aclrtSetDevice会切换为默认ctx，保存调用前ctx并在创建后恢复，
    // 保证协商stream/buffer（属调用方ctx）与执行ctx一致，且不污染调用线程
    aclrtContext curCtx = nullptr;
    aclError aclRet = aclrtGetCurrentContext(&curCtx);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtGetCurrentContext failed, ret[%d].", __func__, aclRet),
        HCCL_E_RUNTIME);

    HcclResult subRet
        = HcclCreateSubCommConfig(&comm, rankSize, rankIds.data(), 0, myRank, &subCommConfig, &negCtx->subComm);
    // 无论创建成败，内部ctx切换均已发生，须先恢复；ctx为null时跳过（ST仿真无ctx，走降级路径）
    if (curCtx != nullptr) {
        aclRet = aclrtSetCurrentContext(curCtx);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtSetCurrentContext failed, ret[%d].", __func__, aclRet),
            HCCL_E_RUNTIME);
    }
    CHK_PRT_RET(
        subRet != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclCreateSubCommConfig failed, ret[%d].", __func__, subRet),
        HCCL_E_UNAVAIL);

    // 一次注册覆盖通信域全部阶段：销毁前(DESTROY_PRE)清理协商资源，快恢后(RESUME_POST)级联恢复协商子通信域；
    // 统一注册名+同名覆盖：各comm创建协商资源时重复注册均覆盖同一注册项（cb与args恒定，幂等），
    // 注册项不随comm数量累积；回调按comm动态查询协商tag判定归属，tag不存在即跳过
    if (HcommIsSupportHcclCommRegCommStateCallback()) {
        HcclResult regRet = HcclCommRegCommStateCallback(CCU_NEGO_RESOURCE_CALLBACK, NegotiationCleanupCb, nullptr);
        CHK_PRT_RET(
            regRet != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclCommRegCommStateCallback failed, ret[%d].", __func__, regRet),
            regRet);
    }
    return HCCL_SUCCESS;
}

static HcclResult GetNegotiationCtx(HcclComm comm, const OpParam& param, u32 rankSize, NegotiationResCtx*& negCtx)
{
    std::string negTag = std::string(param.commName) + NEGOTIATION_COMM_SUFFIX;
    void* ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    bool needCreate = false;
    HcclResult getRet = HcclEngineCtxGet(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
    if (getRet != HCCL_SUCCESS || ctxPtr == nullptr || ctxSize < sizeof(NegotiationResCtx)) {
        needCreate = true;
        ctxSize = sizeof(NegotiationResCtx);
        CHK_RET(HcclEngineCtxCreate(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, ctxSize, &ctxPtr));
        negCtx = static_cast<NegotiationResCtx*>(ctxPtr);
        errno_t memsetRet = memset_s(negCtx, sizeof(NegotiationResCtx), 0, sizeof(NegotiationResCtx));
        CHK_PRT_RET(memsetRet != EOK, HCCL_ERROR("[%s] memset_s failed, ret[%d].", __func__, memsetRet), HCCL_E_MEMORY);
    } else {
        negCtx = static_cast<NegotiationResCtx*>(ctxPtr);
    }

    if (needCreate) {
        HcclResult ret = AllocNegotiationStreamAndBuf(negCtx, rankSize);
        if (ret != HCCL_SUCCESS) {
            // ctx销毁与worker回收同步，维持tag不存在即worker已回收的不变式（防御懒重建失败时worker存活）
            CommWorkerMgr::GetInstance().Remove(comm);
            CleanupNegotiationRes(negCtx);
            (void)HcclEngineCtxDestroy(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
            return ret;
        }
        ret = CreateNegotiationSubCommAndRegCb(comm, negTag, rankSize, negCtx);
        if (ret != HCCL_SUCCESS) {
            CommWorkerMgr::GetInstance().Remove(comm);
            CleanupNegotiationRes(negCtx);
            (void)HcclEngineCtxDestroy(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
            return ret;
        }
        HCCL_INFO(
            "[%s] negotiation resources created, subComm[%p], stream[%p].", __func__, negCtx->subComm, negCtx->stream);
    }
    CHK_RET(ClearNegotiationBuf(negCtx, rankSize));
    return HCCL_SUCCESS;
}

static HcclResult RunInCommWorker(HcclComm comm, const std::function<HcclResult()>& task)
{
    return CommWorkerMgr::GetInstance().Submit(comm, task);
}

static HcclResult
ExecuteNegotiationOp(HcclComm comm, const OpParam& param, u32 rankSize, bool localResOk, int32_t& result)
{
    NegotiationResCtx* negCtx = nullptr;
    CHK_RET(GetNegotiationCtx(comm, param, rankSize, negCtx));

    int32_t localVal = localResOk ? NEGOTIATION_SUCCESS_VAL : NEGOTIATION_FAIL_VAL;
    HCCL_INFO("[%s] localVal[0x%x].", __func__, localVal);

    HcclResult threadRet = RunInCommWorker(comm, [&]() -> HcclResult {
        errno_t memRet = memcpy_s(negCtx->hostSendBuf, sizeof(int32_t), &localVal, sizeof(int32_t));
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[ExecuteNegotiationOp] memcpy_s for hostSendBuf failed, ret[%d].", memRet),
            HCCL_E_MEMORY);

        HcclResult cpyRet = haclrtMemcpy(
            negCtx->deviceSendBuf, sizeof(int32_t), negCtx->hostSendBuf, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        CHK_PRT_RET(
            cpyRet != HCCL_SUCCESS, HCCL_ERROR("[ExecuteNegotiationOp] haclrtMemcpy H2D failed, ret[%d].", cpyRet),
            HCCL_E_RUNTIME);

        HcclResult arRet = HcclAllReduce(
            negCtx->deviceSendBuf, negCtx->deviceRecvBuf, NEGOTIATION_DATA_COUNT, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_MIN,
            negCtx->subComm, negCtx->stream);
        CHK_PRT_RET(
            arRet != HCCL_SUCCESS, HCCL_ERROR("[ExecuteNegotiationOp] HcclAllReduce failed, ret[%d].", arRet), arRet);

        aclError aclRet = aclrtSynchronizeStream(negCtx->stream);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[ExecuteNegotiationOp] aclrtSynchronizeStream failed, ret[%d].", aclRet),
            HCCL_E_RUNTIME);

        cpyRet = haclrtMemcpy(
            negCtx->hostRecvBuf, sizeof(int32_t), negCtx->deviceRecvBuf, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHK_PRT_RET(
            cpyRet != HCCL_SUCCESS, HCCL_ERROR("[ExecuteNegotiationOp] haclrtMemcpy D2H failed, ret[%d].", cpyRet),
            HCCL_E_RUNTIME);

        memRet = memcpy_s(&result, sizeof(int32_t), negCtx->hostRecvBuf, sizeof(int32_t));
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[ExecuteNegotiationOp] memcpy_s for result failed, ret[%d].", memRet),
            HCCL_E_MEMORY);
        return HCCL_SUCCESS;
    });
    CHK_RET(threadRet);

    HCCL_INFO("[%s] negotiation result[0x%x].", __func__, result);
    return HCCL_SUCCESS;
}

static HcclResult CompareNegotiationResult(int32_t result)
{
    if (result == NEGOTIATION_FAIL_VAL) {
        HCCL_WARNING("[%s] negotiation failed, some NPUs have insufficient resources, fallback.", __func__);
        return HCCL_E_UNAVAIL;
    }

    HCCL_INFO("[%s] negotiation success, all NPUs have sufficient resources.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CheckCcuResNegotiation(HcclComm comm, const OpParam& param, bool localResAvailable)
{
    HCCL_INFO(
        "[%s] start, comm[%p], commName[%s], tag[%s], opType[%u], localResAvailable[%d].", __func__, comm,
        param.commName, param.tag, static_cast<u32>(param.opType), static_cast<int>(localResAvailable));

    u32 rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    if (rankSize <= 1) {
        HCCL_INFO("[%s] single rank, skip negotiation.", __func__);
        return localResAvailable ? HCCL_SUCCESS : HCCL_E_UNAVAIL;
    }

    int32_t result = 0;
    CHK_RET(ExecuteNegotiationOp(comm, param, rankSize, localResAvailable, result));

    return CompareNegotiationResult(result);
}

static HcclResult IsCommExpansionModeCcu(HcclComm comm, bool& isCommCcuMode)
{
    isCommCcuMode = false;
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (!hcommFunction.dlHcclConfigGetInfo) {
        HCCL_WARNING("[%s] HcclConfigGetInfo is not supported.", __func__);
        return HCCL_SUCCESS;
    }

    HcclOpExpansionMode mode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    uint32_t infoLen = sizeof(HcclOpExpansionMode);
    CHK_RET(
        hcommFunction.dlHcclConfigGetInfo(comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, infoLen, &mode));

    if (mode == static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs)
        || mode == static_cast<HcclOpExpansionMode>(opExpansionModeCcuSched)) {
        isCommCcuMode = true;
    }

    HCCL_INFO(
        "[%s] expansionMode[%d], isCommCcuMode[%d].", __func__, static_cast<int>(mode),
        static_cast<int>(isCommCcuMode));
    return HCCL_SUCCESS;
}

static HcclResult ExecuteParamCheckOp(HcclComm comm, const OpParam& param, u32 rankSize, CheckParamInfo*& recvInfos)
{
    NegotiationResCtx* negCtx = nullptr;
    CHK_RET(GetNegotiationCtx(comm, param, rankSize, negCtx));

    CheckParamInfo localInfo{};
    localInfo.count = param.DataDes.count;
    localInfo.opExecuteConfig = static_cast<uint32_t>(param.opExecuteConfig);
    localInfo.dataType = static_cast<uint32_t>(param.DataDes.dataType);
    localInfo.opType = static_cast<uint32_t>(param.opType);
    HCCL_INFO(
        "[%s] param negotiation, tag[%s], count[%llu], config[%u], dataType[%u], opType[%u].", __func__, param.tag,
        localInfo.count, localInfo.opExecuteConfig, localInfo.dataType, localInfo.opType);

    HcclResult threadRet = RunInCommWorker(comm, [&]() -> HcclResult {
        errno_t memRet = memcpy_s(negCtx->hostSendBuf, SEND_BUF_SIZE, &localInfo, sizeof(CheckParamInfo));
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[ExecuteParamCheckOp] memcpy_s for hostSendBuf failed, ret[%d].", memRet),
            HCCL_E_MEMORY);

        HcclResult cpyRet = haclrtMemcpy(
            negCtx->deviceSendBuf, SEND_BUF_SIZE, negCtx->hostSendBuf, sizeof(CheckParamInfo),
            ACL_MEMCPY_HOST_TO_DEVICE);
        CHK_PRT_RET(
            cpyRet != HCCL_SUCCESS, HCCL_ERROR("[ExecuteParamCheckOp] haclrtMemcpy H2D failed, ret[%d].", cpyRet),
            HCCL_E_RUNTIME);

        HcclResult agRet = HcclAllGather(
            negCtx->deviceSendBuf, negCtx->deviceRecvBuf, sizeof(CheckParamInfo), HCCL_DATA_TYPE_UINT8, negCtx->subComm,
            negCtx->stream);
        CHK_PRT_RET(
            agRet != HCCL_SUCCESS, HCCL_ERROR("[ExecuteParamCheckOp] HcclAllGather failed, ret[%d].", agRet), agRet);

        aclError aclRet = aclrtSynchronizeStream(negCtx->stream);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[ExecuteParamCheckOp] aclrtSynchronizeStream failed, ret[%d].", aclRet),
            HCCL_E_RUNTIME);

        uint64_t recvBufSize = static_cast<uint64_t>(rankSize) * sizeof(CheckParamInfo);
        cpyRet = haclrtMemcpy(
            negCtx->hostRecvBuf, recvBufSize, negCtx->deviceRecvBuf, recvBufSize, ACL_MEMCPY_DEVICE_TO_HOST);
        CHK_PRT_RET(
            cpyRet != HCCL_SUCCESS, HCCL_ERROR("[ExecuteParamCheckOp] haclrtMemcpy D2H failed, ret[%d].", cpyRet),
            HCCL_E_RUNTIME);
        return HCCL_SUCCESS;
    });
    CHK_RET(threadRet);

    recvInfos = static_cast<CheckParamInfo*>(negCtx->hostRecvBuf);
    return HCCL_SUCCESS;
}

static HcclResult
CompareCcuParam(const CheckParamInfo* recvInfos, u32 rankSize, const OpParam& param, OpExecuteConfig& lowestConfig)
{
    bool isVariableCountOp
        = (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC
           || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V
           || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V
           || param.opType == HcclCMDType::HCCL_CMD_HALF_ALLTOALLV
           || param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);

    if (!isVariableCountOp) {
        for (u32 i = 0; i < rankSize; i++) {
            if (recvInfos[i].count != param.DataDes.count) {
                OpExchangeInfo exchangeInfo{};
                exchangeInfo.opType = param.opType;
                exchangeInfo.count = param.DataDes.count;
                exchangeInfo.dataType = param.DataDes.dataType;
                errno_t cpyRet = strncpy_s(exchangeInfo.group, MAX_LENGTH, param.commName, COMM_INDENTIFIER_MAX_LENGTH);
                CHK_PRT_RET(
                    cpyRet != EOK, HCCL_ERROR("[%s] strncpy_s for group failed, ret[%d].", __func__, cpyRet),
                    HCCL_E_MEMORY);
                // opType、dataType、datacount均会导致数据量不一致，参照一致性校验进行规范格式报错
                CHK_RET(InconsistentCheckOpType(i, exchangeInfo, static_cast<HcclCMDType>(recvInfos[i].opType)));
                if (recvInfos[i].dataType != static_cast<uint32_t>(param.DataDes.dataType)) {
                    CHK_RET(ReportOpExchangeInfoCheckFailed(
                        i, exchangeInfo, "HcclDataType", static_cast<uint32_t>(param.DataDes.dataType),
                        recvInfos[i].dataType));
                } else {
                    CHK_RET(ReportOpExchangeInfoCheckFailed(
                        i, exchangeInfo, "DataCount", std::to_string(param.DataDes.count),
                        std::to_string(recvInfos[i].count)));
                }
            }
        }
    }

    lowestConfig = static_cast<OpExecuteConfig>(recvInfos[0].opExecuteConfig);
    uint32_t minLevel = GetOpExecuteConfigLevel(lowestConfig);
    for (u32 i = 1; i < rankSize; i++) {
        OpExecuteConfig config = static_cast<OpExecuteConfig>(recvInfos[i].opExecuteConfig);
        uint32_t level = GetOpExecuteConfigLevel(config);
        if (level < minLevel) {
            minLevel = level;
            lowestConfig = config;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CheckCcuParamAndFallback(
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName)
{
    HCCL_INFO(
        "[%s] entry, comm[%p], commName[%s], tag[%s], opType[%u], opExecuteConfig[%u], count[%llu].", __func__, comm,
        param.commName, param.tag, static_cast<u32>(param.opType), static_cast<uint32_t>(param.opExecuteConfig),
        param.DataDes.count);

    if (strncmp(param.tag, "SelectAlg_", strlen("SelectAlg_")) == 0) {
        HCCL_INFO("[%s] aiv sk selector, skip ccu param check, tag[%s].", __func__, param.tag);
        return HCCL_SUCCESS;
    }

    bool isCcuMode = false;
    CHK_RET(IsCommExpansionModeCcu(comm, isCcuMode));
    if (!isCcuMode) {
        HCCL_INFO("[%s] not ccu mode, skip ccu param check, tag[%s].", __func__, param.tag);
        return HCCL_SUCCESS;
    }

    u32 rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    if (rankSize <= 1) {
        HCCL_INFO("[%s] single rank, skip param check.", __func__);
        return HCCL_SUCCESS;
    }

    // 以传入的algName与opExecuteConfig为key查询协商结果缓存，命中则跳过跨rank协商
    std::string cacheTag
        = algName + "_" + std::to_string(static_cast<uint32_t>(param.opExecuteConfig)) + "_ccuParamCache";
    void* cacheCtx = nullptr;
    uint64_t cacheCtxSize = 0;
    HcclResult getRet = HcclEngineCtxGet(comm, cacheTag.c_str(), CommEngine::COMM_ENGINE_CCU, &cacheCtx, &cacheCtxSize);
    if (getRet == HCCL_SUCCESS && cacheCtx != nullptr && cacheCtxSize >= sizeof(FallbackCtxData)) {
        auto* ctxData = static_cast<FallbackCtxData*>(cacheCtx);
        if (ctxData->opExecuteConfig == param.opExecuteConfig) {
            HCCL_INFO(
                "[%s] cache hit, no fallback needed, algName[%s], opExecuteConfig[%u].", __func__, ctxData->algName,
                static_cast<uint32_t>(ctxData->opExecuteConfig));
            return HCCL_SUCCESS;
        }
        // 此前该key发生过回退，直接应用缓存的回退结果
        HCCL_INFO(
            "[%s] cache hit, apply cached fallback, algName[%s], opExecuteConfig[%u].", __func__, ctxData->algName,
            static_cast<uint32_t>(ctxData->opExecuteConfig));
        param.opExecuteConfig = ctxData->opExecuteConfig;
        CHK_RET(SetCommEngine(param));
        algName = ctxData->algName;
        CHK_RET(SetOpParamAlgTag(param, algName));
        return HCCL_SUCCESS;
    }

    CheckParamInfo* recvInfos = nullptr;
    CHK_RET(ExecuteParamCheckOp(comm, param, rankSize, recvInfos));

    OpExecuteConfig lowestConfig = OpExecuteConfig::DEFAULT;
    CHK_RET(CompareCcuParam(recvInfos, rankSize, param, lowestConfig));

    if (lowestConfig != param.opExecuteConfig) {
        HCCL_INFO(
            "[%s] config mismatch, local[%u] vs lowest[%u], fallback.", __func__,
            static_cast<uint32_t>(param.opExecuteConfig), static_cast<uint32_t>(lowestConfig));
        CHK_RET(ReSelector(comm, param, topoInfo, algName, lowestConfig));
    } else {
        HCCL_INFO(
            "[%s] param check success, all ranks consistent, opExecuteConfig[%u].", __func__,
            static_cast<uint32_t>(lowestConfig));
    }

    // 缓存协商结束后的最终algName与opExecuteConfig，后续同key调用跳过协商
    void* newCtx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, cacheTag.c_str(), CommEngine::COMM_ENGINE_CCU, sizeof(FallbackCtxData), &newCtx));
    auto* ctxData = static_cast<FallbackCtxData*>(newCtx);
    int copyRet = sprintf_s(ctxData->algName, sizeof(ctxData->algName), "%s", algName.c_str());
    if (copyRet <= 0) {
        HCCL_ERROR("[%s] sprintf_s for algName failed, ret[%d].", __func__, copyRet);
        (void)HcclEngineCtxDestroy(comm, cacheTag.c_str(), CommEngine::COMM_ENGINE_CCU);
        return HCCL_E_INTERNAL;
    }
    ctxData->opExecuteConfig = param.opExecuteConfig;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

__attribute__((weak)) HcclResult CheckCcuResNegotiationC(HcclComm comm, const void* param, bool localResAvailable)
{
    if (param == nullptr) {
        HCCL_ERROR("[%s] param is null.", __func__);
        return HCCL_E_PARA;
    }
    const auto& opParam = *static_cast<const ops_hccl::OpParam*>(param);
    return ops_hccl::CheckCcuResNegotiation(comm, opParam, localResAvailable);
}

__attribute__((weak)) HcclResult
CheckCcuParamAndFallbackC(HcclComm comm, void* param, void** topoInfo, char* algNameBuf, uint32_t algNameBufLen)
{
    if (param == nullptr || topoInfo == nullptr || algNameBuf == nullptr) {
        HCCL_ERROR("[%s] invalid param.", __func__);
        return HCCL_E_PARA;
    }
    auto& opParam = *static_cast<ops_hccl::OpParam*>(param);
    auto& topoInfoPtr = *static_cast<std::unique_ptr<ops_hccl::TopoInfoWithNetLayerDetails>*>(*topoInfo);
    std::string algName(algNameBuf);
    HcclResult ret = ops_hccl::CheckCcuParamAndFallback(comm, opParam, topoInfoPtr, algName);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    errno_t copyRet = strncpy_s(algNameBuf, algNameBufLen, algName.c_str(), algName.size());
    if (copyRet != EOK) {
        HCCL_ERROR("[%s] strncpy_s for algName failed.", __func__);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}
