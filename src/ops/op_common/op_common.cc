/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <future>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <cstdlib> // 包含getenv函数
#include <cstring> // 包含strcmp函数
#include <stdexcept>

#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include "dev_type.h"
#include "sal.h"
#include "error_codes/rt_error_codes.h"
#include "param_check.h"
#include "inconsistent_check.h"
#include "executor_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs.h"
#include "alg_env_config.h"
#include "adapter_acl.h"
#include "topo_host.h"
#include "adapter_error_manager_pub.h"
#include "hccl_inner.h"
#include "hccl.h"
#include "config_log.h"
#include "load_kernel.h"
#include "alg_param.h"
#include "alg_type.h"
#include "op_common.h"
#include "aicpu_timeout.h"
#include "exec_timeout_manager.h"
#include "hccl_aiv_utils.h"
#include "dpu/kernel_launch.h"
#include "order_launch.h"
#include "hcomm_host_profiling_dl.h"
#include "hccl_host_comm_dl.h"
#include "hccl_res_dl.h"
#include "hccl_rank_graph_dl.h"
#include "rt_external.h"
#include "dlhcomm_function.h"
#include "hcomm_primitives_dl.h"
#include "hcomm_diag_dl.h"
#include "hcom.h"
#include "hccl_res_expt_dl.h"
#include "ccu_launch_dl.h"
#include "ccu_res_dl.h"
#include "hccl_ccu_res_dl.h"
#include "comm_engine_utils.h"
#include "utils.h"
#include "selector_engine.h"
#include "hcomm_dlsym.h"

namespace ops_hccl {
thread_local bool needInconsistentCheck = false;
// 用于维护增量建链算子的host ctx信息
constexpr u32 HOST_WAIT_AICPU_NOTIFYIDX = 0;   // host主流wait aicpu流的notify idx
constexpr u32 HOST_NOTIFY_TIMEOUT_OFFSET = 27; // host等待Device通知的超时时间偏移量
constexpr u32 KERNEL_TIMEOUT_OFFSET = 25;      // kernel启动超时时间偏移量
constexpr u32 CPU_TS_NOTIFY_NUM = 3;           // CPU TS thread notify数量

void UpdateAicpuTimeoutCtx(const OpParam& param, AlgResourceCtxSerializable& resCtx)
{
    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    resCtx.waitTimeout = timeout.waitTimeout;
    resCtx.fullTimeout = timeout.fullTimeout;
    HCCL_INFO(
        "[AicpuTimeout] execTimeout[%u]s, waitTimeout[%u]s, fullTimeout[%u]s, "
        "hostNotifyTimeout[%u]s, kernelLaunchTimeout[%u]s, hcommDefaultTimeoutSupported[%u].",
        param.opConfig.execTimeout, timeout.waitTimeout, timeout.fullTimeout, timeout.hostNotifyTimeout,
        timeout.kernelLaunchTimeout, static_cast<u32>(IsHcommDefaultTimeoutSupported()));
}

HcclResult
Selector(HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName)
{
    // 判断通信域状态
    HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
    if (HcommIsSupportHcclCommGetStatus()) {
        CHK_RET(HcclCommGetStatus(param.commName, &commStatus));
        if (commStatus != HCCL_COMM_STATUS_READY) {
            HCCL_ERROR("commStatus is not ready!, commStatus = %d", static_cast<int>(commStatus));
            return HCCL_E_SUSPENDING;
        }
    }
    HCCL_INFO("Start to execute Selector.");
    param.hcclComm = comm;
    // 获取基础拓扑
    CHK_RET(HcclCalcTopoInfo(comm, param, topoInfo));
    if (topoInfo->topLevelUboe) {
        param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
    }

    // 算法选择，选择完后顺便param.algTag设置了，资源的保存是以算子+算法为单位
    if (IsNewSelectorEnabled() && SelectorEngine::IsOpSupported(param.opType)) {
        CHK_RET(SelectorEngine::Global()->Run(comm, param, topoInfo.get(), algName));
    } else {
        std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
        CHK_RET(collAlgSelector->Run(param, topoInfo.get(), algName));
    }
    if (algName == "") {
        HCCL_ERROR("[Selector] select algname fail!");
        return HCCL_E_PTR;
    }
    CHK_RET(SetCommEngine(param));
    // AIV_ONLY 模式下禁止回退到非 AIV 引擎，未选中 AIV 时直接返回不支持。
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY
        && param.engine != CommEngine::COMM_ENGINE_AIV) {
        HCCL_ERROR(
            "[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.",
            static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    // 如果一开始读取到的Engine不是aicpu，经过算法选择后回退到aicpu，则需要重新LoadAICPUKernel
    if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU)) {
        HCCL_DEBUG("[Selector] is aicpu mode");
        CHK_RET(LoadAICPUKernel()); // 该函数内部有防止重复加载的逻辑
        // 在三级组网下走clos链路时，aicpu引擎目前仅支持NHR算法
        HCCL_WARNING("under 3-level topology with CLOS link, aicpu engine currently only supports NHR algorithm.");
    }
    // 如果一开始读取到的Engine不是aiv，经过算法选择后回退到aiv，则需要重新RegisterKernel
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        HCCL_DEBUG("[Selector] is aiv mode");
        CHK_RET(RegisterKernel()); // 该函数内部有防止重复加载的逻辑
    }
    CHK_RET(SetOpParamAlgTag(param, algName));
    // 设定执行超时时间
    CHK_RET(SetExecTimeout(param));
    // 获取多维度切分比例
    CHK_RET(SetMultipleDimensionSplitRatio(comm, param));
    HCCL_INFO("Success to execute Selector.");
    return HCCL_SUCCESS;
}

HcclResult GetHcclDfxOpInfoDataCount(const OpParam& param, const u32& rankSize, uint64_t& sendCount)
{
    sendCount = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
        sendCount
            += *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts)); // 非v类算子，只上报入参里的count
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
        for (u64 i = 0; i < rankSize; i++) { // v类算子上报累加的count
            sendCount += *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        CHK_PTR_NULL(param.varData);
        for (u64 i = 0; i < rankSize; i++) {
            sendCount += *(reinterpret_cast<const uint64_t*>(param.varData) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        CHK_PTR_NULL(param.varData);
        for (u64 i = rankSize; i < 2 * rankSize; i++) {
            sendCount += *(reinterpret_cast<const uint64_t*>(param.varData) + i);
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        CHK_PRT_RET(
            param.batchSendRecvDataDes.sendRecvItemsPtr == nullptr,
            HCCL_ERROR(
                "[%s]fail, tag[%s] sendRecvItemsPtr is nullptr, itemNum[%u]", __func__, param.tag,
                param.batchSendRecvDataDes.itemNum),
            HCCL_E_PTR);
        for (u32 idx = 0; idx < param.batchSendRecvDataDes.itemNum; idx++) {
            HcclSendRecvItem* item = param.batchSendRecvDataDes.sendRecvItemsPtr + idx;
            sendCount += item->count;
        }
    } else {
        sendCount = param.DataDes.count;
    }
    HCCL_INFO(
        "[%s]tag[%s], sendCount[%llu], opType[%u], rankSize[%u]", __func__, param.tag, sendCount, param.opType,
        rankSize);
    return HCCL_SUCCESS;
}

HcclResult GetHcclDfxOpInfoDataType(const OpParam& param, uint32_t& dataType)
{
    dataType = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        dataType = static_cast<u32>(param.vDataDes.dataType);
    } else if (
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dataType = static_cast<u32>(param.all2AllVDataDes.sendType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        CHK_PRT_RET(
            param.batchSendRecvDataDes.itemNum == 0, HCCL_INFO("[%s]tag[%s] itemNum is 0, skip", __func__, param.tag),
            HCCL_SUCCESS);
        CHK_PRT_RET(
            param.batchSendRecvDataDes.sendRecvItemsPtr == nullptr,
            HCCL_ERROR("[%s]fail, tag[%s] sendRecvItemsPtr is nullptr", __func__, param.tag), HCCL_E_PTR);
        dataType
            = static_cast<u32>(param.batchSendRecvDataDes.sendRecvItemsPtr->dataType); // dfx功能只能上报一个数据类型
    } else {
        dataType = static_cast<u32>(param.DataDes.dataType);
    }
    HCCL_INFO("[%s]tag[%s], dataType[%u], opType[%u]", __func__, param.tag, dataType, param.opType);
    return HCCL_SUCCESS;
}

HcclResult AppendFastLaunchTag(
    OpParam& param, const char* dataTypeStr, const char* reduceOpStr, const char* countStr, const char* rootStr)
{
    char* dst = param.fastLaunchTag;
    size_t remain = sizeof(param.fastLaunchTag);

    auto append_str = [&](const char* s) -> bool {
        if (!s)
            return true;
        size_t len = strlen(s);
        if (len >= remain)
            return false;
        if (memcpy_s(dst, remain, s, len) != EOK) {
            HCCL_ERROR("memcpy_s failed in append_str.");
            return false;
        }
        dst += len;
        remain -= len;
        return true;
    };
    if (!append_str(param.tag) || !append_str("_") || !append_str(dataTypeStr)) {
        goto fail;
    }
    if (reduceOpStr && (!append_str("_") || !append_str(reduceOpStr))) {
        goto fail;
    }
    if (countStr && (!append_str("_") || !append_str(countStr))) {
        goto fail;
    }
    if (rootStr && (!append_str("_r") || !append_str(rootStr))) {
        goto fail;
    }
    *dst = '\0';
    HCCL_INFO("[SetOpParamFastLaunchTag] fastLaunchTag: [%s]", param.fastLaunchTag);
    return HcclResult::HCCL_SUCCESS;

fail:
    HCCL_ERROR("failed to fill fastLaunchTag");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult SetOpParamFastLaunchTag(OpParam& param)
{
    // 1. 数据类型
    const char* dataTypeStr = nullptr;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dataTypeStr = GetHcclDataTypeStr(param.all2AllVDataDes.sendType);
    } else {
        dataTypeStr = GetHcclDataTypeStr(param.DataDes.dataType);
    }
    CHK_PRT_RET((!dataTypeStr), HCCL_ERROR("unsupported data type"), HcclResult::HCCL_E_INTERNAL);
    // 2. reduce op
    const char* reduceOpStr = nullptr;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        reduceOpStr = GetHcclReduceOpStr(param.reduceType);
        CHK_PRT_RET((!reduceOpStr), HCCL_ERROR("unsupported reduce op"), HcclResult::HCCL_E_INTERNAL);
    }
    // 3. count
    char countBuf[32];
    const char* countStr = nullptr;
    if (param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV) {
        u64 count = (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) ?
                        *reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts) :
                        param.DataDes.count;
        int countLen = snprintf_s(countBuf, sizeof(countBuf), sizeof(countBuf) - 1, "%llu", count);
        CHK_PRT_RET((countLen <= 0), HCCL_ERROR("failed to format count"), HcclResult::HCCL_E_INTERNAL);
        countStr = countBuf;
    }
    // 4. root
    char rootBuf[10];
    const char* rootStr = nullptr;
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_SCATTER
        || param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        int rootLen
            = snprintf_s(rootBuf, sizeof(rootBuf), sizeof(rootBuf) - 1, "%llu", static_cast<uint64_t>(param.root));
        CHK_PRT_RET((rootLen <= 0), HCCL_ERROR("failed to format root"), HcclResult::HCCL_E_INTERNAL);
        rootStr = rootBuf;
    }
    // 5 一次性拼接
    return AppendFastLaunchTag(param, dataTypeStr, reduceOpStr, countStr, rootStr);
}

static constexpr uint32_t opExpansionModeCcuSched = 5;
static constexpr uint32_t opExpansionModeCcuMs = 4;

bool ShouldGoCcuFastLaunch(HcclComm comm, OpParam& param, CcuFastLaunchCtx** ccuFastLaunchCtx)
{
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    param.hcclComm = comm;
    if (param.opMode == OpMode::OFFLOAD) {
        return false;
    }
    // 1. 引擎为ccu模式
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        return false;
    }
    if (SetOpParamFastLaunchTag(param) != HCCL_SUCCESS) {
        return false;
    }

    // 2. 查到engineCtx
    uint64_t size = 0;
    void* fastLaunchCtxPtr = nullptr;
    if (HcclEngineCtxGet(comm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, &fastLaunchCtxPtr, &size)
        == HCCL_SUCCESS) {
        HCCL_INFO("[ShouldGoCcuFastLaunch] get fastLaunchCtx success, size is %u", size);
        *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(fastLaunchCtxPtr);
        return true;
    }
    return false;
#else
    (void)comm;
    (void)param;
    (void)ccuFastLaunchCtx;
    return false;
#endif
}

HcclResult ConstructHcclDfxOpInfo(
    const OpParam& param, const char* tag, u32 tagSize, HcclDfxOpInfoCompat& hcclDfxOpInfo, ThreadHandle cpuTsThread)
{
    bool isAclGraph = IsStreamInCaptureMode(param.stream);
    hcclDfxOpInfo.opMode = isAclGraph ? static_cast<u32>(ops_hccl::OpMode::ACLGRAPH) : static_cast<u32>(param.opMode);
    hcclDfxOpInfo.opType = static_cast<u32>(param.opType);
    hcclDfxOpInfo.reduceOp = static_cast<u32>(param.reduceType);
    CHK_RET(GetHcclDfxOpInfoDataType(param, hcclDfxOpInfo.dataType));

    // rankSize获取指定算子的dataCount
    u32 userRankSize{0};
    CHK_RET(HcclGetRankSize(param.hcclComm, &userRankSize));
    CHK_RET(GetHcclDfxOpInfoDataCount(param, userRankSize, hcclDfxOpInfo.dataCount));
    hcclDfxOpInfo.root = param.root;
    hcclDfxOpInfo.engine = param.engine;

    hcclDfxOpInfo.inputMemAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    hcclDfxOpInfo.inputMemSize = param.inputSize;
    hcclDfxOpInfo.outputMemAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    hcclDfxOpInfo.outputMemSize = param.outputSize;

    hcclDfxOpInfo.cpuTsThread = cpuTsThread;
    hcclDfxOpInfo.cpuWaitAicpuNotifyIdx = HOST_WAIT_AICPU_NOTIFYIDX;
    s32 sRet = strncpy_s(hcclDfxOpInfo.algTag, ALG_TAG_LENGTH, tag, tagSize);
    CHK_PRT_RET(
        sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, tag:%s, tagSize:%u, sRet:%d.", __func__, tag, tagSize, sRet),
        HCCL_E_MEMORY);
    HCCL_INFO(
        "[%s]HcclDfxOpInfo param: algTag[%s], opMode[%u], opType[%u], reduceOp[%u], dataType[%u], dataCount[%llu], "
        "root[%u], engine[%s], inputMemAddr[0x%llx], inputMemSize[%llu], outputMemAddr[0x%llx], outputMemSize[%llu], "
        "cpuTsThread[0x%llu], cpuWaitAicpuNotifyIdx[%u]",
        __func__, hcclDfxOpInfo.algTag, hcclDfxOpInfo.opMode, hcclDfxOpInfo.opType, hcclDfxOpInfo.reduceOp,
        hcclDfxOpInfo.dataType, hcclDfxOpInfo.dataCount, hcclDfxOpInfo.root,
        GetEnumToString(GetCommEngineStatusStrMap(), hcclDfxOpInfo.engine).c_str(), hcclDfxOpInfo.inputMemAddr,
        hcclDfxOpInfo.inputMemSize, hcclDfxOpInfo.outputMemAddr, hcclDfxOpInfo.outputMemSize, hcclDfxOpInfo.cpuTsThread,
        hcclDfxOpInfo.cpuWaitAicpuNotifyIdx);
    return HCCL_SUCCESS;
}

bool IsStreamInCaptureMode(aclrtStream stream)
{
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI rtModel = nullptr;
    aclError aclRet = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (aclRet == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        return false;
    }
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[%s] aclmdlRICaptureGetInfo fail, ret[%d]", __func__, aclRet);
        return false;
    }
    return captureStatus == aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
}

bool IsAivCacheSupported(const OpParam& param)
{
    return (param.opType == HCCL_CMD_ALLGATHER || param.opType == HCCL_CMD_ALLREDUCE
            || param.opType == HCCL_CMD_REDUCE_SCATTER || param.opType == HCCL_CMD_BROADCAST
            || param.opType == HCCL_CMD_REDUCE || param.opType == HCCL_CMD_ALLTOALL
            || param.opType == HCCL_CMD_ALLTOALLV || param.opType == HCCL_CMD_SCATTER)
           && param.opMode == OpMode::OPBASE && !IsStreamInCaptureMode(param.stream);
}

HcclResult HcclAivCacheCheckAndReplay(HcclComm comm, OpParam& param, bool& cacheHit)
{
    cacheHit = false;
    if (!IsAivCacheSupported(param)) {
        return HCCL_SUCCESS;
    }

    // 提前获取 numBlocksLimit，作为 cache key 的一部分
    // 与 HcclAivKernelEntranceLaunch 逻辑保持一致：先查 aivParam，再 fallback 到 runtime
    u32 numBlocksLimit = 0;
    AivParamStorage* aivParam = nullptr;
    HcclResult aivParamRet = GetAivParamStorageByComm(comm, &aivParam, false);
    if (aivParamRet == HCCL_SUCCESS && aivParam != nullptr) {
        numBlocksLimit = aivParam->aivCoreLimit;
    }
    if (numBlocksLimit == 0) {
        ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit));
    }
    param.numBlocksLimit = numBlocksLimit;

    AivOpCacheArgs cacheKey = {};
    cacheKey.commName = param.commName;
    cacheKey.opType = param.opType;
    cacheKey.root = param.root;
    cacheKey.reduceOp = param.reduceType;
    cacheKey.numBlocksLimit = numBlocksLimit;
    if (param.opType == HCCL_CMD_ALLTOALL) {
        cacheKey.dataType = param.all2AllVDataDes.sendType;
        cacheKey.count = static_cast<const u64*>(param.all2AllVDataDes.sendCounts)[0];
    } else if (param.opType == HCCL_CMD_ALLTOALLV) {
        cacheKey.dataType = param.all2AllVDataDes.sendType;
        cacheKey.count = 0; // counts 在 replay 时动态刷新,cache key 不区分
    } else {
        cacheKey.count = param.DataDes.count;
        cacheKey.dataType = param.DataDes.dataType;
    }

    u64 keyHash = CalcAivCacheKeyHash(cacheKey);
    std::string ctxTag;
    CHK_RET(BuildAivCacheCtxTag(keyHash, ctxTag));

    std::string cachedAlgName;
    AivInstruction* instructions = nullptr;
    u32 insCount = 0;
    CHK_RET(LookupAivCacheCtx(comm, ctxTag, keyHash, cacheHit, cachedAlgName, instructions, insCount));
    if (!cacheHit) {
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[HcclAivCacheCheckAndReplay] cache hit, algName[%s], insCount[%u]", cachedAlgName.c_str(), insCount);

    int result = sprintf_s(param.algName, sizeof(param.algName), "%s", cachedAlgName.c_str());
    CHK_PRT_RET(result <= 0, HCCL_ERROR("[%s] failed to fill param.algName", __func__), HCCL_E_INTERNAL);
    CHK_RET(SetOpParamAlgTag(param, cachedAlgName));
    param.hcclComm = comm;

    uint64_t beginTime = HcommGetProfilingSysCycleTime();

    HcclDfxOpInfoCompat hcclDfxOpInfo{};
    CHK_RET(ConstructHcclDfxOpInfo(param, param.algTag, ALG_TAG_LENGTH, hcclDfxOpInfo, 0));
    param.dataCount = hcclDfxOpInfo.dataCount;
    CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));

    if (param.opType == HCCL_CMD_ALLTOALLV) {
        CHK_RET(ReplayAivInstructionsV(instructions, insCount, param));
    } else {
        CHK_RET(ReplayAivInstructions(instructions, insCount, param));
    }

    CHK_RET(HcclReportAivKernel(comm, beginTime));
    CHK_RET(HcclProfilingReportOp(comm, beginTime));
    return HCCL_SUCCESS;
}

HcclResult HcclExecOpCcuFastLaunch(HcclComm comm, OpParam& param, const CcuFastLaunchCtx* ccuFastLaunchCtx)
{
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    HCCL_INFO("[HcclExecOpCcuFastLaunch] HcclExecOpCcuFastLaunch start");
    std::string algName = ccuFastLaunchCtx->algName;
    HCCL_DEBUG("[HcclExecOpCcuFastLaunch] algName: [%s]", algName.c_str());
    std::unique_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    param.hcclBuff = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    // 覆盖主流
    ThreadHandle mainThread;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, param.engine, param.stream, ccuFastLaunchCtx->notifyNumOnMainThread, &mainThread));
    ThreadHandle* threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    threads[0] = mainThread;
    std::vector<ThreadHandle> threadTemps;
    threadTemps.assign(threads, threads + ccuFastLaunchCtx->threadNum);
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    // Op注册
    HcclDfxOpInfoCompat hcclDfxOpInfo{};
    CHK_RET(ConstructHcclDfxOpInfo(param, param.fastLaunchTag, ALG_TAG_LENGTH, hcclDfxOpInfo, 0));
    param.dataCount = hcclDfxOpInfo.dataCount;
    CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));
    if (IsStreamInCaptureMode(param.stream) && threadTemps.size() > 1) {
        HCCL_INFO("HcclExecOpCcuFastLaunch streamnum %d add slavestream", threadTemps.size());
        CHK_RET(CaptureSlaveStreams(comm, param.stream, threadTemps, param.isCapture));
    }

    HCCL_INFO("[HcclExecOpCcuFastLaunch] FastLaunch start");
    CHK_RET(executor->FastLaunch(param, ccuFastLaunchCtx));
    CHK_RET(HcclProfilingReportOp(comm, beginTime));
    HCCL_INFO("[HcclExecOpCcuFastLaunch] HcclExecOpCcuFastLaunch end");
    return HCCL_SUCCESS;
#else
    (void)comm;
    (void)param;
    (void)ccuFastLaunchCtx;
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult ExecuteAivCacheLogic(
    HcclComm comm, OpParam& param, const std::string& algName, std::unique_ptr<InsCollAlgBase>& executor,
    AlgResourceCtxSerializable& resCtxHost)
{
    bool useCache = IsAivCacheSupported(param);

    std::string ctxTag;
    u64 keyHash = 0;
    if (useCache) {
        AivOpCacheArgs cacheKey = {};
        cacheKey.root = param.root;
        cacheKey.commName = param.commName;
        cacheKey.opType = param.opType;
        cacheKey.reduceOp = param.reduceType;
        cacheKey.numBlocksLimit = param.numBlocksLimit;
        if (param.opType == HCCL_CMD_ALLTOALL) {
            cacheKey.dataType = param.all2AllVDataDes.sendType;
            cacheKey.count = static_cast<const u64*>(param.all2AllVDataDes.sendCounts)[0];
        } else if (param.opType == HCCL_CMD_ALLTOALLV) {
            cacheKey.dataType = param.all2AllVDataDes.sendType;
            cacheKey.count = 0; // counts 在 replay 时动态刷新,cache key 不区分
        } else {
            cacheKey.count = param.DataDes.count;
            cacheKey.dataType = param.DataDes.dataType;
        }
        keyHash = CalcAivCacheKeyHash(cacheKey);
        CHK_RET(BuildAivCacheCtxTag(keyHash, ctxTag));
        g_recordingQueue = std::make_shared<InsQueue>();
        g_baseInputAddr = reinterpret_cast<u64>(param.inputPtr);
        g_baseOutputAddr = reinterpret_cast<u64>(param.outputPtr);
    }

    CHK_RET(executor->Orchestrate(param, resCtxHost));

    if (useCache && g_recordingQueue) {
        AivCacheIndexCtx* indexCtx = nullptr;
        CHK_RET(GetOrCreateAivCacheIndexCtx(comm, &indexCtx));
        CHK_RET(EvictAivCacheIfNeeded(comm, indexCtx));
        CHK_RET(StoreAivCacheCtx(comm, ctxTag, keyHash, algName, indexCtx));
        g_recordingQueue = nullptr;
        g_baseInputAddr = 0;
        g_baseOutputAddr = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult FallbackOp(
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName,
    const ResPackGraphMode& resPack)
{
    void* fallbackCtx = nullptr;
    uint64_t fallbackCtxSize = ALG_MAX_LENGTH;
    CHK_RET(HcclEngineCtxCreate(comm, param.fallbackTag, CommEngine::COMM_ENGINE_CCU, fallbackCtxSize, &fallbackCtx));
    char* newAlgName = static_cast<char*>(fallbackCtx);
    CHK_RET(ReSelector(comm, param, topoInfo, algName));
    auto copyRet = sprintf_s(newAlgName, fallbackCtxSize, "%s", algName.c_str());
    if (copyRet <= 0) {
        HCCL_ERROR("[%s] failed to fill newAlgName", __func__);
        return HCCL_E_INTERNAL;
    }
    CHK_RET(HcclExecOp(comm, param, topoInfo, algName, resPack));
    return HCCL_SUCCESS;
}

HcclResult
ReSelector(HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName)
{
    (void)comm;
    HCCL_INFO("Start to execute ReSelector.");
    // 回退AICPU
    param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
    // 拓扑已有，无需再计算

    // 算法选择，选择完后顺便param.algTag设置了，资源的保存是以算子+算法为单位
    if (IsNewSelectorEnabled() && SelectorEngine::IsOpSupported(param.opType)) {
        CHK_RET(SelectorEngine::Global()->Run(comm, param, topoInfo.get(), algName));
    } else {
        std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
        CHK_RET(collAlgSelector->Run(param, topoInfo.get(), algName));
    }
    if (algName == "") {
        HCCL_ERROR("[ReSelector] select algname fail!");
        return HCCL_E_PTR;
    }
    CHK_RET(SetCommEngine(param));
    // AIV_ONLY 模式下禁止回退到非 AIV 引擎，未选中 AIV 时直接返回不支持。
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY
        && param.engine != CommEngine::COMM_ENGINE_AIV) {
        HCCL_ERROR(
            "[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.",
            static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    // 如果一开始读取到的Engine不是aicpu，经过算法选择后回退到aicpu，则需要重新LoadAICPUKernel
    if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU)) {
        HCCL_DEBUG("[ReSelector] is aicpu mode");
        CHK_RET(LoadAICPUKernel()); // 该函数内部有防止重复加载的逻辑
    }
    CHK_RET(SetOpParamAlgTag(param, algName));
    HCCL_INFO("Success to execute ReSelector.");
    return HCCL_SUCCESS;
}

HcclResult SetOpParamFallbackTag(OpParam& param, const std::string& algName)
{
    auto fallbackRet = sprintf_s(param.fallbackTag, sizeof(param.fallbackTag), "%s_%s", algName.c_str(), "fallback");
    if (fallbackRet <= 0) {
        HCCL_ERROR("[%s] failed to fill fallbackTag", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

static HcclResult ReportOpProfilingInfo(HcclComm comm, HcclCMDType opType, uint64_t beginTime)
{
    bool isGroupEnabled = false;
    if (HcommIsSupportHcclGroupStatusGet()) {
        CHK_RET(HcclGroupStatusGet(&isGroupEnabled));
    }
    bool isSendOrRecv = (opType == HcclCMDType::HCCL_CMD_SEND || opType == HcclCMDType::HCCL_CMD_RECEIVE);
    // group模式下aicpu的p2p算子reportOp在groupEnd中作为整体算子上报，不在这里处理
    if (!(isGroupEnabled && isSendOrRecv)) {
        CHK_RET(HcclProfilingReportOp(comm, beginTime));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclExecOp(
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName,
    const ResPackGraphMode& resPack)
{
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    HCCL_INFO("[HcclExecOp]Start to execute HcclExecOp. HcommGetProfilingSysCycleTime[%llu]", beginTime);
    // 当前通信域的某个算法回退过，则下次直接回退
    void* fallbackCtx = nullptr;
    uint64_t fallbackCtxSize = 0;
    CHK_RET(SetOpParamFallbackTag(param, algName));
    if (HcclEngineCtxGet(comm, param.fallbackTag, param.engine, &fallbackCtx, &fallbackCtxSize) == HCCL_SUCCESS) {
        HCCL_INFO("[HcclExecOp] Engine ctx exists, try to fallback.");
        std::string newAlgName = static_cast<char*>(fallbackCtx);
        HCCL_INFO("[HcclExecOp] Cached algo type is %s.", newAlgName.c_str());
        param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
        param.engine = COMM_ENGINE_AICPU_TS;
        CHK_RET(SetOpParamAlgTag(param, newAlgName));
        CHK_RET(HcclExecOp(comm, param, topoInfo, newAlgName, resPack));
        return HCCL_SUCCESS;
    }
    // 将算法名字放在param参数中
    int result = sprintf_s(param.algName, sizeof(param.algName), "%s", algName.c_str());
    if (result <= 0) {
        HCCL_ERROR("failed to fill param.algName");
        return HCCL_E_INTERNAL;
    }
    // 在原先的commName中添加执行模式，得到commModeTag
    param.hcclComm = comm;
    bool isOpBase = param.opMode == OpMode::OPBASE;
    const char* opModeStr = isOpBase ? "_opbase" : "_offload";
    auto ret = sprintf_s(param.commModeTag, sizeof(param.commModeTag), "%s_%s", param.commName, opModeStr);
    if (ret <= 0) {
        HCCL_ERROR("[%s] failed to fill param.commModeTag", __func__);
        return HCCL_E_INTERNAL;
    }

    std::unique_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    // 资源结构体
    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    resCtxHost->isHcommBatchTransferOnThreadSupported = HcommIsSupportHcommBatchTransferOnThread();
    // 资源序列化结果
    void* resCtxSequence = nullptr;
    bool isResourceReused = false;

    ThreadHandle cpuTsThread{0};
    ThreadHandle exportedAicpuTsThread{0};
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, CPU_TS_NOTIFY_NUM, &cpuTsThread));
        // Export cpuTsThread
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
    }

    auto resRet
        = HcclGetAlgRes(comm, param, executor, topoInfo.get(), resCtxHost, &resCtxSequence, isResourceReused, resPack);
    if (resRet == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[HcclGetAlgRes] resource unavailable, try to fallback.");
        CHK_RET(FallbackOp(comm, param, topoInfo, algName, resPack));
        return HCCL_SUCCESS;
    } else {
        CHK_RET(resRet);
    }

    param.cacheValid = isResourceReused;

    // Op注册
    HcclDfxOpInfoCompat hcclDfxOpInfo{};
    CHK_RET(ConstructHcclDfxOpInfo(param, param.algTag, ALG_TAG_LENGTH, hcclDfxOpInfo, cpuTsThread));
    param.dataCount = hcclDfxOpInfo.dataCount;
    CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));
    ThreadHandle exportedCpuTsThread;
    ThreadHandle mainThread;
    u32 notifyNumOnMainThread;
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        // 获取主流信息
        CHK_RET(GetMainThreadInfo(comm, param, mainThread, notifyNumOnMainThread));
        // Export mainThread
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &mainThread, COMM_ENGINE_CPU_TS, &exportedCpuTsThread));
        // cpuTsThread 添加到param里
        param.opThread = exportedAicpuTsThread;
    }

    // 算法执行
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        ThreadHandle unfoldThread;
        CHK_RET(GetUnfoldThreadInfo(comm, param, unfoldThread));
        // 根据主流的捕获状态决定展开流的状态
        CHK_RET(CaptureSlaveStreams(comm, param.stream, {mainThread, unfoldThread}, param.isCapture));
        // aicpu task cache使能
        param.aicpuCacheEnable = GetExternalInputHcclAicpuCacheEnable();
        CHK_RET(HcclAicpuKernelEntranceLaunch(
            comm, param, cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread, resCtxSequence, algName,
            unfoldThread));
    } else if (param.engine == COMM_ENGINE_AIV) {
        uint64_t aivBeginTime = HcommGetProfilingSysCycleTime();
        param.resCtx = resCtxSequence;
        AlgResourceCtxSerializable& aivResCtxHost = *static_cast<AlgResourceCtxSerializable*>(resCtxSequence);
        CHK_RET(HcclAivKernelEntranceLaunch(comm, param, topoInfo, aivResCtxHost));
        CHK_RET(ExecuteAivCacheLogic(comm, param, algName, executor, aivResCtxHost));
        CHK_RET(HcclReportAivKernel(comm, aivBeginTime));
    } else if (param.engine == COMM_ENGINE_CCU) {
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            char* ctx = static_cast<char*>(resCtxSequence);
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            resCtxHost->DeSerialize(seq);
            // 覆盖主流
            ThreadHandle thread;
            CHK_RET(HcclThreadAcquireWithStream(
                comm, param.engine, param.stream, resCtxHost->notifyNumOnMainThread, &thread));
            if (resCtxHost->threads.empty()) {
                HCCL_ERROR("[%s] reused threads is empty after DeSerialize, cannot overwrite main thread.", __func__);
                return HCCL_E_UNAVAIL;
            }
            resCtxHost->threads[0] = thread;
            // 图模式要全部覆盖
            if (param.opMode != OpMode::OPBASE) {
                CHK_RET(GeReuseResource(comm, param, executor, resCtxHost, topoInfo.get(), resPack));
            }
        }
        if (resCtxHost->slaveThreadNum > 0) {
            CHK_RET(CaptureSlaveStreams(comm, param.stream, resCtxHost->threads, param.isCapture));
        }
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    } else {
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            char* ctx = static_cast<char*>(resCtxSequence);
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            resCtxHost->DeSerialize(seq);
        }
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    }
    // op上报
    CHK_RET(ReportOpProfilingInfo(comm, param.opType, beginTime));
    HCCL_INFO("Execute HcclExecOp success.");
    return HCCL_SUCCESS;
}

HcclResult GeReuseResource(
    HcclComm comm, OpParam& param, std::unique_ptr<InsCollAlgBase>& executor,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    const ResPackGraphMode& resPack)
{
    // 计算AlgHierarchyInfo
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // 分级通信域信息{localRankId, localRankSize}
    if (param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        AlgAttrs algAttrs = executor->GetAlgoMeta(std::string(param.algName));
        CHK_RET(executor->CalcAlgHierarchyInfoV2(topoInfo, algHierarchyInfo, algAttrs));
    } else {
        CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo));
    }
    // 资源计算
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest));

    u32 maxNotifyNum = 0;
    for (u32 i = 0; i < resRequest.notifyNumPerThread.size(); i++) {
        if (resRequest.notifyNumPerThread[i] > maxNotifyNum) {
            maxNotifyNum = resRequest.notifyNumPerThread[i];
        }
    }

    u32 threadNum = resRequest.slaveThreadNum;
    u32 slaveStreams = resPack.streams.size();
    if (threadNum > slaveStreams) {
        HCCL_ERROR("[%s] threadNum[%u] exceeds slaveStreams[%u].", __func__, threadNum, slaveStreams);
        return HCCL_E_UNAVAIL;
    }
    if (resCtxHost->threads.size() < static_cast<size_t>(threadNum) + 1) {
        HCCL_ERROR(
            "[%s] reused threads size[%zu] is less than threadNum+1[%u].", __func__, resCtxHost->threads.size(),
            threadNum + 1);
        return HCCL_E_UNAVAIL;
    }
    for (u32 i = 0; i < threadNum; i++) {
        ThreadHandle slaveThread;
        CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, resPack.streams[i], maxNotifyNum, &slaveThread));
        resCtxHost->threads[i + 1] = slaveThread;
    }
    return HCCL_SUCCESS;
}

static HcclResult GetUnfoldStream(HcclComm comm, OpParam& param, ThreadHandle unfoldThread, aclrtStream& resolvedStream)
{
    void* unfoldStream = nullptr;
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    HcclResult ret;
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo || param.opMode == OpMode::OFFLOAD) { // 不走提前展开
        resolvedStream = param.stream;
    } else {
        ret = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*), &unfoldStream);
        if (ret == HCCL_E_NOT_SUPPORT) {
            resolvedStream = param.stream;
        } else if (ret != HCCL_SUCCESS) {
            resolvedStream = param.stream;
            return ret;
        } else {
            resolvedStream = unfoldStream;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAicpuKernelEntranceLaunch(
    HcclComm comm, OpParam& param, ThreadHandle cpuTsThread, ThreadHandle exportedCpuTsThread,
    u32 notifyNumOnMainThread, void* resCtxSequence, std::string& algName, ThreadHandle unfoldThread)
{
    HCCL_DEBUG("[HcclAicpuKernelEntranceLaunch]start to run aicpu kernel");
    (void)algName;
    // 当前aicpu launch接口只能有一个输入参数，将Context指针放在param参数中
    param.resCtx = resCtxSequence;
    param.aicpuRecordCpuIdx = HOST_WAIT_AICPU_NOTIFYIDX;

    if (param.engine == COMM_ENGINE_CPU) {
        // 注册dpu回调函数
        CHK_RET(static_cast<HcclResult>(HcclTaskRegister(comm, param.algTag, HcclLaunchDPUKernel)));
    }

    if (HcommIsSupportHcclAicpuKernelLaunch()
        && (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE)) {
        HCCL_INFO(
            "[HcclAicpuKernelEntranceLaunch] P2P opType[%d], use HcclAicpuKernelLaunch",
            static_cast<int>(param.opType));

        // 构造 HcclOpDesc
        HcclOpDesc opInfo;

        CHK_SAFETY_FUNC_RET(memset_s(&opInfo, sizeof(HcclOpDesc), 0, sizeof(HcclOpDesc)));
        opInfo.opDescType = 1; // 1: P2P

        std::string opNameStr = (param.opType == HcclCMDType::HCCL_CMD_SEND) ? "HcclSend" : "HcclRecv";
        CHK_SAFETY_FUNC_RET(
            strncpy_s(opInfo.opName, HCCL_OP_DESC_OP_NAME_MAX_LEN, opNameStr.c_str(), opNameStr.size()));

        opInfo.p2p.buffer = (param.opType == HcclCMDType::HCCL_CMD_SEND) ? param.inputPtr : param.outputPtr;
        opInfo.p2p.cmdType = param.opType;
        opInfo.p2p.dataType = param.DataDes.dataType;
        opInfo.p2p.count = param.DataDes.count;
        opInfo.p2p.remoteRank = param.sendRecvRemoteRank;
        aclrtStream resolvedStream;
        (void)GetUnfoldStream(comm, param, unfoldThread, resolvedStream);
        HCCL_INFO("unfoldThread[%llu]", unfoldThread);

        opInfo.p2p.unfoldStream = resolvedStream;
        // 构造 HcclKernelFuncInfo
        HcclKernelFuncInfo funcInfo;
        CHK_SAFETY_FUNC_RET(memset_s(&funcInfo, sizeof(HcclKernelFuncInfo), 0, sizeof(HcclKernelFuncInfo)));

        int soRet = sprintf_s(funcInfo.kernelSoName, sizeof(funcInfo.kernelSoName), "libscatter_aicpu_kernel.so");
        CHK_PRT_RET(soRet <= 0, HCCL_ERROR("[%s] failed to fill kernelSoName", __func__), HCCL_E_INTERNAL);

        int funcRet = sprintf_s(funcInfo.kernelFuncName, sizeof(funcInfo.kernelFuncName), "HcclLaunchP2pAicpuKernel");
        CHK_PRT_RET(funcRet <= 0, HCCL_ERROR("[%s] failed to fill kernelFuncName", __func__), HCCL_E_INTERNAL);

        // 获取 aicpuThreadHandle
        ThreadHandle aicpuThreadHandle;
        u32 mainNotifyNum;
        CHK_RET(GetMainThreadInfo(comm, param, aicpuThreadHandle, mainNotifyNum));

        // 调用 HcclAicpuKernelLaunch
        void* args = &param;
        uint32_t argSize = sizeof(OpParam) + param.varMemSize;

        funcInfo.args = args;
        funcInfo.argSize = argSize;

        HcclKernelLaunchCfg kernelLaunchCfg;
        AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
        u16 kernelLaunchTimeout
            = IsHcommDefaultTimeoutSupported() ?
                  timeout.kernelLaunchTimeout :
                  ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
        kernelLaunchCfg.timeOut = kernelLaunchTimeout;

        CHK_RET(HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, param.stream, &kernelLaunchCfg));

        HCCL_INFO("[HcclAicpuKernelEntranceLaunch] P2P launch success, algTag[%s]", param.algTag);
        return HCCL_SUCCESS;
    }

    // Host stream通知Device主thread，使用主流上idx最大的notify
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread - 1)));

    // OrderLaunch第一阶段
    // 获取执行超时时间
    u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();

    OrderLaunchMode launchMode = param.isCapture ?
                                     OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH :
                                     (param.opMode == OpMode::OFFLOAD ? OrderLaunchMode::ORDER_LAUNCH_GE :
                                                                        OrderLaunchMode::ORDER_LAUNCH_OPBASE);

    HcclRtEventGuard event0Guard;
    HcclRtEventGuard event1Guard;
    if (launchMode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        CHK_RET(event0Guard.Create());
        CHK_RET(event1Guard.Create());
    }
    CHK_RET(HcclOrderLaunchToOrderStream(
        comm, param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX, execTimeout, launchMode, event0Guard.Get()));

    // AicpuKernel report
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    CHK_RET(AicpuKernelLaunch(comm, param, unfoldThread));
    CHK_PTR_NULL(comm);

    // OrderLaunch第二阶段
    CHK_RET(HcclOrderLaunchToKernelStream(
        comm, unfoldThread, HOST_ORDER_THREAD_NOTIFY_IDX, execTimeout, launchMode, event1Guard.Get()));

    std::string kernelName = "HcclLaunchAicpuKernel";
    char* kernelNameCStr = const_cast<char*>(kernelName.c_str());
    HcclResult ret = HcclReportAicpuKernel(comm, beginTime, kernelNameCStr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclAicpuKernelEntranceLaunch] HcclReportAicpuKernel failed, beginTime %lu, kernelNameCStr %s, ret %d ",
            beginTime, kernelNameCStr, ret);
        return ret;
    }
    // Host stream等待Device的通知
    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    u32 hostNotifyWaitTime = IsHcommDefaultTimeoutSupported() ?
                                 timeout.hostNotifyTimeout :
                                 AddAicpuTimeoutOffset(param.opConfig.execTimeout, HOST_NOTIFY_TIMEOUT_OFFSET);
    if (HcommIsSupportHcommSetNotifyWaitTimeOut()) {
        CHK_RET(HcclSetNotifyWaitTimeOut(hostNotifyWaitTime));
    }
    CHK_RET(HcclThreadNotifyWaitOnThreadDefault(cpuTsThread, param.aicpuRecordCpuIdx, hostNotifyWaitTime));

    return HCCL_SUCCESS;
}

HcclResult AicpuKernelLaunch(HcclComm comm, OpParam& param, ThreadHandle unfoldThread)
{
    std::string kernelName = "HcclLaunchAicpuKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    // 注意，目前开源HCCL加载AICPU kernel使用的是从json文件加载
    // 详见load_kernel.cc中的LoadAICPUKernel函数，且只实现了scatter的，先共用scatter的
    aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, "
            "kernelName:%s",
            ret, kernelName.c_str()),
        HCCL_E_RUNTIME);
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, "
            "kernelName:%s",
            ret, kernelName.c_str()),
        HCCL_E_RUNTIME);
    aclrtParamHandle paraHandle;
    size_t paramSize = sizeof(OpParam) + param.varMemSize;
    ret = aclrtKernelArgsAppend(argsHandle, &param, paramSize, &paraHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append "
            "size %u, kernelName:%s",
            ret, paramSize, kernelName.c_str()),
        HCCL_E_RUNTIME);
    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, "
            "kernelName:%s",
            ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    u16 kernelLaunchTimeout
        = IsHcommDefaultTimeoutSupported() ?
              timeout.kernelLaunchTimeout :
              ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = kernelLaunchTimeout;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;
    HCCL_INFO("[AicpuKernelLaunch] unfoldThread [%lu]", unfoldThread); // 通过Thread获取展开流stream
    void* unfoldStream = nullptr;
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo || param.opMode == OpMode::OFFLOAD) { // 不走提前展开
        ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
    } else {
        HcclResult ret1
            = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*), &unfoldStream);
        if (ret1 == HCCL_E_NOT_SUPPORT) {
            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
        } else if (ret1 != HCCL_SUCCESS) {
            return ret1;
        } else {
            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, unfoldStream, &cfg, argsHandle, nullptr);
        }
    }
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[LoadCustomKernel][aclrtLaunchKernelWithConfig]"
            "errNo[0x%016llx] launch kernel failed",
            ret),
        HCCL_E_OPEN_FILE_FAILURE);
    return HCCL_SUCCESS;
}

HcclResult HcclAivKernelEntranceLaunch(
    HcclComm comm, OpParam& param, const std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo,
    AlgResourceCtxSerializable& resCtxHost)
{
    (void)topoInfo;
    HCCL_INFO(
        "[%s] algTag[%s] commModeTag[%s] resCtx(Host)[%p] aivCommInfoPtr(Device)[%p]", __func__, param.algTag,
        param.commModeTag, param.resCtx, resCtxHost.aivCommInfoPtr);
    u32 numBlocksLimit = 0;
    AivParamStorage* aivParam = nullptr;
    HcclResult ret = GetAivParamStorageByComm(comm, &aivParam, false);
    if (ret == HCCL_SUCCESS && aivParam != nullptr) {
        numBlocksLimit = aivParam->aivCoreLimit;
    } else if (param.opMode == OpMode::OFFLOAD) {
        HCCL_ERROR("[%s] GetAivParamStorageByComm faile, ret[%d], aivParam[%p]", __func__, ret, aivParam);
        return HCCL_E_INTERNAL;
    }
    if (numBlocksLimit == 0 && param.opMode == OpMode::OPBASE) {
        ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit));
    }
    CHK_PRT_RET(
        numBlocksLimit < 1, HCCL_ERROR("[%s] block num less than 1, block num[%d]", __func__, numBlocksLimit),
        HCCL_E_PARA);
    param.numBlocksLimit = numBlocksLimit;
    HCCL_INFO("[%s] Aiv core limit is [%d].", __func__, numBlocksLimit);
    return HCCL_SUCCESS;
}

HcclResult
CaptureSlaveStreams(HcclComm comm, aclrtStream mainStream, const std::vector<ThreadHandle>& threads, bool& isCapture)
{
    isCapture = false;
    aclmdlRI rtModel = nullptr;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclError ret = aclmdlRICaptureGetInfo(mainStream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture not support.", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(
            ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclmdlRICaptureGetInfo fail. return[%d].", __func__, ret),
            HCCL_E_RUNTIME);
    }
    if (captureStatus != aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        HCCL_INFO("[%s]captureStatus is not active, captureStatus[%d]", __func__, captureStatus);
        return HCCL_SUCCESS;
    }
    isCapture = true;
    // thread[0] is main thread
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    for (size_t i = 1; i < threads.size(); ++i) {
        void* stream = nullptr;
        CHK_PRT_RET(
            !HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo, HCCL_ERROR("AclGraph is not support."),
            HCCL_E_NOT_SUPPORT);
        CHK_RET(HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, threads[i], 0, sizeof(void*), &stream));
        rtError_t addRet = rtStreamAddToModel(stream, rtModel);
        CHK_PRT_RET(
            addRet != RT_ERROR_NONE, HCCL_ERROR("[%s]rtStreamAddToModel fail. return[%d].", __func__, addRet),
            HCCL_E_RUNTIME);
        HCCL_DEBUG(
            "[%s]add slaveStream to model success, idx[%zu], stream[%p], rtModel[%p]", __func__, i, stream, rtModel);
    }
    HCCL_INFO(
        "[%s]success, captured streams to rtmodel:[%p], slaveStreamNum:[%zu]", __func__, rtModel,
        threads.size() > 0 ? threads.size() - 1 : 0);
    return HCCL_SUCCESS;
}

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo)
{
    HCCL_INFO("[%s] HcclCalcTopoInfo start.", __func__);
    uint64_t size = 0;
    void* ctx = nullptr;
    // 若获取Context失败，表示对应Context尚未缓存
    HcclResult ret = HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size);
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        // 初始化topoInfo
        CHK_RET(InitRankInfo(comm, topoInfo.get()));
        // 序列化
        std::vector<char> seq = topoInfo->Serialize();
        size = seq.size();
        // 创建新的Context保存
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
        CHK_SAFETY_FUNC_RET(memcpy_s(ctx, size, seq.data(), size));
        return HCCL_SUCCESS;
    }
    char* ctxTemp = reinterpret_cast<char*>(ctx);
    std::vector<char> seq(ctxTemp, ctxTemp + size);
    TopoInfoWithNetLayerDetails topoInfoTemp;
    topoInfoTemp.DeSerialize(seq);
    topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>(std::move(topoInfoTemp));
    HCCL_INFO("[%s] HcclCalcTopoInfo end.", __func__);
    return HCCL_SUCCESS;
}

void CompReqChannelWithExistChannel(
    const std::vector<std::vector<ChannelInfo>>& existChannels, AlgResourceRequest& resRequest)
{
    std::set<u32> existRemoteRankSet = {};
    std::vector<HcclChannelDesc> needAllocChannelDesc;
    // 先把所有已存在的channel的remoteRank整理成集合
    for (const ChannelInfo& channel : existChannels[0]) {
        existRemoteRankSet.insert(channel.remoteRank);
    }
    // 在集合中查找有没有request的channel
    for (const HcclChannelDesc& channelDesc : resRequest.channels[0]) {
        if (existRemoteRankSet.find(channelDesc.remoteRank) == existRemoteRankSet.end()) {
            needAllocChannelDesc.push_back(channelDesc);
        }
    }
    resRequest.channels = {needAllocChannelDesc};
    return;
}

static HcclResult TryReuseResource(
    HcclComm comm, OpParam& param, bool& increCreateChannelFlag, void** resCtxSequence, uint64_t& size,
    bool& isResourceReused)
{
    // 增量建链模式下不能复用资源
    if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && param.opMode == OpMode::OPBASE) {
        increCreateChannelFlag = true;
        return HCCL_E_NOT_FOUND;
    }
    // 非OPBASE模式且非CCU引擎不能复用资源
    if (param.opMode != OpMode::OPBASE && param.engine != CommEngine::COMM_ENGINE_CCU) {
        return HCCL_E_NOT_FOUND;
    }
    void* ctx = nullptr;
    // 这种情况下资源已经有了
    CommEngine ctxEngine = param.engine;
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        // AIV模式固定利用利用algTag申请1块host内存resCtx
        ctxEngine = COMM_ENGINE_CPU_TS;
    } else if (param.engine == COMM_ENGINE_CPU) {
        // host dpu申请device内存用于存放resctx
        ctxEngine = COMM_ENGINE_AICPU_TS;
    }
    if (HcclEngineCtxGet(comm, param.algTag, ctxEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_DEBUG("Already have context, skip create, ctxSize is %llu", size);
        isResourceReused = true;
        *resCtxSequence = ctx;
        param.ctxSize = size;
        return HCCL_SUCCESS;
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult HcclGetAlgRes(
    HcclComm comm, OpParam& param, std::unique_ptr<InsCollAlgBase>& executor, TopoInfoWithNetLayerDetails* topoInfo,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void** resCtxSequence, bool& isResourceReused,
    const ResPackGraphMode& resPack)
{
    HCCL_INFO("[HcclGetAlgRes] Start to execute HcclGetAlgRes.");

    // 获取当前算法的完整属性
    AlgAttrs algoMeta = executor->GetAlgoMeta(std::string(param.algName));
    HCCL_INFO(
        "[HcclGetAlgRes] algName=%s, engine=%d, opType=%d, algoTypes.size=%zu.", param.algName,
        static_cast<int>(algoMeta.engine), static_cast<int>(algoMeta.opType), algoMeta.algoTypes.size());

    bool increCreateChannelFlag = false;
    uint64_t size = 0;
    if (TryReuseResource(comm, param, increCreateChannelFlag, resCtxSequence, size, isResourceReused) == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    // context创建前做是否需要参数一致性的判断，context未创建则判断为首次下发该算子
    needInconsistentCheck = NeedInconsistentCheck(comm, param);

    // 计算AlgHierarchyInfo
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // 分级通信域信息{localRankId, localRankSize}
    if (param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        CHK_RET(executor->CalcAlgHierarchyInfoV2(topoInfo, algHierarchyInfo, algoMeta));
    } else {
        CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo));
    }
    // 资源计算
    HCCL_INFO("[HcclGetAlgRes] executor->CalcRes.");
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest));
    auto ret = GetAlgResWithEngine(
        comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size, increCreateChannelFlag,
        resPack);
    if (ret == HCCL_E_UNAVAIL) {
        return HCCL_E_UNAVAIL;
    }
    CHK_RET(ret);

    if (resCtxHost != nullptr) {
        // 拼接各level的channel数量信息
        std::string channelNumInfo;
        for (size_t i = 0; i < resCtxHost->channels.size(); i++) {
            if (i > 0)
                channelNumInfo += ", ";
            channelNumInfo += "level" + std::to_string(i) + "[" + std::to_string(resCtxHost->channels[i].size()) + "]";
        }
        HCCL_RUN_INFO(
            "[HcclGetAlgRes] engine[%s], algTag[%s], resource allocated: thread num[%u], "
            "channel num per level[%s], ccu kernel num[%u].",
            GetEnumToString(GetCommEngineStatusStrMap(), param.engine).c_str(), param.algTag,
            resCtxHost->threads.size(), channelNumInfo.c_str(), resCtxHost->ccuKernels.size());
    }

    // 参数一致性校验
    if (needInconsistentCheck) {
        OpExchangeInfo exchangeInfo{};
        CHK_RET(FillOpExchangeInfo(comm, param, exchangeInfo));
        CHK_RET(CompareOpExchangeInfos(comm, param, resRequest, exchangeInfo));
    }

    return HCCL_SUCCESS;
}

HcclResult FillOpExchangeInfo(HcclComm comm, const OpParam& param, OpExchangeInfo& exchangeInfo)
{
    CHK_PTR_NULL(comm);
    void* cclBufferAddr = nullptr; // 不使用，仅为调用HcclGetHcclBuffer获取cclBufferSize
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &exchangeInfo.cclBufferSize));
    exchangeInfo.root = param.root;
    exchangeInfo.opType = param.opType;
    exchangeInfo.opExecuteConfig = param.opExecuteConfig;
    exchangeInfo.reduceType = param.reduceType;
    CHK_RET(FillOpExchangeInfoWithDataDes(param, exchangeInfo));

    u32 numBlocksLimit = 0;
    AivParamStorage* aivParam = nullptr;
    HcclResult ret = GetAivParamStorageByComm(comm, &aivParam, false);
    if (ret == HCCL_SUCCESS && aivParam != nullptr) {
        numBlocksLimit = aivParam->aivCoreLimit;
        exchangeInfo.aivCoreLimit = numBlocksLimit;
    }
    if (numBlocksLimit == 0 && param.opMode == OpMode::OPBASE) {
        ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit));
        exchangeInfo.aivCoreLimit = numBlocksLimit;
    }

    CHK_RET(HcclGetCommName(comm, exchangeInfo.group));
    exchangeInfo.group[MAX_LENGTH - 1] = '\0';
    s32 sRet = strncpy_s(exchangeInfo.tag, TAG_LENGTH, param.tag, TAG_LENGTH);
    CHK_PRT_RET(
        sRet != EOK, HCCL_ERROR("[%s] call strncpy_s failed, param.tag[%s], return[%d].", __func__, param.tag, sRet),
        HCCL_E_MEMORY);

    HCCL_INFO(
        "[%s] success. exchangeInfo dump: cclBufferSize[%llu], root[%u], opType[%u], opExecuteConfig[%u], "
        "reduceType[%u], dataType[%u], count[%llu], aivCoreLimit[%u], group[%s], tag[%s]",
        __func__, exchangeInfo.cclBufferSize, exchangeInfo.root, exchangeInfo.opType, exchangeInfo.opExecuteConfig,
        exchangeInfo.reduceType, exchangeInfo.dataType, exchangeInfo.count, exchangeInfo.aivCoreLimit,
        exchangeInfo.group, exchangeInfo.tag);
    return HCCL_SUCCESS;
}

HcclResult FillOpExchangeInfoWithDataDes(const OpParam& param, OpExchangeInfo& exchangeInfo)
{
    switch (param.opType) {
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALL:
            exchangeInfo.dataType = param.all2AllVDataDes.sendType;
            CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
            exchangeInfo.count = static_cast<u64*>(param.all2AllVDataDes.sendCounts)[0];
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            exchangeInfo.dataType = param.all2AllVDataDes.sendType;
            break;
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
            exchangeInfo.dataType = param.vDataDes.dataType;
            break;
        default:
            exchangeInfo.dataType = param.DataDes.dataType;
            exchangeInfo.count = param.DataDes.count;
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult AddExchangeInfo(HcclComm comm, const OpParam& param)
{
    CHK_PTR_NULL(comm);
    if (needInconsistentCheck) {
        OpExchangeInfo exchangeInfo{};
        CHK_RET(FillOpExchangeInfo(comm, param, exchangeInfo));
        CHK_RET(HcclCommAddExchangeInfo(comm, &exchangeInfo, sizeof(exchangeInfo)));
        HCCL_INFO("[%s] success.", __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult GetAlgResWithEngine(
    HcclComm comm, OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& size, bool increCreateChannelFlag,
    const ResPackGraphMode& resPack)
{
    // host侧资源
    if (param.engine == COMM_ENGINE_RESERVED) {
        // COMM_ENGINE_RESERVED
    } else if (param.engine == COMM_ENGINE_CPU) {
        CHK_RET(GetAlgResDPU(
            comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size,
            increCreateChannelFlag, resPack));
    } else if (param.engine == COMM_ENGINE_CPU_TS) {
        // COMM_ENGINE_CPU_TS
    } else if (param.engine == COMM_ENGINE_AICPU) {
        // COMM_ENGINE_AICPU
    } else if (param.engine == COMM_ENGINE_AICPU_TS) {
        CHK_RET(GetAlgResAICPU(
            comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size,
            increCreateChannelFlag, resPack));
    } else if (param.engine == COMM_ENGINE_AIV) {
        CHK_RET(GetAlgResAiv(comm, param, resRequest, topoInfo, algHierarchyInfo, resCtxSequence));
    } else if (param.engine == COMM_ENGINE_CCU) {
        // 添加资源回退。SetCommEngine
        auto ret = GetAlgResCcu(
            comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size, resPack);
        if (ret == HCCL_E_UNAVAIL) {
            return HCCL_E_UNAVAIL;
        }
        CHK_RET(ret);
    } else {
        HCCL_ERROR(
            "fail to get engine, invalid engine type[%s].",
            GetEnumToString(GetCommEngineStatusStrMap(), param.engine).c_str());
        return HCCL_E_PARA;
    }
    param.ctxSize = size;
    return HCCL_SUCCESS;
}

HcclResult CacheHostCtxToEngine(
    HcclComm comm, const char* algTag, const std::string& hostCacheTag, const std::vector<char>& hostCtxSeq)
{
    void* hostCtxPtr = nullptr;
    HcclResult createRet = HcclEngineCtxCreate(
        comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, hostCtxSeq.size(), &hostCtxPtr);
    if (createRet != HCCL_SUCCESS) {
        HCCL_ERROR("failed to create host EngineCtx for caching, ret[%d].", createRet);
        HcclResult destroyRet = HcclEngineCtxDestroy(comm, algTag, COMM_ENGINE_AICPU_TS);
        if (destroyRet != HCCL_SUCCESS) {
            HCCL_ERROR("failed to destroy device ctx on host ctx create failure rollback, ret[%d].", destroyRet);
        }
        return createRet;
    }
    errno_t memcpyRet = memcpy_s(hostCtxPtr, hostCtxSeq.size(), hostCtxSeq.data(), hostCtxSeq.size());
    if (memcpyRet != EOK) {
        HCCL_ERROR("memcpy_s failed writing to host EngineCtx cache, ret=%d.", memcpyRet);
        HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
        HcclEngineCtxDestroy(comm, algTag, COMM_ENGINE_AICPU_TS);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult ReuseCachedDeviceCtx(HcclComm comm, const OpParam& param, void** resCtxSequence, uint64_t& ctxSize)
{
    void* ctx = nullptr;
    uint64_t size = 0;
    HcclResult ret;
    if (param.engine == COMM_ENGINE_CPU) {
        ret = HcclEngineCtxGet(comm, param.algTag, COMM_ENGINE_AICPU_TS, &ctx, &size);
    } else {
        ret = HcclEngineCtxGet(comm, param.algTag, param.engine, &ctx, &size);
    }
    if (ret == HCCL_SUCCESS) {
        *resCtxSequence = ctx;
        ctxSize = size;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("failed to get device ctx.");
    return ret;
}

HcclResult IncrementalCreateChannel(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, AlgResourceCtxSerializable& hostCtxObj,
    const std::string& hostCacheTag, void** resCtxSequence, uint64_t& ctxSize)
{
    HcclResult ret = HcclGetChannel(comm, param, resRequest, &hostCtxObj);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to incrementally create channel."), ret);
    if (param.engine == COMM_ENGINE_CPU) {
        ret = HcclEngineCtxDestroy(comm, param.algTag, COMM_ENGINE_AICPU_TS);
    } else {
        ret = HcclEngineCtxDestroy(comm, param.algTag, param.engine);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to destroy device Ctx, ret[%d].", ret);
    }
    std::vector<char> newSeq = hostCtxObj.Serialize();
    ret = HcclMemcpyCtxHostToDevice(comm, param, newSeq, resCtxSequence, ctxSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to memcpy hostCtx to device after incremental channel creation, ret[%d].", ret);
        HcclResult destroyRet = HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
        if (destroyRet != HCCL_SUCCESS) {
            HCCL_ERROR("failed to destroy host ctx on incremental path failure rollback, ret[%d].", destroyRet);
        }
        return ret;
    }
    HcclResult destroyRet = HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
    if (destroyRet != HCCL_SUCCESS) {
        HCCL_ERROR("failed to destroy old host EngineCtx for cache update, ret[%d].", destroyRet);
    }
    void* newHostCtxPtr = nullptr;
    HcclResult cacheRet = HcclEngineCtxCreate(
        comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, newSeq.size(), &newHostCtxPtr);
    if (cacheRet != HCCL_SUCCESS) {
        HCCL_ERROR("failed to create host EngineCtx for cache update, ret[%d].", cacheRet);
        HcclResult devDestroyRet = HcclEngineCtxDestroy(comm, param.algTag, param.engine);
        if (devDestroyRet != HCCL_SUCCESS) {
            HCCL_ERROR("failed to destroy device ctx on host cache update failure rollback, ret[%d].", devDestroyRet);
        }
        return cacheRet;
    }
    errno_t memcpyRet = memcpy_s(newHostCtxPtr, newSeq.size(), newSeq.data(), newSeq.size());
    if (memcpyRet != EOK) {
        HCCL_ERROR("memcpy_s failed writing to updated host EngineCtx cache, ret=%d.", memcpyRet);
        HcclEngineCtxDestroy(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
        HcclEngineCtxDestroy(comm, param.algTag, param.engine);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("Incrementally add channel success");
    return HCCL_SUCCESS;
}

HcclResult GetAlgResAICPU(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag, const ResPackGraphMode& resPack)
{
    std::string hostCacheTag = std::string(param.algTag) + "_hostCache";
    void* hostCtxPtr = nullptr;
    uint64_t hostCtxSize = 0;
    HcclResult hostCtxRet
        = HcclEngineCtxGet(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &hostCtxPtr, &hostCtxSize);
    if (!increCreateChannelFlag || hostCtxRet != HCCL_SUCCESS) {
        resCtxHost->commInfoPtr = static_cast<void*>(comm);
        resCtxHost->topoInfo = *topoInfo;
        resCtxHost->algHierarchyInfo = algHierarchyInfo;
        HcclResult ret = HcclAllocAlgResourceAICPU(comm, param, resRequest, resCtxHost, resPack);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to alloc alg resource."), ret);
        std::vector<char> hostCtxSeq = resCtxHost->Serialize();
        ret = HcclMemcpyCtxHostToDevice(comm, param, hostCtxSeq, resCtxSequence, ctxSize);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("failed to memcpy hostCtx to device."), ret);
        if (increCreateChannelFlag) {
            CHK_RET(CacheHostCtxToEngine(comm, param.algTag, hostCacheTag, hostCtxSeq));
        }
    } else {
        std::vector<char> cachedData(static_cast<char*>(hostCtxPtr), static_cast<char*>(hostCtxPtr) + hostCtxSize);
        AlgResourceCtxSerializable hostCtxObj;
        hostCtxObj.DeSerialize(cachedData);
        CompReqChannelWithExistChannel(hostCtxObj.channels, resRequest);
        if (resRequest.channels[0].size() == 0) {
            return ReuseCachedDeviceCtx(comm, param, resCtxSequence, ctxSize);
        }
        CHK_RET(IncrementalCreateChannel(comm, param, resRequest, hostCtxObj, hostCacheTag, resCtxSequence, ctxSize));
    }

    HCCL_INFO("Execute GetAlgResAICPU success.");
    return HCCL_SUCCESS;
}

HcclResult HcclMemcpyCtxHostToDevice(
    HcclComm comm, const OpParam& param, const std::vector<char>& seq, void** resCtxSequence, uint64_t& ctxSize)
{
    uint64_t size = seq.size();
    void* ctx = nullptr;
    // 创建Context, aicpu和host dpu申请device内存
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, COMM_ENGINE_AICPU_TS, size, &ctx));
    // 从Host内存拷贝到Device Context内存上
    CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_AICPU_TS, param.algTag, seq.data(), size, 0));
    // 将内存强转为AlgResourceCtx结构体
    *resCtxSequence = ctx;
    ctxSize = size;
    HCCL_INFO("Memcpy hostCtx to device success.");
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceAICPU(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode& resPack)
{
    HCCL_INFO("Start to execute AllocAlgResource.");
    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    UpdateAicpuTimeoutCtx(param, *resCtxHost);
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;
    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost, resPack));
    CHK_RET(HcclGetChannel(comm, param, resRequest, resCtxHost.get()));
    return HCCL_SUCCESS;
}

static HcclResult HcclGetThreadWithConfig(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, u32 threadNum,
    std::vector<ThreadHandle>& threads, std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, bool unfoldReady)
{
    std::vector<ThreadConfig> threadConfigs(threadNum);
    CHK_RET(static_cast<HcclResult>(ThreadConfigInit(threadConfigs.data(), threadNum)));
    threadConfigs[0].notifyNumPerThread = resRequest.notifyNumOnMainThread + 1; // 主流上多一个用于host-device同步
    HCCL_DEBUG("[HcclGetThread] AICPU thread[0] notify num[%u].", threadConfigs[0].notifyNumPerThread);
    CHK_PRT_RET(
        resRequest.notifyNumPerThread.size() < threadNum - 1,
        HCCL_ERROR(
            "[HcclGetThread] notifyNumPerThread size[%zu] is less than slaveThreadNum[%u].",
            resRequest.notifyNumPerThread.size(), threadNum - 1),
        HCCL_E_INTERNAL);
    for (u32 i = 1; i < threadNum; i++) {
        threadConfigs[i].notifyNumPerThread = resRequest.notifyNumPerThread[i - 1];
        HCCL_DEBUG("[HcclGetThread] AICPU thread[%u] notify num[%u].", i, threadConfigs[i].notifyNumPerThread);
    }
    CHK_RET(HcclThreadAcquireWithConfig(
        comm, COMM_ENGINE_AICPU, threadNum, THREAD_TYPE_TS, threadConfigs.data(), threads.data()));
    // 申请展开流对应的Thread
    if (!unfoldReady) {
        ThreadConfig unfoldThreadConfig;
        CHK_RET(static_cast<HcclResult>(ThreadConfigInit(&unfoldThreadConfig, 1)));
        // 展开流需要一个Notify用于 AICPU按序下发
        unfoldThreadConfig.notifyNumPerThread = ORDER_UNFOLD_THREAD_NOTIFY_NUM;
        CHK_RET(HcclThreadAcquireWithConfig(
            comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS, &unfoldThreadConfig, &resCtxHost->unfoldThread));
    }
    CHK_RET(SaveMainThreadInfo(comm, param, threads[0], resRequest.notifyNumOnMainThread + 1));
    return HCCL_SUCCESS;
}

static u32 GetMaxNotifyNum(const std::vector<u32>& notifyNumPerThread, u32 initNotifyNum)
{
    u32 maxNotifyNum = initNotifyNum;
    for (u32 notifyNum : notifyNumPerThread) {
        if (notifyNum > maxNotifyNum) {
            maxNotifyNum = notifyNum;
        }
    }
    return maxNotifyNum;
}

static HcclResult HcclGetAicpuThread(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    u32 threadNum = resRequest.slaveThreadNum + 1;
    std::vector<ThreadHandle> threads(threadNum);
    bool unfoldReady = false;
    ThreadHandle existingUnfoldThread = 0;
    if (GetUnfoldThreadInfo(comm, param, existingUnfoldThread) == HCCL_SUCCESS) {
        resCtxHost->unfoldThread = existingUnfoldThread;
        unfoldReady = true;
        HCCL_INFO("[HcclGetThread] reuse unfoldThread [%lu]", resCtxHost->unfoldThread);
    }
    if (HcommIsSupportHcclThreadAcquireWithConfig()) {
        CHK_RET(HcclGetThreadWithConfig(comm, param, resRequest, threadNum, threads, resCtxHost, unfoldReady));
    } else {
        u32 maxNotifyNum = GetMaxNotifyNum(resRequest.notifyNumPerThread, resRequest.notifyNumOnMainThread);
        HCCL_DEBUG("[HcclGetThread] require maxNotifyNum[%u] for all AICPU threads.", maxNotifyNum);
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, threadNum, maxNotifyNum + 1, threads.data()));
        if (!unfoldReady) {
            // 展开流需要一个Notify用于 AICPU按序下发
            CHK_RET(
                HcclThreadAcquire(comm, COMM_ENGINE_CPU, 1, ORDER_UNFOLD_THREAD_NOTIFY_NUM, &resCtxHost->unfoldThread));
        }
        CHK_RET(SaveMainThreadInfo(comm, param, threads[0], maxNotifyNum + 1));
    }
    if (!unfoldReady) {
        CHK_RET(SaveUnfoldThreadInfo(comm, param, resCtxHost->unfoldThread));
    }
    HCCL_INFO("[HcclGetThread] unfoldThread [%lu]", resCtxHost->unfoldThread);
    HCCL_DEBUG("threads ptr is %p\n", threads.data());
    for (u32 i = 0; i < threadNum; i++) {
        resCtxHost->threads.push_back(threads[i]);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetThread(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode& resPack)
{
    resCtxHost->isHcclThreadAcquireWithConfigSupported = HcommIsSupportHcclThreadAcquireWithConfig();
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        CHK_RET(HcclGetAicpuThread(comm, param, resRequest, resCtxHost));
    } else {
        // host模式下，将主流封装为thread，并创建主流上的notify
        ThreadHandle thread;
        CHK_RET(
            HcclThreadAcquireWithStream(comm, param.engine, param.stream, resRequest.notifyNumOnMainThread, &thread));
        resCtxHost->threads.push_back(thread);

        u32 maxNotifyNum = GetMaxNotifyNum(resRequest.notifyNumPerThread, 0);
        CHK_RET(GeGetThread(comm, param, resRequest, resCtxHost, resPack, maxNotifyNum));
    }

    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        HCCL_DEBUG("[HcclGetThread] slaveThreadNum[%u]", resRequest.slaveThreadNum);
        for (u32 i = 0; i < resRequest.slaveThreadNum + 1; i++) {
            HCCL_DEBUG("[HcclGetThread] threads[%u]=[%llu]", i, resCtxHost->threads[i]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult GeGetThread(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode& resPack, u32 maxNotifyNum)
{
    if (param.opMode == OpMode::OPBASE) {
        u32 threadNum = resRequest.slaveThreadNum;
        if (threadNum > 0) {
            std::vector<ThreadHandle> threads(threadNum);
            if (HcommIsSupportHcclThreadAcquireWithConfig()) {
                std::vector<ThreadConfig> threadConfigs(threadNum);
                CHK_RET(static_cast<HcclResult>(ThreadConfigInit(threadConfigs.data(), threadNum)));
                CHK_PRT_RET(
                    resRequest.notifyNumPerThread.size() < threadNum,
                    HCCL_ERROR(
                        "[GeGetThread] notifyNumPerThread size[%zu] is less than slaveThreadNum[%u].",
                        resRequest.notifyNumPerThread.size(), threadNum),
                    HCCL_E_INTERNAL);
                for (u32 i = 0; i < threadNum; i++) {
                    threadConfigs[i].notifyNumPerThread = resRequest.notifyNumPerThread[i];
                }
                CHK_RET(HcclThreadAcquireWithConfig(
                    comm, COMM_ENGINE_CPU, threadNum, THREAD_TYPE_TS, threadConfigs.data(), threads.data()));
            } else {
                CHK_RET(HcclThreadAcquire(comm, param.engine, threadNum, maxNotifyNum, threads.data()));
            }
            for (u32 i = 0; i < threadNum; i++) {
                resCtxHost->threads.push_back(threads[i]);
            }
        }
    } else {
        u32 slaveStreams = resPack.streams.size();
        u32 threadNum = resRequest.slaveThreadNum;
        if (threadNum > slaveStreams) {
            HCCL_ERROR(
                "Thread Num Should less than slave streams. slaveStreams[%llu], threadNums[%llu]", slaveStreams,
                threadNum);
            return HCCL_E_UNAVAIL;
        }

        for (u32 i = 0; i < threadNum; i++) {
            ThreadHandle slaveThread;
            CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, resPack.streams[i], maxNotifyNum, &slaveThread));
            resCtxHost->threads.push_back(slaveThread);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult SaveMainThreadInfo(HcclComm comm, const OpParam& param, ThreadHandle thread, u32 notifyNum)
{
    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);
    void* ctx = nullptr;
    // 申请一块host类型内存，保存主流信息
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    // 填充主流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
    *threadPtr = thread;
    // 填充主流notify数量信息
    char* curPtr = reinterpret_cast<char*>(ctx);
    curPtr += sizeof(ThreadHandle);
    u32* notifyNumPtr = reinterpret_cast<u32*>(curPtr);
    *notifyNumPtr = notifyNum;
    HCCL_INFO(
        "[SaveMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]", threadPtr, thread,
        notifyNumPtr, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult SaveUnfoldThreadInfo(HcclComm comm, const OpParam& param, ThreadHandle unfoldThread)
{
    uint64_t size = sizeof(ThreadHandle);
    void* ctx = nullptr;
    // 申请一块host类型内存，保存展开流信息
    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, "%s_unfold", param.commName);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill unfoldAlgTag", __func__), HCCL_E_INTERNAL);
    CHK_RET(HcclEngineCtxCreate(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    // 填充展开流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
    *threadPtr = unfoldThread;
    HCCL_INFO(
        "[SaveUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]", unfoldAlgTag, threadPtr,
        unfoldThread);
    return HCCL_SUCCESS;
}

HcclResult GetUnfoldThreadInfo(HcclComm comm, const OpParam& param, ThreadHandle& unfoldThread)
{
    uint64_t size = sizeof(ThreadHandle);
    void* ctx = nullptr;
    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, "%s_unfold", param.commName);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill unfoldAlgTag", __func__), HCCL_E_INTERNAL);
    HcclResult getRet = HcclEngineCtxGet(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size);
    if (getRet != HCCL_SUCCESS) {
        return getRet;
    }
    // 获取展开流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
    unfoldThread = *threadPtr;
    HCCL_INFO(
        "[GetUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]", unfoldAlgTag, threadPtr,
        unfoldThread);
    return HCCL_SUCCESS;
}

HcclResult GetMainThreadInfo(HcclComm comm, const OpParam& param, ThreadHandle& thread, u32& notifyNum)
{
    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);
    void* ctx = nullptr;
    CHK_RET(HcclEngineCtxGet(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size));

    // 获取主流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle*>(ctx);
    thread = *threadPtr;
    // 获取主流notify数量信息
    char* curPtr = reinterpret_cast<char*>(ctx);
    curPtr += sizeof(ThreadHandle);
    u32* notifyNumPtr = reinterpret_cast<u32*>(curPtr);
    notifyNum = *notifyNumPtr;
    HCCL_INFO(
        "[GetMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]", threadPtr, thread,
        notifyNumPtr, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannel(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, AlgResourceCtxSerializable* resCtxHost)
{
    MemRegInfo memRegInfo;
    if (param.opMode == OpMode::OFFLOAD) {
        HCCL_INFO("[HcclGetChannelImpl] start to RegGraphModeBuffers");
        CHK_RET(
            RegGraphModeBuffers(comm, param, memRegInfo.inputBuffTag, memRegInfo.outputBuffTag, memRegInfo.memHandles));
    }
    resCtxHost->channels.resize(resRequest.channels.size());
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc>& levelNChannelRequest = resRequest.channels[level];
        std::vector<HcclChannelDesc> deviceChannelRequest;
        std::vector<HcclChannelDesc> hostChannelRequest;
        for (auto& channelRequest : levelNChannelRequest) {
            if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                deviceChannelRequest.emplace_back(channelRequest);
            } else if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                hostChannelRequest.emplace_back(channelRequest);
            }
        }
        // device建链
        CHK_RET(
            HcclGetChannelImpl(level, comm, param, deviceChannelRequest, COMM_ENGINE_AICPU_TS, resCtxHost, memRegInfo));
        // host建链
        CHK_RET(HcclGetChannelImpl(level, comm, param, hostChannelRequest, COMM_ENGINE_CPU, resCtxHost, memRegInfo));
    }
    return HCCL_SUCCESS;
}

static HcclResult BuildChannelInfo(
    HcclComm comm, const OpParam& param, const HcclChannelDesc& channelDesc, ChannelHandle channelHandle, u32 userRank,
    MemRegInfo& memRegInfo, ChannelInfo& channel)
{
    // 对于真实建链的链路进行填充
    channel.isValid = true;
    channel.remoteRank = channelDesc.remoteRank;
    channel.protocol = channelDesc.channelProtocol;
    channel.locationType = channelDesc.remoteEndpoint.loc.locType;
    channel.notifyNum = channelDesc.notifyNum;
    channel.handle = channelHandle;
#ifndef AICPU_COMPILE
    EndpointDesc localEndpoint = channelDesc.localEndpoint;
    using portSizeType = uint32_t;
    const uint32_t portSizeTypeSize = sizeof(portSizeType);
    portSizeType portSize = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, userRank, &localEndpoint, ENDPOINT_ATTR_BW_COEFF, portSizeTypeSize, static_cast<void*>(&portSize)));
    channel.portGroupSize = portSize;
    CHK_PRT_RET(
        portSize == 0, HCCL_ERROR("[HcclGetChannelImpl] userRank [%d], portSize [%u] is 0.", userRank, portSize),
        HcclResult::HCCL_E_INTERNAL);
    EndpointAttrDieId dieId = INVALID_VALUE_RANKID;
    const uint32_t dieIdSize = sizeof(EndpointAttrDieId);
    HcclResult dieIdRet = HcclRankGraphGetEndpointInfo(
        comm, userRank, &localEndpoint, ENDPOINT_ATTR_DIE_ID, dieIdSize, static_cast<void*>(&dieId));
    if (dieIdRet == HCCL_SUCCESS) {
        channel.dieId = dieId;
    } else {
        HCCL_WARNING(
            "[HcclGetChannelImpl] failed to get dieId for userRank[%u], remoteRank[%u], "
            "ret[0x%016llx]. POD convergence adjustment will not be used for this channel.",
            userRank, channel.remoteRank, HCCL_ERROR_CODE(dieIdRet));
    }
#endif
    void* remoteCclBufferAddr = nullptr;
    uint64_t remoteCclBufferSize = 0;
    CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandle, &remoteCclBufferAddr, &remoteCclBufferSize));
    channel.remoteCclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteCclBufferAddr, remoteCclBufferSize};
    HCCL_INFO(
        "[%s]remoteRank[%u] protocol[%u] portGroupSize[%u] dieId[%u] "
        "remoteCclBufferAddr[0x%llx] remoteCclBufferSize[%u]",
        __func__, channelDesc.remoteRank, channelDesc.channelProtocol, channel.portGroupSize, channel.dieId,
        remoteCclBufferAddr, remoteCclBufferSize);

    if (param.opMode == OpMode::OFFLOAD) {
        CHK_RET(GetGraphModeBuffers(comm, channelHandle, memRegInfo.inputBuffTag, memRegInfo.outputBuffTag, channel));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannelImpl(
    const u32 level, HcclComm comm, const OpParam& param, std::vector<HcclChannelDesc>& channelRequest,
    const CommEngine commEngine, AlgResourceCtxSerializable* resCtxHost, MemRegInfo& memRegInfo)
{
    // 获取子通信域的建链数量
    if (channelRequest.empty()) {
        HCCL_INFO("[HcclGetChannelImpl] channelRequest is empty");
        return HCCL_SUCCESS;
    }
    u32 channelNum = channelRequest.size();
    std::vector<ChannelHandle> levelNChannels;
    levelNChannels.resize(channelNum);
    if (param.opMode == OpMode::OFFLOAD) {
        for (auto& channelDesc : channelRequest) {
            channelDesc.memHandles = memRegInfo.memHandles.data();
            channelDesc.memHandleNum = memRegInfo.memHandles.size();
        }
    }
    if (channelNum > 0) {
        // 参数一致性校验信息注册到通信域，HcclChannelAcquire内部存在读清动作，每次调用前均需注册
        CHK_RET(AddExchangeInfo(comm, param));
        CHK_RET(HcclChannelAcquire(comm, commEngine, channelRequest.data(), channelNum, levelNChannels.data()));
    }

    for (u32 idx = 0; idx < channelNum; idx++) {
        ChannelInfo channel;
        CHK_RET(BuildChannelInfo(
            comm, param, channelRequest[idx], levelNChannels[idx], resCtxHost->topoInfo.userRank, memRegInfo, channel));
        resCtxHost->channels[level].push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult RegGraphModeBuffers(
    HcclComm comm, const OpParam& param, char* inputBuffTag, char* outputBuffTag,
    std::vector<HcclMemHandle>& memHandles)
{
    HCCL_INFO("[RegGraphModeBuffers] param.tag[%s]", param.tag);
    auto retIn = sprintf_s(inputBuffTag, MAX_MEM_TAG_LENGTH, "%s_%s", param.tag, "InputBuffer");
    auto retOut = sprintf_s(outputBuffTag, MAX_MEM_TAG_LENGTH, "%s_%s", param.tag, "OutputBuffer");
    if (retIn <= 0 || retOut <= 0) {
        HCCL_ERROR("[RegGraphModeBuffers]failed to fill BuffTag");
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_INFO("[RegGraphModeBuffers] graph mode regstry remote buuffer");
    if (param.inputPtr != nullptr && param.inputSize != 0) {
        HcclMemHandle inputHandle = nullptr;
        CHK_RET(HcclRegstryBuff(comm, inputBuffTag, param.inputPtr, param.inputSize, &inputHandle));
        CHK_PTR_NULL(inputHandle);
        memHandles.emplace_back(inputHandle);
    }
    if (param.outputPtr != nullptr && param.outputSize != 0) {
        HcclMemHandle outputHandle = nullptr;
        CHK_RET(HcclRegstryBuff(comm, outputBuffTag, param.outputPtr, param.outputSize, &outputHandle));
        CHK_PTR_NULL(outputHandle);
        memHandles.emplace_back(outputHandle);
    }
    HCCL_INFO("[RegGraphModeBuffers]memHandles size[%d]", memHandles.size());
    return HCCL_SUCCESS;
}

HcclResult GetGraphModeBuffers(
    HcclComm comm, ChannelHandle channelHandle, const char* inputBuffTag, const char* outputBuffTag,
    ChannelInfo& channel)
{
    void* remoteInputBufferAddr = nullptr;
    uint64_t remoteInputBufferSize = 0;
    CHK_RET(HcclGetRemoteBuff(comm, channelHandle, inputBuffTag, &remoteInputBufferAddr, &remoteInputBufferSize));
    if (remoteInputBufferAddr != nullptr && remoteInputBufferSize > 0) {
        channel.remoteInputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteInputBufferAddr, remoteInputBufferSize};
    }

    void* remoteOutputBufferAddr = nullptr;
    uint64_t remoteOutputBufferSize = 0;
    CHK_RET(HcclGetRemoteBuff(comm, channelHandle, outputBuffTag, &remoteOutputBufferAddr, &remoteOutputBufferSize));
    if (remoteOutputBufferAddr != nullptr && remoteOutputBufferSize > 0) {
        channel.remoteOutputGraphMode = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteOutputBufferAddr, remoteOutputBufferSize};
    }
    return HCCL_SUCCESS;
}

HcclResult GetAlgResCcu(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& ctxSize,
    const ResPackGraphMode& resPack)
{
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    // 创建资源，并填充到Host内存上
    HcclResult ret = HcclAllocAlgResourceCcu(comm, param, resRequest, resCtxHost, resPack);
    if (ret == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[HcclAllocAlgResourceCcu] resource unavailable, try to fallback.");
        return HCCL_E_UNAVAIL;
    } else if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("failed to alloc alg resource.");
        return ret;
    }
    // 序列化
    std::vector<char> seq = resCtxHost->Serialize();
    uint64_t size = seq.size();

    void* ctx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, param.engine, size, &ctx));
    CHK_SAFETY_FUNC_RET(memcpy_s(ctx, size, seq.data(), size));
    *resCtxSequence = ctx;
    ctxSize = size;
    HCCL_INFO("Execute GetAlgResCCU success.");
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceCcu(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode& resPack)
{
    HCCL_INFO("Start to execute AllocAlgResource.");
    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;
    resCtxHost->dieSplitRatio = resRequest.dieSplitRatio;
    CHK_RET(HcclGetThread(comm, param, resRequest, resCtxHost, resPack));
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    // 资源回退
    auto ret = HcclGetChannelForCcu(comm, param, resRequest);
    if (ret == HCCL_E_UNAVAIL) {
        // 进行资源回退
        HCCL_WARNING("[HcclGetChannelForCcu] channel unavailable, try to fallback.");
        return HCCL_E_UNAVAIL;
    } else {
        CHK_RET(ret);
    }

    // opMode 透传：CCU_MS 与 CCU_SCHED 使用不同的默认资源阈值（MS 模式 LOOP/CCU_BUF 阈值更大）
    ret = HcclGetCcuKernel(comm, param.commOpExpansionMode, resRequest, resCtxHost);
    if (ret == HCCL_E_UNAVAIL) {
        // 资源不足导致回退：打印算子信息方便定位
        HCCL_RUN_INFO(
            "[HcclGetCcuKernel] ccu resource insufficient, fallback to AICPU. "
            "algTag[%s], algName[%s], opType[%u], kernelNum[%zu], kernel[0] name[%s].",
            param.algTag, param.algName, static_cast<uint32_t>(param.opType), resRequest.ccuKernelInfos.size(),
            resRequest.ccuKernelInfos.empty() ? "N/A" : resRequest.ccuKernelInfos[0].kernelFuncName);
        return HCCL_E_UNAVAIL;
    } else {
        CHK_RET(ret);
    }
#endif
    return HCCL_SUCCESS;
}

#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
HcclResult HcclGetChannelForCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest)
{
    // OpParam.userRank 并非所有算子路径都会赋值（仅 Reduce 赋值），这里直接从 comm 查询本端全局 rank
    u32 userRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &userRank));

    // 以kernel为粒度申请channel
    for (CcuKernelInfo& kernelInfo : resRequest.ccuKernelInfos) {
        std::vector<HcclChannelDesc>& kernelChannelRequest = kernelInfo.channels;

        u32 channelNum = kernelChannelRequest.size();
        std::vector<ChannelHandle> kernelChannels;
        kernelChannels.resize(channelNum);

        if (channelNum > 0) {
            // 参数一致性校验信息注册到通信域，HcclChannelAcquire内部存在读清动作，每次调用前均需注册
            CHK_RET(AddExchangeInfo(comm, param));
            auto ret = HcclChannelAcquire(
                comm, param.engine, kernelChannelRequest.data(), channelNum, kernelChannels.data());
            // 需要资源回退。返回资源不够
            if (ret == HCCL_E_UNAVAIL) {
                HCCL_WARNING("[HcclChannelAcquire] channel unavailable, channel num[%u].", channelNum);
                return HCCL_E_UNAVAIL;
            } else {
                CHK_RET(ret);
            }
            // 从首条channel获取dieId，作为kernel所属dieId保存（同一kernel的所有channel在同一die上）
            EndpointDesc localEndpoint = kernelChannelRequest[0].localEndpoint;
            using DieIdType = uint32_t;
            const uint32_t dieIdTypeSize = sizeof(DieIdType);
            DieIdType dieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(
                comm, userRank, &localEndpoint, ENDPOINT_ATTR_DIE_ID, dieIdTypeSize, static_cast<void*>(&dieId)));
            kernelInfo.dieId = dieId;
        }
        auto* kernelArgBase = static_cast<CcuKernelArgBase*>(kernelInfo.kernelArg);
        if (!kernelArgBase) {
            HCCL_ERROR("[HcclGetChannelForCcu] kernelArg ptr is err.");
            return HCCL_E_INTERNAL;
        }
        for (u32 i = 0; i < channelNum; ++i) {
            kernelArgBase->channels[i] = kernelChannels[i];
        }
        kernelArgBase->channelCount = channelNum;
        HCCL_INFO("[HcclGetChannelForCcu] Get [%lu] channels, dieId[%u]", channelNum, kernelInfo.dieId);
    }
    return HCCL_SUCCESS;
}

static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_ADDRESS = 400;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP = 16;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF = 128;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_VARIABLE = 400;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT = 48;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_THREAD = 2;

static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP_MS = 128;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF_MS = 1024;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT_MS = 160;

// V2(960) 资源规格，与 V1 一致的也单独定义，方便后续独立调整
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_ADDRESS_V2 = 0;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP_V2 = 16;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF_V2 = 128;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_VARIABLE_V2 = 800;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT_V2 = 64;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_THREAD_V2 = 2;

static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_LOOP_MS_V2 = 128;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_CCU_BUF_MS_V2 = 1024;
static constexpr uint32_t CCU_DEFAULT_RES_FRACTION_EVENT_MS_V2 = 400;

// 即使本算子未在所有 die 上下 kernel，
// 也需为所有 die 创建 reqDesc，保证后续算子在该 die 上有 kernel 时容量充足。
static constexpr uint32_t CCU_DEFAULT_DIE_NUM = 2;

static const std::vector<HcommCcuResType>& GetCcuResTypes()
{
    static const std::vector<HcommCcuResType> types = {
        HCOMM_CCU_RES_TYPE_LOOP,        HCOMM_CCU_RES_TYPE_CCU_BUF, HCOMM_CCU_RES_TYPE_VARIABLE,
        HCOMM_CCU_RES_TYPE_ADDRESS,     HCOMM_CCU_RES_TYPE_EVENT,   HCOMM_CCU_RES_TYPE_CCU_THREAD,
        HCOMM_CCU_RES_TYPE_INSTRUCTION,
    };
    return types;
}

// 实例创建相关的资源类型列表（不含 INSTRUCTION）。
// INSTRUCTION 仅用于查询
static const std::vector<HcommCcuResType>& GetCcuInsCreateResTypes()
{
    static const std::vector<HcommCcuResType> types = {
        HCOMM_CCU_RES_TYPE_LOOP,    HCOMM_CCU_RES_TYPE_CCU_BUF, HCOMM_CCU_RES_TYPE_VARIABLE,
        HCOMM_CCU_RES_TYPE_ADDRESS, HCOMM_CCU_RES_TYPE_EVENT,   HCOMM_CCU_RES_TYPE_CCU_THREAD,
    };
    return types;
}

// V1(950) 默认资源规格
static uint32_t GetDefaultResFractionV1(HcommCcuResType resType, HcclOpExpansionMode opExpansionMode)
{
    bool isCcuMs = (opExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_CCU_MS);
    switch (resType) {
        case HCOMM_CCU_RES_TYPE_ADDRESS:
            return CCU_DEFAULT_RES_FRACTION_ADDRESS;
        case HCOMM_CCU_RES_TYPE_LOOP:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_LOOP_MS : CCU_DEFAULT_RES_FRACTION_LOOP;
        case HCOMM_CCU_RES_TYPE_CCU_BUF:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_CCU_BUF_MS : CCU_DEFAULT_RES_FRACTION_CCU_BUF;
        case HCOMM_CCU_RES_TYPE_VARIABLE:
            return CCU_DEFAULT_RES_FRACTION_VARIABLE;
        case HCOMM_CCU_RES_TYPE_EVENT:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_EVENT_MS : CCU_DEFAULT_RES_FRACTION_EVENT;
        case HCOMM_CCU_RES_TYPE_CCU_THREAD:
            return CCU_DEFAULT_RES_FRACTION_CCU_THREAD;
        default:
            return 0;
    }
}

// V2(960) 默认资源规格
static uint32_t GetDefaultResFractionV2(HcommCcuResType resType, HcclOpExpansionMode opExpansionMode)
{
    bool isCcuMs = (opExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_CCU_MS);
    switch (resType) {
        case HCOMM_CCU_RES_TYPE_ADDRESS:
            return CCU_DEFAULT_RES_FRACTION_ADDRESS_V2;
        case HCOMM_CCU_RES_TYPE_LOOP:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_LOOP_MS_V2 : CCU_DEFAULT_RES_FRACTION_LOOP_V2;
        case HCOMM_CCU_RES_TYPE_CCU_BUF:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_CCU_BUF_MS_V2 : CCU_DEFAULT_RES_FRACTION_CCU_BUF_V2;
        case HCOMM_CCU_RES_TYPE_VARIABLE:
            return CCU_DEFAULT_RES_FRACTION_VARIABLE_V2;
        case HCOMM_CCU_RES_TYPE_EVENT:
            return isCcuMs ? CCU_DEFAULT_RES_FRACTION_EVENT_MS_V2 : CCU_DEFAULT_RES_FRACTION_EVENT_V2;
        case HCOMM_CCU_RES_TYPE_CCU_THREAD:
            return CCU_DEFAULT_RES_FRACTION_CCU_THREAD_V2;
        default:
            return 0;
    }
}

// 根据设备类型分发：960 走 V2，其余走 V1
// opExpansionMode 为 CCU_MS 时 LOOP/CCU_BUF/EVENT 使用 MS 模式专用阈值，其他资源类型与其他模式保持一致
static uint32_t GetDefaultResFraction(HcommCcuResType resType, HcclOpExpansionMode opExpansionMode)
{
    HcclDevType deviceType;
    bool isV2 = (HcclGetDeviceType(deviceType) == HCCL_SUCCESS) && (deviceType == HcclDevType::DEV_TYPE_960);
    return isV2 ? GetDefaultResFractionV2(resType, opExpansionMode) : GetDefaultResFractionV1(resType, opExpansionMode);
}

// 将 HcommCcuResType 转字符串
static const char* GetCcuResTypeName(HcommCcuResType resType)
{
    switch (resType) {
        case HCOMM_CCU_RES_TYPE_LOOP:
            return "LOOP";
        case HCOMM_CCU_RES_TYPE_CCU_BUF:
            return "CCU_BUF";
        case HCOMM_CCU_RES_TYPE_VARIABLE:
            return "VARIABLE";
        case HCOMM_CCU_RES_TYPE_ADDRESS:
            return "ADDRESS";
        case HCOMM_CCU_RES_TYPE_EVENT:
            return "EVENT";
        case HCOMM_CCU_RES_TYPE_CCU_THREAD:
            return "CCU_THREAD";
        case HCOMM_CCU_RES_TYPE_INSTRUCTION:
            return "INSTRUCTION";
        default:
            return "UNKNOWN";
    }
}

static bool IsCcuDynamicResApiSupported()
{
    return HcommIsSupportHcommCcuInsResDescCreate() && HcommIsSupportHcommCcuInsResDescDestroy()
           && HcommIsSupportHcommCcuInsResDescSetNum() && HcommIsSupportHcommCcuInsResDescQueryNum()
           && HcommIsSupportHcommCcuInsCreate() && HcommIsSupportHcommCcuInsDestroy()
           && HcommIsSupportHcommCcuInsQueryResDesc() && HcommIsSupportHcommCcuQueryRemainResDesc()
           && HcommIsSupportHcommCcuKernelQueryResReq() && HcommIsSupportHcclCommAssignCcuIns()
           && HcommIsSupportHcclCommQueryAssignedCcuIns();
}

// 按 dieId 维护资源描述符集合；HcommCcuInsResDescCreate 接口要求每个 desc 必须绑定一个 dieId，
// 因此同一 die 上多个 kernel 的需求聚合到同一个 desc，不同 die 各自维护独立 desc。
using ResDescByDie = std::map<uint32_t, HcommCcuResDescHandle>;

// 销毁集合中所有 desc 并清空，避免资源泄漏
static void DestroyAllDescs(ResDescByDie& descs)
{
    for (auto& kv : descs) {
        if (kv.second != 0) {
            HcommCcuInsResDescDestroy(kv.second);
            kv.second = 0;
        }
    }
    descs.clear();
}

// 查询单个 kernel 的资源需求，按 (dieId, resGroup, resType) 累加到 groupedResMap。
static HcclResult AccumulateKernelRes(
    const CcuKernelInfo& kernelInfo,
    std::map<uint32_t, std::map<u32, std::map<HcommCcuResType, uint32_t>>>& groupedResMap)
{
    HcommCcuResDescHandle kernelDesc = 0;
    CcuResult createRet = HcommCcuInsResDescCreate(kernelInfo.dieId, &kernelDesc);
    CHK_PRT_RET(
        createRet != CCU_SUCCESS,
        HCCL_ERROR(
            "[AccumulateKernelRes] HcommCcuInsResDescCreate dieId[%u] failed: ccuRet -> %d", kernelInfo.dieId,
            createRet),
        ConvertCcuToHccl(createRet));

    const void* kernelArgs[] = {kernelInfo.kernelArg};
    constexpr uint32_t kernelArgNum = 1;
    CcuResult queryRet = HcommCcuKernelQueryResReq(
        reinterpret_cast<void*>(kernelInfo.kernelFunc), kernelArgs, kernelArgNum, kernelDesc);
    if (queryRet != CCU_SUCCESS) {
        HCCL_ERROR("[AccumulateKernelRes] HcommCcuKernelQueryResReq failed: ccuRet -> %d", queryRet);
        HcommCcuInsResDescDestroy(kernelDesc);
        return ConvertCcuToHccl(queryRet);
    }

    for (HcommCcuResType resType : GetCcuResTypes()) {
        uint32_t resNum = 0;
        CcuResult qRet = HcommCcuInsResDescQueryNum(kernelDesc, resType, &resNum);
        if (qRet != CCU_SUCCESS) {
            HCCL_ERROR("[AccumulateKernelRes] HcommCcuInsResDescQueryNum failed: ccuRet -> %d", qRet);
            HcommCcuInsResDescDestroy(kernelDesc);
            return ConvertCcuToHccl(qRet);
        }
        HCCL_INFO(
            "[AccumulateKernelRes] kernel[%s] dieId[%u] resGroup[%u] resType[%s] resNum[%u].",
            kernelInfo.kernelFuncName, kernelInfo.dieId, kernelInfo.resGroup, GetCcuResTypeName(resType), resNum);
        groupedResMap[kernelInfo.dieId][kernelInfo.resGroup][resType] += resNum;
    }
    HcommCcuInsResDescDestroy(kernelDesc);
    return HCCL_SUCCESS;
}

static HcclResult IsResCapSufficient(
    uint32_t dieId, HcommCcuResDescHandle resCap, HcommCcuResDescHandle resReq, bool& sufficient,
    std::string& insuffSummary)
{
    sufficient = true;
    insuffSummary.clear();
    HCCL_INFO("[IsResCapSufficient] start, dieId[%u].", dieId);
    for (HcommCcuResType resType : GetCcuInsCreateResTypes()) {
        uint32_t capNum = 0;
        uint32_t reqNum = 0;
        CcuResult capRet = HcommCcuInsResDescQueryNum(resCap, resType, &capNum);
        if (capRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[IsResCapSufficient] dieId[%u] query cap failed, resType[%s]: ccuRet -> %d", dieId,
                GetCcuResTypeName(resType), capRet);
            return ConvertCcuToHccl(capRet);
        }
        CcuResult reqRet = HcommCcuInsResDescQueryNum(resReq, resType, &reqNum);
        if (reqRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[IsResCapSufficient] dieId[%u] query req failed, resType[%s]: ccuRet -> %d", dieId,
                GetCcuResTypeName(resType), reqRet);
            return ConvertCcuToHccl(reqRet);
        }
        HCCL_INFO(
            "[IsResCapSufficient] dieId[%u] resType[%s] cap[%u] req[%u] %s.", dieId, GetCcuResTypeName(resType), capNum,
            reqNum, capNum >= reqNum ? "sufficient" : "insufficient");
        if (capNum < reqNum) {
            sufficient = false;
            if (!insuffSummary.empty()) {
                insuffSummary += ",";
            }
            insuffSummary += std::string(GetCcuResTypeName(resType)) + "(need=" + std::to_string(reqNum)
                             + ",remain=" + std::to_string(capNum) + ")";
        }
    }
    HCCL_INFO(
        "[IsResCapSufficient] dieId[%u] %s.", dieId,
        sufficient ? "all resTypes sufficient" : "some resTypes insufficient");
    return HCCL_SUCCESS;
}

static HcclResult CalcMaxResReqWithDefault(
    uint32_t dieId, HcclOpExpansionMode opExpansionMode, HcommCcuResDescHandle resReq, HcommCcuResDescHandle outMax)
{
    HCCL_INFO(
        "[CalcMaxResReqWithDefault] start, dieId[%u], opExpansionMode[%u].", dieId,
        static_cast<uint32_t>(opExpansionMode));
    for (HcommCcuResType resType : GetCcuInsCreateResTypes()) {
        uint32_t reqNum = 0;
        CcuResult qRet = HcommCcuInsResDescQueryNum(resReq, resType, &reqNum);
        if (qRet != CCU_SUCCESS) {
            HCCL_ERROR("[CalcMaxResReqWithDefault] dieId[%u] query req failed: ccuRet -> %d", dieId, qRet);
            return ConvertCcuToHccl(qRet);
        }
        uint32_t defaultNum = GetDefaultResFraction(resType, opExpansionMode);
        uint32_t maxNum = std::max(reqNum, defaultNum);
        CcuResult setRet = HcommCcuInsResDescSetNum(outMax, resType, maxNum);
        if (setRet != CCU_SUCCESS) {
            HCCL_ERROR("[CalcMaxResReqWithDefault] dieId[%u] set failed: ccuRet -> %d", dieId, setRet);
            return ConvertCcuToHccl(setRet);
        }
        HCCL_INFO(
            "[CalcMaxResReqWithDefault] dieId[%u] resType[%s] req[%u] default[%u] -> max[%u].", dieId,
            GetCcuResTypeName(resType), reqNum, defaultNum, maxNum);
    }
    HCCL_INFO("[CalcMaxResReqWithDefault] dieId[%u] finish.", dieId);
    return HCCL_SUCCESS;
}

static HcclResult RegisterCcuKernels(
    CcuInsHandle insHandle, AlgResourceRequest& resRequest, std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    u32 totalKernelNum = std::accumulate(resRequest.ccuKernelNum.begin(), resRequest.ccuKernelNum.end(), 0u);
    CHK_PRT_RET(
        totalKernelNum != resRequest.ccuKernelInfos.size(), HCCL_ERROR("[RegisterCcuKernels]ccuKernel num not match!"),
        HCCL_E_INTERNAL);
    HCCL_INFO("[RegisterCcuKernels] start, totalKernelNum[%u], insHandle[%p].", totalKernelNum, insHandle);

    // 遍历计算最大resGroup号
    auto maxIt = std::max_element(
        resRequest.ccuKernelInfos.begin(), resRequest.ccuKernelInfos.end(),
        [](const CcuKernelInfo& a, const CcuKernelInfo& b) {
            return a.resGroup < b.resGroup;
        });
    u32 maxResGroup = (maxIt != resRequest.ccuKernelInfos.end()) ? maxIt->resGroup : 0;
    resCtxHost->ccuKernels.resize(totalKernelNum);

    for (u32 currentResGroup = 0; currentResGroup <= maxResGroup; currentResGroup++) {
        HCCL_INFO("[RegisterCcuKernels] register resGroup[%u] start, maxResGroup[%u].", currentResGroup, maxResGroup);
        CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
        CHK_PRT_RET(
            regStartRet != CCU_SUCCESS, HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regStartRet),
            ConvertCcuToHccl(regStartRet));
        for (u32 i = 0; i < totalKernelNum; i++) {
            CcuKernelInfo& kernelInfo = resRequest.ccuKernelInfos[i];
            if (kernelInfo.resGroup != currentResGroup)
                continue;
            CcuKernelHandle kernelHandle;
            const void* kernelArgs[] = {kernelInfo.kernelArg};
            CcuResult regRet = HcommCcuKernelRegister(
                insHandle, kernelInfo.dieId, kernelInfo.kernelFuncName, reinterpret_cast<void*>(kernelInfo.kernelFunc),
                kernelArgs, 1, &kernelHandle);
            if (regRet == CCU_E_UNAVAIL) {
                HCCL_WARNING("[RegisterCcuKernels] kernel[%s] unavailable, fallback.", kernelInfo.kernelFuncName);
                return HCCL_E_UNAVAIL;
            }
            CHK_PRT_RET(
                regRet != CCU_SUCCESS,
                HCCL_ERROR("ccu kernel register failed: ccuRet -> %d, kernel[%s]", regRet, kernelInfo.kernelFuncName),
                ConvertCcuToHccl(regRet));
            resCtxHost->ccuKernels[i] = kernelHandle;
        }
        CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
        CHK_PRT_RET(
            regEndRet != CCU_SUCCESS, HCCL_ERROR("ccu kernel register end failed: ccuRet -> %d", regEndRet),
            ConvertCcuToHccl(regEndRet));
        HCCL_INFO("[RegisterCcuKernels] register resGroup[%u] finish.", currentResGroup);
    }
    resCtxHost->ccuKernelNum = resRequest.ccuKernelNum;
    HCCL_INFO("[RegisterCcuKernels] finish, totalKernelNum[%u] registered.", totalKernelNum);
    return HCCL_SUCCESS;
}

// 探测 die 是否使能：调用 HcommCcuQueryRemainResDesc，返回 CCU_E_UNAVAIL 表示未使能。
// support flag 已在 IsCcuDynamicResApiSupported 中校验，本函数不重复判断。
static HcclResult IsDieEnabledForPadding(uint32_t dieId, bool& enabled)
{
    enabled = false;
    // 创建临时 probe desc 绑定 dieId，查询后立即销毁
    HcommCcuResDescHandle probeDesc = 0;
    CcuResult createRet = HcommCcuInsResDescCreate(dieId, &probeDesc);
    CHK_PRT_RET(
        createRet != CCU_SUCCESS,
        HCCL_ERROR(
            "[IsDieEnabledForPadding] HcommCcuInsResDescCreate dieId[%u] failed: ccuRet -> %d", dieId, createRet),
        ConvertCcuToHccl(createRet));
    CcuResult queryRet = HcommCcuQueryRemainResDesc(probeDesc);
    // 立即销毁 probe desc，避免句柄泄漏
    HcommCcuInsResDescDestroy(probeDesc);
    if (queryRet == CCU_E_UNAVAIL) {
        enabled = false;
        HCCL_INFO("[IsDieEnabledForPadding] dieId[%u] is not enabled (CCU_E_UNAVAIL), skip padding.", dieId);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        queryRet != CCU_SUCCESS,
        HCCL_ERROR(
            "[IsDieEnabledForPadding] HcommCcuQueryRemainResDesc dieId[%u] failed: ccuRet -> %d", dieId, queryRet),
        ConvertCcuToHccl(queryRet));
    enabled = true;
    HCCL_INFO("[IsDieEnabledForPadding] dieId[%u] is enabled, need padding.", dieId);
    return HCCL_SUCCESS;
}

// 聚合所有 kernel 的资源需求到 reqDescs（按 dieId 分组）。
// 聚合规则：同 (dieId, resGroup) 内逐 kernel 相加、同 dieId 不同 resGroup 之间取最大。
// 为硬件上所有使能的 die 创建 reqDesc（未使能 die 跳过），防止后续算子容量不足触发回退。
static HcclResult BuildAggregatedResReq(AlgResourceRequest& resRequest, ResDescByDie& reqDescs)
{
    u32 totalKernelNum = resRequest.ccuKernelInfos.size();
    HCCL_INFO("[BuildAggregatedResReq] start, kernelNum[%u].", totalKernelNum);

    // 按 (dieId, resGroup, resType) 聚合所有 kernel 的资源需求
    std::map<uint32_t, std::map<u32, std::map<HcommCcuResType, uint32_t>>> groupedResMap;
    for (u32 i = 0; i < totalKernelNum; i++) {
        CHK_RET(AccumulateKernelRes(resRequest.ccuKernelInfos[i], groupedResMap));
    }

    // 补齐缺失的 die。区分两类 die：
    // 1. 首算子 kernel 所在的 die：已在 groupedResMap 中，必须申请资源，不判断使能状态。
    // 2. kernel 不所在的 die（groupedResMap 中缺失的 die）：通过 IsDieEnabledForPadding 探测是否使能，
    //    使能才创建空条目，后续 CreateFinalReqDescs 会按默认阈值为其申请资源（防止后续算子回退）；
    //    未使能（单 die 场景）跳过补齐，避免冗余申请。
    for (uint32_t dieId = 0; dieId < CCU_DEFAULT_DIE_NUM; dieId++) {
        if (groupedResMap.find(dieId) != groupedResMap.end())
            continue;
        bool enabled = false;
        CHK_RET(IsDieEnabledForPadding(dieId, enabled));
        if (enabled) {
            groupedResMap[dieId] = {};
            HCCL_INFO("[BuildAggregatedResReq] dieId[%u] has no kernel but enabled, create empty entry.", dieId);
        }
    }

    // 为每个 die 创建独立 reqDesc，写入同 dieId 不同 resGroup 取最大后的资源数
    for (auto& dieEntry : groupedResMap) {
        uint32_t dieId = dieEntry.first;
        auto& resGroupMap = dieEntry.second;
        HcommCcuResDescHandle reqDesc = 0;
        CcuResult createRet = HcommCcuInsResDescCreate(dieId, &reqDesc);
        if (createRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[BuildAggregatedResReq] HcommCcuInsResDescCreate dieId[%u] failed: ccuRet -> %d", dieId, createRet);
            DestroyAllDescs(reqDescs);
            return ConvertCcuToHccl(createRet);
        }
        for (HcommCcuResType resType : GetCcuInsCreateResTypes()) {
            uint32_t maxNum = 0;
            for (auto& groupEntry : resGroupMap) {
                auto it = groupEntry.second.find(resType);
                if (it != groupEntry.second.end() && it->second > maxNum)
                    maxNum = it->second;
            }
            CcuResult setRet = HcommCcuInsResDescSetNum(reqDesc, resType, maxNum);
            if (setRet != CCU_SUCCESS) {
                HCCL_ERROR("[BuildAggregatedResReq] HcommCcuInsResDescSetNum failed: ccuRet -> %d", setRet);
                HcommCcuInsResDescDestroy(reqDesc);
                DestroyAllDescs(reqDescs);
                return ConvertCcuToHccl(setRet);
            }
            HCCL_INFO(
                "[BuildAggregatedResReq] dieId[%u] resType[%s] aggregated maxNum[%u].", dieId,
                GetCcuResTypeName(resType), maxNum);
        }
        reqDescs[dieId] = reqDesc;
    }
    HCCL_INFO("[BuildAggregatedResReq] finish, dieNum[%zu].", reqDescs.size());
    return HCCL_SUCCESS;
}

// 复用已有 CcuIns：对每个 die 创建 capDesc 查询容量并与 reqDesc 比较，全部充足才注册 kernels；
// 任一 die 不足返回 HCCL_E_UNAVAIL 触发回退。函数内部销毁 reqDescs。
static HcclResult ReuseExistingCcuIns(
    CcuInsHandle insHandle, ResDescByDie& reqDescs, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO("[ReuseExistingCcuIns] reuse existing CcuIns, insHandle[%p], dieNum[%zu].", insHandle, reqDescs.size());
    bool allSufficient = true;
    std::string allDieInsuffSummary; // 收集所有 die 的不足资源摘要，用于回退时记 RUN_INFO
    for (auto& dieEntry : reqDescs) {
        uint32_t dieId = dieEntry.first;
        HcommCcuResDescHandle reqDesc = dieEntry.second;
        HcommCcuResDescHandle capDesc = 0;
        CcuResult capCreateRet = HcommCcuInsResDescCreate(dieId, &capDesc);
        if (capCreateRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[ReuseExistingCcuIns] HcommCcuInsResDescCreate capDesc dieId[%u] failed: ccuRet -> %d", dieId,
                capCreateRet);
            DestroyAllDescs(reqDescs);
            return ConvertCcuToHccl(capCreateRet);
        }
        CcuResult qRet = HcommCcuInsQueryResDesc(insHandle, capDesc);
        if (qRet != CCU_SUCCESS) {
            HCCL_ERROR("[ReuseExistingCcuIns] HcommCcuInsQueryResDesc dieId[%u] failed: ccuRet -> %d", dieId, qRet);
            HcommCcuInsResDescDestroy(capDesc);
            DestroyAllDescs(reqDescs);
            return ConvertCcuToHccl(qRet);
        }
        bool sufficient = false;
        std::string insuffSummary;
        HcclResult capRet = IsResCapSufficient(dieId, capDesc, reqDesc, sufficient, insuffSummary);
        if (capRet != HCCL_SUCCESS) {
            // 查询接口本身失败，当作错误上报，不继续检查其他 die，不触发回退
            HCCL_ERROR("[ReuseExistingCcuIns] IsResCapSufficient dieId[%u] failed: ret -> %d", dieId, capRet);
            HcommCcuInsResDescDestroy(capDesc);
            DestroyAllDescs(reqDescs);
            return capRet;
        }
        HCCL_INFO("[ReuseExistingCcuIns] dieId[%u] sufficient[%d].", dieId, sufficient);
        if (!sufficient) {
            allSufficient = false;
            if (!allDieInsuffSummary.empty()) {
                allDieInsuffSummary += "; ";
            }
            allDieInsuffSummary += "dieId[" + std::to_string(dieId) + "]: " + insuffSummary;
        }
        HcommCcuInsResDescDestroy(capDesc);
    }
    DestroyAllDescs(reqDescs);

    if (!allSufficient) {
        HCCL_WARNING("[ReuseExistingCcuIns] existing CcuIns resource insufficient, try to fallback.");
        HCCL_RUN_INFO("[ReuseExistingCcuIns] insufficient res detail: %s", allDieInsuffSummary.c_str());
        return HCCL_E_UNAVAIL;
    }
    return RegisterCcuKernels(insHandle, resRequest, resCtxHost);
}

// 为每个 die 创建 finalReqDesc = max(reqDesc, 默认阈值)，避免按实际需求申请造成资源碎片。出参由调用方销毁。
static HcclResult CreateFinalReqDescs(
    ResDescByDie& reqDescs, HcclOpExpansionMode opExpansionMode, std::vector<HcommCcuResDescHandle>& finalReqDescs)
{
    for (auto& dieEntry : reqDescs) {
        uint32_t dieId = dieEntry.first;
        HcommCcuResDescHandle reqDesc = dieEntry.second;
        HcommCcuResDescHandle finalReqDesc = 0;
        CcuResult fCreateRet = HcommCcuInsResDescCreate(dieId, &finalReqDesc);
        if (fCreateRet != CCU_SUCCESS) {
            HCCL_ERROR(
                "[CreateFinalReqDescs] HcommCcuInsResDescCreate dieId[%u] failed: ccuRet -> %d", dieId, fCreateRet);
            for (auto d : finalReqDescs) {
                HcommCcuInsResDescDestroy(d);
            }
            finalReqDescs.clear();
            return ConvertCcuToHccl(fCreateRet);
        }
        HcclResult maxRet = CalcMaxResReqWithDefault(dieId, opExpansionMode, reqDesc, finalReqDesc);
        if (maxRet != HCCL_SUCCESS) {
            HcommCcuInsResDescDestroy(finalReqDesc);
            for (auto d : finalReqDescs) {
                HcommCcuInsResDescDestroy(d);
            }
            finalReqDescs.clear();
            return maxRet;
        }
        finalReqDescs.push_back(finalReqDesc);
    }
    return HCCL_SUCCESS;
}

// 遍历每个 die 查询硬件剩余资源（HcommCcuQueryRemainResDesc），与 finalReqDescs 对比收集不足资源摘要，
// 拼成 "dieId[x]: LOOP(need=128,remain=16); dieId[y]: ..."。任一 die 查询/对比失败仅 WARNING 并 continue。
static void CollectInsufficientResFromRemain(
    const std::vector<HcommCcuResDescHandle>& finalReqDescs, const std::vector<uint32_t>& finalReqDieIds,
    std::string& allDieInsuffSummary)
{
    for (size_t i = 0; i < finalReqDescs.size() && i < finalReqDieIds.size(); i++) {
        uint32_t dieId = finalReqDieIds[i];
        HcommCcuResDescHandle remainDesc = 0;
        CcuResult rCreateRet = HcommCcuInsResDescCreate(dieId, &remainDesc);
        if (rCreateRet != CCU_SUCCESS) {
            HCCL_WARNING(
                "[CollectInsufficientResFromRemain] create remainDesc dieId[%u] failed: ccuRet -> %d, skip.", dieId,
                rCreateRet);
            continue;
        }
        CcuResult rQueryRet = HcommCcuQueryRemainResDesc(remainDesc);
        if (rQueryRet != CCU_SUCCESS) {
            HCCL_WARNING(
                "[CollectInsufficientResFromRemain] query remainDesc dieId[%u] failed: ccuRet -> %d, skip.", dieId,
                rQueryRet);
            HcommCcuInsResDescDestroy(remainDesc);
            continue;
        }
        bool sufficient = false;
        std::string insuffSummary;
        HcclResult cmpRet = IsResCapSufficient(dieId, remainDesc, finalReqDescs[i], sufficient, insuffSummary);
        HcommCcuInsResDescDestroy(remainDesc);
        if (cmpRet != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CollectInsufficientResFromRemain] compare remainDesc dieId[%u] failed: ret -> %d, skip.", dieId,
                cmpRet);
            continue;
        }
        if (!sufficient) {
            if (!allDieInsuffSummary.empty()) {
                allDieInsuffSummary += "; ";
            }
            allDieInsuffSummary += "dieId[" + std::to_string(dieId) + "]: " + insuffSummary;
        }
    }
}

// 新建 CcuIns：取需求与默认阈值的最大值创建实例并绑定到 comm，使后续算子走复用路径。
// 函数内部销毁 reqDescs。
static HcclResult CreateAndAssignNewCcuIns(
    HcclComm comm, HcclOpExpansionMode opExpansionMode, ResDescByDie& reqDescs, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO(
        "[CreateAndAssignNewCcuIns] no existing CcuIns, create new one, opExpansionMode[%u], dieNum[%zu].",
        static_cast<uint32_t>(opExpansionMode), reqDescs.size());
    // 收集 dieId 顺序（CreateFinalReqDescs 按 reqDescs 即 map 升序遍历，与此处顺序一致），
    // 用于资源不足时查询每个 die 的剩余资源做对比
    std::vector<uint32_t> finalReqDieIds;
    for (const auto& entry : reqDescs) {
        finalReqDieIds.push_back(entry.first);
    }

    std::vector<HcommCcuResDescHandle> finalReqDescs;
    HcclResult createRet = CreateFinalReqDescs(reqDescs, opExpansionMode, finalReqDescs);
    DestroyAllDescs(reqDescs);
    if (createRet != HCCL_SUCCESS) {
        return createRet;
    }

    CcuInsHandle newInsHandle = 0;
    CcuResult createInsRet = HcommCcuInsCreate(finalReqDescs.data(), finalReqDescs.size(), &newInsHandle);
    HCCL_INFO("[CreateAndAssignNewCcuIns] HcommCcuInsCreate ret[%d], newInsHandle[%p].", createInsRet, newInsHandle);
    if (createInsRet == CCU_E_UNAVAIL) {
        HCCL_WARNING("[CreateAndAssignNewCcuIns] HcommCcuInsCreate unavailable, try to fallback.");
        std::string allDieInsuffSummary;
        CollectInsufficientResFromRemain(finalReqDescs, finalReqDieIds, allDieInsuffSummary);
        HCCL_RUN_INFO("[CreateAndAssignNewCcuIns] insufficient res detail: %s", allDieInsuffSummary.c_str());
        for (auto d : finalReqDescs) {
            HcommCcuInsResDescDestroy(d);
        }
        return HCCL_E_UNAVAIL;
    } else if (createInsRet != CCU_SUCCESS) {
        HCCL_ERROR("[CreateAndAssignNewCcuIns] HcommCcuInsCreate failed: ccuRet -> %d", createInsRet);
        for (auto d : finalReqDescs) {
            HcommCcuInsResDescDestroy(d);
        }
        return ConvertCcuToHccl(createInsRet);
    }

    HcclResult assignRet = HcclCommAssignCcuIns(comm, newInsHandle);
    HCCL_INFO("[CreateAndAssignNewCcuIns] HcclCommAssignCcuIns ret[%d].", assignRet);
    if (assignRet != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateAndAssignNewCcuIns] HcclCommAssignCcuIns failed: ret -> %d", assignRet);
        HcommCcuInsDestroy(newInsHandle);
        for (auto d : finalReqDescs) {
            HcommCcuInsResDescDestroy(d);
        }
        return assignRet;
    }
    for (auto d : finalReqDescs) {
        HcommCcuInsResDescDestroy(d);
    }
    return RegisterCcuKernels(newInsHandle, resRequest, resCtxHost);
}

// CCU kernel 动态资源申请主流程：1.聚合资源需求 -> 2.查询可复用 CcuIns
// -> 3a.容量充足则复用并注册 / 3b.新建实例并绑定 comm 后注册。接口返回 CCU_E_UNAVAIL 时触发回退。
static HcclResult HcclGetCcuKernelDynamic(
    HcclComm comm, HcclOpExpansionMode opExpansionMode, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    HCCL_INFO(
        "[HcclGetCcuKernelDynamic] start, opExpansionMode[%u], kernelNum[%zu].", static_cast<uint32_t>(opExpansionMode),
        resRequest.ccuKernelInfos.size());

    // 步骤1：聚合资源需求，按 dieId 分组
    ResDescByDie reqDescs;
    HcclResult buildRet = BuildAggregatedResReq(resRequest, reqDescs);
    if (buildRet != HCCL_SUCCESS) {
        // 防御性清理：BuildAggregatedResReq 失败时契约上已清理，这里再清理一次防止内部契约被破坏
        DestroyAllDescs(reqDescs);
        return buildRet;
    }

    // 步骤2：查询当前 comm 是否已绑定 CcuIns 实例
    // 接口语义：未绑定 CcuIns 时返回 HCCL_E_UNAVAIL（不是 insNum=0），需走新建路径
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    bool hasReusableIns = false;
    HcclResult queryRet = HcclCommQueryAssignedCcuIns(comm, &insHandle, &insNum);
    if (queryRet == HCCL_SUCCESS) {
        hasReusableIns = (insNum != 0);
        HCCL_INFO(
            "[HcclGetCcuKernelDynamic] HcclCommQueryAssignedCcuIns success, insHandle[%p] insNum[%u].", insHandle,
            insNum);
    } else if (queryRet == HCCL_E_UNAVAIL) {
        HCCL_INFO(
            "[HcclGetCcuKernelDynamic] HcclCommQueryAssignedCcuIns returns UNAVAIL, no reusable CcuIns, will create "
            "new.");
    } else {
        HCCL_ERROR("[HcclGetCcuKernelDynamic] HcclCommQueryAssignedCcuIns failed: ret -> %d", queryRet);
        DestroyAllDescs(reqDescs);
        return queryRet;
    }

    // 步骤3：有可复用实例走复用路径，否则新建；reqDescs 所有权转移给子函数
    // opExpansionMode 透传下去：新建路径需要根据模式取不同的默认阈值（MS 模式 LOOP/CCU_BUF 阈值更大）
    HcclResult finalRet = hasReusableIns ?
                              ReuseExistingCcuIns(insHandle, reqDescs, resRequest, resCtxHost) :
                              CreateAndAssignNewCcuIns(comm, opExpansionMode, reqDescs, resRequest, resCtxHost);

    // 资源不足导致回退时记一条 run info，便于运维统计动态资源申请的回退频率
    if (finalRet == HCCL_E_UNAVAIL) {
        HCCL_RUN_INFO(
            "[HcclGetCcuKernelDynamic] ccu dynamic resource unavailable, fallback to legacy flow, "
            "hasReusableIns[%d], kernelNum[%zu].",
            hasReusableIns, resRequest.ccuKernelInfos.size());
    }
    HCCL_INFO("[HcclGetCcuKernelDynamic] finish, finalRet[%d].", finalRet);
    return finalRet;
}

HcclResult HcclGetCcuKernel(
    HcclComm comm, HcclOpExpansionMode opExpansionMode, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    if (IsCcuDynamicResApiSupported()) {
        HCCL_INFO(
            "[HcclGetCcuKernel] use dynamic resource apply flow, opExpansionMode[%u].",
            static_cast<uint32_t>(opExpansionMode));
        return HcclGetCcuKernelDynamic(comm, opExpansionMode, resRequest, resCtxHost);
    }

    // 兼容旧 hcomm 包
    HCCL_INFO("[HcclGetCcuKernel] use legacy pre-allocated resource flow.");
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(
        insNum != 1, HCCL_ERROR("[HcclGetCcuKernel] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
        HCCL_E_INTERNAL);
    return RegisterCcuKernels(insHandle, resRequest, resCtxHost);
}
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0) */

HcclResult GetAlgResAiv(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence)
{
    uint64_t size = sizeof(AlgResourceCtxSerializable);
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, resCtxSequence));

    AlgResourceCtxSerializable* resCtxHost = static_cast<AlgResourceCtxSerializable*>(*resCtxSequence);
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    CHK_RET(HcclAllocAlgResourceAiv(comm, param, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceAiv(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest, AlgResourceCtxSerializable* resCtxHost)
{
    HCCL_INFO("[%s]Start to execute.", __func__);
    HcclMemHandle memHandle; // 注册到通信域内存的handle，用于建链
    // 获取存放AIV对端信息和标记区的空间
    void* buffersIn[MAX_RANK_SIZE] = {};
    void* buffersOut[MAX_RANK_SIZE] = {};
    uint64_t commInfoSize = 0;
    HcclResult ret
        = HcclEngineCtxGet(comm, param.commModeTag, param.engine, &(resCtxHost->aivCommInfoPtr), &commInfoSize);
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        CHK_RET(HcclEngineCtxCreate(
            comm, param.commModeTag, param.engine, AIV_TAG_BUFF_LEN, &(resCtxHost->aivCommInfoPtr)));
        // 清零
        ACLCHECK(haclrtMemset(resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN));
        if (HcommIsSupportHcclCommRegCommStateCallback()) {
            CHK_RET(HcclCommRegCommStateCallback(param.commModeTag, ClearAivTagCb, resCtxHost->aivCommInfoPtr));
        }
        // 注册到通信域，支持建链时交换
        CommMem regMem{COMM_MEM_TYPE_DEVICE, resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN};
        CHK_RET(HcclCommMemReg(comm, param.commModeTag, &regMem, &memHandle));
        void* memHandleCachePtr = nullptr; // 当前AIV存放注册内存的memHandle使用
        CHK_RET(HcclEngineCtxCreate(
            comm, param.commModeTag, CommEngine::COMM_ENGINE_CPU_TS, sizeof(HcclMemHandle), &memHandleCachePtr));
        static_cast<HcclMemHandle*>(memHandleCachePtr)[0] = memHandle;
    } else {
        void* memHandleCachePtr = nullptr;
        uint64_t memHandleCacheSize = 0;
        HcclResult ret = HcclEngineCtxGet(
            comm, param.commModeTag, CommEngine::COMM_ENGINE_CPU_TS, &memHandleCachePtr, &memHandleCacheSize);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS || memHandleCacheSize != sizeof(HcclMemHandle),
            HCCL_ERROR(
                "[%s]commModeTag[%s] aiv memHandle not found in cache, ptr[%p] size[%llu]", __func__, param.commModeTag,
                memHandleCachePtr, memHandleCacheSize),
            HCCL_E_INTERNAL);
        memHandle = static_cast<HcclMemHandle*>(memHandleCachePtr)[0];

        // 有的算子，如send recv，可能多个算子对端不同导致buffer被覆盖，需先获取一遍之前的信息
        CHK_RET(haclrtMemcpy(
            buffersIn, MAX_RANK_SIZE * sizeof(void*), resCtxHost->aivCommInfoPtr, MAX_RANK_SIZE * sizeof(void*),
            ACL_MEMCPY_DEVICE_TO_HOST));
        CHK_RET(haclrtMemcpy(
            buffersOut, MAX_RANK_SIZE * sizeof(void*),
            static_cast<u8*>(resCtxHost->aivCommInfoPtr) + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(void*),
            ACL_MEMCPY_DEVICE_TO_HOST));
    }
    HCCL_INFO(
        "[%s]commModeTag[%s] regMemAddr[%p] memHandle[%p]", __func__, param.commModeTag, resCtxHost->aivCommInfoPtr,
        memHandle);

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    HCCL_INFO("[%s]local cclBufferAddr[%p] cclBufferSize[%llu]", __func__, cclBufferAddr, cclBufferSize);
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    buffersIn[resCtxHost->topoInfo.userRank] = cclBufferAddr;
    buffersOut[resCtxHost->topoInfo.userRank] = resCtxHost->aivCommInfoPtr;

    // 迭代每个子通信域的建链请求，创建链路
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc>& levelNChannelRequest = resRequest.channels[level];
        for (auto& channelDesc : levelNChannelRequest) {
            channelDesc.memHandles = &memHandle;
            channelDesc.memHandleNum = 1;
        }
        // 获取子通信域的建链数量
        u32 validChannelNum = levelNChannelRequest.size();
        std::vector<ChannelHandle> levelNChannels;
        levelNChannels.resize(validChannelNum);
        HCCL_INFO("[%s]level[%u] validChannelNum[%u]", __func__, level, validChannelNum);

        if (validChannelNum > 0) {
            // 参数一致性校验信息注册到通信域，HcclChannelAcquire内部存在读清动作，每次调用前均需注册
            CHK_RET(AddExchangeInfo(comm, param));
            CHK_RET(HcclChannelAcquire(
                comm, param.engine, levelNChannelRequest.data(), validChannelNum, levelNChannels.data()));
        }

        for (u32 idx = 0; idx < validChannelNum; idx++) {
            HcclChannelDesc& channelDesc = levelNChannelRequest[idx];
            CHK_PRT_RET(
                channelDesc.remoteRank >= MAX_RANK_SIZE,
                HCCL_ERROR(
                    "[%s] remoteRank[%u] exceeds MAX_RANK_SIZE[%u]", __func__, channelDesc.remoteRank, MAX_RANK_SIZE),
                HCCL_E_PARA);
            void* remoteBufferAddr;
            uint64_t remoteBufferSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
            HCCL_INFO(
                "[%s]remoteRank[%u] cclBufferAddr[%p] cclBufferSize[%llu]", __func__, channelDesc.remoteRank,
                remoteBufferAddr, remoteBufferSize);
            buffersIn[channelDesc.remoteRank] = remoteBufferAddr;

            u32 memNum;
            CommMem* remoteMems;
            char** memTags;
            CHK_RET(HcclChannelGetRemoteMems(comm, levelNChannels[idx], &memNum, &remoteMems, &memTags));
            CHK_PRT_RET(memNum == 0, HCCL_ERROR("[%s] HcclChannelGetRemoteMems memNum is 0", __func__), HCCL_E_PARA);
            HCCL_RUN_INFO(
                "[%s]remoteRank[%u] memNum[%u] regMemAddr[%p] regMemSize[%llu] memTag[%s]", __func__,
                channelDesc.remoteRank, memNum, remoteMems[memNum - 1].addr, remoteMems[memNum - 1].size,
                memTags[memNum - 1]);
            buffersOut[channelDesc.remoteRank] = remoteMems[memNum - 1].addr;
        }
    }

    CHK_RET(haclrtMemcpy(
        resCtxHost->aivCommInfoPtr, MAX_RANK_SIZE * sizeof(void*), buffersIn, MAX_RANK_SIZE * sizeof(void*),
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHK_RET(haclrtMemcpy(
        static_cast<u8*>(resCtxHost->aivCommInfoPtr) + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(void*), buffersOut,
        MAX_RANK_SIZE * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE));

    HCCL_INFO("[%s] Alloc res success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult GetAlgResDPU(
    HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, void** resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag, const ResPackGraphMode& resPack)
{
    // 申请共享内存
    uint64_t shmemSize = 100 * 1024 * 1024;
    void* shmemPtr = nullptr;
    bool newCreated;
    CHK_RET(HcclDevMemAcquire(comm, "DPUTAG", &shmemSize, &shmemPtr, &newCreated));
    resCtxHost->npu2DpuShmemPtr = shmemPtr;
    constexpr uint64_t DPU2NPU_SHMEM_RATIO = 2;
    resCtxHost->dpu2NpuShmemPtr = static_cast<void*>(static_cast<uint8_t*>(shmemPtr) + shmemSize / DPU2NPU_SHMEM_RATIO);

    CHK_RET(GetAlgResAICPU(
        comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, ctxSize,
        increCreateChannelFlag, resPack));

    HCCL_INFO("Execute GetAlgResAICPU success.");
    return HCCL_SUCCESS;
}

HcclResult CheckCount(const u64 count)
{
    if (UNLIKELY(count > SYS_MAX_COUNT)) {
        HCCL_ERROR(
            "[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce)
{
    const std::vector<std::string> infoTitle({"ccl_op", "value", "parameter", "expect"});
    // 查询是否为合法的HcclDataType枚举值
    bool notValid = VALID_HCCL_DATA_TYPES.find(dataType) == VALID_HCCL_DATA_TYPES.end();
    if (needReduce) {
        // reduce场景不支持的数据类型
        static const std::set<HcclDataType> REDUCE_UNSUPPORTED
            = {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,  HCCL_DATA_TYPE_UINT32,  HCCL_DATA_TYPE_INT128,
               HCCL_DATA_TYPE_HIF8,  HCCL_DATA_TYPE_FP8E4M3, HCCL_DATA_TYPE_FP8E5M2, HCCL_DATA_TYPE_FP8E8M0};
        if (notValid || REDUCE_UNSUPPORTED.find(dataType) != REDUCE_UNSUPPORTED.end()) {
            RPT_INPUT_ERR(
                true, "EI0003", infoTitle,
                std::vector<std::string>(
                    {"CheckDataType", GetDataTypeEnumStr(dataType), "dataType", GetSupportDataType(needReduce)}));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        // 非reduce场景不支持INT128
        if (notValid || dataType == HCCL_DATA_TYPE_INT128) {
            RPT_INPUT_ERR(
                true, "EI0003", infoTitle,
                std::vector<std::string>(
                    {"CheckDataType", GetDataTypeEnumStr(dataType), "dataType",
                     GetSupportDataType(needReduce).c_str()}));
            HCCL_ERROR(
                "[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

std::string GetSupportDataType(bool needReduce)
{
    std::vector<HcclDataType> supportList = {HCCL_DATA_TYPE_INT8,  HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32,
                                             HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_FP16,  HCCL_DATA_TYPE_FP32};
    if (needReduce) {
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64});
    } else {
        supportList.insert(
            supportList.end(), {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16, HCCL_DATA_TYPE_UINT32,
                                HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64, HCCL_DATA_TYPE_HIF8, HCCL_DATA_TYPE_FP8E4M3,
                                HCCL_DATA_TYPE_FP8E5M2, HCCL_DATA_TYPE_FP8E8M0});
        supportList.push_back(HCCL_DATA_TYPE_BFP16);
    }

    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

HcclResult CheckReduceOp(const HcclDataType dataType, const HcclReduceOp op)
{
    std::vector<HcclDataType> prodSupportList
        = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,
           HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32,  HCCL_DATA_TYPE_FP64};
    const std::vector<std::string> infoTitle({"ccl_op", "value", "parameter", "expect"});
    if (op == HcclReduceOp::HCCL_REDUCE_PROD) {
        if (std::find(prodSupportList.begin(), prodSupportList.end(), dataType) == prodSupportList.end()) {
            RPT_INPUT_ERR(
                true, "EI0003", infoTitle,
                std::vector<std::string>(
                    {"CheckReduceDataType", GetDataTypeEnumStr(dataType), "dataType", GetReduceProdSupportDataType()}));
            HCCL_ERROR(
                "[Check][ReduceOp][DataType]errNo[0x%016llx] reduceop is [%s] data type[%s] not supported, support "
                "range=[%s]",
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetReduceOpEnumStr(op).c_str(),
                GetDataTypeEnumStr(dataType).c_str(), GetReduceProdSupportDataType().c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

std::string GetReduceProdSupportDataType()
{
    std::vector<HcclDataType> supportList
        = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,
           HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32,  HCCL_DATA_TYPE_FP64};
    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

HcclResult SetCommEngine(OpParam& param)
{
    // 使用一个静态的映射表来关联配置和引擎值
    static const std::unordered_map<OpExecuteConfig, CommEngine> ConfigToEngineMap = {
        {OpExecuteConfig::HOSTCPU_TS, COMM_ENGINE_CPU_TS},
        {OpExecuteConfig::AICPU_TS, COMM_ENGINE_AICPU_TS},
        {OpExecuteConfig::AIV, COMM_ENGINE_AIV},
        {OpExecuteConfig::AIV_ONLY, COMM_ENGINE_AIV}, // AIV_ONLY 和 AIV 映射到同一引擎
        {OpExecuteConfig::CCU_MS, COMM_ENGINE_CCU},
        {OpExecuteConfig::CCU_SCHED, COMM_ENGINE_CCU},
        {OpExecuteConfig::AICPU, COMM_ENGINE_AICPU},
        {OpExecuteConfig::HOSTCPU, COMM_ENGINE_CPU},
    };

    auto it = ConfigToEngineMap.find(param.opExecuteConfig);
    if (it != ConfigToEngineMap.end()) {
        param.engine = it->second;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR(
        "[op_common][SetCommEngine] Unsupported or unknown opExecuteConfig: {%d}",
        static_cast<int>(param.opExecuteConfig));
    return HCCL_E_NOT_SUPPORT;
}

HcclResult SingleRankProc(HcclComm comm, OpParam& param)
{
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    HCCL_INFO("[SingleRankProc]Start to execute HcclExecOp. HcommGetProfilingSysCycleTime[%llu]", beginTime);
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY) {
        HCCL_ERROR(
            "[SingleRankProc] opType[%d] currently do not select aiv mode, aiv only not support, "
            "please ensure rankNum is greater than one",
            static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
        HCCL_WARNING("[%s] ranksize == 1 is not support BATCHSENDRECV SEND RECV", __func__);
        return HcclResult::HCCL_SUCCESS;
    }
    if (param.inputPtr == param.outputPtr) {
        HCCL_WARNING("[%s] sendBuf == recvBuf, return success", __func__);
        return HcclResult::HCCL_SUCCESS;
    }
    u64 len{0};
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        len = DATATYPE_SIZE_TABLE[param.all2AllVDataDes.sendType]
              * *(static_cast<const u64*>(param.all2AllVDataDes.sendCounts));
    } else if (param.opType == HCCL_CMD_ALLGATHER_V || param.opType == HCCL_CMD_REDUCE_SCATTER_V) {
        len = DATATYPE_SIZE_TABLE[param.vDataDes.dataType] * *(static_cast<const u64*>(param.vDataDes.counts));
    } else {
        len = DATATYPE_SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;
    }
    HCCL_INFO("[%s] sendBuf[%p], recvBuf[%p], len[%llu]", __func__, param.inputPtr, param.outputPtr, len);
    if (len > 0) {
        ThreadHandle cpuTsThread{0};
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
        HcclDfxOpInfoCompat hcclDfxOpInfo{}; // Op注册
        hcclDfxOpInfo.opMode = static_cast<u32>(param.opMode);
        hcclDfxOpInfo.opType = static_cast<u32>(param.opType);
        hcclDfxOpInfo.reduceOp = static_cast<u32>(param.reduceType);
        CHK_RET(GetHcclDfxOpInfoDataType(param, hcclDfxOpInfo.dataType));
        u32 userRankSize{0}; // rankSize获取指定算子的dataCount
        CHK_RET(HcclGetRankSize(comm, &userRankSize));
        CHK_RET(GetHcclDfxOpInfoDataCount(param, userRankSize, hcclDfxOpInfo.dataCount));
        hcclDfxOpInfo.root = param.root;
        hcclDfxOpInfo.engine = param.engine;
        hcclDfxOpInfo.cpuTsThread = cpuTsThread;
        hcclDfxOpInfo.cpuWaitAicpuNotifyIdx = HOST_WAIT_AICPU_NOTIFYIDX;
        CHK_RET(SetOpParamAlgTag(param, "SingleRankProc"));
        s32 sRet = strncpy_s(hcclDfxOpInfo.algTag, ALG_TAG_LENGTH, param.algTag, ALG_TAG_LENGTH);
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR("%s call strncpy_s failed, param.algTag %s, return %d.", __func__, param.algTag, sRet),
            HCCL_E_MEMORY);
        CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(cpuTsThread, param.outputPtr, param.inputPtr, len)));
    }
    CHK_RET(HcclProfilingReportOp(comm, beginTime));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCheckTag(const char* tag)
{
    CHK_PTR_NULL(tag);

    u32 tagLen = strnlen(tag, HCCL_TAG_MAX_LEN + 1);
    if (UNLIKELY((tagLen == (HCCL_TAG_MAX_LEN + 1) || tagLen == 0))) {
        HCCL_ERROR("[Check][Tag]errNo[0x%016llx] tag is too long", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

static HcclResult BuildCcuExtraTag(const OpParam& param, std::string& ccuExtraTag)
{
    HcclDataType tmpDataType;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        tmpDataType = param.all2AllVDataDes.sendType;
    } else if (
        param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        tmpDataType = param.vDataDes.dataType;
    } else {
        tmpDataType = param.DataDes.dataType;
    }
    ccuExtraTag = "_" + HCOM_DATA_TYPE_STR_MAP.at(tmpDataType);

    if (param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE
        || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        ccuExtraTag += "_" + HCOM_REDUCE_OP_STR_MAP.at(param.reduceType);
    }

    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_SCATTER
        || param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        ccuExtraTag += "_r" + std::to_string(param.root);
    }
    return HCCL_SUCCESS;
}

HcclResult SetOpParamAlgTag(OpParam& param, const std::string& algName)
{
    std::string temp = algName; // 创建algName的副本

    const char* launchMode
        = (((param.engine == CommEngine::COMM_ENGINE_AICPU) || (param.engine == CommEngine::COMM_ENGINE_AICPU_TS)) ?
               "device" :
               "host");
    int len;
    // 图模式下去掉param.tag前缀，避免tag不同导致algTag不同而无法复用资源
    if (param.opMode == OpMode::OFFLOAD && param.engine == CommEngine::COMM_ENGINE_CCU) {
        len = snprintf_s(
            param.algTag, sizeof(param.algTag), sizeof(param.algTag), "Graph_%s_%s", temp.c_str(), launchMode);
    } else {
        len = snprintf_s(
            param.algTag, sizeof(param.algTag), sizeof(param.algTag), "%s_%s_%s", param.tag, temp.c_str(), launchMode);
    }
    if (len < 0 || len >= sizeof(param.algTag)) {
        HCCL_ERROR("failed to fill param.algTag");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // ccu模式，考虑kernel是否能复用，需要添加dataType和reduceType
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        try {
            std::string ccuExtraTag;
            CHK_RET(BuildCcuExtraTag(param, ccuExtraTag));
            size_t remainBytes = sizeof(param.algTag) - len;

            int len_ccu = snprintf_s(param.algTag + len, remainBytes, remainBytes, "%s", ccuExtraTag.c_str());
            CHK_PRT_RET(
                (len_ccu < 0 || len_ccu >= sizeof(param.algTag) - len),
                HCCL_ERROR("failed to fill alg tag with ccu dataType"), HCCL_E_INTERNAL);
        } catch (const std::out_of_range& e) {
            HCCL_ERROR("[SetOpParamAlgTag] dataType or reduceType out of range: %s", e.what());
            return HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclGetOpExpansionMode(HcclComm comm, OpParam& param)
{
    HcclOpExpansionMode finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    // 第一步：决定使用哪种模式
    HcclResult ret = DecideHcclOpExpansionMode(comm, finalMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("DecideHcclOpExpansionMode failed, ret: %d", ret);
        return ret;
    }
    param.commOpExpansionMode = finalMode;

    // 第二步：应用选择的模式到param
    ret = ApplyOpExpansionMode(param, finalMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("ApplyOpExpansionMode failed, ret: %d", ret);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetHcclAlgo(HcclComm comm, std::string& hcclAlgo)
{
    if (GetHcommVersion() <= CANN_VERSION(9, 2, 0, 1)) {
        HCCL_INFO("[HcclGetHcclAlgo] HcclConfigGetInfo not supported, skip.");
        return HcclResult::HCCL_SUCCESS;
    }
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (!hcommFunction.dlHcclConfigGetInfo) {
        HCCL_INFO("[HcclGetHcclAlgo] HcclConfigGetInfo not supported, skip.");
        return HcclResult::HCCL_SUCCESS;
    }
    std::vector<char> buf(HCCL_COMM_ALGO_MAX_LENGTH, '\0');
    uint32_t infoLen = static_cast<uint32_t>(buf.size());
    HcclResult ret = hcommFunction.dlHcclConfigGetInfo(
        comm, static_cast<HcclConfigType>(HCCL_CONFIG_TYPE_HCCL_ALGO), infoLen, buf.data());
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING("[HcclGetHcclAlgo] HcclConfigGetInfo failed, ret: %d", ret);
        return HcclResult::HCCL_SUCCESS;
    }
    hcclAlgo.assign(buf.data(), strnlen(buf.data(), buf.size()));
    HCCL_DEBUG("[HcclGetHcclAlgo] hcclAlgo from comm: [%s]", hcclAlgo.c_str());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult DecideHcclOpExpansionMode(HcclComm comm, HcclOpExpansionMode& finalMode)
{
    HcclOpExpansionMode configOpExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    bool useConfigOpExpansionMode = false;
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (hcommFunction.dlHcclConfigGetInfo) {
        uint32_t infoLen = sizeof(HcclOpExpansionMode);
        CHK_RET(hcommFunction.dlHcclConfigGetInfo(
            comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, infoLen, &configOpExpansionMode));
        finalMode = configOpExpansionMode;
        useConfigOpExpansionMode = true;
    } else {
        HCCL_INFO("[DecideHcclOpExpansionMode] HcclConfigGetInfo is not supported, use environment mode.");
        finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs);
    }

    // A5仅通过HcclConfigGetInfo获取展开模式，其他型号保留环境变量方式
    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    CHK_RET(HcclGetDeviceType(deviceType));
    if (!shouldGoOutPlace(deviceType) || !useConfigOpExpansionMode) {
        if (GetExternalInputHcclAicpuUnfold() == true) {
            finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU;
        } else if (GetExternalInputHcclAivOnlyMode() == true) {
            finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY;
        } else if (GetExternalInputHcclAivMode() == true) {
            finalMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV;
        } else if (GetExternalInputHcclCcuMSMode()) {
            finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs);
        } else if (GetExternalInputHcclCcuSchedMode()) {
            finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuSched);
        }
        if (useConfigOpExpansionMode && configOpExpansionMode != finalMode) {
            HCCL_DEBUG(
                "[DecideHcclOpExpansionMode] configOpExpansionMode: %d, environment mode: %d, conflict, use "
                "environment mode.",
                configOpExpansionMode, finalMode);
        }
    }
    HCCL_INFO("[DecideHcclOpExpansionMode] finalMode: %d.", finalMode);

    return HCCL_SUCCESS;
}

HcclResult ApplyOpExpansionMode(OpParam& param, HcclOpExpansionMode finalMode)
{
    switch (finalMode) {
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AI_CPU:
            param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
            param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
            CHK_RET(LoadAICPUKernel());
            HCCL_DEBUG("[ApplyOpExpansionMode] AICPU mode selected.");
            break;
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_AIV:
            param.opExecuteConfig = OpExecuteConfig::AIV;
            param.engine = CommEngine::COMM_ENGINE_AIV;
            CHK_RET(RegisterKernel());
            HCCL_DEBUG("[ApplyOpExpansionMode] AIV mode selected.");
            break;
        case HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY:
            param.opExecuteConfig = OpExecuteConfig::AIV_ONLY;
            param.engine = CommEngine::COMM_ENGINE_AIV;
            CHK_RET(RegisterKernel());
            HCCL_DEBUG("[ApplyOpExpansionMode] AIV_ONLY mode selected.");
            break;
        case static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs):
            param.opExecuteConfig = OpExecuteConfig::CCU_MS;
            param.engine = CommEngine::COMM_ENGINE_CCU;
            HCCL_DEBUG("[ApplyOpExpansionMode] CCU_MS mode selected.");
            break;
        case static_cast<HcclOpExpansionMode>(opExpansionModeCcuSched):
            param.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
            param.engine = CommEngine::COMM_ENGINE_CCU;
            HCCL_DEBUG("[ApplyOpExpansionMode] CCU_SCHED mode selected.");
            break;
        default:
            // 回退到aicpu
            HCCL_WARNING("[ApplyOpExpansionMode] Invalid HcclOpExpansionMode: %d, fallback to AICPU_TS.", finalMode);
            param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
            param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
            CHK_RET(LoadAICPUKernel());
            break;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
HcclRegstryBuff(HcclComm comm, const char* memTag, void* bufferPtr, uint64_t bufferSize, HcclMemHandle* memHandle)
{
    CHK_PTR_NULL(memHandle);
    CommMem regMem{COMM_MEM_TYPE_DEVICE, bufferPtr, bufferSize};
    CHK_RET(HcclCommMemReg(comm, memTag, &regMem, memHandle));
    HCCL_INFO("[%s] regMemAddr[%p] regMemSize[%llu]", __func__, regMem.addr, regMem.size);
    CHK_PTR_NULL(*memHandle);
    return HCCL_SUCCESS;
}

HcclResult
HcclGetRemoteBuff(HcclComm comm, ChannelHandle channel, const char* memTag, void** bufferPtr, uint64_t* bufferSize)
{
    CHK_PTR_NULL(bufferPtr);
    CHK_PTR_NULL(bufferSize);

    u32 memNum;
    CommMem* remoteMemList;
    char** memTags;
    CHK_RET(HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMemList, &memTags));
    HCCL_INFO("[%s] HcclChannelGetRemoteMems memNum[%u]", __func__, memNum);
    for (u32 i = 0; i < memNum; i++) {
        HCCL_INFO("[%s] memNum[%u/%u] memTags[%s]", __func__, i + 1, memNum, memTags[i]);
        if (strcmp(memTags[i], memTag) == 0) {
            *bufferPtr = remoteMemList[i].addr;
            *bufferSize = remoteMemList[i].size;
            HCCL_INFO(
                "[%s] Found memNum[%u/%u]: addr=%p, size=%llu", __func__, i + 1, memNum, remoteMemList[i].addr,
                remoteMemList[i].size);
            break;
        }
    }
    if (*bufferPtr == nullptr) {
        HCCL_WARNING("[%s] Failed to find %s in remote mem list", __func__, memTag);
    }
    return HCCL_SUCCESS;
}

HcclResult LogHcclExit(const std::string& opName, const char* tag, HcclUs startut, bool forceLog)
{
    if (forceLog || GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        std::string endInfo = opName + ":success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us, tag: " + tag;
        HCCL_RUN_INFO("%s", endInfo.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult GetAivParamStorageByComm(HcclComm comm, AivParamStorage** aivParam, bool ifCreate)
{
    if (comm == nullptr || aivParam == nullptr) {
        HCCL_ERROR("[GetAivParamStorageByComm] Invalid parameters");
        return HCCL_E_PARA;
    }

    void* aivParamCtx = nullptr;
    uint64_t size = sizeof(AivParamStorage);

    const char* aivParamTag = "AivParamStorage";
    if (HcclEngineCtxGet(comm, aivParamTag, CommEngine::COMM_ENGINE_CPU_TS, &aivParamCtx, &size) != HCCL_SUCCESS) {
        if (ifCreate) {
            CHK_RET(HcclEngineCtxCreate(comm, aivParamTag, CommEngine::COMM_ENGINE_CPU_TS, size, &aivParamCtx));
        } else {
            HCCL_WARNING("[GetAivParamStorageByComm] Call HcclEngineCtxGet failed.");
            return HCCL_E_PARA;
        }
    }

    *aivParam = static_cast<AivParamStorage*>(aivParamCtx);

    return HCCL_SUCCESS;
}

HcclResult GetAivParamStorage(const char* group, AivParamStorage** aivParam)
{
    if (group == nullptr || aivParam == nullptr) {
        HCCL_ERROR("[GetAivParamStorage] Invalid parameters");
        return HCCL_E_PARA;
    }

    HcclComm comm = nullptr;
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));

    return GetAivParamStorageByComm(comm, aivParam, true);
}

template <typename...>
using VoidT = void;

template <typename T, typename = void>
struct HasSplitRatioConfigType : std::false_type {};

template <typename T>
struct HasSplitRatioConfigType<T, VoidT<decltype(T::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO)>> :
    std::true_type {};

HcclResult QuerySplitRatioByConfigGetInfo(HcclComm comm, HcclConfigType cfgType, double& ratio, bool& isConfigured)
{
    ratio = 0.0;
    isConfigured = false;
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (!hcommFunction.dlHcclConfigGetInfo) {
        HCCL_INFO("[QuerySplitRatioByConfigGetInfo] HcclConfigGetInfo is not supported, skip comm config.");
        return HCCL_SUCCESS;
    }
    double commRatio = 0.0;
    const uint32_t infoLen = sizeof(commRatio);
    HcclResult ret = hcommFunction.dlHcclConfigGetInfo(comm, cfgType, infoLen, &commRatio);
    if (ret == HCCL_E_NOT_SUPPORT) {
        HCCL_INFO("[QuerySplitRatioByConfigGetInfo] comm config not set or not supported, ret[%d].", ret);
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[QuerySplitRatioByConfigGetInfo] HcclConfigGetInfo failed, ret[%d].", ret);
        return ret;
    }
    if (!std::isfinite(commRatio) || commRatio < 0.0 || commRatio > 1.0) {
        HCCL_ERROR("[QuerySplitRatioByConfigGetInfo] comm ratio[%f] is not finite or out of range[0, 1].", commRatio);
        return HCCL_E_PARA;
    }
    if (IsDoubleEqual(commRatio, 0.0)) {
        HCCL_INFO("[QuerySplitRatioByConfigGetInfo] comm split ratio is not configured.");
        return HCCL_SUCCESS;
    }
    ratio = commRatio;
    isConfigured = true;
    HCCL_INFO("[QuerySplitRatioByConfigGetInfo] comm ratio[%f] is configured.", commRatio);
    return HCCL_SUCCESS;
}

template <typename ConfigType>
HcclResult QueryCommSplitRatio(HcclComm comm, double& ratio, bool& isConfigured, std::false_type)
{
    HCCL_INFO("[QueryCommSplitRatio] Current Hcomm headers do not support split ratio config, skip comm config.");
    ratio = 0.0;
    isConfigured = false;
    return HCCL_SUCCESS;
}

template <typename ConfigType>
HcclResult QueryCommSplitRatio(HcclComm comm, double& ratio, bool& isConfigured, std::true_type)
{
    return QuerySplitRatioByConfigGetInfo(
        comm, ConfigType::HCCL_CONFIG_TYPE_MULTIPLE_DIMENSION_SPLIT_RATIO, ratio, isConfigured);
}

HcclResult GetCommMultipleDimensionSplitRatio(HcclComm comm, double& ratio, bool& isConfigured)
{
    return QueryCommSplitRatio<HcclConfigType>(comm, ratio, isConfigured, HasSplitRatioConfigType<HcclConfigType>{});
}

HcclResult SetMultipleDimensionSplitRatio(HcclComm comm, OpParam& param)
{
    constexpr double defaultRatio = 0.5;

    double commRatio = 0.0;
    bool isCommConfigured = false;
    HcclResult ret = GetCommMultipleDimensionSplitRatio(comm, commRatio, isCommConfigured);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (isCommConfigured) {
        param.opConfig.multipleDimensionSplitRatio = commRatio;
        param.opConfig.multipleDimensionSplitRatioSource = MultipleDimensionSplitRatioSource::COMM_CONFIG;
        HCCL_INFO("[SetMultipleDimensionSplitRatio] ratioSource[COMM_CONFIG], configuredRatio[%f]", commRatio);
        return HCCL_SUCCESS;
    }

    double envRatio = 0.0;
    if (GetExternalInputMultipleDimensionSplitRatio(envRatio)) {
        if (!std::isfinite(envRatio) || envRatio < 0.0 || envRatio > 1.0) {
            HCCL_WARNING(
                "[SetMultipleDimensionSplitRatio] env ratio[%f] is out of range, use default ratio[%f]", envRatio,
                defaultRatio);
            envRatio = defaultRatio;
        }
        param.opConfig.multipleDimensionSplitRatio = envRatio;
        param.opConfig.multipleDimensionSplitRatioSource = MultipleDimensionSplitRatioSource::ENV_CONFIG;
        HCCL_INFO("[SetMultipleDimensionSplitRatio] ratioSource[ENV_CONFIG], configuredRatio[%f]", envRatio);
        return HCCL_SUCCESS;
    }

    param.opConfig.multipleDimensionSplitRatio = defaultRatio;
    param.opConfig.multipleDimensionSplitRatioSource = MultipleDimensionSplitRatioSource::BUILTIN_FORMULA;
    HCCL_INFO("[SetMultipleDimensionSplitRatio] ratioSource[BUILTIN_FORMULA], configuredRatio[%f]", defaultRatio);
    return HCCL_SUCCESS;
}

// 判断通过最高一个level的网络全部没有device的可达链路，并且有host的可达链路
HcclResult CheckHostDPUOnly(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool& hostDPUOnly)
{
    hostDPUOnly = false;
    HCCL_INFO("Start CheckHostDPUOnly");
    // 只有一个server，不使用DPU
    if (topoInfo->serverNum == 1) {
        HCCL_INFO("Not using hostdpu because serverNum is 1");
        return HCCL_SUCCESS;
    }

    // 只有一层topo，不使用DPU
    if (topoInfo->topoLevelNums == 1) {
        HCCL_INFO("Not using hostdpu because topoLevelNums is 1");
        return HCCL_SUCCESS;
    }

    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if ((netLayers == nullptr) || (netLayerNum == 0)) {
        HCCL_WARNING("HcclRankGraphGetLayers fail");
        return HCCL_E_INTERNAL;
    }

    bool hostDPU = false;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        uint32_t netLayer = netLayers[layerIdx];
        // 只校验最后一个level
        if (netLayer < (topoInfo->topoLevelNums - 1)) {
            HCCL_INFO("Skip checking layer[%u], topoLevelNums is [%u]", netLayer, topoInfo->topoLevelNums);
            continue;
        }
        uint32_t* topoInsts = nullptr;
        uint32_t topoInsNum = 0;
        CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInsNum));
        if ((topoInsts == nullptr) || (topoInsNum == 0)) {
            HCCL_WARNING("HcclRankGraphGetTopoInstsByLayer fail, netLayer[%u]", netLayer);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t topoInsIdx = 0; topoInsIdx < topoInsNum; topoInsIdx++) {
            uint32_t topoInstId = topoInsts[topoInsIdx];
            HCCL_INFO("Start checking topoInstId[%u]", topoInstId);
            CommTopo topoType;
            CHK_RET(HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &topoType));
            if (topoType != COMM_TOPO_CLOS) {
                HCCL_INFO("Not using hostdpu because topo type is not COMM_TOPO_CLOS");
                continue;
            }
            uint32_t* ranks = nullptr;
            uint32_t rankNum = 0;
            CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum));
            // 校验当前rank与其他所有rank连通
            if (rankNum != topoInfo->userRankSize) {
                HCCL_INFO("Not using hostdpu because current rank is not fully connected to all other ranks");
                continue;
            }
            uint32_t endPointNums = 0;
            CHK_RET(HcclRankGraphGetEndpointNum(comm, netLayer, topoInstId, &endPointNums));
            EndpointDesc endPointDescs[endPointNums];
            CHK_RET(HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &endPointNums, endPointDescs));
            for (uint32_t endPointIdx = 0; endPointIdx < endPointNums; endPointIdx++) {
                EndpointDesc endPointDesc = endPointDescs[endPointIdx];
                if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                    HCCL_INFO(
                        "Not using hostdpu because there is links on device in netLayer[%u] in endPointIdx[%u]",
                        netLayer, endPointIdx);
                    return HCCL_SUCCESS;
                } else if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                    HCCL_INFO("Found a host endPoint in netLayer[%u] endPointIdx[%u]", netLayer, endPointIdx);
                    hostDPU = true;
                }
            }
        }
    }
    if (hostDPU) {
        HCCL_INFO("Using host dpu trans.");
        hostDPUOnly = true;
    }
    return HCCL_SUCCESS;
}

// 设置执行超时时间
HcclResult SetExecTimeout(OpParam& param)
{
    double execTimeoutValue = 0;
    if (!GetExternalInputExecTimeout(execTimeoutValue)) {
        param.opConfig.execTimeout = CUSTOM_TIMEOUT;
        HCCL_INFO("[OpCommon] Exec timeout is not set, use default value: %u seconds", CUSTOM_TIMEOUT);
    } else {
        // 验证转换后的值是否合理
        if (execTimeoutValue < 0 || execTimeoutValue > UINT32_MAX) {
            HCCL_WARNING(
                "[OpCommon] Exec timeout value %.2f s out of range, use default: %u seconds", execTimeoutValue,
                CUSTOM_TIMEOUT);
            param.opConfig.execTimeout = CUSTOM_TIMEOUT;
        } else {
            param.opConfig.execTimeout = static_cast<uint32_t>(execTimeoutValue);
            if (param.opConfig.execTimeout == 0) {
                HCCL_INFO("[OpCommon] Exec timeout is disabled (never timeout).");
            } else {
                HCCL_INFO("[OpCommon] Set exec timeout to: %u seconds", param.opConfig.execTimeout);
            }
        }
    }
    return HCCL_SUCCESS;
}

bool IsHostDpu(HcclComm comm)
{
    HcclResult ret;
    bool hostDpuOnly = false;

    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    ret = HcclGetDeviceType(deviceType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[IsHostDpu]HcclGetDeviceType fail, ret:%d", ret);
        return false;
    }
    if (deviceType != HcclDevType::DEV_TYPE_910B) {
        return false;
    }

    uint32_t* level0SizeList = nullptr;
    uint32_t level0RankListNum = 0;
    ret = HcclRankGraphGetInstSizeListByLayer(
        comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), &level0SizeList, &level0RankListNum);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    // 获取 rankSize
    u32 rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    // 获取 topoLevelNums
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    TopoInfoWithNetLayerDetails topoInfo;
    topoInfo.serverNum = level0RankListNum;
    topoInfo.topoLevelNums = netLayerNum;
    topoInfo.userRankSize = rankSize;
    ret = CheckHostDPUOnly(comm, &topoInfo, hostDpuOnly);
    if (ret == HCCL_SUCCESS && hostDpuOnly) {
        return true;
    }
    return false;
}

// 判定当前通信域是否为「框间 host-DPU」场景。逻辑与 IsHostDpu 一致，
// 但不限定 910B —— 950 Barrier 新流程（框内 AICPU + 框间 DPU）仅在该场景启用。
bool IsBarrierHostDpu(HcclComm comm)
{
    HcclResult ret;
    bool hostDpuOnly = false;

    uint32_t* level0SizeList = nullptr;
    uint32_t level0RankListNum = 0;
    ret = HcclRankGraphGetInstSizeListByLayer(
        comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), &level0SizeList, &level0RankListNum);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    u32 rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    ret = HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    TopoInfoWithNetLayerDetails topoInfo;
    topoInfo.serverNum = level0RankListNum;
    topoInfo.topoLevelNums = netLayerNum;
    topoInfo.userRankSize = rankSize;
    ret = CheckHostDPUOnly(comm, &topoInfo, hostDpuOnly);
    if (ret == HCCL_SUCCESS && hostDpuOnly) {
        return true;
    }
    return false;
}
} // namespace ops_hccl
