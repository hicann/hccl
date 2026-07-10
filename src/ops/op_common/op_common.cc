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
#include <future>
#include <map>
#include <string>
#include <memory>
#include <cstdlib>  // 包含getenv函数
#include <cstring>  // 包含strcmp函数
#include <stdexcept>

#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include "hccl/base.h"
#include "sal.h"
#include "error_codes/rt_error_codes.h"
#include "param_check.h"
#include "inconsistent_check.h"
#include "executor_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_env_config.h"
#include "adapter_acl.h"
#include "topo_host.h"
#include "adapter_error_manager_pub.h"
#include "hccl_inner.h"
#include "hccl.h"
#include "config_log.h"
#include "workflow.h"
#include "load_kernel.h"
#include "alg_param.h"
#include "alg_type.h"
#include "op_common.h"
#include "aicpu_timeout.h"
#include "hccl_aiv_utils.h"
#include "dpu/kernel_launch.h"
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
#include "hccl_ccu_res_dl.h"
#include "comm_engine_utils.h"

namespace ops_hccl {
thread_local bool needInconsistentCheck = false;
// 用于维护增量建链算子的host ctx信息
constexpr u32 HOST_WAIT_AICPU_NOTIFYIDX = 0;// host主流wait aicpu流的notify idx
constexpr u32 HOST_NOTIFY_TIMEOUT_OFFSET = 27;  // host等待Device通知的超时时间偏移量
constexpr u32 KERNEL_TIMEOUT_OFFSET = 25;       // kernel启动超时时间偏移量
constexpr u32 CPU_TS_NOTIFY_NUM = 3;            // CPU TS thread notify数量

void UpdateAicpuTimeoutCtx(const OpParam &param, AlgResourceCtxSerializable &resCtx)
{
    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    resCtx.waitTimeout = timeout.waitTimeout;
    resCtx.fullTimeout = timeout.fullTimeout;
    HCCL_INFO("[AicpuTimeout] execTimeout[%u], waitTimeout[%u], fullTimeout[%u], "
        "hostNotifyTimeout[%u], kernelLaunchTimeout[%u], hcommDefaultTimeoutSupported[%u].",
        param.opConfig.execTimeout, timeout.waitTimeout, timeout.fullTimeout, timeout.hostNotifyTimeout,
        timeout.kernelLaunchTimeout, static_cast<u32>(IsHcommDefaultTimeoutSupported()));
}

// 检查非对称拓扑支持情况
// 仅 AllGather, AllReduce, ReduceScatter, AllToAll(V/VC), Scatter, Broadcast, Reduce 支持跨框非对称拓扑，其他算子拦截
HcclResult CheckAsymmetricTopoSupport(HcclCMDType opType, const TopoInfoWithNetLayerDetails* topoInfo)
{
    // 仅在跨框非对称场景下检查
    if (topoInfo->topoLevelNums > 1 && topoInfo->multiModuleDiffDeviceNumMode) {
        // 已适配非对称的算子：AllGather, AllReduce, ReduceScatter, AllToAll(V/VC), Scatter, Broadcast, Reduce
        bool isSupportedOp = (opType == HcclCMDType::HCCL_CMD_ALLGATHER ||
                             opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
                             opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
                             opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
                             opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
                             opType == HcclCMDType::HCCL_CMD_ALLTOALLVC ||
                             opType == HcclCMDType::HCCL_CMD_SCATTER ||
                             opType == HcclCMDType::HCCL_CMD_BROADCAST ||
                             opType == HcclCMDType::HCCL_CMD_REDUCE);
        if (!isSupportedOp) {
            HCCL_ERROR("[CheckAsymmetricTopoSupport] OpType[%d] does not support asymmetric topology "
                "(multi-module diff device num mode) "
                "only ALLGATHER/ALLREDUCE/REDUCE_SCATTER/ALLTOALL/ALLTOALLV/ALLTOALLVC/SCATTER/BROADCAST/REDUCE "
                "are supported.", opType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult Selector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    std::string &algName)
{
    //判断通信域状态
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

    // 检查非对称拓扑支持情况，非对称场景仅 AllGather/AllReduce/ReduceScatter/AllToAll(V/VC)/Scatter/Broadcast/Reduce 可用
    CHK_RET(CheckAsymmetricTopoSupport(param.opType, topoInfo.get()));

    // 算法选择，选择完后顺便param.algTag设置了，资源的保存是以算子+算法为单位
    std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
    CHK_RET(collAlgSelector->Run(param, topoInfo.get(), algName));
    if (algName == "") {
        HCCL_ERROR("[Selector] select algname fail!");
        return HCCL_E_PTR;
    }
    CHK_RET(SetCommEngine(param));
    // AIV_ONLY 模式下禁止回退到非 AIV 引擎，未选中 AIV 时直接返回不支持。
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY && param.engine != CommEngine::COMM_ENGINE_AIV) {
        HCCL_ERROR("[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.",
            static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    // 如果一开始读取到的Engine不是aicpu，经过算法选择后回退到aipcu，则需要重新LoadAICPUKernel
    if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU)) {
        HCCL_DEBUG("[Selector] is aicpu mode");
        CHK_RET(LoadAICPUKernel()); // 该函数内部有防止重复加载的逻辑
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
    CHK_RET(SetMultipleDimensionSplitRatio(param));
    HCCL_INFO("Success to execute Selector.");
    return HCCL_SUCCESS;
}

HcclResult GetHcclDfxOpInfoDataCount(const OpParam &param, const u32 &rankSize, uint64_t &sendCount) {
    sendCount = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
        sendCount += *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts)); // 非v类算子，只上报入参里的count
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
        for (u32 idx = 0; idx < param.batchSendRecvDataDes.itemNum; idx++) {
            HcclSendRecvItem* item = param.batchSendRecvDataDes.sendRecvItemsPtr + idx;
            CHK_PRT_RET(item == nullptr, HCCL_ERROR("[%s]fail, item is nullptr, idx[%u], itemNum[%u], tag[%s]",
                __func__, idx, param.batchSendRecvDataDes.itemNum, param.tag), HCCL_E_PTR);
            sendCount += item->count;
        }
    } else {
        sendCount = param.DataDes.count;
    }
    HCCL_INFO("[%s]tag[%s], sendCount[%u], opType[%u], rankSize[%u]",
        __func__, param.tag, sendCount, param.opType, rankSize);
    return HCCL_SUCCESS;
}

HcclResult GetHcclDfxOpInfoDataType(const OpParam &param, uint32_t &dataType) {
    dataType = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V
        || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        dataType = static_cast<u32>(param.vDataDes.dataType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dataType = static_cast<u32>(param.all2AllVDataDes.sendType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        CHK_PRT_RET(param.batchSendRecvDataDes.itemNum == 0, HCCL_INFO("[%s]tag[%s] itemNum is 0, skip",
            __func__, param.tag), HCCL_SUCCESS);
        CHK_PRT_RET(param.batchSendRecvDataDes.sendRecvItemsPtr == nullptr,
            HCCL_ERROR("[%s]fail, tag[%s] sendRecvItemsPtr is nullptr", __func__, param.tag), HCCL_E_PTR);
        dataType = static_cast<u32>(param.batchSendRecvDataDes.sendRecvItemsPtr->dataType); // dfx功能只能上报一个数据类型
    } else {
        dataType = static_cast<u32>(param.DataDes.dataType);
    }
    HCCL_INFO("[%s]tag[%s], dataType[%u], opType[%u]", __func__, param.tag, dataType, param.opType);
    return HCCL_SUCCESS;
}

HcclResult AppendFastLaunchTag(OpParam &param, const char* dataTypeStr,
    const char* reduceOpStr, const char* countStr, const char* rootStr)
{
    char* dst = param.fastLaunchTag;
    size_t remain = sizeof(param.fastLaunchTag);

    auto append_str = [&](const char* s) -> bool {
        if (!s) return true;
        size_t len = strlen(s);
        if (len >= remain) return false;
        memcpy_s(dst, remain, s, len);
        dst += len;
        remain -= len;
        return true;
    };
    if (!append_str(param.tag) || !append_str("_") || !append_str(dataTypeStr)) {
        goto fail;
    }
    if (reduceOpStr && (!append_str("_")) || !append_str(reduceOpStr)) {
        goto fail;
    }
    if (countStr && (!append_str("_")) || !append_str(countStr)) {
        goto fail;
    }
    if (rootStr && (!append_str("_r")) || !append_str(rootStr)) {
        goto fail;
    }
    *dst = '\0';
    HCCL_INFO("[SetOpParamFastLaunchTag] fastLaunchTag: [%s]", param.fastLaunchTag);
    return HcclResult::HCCL_SUCCESS;

fail:
    HCCL_ERROR("failed to fill fastLaunchTag");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult SetOpParamFastLaunchTag(OpParam &param)
{
    // 1. 数据类型
    const char* dataTypeStr = nullptr;
    if(param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dataTypeStr = GetHcclDataTypeStr(param.all2AllVDataDes.sendType);
    } else {
        dataTypeStr = GetHcclDataTypeStr(param.DataDes.dataType);
    }
    CHK_PRT_RET((!dataTypeStr), HCCL_ERROR("unsupported data type"), HcclResult::HCCL_E_INTERNAL);
    // 2. reduce op
    const char* reduceOpStr = nullptr;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
        param.opType == HcclCMDType::HCCL_CMD_REDUCE    || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        reduceOpStr = GetHcclReduceOpStr(param.reduceType);
        CHK_PRT_RET((!reduceOpStr), HCCL_ERROR("unsupported reduce op"), HcclResult::HCCL_E_INTERNAL);
    }
    // 3. count
    char countBuf[32];
    const char* countStr = nullptr;
    if (param.opType != HcclCMDType::HCCL_CMD_ALLTOALLV) {
        u64 count = (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) ? *reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts)
                                                                     : param.DataDes.count;
        int countLen = snprintf_s(countBuf, sizeof(countBuf), sizeof(countBuf) - 1, "%llu", count);
        CHK_PRT_RET((countLen <= 0), HCCL_ERROR("failed to format count"), HcclResult::HCCL_E_INTERNAL);
        countStr = countBuf;
    }
    // 4. root
    char rootBuf[10];
    const char* rootStr = nullptr;
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_SCATTER ||
        param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        int rootLen = snprintf_s(rootBuf, sizeof(rootBuf), sizeof(rootBuf) - 1, "%llu", static_cast<uint64_t>(param.root));
        CHK_PRT_RET((rootLen <= 0), HCCL_ERROR("failed to format root"), HcclResult::HCCL_E_INTERNAL);
        rootStr = rootBuf;
    }
    // 5 一次性拼接
    return AppendFastLaunchTag(param, dataTypeStr, reduceOpStr, countStr, rootStr);
}

static constexpr uint32_t opExpansionModeCcuSched = 5;
static constexpr uint32_t opExpansionModeCcuMs = 4;

bool ShouldGoCcuFastLaunch(HcclComm comm, OpParam &param, CcuFastLaunchCtx **ccuFastLaunchCtx)
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
    CHK_RET(SetOpParamFastLaunchTag(param));

    // 2. 查到engineCtx
    uint64_t size = 0;
    void *fastLaunchCtxPtr = nullptr;
    if (HcclEngineCtxGet(comm, param.fastLaunchTag, CommEngine::COMM_ENGINE_CCU, &fastLaunchCtxPtr, &size) == HCCL_SUCCESS) {
        HCCL_INFO("[ShouldGoCcuFastLaunch] get fastLaunchCtx success, size is %u", size);
        *ccuFastLaunchCtx = reinterpret_cast<CcuFastLaunchCtx*>(fastLaunchCtxPtr);
        return true;
    }
    return false;
#else
    (void)comm; (void)param; (void)ccuFastLaunchCtx;
    return false;
#endif
}

HcclResult ConstructHcclDfxOpInfo(const OpParam &param, const char* tag, u32 tagSize, HcclDfxOpInfoCompat& hcclDfxOpInfo,
    ThreadHandle cpuTsThread)
{
    hcclDfxOpInfo.opMode = static_cast<u32>(param.opMode);
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
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, tag:%s, tagSize:%u, sRet:%d.",
        __func__, tag, tagSize, sRet), HCCL_E_MEMORY);
    HCCL_INFO("[%s]HcclDfxOpInfo param: algTag[%s], opMode[%u], opType[%u], reduceOp[%u], dataType[%u], dataCount[%llu], "
        "root[%u], engine[%s], inputMemAddr[0x%llx], inputMemSize[%llu], outputMemAddr[0x%llx], outputMemSize[%llu], "
        "cpuTsThread[0x%llu], cpuWaitAicpuNotifyIdx[%u]",
        __func__, hcclDfxOpInfo.algTag, hcclDfxOpInfo.opMode, hcclDfxOpInfo.opType, hcclDfxOpInfo.reduceOp,
        hcclDfxOpInfo.dataType, hcclDfxOpInfo.dataCount, hcclDfxOpInfo.root, GetEnumToString(GetCommEngineStatusStrMap(), hcclDfxOpInfo.engine).c_str(),
        hcclDfxOpInfo.inputMemAddr, hcclDfxOpInfo.inputMemSize, hcclDfxOpInfo.outputMemAddr,
        hcclDfxOpInfo.outputMemSize, hcclDfxOpInfo.cpuTsThread, hcclDfxOpInfo.cpuWaitAicpuNotifyIdx);
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

bool IsAivCacheSupported(const OpParam &param)
{
    return (param.opType == HCCL_CMD_ALLGATHER || param.opType == HCCL_CMD_ALLREDUCE ||
            param.opType == HCCL_CMD_REDUCE_SCATTER || param.opType == HCCL_CMD_BROADCAST ||
            param.opType == HCCL_CMD_REDUCE || param.opType == HCCL_CMD_ALLTOALL ||
            param.opType == HCCL_CMD_SCATTER) &&
           param.opMode == OpMode::OPBASE && !IsStreamInCaptureMode(param.stream);
}

HcclResult HcclAivCacheCheckAndReplay(HcclComm comm, OpParam &param, bool &cacheHit)
{
    cacheHit = false;
    if (!IsAivCacheSupported(param)) {
        return HCCL_SUCCESS;
    }

    AivOpCacheArgs cacheKey = {};
    cacheKey.commName = param.commName;
    cacheKey.opType = param.opType;
    cacheKey.root = param.root;
    cacheKey.reduceOp = param.reduceType;
    if (param.opType == HCCL_CMD_ALLTOALL) {
        cacheKey.dataType = param.all2AllVDataDes.sendType;
        cacheKey.count = static_cast<const u64 *>(param.all2AllVDataDes.sendCounts)[0];
    } else {
        cacheKey.count = param.DataDes.count;
        cacheKey.dataType = param.DataDes.dataType;
    }

    u64 keyHash = CalcAivCacheKeyHash(cacheKey);
    std::string ctxTag;
    CHK_RET(BuildAivCacheCtxTag(keyHash, ctxTag));

    std::string cachedAlgName;
    AivInstruction *instructions = nullptr;
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

    u32 numBlocksLimit = MAX_NUM_BLOCKS;
    ACLCHECK(aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit));
    param.numBlocksLimit = numBlocksLimit;

    CHK_RET(ReplayAivInstructions(instructions, insCount, param));

    CHK_RET(HcclReportAivKernel(comm, beginTime));
    CHK_RET(HcclProfilingReportOp(comm, beginTime));
    return HCCL_SUCCESS;
}

HcclResult HcclExecOpCcuFastLaunch(HcclComm comm, OpParam &param, const CcuFastLaunchCtx *ccuFastLaunchCtx)
{
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    HCCL_INFO("[HcclExecOpCcuFastLaunch] HcclExecOpCcuFastLaunch start");
    std::string algName = ccuFastLaunchCtx->algName;
    HCCL_DEBUG("[HcclExecOpCcuFastLaunch] algName: [%s]", algName.c_str());
    std::unique_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr, HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    void *cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    param.hcclBuff = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    // 覆盖主流
    ThreadHandle mainThread;
    CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, param.stream,
        ccuFastLaunchCtx->notifyNumOnMainThread, &mainThread));
    ThreadHandle *threads = ccuFastLaunchCtx->GetThreadHandlePtr();
    threads[0] = mainThread;

    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    // Op注册
    HcclDfxOpInfoCompat hcclDfxOpInfo{};
    CHK_RET(ConstructHcclDfxOpInfo(param, param.fastLaunchTag, ALG_TAG_LENGTH, hcclDfxOpInfo, 0));
    param.dataCount = hcclDfxOpInfo.dataCount;
    CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));

    HCCL_INFO("[HcclExecOpCcuFastLaunch] FastLaunch start");
    CHK_RET(executor->FastLaunch(param, ccuFastLaunchCtx));
    HcclProfilingReportOp(comm, beginTime);
    HCCL_INFO("[HcclExecOpCcuFastLaunch] HcclExecOpCcuFastLaunch end");
    return HCCL_SUCCESS;
#else
    (void)comm; (void)param; (void)ccuFastLaunchCtx;
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult ExecuteAivCacheLogic(HcclComm comm, OpParam &param, const std::string &algName,
                                std::unique_ptr<InsCollAlgBase> &executor,
                                AlgResourceCtxSerializable &resCtxHost)
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
        if (param.opType == HCCL_CMD_ALLTOALL) {
            cacheKey.dataType = param.all2AllVDataDes.sendType;
            cacheKey.count = static_cast<const u64 *>(param.all2AllVDataDes.sendCounts)[0];
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
        AivCacheIndexCtx *indexCtx = nullptr;
        CHK_RET(GetOrCreateAivCacheIndexCtx(comm, &indexCtx));
        CHK_RET(EvictAivCacheIfNeeded(comm, indexCtx));
        CHK_RET(StoreAivCacheCtx(comm, ctxTag, keyHash, algName, indexCtx));
        g_recordingQueue = nullptr;
        g_baseInputAddr = 0;
        g_baseOutputAddr = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult FallbackOp(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, 
    std::string &algName, const ResPackGraphMode &resPack)
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

HcclResult ReSelector(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    std::string &algName)
{
    (void) comm;
    HCCL_INFO("Start to execute ReSelector.");
    // 回退AICPU
    param.opExecuteConfig = OpExecuteConfig::AICPU_TS;
    // 拓扑已有，无需再计算

    // 算法选择，选择完后顺便param.algTag设置了，资源的保存是以算子+算法为单位
    std::shared_ptr<ExecuteSelector> collAlgSelector = std::make_shared<ExecuteSelector>(ExecuteSelector());
    CHK_RET(collAlgSelector->Run(param, topoInfo.get(), algName));
    if (algName == "") {
        HCCL_ERROR("[ReSelector] select algname fail!");
        return HCCL_E_PTR;
    }
    CHK_RET(SetCommEngine(param));
    // AIV_ONLY 模式下禁止回退到非 AIV 引擎，未选中 AIV 时直接返回不支持。
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY && param.engine != CommEngine::COMM_ENGINE_AIV) {
        HCCL_ERROR("[HcclExecOp] opType[%d] currently do not select aiv mode, aiv only not support.",
            static_cast<int>(param.opType));
        return HCCL_E_NOT_SUPPORT;
    }
    // 如果一开始读取到的Engine不是aicpu，经过算法选择后回退到aipcu，则需要重新LoadAICPUKernel
    if ((param.engine == CommEngine::COMM_ENGINE_AICPU_TS) || (param.engine == CommEngine::COMM_ENGINE_CPU)) {
        HCCL_DEBUG("[ReSelector] is aicpu mode");
        CHK_RET(LoadAICPUKernel()); // 该函数内部有防止重复加载的逻辑
    }
    CHK_RET(SetOpParamAlgTag(param, algName));
    HCCL_INFO("Success to execute ReSelector.");
    return HCCL_SUCCESS;
}

HcclResult SetOpParamFallbackTag(OpParam &param, const std::string &algName)
{
    auto fallbackRet = sprintf_s(param.fallbackTag, sizeof(param.fallbackTag), "%s_%s", algName.c_str(), "fallback");
    if (fallbackRet <= 0) {
        HCCL_ERROR("[%s] failed to fill fallbackTag", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclExecOp(HcclComm comm, OpParam &param,
                      std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo, std::string &algName, const ResPackGraphMode &resPack)
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
    resCtxHost->isHcommBatchTransferOnThreadSupported =
        HcommIsSupportHcommBatchTransferOnThread();
    // 资源序列化结果
    void *resCtxSequence = nullptr;
    bool isResourceReused = false;

    ThreadHandle cpuTsThread{0};
    ThreadHandle exportedAicpuTsThread{0};
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, CPU_TS_NOTIFY_NUM, &cpuTsThread));
        // Export cpuTsThread
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &cpuTsThread, COMM_ENGINE_AICPU_TS, &exportedAicpuTsThread));
    }

    auto resRet = HcclGetAlgRes(comm, param, executor, topoInfo.get(), resCtxHost, &resCtxSequence, isResourceReused, resPack);
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
        CHK_RET(CaptureSlaveStreams(comm, param.stream, {mainThread, unfoldThread}));
        CHK_RET(HcclAicpuKernelEntranceLaunch(comm, param, cpuTsThread, exportedCpuTsThread, notifyNumOnMainThread,
            resCtxSequence, algName, unfoldThread));
    } else if (param.engine == COMM_ENGINE_AIV) {
        uint64_t aivBeginTime = HcommGetProfilingSysCycleTime();
        param.resCtx = resCtxSequence;
        AlgResourceCtxSerializable &aivResCtxHost = *static_cast<AlgResourceCtxSerializable *>(resCtxSequence);
        CHK_RET(HcclAivKernelEntranceLaunch(comm, param, topoInfo, aivResCtxHost));
        CHK_RET(ExecuteAivCacheLogic(comm, param, algName, executor, aivResCtxHost));
        CHK_RET(HcclReportAivKernel(comm, aivBeginTime));
    } else if (param.engine == COMM_ENGINE_CCU) {
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            char *ctx = static_cast<char*>(resCtxSequence);
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            resCtxHost->DeSerialize(seq);
            // 覆盖主流
            ThreadHandle thread;
            CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, param.stream,
                resCtxHost->notifyNumOnMainThread, &thread));
            resCtxHost->threads[0] = thread;
            // 图模式要全部覆盖
            if (param.opMode != OpMode::OPBASE) {
                CHK_RET(GeReuseResource(comm, param, executor, resCtxHost, topoInfo.get(), resPack));
            }
        }
        if (resCtxHost->slaveThreadNum > 0) {
            CHK_RET(CaptureSlaveStreams(comm, param.stream, resCtxHost->threads));
        }
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    } else {
        if (isResourceReused) {
            // 复用资源，则需从engineCtx取得res，进行反序列化
            char *ctx = static_cast<char*>(resCtxSequence);
            std::vector<char> seq(ctx, ctx + param.ctxSize);
            resCtxHost->DeSerialize(seq);
        }
        CHK_RET(executor->Orchestrate(param, *resCtxHost));
    }
    // op上报
    CHK_RET(HcclProfilingReportOp(comm, beginTime));
    HCCL_INFO("Execute HcclExecOp success.");
    return HCCL_SUCCESS;
}

HcclResult GeReuseResource(HcclComm comm, OpParam &param, std::unique_ptr<InsCollAlgBase>& executor,
        std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo, const ResPackGraphMode &resPack)
{
    // 计算AlgHierarchyInfo
    AlgHierarchyInfoForAllLevel algHierarchyInfo;  // 分级通信域信息{localRankId, localRankSize}
    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo));
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
    for (u32 i = 0; i < threadNum; i++) {
        ThreadHandle slaveThread;
        CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, resPack.streams[i], maxNotifyNum, &slaveThread));
        resCtxHost->threads[i + 1] = slaveThread;
    }
    return HCCL_SUCCESS;
}

static HcclResult GetUnfoldStream(HcclComm comm, OpParam &param, ThreadHandle unfoldThread, aclrtStream &resolvedStream)
{
    void *unfoldStream = nullptr;
    auto &HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    HcclResult ret;
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo || param.opMode == OpMode::OFFLOAD) { // 不走提前展开
        resolvedStream = param.stream;
    } else {
        ret = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void *), &unfoldStream);
        if (ret == HCCL_E_NOT_SUPPORT) {
            resolvedStream = param.stream;
        } else if (ret != HCCL_SUCCESS) {
            return ret;
        } else {
            resolvedStream = unfoldStream;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAicpuKernelEntranceLaunch(HcclComm comm, OpParam &param, ThreadHandle cpuTsThread,
    ThreadHandle exportedCpuTsThread, u32 notifyNumOnMainThread, void *resCtxSequence, std::string &algName, ThreadHandle unfoldThread)
{
    HCCL_DEBUG("[HcclAicpuKernelEntranceLaunch]start to run aicpu kernel");
    (void) algName;
    // 当前aicpu launch接口只能有一个输入参数，将Context指针放在param参数中
    param.resCtx = resCtxSequence;
    param.aicpuRecordCpuIdx = HOST_WAIT_AICPU_NOTIFYIDX;

    if (param.engine == COMM_ENGINE_CPU) {
        // 注册dpu回调函数
        CHK_RET(static_cast<HcclResult>(HcclTaskRegister(comm, param.algTag, HcclLaunchDPUKernel)));
    }

    if (HcommIsSupportHcclAicpuKernelLaunch() && 
        (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE)) {
        HCCL_INFO("[HcclAicpuKernelEntranceLaunch] P2P opType[%d], use HcclAicpuKernelLaunch",
            static_cast<int>(param.opType));
 
        // 构造 HcclOpDesc
        HcclOpDesc opInfo;
        
        (void)memset_s(&opInfo, sizeof(HcclOpDesc), 0, sizeof(HcclOpDesc));
        opInfo.opDescType = 1;  // 1: P2P

        std::string opNameStr = (param.opType == HcclCMDType::HCCL_CMD_SEND) ? "HcclSend" : "HcclRecv";
        (void)strncpy_s(opInfo.opName, HCCL_OP_DESC_OP_NAME_MAX_LEN, opNameStr.c_str(), opNameStr.size());

        opInfo.p2p.buffer = (param.opType == HcclCMDType::HCCL_CMD_SEND) ? 
                            param.inputPtr : param.outputPtr;
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
        (void)memset_s(&funcInfo, sizeof(HcclKernelFuncInfo), 0, sizeof(HcclKernelFuncInfo));
        
        (void)sprintf_s(funcInfo.kernelSoName, sizeof(funcInfo.kernelSoName), 
                          "libscatter_aicpu_kernel.so");

        (void)sprintf_s(funcInfo.kernelFuncName, sizeof(funcInfo.kernelFuncName), 
                          "HcclLaunchP2pAicpuKernel");
  
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
        u16 kernelLaunchTimeout = IsHcommDefaultTimeoutSupported() ? timeout.kernelLaunchTimeout :
        ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
        kernelLaunchCfg.timeOut = kernelLaunchTimeout;

        CHK_RET(HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, param.stream, &kernelLaunchCfg));

        HCCL_INFO("[HcclAicpuKernelEntranceLaunch] P2P launch success, algTag[%s]", param.algTag);
        return HCCL_SUCCESS;
    }

    // Host stream通知Device主thread，使用主流上idx最大的notify
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(cpuTsThread, exportedCpuTsThread,
        notifyNumOnMainThread - 1)));
    // AicpuKernel report
    uint64_t beginTime = HcommGetProfilingSysCycleTime();
    CHK_RET(AicpuKernelLaunch(comm, param, unfoldThread));
    CHK_PTR_NULL(comm);
    std::string kernelName = "HcclLaunchAicpuKernel";
    char* kernelNameCStr = const_cast<char*>(kernelName.c_str());
    HcclResult ret = HcclReportAicpuKernel(comm, beginTime, kernelNameCStr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclAicpuKernelEntranceLaunch] HcclReportAicpuKernel failed, beginTime %lu, kernelNameCStr %s, ret %d ", beginTime, kernelNameCStr, ret);
        return ret;
    }
    // Host stream等待Device的通知
    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    u32 hostNotifyWaitTime = IsHcommDefaultTimeoutSupported() ? timeout.hostNotifyTimeout :
        AddAicpuTimeoutOffset(param.opConfig.execTimeout, HOST_NOTIFY_TIMEOUT_OFFSET);
    if (HcommIsSupportHcommSetNotifyWaitTimeOut()) {
        CHK_RET(HcclSetNotifyWaitTimeOut(hostNotifyWaitTime));
    }
    CHK_RET(HcclThreadNotifyWaitOnThreadDefault(cpuTsThread, param.aicpuRecordCpuIdx, hostNotifyWaitTime));

    return HCCL_SUCCESS;
}

HcclResult AicpuKernelLaunch(HcclComm comm, OpParam &param, ThreadHandle unfoldThread)
{
    std::string kernelName = "HcclLaunchAicpuKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    // 注意，目前开源HCCL加载AICPU kernel使用的是从json文件加载
    // 详见load_kernel.cc中的LoadAICPUKernel函数，且只实现了scatter的，先共用scatter的
    aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, "
        "kernelName:%s", ret, kernelName.c_str()), HCCL_E_RUNTIME);
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, "
        "kernelName:%s", ret, kernelName.c_str()), HCCL_E_RUNTIME);
    aclrtParamHandle paraHandle;
    size_t paramSize = sizeof(OpParam) + param.varMemSize;
    ret = aclrtKernelArgsAppend(argsHandle, &param, paramSize, &paraHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append "
        "size %u, kernelName:%s", ret, paramSize, kernelName.c_str()), HCCL_E_RUNTIME);
    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, "
        "kernelName:%s", ret, kernelName.c_str()), HCCL_E_RUNTIME);

    AicpuTimeout timeout = DeriveAicpuTimeout(param.opConfig.execTimeout);
    u16 kernelLaunchTimeout = IsHcommDefaultTimeoutSupported() ? timeout.kernelLaunchTimeout :
        ToKernelLaunchTimeout(AddAicpuTimeoutOffset(param.opConfig.execTimeout, KERNEL_TIMEOUT_OFFSET));
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = kernelLaunchTimeout;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;
    HCCL_INFO("[AicpuKernelLaunch] unfoldThread [%lu]", unfoldThread);  // 通过Thread获取展开流stream
    void* unfoldStream = nullptr;
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo || param.opMode == OpMode::OFFLOAD) { // 不走提前展开
        ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
    } else {
        HcclResult ret1 = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*),
            &unfoldStream);
        if (ret1 == HCCL_E_NOT_SUPPORT) {
            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
        } else if (ret1 != HCCL_SUCCESS) {
            return ret1;
        } else {
            ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, unfoldStream, &cfg, argsHandle, nullptr);
        }
    }
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[LoadCustomKernel][aclrtLaunchKernelWithConfig]"
        "errNo[0x%016llx] launch kernel failed", ret), HCCL_E_OPEN_FILE_FAILURE);
    return HCCL_SUCCESS;
}

HcclResult HcclAivKernelEntranceLaunch(HcclComm comm, OpParam &param, const std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo,
    AlgResourceCtxSerializable &resCtxHost)
{
    (void) topoInfo;
    HCCL_INFO("[%s] algTag[%s] commModeTag[%s] resCtx(Host)[%p] aivCommInfoPtr(Device)[%p]", __func__,
        param.algTag, param.commModeTag, param.resCtx, resCtxHost.aivCommInfoPtr);
    u32 numBlocksLimit = 0;
    AivParamStorage *aivParam = nullptr;
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
    CHK_PRT_RET(numBlocksLimit < 1,
        HCCL_ERROR("[%s] block num less than 1, block num[%d]", __func__, numBlocksLimit), HCCL_E_PARA);
    param.numBlocksLimit = numBlocksLimit;
    HCCL_INFO("[%s] Aiv core limit is [%d].", __func__, numBlocksLimit);
    return HCCL_SUCCESS;
}

HcclResult CaptureSlaveStreams(HcclComm comm, aclrtStream mainStream, const std::vector<ThreadHandle>& threads)
{
    aclmdlRI rtModel = nullptr;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclError ret = aclmdlRICaptureGetInfo(mainStream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture not support.", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclmdlRICaptureGetInfo fail. return[%d].", __func__, ret),
            HCCL_E_RUNTIME);
    }
    if (captureStatus != aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        HCCL_INFO("[%s]captureStatus is not active, captureStatus[%d]", __func__, captureStatus);
        return HCCL_SUCCESS;
    }
    //thread[0] is main thread
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    for (size_t i = 1; i < threads.size(); ++i) {
        void* stream = nullptr;
        CHK_PRT_RET(!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo, HCCL_ERROR("AclGraph is not support."),
            HCCL_E_NOT_SUPPORT);
        CHK_RET(HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, threads[i], 0, sizeof(void*), &stream));
        rtError_t addRet = rtStreamAddToModel(stream, rtModel);
        CHK_PRT_RET(addRet != RT_ERROR_NONE, HCCL_ERROR("[%s]rtStreamAddToModel fail. return[%d].", __func__, addRet),
            HCCL_E_RUNTIME);
        HCCL_DEBUG("[%s]add slaveStream to model success, idx[%zu], stream[%p], rtModel[%p]", __func__, i, stream, rtModel);
    }
    HCCL_INFO("[%s]success, captured streams to rtmodel:[%p], slaveStreamNum:[%zu]", __func__, rtModel, threads.size() > 0 ? threads.size() - 1 : 0);
    return HCCL_SUCCESS;
}

HcclResult HcclCalcTopoInfo(HcclComm comm, OpParam &param, std::unique_ptr<TopoInfoWithNetLayerDetails> &topoInfo)
{
    HCCL_INFO("[%s] HcclCalcTopoInfo start.", __func__);
    uint64_t size = 0;
    void *ctx = nullptr;
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
    char *ctxTemp = reinterpret_cast<char*>(ctx);
    std::vector<char> seq(ctxTemp, ctxTemp + size);
    TopoInfoWithNetLayerDetails topoInfoTemp;
    topoInfoTemp.DeSerialize(seq);
    topoInfo = std::make_unique<TopoInfoWithNetLayerDetails>(std::move(topoInfoTemp));
    HCCL_INFO("[%s] HcclCalcTopoInfo end.", __func__);
    return HCCL_SUCCESS;
}

void CompReqChannelWithExistChannel(const std::vector<std::vector<ChannelInfo>>& existChannels,
    AlgResourceRequest &resRequest)
{
    std::set<u32> existRemoteRankSet = {};
    std::vector<HcclChannelDesc> needAllocChannelDesc;
    // 先把所有已存在的channel的remoteRank整理成集合
    for (const ChannelInfo& channel: existChannels[0]) {
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

static HcclResult TryReuseResource(HcclComm comm, OpParam& param, bool& increCreateChannelFlag,
    void** resCtxSequence, uint64_t& size, bool &isResourceReused)
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
    void *ctx = nullptr;
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

HcclResult HcclGetAlgRes(HcclComm comm, OpParam& param, std::unique_ptr<InsCollAlgBase>& executor, TopoInfoWithNetLayerDetails* topoInfo,
                         std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, void** resCtxSequence, bool &isResourceReused, const ResPackGraphMode &resPack)
{
    HCCL_INFO("[HcclGetAlgRes] Start to execute HcclGetAlgRes.");

    bool increCreateChannelFlag = false;
    uint64_t size = 0;
    if (TryReuseResource(comm, param, increCreateChannelFlag, resCtxSequence, size, isResourceReused) == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    // context创建前做是否需要参数一致性的判断，context未创建则判断为首次下发该算子
    needInconsistentCheck = NeedInconsistentCheck(comm, param);

    // 计算AlgHierarchyInfo
    AlgHierarchyInfoForAllLevel algHierarchyInfo;  // 分级通信域信息{localRankId, localRankSize}
    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo));
    // 资源计算
    HCCL_INFO("[HcclGetAlgRes] executor->CalcRes.");
    AlgResourceRequest resRequest;
    CHK_RET(executor->CalcRes(comm, param, topoInfo, algHierarchyInfo, resRequest));
    auto ret = GetAlgResWithEngine(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
        size, increCreateChannelFlag, resPack);
    if (ret == HCCL_E_UNAVAIL) {
        return HCCL_E_UNAVAIL;
    }
    CHK_RET(ret);

    if (resCtxHost != nullptr) {
        // 拼接各level的channel数量信息
        std::string channelNumInfo;
        for (size_t i = 0; i < resCtxHost->channels.size(); i++) {
            if (i > 0) channelNumInfo += ", ";
            channelNumInfo += "level" + std::to_string(i) + "[" + std::to_string(resCtxHost->channels[i].size()) + "]";
        }
        HCCL_RUN_INFO("[HcclGetAlgRes] engine[%s], algTag[%s], resource allocated: thread num[%u], "
            "channel num per level[%s], ccu kernel num[%u].", GetEnumToString(GetCommEngineStatusStrMap(), param.engine).c_str(), param.algTag,
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

HcclResult FillOpExchangeInfo(HcclComm comm, const OpParam &param, OpExchangeInfo &exchangeInfo)
{
    CHK_PTR_NULL(comm);
    void *cclBufferAddr = nullptr; // 不使用，仅为调用HcclGetHcclBuffer获取cclBufferSize
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &exchangeInfo.cclBufferSize));
    exchangeInfo.root = param.root;
    exchangeInfo.opType = param.opType;
    exchangeInfo.opExecuteConfig = param.opExecuteConfig;
    exchangeInfo.reduceType = param.reduceType;
    CHK_RET(FillOpExchangeInfoWithDataDes(param, exchangeInfo));

    u32 numBlocksLimit = 0;
    AivParamStorage *aivParam = nullptr;
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
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] call strncpy_s failed, param.tag[%s],  return[%d].",
        __func__, param.tag, sRet), HCCL_E_MEMORY);

    HCCL_INFO("[%s] success. exchangeInfo dump: cclBufferSize[%llu], root[%u], opType[%u], opExecuteConfig[%u], "
        "reduceType[%u], dataType[%u], count[%llu], aivCoreLimit[%u], group[%s], tag[%s]",
        __func__, exchangeInfo.cclBufferSize, exchangeInfo.root, exchangeInfo.opType, exchangeInfo.opExecuteConfig,
        exchangeInfo.reduceType, exchangeInfo.dataType, exchangeInfo.count, exchangeInfo.aivCoreLimit,
        exchangeInfo.group, exchangeInfo.tag);
    return HCCL_SUCCESS;
}

HcclResult FillOpExchangeInfoWithDataDes(const OpParam &param, OpExchangeInfo &exchangeInfo)
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

HcclResult AddExchangeInfo(HcclComm comm, const OpParam &param)
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

HcclResult GetAlgResWithEngine(HcclComm comm, OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable> &resCtxHost, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t &size, bool increCreateChannelFlag,
    const ResPackGraphMode &resPack)
{
    // host侧资源
    if (param.engine == COMM_ENGINE_RESERVED) {
        // COMM_ENGINE_RESERVED
    } else if (param.engine == COMM_ENGINE_CPU) {
        CHK_RET(GetAlgResDPU(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
            size, increCreateChannelFlag, resPack));
    } else if (param.engine == COMM_ENGINE_CPU_TS) {
        // COMM_ENGINE_CPU_TS
    } else if (param.engine == COMM_ENGINE_AICPU) {
        // COMM_ENGINE_AICPU
    } else if (param.engine == COMM_ENGINE_AICPU_TS) {
        CHK_RET(GetAlgResAICPU(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
                               size, increCreateChannelFlag, resPack));
    } else if (param.engine == COMM_ENGINE_AIV) {
        CHK_RET(GetAlgResAiv(comm, param, resRequest, topoInfo, algHierarchyInfo, resCtxSequence));
    } else if (param.engine == COMM_ENGINE_CCU) {
        // 添加资源回退。SetCommEngine
        auto ret = GetAlgResCcu(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence, size, resPack);
        if (ret == HCCL_E_UNAVAIL) {
            return HCCL_E_UNAVAIL;
        }
        CHK_RET(ret);
    } else {
        HCCL_ERROR("fail to get engine, invalid engine type[%s].", GetEnumToString(GetCommEngineStatusStrMap(), param.engine).c_str());
        return HCCL_E_PARA;
    }
    param.ctxSize = size;
    return HCCL_SUCCESS;
}

HcclResult CacheHostCtxToEngine(HcclComm comm, const char *algTag, const std::string &hostCacheTag,
    const std::vector<char> &hostCtxSeq)
{
    void *hostCtxPtr = nullptr;
    HcclResult createRet = HcclEngineCtxCreate(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS,
        hostCtxSeq.size(), &hostCtxPtr);
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

HcclResult ReuseCachedDeviceCtx(HcclComm comm, const OpParam &param, void **resCtxSequence, uint64_t &ctxSize)
{
    void *ctx = nullptr;
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

HcclResult IncrementalCreateChannel(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    AlgResourceCtxSerializable &hostCtxObj, const std::string &hostCacheTag, void **resCtxSequence,
    uint64_t &ctxSize)
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
    void *newHostCtxPtr = nullptr;
    HcclResult cacheRet = HcclEngineCtxCreate(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS,
        newSeq.size(), &newHostCtxPtr);
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

HcclResult GetAlgResAICPU(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag, const ResPackGraphMode &resPack)
{
    std::string hostCacheTag = std::string(param.algTag) + "_hostCache";
    void *hostCtxPtr = nullptr;
    uint64_t hostCtxSize = 0;
    HcclResult hostCtxRet = HcclEngineCtxGet(comm, hostCacheTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS,
        &hostCtxPtr, &hostCtxSize);
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
        CHK_RET(IncrementalCreateChannel(comm, param, resRequest, hostCtxObj, hostCacheTag,
            resCtxSequence, ctxSize));
    }

    HCCL_INFO("Execute GetAlgResAICPU success.");
    return HCCL_SUCCESS;
}

HcclResult HcclMemcpyCtxHostToDevice(HcclComm comm, const OpParam &param,
    const std::vector<char>& seq, void **resCtxSequence, uint64_t& ctxSize)
{
    uint64_t size = seq.size();
    void *ctx = nullptr;
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
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute AllocAlgResource.");
    void *cclBufferAddr;
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

static HcclResult HcclGetThreadWithConfig(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    u32 threadNum, std::vector<ThreadHandle> &threads, std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost,
    bool unfoldReady)
{
    std::vector<ThreadConfig> threadConfigs(threadNum);
    CHK_RET(static_cast<HcclResult>(ThreadConfigInit(threadConfigs.data(), threadNum)));
    threadConfigs[0].notifyNumPerThread = resRequest.notifyNumOnMainThread + 1; // 主流上多一个用于host-device同步
    HCCL_DEBUG("[HcclGetThread] AICPU thread[0] notify num[%u].", threadConfigs[0].notifyNumPerThread);
    CHK_PRT_RET(resRequest.notifyNumPerThread.size() < threadNum - 1,
        HCCL_ERROR("[HcclGetThread] notifyNumPerThread size[%zu] is less than slaveThreadNum[%u].",
            resRequest.notifyNumPerThread.size(), threadNum - 1), HCCL_E_INTERNAL);
    for (u32 i = 1; i < threadNum; i++) {
        threadConfigs[i].notifyNumPerThread = resRequest.notifyNumPerThread[i - 1];
        HCCL_DEBUG("[HcclGetThread] AICPU thread[%u] notify num[%u].", i, threadConfigs[i].notifyNumPerThread);
    }
    CHK_RET(HcclThreadAcquireWithConfig(comm, COMM_ENGINE_AICPU, threadNum, THREAD_TYPE_TS,
        threadConfigs.data(), threads.data()));
    // 申请展开流对应的Thread
    if (!unfoldReady) {
        ThreadConfig unfoldThreadConfig;
        CHK_RET(static_cast<HcclResult>(ThreadConfigInit(&unfoldThreadConfig, 1)));
        unfoldThreadConfig.notifyNumPerThread = 0;
        CHK_RET(HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, 1, THREAD_TYPE_TS,
            &unfoldThreadConfig, &resCtxHost->unfoldThread));
    }
    CHK_RET(SaveMainThreadInfo(comm, param, threads[0], resRequest.notifyNumOnMainThread + 1));
    return HCCL_SUCCESS;
}

static u32 GetMaxNotifyNum(const std::vector<u32> &notifyNumPerThread, u32 initNotifyNum)
{
    u32 maxNotifyNum = initNotifyNum;
    for (u32 notifyNum : notifyNumPerThread) {
        if (notifyNum > maxNotifyNum) {
            maxNotifyNum = notifyNum;
        }
    }
    return maxNotifyNum;
}

static HcclResult HcclGetAicpuThread(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
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
            CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CPU, 1, 0, &resCtxHost->unfoldThread));
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
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode &resPack)
{
    resCtxHost->isHcclThreadAcquireWithConfigSupported = HcommIsSupportHcclThreadAcquireWithConfig();
    if ((param.engine == COMM_ENGINE_AICPU_TS) || (param.engine == COMM_ENGINE_CPU)) {
        CHK_RET(HcclGetAicpuThread(comm, param, resRequest, resCtxHost));
    } else {
        // host模式下，将主流封装为thread，并创建主流上的notify
        ThreadHandle thread;
        CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, param.stream, resRequest.notifyNumOnMainThread, &thread));
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

HcclResult GeGetThread(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode &resPack, u32 maxNotifyNum)
{
    if (param.opMode == OpMode::OPBASE) {
        u32 threadNum = resRequest.slaveThreadNum;
        if (threadNum > 0) {
            std::vector<ThreadHandle> threads(threadNum);
            if (HcommIsSupportHcclThreadAcquireWithConfig()) {
                std::vector<ThreadConfig> threadConfigs(threadNum);
                CHK_RET(static_cast<HcclResult>(ThreadConfigInit(threadConfigs.data(), threadNum)));
                CHK_PRT_RET(resRequest.notifyNumPerThread.size() < threadNum,
                    HCCL_ERROR("[GeGetThread] notifyNumPerThread size[%zu] is less than slaveThreadNum[%u].",
                        resRequest.notifyNumPerThread.size(), threadNum), HCCL_E_INTERNAL);
                for (u32 i = 0; i < threadNum; i++) {
                    threadConfigs[i].notifyNumPerThread = resRequest.notifyNumPerThread[i];
                }
                CHK_RET(HcclThreadAcquireWithConfig(comm, COMM_ENGINE_CPU, threadNum, THREAD_TYPE_TS,
                    threadConfigs.data(), threads.data()));
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
            HCCL_ERROR("Thread Num Should less than slave streams. slaveStreams[%llu], threadNums[%llu]", slaveStreams, threadNum);
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

HcclResult SaveMainThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle thread, u32 notifyNum)
{
    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);
    void *ctx = nullptr;
    // 申请一块host类型内存，保存主流信息
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    // 填充主流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    *threadPtr = thread;
    // 填充主流notify数量信息
    char* curPtr = reinterpret_cast<char *>(ctx);
    curPtr += sizeof(ThreadHandle);
    u32 *notifyNumPtr = reinterpret_cast<u32 *>(curPtr);
    *notifyNumPtr = notifyNum;
    HCCL_INFO("[SaveMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]",
        threadPtr, thread, notifyNumPtr, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult SaveUnfoldThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle unfoldThread)
{
    uint64_t size = sizeof(ThreadHandle);
    void *ctx = nullptr;
    // 申请一块host类型内存，保存展开流信息
    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, "%s_unfold", param.commName);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill unfoldAlgTag", __func__), HCCL_E_INTERNAL);
    CHK_RET(HcclEngineCtxCreate(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx));
    // 填充主流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    *threadPtr = unfoldThread;
    HCCL_INFO("[SaveUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]",
        unfoldAlgTag, threadPtr, unfoldThread);
    return HCCL_SUCCESS;
}

HcclResult GetUnfoldThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle& unfoldThread)
{
    uint64_t size = sizeof(ThreadHandle);
    void *ctx = nullptr;
    char unfoldAlgTag[ALG_TAG_LENGTH] = {0};
    int ret = snprintf_s(unfoldAlgTag, sizeof(unfoldAlgTag), sizeof(unfoldAlgTag) - 1, "%s_unfold", param.commName);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill unfoldAlgTag", __func__), HCCL_E_INTERNAL);
    HcclResult getRet = HcclEngineCtxGet(comm, unfoldAlgTag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size);
    if (getRet != HCCL_SUCCESS) {
        return getRet;
    }
    // 获取展开流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    unfoldThread = *threadPtr;
    HCCL_INFO("[GetUnfoldThreadInfo]unfoldAlgTag[%s], threadPtr[%p], unfoldThread[%lu]",
        unfoldAlgTag, threadPtr, unfoldThread);
    return HCCL_SUCCESS;
}

HcclResult GetMainThreadInfo(HcclComm comm, const OpParam &param, ThreadHandle &thread, u32 &notifyNum)
{
    uint64_t size = sizeof(ThreadHandle) + sizeof(u32);
    void *ctx = nullptr;
    CHK_RET(HcclEngineCtxGet(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &size));

    // 获取主流handle信息
    ThreadHandle* threadPtr = reinterpret_cast<ThreadHandle *>(ctx);
    thread = *threadPtr;
    // 获取主流notify数量信息
    char* curPtr = reinterpret_cast<char *>(ctx);
    curPtr += sizeof(ThreadHandle);
    u32 *notifyNumPtr = reinterpret_cast<u32 *>(curPtr);
    notifyNum = *notifyNumPtr;
    HCCL_INFO("[GetMainThreadInfo]threadPtr[%p], thread[%lu], notifyNumPtr[%p], notifyNum[%lu]",
        threadPtr, thread, notifyNumPtr, notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannel(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
                          AlgResourceCtxSerializable* resCtxHost)
{
    MemRegInfo memRegInfo;
    if (param.opMode == OpMode::OFFLOAD) {
        HCCL_INFO("[HcclGetChannelImpl] start to RegGraphModeBuffers");
        CHK_RET(RegGraphModeBuffers(comm, param, memRegInfo.inputBuffTag, memRegInfo.outputBuffTag, memRegInfo.memHandles));
    }
    resCtxHost->channels.resize(resRequest.channels.size());
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc> &levelNChannelRequest = resRequest.channels[level];
        std::vector<HcclChannelDesc> deviceChannelRequest;
        std::vector<HcclChannelDesc> hostChannelRequest;
        for (auto &channelRequest : levelNChannelRequest) {
            if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                deviceChannelRequest.emplace_back(channelRequest);
            } else if (channelRequest.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                hostChannelRequest.emplace_back(channelRequest);
            }
        }
        // device建链
        CHK_RET(HcclGetChannelImpl(level, comm, param, deviceChannelRequest, COMM_ENGINE_AICPU_TS, resCtxHost, memRegInfo));
        // host建链
        CHK_RET(HcclGetChannelImpl(level, comm, param, hostChannelRequest, COMM_ENGINE_CPU, resCtxHost, memRegInfo));

    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannelImpl(const u32 level, HcclComm comm, const OpParam &param, std::vector<HcclChannelDesc>& channelRequest,
                              const CommEngine commEngine, AlgResourceCtxSerializable* resCtxHost, MemRegInfo &memRegInfo) {
    // 获取子通信域的建链数量
    if (channelRequest.empty()) {
        HCCL_INFO("[HcclGetChannelImpl] channelRequest is empty");
        return HCCL_SUCCESS;
    }
    u32 channelNum = channelRequest.size();
    std::vector<ChannelHandle> levelNChannels;
    levelNChannels.resize(channelNum);
    if (param.opMode == OpMode::OFFLOAD) {
        for (auto &channelDesc : channelRequest) {
            channelDesc.memHandles = memRegInfo.memHandles.data();
            channelDesc.memHandleNum = memRegInfo.memHandles.size();
        }
    }
    if (channelNum > 0) {
        // 参数一致性校验信息注册到通信域，HcclChannelAcquire内部存在读清动作，每次调用前均需注册
        CHK_RET(AddExchangeInfo(comm, param));
        CHK_RET(HcclChannelAcquire(comm, commEngine, channelRequest.data(),
            channelNum, levelNChannels.data()));
    }

    for (u32 idx = 0; idx < channelNum; idx++) {
        ChannelInfo channel;
        // 对于真实建链的链路进行填充
        const HcclChannelDesc &channelDescNew = channelRequest[idx];
        channel.isValid = true;
        channel.remoteRank = channelDescNew.remoteRank;
        channel.protocol = channelDescNew.channelProtocol;
        channel.locationType = channelDescNew.remoteEndpoint.loc.locType;
        channel.notifyNum = channelDescNew.notifyNum;
        channel.handle = levelNChannels[idx];
#ifndef AICPU_COMPILE
        EndpointDesc localEndpoint = channelDescNew.localEndpoint;
        using portSizeType = uint32_t;
        const uint32_t portSizeTypeSize = sizeof(portSizeType);
        portSizeType portSize = 0;
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, resCtxHost->topoInfo.userRank, &localEndpoint,
                ENDPOINT_ATTR_BW_COEFF, portSizeTypeSize, static_cast<void*>(&portSize)));
        channel.portGroupSize = portSize;
        CHK_PRT_RET(portSize == 0, HCCL_ERROR("[HcclGetChannelImpl] userRank [%d], portSize [%u] is 0.", resCtxHost->topoInfo.userRank, portSize), HcclResult::HCCL_E_INTERNAL);
#endif
        void* remoteCclBufferAddr = nullptr;
        uint64_t remoteCclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteCclBufferAddr, &remoteCclBufferSize));
        channel.remoteCclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, remoteCclBufferAddr, remoteCclBufferSize};
        HCCL_INFO("[%s]remoteRank[%u] protocol[%u] remoteCclBufferAddr[0x%llx] remoteCclBufferSize[%u]",
            __func__, channelDescNew.remoteRank,channelDescNew.channelProtocol, remoteCclBufferAddr, remoteCclBufferSize);

        if (param.opMode == OpMode::OFFLOAD) {
            CHK_RET(GetGraphModeBuffers(comm, levelNChannels[idx], memRegInfo.inputBuffTag, memRegInfo.outputBuffTag, channel));
        }
        resCtxHost->channels[level].push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult RegGraphModeBuffers(HcclComm comm, const OpParam &param,char* inputBuffTag, char* outputBuffTag, std::vector<HcclMemHandle>& memHandles) {
    HCCL_INFO("[RegGraphModeBuffers] param.tag[%s]", param.tag);
    auto retIn = sprintf_s(inputBuffTag, MAX_MEM_TAG_LENGTH, "%s_%s", param.tag, "InputBuffer");
    auto retOut =  sprintf_s(outputBuffTag, MAX_MEM_TAG_LENGTH, "%s_%s", param.tag, "OutputBuffer");
    if (retIn <= 0 || retOut <= 0){
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

HcclResult GetGraphModeBuffers(HcclComm comm, ChannelHandle channelHandle, const char* inputBuffTag, const char* outputBuffTag, ChannelInfo& channel) {
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

HcclResult GetAlgResCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                        std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails* topoInfo,
                        AlgHierarchyInfoForAllLevel& algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize, const ResPackGraphMode &resPack)
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

    void *ctx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, param.engine, size, &ctx));
    CHK_SAFETY_FUNC_RET(memcpy_s(ctx, size, seq.data(), size));
    *resCtxSequence = ctx;
    ctxSize = size;
    HCCL_INFO("Execute GetAlgResCCU success.");
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceCcu(HcclComm comm, const OpParam& param, AlgResourceRequest& resRequest,
                                   std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, const ResPackGraphMode &resPack)
{
    HCCL_INFO("Start to execute AllocAlgResource.");
    void *cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    // CCL IN使用所有的CCL Buffer，这个其实就是scratch buffer
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};
    resCtxHost->notifyNumOnMainThread = resRequest.notifyNumOnMainThread;
    resCtxHost->slaveThreadNum = resRequest.slaveThreadNum;
    resCtxHost->notifyNumPerThread = resRequest.notifyNumPerThread;
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

    ret = HcclGetCcuKernel(comm, resRequest, resCtxHost);
    if (ret == HCCL_E_UNAVAIL) {
        // 进行资源回退
        HCCL_WARNING("[HcclGetCcuKernel] ccu kernel unavailable, try to fallback.");
        return HCCL_E_UNAVAIL;
    } else {
        CHK_RET(ret);
    }
#endif
    return HCCL_SUCCESS;
}

#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
HcclResult HcclGetChannelForCcu(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest)
{
    // 以kernel为粒度申请channel
    for (CcuKernelInfo& kernelInfo: resRequest.ccuKernelInfos) {
        std::vector<HcclChannelDesc> &kernelChannelRequest = kernelInfo.channels;

        u32 channelNum = kernelChannelRequest.size();
        std::vector<ChannelHandle> kernelChannels;
        kernelChannels.resize(channelNum);

        if (channelNum > 0) {
            // 参数一致性校验信息注册到通信域，HcclChannelAcquire内部存在读清动作，每次调用前均需注册
            CHK_RET(AddExchangeInfo(comm, param));
            auto ret = HcclChannelAcquire(comm, param.engine, kernelChannelRequest.data(),
                channelNum, kernelChannels.data());
            // 需要资源回退。返回资源不够
            if (ret == HCCL_E_UNAVAIL) {
                HCCL_WARNING("[HcclChannelAcquire] channel unavailable, channel num[%u].", channelNum);
                return HCCL_E_UNAVAIL;
            } else {
                CHK_RET(ret);
            }
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
        HCCL_INFO("[HcclGetChannelForCcu] Get [%lu] channels", channelNum);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetCcuKernel(HcclComm comm, AlgResourceRequest &resRequest,
                          std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("[HcclGetCcuKernel] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
                HCCL_E_INTERNAL);

    u32 totalKernelNum = 0;
    for (auto t: resRequest.ccuKernelNum) {
        totalKernelNum += t;
    }
    CHK_PRT_RET(totalKernelNum != resRequest.ccuKernelInfos.size(),
                HCCL_ERROR("[HcclGetCcuKernel]ccuKernel num not match!"), HCCL_E_INTERNAL);

    // 按照resgroup进行注册
    u32 currentResGroup = 0;
    u32 maxResGroup = 0;
    resCtxHost->ccuKernels.resize(totalKernelNum);

    while (currentResGroup <= maxResGroup) {
        CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
        if (regStartRet != CCU_SUCCESS) {
            HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regStartRet);
            return ConvertCcuToHccl(regStartRet);
        }
        for (u32 i = 0; i < totalKernelNum; i++) {
            CcuKernelInfo& kernelInfo = resRequest.ccuKernelInfos[i];
            if (kernelInfo.resGroup > maxResGroup) {
                maxResGroup = kernelInfo.resGroup;
            }
            if (kernelInfo.resGroup != currentResGroup) {
                continue;
            }

            HCCL_DEBUG("[HcclGetCcuKernel] kernelFuncName[%s]", kernelInfo.kernelFuncName);
            CcuKernelHandle kernelHandle;
            const void *kernelArgs[] = {kernelInfo.kernelArg};

            constexpr uint32_t dieId = 0; // 预留接口，暂无含义
            constexpr uint32_t kernelArgNum = 1;
            CcuResult regRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
                                                      reinterpret_cast<void*>(kernelInfo.kernelFunc),
                                                      kernelArgs, kernelArgNum, &kernelHandle);
            if (regRet != CCU_SUCCESS) {
                HCCL_ERROR("ccu kernel register failed: ccuRet -> %d", regRet);
                return ConvertCcuToHccl(regRet);
            }
            resCtxHost->ccuKernels[i] = kernelHandle;
        }
        CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
        if (regEndRet == CCU_E_UNAVAIL) {
            HCCL_WARNING("[HcclGetCcuKernel] ccu kernel register end unavailable, try to fallback.");
            return HCCL_E_UNAVAIL;
        } else if (regEndRet != CCU_SUCCESS) {
            HCCL_ERROR("ccu kernel register end failed: ccuRet -> %d", regEndRet);
            return ConvertCcuToHccl(regEndRet);
        }
        currentResGroup++;
    }
    resCtxHost->ccuKernelNum = resRequest.ccuKernelNum;
    return HCCL_SUCCESS;
}
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0) */

HcclResult GetAlgResAiv(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence)
{
    uint64_t size = sizeof(AlgResourceCtxSerializable);
    CHK_RET(HcclEngineCtxCreate(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS, size, resCtxSequence));

    AlgResourceCtxSerializable* resCtxHost = static_cast<AlgResourceCtxSerializable *>(*resCtxSequence);
    resCtxHost->topoInfo = *topoInfo;
    resCtxHost->algHierarchyInfo = algHierarchyInfo;

    CHK_RET(HcclAllocAlgResourceAiv(comm, param, resRequest, resCtxHost));
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceAiv(
    HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest, AlgResourceCtxSerializable* resCtxHost)
{
    HCCL_INFO("[%s]Start to execute.", __func__);
    HcclMemHandle memHandle; // 注册到通信域内存的handle，用于建链
    // 获取存放AIV对端信息和标记区的空间
    uint64_t commInfoSize = 0;
    HcclResult ret = HcclEngineCtxGet(comm, param.commModeTag, param.engine, &(resCtxHost->aivCommInfoPtr), &commInfoSize);
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        CHK_RET(HcclEngineCtxCreate(comm, param.commModeTag, param.engine, AIV_TAG_BUFF_LEN, &(resCtxHost->aivCommInfoPtr)));
        // 清零
        ACLCHECK(haclrtMemset(resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN));
        if (HcommIsSupportHcclCommRegCommStateCallback()) {
            CHK_RET(HcclCommRegCommStateCallback(param.commModeTag, ClearAivTagCb, resCtxHost->aivCommInfoPtr));
        }
        // 注册到通信域，支持建链时交换
        CommMem regMem{COMM_MEM_TYPE_DEVICE, resCtxHost->aivCommInfoPtr, AIV_TAG_BUFF_LEN};
        CHK_RET(HcclCommMemReg(comm, param.commModeTag, &regMem, &memHandle));
        void* memHandleCachePtr = nullptr; // 当前AIV存放注册内存的memHandle使用
        CHK_RET(HcclEngineCtxCreate(comm, param.commModeTag, CommEngine::COMM_ENGINE_CPU_TS, sizeof(HcclMemHandle), &memHandleCachePtr));
        static_cast<HcclMemHandle*>(memHandleCachePtr)[0] = memHandle;
    } else {
        void* memHandleCachePtr = nullptr;
        uint64_t memHandleCacheSize = 0;
        HcclResult ret = HcclEngineCtxGet(comm, param.commModeTag, CommEngine::COMM_ENGINE_CPU_TS, &memHandleCachePtr, &memHandleCacheSize);
        CHK_PRT_RET(ret != HCCL_SUCCESS || memHandleCacheSize != sizeof(HcclMemHandle),
            HCCL_ERROR("[%s]commModeTag[%s] aiv memHandle not found in cache, ptr[%p] size[%llu]",
                __func__, param.commModeTag, memHandleCachePtr, memHandleCacheSize),
                HCCL_E_INTERNAL);
        memHandle = static_cast<HcclMemHandle*>(memHandleCachePtr)[0];
    }
    HCCL_INFO("[%s]commModeTag[%s] regMemAddr[%p] memHandle[%p]", __func__, param.commModeTag, resCtxHost->aivCommInfoPtr,
        memHandle);

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    HCCL_INFO("[%s]local cclBufferAddr[%p] cclBufferSize[%llu]", __func__, cclBufferAddr, cclBufferSize);
    resCtxHost->cclMem = HcclMem{HCCL_MEM_TYPE_DEVICE, cclBufferAddr, cclBufferSize};

    void* buffersIn[MAX_RANK_SIZE] = {};
    void* buffersOut[MAX_RANK_SIZE] = {};
    buffersIn[resCtxHost->topoInfo.userRank] = cclBufferAddr;
    buffersOut[resCtxHost->topoInfo.userRank] = resCtxHost->aivCommInfoPtr;

    // 迭代每个子通信域的建链请求，创建链路
    for (u32 level = 0; level < resRequest.channels.size(); level++) {
        // 获取子通信域的建链请求
        std::vector<HcclChannelDesc> &levelNChannelRequest = resRequest.channels[level];
        for (auto &channelDesc : levelNChannelRequest) {
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
            CHK_RET(HcclChannelAcquire(comm, param.engine, levelNChannelRequest.data(),
                validChannelNum, levelNChannels.data()));
        }

        for (u32 idx = 0; idx < validChannelNum; idx++) {
            HcclChannelDesc &channelDesc = levelNChannelRequest[idx];
            CHK_PRT_RET(channelDesc.remoteRank >= MAX_RANK_SIZE,
                HCCL_ERROR("[%s] remoteRank[%u] exceeds MAX_RANK_SIZE[%u]", __func__, channelDesc.remoteRank, MAX_RANK_SIZE), HCCL_E_PARA);
            void* remoteBufferAddr;
            uint64_t remoteBufferSize;
            CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
            HCCL_RUN_INFO("[%s]remoteRank[%u] cclBufferAddr[%p] cclBufferSize[%llu]", __func__, channelDesc.remoteRank,
                remoteBufferAddr, remoteBufferSize);
            buffersIn[channelDesc.remoteRank] = remoteBufferAddr;

            u32 memNum;
            CommMem* remoteMems;
            char** memTags;
            CHK_RET(HcclChannelGetRemoteMems(comm, levelNChannels[idx], &memNum, &remoteMems, &memTags));
            CHK_PRT_RET(memNum == 0,
                HCCL_ERROR("[%s] HcclChannelGetRemoteMems memNum is 0", __func__), HCCL_E_PARA);
            HCCL_RUN_INFO("[%s]remoteRank[%u] memNum[%u] regMemAddr[%p] regMemSize[%llu] memTag[%s]", __func__,
                channelDesc.remoteRank, memNum, remoteMems[memNum - 1].addr, remoteMems[memNum - 1].size,
                memTags[memNum - 1]);
            buffersOut[channelDesc.remoteRank] = remoteMems[memNum - 1].addr;
        }
    }

    CHK_RET(haclrtMemcpy(resCtxHost->aivCommInfoPtr, MAX_RANK_SIZE * sizeof(void*), buffersIn, MAX_RANK_SIZE * sizeof(void*),
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHK_RET(haclrtMemcpy(static_cast<u8*>(resCtxHost->aivCommInfoPtr) + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(void*),
        buffersOut, MAX_RANK_SIZE * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE));

    HCCL_INFO("[%s] Alloc res success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult GetAlgResDPU(HcclComm comm, const OpParam &param, AlgResourceRequest &resRequest,
    std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost, TopoInfoWithNetLayerDetails *topoInfo,
    AlgHierarchyInfoForAllLevel &algHierarchyInfo, void **resCtxSequence, uint64_t& ctxSize,
    bool increCreateChannelFlag, const ResPackGraphMode &resPack)
{
    // 申请共享内存
    uint64_t shmemSize = 100 * 1024 * 1024;
    void *shmemPtr = nullptr;
    bool newCreated;
    CHK_RET(HcclDevMemAcquire(comm, "DPUTAG", &shmemSize, &shmemPtr, &newCreated));
    resCtxHost->npu2DpuShmemPtr = shmemPtr;
    constexpr uint64_t DPU2NPU_SHMEM_RATIO = 2;
    resCtxHost->dpu2NpuShmemPtr = static_cast<void*>(static_cast<uint8_t*>(shmemPtr) + shmemSize / DPU2NPU_SHMEM_RATIO);

    CHK_RET(GetAlgResAICPU(comm, param, resRequest, resCtxHost, topoInfo, algHierarchyInfo, resCtxSequence,
                           ctxSize, increCreateChannelFlag, resPack));

    HCCL_INFO("Execute GetAlgResAICPU success.");
    return HCCL_SUCCESS;
}

HcclResult CheckCount(const u64 count)
{
    if (UNLIKELY(count > SYS_MAX_COUNT)) {
        HCCL_ERROR("[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
                    HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckDataType(const HcclDataType dataType, bool needReduce)
{
    const std::vector<std::string> infoTitle({"ccl_op", "value", "parameter", "expect"});
    if (needReduce) {
        if ((dataType == HCCL_DATA_TYPE_UINT8)   || (dataType == HCCL_DATA_TYPE_UINT16)  ||
            (dataType == HCCL_DATA_TYPE_UINT32)  || (dataType == HCCL_DATA_TYPE_INT128)  ||
            (dataType == HCCL_DATA_TYPE_HIF8)    || (dataType == HCCL_DATA_TYPE_FP8E4M3) ||
            (dataType == HCCL_DATA_TYPE_FP8E5M2) || (dataType == HCCL_DATA_TYPE_FP8E8M0) ||
            (dataType == HCCL_DATA_TYPE_RESERVED)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, std::vector<std::string>({"CheckDataType", GetDataTypeEnumStr(dataType), "dataType",
                GetSupportDataType(needReduce)}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8) ||
            (dataType == HCCL_DATA_TYPE_INT128)) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, std::vector<std::string>({"CheckDataType", GetDataTypeEnumStr(dataType), "dataType",
                GetSupportDataType(needReduce).c_str()}));
            HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                        GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

std::string GetSupportDataType(bool needReduce)
{
    std::vector<HcclDataType> supportList = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32,
                                             HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32};
    if (needReduce) {
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_UINT64,
                                               HCCL_DATA_TYPE_FP64});
    } else {
        supportList.insert(supportList.end(), {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,
                                               HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64,
                                               HCCL_DATA_TYPE_HIF8, HCCL_DATA_TYPE_FP8E4M3,  HCCL_DATA_TYPE_FP8E5M2,
                                               HCCL_DATA_TYPE_FP8E8M0});
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
    std::vector<HcclDataType> prodSupportList = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,
                                                 HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP64};
    const std::vector<std::string> infoTitle({"ccl_op", "value", "parameter", "expect"});
    if (op == HcclReduceOp::HCCL_REDUCE_PROD) {
        if (std::find(prodSupportList.begin(), prodSupportList.end(), dataType) == prodSupportList.end()) {
            RPT_INPUT_ERR(true, "EI0003", infoTitle, std::vector<std::string>({"CheckReduceDataType", GetDataTypeEnumStr(dataType), "dataType",
                GetReduceProdSupportDataType()}));
            HCCL_ERROR("[Check][ReduceOp][DataType]errNo[0x%016llx] reduceop is [%s] data type[%s] not supported, support range=[%s]",
                        HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetReduceOpEnumStr(op).c_str(), GetDataTypeEnumStr(dataType).c_str(),
                        GetReduceProdSupportDataType().c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

std::string GetReduceProdSupportDataType()
{
    std::vector<HcclDataType> supportList = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64,
                                             HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP64};
    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

HcclResult SetCommEngine(OpParam &param)
{
    // 使用一个静态的映射表来关联配置和引擎值
    static const std::unordered_map<OpExecuteConfig, CommEngine> ConfigToEngineMap = {
        {OpExecuteConfig::HOSTCPU_TS, COMM_ENGINE_CPU_TS},
        {OpExecuteConfig::AICPU_TS,   COMM_ENGINE_AICPU_TS},
        {OpExecuteConfig::AIV,        COMM_ENGINE_AIV},
        {OpExecuteConfig::AIV_ONLY,  COMM_ENGINE_AIV}, // AIV_ONLY 和 AIV 映射到同一引擎
        {OpExecuteConfig::CCU_MS,     COMM_ENGINE_CCU},
        {OpExecuteConfig::CCU_SCHED,  COMM_ENGINE_CCU},
        {OpExecuteConfig::AICPU,      COMM_ENGINE_AICPU},
        {OpExecuteConfig::HOSTCPU,    COMM_ENGINE_CPU},
    };

    auto it = ConfigToEngineMap.find(param.opExecuteConfig);
    if (it != ConfigToEngineMap.end()) {
        param.engine = it->second;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[op_common][SetCommEngine] Unsupported or unknown opExecuteConfig: {%d}", static_cast<int>(param.opExecuteConfig));
    return HCCL_E_NOT_SUPPORT;
}

HcclResult SingleRankProc(HcclComm comm, OpParam &param)
{
    if (param.commOpExpansionMode == HcclOpExpansionMode::HCCL_OP_EXPANSION_AIV_ONLY) {
        HCCL_ERROR("[SingleRankProc] opType[%d] currently do not select aiv mode, aiv only not support, "
            "please ensure rankNum is greater than one", static_cast<int>(param.opType));
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
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        len = DATATYPE_SIZE_TABLE[param.all2AllVDataDes.sendType] * *(static_cast<const u64 *>(param.all2AllVDataDes.sendCounts));
    } else if (param.opType == HCCL_CMD_ALLGATHER_V || param.opType == HCCL_CMD_REDUCE_SCATTER_V) {
        len = DATATYPE_SIZE_TABLE[param.vDataDes.dataType] * *(static_cast<const u64 *>(param.vDataDes.counts));
    } else {
        len = DATATYPE_SIZE_TABLE[param.DataDes.dataType] * param.DataDes.count;
    }

    HCCL_INFO("[%s] sendBuf[%p], recvBuf[%p], len[%llu]", __func__, param.inputPtr, param.outputPtr, len);
    if (len > 0) {
        ThreadHandle cpuTsThread{0};
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, param.stream, 1, &cpuTsThread));
        // Op注册
        HcclDfxOpInfoCompat hcclDfxOpInfo{};
        hcclDfxOpInfo.opMode = static_cast<u32>(param.opMode);
        hcclDfxOpInfo.opType = static_cast<u32>(param.opType);
        hcclDfxOpInfo.reduceOp = static_cast<u32>(param.reduceType);
        CHK_RET(GetHcclDfxOpInfoDataType(param, hcclDfxOpInfo.dataType));

        // rankSize获取指定算子的dataCount
        u32 userRankSize{0};
        CHK_RET(HcclGetRankSize(comm, &userRankSize));
        CHK_RET(GetHcclDfxOpInfoDataCount(param, userRankSize, hcclDfxOpInfo.dataCount));
        hcclDfxOpInfo.root = param.root;
        hcclDfxOpInfo.engine = param.engine;
        hcclDfxOpInfo.cpuTsThread = cpuTsThread;
        hcclDfxOpInfo.cpuWaitAicpuNotifyIdx = HOST_WAIT_AICPU_NOTIFYIDX;
        CHK_RET(SetOpParamAlgTag(param, "SingleRankProc"));
        s32 sRet = strncpy_s(hcclDfxOpInfo.algTag, ALG_TAG_LENGTH, param.algTag, ALG_TAG_LENGTH);
        CHK_PRT_RET(sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, param.algTag %s,  return %d.",
            __func__, param.algTag, sRet), HCCL_E_MEMORY);

        CHK_RET(HcclDfxRegOpInfoByCommId(param.commName, reinterpret_cast<void*>(&hcclDfxOpInfo)));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(cpuTsThread, param.outputPtr, param.inputPtr, len)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclCheckTag(const char *tag)
{
    CHK_PTR_NULL(tag);

    u32 tagLen = strnlen(tag, TAG_MAX_LEN + 1);
    if (UNLIKELY((tagLen == (TAG_MAX_LEN + 1) || tagLen == 0))) {
        HCCL_ERROR("[Check][Tag]errNo[0x%016llx] tag is too long", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

static HcclResult BuildCcuExtraTag(const OpParam &param, std::string &ccuExtraTag)
{
    HcclDataType tmpDataType;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        tmpDataType = param.all2AllVDataDes.sendType;
    } else if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V ||
               param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        tmpDataType = param.vDataDes.dataType;
    } else {
        tmpDataType = param.DataDes.dataType;
    }
    ccuExtraTag = "_" + HCOM_DATA_TYPE_STR_MAP.at(tmpDataType);

    if (param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE ||
        param.opType == HcclCMDType::HCCL_CMD_REDUCE ||
        param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
        param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        ccuExtraTag += "_" + HCOM_REDUCE_OP_STR_MAP.at(param.reduceType);
    }

    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE || param.opType == HcclCMDType::HCCL_CMD_SCATTER ||
        param.opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        ccuExtraTag += "_r" + std::to_string(param.root);
    }
    return HCCL_SUCCESS;
}

HcclResult SetOpParamAlgTag(OpParam &param, const std::string &algName)
{
    std::string temp = algName; // 创建algName的副本

    const char* launchMode = (((param.engine == CommEngine::COMM_ENGINE_AICPU) ||
                                (param.engine == CommEngine::COMM_ENGINE_AICPU_TS)) ? "device" : "host");
    int len;
    // 图模式下去掉param.tag前缀，避免tag不同导致algTag不同而无法复用资源
    if (param.opMode == OpMode::OFFLOAD && param.engine == CommEngine::COMM_ENGINE_CCU) {
        len = snprintf_s(param.algTag, sizeof(param.algTag), sizeof(param.algTag), "Graph_%s_%s", temp.c_str(), launchMode);
    } else {
        len = snprintf_s(param.algTag, sizeof(param.algTag), sizeof(param.algTag), "%s_%s_%s", param.tag, temp.c_str(), launchMode);
    }
    if (len < 0|| len >= sizeof(param.algTag)) {
        HCCL_ERROR("failed to fill param.algTag");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // ccu模式，考虑kernel是否能复用，需要添加dataType和reduceType
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        try{
            std::string ccuExtraTag;
            CHK_RET(BuildCcuExtraTag(param, ccuExtraTag));
            size_t remainBytes = sizeof(param.algTag) - len;

            int len_ccu = snprintf_s(param.algTag + len, remainBytes, remainBytes, "%s", ccuExtraTag.c_str());
            CHK_PRT_RET((len_ccu < 0 || len_ccu >= sizeof(param.algTag) - len),
                HCCL_ERROR("failed to fill alg tag with ccu dataType"), HCCL_E_INTERNAL);
        }
        catch (const std::out_of_range& e) {
            HCCL_ERROR("[SetOpParamAlgTag] dataType or reduceType out of range: %s", e.what());
            return HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HcclGetOpExpansionMode(HcclComm comm, OpParam &param)
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

HcclResult DecideHcclOpExpansionMode(HcclComm comm, HcclOpExpansionMode &finalMode)
{
    HcclOpExpansionMode configOpExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    bool useConfigOpExpansionMode = false;
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (hcommFunction.dlHcclConfigGetInfo) {
        uint32_t infoLen = sizeof(HcclOpExpansionMode);
        CHK_RET(hcommFunction.dlHcclConfigGetInfo(comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, infoLen,
            &configOpExpansionMode));
        finalMode = configOpExpansionMode;
        useConfigOpExpansionMode = true;
    } else {
        HCCL_INFO("[DecideHcclOpExpansionMode] HcclConfigGetInfo is not supported, use environment mode.");
        finalMode = static_cast<HcclOpExpansionMode>(opExpansionModeCcuMs);
    }

    // A5仅通过HcclConfigGetInfo获取展开模式，其他型号保留环境变量方式
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    #ifdef MACRO_DEV_TYPE_NEW
    if (deviceType != DevType::DEV_TYPE_950 || !useConfigOpExpansionMode) {
    #else
    if (deviceType != DevType::DEV_TYPE_910_95 || !useConfigOpExpansionMode) {
    #endif
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
            HCCL_DEBUG("[DecideHcclOpExpansionMode] configOpExpansionMode: %d, environment mode: %d, conflict, use environment mode.",
                configOpExpansionMode, finalMode);
        }
    }
    HCCL_INFO("[DecideHcclOpExpansionMode] finalMode: %d.", finalMode);

    return HCCL_SUCCESS;
}

HcclResult ApplyOpExpansionMode(OpParam &param, HcclOpExpansionMode finalMode)
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

bool HcclCheckAicpuEnableOpen()
{
    const char* envValue = std::getenv("HCCL_ENABLE_OPEN_AICPU");

    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return false;
    }

    return true;
}

HcclResult HcclRegstryBuff(HcclComm comm, const char *memTag, void *bufferPtr, uint64_t bufferSize, HcclMemHandle *memHandle)
{
    CHK_PTR_NULL(memHandle);
    CommMem regMem{COMM_MEM_TYPE_DEVICE, bufferPtr, bufferSize};
    CHK_RET(HcclCommMemReg(comm, memTag, &regMem, memHandle));
    HCCL_INFO("[%s] regMemAddr[%p] regMemSize[%llu]", __func__, regMem.addr, regMem.size);
    CHK_PTR_NULL(*memHandle);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRemoteBuff(HcclComm comm, ChannelHandle channel, const char *memTag, void **bufferPtr, uint64_t *bufferSize)
{
    CHK_PTR_NULL(bufferPtr);
    CHK_PTR_NULL(bufferSize);

    u32 memNum;
    CommMem *remoteMemList;
    char **memTags;
    CHK_RET(HcclChannelGetRemoteMems(comm, channel, &memNum, &remoteMemList, &memTags));
    HCCL_INFO("[%s] HcclChannelGetRemoteMems memNum[%u]", __func__, memNum);
    for (u32 i=0; i< memNum; i++) {
        HCCL_INFO("[%s] memNum[%u/%u] memTags[%s]", __func__, i + 1, memNum, memTags[i]);
        if (strcmp(memTags[i], memTag) == 0) {
            *bufferPtr = remoteMemList[i].addr;
            *bufferSize = remoteMemList[i].size;
            HCCL_INFO("[%s] Found %u memNum[%u/%u] is %u at index %u: addr=%p, size=%llu", __func__, *memTag,
                i + 1, memNum, remoteMemList[i].addr, remoteMemList[i].size);
            break;
        }
    }
    if (*bufferPtr == nullptr) {
        HCCL_WARNING("[%s] Failed to find %s in remote mem list", __func__, memTag);
    }
    return HCCL_SUCCESS;
}

bool HcclCheckCcuEnableOpen()
{
    return true;
}

bool HcclCheckAivEnableOpen()
{
    const char* envValue = std::getenv("HCCL_ENABLE_OPEN_AIV");

    if (envValue != nullptr && std::strcmp(envValue, "1") == 0) {
        return false;
    }

    return true;
}

bool ShouldUseInnerOp(OpExecuteConfig opExecuteConfig)
{
    bool isAicpuOrHostMode = (opExecuteConfig == OpExecuteConfig::AICPU_TS ||
                              opExecuteConfig == OpExecuteConfig::HOSTCPU);
    bool isCcuMode = (opExecuteConfig == OpExecuteConfig::CCU_MS ||
                      opExecuteConfig == OpExecuteConfig::CCU_SCHED);
    bool isAivMode = (opExecuteConfig == OpExecuteConfig::AIV ||
                      opExecuteConfig == OpExecuteConfig::AIV_ONLY);

    if (isAicpuOrHostMode) {
        return !HcclCheckAicpuEnableOpen();
    } else if (isCcuMode) {
        return !HcclCheckCcuEnableOpen();
    } else if (isAivMode) {
        return !HcclCheckAivEnableOpen();
    }

    return false;
}

HcclResult LogHcclExit(const std::string &opName, const char *tag, HcclUs startut, bool forceLog)
{
    if (forceLog || GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        std::string endInfo = opName + ":success,take time: " +
            std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + tag;
        HCCL_RUN_INFO("%s", endInfo.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult GetAivParamStorageByComm(HcclComm comm, AivParamStorage **aivParam, bool ifCreate)
{
    if (comm == nullptr || aivParam == nullptr) {
        HCCL_ERROR("[GetAivParamStorageByComm] Invalid parameters");
        return HCCL_E_PARA;
    }

    void *aivParamCtx = nullptr;
    uint64_t size = sizeof(AivParamStorage);

    const char *aivParamTag = "AivParamStorage";
    if (HcclEngineCtxGet(comm, aivParamTag, CommEngine::COMM_ENGINE_CPU_TS, &aivParamCtx, &size) != HCCL_SUCCESS) {
        if (ifCreate) {
            CHK_RET(HcclEngineCtxCreate(comm, aivParamTag, CommEngine::COMM_ENGINE_CPU_TS, size, &aivParamCtx));
        } else {
            HCCL_WARNING("[GetAivParamStorageByComm] Call HcclEngineCtxGet failed.");
            return HCCL_E_PARA;
        }
    }

    *aivParam = static_cast<AivParamStorage *>(aivParamCtx);

    return HCCL_SUCCESS;
}

HcclResult GetAivParamStorage(const char *group, AivParamStorage **aivParam)
{
    if (group == nullptr || aivParam == nullptr) {
        HCCL_ERROR("[GetAivParamStorage] Invalid parameters");
        return HCCL_E_PARA;
    }

    HcclComm comm = nullptr;
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));

    return GetAivParamStorageByComm(comm, aivParam, true);
}

HcclResult SetMultipleDimensionSplitRatio(OpParam &param) {
    double ratioValue = 0;
    const double DEFAULT_MULT_RATIO = 0.5;
    if (!GetExternalInputMultipleDimensionSplitRatio(ratioValue)) {
        param.opConfig.multipleDimensionSplitRatio = DEFAULT_MULT_RATIO;
        HCCL_INFO("[OpCommon] Ratio is not set, use default value: %f", DEFAULT_MULT_RATIO);
    } else {
        // 验证转换后的值是否合理
        if (ratioValue < 0 || ratioValue > 1) {
            HCCL_WARNING("[OpCommon] Ratio value %.2f out of range, use default value: %f", 
                        ratioValue, DEFAULT_MULT_RATIO);
            param.opConfig.multipleDimensionSplitRatio = DEFAULT_MULT_RATIO;
        } else {
            param.opConfig.multipleDimensionSplitRatio = ratioValue;
            HCCL_INFO("[OpCommon] Set ratio to: %f", param.opConfig.multipleDimensionSplitRatio);
        }
    }
    return HCCL_SUCCESS;
}

// 判断通过最高一个level的网络全部没有device的可达链路，并且有host的可达链路
HcclResult CheckHostDPUOnly(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool &hostDPUOnly)
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

    uint32_t *netLayers = nullptr;
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
        uint32_t *topoInsts = nullptr;
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
            uint32_t *ranks = nullptr;
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
                    HCCL_INFO("Not using hostdpu because there is links on device in netLayer[%u] in endPointIdx[%u]",
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
HcclResult SetExecTimeout(OpParam &param) {
    double execTimeoutValue = 0;
    if (!GetExternalInputExecTimeout(execTimeoutValue)) {
        param.opConfig.execTimeout = CUSTOM_TIMEOUT;
        HCCL_INFO("[OpCommon] Exec timeout is not set, use default value: %u seconds", CUSTOM_TIMEOUT);
    } else {
        // 验证转换后的值是否合理
        if (execTimeoutValue < 0 || execTimeoutValue > UINT32_MAX) {
            HCCL_WARNING("[OpCommon] Exec timeout value %.2f out of range, use default: %u seconds", 
                         execTimeoutValue, CUSTOM_TIMEOUT);
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

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    ret = hrtGetDeviceType(deviceType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[IsHostDpu]hrtGetDeviceType fail, ret:%d", ret);
        return false;
    }
    if (deviceType != DevType::DEV_TYPE_910B) {
        return false;
    }

    uint32_t *level0SizeList = nullptr;
    uint32_t level0RankListNum = 0;
    ret = HcclRankGraphGetInstSizeListByLayer(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0),
        &level0SizeList, &level0RankListNum);
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
    uint32_t *netLayers = nullptr;
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

    uint32_t *level0SizeList = nullptr;
    uint32_t level0RankListNum = 0;
    ret = HcclRankGraphGetInstSizeListByLayer(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0),
        &level0SizeList, &level0RankListNum);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    u32 rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        return false;
    }

    uint32_t *netLayers = nullptr;
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
}  // namespace ops_hccl
