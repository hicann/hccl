/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_resource_graph_mode.h"
#include "hccl/hcom.h"
#include <cstddef>
#include <cstring>
#include "hcom.h"
#include "op_common.h"
#include "alg_env_config.h"
#include "adapter_acl.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "hccl_aiv_utils.h"
#include "aiv_kernel_def.h"
#include "hccl_res_dl.h"

namespace {
struct AclHostMemGuard {
    void* ptr = nullptr;

    AclHostMemGuard() = default;
    AclHostMemGuard(const AclHostMemGuard&) = delete;
    AclHostMemGuard& operator=(const AclHostMemGuard&) = delete;
    AclHostMemGuard(AclHostMemGuard&&) = delete;
    AclHostMemGuard& operator=(AclHostMemGuard&&) = delete;

    ~AclHostMemGuard()
    {
        if (ptr == nullptr) {
            return;
        }
        aclError ret = aclrtFreeHost(ptr);
        if (ret != ACL_SUCCESS) {
            HCCL_WARNING("[AclHostMemGuard] aclrtFreeHost failed, ret[%d]", ret);
        }
    }
};
} // namespace

HcclResult HcclCreateOpParamGraphMode(OpParamGraphMode** opParam)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void**转换为OpParamGraphMode**
    OpParamGraphMode** paramPtr = reinterpret_cast<OpParamGraphMode**>(opParam);
    *paramPtr = new OpParamGraphMode();
    if (*paramPtr == nullptr) {
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclDestroyOpParamGraphMode(OpParamGraphMode* opParam)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode* paramPtr = reinterpret_cast<OpParamGraphMode*>(opParam);
    delete paramPtr;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeOpType(OpParamGraphMode* opParam, const char* opType)
{
    if (opParam == nullptr || opType == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode* paramPtr = reinterpret_cast<OpParamGraphMode*>(opParam);
    CHK_SAFETY_FUNC_RET(strncpy_s(paramPtr->opType, sizeof(paramPtr->opType), opType, sizeof(paramPtr->opType) - 1));
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeDataCount(OpParamGraphMode* opParam, const u64* dataCount)
{
    if (opParam == nullptr || dataCount == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode* paramPtr = reinterpret_cast<OpParamGraphMode*>(opParam);
    CHK_SAFETY_FUNC_RET(memcpy_s(&paramPtr->dataCount, sizeof(paramPtr->dataCount), dataCount, sizeof(u64)));
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeDataType(OpParamGraphMode* opParam, const HcclDataType dataType)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode* paramPtr = reinterpret_cast<OpParamGraphMode*>(opParam);
    paramPtr->dataType = dataType;
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeRankSize(OpParamGraphMode* opParam, const u32* rankSize)
{
    if (opParam == nullptr || rankSize == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode* paramPtr = reinterpret_cast<OpParamGraphMode*>(opParam);
    CHK_SAFETY_FUNC_RET(memcpy_s(&paramPtr->rankSize, sizeof(paramPtr->rankSize), rankSize, sizeof(u32)));
    return HCCL_SUCCESS;
}

HcclResult HcclSetOpParamGraphModeHCCLBufferSize(OpParamGraphMode* opParam, const u64* hcclBufferSize)
{
    if (opParam == nullptr || hcclBufferSize == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode* paramPtr = reinterpret_cast<OpParamGraphMode*>(opParam);
    CHK_SAFETY_FUNC_RET(
        memcpy_s(&paramPtr->hcclBufferSize, sizeof(paramPtr->hcclBufferSize), hcclBufferSize, sizeof(u64)));
    return HCCL_SUCCESS;
}
HcclResult HcclSetAivSelectOpParamGraphMode(OpParamGraphMode* opParam, u32 aivCoreLimit)
{
    if (opParam == nullptr) {
        return HCCL_E_PARA;
    }
    // 将void*转换为OpParamGraphMode*
    OpParamGraphMode* paramPtr = reinterpret_cast<OpParamGraphMode*>(opParam);
    paramPtr->aivCoreLimit = aivCoreLimit;
    return HCCL_SUCCESS;
}

HcclResult
HcclCalcOpResOnlineGraphMode(OpParamGraphMode* opParam, u64* opMemSize, u32* streamNum, u32* taskNum, u32* aivCoreNum)
{
    HCCL_INFO("Enter HcclCalcOpResOnlineGraphMode.");
    CHK_RET(CheckCalcResInputGraphMode(opParam, opMemSize, streamNum, taskNum, aivCoreNum));
    // 将void**转换为OpParamGraphMode**
    OpParamGraphMode* paramPtr = reinterpret_cast<OpParamGraphMode*>(opParam);
    if (paramPtr == nullptr) {
        return HCCL_E_PARA;
    }
    // 为了兼容，创建临时的 ResResponseGraphMode 结构
    ResResponseGraphMode resResponse = {0, 0, 0, 0};
    HCCL_INFO("Start to calc op resource online.");
    // aicpu引擎计算资源
    ops_hccl::HcclCalcAicpuResOffline(&resResponse);

    // ccu引擎计算资源
    ops_hccl::HcclCalcCcuResOffline(opParam, &resResponse);

    // aiv引擎计算资源
    ops_hccl::HcclCalcAivResOffline(&resResponse, paramPtr);

    // 将结果复制到输出参数
    *opMemSize = resResponse.opMemSize;
    *streamNum = resResponse.streamNum;
    *taskNum = resResponse.taskNum;
    *aivCoreNum = resResponse.aivCoreNum;

    return HCCL_SUCCESS;
}

HcclResult
HcclCalcOpResOfflineGraphMode(OpParamGraphMode* opParam, u64* opMemSize, u32* streamNum, u32* taskNum, u32* aivCoreNum)
{
    HCCL_INFO("Enter HcclCalcOpResOfflineGraphMode.");
    CHK_RET(CheckCalcResInputGraphMode(opParam, opMemSize, streamNum, taskNum, aivCoreNum));
    // 将void**转换为OpParamGraphMode**
    OpParamGraphMode* paramPtr = reinterpret_cast<OpParamGraphMode*>(opParam);
    if (paramPtr == nullptr) {
        return HCCL_E_PARA;
    }
    // 为了兼容，创建临时的 ResResponseGraphMode 结构
    ResResponseGraphMode resResponse = {0, 0, 0, 0};
    HCCL_INFO("Start to calc op resource offline.");
    // aicpu引擎计算资源
    ops_hccl::HcclCalcAicpuResOffline(&resResponse);

    // ccu引擎计算资源
    ops_hccl::HcclCalcCcuResOffline(opParam, &resResponse);

    // 其他引擎补充在下面
    // aiv引擎计算资源
    ops_hccl::HcclCalcAivResOffline(&resResponse, paramPtr);

    // 将结果复制到输出参数
    *opMemSize = resResponse.opMemSize;
    *streamNum = resResponse.streamNum;
    *taskNum = resResponse.taskNum;
    *aivCoreNum = resResponse.aivCoreNum;

    return HCCL_SUCCESS;
}

HcclResult HcclSetAivCoreLimitGraphMode(const char* group, u32 aivCoreLimit)
{
    if (group == nullptr) {
        HCCL_ERROR("[HcclSetAivCoreLimitGraphMode] group is nullptr");
        return HCCL_E_PARA;
    }

    ops_hccl::AivParamStorage* aivParam = nullptr;
    CHK_RET(ops_hccl::GetAivParamStorage(group, &aivParam));

    aivParam->aivCoreLimit = aivCoreLimit;

    HCCL_INFO("[HcclSetAivCoreLimitGraphMode] Set aivCoreLimit[%u] for group[%s]", aivCoreLimit, group);

    return HCCL_SUCCESS;
}

HcclResult HcclSelectAlgGraphMode(
    const char* group, u64 count, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType, u32 aivCoreLimit,
    bool* ifAiv, char* algName)
{
    if (group == nullptr || ifAiv == nullptr || algName == nullptr) {
        HCCL_ERROR("[HcclSelectAlgGraphMode] Invalid parameters");
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "[HcclSelectAlgGraphMode] Start: group[%s] count[%llu] dataType[%u] reduceOp[%u] opType[%u] aivCoreLimit[%u]",
        group, count, dataType, op, opType, aivCoreLimit);

    if (g_aivKernelInfoMap.find(opType) == g_aivKernelInfoMap.end()) {
        HCCL_INFO("[HcclSelectAlgGraphMode] Unsupported aiv op.");
        return HCCL_SUCCESS;
    }

    s32 deviceId = 0;
    CHK_PRT_RET(
        aclrtGetDevice(&deviceId) != ACL_SUCCESS, HCCL_WARNING("[HcclSelectAlgGraphMode] device is not set."),
        HCCL_SUCCESS);

    HcclComm hcclComm = nullptr;
    CHK_RET(HcomGetCommHandleByGroup(group, &hcclComm));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(hcclComm, &rankSize));

    CHK_RET(InitEnvConfig());

    ops_hccl::OpParam param;
    CHK_RET(HcclGetCommName(hcclComm, param.commName));

    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    CHK_RET(HcclGetDeviceType(deviceType));

    param.opType = opType;
    param.DataDes.count = count;
    param.DataDes.dataType = dataType;
    param.reduceType = op;
    param.opMode = ops_hccl::OpMode::OFFLOAD;
    param.numBlocksLimit = aivCoreLimit;
    param.enableDetour = false;
    param.deviceType = deviceType;

    AclHostMemGuard sendCountsHost;
    AclHostMemGuard recvCountsHost;
    AclHostMemGuard sdisplsHost;
    AclHostMemGuard rdisplsHost;
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        param.varMemSize = ops_hccl::ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64);
        param.all2AllVDataDes.sendType = dataType;
        param.all2AllVDataDes.recvType = dataType;

        u64 arrSize = rankSize * sizeof(u64);
        ACLCHECK(aclrtMallocHost(&sendCountsHost.ptr, arrSize));
        ACLCHECK(aclrtMallocHost(&recvCountsHost.ptr, arrSize));
        ACLCHECK(aclrtMallocHost(&sdisplsHost.ptr, arrSize));
        ACLCHECK(aclrtMallocHost(&rdisplsHost.ptr, arrSize));

        u64* sendCountsPtr = static_cast<u64*>(sendCountsHost.ptr);
        u64* recvCountsPtr = static_cast<u64*>(recvCountsHost.ptr);
        u64* sdisplsPtr = static_cast<u64*>(sdisplsHost.ptr);
        u64* rdisplsPtr = static_cast<u64*>(rdisplsHost.ptr);

        u64 dataCountOffset = 0;
        for (u32 i = 0; i < rankSize; i++) {
            sendCountsPtr[i] = count;
            recvCountsPtr[i] = count;
            sdisplsPtr[i] = dataCountOffset;
            rdisplsPtr[i] = dataCountOffset;
            dataCountOffset += count;
        }

        param.all2AllVDataDes.sendCounts = sendCountsHost.ptr;
        param.all2AllVDataDes.recvCounts = recvCountsHost.ptr;
        param.all2AllVDataDes.sdispls = sdisplsHost.ptr;
        param.all2AllVDataDes.rdispls = rdisplsHost.ptr;
    }

    int ret = sprintf_s(param.tag, sizeof(param.tag), "SelectAlg_%d_%s", static_cast<int>(opType), param.commName);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[HcclSelectAlgGraphMode] failed to fill param.tag"), HCCL_E_INTERNAL);

    CHK_RET(ops_hccl::HcclGetOpExpansionMode(hcclComm, param));

    std::unique_ptr<ops_hccl::TopoInfoWithNetLayerDetails> topoInfo
        = std::make_unique<ops_hccl::TopoInfoWithNetLayerDetails>();
    std::string localAlgName;
    CHK_RET(ops_hccl::Selector(hcclComm, param, topoInfo, localAlgName));

    *ifAiv = (param.engine == CommEngine::COMM_ENGINE_AIV);

    // 拷贝字符串
    CHK_SAFETY_FUNC_RET(strncpy_s(algName, ALG_NAME_MAX_LEN, localAlgName.c_str(), ALG_NAME_MAX_LEN - 1));

    HCCL_INFO("[HcclSelectAlgGraphMode] Success. ifAiv=%d, algName=%s", *ifAiv, algName);
    return HCCL_SUCCESS;
}

namespace {
// 图模式下算法选择+录制模式的公共流程，返回录制的aivOpArgs与cclBufferSize
HcclResult RecordAivOpArgsGraphMode(
    const char* group, u64 count, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType, u32 aivCoreLimit,
    void* inputPtr, void* outputPtr, const char* tag, ops_hccl::AivOpArgs& aivOpArgs, u64& cclBufferSize)
{
    HcclComm comm = nullptr;
    CHK_RET(HcomGetCommHandleByGroup(group, &comm));
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET(HcclGetRankSize(comm, &rankSize));

    ops_hccl::OpParam param;
    param.hcclComm = comm;
    param.opType = opType;
    param.inputPtr = inputPtr;
    param.outputPtr = outputPtr;
    param.DataDes.count = count;
    param.DataDes.dataType = dataType;
    param.reduceType = op;
    param.opMode = ops_hccl::OpMode::OFFLOAD;
    param.numBlocksLimit = aivCoreLimit;

    AclHostMemGuard sendCountsHost;
    AclHostMemGuard recvCountsHost;
    AclHostMemGuard sdisplsHost;
    AclHostMemGuard rdisplsHost;
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        param.varMemSize = ops_hccl::ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64);
        param.all2AllVDataDes.sendType = dataType;
        param.all2AllVDataDes.recvType = dataType;

        u64 arrSize = rankSize * sizeof(u64);
        ACLCHECK(aclrtMallocHost(&sendCountsHost.ptr, arrSize));
        ACLCHECK(aclrtMallocHost(&recvCountsHost.ptr, arrSize));
        ACLCHECK(aclrtMallocHost(&sdisplsHost.ptr, arrSize));
        ACLCHECK(aclrtMallocHost(&rdisplsHost.ptr, arrSize));

        u64* recvCountsPtr = static_cast<u64*>(recvCountsHost.ptr);
        u64* sendCountsPtr = static_cast<u64*>(sendCountsHost.ptr);
        u64* sdisplsPtr = static_cast<u64*>(sdisplsHost.ptr);
        u64* rdisplsPtr = static_cast<u64*>(rdisplsHost.ptr);

        u64 dataCountOffset = 0;
        for (u32 i = 0; i < rankSize; i++) {
            sendCountsPtr[i] = count;
            recvCountsPtr[i] = count;
            sdisplsPtr[i] = dataCountOffset;
            rdisplsPtr[i] = dataCountOffset;
            dataCountOffset += count;
        }

        param.all2AllVDataDes.recvCounts = recvCountsHost.ptr;
        param.all2AllVDataDes.sendCounts = sendCountsHost.ptr;
        param.all2AllVDataDes.sdispls = sdisplsHost.ptr;
        param.all2AllVDataDes.rdispls = rdisplsHost.ptr;
    }

    CHK_RET(InitEnvConfig());
    HcclDevType deviceType = HcclDevType::DEV_TYPE_COUNT;
    CHK_RET(HcclGetDeviceType(deviceType));
    param.deviceType = deviceType;

    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", tag);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[RecordAivOpArgsGraphMode] failed to fill param.tag"), HCCL_E_INTERNAL);
    CHK_RET(HcclGetCommName(comm, param.commName));
    ret = sprintf_s(param.commModeTag, sizeof(param.commModeTag), "%s_offload", param.commName);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[RecordAivOpArgsGraphMode] failed to fill param.commModeTag"), HCCL_E_INTERNAL);

    CHK_RET(ops_hccl::HcclGetOpExpansionMode(comm, param));

    // 算法选择
    std::unique_ptr<ops_hccl::TopoInfoWithNetLayerDetails> topoInfo
        = std::make_unique<ops_hccl::TopoInfoWithNetLayerDetails>();
    std::string algName;
    CHK_RET(ops_hccl::Selector(comm, param, topoInfo, algName));

    std::unique_ptr<ops_hccl::InsCollAlgBase> executor
        = ops_hccl::CollAlgExecRegistryV2::Instance().GetAlgExec(param.opType, algName);
    CHK_PRT_RET(
        executor.get() == nullptr,
        HCCL_ERROR("[RecordAivOpArgsGraphMode] Failed to find executor for algName[%s]", algName.c_str()), HCCL_E_PARA);

    // 启用Only录制模式
    ops_hccl::g_recordingQueue = std::make_shared<ops_hccl::InsQueue>();
    ops_hccl::g_baseInputAddr = reinterpret_cast<u64>(inputPtr);
    ops_hccl::g_baseOutputAddr = reinterpret_cast<u64>(outputPtr);
    ops_hccl::g_recordOnlyMode = true;

    // 计算AlgHierarchyInfo
    ops_hccl::AlgHierarchyInfoForAllLevel algHierarchyInfo;
    CHK_RET(executor->CalcAlgHierarchyInfo(comm, topoInfo.get(), algHierarchyInfo));
    // 资源计算
    ops_hccl::AlgResourceRequest resRequest;
    CHK_RET(executor->CalcRes(comm, param, topoInfo.get(), algHierarchyInfo, resRequest));
    // host侧资源
    void* resCtxSequence = nullptr;
    CHK_RET(ops_hccl::GetAlgResAiv(comm, param, resRequest, topoInfo.get(), algHierarchyInfo, &resCtxSequence));
    // 编排
    ops_hccl::AlgResourceCtxSerializable* resCtxHost
        = static_cast<ops_hccl::AlgResourceCtxSerializable*>(resCtxSequence);
    CHK_RET(executor->Orchestrate(param, *resCtxHost));

    // 从录制的指令队列中获取aivOpArgs
    if (ops_hccl::g_recordingQueue && !ops_hccl::g_recordingQueue->empty()) {
        aivOpArgs = (*ops_hccl::g_recordingQueue)[0].opArgs;
    }
    cclBufferSize = resCtxHost->cclMem.size;

    // 清除录制
    ops_hccl::g_recordingQueue = nullptr;
    ops_hccl::g_baseInputAddr = 0;
    ops_hccl::g_baseOutputAddr = 0;
    ops_hccl::g_recordOnlyMode = false;

    // 销毁录制期间创建的algTag context，避免相同tag重复注册导致already exist错误
    HcclEngineCtxDestroy(comm, param.algTag, CommEngine::COMM_ENGINE_CPU_TS);

    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclCalcAivCoreNumGraphMode(
    const char* group, u64 count, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType, u32 aivCoreLimit,
    u32* numBlocks)
{
    HCCL_INFO(
        "[HcclCalcAivCoreNumGraphMode] group[%s] count[%llu] dataType[%u] reduceOp[%u] opType[%u] aivCoreLimit[%u]",
        group != nullptr ? group : "nullptr", count, static_cast<int>(dataType), static_cast<int>(op),
        static_cast<int>(opType), aivCoreLimit);

    if (numBlocks == nullptr) {
        HCCL_ERROR("[HcclCalcAivCoreNumGraphMode] Invalid parameter: numBlocks is null.");
        return HCCL_E_PARA;
    }
    *numBlocks = 0;

    char tag[TAG_LENGTH];
    int ret = sprintf_s(tag, sizeof(tag), "CalcAivCoreNum_%d", static_cast<int>(opType));
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[HcclCalcAivCoreNumGraphMode] failed to fill tag"), HCCL_E_INTERNAL);

    ops_hccl::AivOpArgs aivOpArgs;
    u64 cclBufferSize = 0;
    CHK_RET(RecordAivOpArgsGraphMode(
        group, count, dataType, op, opType, aivCoreLimit, nullptr, nullptr, tag, aivOpArgs, cclBufferSize));

    *numBlocks = aivOpArgs.numBlocks;
    HCCL_INFO("[HcclCalcAivCoreNumGraphMode] Success. numBlocks=%u", *numBlocks);
    return HCCL_SUCCESS;
}

HcclResult HcclGetAlgExecParamGraphMode(
    const char* tag, const char* group, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType,
    bool clearEnable, HcclDataType dataType, HcclReduceOp op, void** commContext, u64* len, u32 aivCoreLimit)
{
    HCCL_INFO(
        "[HcclGetAlgExecParamGraphMode] tag[%s], group[%s], count[%llu], opType[%d], dataType[%d], "
        "reduceOp[%d], clearEnable[%d], aivCoreLimit[%u]",
        tag != nullptr ? tag : "nullptr", group != nullptr ? group : "nullptr", count, static_cast<int>(opType),
        static_cast<int>(dataType), static_cast<int>(op), clearEnable, aivCoreLimit);

    CHK_PTR_NULL(tag);
    CHK_PTR_NULL(group);
    CHK_PTR_NULL(commContext);
    CHK_PTR_NULL(len);
    *commContext = nullptr;
    *len = 0;

    ops_hccl::AivOpArgs aivOpArgs;
    u64 cclBufferSize = 0;
    CHK_RET(RecordAivOpArgsGraphMode(
        group, count, dataType, op, opType, aivCoreLimit, inputPtr, outputPtr, tag, aivOpArgs, cclBufferSize));

    ops_hccl::AivSuperKernelArgs superKernelArgs;
    superKernelArgs.buffersIn = aivOpArgs.buffersIn;
    superKernelArgs.rank = aivOpArgs.rank;
    superKernelArgs.rankSize = aivOpArgs.rankSize;
    superKernelArgs.len = count;
    superKernelArgs.dataType = dataType;
    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[HcclGetAlgExecParamGraphMode] dataType[%u] is out of range[0, %d)", static_cast<u32>(dataType),
            static_cast<int>(HCCL_DATA_TYPE_RESERVED));
        return HCCL_E_PARA;
    }
    superKernelArgs.unitSize = ops_hccl::DATATYPE_SIZE_TABLE[dataType];
    superKernelArgs.reduceOp = op;
    superKernelArgs.numBlocks = aivOpArgs.numBlocks;
    superKernelArgs.tag = 0;
    superKernelArgs.clearEnable = clearEnable;
    superKernelArgs.inputSliceStride = 0;
    superKernelArgs.outputSliceStride = 0;
    superKernelArgs.repeatNum = 1;
    superKernelArgs.inputRepeatStride = 0;
    superKernelArgs.outputRepeatStride = 0;
    superKernelArgs.input = aivOpArgs.input;
    superKernelArgs.output = aivOpArgs.output;
    superKernelArgs.cclBufferSize = cclBufferSize;

    HCCL_INFO(
        "[HcclGetAlgExecParamGraphMode] superKernelArgs: buffersIn[%p], rank[%u], rankSize[%u], "
        "len[%llu], dataType[%u], unitSize[%u], reduceOp[%u], numBlocks[%u], tag[%d], "
        "clearEnable[%d], inputSliceStride[%llu], outputSliceStride[%llu], repeatNum[%llu], "
        "inputRepeatStride[%llu], outputRepeatStride[%llu], input[%p], output[%p], cclBufferSize[%llu]",
        superKernelArgs.buffersIn, superKernelArgs.rank, superKernelArgs.rankSize, superKernelArgs.len,
        superKernelArgs.dataType, superKernelArgs.unitSize, superKernelArgs.reduceOp, superKernelArgs.numBlocks,
        superKernelArgs.tag, superKernelArgs.clearEnable, superKernelArgs.inputSliceStride,
        superKernelArgs.outputSliceStride, superKernelArgs.repeatNum, superKernelArgs.inputRepeatStride,
        superKernelArgs.outputRepeatStride, superKernelArgs.input, superKernelArgs.output,
        superKernelArgs.cclBufferSize);

    // 分配设备内存
    void* deviceMem = nullptr;
    aclError aclRet = aclrtMalloc(&deviceMem, sizeof(ops_hccl::AivSuperKernelArgs), ACL_MEM_MALLOC_HUGE_FIRST);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[HcclGetAlgExecParamGraphMode] aclrtMalloc failed, ret[%d]", aclRet),
        HCCL_E_RUNTIME);
    // 拷贝到设备内存
    aclRet = aclrtMemcpy(
        deviceMem, sizeof(ops_hccl::AivSuperKernelArgs), &superKernelArgs, sizeof(ops_hccl::AivSuperKernelArgs),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[HcclGetAlgExecParamGraphMode] aclrtMemcpy failed, ret[%d]", aclRet);
        aclError freeRet = aclrtFree(deviceMem);
        if (freeRet != ACL_SUCCESS) {
            HCCL_ERROR("[HcclGetAlgExecParamGraphMode] aclrtFree failed, ret[%d]", freeRet);
        }
        return HCCL_E_RUNTIME;
    }

    *commContext = deviceMem;
    *len = sizeof(ops_hccl::AivSuperKernelArgs);

    HCCL_INFO("[HcclGetAlgExecParamGraphMode] success, commContext[%p], len[%llu]", *commContext, *len);
    return HCCL_SUCCESS;
}

namespace ops_hccl {
HcclResult HcclCalcAicpuResOffline(ResResponseGraphMode* resResponse)
{
    if (resResponse == nullptr) {
        return HCCL_E_PARA;
    }
    u64 aicpuOpMemSize = 0;
    u32 aicpuStreamNum = 0;
    u32 aicpuTaskNum = 3;

    resResponse->opMemSize = std::max(resResponse->opMemSize, aicpuOpMemSize);
    resResponse->streamNum = std::max(resResponse->streamNum, aicpuStreamNum);
    resResponse->taskNum = std::max(resResponse->taskNum, aicpuTaskNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCalcAivResOffline(ResResponseGraphMode* resResponse, OpParamGraphMode* paramPtr)
{
    if (resResponse == nullptr || paramPtr == nullptr || paramPtr->aivCoreLimit == 0) {
        return HCCL_E_PARA;
    }
    constexpr u64 AIV_WORKSPACE_MEM_SIZE = 512;
    constexpr u32 AIV_STREAM_NUM = 0;
    constexpr u32 AIV_TASK_NUM = 3;

    resResponse->opMemSize = std::max(resResponse->opMemSize, AIV_WORKSPACE_MEM_SIZE);
    resResponse->streamNum = std::max(resResponse->streamNum, AIV_STREAM_NUM);
    resResponse->taskNum = std::max(resResponse->taskNum, AIV_TASK_NUM);
    resResponse->aivCoreNum = paramPtr->aivCoreLimit;
    return HCCL_SUCCESS;
}

HcclResult CheckCalcResInputGraphMode(
    const OpParamGraphMode* opParam, const u64* opMemSize, const u32* streamNum, const u32* taskNum,
    const u32* aivCoreNum)
{
    CHK_PTR_NULL(opParam);
    CHK_PTR_NULL(opMemSize);
    CHK_PTR_NULL(streamNum);
    CHK_PTR_NULL(taskNum);
    CHK_PTR_NULL(aivCoreNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCalcCcuResOffline(OpParamGraphMode* opParam, ResResponseGraphMode* resResponse)
{
    HCCL_INFO("Entry HcclCalcCcuResOffline.");
    if (resResponse == nullptr || opParam == nullptr) {
        return HCCL_E_PARA;
    }

    // ccu的资源申请
    u64 ccuOpMemSize = 0;
    u32 ccuStreamNum = 6;
    u32 ccuTaskNum = 0;

    CHK_PRT(CalcTaskNum(opParam, ccuTaskNum));

    resResponse->opMemSize = std::max(resResponse->opMemSize, ccuOpMemSize);
    resResponse->streamNum = std::max(resResponse->streamNum, ccuStreamNum);
    resResponse->taskNum = std::max(resResponse->taskNum, ccuTaskNum);
    HCCL_INFO(
        "[HcclCalcCcuResOffline] opMemSize[%llu], streamNum[%llu], taskNum[%llu]", resResponse->opMemSize,
        resResponse->streamNum, resResponse->taskNum);
    return HCCL_SUCCESS;
}

HcclResult CalcTaskNum(OpParamGraphMode* opParam, u32& ccuTaskNum)
{
    HCCL_INFO("[CalcTaskNum] begin");
    if (opParam->hcclBufferSize == 0 || opParam->rankSize == 0) {
        ccuTaskNum = GE_PARALLEL;
        return HCCL_SUCCESS;
    }

    u64 dataCount = opParam->dataCount;
    u64 rankSize = opParam->rankSize;
    u64 scratchBufferSize = opParam->hcclBufferSize;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 dataType = opParam->dataType;
    if (dataType >= HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[CalcTaskNum] dataType[%llu] is out of range[0, %d)", dataType, static_cast<int>(HCCL_DATA_TYPE_RESERVED));
        return HCCL_E_PARA;
    }
    u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    u64 maxDataSizePerLoop = 0;
    u64 maxDataCountPerLoop = 0;
    u64 loopTimes = 0;
    HCCL_INFO(
        "[CalcTaskNum] opType[%s] scratchBufferSize[%llu] dataCount[%llu] rankSize[%llu]", opParam->opType,
        scratchBufferSize, dataCount, rankSize);
    if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLTOALL) {
        maxDataSizePerLoop = transportBoundDataSize;
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize / rankSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLTOALLV || opParam->opType == HCCL_KERNEL_OP_TYPE_ALLTOALLVC) {
        ccuTaskNum = 1;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_REDUCE) {
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_BROADCAST) {
        maxDataSizePerLoop = transportBoundDataSize;
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
        maxDataSizePerLoop = transportBoundDataSize;
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER) {
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLREDUCE) {
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        u64 scratchBoundDataSize = scratchBufferSize / rankSize / 128 * 128;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_SCATTER) {
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_ALLGATHERV) {
        maxDataSizePerLoop = transportBoundDataSize;
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    } else if (opParam->opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTERV) {
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBufferSize);
        maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize;
        loopTimes = dataCount / maxDataCountPerLoop + static_cast<u64>(dataCount % maxDataCountPerLoop != 0);
        ccuTaskNum = loopTimes * GE_PARALLEL;
    }
    HCCL_INFO(
        "[CalcTaskNum] maxDataSizePerLoop[%llu] maxDataCountPerLoop[%llu] loopTimes[%llu] ccuTaskNum[%llu]",
        maxDataSizePerLoop, maxDataCountPerLoop, loopTimes, ccuTaskNum);
    HCCL_INFO("[CalcTaskNum] end.");
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
