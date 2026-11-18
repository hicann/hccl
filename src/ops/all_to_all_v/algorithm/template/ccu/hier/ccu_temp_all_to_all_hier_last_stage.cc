/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_alg_template_base.h"
#include "ccu_launch_dl.h"
#include "ccu_alltoall_stage_template_registry.h"
#include "ccu_temp_all_to_all_hier_last_stage.h"
#include "kernel/ccu_kernel_all_to_all_hier_last_stage.h"

namespace ops_hccl {

CcuTempAlltoAllHierLastStage::CcuTempAlltoAllHierLastStage(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks, u32 stageIndex)
    : CcuTempAlltoAllHierStageBase(param, rankId, subCommRanks, stageIndex)
{}

HcclResult CcuTempAlltoAllHierLastStage::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_RET(ValidateStageTopo(topoInfo));
    srcSliceNum_ = topoInfo->userRankSize / templateRankSize_;

    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.ccuKernelNum.push_back(1);

    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllToAllHierLastStageKernel"));
    kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuAllToAllHierLastStageKernel);

    std::vector<HcclChannelDesc> channelDescs;
    // 通道契约：按 subCommRanks_ 组序申请，与 kernel 组序遍历一一对齐（每对端恰一条）
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    auto kernelArg = std::make_shared<CcuKernelArgAllToAllHierLastStage>();
    kernelArg->rankId = myRank_;
    kernelArg->myAlgRank = myAlgRank_;
    kernelArg->groupSize = templateRankSize_;
    kernelArg->srcSliceNum = srcSliceNum_;
    kernelArg->rankGroup = subCommRanks_[0];
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    resourceRequest.channels.emplace_back(channelDescs);

    HCCL_INFO(
        "[CcuTempAlltoAllHierLastStage][CalcRes] rank[%u], channels[%zu], myAlgRank[%u], groupSize[%u], "
        "srcSliceNum[%u]",
        myRank_, channelDescs.size(), myAlgRank_, templateRankSize_, srcSliceNum_);
    return HCCL_SUCCESS;
}

u64 CcuTempAlltoAllHierLastStage::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

static HcclResult ValidateAndCalcParams(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource,
    u64& totalCount, u64& dataTypeSize, u64& otherDim, u64& cclCount)
{
    CHK_PRT_RET(
        templateResource.ccuKernels.empty(),
        HCCL_ERROR("[CcuTempAlltoAllHierLastStage][%s] ccuKernels is empty.", __func__), HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        templateResource.threads.empty(), HCCL_ERROR("[CcuTempAlltoAllHierLastStage][%s] threads is empty.", __func__),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        tempAlgParams.buffInfo.hcclBuff.addr == nullptr || tempAlgParams.buffInfo.outputPtr == nullptr,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierLastStage][%s] hcclBuff[%p] or outputPtr[%p] is null", __func__,
            tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.outputPtr),
        HcclResult::HCCL_E_PTR);
    CHK_RET(CalcUniformSliceParams(param, tempAlgParams, totalCount, dataTypeSize, otherDim, cclCount));
    return HCCL_SUCCESS;
}

HcclResult CcuTempAlltoAllHierLastStage::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    buffInfo_ = tempAlgParams.buffInfo;
    u64 totalCount = 0;
    u64 dataTypeSize = 0;
    u64 otherDim = 0;
    u64 cclCount = 0;
    CHK_RET(
        ValidateAndCalcParams(param, tempAlgParams, templateResource, totalCount, dataTypeSize, otherDim, cclCount));

    uint64_t cclBuffAddr = PointerToAddr(buffInfo_.hcclBuff.addr);
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr);
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    u64 blockStride = totalCount * dataTypeSize; // OUTPUT 块步长
    u64 slotStride = cclCount * dataTypeSize;    // cclBuff 单槽步长（字节数，与首 Stage 一致）

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
    auto goSize = CalGoSize(tempAlgParams.sliceSize, config, GetCcuVersion());

    std::vector<uint64_t> taskArgs
        = {cclBuffAddr,
           outputAddr,
           token,
           tempAlgParams.sliceSize,
           buffInfo_.hcclBuffBaseOff,
           buffInfo_.outBuffBaseOff,
           blockStride,
           slotStride,
           goSize[0],
           goSize[1],
           goSize[2],
           goSize[3]};
    // otherDim 即末 Stage 语义的 srcSliceNum（=totalRankSize/stageRankSize=d1）；成员 srcSliceNum_ 执行路径实例恒
    // 0，不可直打
    HCCL_INFO(
        "[CcuTempAlltoAllHierLastStage][KernelRun] sliceSize[%llu], blockStride[%llu], slotStride[%llu], "
        "myAlgRank[%u], otherDimRankSize[%llu]",
        tempAlgParams.sliceSize, blockStride, slotStride, myAlgRank_, otherDim);
    CcuResult launchRet = HcommCcuKernelLaunch(
        templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(),
        static_cast<uint32_t>(taskArgs.size()));
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAlltoAllHierLastStage][KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CcuKernelSubmitInfo submitInfo{};
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(
        submitInfo, taskArgs[0], taskArgs[1], taskArgs[2], taskArgs[3], taskArgs[4], taskArgs[5], taskArgs[6],
        taskArgs[7], taskArgs[8], taskArgs[9], taskArgs[10], taskArgs[11]));
    templateResource.submitInfos.push_back(submitInfo);
    return HCCL_SUCCESS;
}

HcclResult
CcuTempAlltoAllHierLastStage::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    (void)param;
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.empty()) {
        HCCL_INFO("[CcuTempAlltoAllHierLastStage][FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        tempFastLaunchCtx.threads.empty(), HCCL_ERROR("[CcuTempAlltoAllHierLastStage][FastLaunch] threads is empty."),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        tempFastLaunchCtx.buffInfo.hcclBuff.addr == nullptr || tempFastLaunchCtx.buffInfo.outputPtr == nullptr,
        HCCL_ERROR(
            "[CcuTempAlltoAllHierLastStage][FastLaunch] hcclBuff[%p] or outputPtr[%p] is null",
            tempFastLaunchCtx.buffInfo.hcclBuff.addr, tempFastLaunchCtx.buffInfo.outputPtr),
        HcclResult::HCCL_E_PTR);

    constexpr u32 cclBuffAddrIdx = 0;
    constexpr u32 outputAddrIdx = 1;
    constexpr u32 taskArgSize = 12;

    uint64_t* args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    args[cclBuffAddrIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.hcclBuff.addr);
    args[outputAddrIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr);

    CcuResult launchRet = HcommCcuKernelLaunch(
        tempFastLaunchCtx.threads[0], tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
        reinterpret_cast<void*>(args), taskArgSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAlltoAllHierLastStage][FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    return HCCL_SUCCESS;
}

REGISTER_CCU_A2A_STAGE_TEMPLATE(LastMeshStage, CcuTempAlltoAllHierLastStage);

} // namespace ops_hccl
