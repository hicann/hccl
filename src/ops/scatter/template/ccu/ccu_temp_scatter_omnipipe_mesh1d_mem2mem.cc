/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "ccu_kernel_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_scatter_omnipipe_mesh1d_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempScatterOmniPipeMesh1DMem2Mem::CcuTempScatterOmniPipeMesh1DMem2Mem(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), itRoot);
    }
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    ifRealRoot_ = (rankId == param.root);
    std::string ranksStr = "";
    for (auto r : ranks) {
        ranksStr += std::to_string(r) + ", ";
    }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] subCommRanks[%s] templateRankSize[%u] subCommRootId_[%d] ifRealRoot_[%d]",
        __func__, rankId, mySubCommRank_, ranksStr.c_str(), templateRankSize_, subCommRootId_, ifRealRoot_);
}

CcuTempScatterOmniPipeMesh1DMem2Mem::~CcuTempScatterOmniPipeMesh1DMem2Mem() {}

void CcuTempScatterOmniPipeMesh1DMem2Mem::SetRoot(u32 root)
{
    HCCL_DEBUG("[CcuTempScatterOmniPipeMesh1DMem2Mem][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::string ranksStr = "";
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), itRoot);
    }
    for (auto r : ranks) {
        ranksStr += std::to_string(r) + ", ";
    }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] subCommRanks[%s] subCommRootId_[%d]", __func__, myRank_, mySubCommRank_,
        ranksStr.c_str(), subCommRootId_);
}

void CcuTempScatterOmniPipeMesh1DMem2Mem::UnsetRoot(u32 rank)
{
    HCCL_DEBUG("[CcuTempScatterOmniPipeMesh1DMem2Mem][UnsetRoot] myRank_ [%u], unset root [%u] ", myRank_, rank);
    if (!ifRealRoot_) {
        subCommRootId_ = UINT32_MAX;
    }
}

u64 CcuTempScatterOmniPipeMesh1DMem2Mem::GetThreadNum() const { return 1; }

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);
    resourceRequest.ccuKernelNum.push_back(1);

    HCCL_DEBUG(
        "[%s]notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__, resourceRequest.notifyNumOnMainThread,
        resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(strcpy_s(
        kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuScatterOmniPipeMesh1DMem2MemKernel"));
    kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuScatterOmniPipeMesh1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1DFullMesh(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(
            comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : channelDescs) {
            if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
                HCCL_ERROR(
                    "[CcuTempScatterOmniPipeMesh1DMem2Mem][%s] channel.channelProtocol[%u]", __func__,
                    channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }
    HCCL_DEBUG("[CcuTempScatterOmniPipeMesh1DMem2Mem][%s] Get Mesh channels Success.", __func__);

    auto kernelArg = std::make_shared<CcuKernelArgScatterOmniPipeMesh1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = subCommRootId_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelArg->ifRealRoot = ifRealRoot_;
    kernelArg->myrealrank = myRank_;

    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG(
        "[%s]channelDescs.size()=%llu, ccuKernelInfos.size()=%llu", __func__, channelDescs.size(),
        resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::LaunchOneRepeat(
    const StepSliceInfo& stepSliceInfo, TemplateResource& templateResource, uint32_t rpt, uint64_t repeatNum,
    bool ifNewRoot, uint64_t inputAddr, uint64_t outputAddr, uint64_t token)
{
    uint64_t sliceSize = 0;
    bool isFirstPiece = (rpt == 0);
    bool isLastPiece = (rpt == repeatNum - 1);
    std::vector<uint64_t> inputOmniSliceSizeVec = {};
    std::vector<uint64_t> inputOmniSliceStrideVec = {};
    std::vector<uint64_t> outputOmniSliceStrideVec = {};

    BuildSliceStrideVec(
        stepSliceInfo, rpt, repeatNum, ifNewRoot, sliceSize, inputOmniSliceSizeVec, inputOmniSliceStrideVec,
        outputOmniSliceStrideVec);

    for (uint32_t i = 0; i < inputOmniSliceSizeVec.size(); i++) {
        HCCL_DEBUG(
            "myRank_[%u] subCommRootId_[%u] rpt[%u] sliceSize[%llu] inputOmniSliceSizeVec[%d] = %llu "
            "isStepone[%d] isLastStep[%d]",
            myRank_, subCommRootId_, rpt, sliceSize, i, inputOmniSliceSizeVec[i], isStepOne_, isLastStep_);
    }
    for (uint32_t i = 0; i < inputOmniSliceStrideVec.size(); i++) {
        HCCL_DEBUG(
            "myRank_[%u] subCommRootId_[%u] rpt[%u] sliceSize[%llu] inputOmniSliceStrideVec[%d] = %llu "
            "isStepone[%d] isLastStep[%d] isFirstPiece[%d] isLastPiece[%d]",
            myRank_, subCommRootId_, rpt, sliceSize, i, inputOmniSliceStrideVec[i], isStepOne_, isLastStep_,
            isFirstPiece, isLastPiece);
    }
    for (uint32_t i = 0; i < outputOmniSliceStrideVec.size(); i++) {
        HCCL_DEBUG(
            "myRank_[%u] subCommRootId_[%u] rpt[%u] sliceSize[%llu] outputOmniSliceStrideVec[%d] = %llu "
            "isStepone[%d] isLastStep[%d] isFirstPiece[%d] isLastPiece[%d]",
            myRank_, subCommRootId_, rpt, sliceSize, i, outputOmniSliceStrideVec[i], isStepOne_, isLastStep_,
            isFirstPiece, isLastPiece);
    }

    std::vector<uint64_t> taskArgs
        = {inputAddr, outputAddr, sliceSize, token, isStepOne_, isLastStep_, ifNewRoot, isFirstPiece, isLastPiece};
    taskArgs.insert(taskArgs.end(), inputOmniSliceStrideVec.begin(), inputOmniSliceStrideVec.end());
    taskArgs.insert(taskArgs.end(), outputOmniSliceStrideVec.begin(), outputOmniSliceStrideVec.end());
    taskArgs.insert(taskArgs.end(), inputOmniSliceSizeVec.begin(), inputOmniSliceSizeVec.end());

    uint64_t argSize = taskArgs.size();
    CcuResult launchRet
        = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[%s] myRank[%u] HcommCcuKernelLaunch failed, ccuRet is:[%d]", __func__, myRank_, launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::RunScatterMeshDispatch(
    const TemplateDataParams& templateDataParams, TemplateResource& templateResource, uint64_t inputAddr,
    uint64_t outputAddrBase, uint64_t outBuffBaseOff, uint64_t token)
{
    auto stepSliceInfo = templateDataParams.stepSliceInfo;
    uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
    bool ifNewRoot = (subCommRootId_ == mySubCommRank_);
    uint64_t totalPieceNum = stepSliceInfo.inputOmniPipeSliceStride[0].size();
    uint64_t peerNum = templateRankSize_ - 1;
    // 防御一下，其实都是能够整除的
    CHK_PRT_RET(
        totalPieceNum % peerNum != 0,
        HCCL_ERROR(
            "inputOmniPipeSliceStride size=%llu not divisible by peerNum=%llu, repeatNum truncated", totalPieceNum,
            peerNum),
        HCCL_E_INTERNAL);
    uint64_t repeatNum = totalPieceNum / peerNum;

    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank_[%u] subCommRootId_[%u] ifNewRoot[%d] isStepone[%d] isLastStep[%d] "
        "repeatNum[%llu]",
        __func__, myRank_, mySubCommRank_, subCommRootId_, ifNewRoot, isStepOne_, isLastStep_, repeatNum);

    for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
        HcclResult ret
            = LaunchOneRepeat(stepSliceInfo, templateResource, rpt, repeatNum, ifNewRoot, inputAddr, outputAddr, token);
        if (ret != HcclResult::HCCL_SUCCESS) {
            return ret;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::RunLocalCopy(
    const TemplateDataParams& templateDataParams, TemplateResource& templateResource, uint64_t inputAddrBase,
    uint64_t outputAddrBase)
{
    HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
    DataSlice dstSlice(
        buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
    DataSlice srcSlice(
        buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
    HCCL_DEBUG(
        "[%s] myRank[%u] TempLocalCopy sliceSize[%llu] inputAddrBase[%llu] inputAddrOffset[%llu] "
        "outputAddrBase[%llu] "
        "outputAddrOffset[%llu]",
        __func__, myRank_, templateDataParams.sliceSize, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase,
        buffInfo_.outBuffBaseOff);
    CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    (void)param;
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    buffInfo_ = templateDataParams.buffInfo;
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank_[%u] isStepone[%d] isLastStep[%d] localCopyFlag[%d] start", __func__, myRank_,
        mySubCommRank_, isStepOne_, isLastStep_, localCopyFlag);
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);
    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);

    uint64_t outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    uint64_t inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;

    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    HCCL_DEBUG(
        "buffInfo_.inputSize[%llu] buffInfo_.inputAddr[%llu] buffInfo_.inBuffType[%llu]", buffInfo_.inputSize,
        inputAddrBase, buffInfo_.inBuffType);
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    if (localCopyFlag == 0) {
        return RunScatterMeshDispatch(
            templateDataParams, templateResource, inputAddr, outputAddrBase, outBuffBaseOff, token);
    } else if (localCopyFlag == 1) {
        return RunLocalCopy(templateDataParams, templateResource, inputAddrBase, outputAddrBase);
    }

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempScatterOmniPipeMesh1DMem2Mem::BuildSliceStrideVec(
    const StepSliceInfo& stepSliceInfo, uint32_t rpt, uint64_t repeatNum, bool ifNewRoot, uint64_t& sliceSize,
    std::vector<uint64_t>& inputOmniSliceSizeVec, std::vector<uint64_t>& inputOmniSliceStrideVec,
    std::vector<uint64_t>& outputOmniSliceStrideVec)
{
    if (!ifNewRoot) {
        for (uint32_t ridx = 0; ridx < templateRankSize_; ridx++) {
            inputOmniSliceSizeVec.push_back(0);
            outputOmniSliceStrideVec.push_back(0);
            inputOmniSliceStrideVec.push_back(0);
        }
        return;
    }
    uint64_t originIndex = 0;
    sliceSize = stepSliceInfo.stepSliceSize[myRank_ / templateRankSize_][rpt];
    for (uint32_t ridx = 0; ridx < templateRankSize_; ridx++) {
        if (ridx == subCommRootId_) {
            inputOmniSliceSizeVec.push_back(0);
            outputOmniSliceStrideVec.push_back(0);
            inputOmniSliceStrideVec.push_back(0);
        } else {
            uint64_t sliceStrideiIndex = repeatNum * originIndex + rpt;
            originIndex = originIndex + 1;
            inputOmniSliceSizeVec.push_back(
                stepSliceInfo.stepSliceSize[myRank_ / templateRankSize_][sliceStrideiIndex]);
            outputOmniSliceStrideVec.push_back(
                stepSliceInfo.outputOmniPipeSliceStride[myRank_ / templateRankSize_][sliceStrideiIndex]);
            inputOmniSliceStrideVec.push_back(
                stepSliceInfo.inputOmniPipeSliceStride[myRank_ / templateRankSize_][sliceStrideiIndex]);
        }
    }
}

u64 CcuTempScatterOmniPipeMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

} // namespace ops_hccl
