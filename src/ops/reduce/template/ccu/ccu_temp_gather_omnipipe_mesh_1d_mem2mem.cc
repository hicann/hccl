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
#include "ccu_kernel_gather_omnipipe_mesh_1d_mem2mem.h"
#include "ccu_temp_gather_omnipipe_mesh_1d_mem2mem.h"
#include "alg_data_trans_wrapper.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempGatherOmniPipeMesh1DMem2Mem::CcuTempGatherOmniPipeMesh1DMem2Mem(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    // 子通信域的root卡号
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_ = std::distance(ranks.begin(), itRoot);
    }

    ifRealRoot_ = (rankId == param.root);
}

CcuTempGatherOmniPipeMesh1DMem2Mem::~CcuTempGatherOmniPipeMesh1DMem2Mem() {}

void CcuTempGatherOmniPipeMesh1DMem2Mem::SetRoot(u32 root)
{
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
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

void CcuTempGatherOmniPipeMesh1DMem2Mem::UnsetRoot(u32 rank)
{
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem][UnsetRoot] myRank_ [%u], unset root [%u] ", myRank_, rank);
    if (!ifRealRoot_) {
        subCommRootId_ = UINT32_MAX;
    }
}

u64 CcuTempGatherOmniPipeMesh1DMem2Mem::GetThreadNum() const { return 1; }

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);

    return HCCL_SUCCESS;
}

uint32_t CcuTempGatherOmniPipeMesh1DMem2Mem::RemoteRankId2RankId(const uint32_t remoteRankId) const
{
    uint32_t subCommRankId = 0;
    std::vector<u32> ranks = subCommRanks_[0];
    auto it = std::find(ranks.begin(), ranks.end(), remoteRankId);
    if (it != ranks.end()) {
        subCommRankId = std::distance(ranks.begin(), it);
    }
    return subCommRankId;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    GetRes(resourceRequest);
    resourceRequest.ccuKernelNum.push_back(1);

    HCCL_DEBUG(
        "[%s]notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__, resourceRequest.notifyNumOnMainThread,
        resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    CHK_SAFETY_FUNC_RET(
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuGatherOmniPipeMesh1DMem2MemKernel"));
    kernelInfo.kernelFunc = reinterpret_cast<void*>(CcuGatherOmniPipeMesh1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1DFullMesh(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(
            comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : channelDescs) {
            if (channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
                HCCL_ERROR("[%s] channel.channelProtocol[%u]", __func__, channel.channelProtocol);
                return HCCL_E_INTERNAL;
            }
        }
    }

    HCCL_DEBUG("[%s] Get Mesh channels Success.", __func__);
    auto kernelArg = std::make_shared<CcuKernelArgGatherOmniPipeMesh1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = subCommRootId_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelArg->myrealrank = myRank_;

    kernelInfo.setKernelArg(kernelArg);

    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    HCCL_DEBUG(
        "[%s]channelDescs.size()=%llu, ccuKernelInfos.size() = %llu", __func__, channelDescs.size(),
        resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun(
    const OpParam& param, const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] mesh start");
    buffInfo_ = templateDataParams.buffInfo;
    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    if (localCopyFlag == 0) {
        CHK_RET(RunGatherComm(
            templateDataParams.stepSliceInfo, inputAddr, outputAddr, token, localCopyFlag, templateResource));
    } else if (localCopyFlag == 1) {
        CHK_RET(RunLocalCopy(templateDataParams, templateResource));
    }
    HCCL_DEBUG("[CcuTempGatherOmniPipeMesh1DMem2Mem::KernelRun] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::RunGatherComm(
    const StepSliceInfo& stepSliceInfo, uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t localCopyFlag,
    TemplateResource& templateResource)
{
    const auto& strides = stepSliceInfo.inputOmniPipeSliceStride;
    for (uint32_t rpt = 0; rpt < strides[0].size(); ++rpt) {
        bool isFirstPiece = (rpt == 0);
        bool isLastPiece = (rpt == (strides[0].size() - 1));
        uint64_t sliceSize = stepSliceInfo.stepSliceSize[0][rpt];
        std::vector<uint64_t> sliceSizeVec;
        std::vector<uint64_t> inputVec;
        std::vector<uint64_t> outputVec;
        for (uint32_t peerId = 0; peerId < templateRankSize_; ++peerId) {
            sliceSizeVec.push_back(stepSliceInfo.stepSliceSize[peerId][rpt]);
            inputVec.push_back(stepSliceInfo.inputOmniPipeSliceStride[peerId][rpt]);
            outputVec.push_back(stepSliceInfo.outputOmniPipeSliceStride[peerId][rpt]);
        }
        bool ifNewRoot = (subCommRootId_ == mySubCommRank_);
        CHK_RET(LaunchGatherKernel(
            templateResource, inputAddr, outputAddr, token, localCopyFlag, sliceSize, isFirstPiece, isLastPiece,
            ifNewRoot, sliceSizeVec, inputVec, outputVec));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::LaunchGatherKernel(
    TemplateResource& templateResource, uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t localCopyFlag,
    uint64_t sliceSize, bool isFirstPiece, bool isLastPiece, bool ifNewRoot, const std::vector<uint64_t>& sliceSizeVec,
    const std::vector<uint64_t>& inputVec, const std::vector<uint64_t>& outputVec)
{
    std::vector<uint64_t> taskArgs
        = {inputAddr, outputAddr, token, localCopyFlag, sliceSize, ifNewRoot, isFirstPiece, isLastPiece};
    taskArgs.insert(taskArgs.end(), sliceSizeVec.begin(), sliceSizeVec.end());
    taskArgs.insert(taskArgs.end(), inputVec.begin(), inputVec.end());
    taskArgs.insert(taskArgs.end(), outputVec.begin(), outputVec.end());
    uint64_t argSize = taskArgs.size();
    CcuResult launchRet
        = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[%s] myRank[%u] HcommCcuKernelLaunch failed, ccuRet is:[%d]", __func__, myRank_, launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempGatherOmniPipeMesh1DMem2Mem::RunLocalCopy(
    const TemplateDataParams& templateDataParams, TemplateResource& templateResource)
{
    HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
    DataSlice srcSlice(
        buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
    DataSlice dstSlice(
        buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
    CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempGatherOmniPipeMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

} // namespace ops_hccl
