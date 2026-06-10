/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "ccu_launch_dl.h"
#include "ccu_kernel_broadcast_mesh1d_mem2mem.h"
#include "ccu/ccu_temp_broadcast_mesh_1D_mem2mem.h"

namespace ops_hccl {

CcuTempBroadcastMesh1DMem2Mem::CcuTempBroadcastMesh1DMem2Mem(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    HCCL_INFO("[CcuTempBroadcastMesh1DMem2Mem] subCommRanksSize[%zu] mySubCommRank[%u] subCommRootId[%u] rankId[%u]",
              subCommRanks.size(), mySubCommRank_, subCommRootId_, rankId);
}

CcuTempBroadcastMesh1DMem2Mem::~CcuTempBroadcastMesh1DMem2Mem()
{
}

HcclResult CcuTempBroadcastMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempBroadcastMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的kernelArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuBroadcastMesh1DMem2MemKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuBroadcastMesh1DMem2MemKernel);

    std::vector<HcclChannelDesc> channelDescs;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs,
            CommTopo::COMM_TOPO_1DMESH));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        }
        HCCL_DEBUG("[CcuTempBroadcastMesh1DMem2Mem::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    }

    auto kernelArg = std::make_shared<CcuKernelArgBroadcastMesh1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->rootId = subCommRootId_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempBroadcastMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu, mySubCommRank_=%u, subCommRootId_=%u root=%u",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size(),
               mySubCommRank_, subCommRootId_, param.root);

    return HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastMesh1DMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempBroadcastMesh1DMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempBroadcastMesh1DMem2Mem::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 inputOffsetIdx = 11;
    constexpr u32 outputOffsetIdx = 12;
    uint64_t argSize = 11;

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];

    void *taskArgs = reinterpret_cast<void*>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempBroadcastMesh1DMem2Mem::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempBroadcastMesh1DMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                    const TemplateDataParams& templateDataParams,
                                                    TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t repeatNum          = templateDataParams.repeatNum;
    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride  = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t curSliceSize       = templateDataParams.sliceSize;
    uint64_t normalSliceSize    = curSliceSize / templateRankSize_;
    uint64_t lastSliceSize      = curSliceSize % templateRankSize_ + normalSliceSize;
    uint64_t repeatNumVar       = UINT64_MAX - repeatNum;

    uint64_t currentRankSliceInputOffset  = inputSliceStride * mySubCommRank_;
    uint64_t currentRankSliceOutputOffset = outputSliceStride * mySubCommRank_;
    uint64_t allgatherOffset              = normalSliceSize * mySubCommRank_;

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, currentRankSliceInputOffset,
                                       currentRankSliceOutputOffset, inputRepeatStride, outputRepeatStride,
                                       normalSliceSize, lastSliceSize, allgatherOffset, repeatNumVar};
    uint64_t argSize = 11;

    HCCL_INFO("[CcuTempBroadcastMesh1DMem2Mem::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], inputRepeatStride[%llu], "
              "outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu], allgatherOffset[%llu], "
              "repeatNumVar[%llu]",
              inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, inputRepeatStride,
              outputRepeatStride, normalSliceSize, lastSliceSize, allgatherOffset, repeatNumVar);
    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
                                                taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempBroadcastMesh1DMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, inputAddr, outputAddr, token, currentRankSliceInputOffset,
                           currentRankSliceOutputOffset, inputRepeatStride, outputRepeatStride,
                           normalSliceSize, lastSliceSize, allgatherOffset, repeatNumVar,
                           buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_DEBUG("[CcuTempBroadcastMesh1DMem2Mem::KernelRun] end");

    return HCCL_SUCCESS;
}

void CcuTempBroadcastMesh1DMem2Mem::SetRoot(u32 root)
{
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    HCCL_INFO("[CcuTempBroadcastMesh1DMem2Mem][SetRoot] myRank_ [%u], set root_ [%u] subCommRanks[%u]", myRank_, root, subCommRootId_);
}

u64 CcuTempBroadcastMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempBroadcastMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
