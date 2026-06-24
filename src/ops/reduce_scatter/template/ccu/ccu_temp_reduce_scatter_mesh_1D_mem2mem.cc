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
#include "ccu_kernel_reduce_scatter_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempReduceScatterMesh1DMem2Mem::CcuTempReduceScatterMesh1DMem2Mem(const OpParam& param, const u32 rankId,
                                                                     const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempReduceScatterMesh1DMem2Mem::~CcuTempReduceScatterMesh1DMem2Mem()
{
}

HcclResult CcuTempReduceScatterMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuReduceScatterMesh1DMem2MemKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuReduceScatterMesh1DMem2MemKernel);
    
    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1DFullMesh(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        }
        HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::CalcRes] Get Mesh Channel Success!");
    }
    
    auto kernelArg = std::make_shared<CcuKernelArgReduceScatterMesh1DMem2Mem>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1DMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempReduceScatterMesh1DMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 scratchIdx = 3;
    
    constexpr u32 inputOffsetIdx = 16;   // inBuffBaseOff
    constexpr u32 outputOffsetIdx = 17;  // outBuffBaseOff
    constexpr u32 scratchOffsetIdx = 18; // hcclBuffBaseOff
    uint64_t argSize = 16;

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
    args[scratchIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.hcclBuff.addr) + args[scratchOffsetIdx];

    void *taskArgs = reinterpret_cast<void*>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempReduceScatterMesh1DMem2Mem::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempReduceScatterMesh1DMem2Mem::KernelRun(const OpParam& param,
                                                        const TemplateDataParams& templateDataParams,
                                                        TemplateResource& templateResource)
{
    if (templateDataParams.sliceSize == 0 && templateDataParams.tailSize == 0) {
        HCCL_INFO("[CcuTempReduceScatterMesh1DMem2Mem] sliceSize is 0, no need to do, just success.");
        return HCCL_SUCCESS;
    }
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t scratchAddr        = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t inputSliceStride   = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride  = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride  = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t normalSliceSize    = templateDataParams.sliceSize;
    uint64_t lastSliceSize      = templateDataParams.tailSize;
    uint64_t repeatNum          = UINT64_MAX - templateDataParams.repeatNum;
    uint64_t scratchRepeatStride = templateRankSize_ * normalSliceSize;

    uint64_t currentRankSliceInputOffset = inputSliceStride * mySubCommRank_;
    uint64_t currentRankSliceOutputOffset = outputSliceStride * mySubCommRank_;

    LoopGroupConfig  config{};
    config.msInterleave = REDUCE_MS_CNT;
    config.loopCount    = REDUCE_SCATTER_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE;
    auto goSize = (mySubCommRank_ == (templateRankSize_ - 1)) ? CalGoSize(lastSliceSize, config) : 
                   CalGoSize(normalSliceSize, config);

    HCCL_INFO("[CcuTempReduceScatterMesh1DMem2Mem::KernelRun] TaskArgs: inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu], "
              "inputOffset[%llu], outputOffset[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], "
              "normalSliceSize[%llu], lastSliceSize[%llu], scratchRepeatStride[%llu], repeatNum[%llu]",
              inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset,
              inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize,
              scratchRepeatStride, UINT64_MAX - repeatNum);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, scratchAddr,
        currentRankSliceInputOffset, currentRankSliceOutputOffset, inputRepeatStride, outputRepeatStride,
        normalSliceSize, lastSliceSize, scratchRepeatStride, repeatNum, goSize[0], goSize[1], goSize[2], goSize[3]};
    
    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], 
                                               taskArgs.data(), taskArgs.size());
    CHK_PRT_RET(launchRet != CCU_SUCCESS, HCCL_ERROR(
        "[CcuTempReduceScatterMesh1DMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet), ConvertCcuToHccl(launchRet));

    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, inputAddr, outputAddr, token, scratchAddr,
        currentRankSliceInputOffset, currentRankSliceOutputOffset, inputRepeatStride, outputRepeatStride,
        normalSliceSize, lastSliceSize, scratchRepeatStride, repeatNum, goSize[0], goSize[1], goSize[2], goSize[3],
        buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, buffInfo_.hcclBuffBaseOff));
    templateResource.submitInfos.push_back(submitInfo);

    HCCL_DEBUG("[CcuTempReduceScatterMesh1DMem2Mem::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempReduceScatterMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // one shot 场景，scratch Buffer 需要是 usrIn的rankSize倍
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

u64 CcuTempReduceScatterMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempReduceScatterMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;

    return HCCL_SUCCESS;
}
} // namespace ops_hccl