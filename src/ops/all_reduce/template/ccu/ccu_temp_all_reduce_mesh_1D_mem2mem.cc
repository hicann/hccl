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
#include "ccu_temp_all_reduce_mesh_1D_mem2mem.h"
#include "ccu_kernel_all_reduce_mesh1d_mem2mem.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempAllReduceMeshMem2Mem1D::CcuTempAllReduceMeshMem2Mem1D(const OpParam& param, 
                                                const u32 rankId, // 传通信域的rankId，userRank
                                                const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
    templateRankSize_ = subCommRanks[0].size();
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType; 
}

CcuTempAllReduceMeshMem2Mem1D::~CcuTempAllReduceMeshMem2Mem1D()
{
}

uint64_t CcuTempAllReduceMeshMem2Mem1D::RoundUp(uint64_t dividend, uint64_t divisor) const
{
    return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
}

HcclResult CcuTempAllReduceMeshMem2Mem1D::CalcSlice(const u64 dataSize, RankSliceInfo &sliceInfoVec)
{
    std::vector<SliceInfo> tmp(subCommRanks_.size());
    sliceInfoVec.resize(templateRankSize_, tmp);

    u64 unitAllignSize = DataTypeSizeGet(dataType_);
    u64 chunkSize      = RoundUp(dataSize, (templateRankSize_ * unitAllignSize)) * unitAllignSize;
    HCCL_INFO("[CcuTempAllReduceMeshMem2Mem1D] chunkSize[%llu], dataSize[%llu], templateRankSize_[%u], unitAllignSize[%llu]",
              chunkSize, dataSize, templateRankSize_, unitAllignSize);
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64       currChunkSize  = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice          = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR(
            "[CcuTempAllReduceMeshMem2Mem1D] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!",
            chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMeshMem2Mem1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                  AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllReduceMeshMem2Mem1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelAllReduceMesh1DMem2Mem");
 	kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllReduceMeshMem2Mem1DKernel);
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
        HCCL_DEBUG("[CcuTempAllReduceMeshMem2Mem1D::CalcRes] Get Mesh Channel Success!");
    }
    auto kernelArg = std::make_shared<CcuKernelArgAllReduceMeshMem2Mem1D>();
    kernelArg->rankSize = subCommRanks_[0].size();
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllReduceMeshMem2Mem1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllReduceMeshMem2Mem1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

u64 CcuTempAllReduceMeshMem2Mem1D::GetThreadNum() const
{
    return 1;
}
 
HcclResult CcuTempAllReduceMeshMem2Mem1D::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMeshMem2Mem1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllReduceMeshMem2Mem1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllReduceMeshMem2Mem1D::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 scratchIdx = 3;
    constexpr u32 inputOffsetIdx = 15;
    constexpr u32 outputOffsetIdx = 16;
    constexpr u32 scratchOffsetIdx = 17;
    uint64_t argSize = 15;

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
    args[scratchIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.hcclBuff.addr) + args[scratchOffsetIdx];

    void *taskArgs = reinterpret_cast<void*>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllReduceMeshMem2Mem1D::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempAllReduceMeshMem2Mem1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempAllReduceMeshMem2Mem1D::BuildTaskArgs(const uint64_t inputAddr, const uint64_t outputAddr,
    const uint64_t token, const uint64_t scratchAddr, const uint64_t currentRankSliceInputOffset,
    const uint64_t currentRankSliceOutputOffset, const uint64_t normalSliceSize, const uint64_t lastSliceSize,
    const uint64_t mySliceSize, const uint64_t sliceOffset, const uint64_t isInputOutputEqual,
    const std::vector<uint64_t>& goSize, std::vector<uint64_t>& taskArgs) const
{
    taskArgs = {inputAddr, outputAddr, token, scratchAddr, currentRankSliceInputOffset,
                currentRankSliceOutputOffset, normalSliceSize, lastSliceSize, mySliceSize,
                sliceOffset, isInputOutputEqual, goSize[0], goSize[1], goSize[2], goSize[3]};
}

void CcuTempAllReduceMeshMem2Mem1D::SaveSubmitInfo(const uint64_t inputAddr, const uint64_t outputAddr,
    const uint64_t token, const uint64_t scratchAddr, const uint64_t currentRankSliceInputOffset,
    const uint64_t currentRankSliceOutputOffset, const uint64_t normalSliceSize, const uint64_t lastSliceSize,
    const uint64_t mySliceSize, const uint64_t sliceOffset, const uint64_t isInputOutputEqual,
    const std::vector<uint64_t>& goSize, TemplateResource& templateResource) const
{
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    (void)FillCachedArgs(submitInfo, inputAddr, outputAddr, token, scratchAddr,
        currentRankSliceInputOffset, currentRankSliceOutputOffset, normalSliceSize,
        lastSliceSize, mySliceSize, sliceOffset, isInputOutputEqual,
        goSize[0], goSize[1], goSize[2], goSize[3],
        buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, buffInfo_.hcclBuffBaseOff);
    templateResource.submitInfos.push_back(submitInfo);
}

HcclResult CcuTempAllReduceMeshMem2Mem1D::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                    TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSlice(templateDataParams.sliceSize, sliceInfoVec));

    const uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    const uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    const uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    const uint64_t isInputOutputEqual = (inputAddr == outputAddr) ? 1 : 0;

    const uint64_t normalSliceSize = sliceInfoVec[0][0].size;
    const uint64_t lastSliceSize = sliceInfoVec[templateRankSize_ - 1][0].size;
    const uint64_t mySliceSize = sliceInfoVec[mySubCommRank_][0].size;
    const uint64_t currentRankSliceInputOffset = templateDataParams.inputSliceStride * myRank_;
    const uint64_t currentRankSliceOutputOffset = templateDataParams.outputSliceStride * myRank_;
    const uint64_t sliceOffset = normalSliceSize * myRank_;

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = 16;
    config.memSlice = CCU_MS_SIZE;
    const std::vector<uint64_t> goSize = (myRank_ != templateRankSize_ - 1) ?
        CalGoSize(normalSliceSize, config) : CalGoSize(lastSliceSize, config);

    std::vector<uint64_t> taskArgs;
    BuildTaskArgs(inputAddr, outputAddr, token, scratchAddr, currentRankSliceInputOffset,
                  currentRankSliceOutputOffset, normalSliceSize, lastSliceSize, mySliceSize,
                  sliceOffset, isInputOutputEqual, goSize, taskArgs);

    CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0],
                                                taskArgs.data(), taskArgs.size());
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllReduceMeshMem2Mem1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    SaveSubmitInfo(inputAddr, outputAddr, token, scratchAddr, currentRankSliceInputOffset,
                   currentRankSliceOutputOffset, normalSliceSize, lastSliceSize, mySliceSize,
                   sliceOffset, isInputOutputEqual, goSize, templateResource);

    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl