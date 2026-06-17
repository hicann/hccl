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
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"
#include "ccu_launch_dl.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

CcuTempAllGatherNHR1DMem2Mem::CcuTempAllGatherNHR1DMem2Mem(const OpParam& param, const u32 rankId,
                                                                   const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域中的虚拟rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
}

CcuTempAllGatherNHR1DMem2Mem::~CcuTempAllGatherNHR1DMem2Mem()
{
}

u64 CcuTempAllGatherNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    return 0;
}

HcclResult CcuTempAllGatherNHR1DMem2Mem::ProcessNHRStepInfo(HcclComm comm,
                                                            std::vector<NHRStepInfo>& stepInfoVector,
                                                            std::map<u32, u32>& rank2ChannelIdx,
                                                            u32 dieNum, u32 enableDieId,
                                                            std::vector<std::vector<HcclChannelDesc>>& channelsPerDie)
{
    constexpr u32 DIE_NUM_1 = 1;
    constexpr u32 DIE_NUM_2 = 2;
    constexpr u32 DIE_0 = 0;
    constexpr u32 DIE_1 = 1;
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (dieNum == DIE_NUM_1) {
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.fromRank, rankIdToChannelDesc_, enableDieId, 
                rank2ChannelIdx, channelsPerDie[DIE_0]));
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.toRank, rankIdToChannelDesc_, enableDieId,
                rank2ChannelIdx, channelsPerDie[DIE_0]));
        } else if (dieNum == DIE_NUM_2) {
            // 加入fromRank 2个die的链路
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.fromRank, rankIdToChannelDesc_, DIE_0,
                rank2ChannelIdx, channelsPerDie[DIE_0]));
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.fromRank, rankIdToChannelDesc_, DIE_1,
                rank2ChannelIdx, channelsPerDie[DIE_1]));
            // 加入toRank 2个die的链路
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.toRank, rankIdToChannelDesc_, DIE_0,
                rank2ChannelIdx, channelsPerDie[DIE_0]));
            CHK_RET(SelectChannelToVec(comm, myRank_, stepInfo.toRank, rankIdToChannelDesc_, DIE_1,
                rank2ChannelIdx, channelsPerDie[DIE_1]));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                         AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> channelDescs;
    std::vector<HcclChannelDesc> myChannelDescs;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_CLOS)); 
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        } 
    } else {
        CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, myChannelDescs));
        channelDescs = myChannelDescs;
    }
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    // 1.从获得的channelDesc，判断kernel发送到几个die上
    uint32_t dieNum = 0;
    uint32_t enableDieId = 0;
    CHK_RET(GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc_, myRank_, dieNum, enableDieId));

    if (dieNum < 1 || dieNum > CCU_DIE_NUM_MAX_2) { // 目前只支持1个或2个die
        HCCL_ERROR("[CcuTempAllGatherNHR1DMem2Mem::CalcRes] get channelDescs fail");
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempAllGatherNHR1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u], kernelNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum, kernelNum);

    // 2.将channelDescs分到2个die
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(dieNum);
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<NHRStepInfo> stepInfoVector;

    CHK_RET(ProcessNHRStepInfo(comm, stepInfoVector, rank2ChannelIdx, dieNum, enableDieId, channelsPerDie));
    if (dieNum > 1) { // 通过端口数划分channel，适配跨框die0连die1的场景，避免建链失败
        CHK_RET(ReverseChannelPerDieIfNeed(comm, myRank_, channelsPerDie));
    }

     // 3.构造kernelInfo
    CHK_RET(BuildCcuKernelInfos(param, dieNum, kernelNum, stepInfoVector, rank2ChannelIdx, channelsPerDie,
                                resourceRequest));

    HCCL_DEBUG("[CcuTempAllGatherNHR1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMem2Mem::BuildCcuKernelInfos(const OpParam& param, uint32_t dieNum, uint32_t kernelNum,
                                                                   const std::vector<NHRStepInfo>& stepInfoVector,
                                                                   const std::map<u32, u32>& rank2ChannelIdx,
                                                                   const std::vector<std::vector<HcclChannelDesc>>& channelsPerDie,
                                                                   AlgResourceRequest& resourceRequest)
{
    for (uint32_t kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        // 创建每个kernel的KernelArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;
        strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllGatherNHR1DMem2MemKernel");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllGatherNHR1DMem2MemKernel);

        auto kernelArg = std::make_shared<CcuKernelArgAllGatherNHR1D>();
        kernelArg->dimSize = subCommRanks_[0].size();
        kernelArg->mySubCommRankId = mySubCommRank_;
        kernelArg->axisId = kernelIdx;
        kernelArg->stepInfoVector = stepInfoVector;
        kernelArg->rank2ChannelIdx = rank2ChannelIdx;
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelArg->axisSize = dieNum;
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channelsPerDie[kernelIdx];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMem2Mem::SplitDataFor2Dies(const OpParam& param,
                                                           const TemplateDataParams& templateDataParams,
                                                           uint64_t& die0Size, uint64_t& die1Size) const
{
    constexpr uint64_t MULTIPLIER = 4;
    uint64_t typeSize = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t dataCount = (templateDataParams.sliceSize / typeSize);

    if (dataCount <= templateRankSize_ * MULTIPLIER) {  // 数据量极小，不划分die
        die0Size = dataCount * typeSize;
        die1Size = 0;
        return HcclResult::HCCL_SUCCESS;
    }
    u8 die0PortGroupSize = 6;
    u8 die1PortGroupSize = 2;

    die0Size = (dataCount * die0PortGroupSize / (die0PortGroupSize + die1PortGroupSize)) * typeSize;
    die1Size = templateDataParams.sliceSize - die0Size;
    HCCL_DEBUG("[CcuTempAllGatherNHR1DMem2Mem::SplitDataFor2Dies] die0Size = %llu, die1Size = %llu", die0Size , die1Size);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMem2Mem::PrepareLaunchArgs(const OpParam& param,
    const TemplateDataParams& templateDataParams, u32 kernelNum,
    std::vector<uint64_t>& taskArgs, uint64_t& argSize)
{
    buffInfo_ = templateDataParams.buffInfo;
    uint64_t die0Size = 0;
    uint64_t die1Size = 0;
    constexpr uint32_t MAX_DIE_NUM_2 = 2;
    if (kernelNum == MAX_DIE_NUM_2) {
        SplitDataFor2Dies(param, templateDataParams, die0Size, die1Size);
    } else {
        die0Size = templateDataParams.sliceSize;
    }

    uint64_t inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));
    uint64_t repeatNum = UINT64_MAX - templateDataParams.repeatNum;
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t die0LastSize = templateDataParams.tailSize / kernelNum;
    uint64_t die1LastSize = templateDataParams.tailSize - die0LastSize;
    bool inputOutputEqual = (inputAddr + inputSliceStride * mySubCommRank_ == outputAddr + outputSliceStride * mySubCommRank_);
    uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);
    
    HCCL_INFO("[CcuTempAllGatherNHR1DMem2Mem] dimSize[%llu], die0Size[%llu], die1Size[%llu], inputAddr[%llu],\
        outputAddr[%llu], repeatNum[%llu], inputSliceStride[%llu], outputSliceStride[%llu],\
        inputRepeatStride[%llu], outputRepeatStride[%llu]",
        templateRankSize_, die0Size, die1Size, inputAddr, outputAddr, repeatNum, inputSliceStride,
        outputSliceStride, inputRepeatStride, outputRepeatStride);

    taskArgs = {inputAddr, outputAddr, token, die0Size, die1Size, repeatNum,
                inputSliceStride, outputSliceStride, inputRepeatStride,
                outputRepeatStride, isInputOutputEqual, die0LastSize, die1LastSize};
    argSize = 13;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMem2Mem::KernelRun(const OpParam& param,
                                                       const TemplateDataParams& templateDataParams,
                                                       TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempAllGatherNHR1DMem2Mem] Template KernelRun start.");
    opMode_ = param.opMode;
    u32 kernelNum = templateResource.ccuKernels.size();

    if (templateDataParams.sliceSize == 0 && templateDataParams.tailSize == 0) {
        HCCL_INFO("[CcuTempAllGatherNHR1DMem2Mem] sliceSize is 0, no need do, just success.");
        return HCCL_SUCCESS;
    }

    std::vector<uint64_t> taskArgs;
    uint64_t argSize = 0;
    CHK_RET(PrepareLaunchArgs(param, templateDataParams, kernelNum, taskArgs, argSize));

    uint64_t die0Size = taskArgs[3];
    uint64_t die1Size = taskArgs[4];

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);

        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (uint32_t axisId = 0; axisId < kernelNum; axisId++) {
        if ((templateDataParams.tailSize == 0) && ((axisId == 0 && die0Size == 0) || (axisId == 1 && die1Size == 0))) {
            // 数据长度为0的kernel不下发
            continue;
        }
        
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[axisId],
                                                    templateResource.ccuKernels[axisId],
                                                    taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGatherNHR1DMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    // 后流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);

        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    // 所有task下发完后再保存参数信息
    CcuKernelSubmitInfo submitInfo;
    CHK_RET(FillCachedArgs(submitInfo, taskArgs[0], taskArgs[1], taskArgs[2], taskArgs[3], taskArgs[4], taskArgs[5],
                           taskArgs[6], taskArgs[7], taskArgs[8], taskArgs[9], taskArgs[10], taskArgs[11], taskArgs[12],
                           buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff, mySubCommRank_));
    for (u32 i = 0; i < kernelNum; i++) {
        // 2个kernel的TaskArg相同
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        templateResource.submitInfos.push_back(submitInfo);
    }

    HCCL_INFO("[CcuTempAllGatherNHR1DMem2Mem] Template Run for all steps Ends.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllGatherNHR1DMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }   
    HCCL_DEBUG("[CcuTempAllGatherNHR1DMem2Mem::FastLaunch] start");
    u32 kernelNum = tempFastLaunchCtx.ccuKernelSubmitInfos.size();
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 inputSliceStrideIdx  = 6;
    constexpr u32 outputSliceStrideIdx  = 7;
    constexpr u32 isInputOutputEqualIdx = 10;
    constexpr u32 inputOffsetIdx   = 13;
    constexpr u32 outputOffsetIdx  = 14;
    constexpr u32 mySubCommRankIdx = 15;
    uint64_t argSize = 13;

    uint64_t inputAddr          = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    uint64_t outputAddr         = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];
    uint64_t inputSliceStride   = args[inputSliceStrideIdx];
    uint64_t outputSliceStride  = args[outputSliceStrideIdx];
    uint64_t mySubCommRank      = args[mySubCommRankIdx];
    bool inputOutputEqual = (inputAddr + inputSliceStride * mySubCommRank == outputAddr + outputSliceStride * mySubCommRank);

    uint64_t isInputOutputEqual = static_cast<uint64_t>(inputOutputEqual);

    args[inputIdx]  = inputAddr;
    args[outputIdx] = outputAddr;
    args[isInputOutputEqualIdx] = isInputOutputEqual;

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

     void *taskArgs = reinterpret_cast<void*>(args);

    for (u32 kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[kernelIdx],
                                                   tempFastLaunchCtx.ccuKernelSubmitInfos[kernelIdx].kernelHandle,
                                                   taskArgs, argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempAllGatherNHR1DMem2Mem::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }
    HCCL_DEBUG("[CcuTempAllGatherNHR1DMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllGatherNHR1DMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
{
    u32 rankIdx = mySubCommRank_;
    std::vector<u32> ranks = subCommRanks_[0];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = mySubCommRank_;

    // 计算通信对象，计算出的是虚拟rankid
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 sendTo = (rankIdx + deltaRank) % templateRankSize_;
    u32 recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = mySubCommRank_;
    u32 rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = ranks[sendTo];  //  从虚拟rankid转换至通信域真实rankid
    stepInfo.fromRank = ranks[recvFrom];

    HCCL_INFO("[CcuTempAllGatherNHR1DMem2Mem][GetStepInfo] nSlices[%u] toRank[%u] fromRank[%u]",
                nSlices, stepInfo.toRank, stepInfo.fromRank);

    // 计算本rank在本轮收/发中的slice编号
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        HCCL_INFO("[AllGatherNHR1D][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);
        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllGatherNHR1DMem2Mem::GetThreadNum() const
{
    u32 threadNum = 2;
    return threadNum;
}

HcclResult CcuTempAllGatherNHR1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl