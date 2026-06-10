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
#include "ccu_kernel_scatter_nhr1d_mem2mem.h"
#include "ccu_temp_scatter_nhr1d_mem2mem.h"
#include "ccu_launch_dl.h"
#include "alg_data_trans_wrapper.h"
#include <iostream>

namespace ops_hccl {

CcuTempScatterNHR1DMem2Mem::CcuTempScatterNHR1DMem2Mem(const OpParam &param, const u32 rankId,
                                                       const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    // 获取本卡在子通信域中的虚拟rankid
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
}

CcuTempScatterNHR1DMem2Mem::~CcuTempScatterNHR1DMem2Mem() {}

void CcuTempScatterNHR1DMem2Mem::SetRoot(u32 root)
{
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::vector<u32> ranks = subCommRanks_[0];
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + " "; }
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem][SetSubCommRoot] ranks = subCommRanks[0] is: %s", ranksStr.c_str());
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
}

u64 CcuTempScatterNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return subCommRanks_[0].size();
}

HcclResult CcuTempScatterNHR1DMem2Mem::GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum)
{
    constexpr u32 LINK_NUM_1 = 2;
    constexpr u32 LINK_NUM_2 = 2;
    auto firstElement = rankIdToChannelDesc_.begin();
    const std::vector<HcclChannelDesc> &firstVector = firstElement->second;
    if (firstVector.size() == 1) {
        dieNum = 1;
        return HcclResult::HCCL_SUCCESS;
    } else if (firstVector.size() == LINK_NUM_2) {
        // 检查2个channel是否在2个die上
        uint32_t dieId0 = 0;
        uint32_t dieId1 = 0;
        GetChannelDieId(comm, myRank_, firstVector[0], dieId0);
        GetChannelDieId(comm, myRank_, firstVector[1], dieId1);
        if (dieId0 == dieId1) {
            dieNum = LINK_NUM_1;
            HCCL_INFO("[CcuTempScatterNHR1DMem2Mem::GetDieNumFromChannelDescs] 2 channels on the same die, dieNum = 1.");
        } else {
            dieNum = LINK_NUM_2;
            HCCL_INFO("[CcuTempScatterNHR1DMem2Mem::GetDieNumFromChannelDescs] 2 channels on 2 dies, dieNum = 2.");
        }
        return HcclResult::HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[CcuTempScatterNHR1DMem2Mem::CalcRes] get channelDescs fail: there are [] link to rank []",
                   firstVector.size(), firstElement->first);
        return HcclResult::HCCL_E_INTERNAL;
    }
}

HcclResult CcuTempScatterNHR1DMem2Mem::ProcessNHRStepInfo(HcclComm comm,
                                                          std::vector<NHRStepInfo> &stepInfoVector,
                                                          std::map<u32, u32> &rank2ChannelIdx, u32 enableDieNum,
                                                          std::vector<std::vector<HcclChannelDesc>> &channelsPerDie)
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        stepInfoVector.push_back(stepInfo);
        if (rank2ChannelIdx.count(stepInfo.fromRank) == 0 && stepInfo.rxSliceIdxs.size() != 0) {
            // 存储 rankid → channelIdx 的索引
            u32 curChannelIdx = channelsPerDie[0].size();
            rank2ChannelIdx[stepInfo.fromRank] = curChannelIdx;
            for (HcclChannelDesc channel : rankIdToChannelDesc_.at(stepInfo.fromRank)) {
                uint32_t dieId = 0;
                CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
                // 如果是2个die的算法，则分别加入到2个vector中，否则只加入到1个vector
                uint32_t vecIdx = dieId % enableDieNum;
                // 限制只加入一个channel
                if (channelsPerDie[vecIdx].size() == curChannelIdx) {
                    channelsPerDie[vecIdx].push_back(channel);
                }
            }
        }
        if (rank2ChannelIdx.count(stepInfo.toRank) == 0 && stepInfo.txSliceIdxs.size() != 0) {
            u32 curChannelIdx = channelsPerDie[0].size();
            rank2ChannelIdx[stepInfo.toRank] = curChannelIdx;

            for (HcclChannelDesc channel : rankIdToChannelDesc_.at(stepInfo.toRank)) {
                u32 dieId = 0;
                CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
                u32 vecIdx = dieId % enableDieNum;
                if (channelsPerDie[vecIdx].size() == curChannelIdx) {
                    channelsPerDie[vecIdx].push_back(channel);
                }
            }
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterNHR1DMem2Mem::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                               AlgResourceRequest &resourceRequest)
{
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    // 1.从获得的channelDesc，判断kernel发送到几个die上
    uint32_t enableDieNum = 0;
    CHK_RET(GetDieNumFromChannelDescs(comm, enableDieNum));

    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) { // 目前只支持1个或2个die
        HCCL_ERROR("[CcuTempScatterNHR1DMem2Mem::CalcRes] get channelDescs fail");
        return HcclResult::HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = enableDieNum;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempScatterNHR1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 2.将channelDescs分到2个die
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(enableDieNum);
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<NHRStepInfo> stepInfoVector;

    CHK_RET(ProcessNHRStepInfo(comm,  stepInfoVector, rank2ChannelIdx, enableDieNum, channelsPerDie));

    // 3.构造kernelInfo
    for (uint32_t kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;
        CHK_SAFETY_FUNC_RET(strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuScatterNHR1DMem2MemKernel"));
        kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuScatterNHR1DMem2MemKernel);
        auto kernelArg = std::make_shared<CcuKernelArgScatterNHRMem2Mem1D>();
        kernelArg->rankSize = subCommRanks_[0].size();
        kernelArg->rankId = mySubCommRank_;
        kernelArg->rootId = subCommRootId_;
        kernelArg->axisId = kernelIdx;
        kernelArg->axisSize = enableDieNum;
        kernelArg->stepInfoVector = stepInfoVector;
        kernelArg->rank2ChannelIdx = rank2ChannelIdx;
        kernelArg->opParam = param;
        kernelArg->subCommRanks = subCommRanks_;
        kernelInfo.setKernelArg(kernelArg);
        kernelInfo.channels = channelsPerDie[kernelIdx];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    }

    HCCL_DEBUG("[CcuTempScatterNHR1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterNHR1DMem2Mem::SplitDataFor2Dies(const OpParam &param,
                                                         const TemplateDataParams &templateDataParams,
                                                         uint64_t &die0Size, uint64_t &die1Size) const
{
    constexpr uint64_t MULTIPLIER = 4;
    uint64_t typeSize = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t dataCount = (templateDataParams.sliceSize / typeSize);

    if (dataCount <= templateRankSize_ * MULTIPLIER) {  // 数据量极小，不划分die
        die0Size = dataCount * typeSize;
        die1Size = 0;
        return HcclResult::HCCL_SUCCESS;
    }
    u8 die0PortGroupSize = 1;
    u8 die1PortGroupSize = 1;

    die0Size = (dataCount * die0PortGroupSize / (die0PortGroupSize + die1PortGroupSize)) * typeSize;
    die1Size = templateDataParams.sliceSize - die0Size;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterNHR1DMem2Mem::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    HCCL_DEBUG("[CcuTempScatterNHR1DMem2Mem::FastLaunch] start");
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempScatterNHR1DMem2Mem::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    u32 kernelNum = tempFastLaunchCtx.ccuKernelSubmitInfos.size();
    buffInfo_ = tempFastLaunchCtx.buffInfo;
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    // 前流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        CHK_RET(PreSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxMainToSub));
    }

    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 scratchIdx = 2;
    args[inputIdx] = PointerToAddr(buffInfo_.inputPtr) + args[inputIdx];
    args[outputIdx] = PointerToAddr(buffInfo_.outputPtr) + args[outputIdx];
    args[scratchIdx] = PointerToAddr(buffInfo_.hcclBuff.addr) + args[scratchIdx];
    void *taskArgs = reinterpret_cast<void*>(args);
    uint64_t argSize = 17;

    for (u32 kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[kernelIdx],
                                                   tempFastLaunchCtx.ccuKernelSubmitInfos[kernelIdx].kernelHandle,
                                                   taskArgs, argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempScatterNHR1DMem2Mem::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }
    // 后流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(tempFastLaunchCtx.threads.begin() + 1, tempFastLaunchCtx.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(tempFastLaunchCtx.threads[0], subThreads, notifyIdxSubToMain));
    }
    HCCL_DEBUG("[CcuTempScatterNHR1DMem2Mem::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuTempScatterNHR1DMem2Mem::FillKernelRunTempArgs(const TemplateDataParams &templateDataParams, KernalRunTempArgs &tempArgs) const
{
    tempArgs.inputAddr = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    tempArgs.outputAddr = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    tempArgs.scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] buffInfo_.inputPtr [%p].", buffInfo_.inputPtr);
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] buffInfo_.inputSize [%llu].", buffInfo_.inputSize);

    tempArgs.sliceSize = templateDataParams.sliceSize;
    tempArgs.repeatNum = templateDataParams.repeatNum;
    tempArgs.inputSliceStride = templateDataParams.inputSliceStride;
    tempArgs.outputSliceStride = templateDataParams.outputSliceStride;
    tempArgs.inputRepeatStride = templateDataParams.inputRepeatStride;
    tempArgs.outputRepeatStride = templateDataParams.outputRepeatStride;
    tempArgs.isOutputScratch = (buffInfo_.outBuffType == BufferType::HCCL_BUFFER) ? 1 : 0;
    tempArgs.isInputOutputEqual = (tempArgs.inputAddr == tempArgs.outputAddr) ? 1 : 0;

    tempArgs.die0TailSize = templateDataParams.tailSize / tempArgs.kernelNum;
    tempArgs.die1TailSize = templateDataParams.tailSize - tempArgs.die0TailSize;
    tempArgs.isSliceSizeZero = (tempArgs.sliceSize == 0);
    
    return;
}

HcclResult CcuTempScatterNHR1DMem2Mem::FillKernelRunArgs(const KernalRunTempArgs &tempArgs, const TemplateDataParams &templateDataParams,
    const TemplateResource& templateResource) const
{
    for (uint32_t axisId = 0; axisId < tempArgs.kernelNum; axisId++) {
        if ((templateDataParams.tailSize == 0) && ((axisId == 0 && tempArgs.die0Size == 0) || (axisId == 1 && tempArgs.die1Size == 0))) {
            // 数据长度为0的kernel不下发
            continue;
        }
        std::vector<uint64_t> taskArgs = {
            tempArgs.inputAddr, tempArgs.outputAddr, tempArgs.scratchAddr, tempArgs.token,
            tempArgs.die0Size, tempArgs.die1Size,
            tempArgs.inputSliceStride, tempArgs.outputSliceStride,
            tempArgs.sliceSize * tempArgs.repeatNum,
            tempArgs.inputRepeatStride, tempArgs.outputRepeatStride,
            UINT64_MAX - tempArgs.repeatNum,
            tempArgs.isOutputScratch, tempArgs.isInputOutputEqual,
            tempArgs.die0TailSize, tempArgs.die1TailSize, tempArgs.isSliceSizeZero
        };
        uint64_t argSize = 17;
        CcuResult launchRet = HcommCcuKernelLaunch(templateResource.threads[axisId],
                                                   templateResource.ccuKernels[axisId],
                                                   taskArgs.data(), argSize);
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[CcuTempScatterNHR1DMem2Mem::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

void CcuTempScatterNHR1DMem2Mem::SaveSubmitInfo(const KernalRunTempArgs &tempArgs, TemplateResource& templateResource) const
{
    CcuKernelSubmitInfo submitInfo;
    submitInfo.cachedArgs[0]=buffInfo_.inBuffBaseOff;  // input、ouput只存对应的偏移
    submitInfo.cachedArgs[1]=buffInfo_.outBuffBaseOff;
    submitInfo.cachedArgs[2]=buffInfo_.hcclBuffBaseOff;
    submitInfo.cachedArgs[3]=tempArgs.token;
    submitInfo.cachedArgs[4]=tempArgs.die0Size;
    submitInfo.cachedArgs[5]=tempArgs.die1Size;
    submitInfo.cachedArgs[6]=tempArgs.inputSliceStride;
    submitInfo.cachedArgs[7]=tempArgs.outputSliceStride;
    submitInfo.cachedArgs[8]=tempArgs.sliceSize * tempArgs.repeatNum; // curScratchStride
    submitInfo.cachedArgs[9]=tempArgs.inputRepeatStride;
    submitInfo.cachedArgs[10]=tempArgs.outputRepeatStride;
    submitInfo.cachedArgs[11]=UINT64_MAX - tempArgs.repeatNum;
    submitInfo.cachedArgs[12]=tempArgs.isOutputScratch;
    submitInfo.cachedArgs[13]=tempArgs.isInputOutputEqual;
    submitInfo.cachedArgs[14]=tempArgs.die0TailSize;
    submitInfo.cachedArgs[15]=tempArgs.die1TailSize;
    submitInfo.cachedArgs[16]=tempArgs.isSliceSizeZero;
    for (u32 i = 0; i < tempArgs.kernelNum; i++) {
        // 2个kernel的TaskArg相同
        submitInfo.kernelHandle = templateResource.ccuKernels[i];
        templateResource.submitInfos.push_back(submitInfo);
    }
    return;
}

HcclResult CcuTempScatterNHR1DMem2Mem::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
                                                 TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] Template KernelRun start.");
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    KernalRunTempArgs tempArgs;

    tempArgs.kernelNum = templateResource.ccuKernels.size();

    if (templateDataParams.sliceSize == 0 && templateDataParams.tailSize == 0) {
        HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] sliceSize is 0, no need do, just success.");
        return HCCL_SUCCESS;
    }
    tempArgs.die0Size = 0;
    tempArgs.die1Size = 0;
    constexpr uint32_t MAX_DIE_NUM_2 = 2;
    if (tempArgs.kernelNum == MAX_DIE_NUM_2) {
        SplitDataFor2Dies(param, templateDataParams, tempArgs.die0Size, tempArgs.die1Size);
    } else {
        tempArgs.die0Size = templateDataParams.sliceSize;
    }
    FillKernelRunTempArgs(templateDataParams, tempArgs);
    tempArgs.token;
    CHK_RET(GetToken(buffInfo_, tempArgs.token));
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] dimSize[%llu], inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu],"
              "sliceSize[%llu], die0Size[%llu], die1Size[%llu], inputSliceStride[%llu], outputSliceStride[%llu],"
              "inputRepeatStride[%llu], outputRepeatStride[%llu], repeatNum[%llu], isOutputScratch[%llu], die0TailSize[%llu],"
              "die1TailSize[%llu]",
              templateRankSize_, tempArgs.inputAddr, tempArgs.outputAddr, tempArgs.scratchAddr, tempArgs.sliceSize, tempArgs.die0Size,
              tempArgs.die1Size, tempArgs.inputSliceStride, tempArgs.outputSliceStride, tempArgs.inputRepeatStride, tempArgs.outputRepeatStride, 
              tempArgs.repeatNum, tempArgs.isOutputScratch, tempArgs.isInputOutputEqual, tempArgs.die0TailSize, tempArgs.die1TailSize);

    // 前流同步
    if (tempArgs.kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }
    CHK_RET(FillKernelRunArgs(tempArgs, templateDataParams, templateResource));

    // 后流同步
    if (tempArgs.kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] Template Run for all steps Ends.");

    // 所有task下发完后再保存参数信息
    SaveSubmitInfo(tempArgs, templateResource);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterNHR1DMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
{
    u32 virtRankIdx = mySubCommRank_;
    std::vector<u32> ranks = subCommRanks_[0];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices       = 0;
    stepInfo.toRank        = templateRankSize_;
    stepInfo.fromRank      = templateRankSize_;
    stepInfo.step          = step;
    stepInfo.myRank        = virtRankIdx;
    uint32_t rootId        = subCommRootId_;
    u32      deltaRoot     = (rootId + templateRankSize_ - virtRankIdx) % templateRankSize_;
    u32      deltaRankPair = 1 << step;
    // 数据份数和数据编号增量
    u32 nSlices         = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1)); // 向上取整设计了下的
    u32 deltaSliceIndex = 1 << (step + 1);
    // 是否为2的幂
    u32  nRanks       = 0;
    bool isPowerOfTwo = (templateRankSize_ & (templateRankSize_ - 1)) == 0;
    if (!isPowerOfTwo && step == nSteps - 1) {
        nRanks = templateRankSize_ - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) { // 需要发
        u32 sendTo     = (virtRankIdx + templateRankSize_ - deltaRankPair) % templateRankSize_;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
        stepInfo.toRank = ranks[sendTo];
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom   = (virtRankIdx + deltaRankPair) % templateRankSize_;
        u32 rxSliceIdx = virtRankIdx;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
        stepInfo.fromRank = ranks[recvFrom];
        stepInfo.nSlices  = nSlices;
    }
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempScatterNHR1DMem2Mem::GetThreadNum() const
{
    constexpr uint32_t KERNEL_NUM_2 = 2;
    return KERNEL_NUM_2;
}

HcclResult CcuTempScatterNHR1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = 1;

    return HCCL_SUCCESS;
}

}  // namespace ops_hccl