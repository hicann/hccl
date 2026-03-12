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
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "ccu_kernel_scatter_nhr1d_mem2mem.h"
#include "ccu_temp_scatter_nhr1d_mem2mem.h"
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
    return 0;
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
            HCCL_INFO("[CcuTempReduceScatterNHR1DMem2Mem::GetDieNumFromChannelDescs] 2 channels on the same die, dieNum = 1.");
        } else {
            dieNum = LINK_NUM_2;
            HCCL_INFO("[CcuTempReduceScatterNHR1DMem2Mem::GetDieNumFromChannelDescs] 2 channels on 2 dies, dieNum = 2.");
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

        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
            return std::make_unique<CcuKernelScatterNHR1DMem2Mem>(arg);
        };
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgScatterNHRMem2Mem1D>(
            subCommRanks_[0].size(),
            mySubCommRank_,  // 子通信域rankid
            subCommRootId_, kernelIdx, enableDieNum, stepInfoVector, rank2ChannelIdx, param, subCommRanks_);
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

HcclResult CcuTempScatterNHR1DMem2Mem::KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
                                                 const TemplateResource &templateResource)
{
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] Template KernelRun start.");
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    u32 kernelNum = templateResource.ccuKernels.size();

    if (templateDataParams.sliceSize == 0) {
        HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] sliceSize is 0, no need do, just success.");
        return HCCL_SUCCESS;
    }
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
    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] buffInfo_.inputPtr [%p].", buffInfo_.inputPtr);
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] buffInfo_.inputSize [%llu].", buffInfo_.inputSize);
    uint64_t token = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                 static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t sliceSize = templateDataParams.sliceSize;
    uint64_t repeatNum = templateDataParams.repeatNum;
    uint64_t inputSliceStride = templateDataParams.inputSliceStride;
    uint64_t outputSliceStride = templateDataParams.outputSliceStride;
    uint64_t inputRepeatStride = templateDataParams.inputRepeatStride;
    uint64_t outputRepeatStride = templateDataParams.outputRepeatStride;
    uint64_t isOutputScratch = (buffInfo_.outBuffType == BufferType::HCCL_BUFFER) ? 1 : 0;
    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] dimSize[%llu], inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu],"
              "sliceSize[%llu], die0Size[%llu], die1Size[%llu], inputSliceStride[%llu],"
              "inputRepeatStride[%llu], outputRepeatStride[%llu], repeatNum[%llu], isOutputScratch[%llu]",
              templateRankSize_, inputAddr, outputAddr, scratchAddr, sliceSize, die0Size, die1Size, inputSliceStride,
              inputRepeatStride, outputRepeatStride, repeatNum, isOutputScratch);

    // 前流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxMainToSub(1, 0);
        
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub));
    }

    for (uint32_t axisId = 0; axisId < kernelNum; axisId++) {
        if ((axisId == 0 && die0Size == 0) || (axisId == 1 && die1Size == 0)) {
            // 数据长度为0的kernel不下发
            continue;
        }
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgScatterNHRMem2Mem1D>(
            inputAddr, outputAddr, scratchAddr, token, sliceSize, die0Size, die1Size, inputSliceStride,
            inputRepeatStride, outputRepeatStride, repeatNum, isOutputScratch);

        void *taskArgPtr = static_cast<void *>(taskArg.get());

        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[axisId],
                                    templateResource.ccuKernels[axisId], taskArgPtr));
    }

    // 后流同步
    if (kernelNum > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    HCCL_INFO("[CcuTempScatterNHR1DMem2Mem] Template Run for all steps Ends.");
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