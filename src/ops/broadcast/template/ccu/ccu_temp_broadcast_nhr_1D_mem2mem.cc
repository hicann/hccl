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
#include "ccu/ccu_temp_broadcast_nhr_1D_mem2mem.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

CcuTempBroadcastNHR1DMem2Mem::CcuTempBroadcastNHR1DMem2Mem(const OpParam& param, const u32 rankId,
                                                                   const std::vector<std::vector<u32>>& subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem] root[%u] rank[%u] myRank_[%u]", param.root, rankId, myRank_);
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
    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem] mySubCommRank[%u] subCommRootId[%u]", mySubCommRank_, subCommRootId_);
}

CcuTempBroadcastNHR1DMem2Mem::~CcuTempBroadcastNHR1DMem2Mem()
{
}

u64 CcuTempBroadcastNHR1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void) inBuffType;
    (void) outBuffType;
    return 0;
}

HcclResult CcuTempBroadcastNHR1DMem2Mem::GetDieNumFromChannelDescs(HcclComm comm, u32 &dieNum)
{
    constexpr u32 LINK_NUM_1 = 2;
    constexpr u32 LINK_NUM_2 = 2;
    auto firstElement = rankIdToChannelDesc_.begin();
    const std::vector<HcclChannelDesc>& firstVector = firstElement->second;
    if (firstVector.size() == 1) {
        dieNum = 1;
        return HCCL_SUCCESS;
    } else if (firstVector.size() == LINK_NUM_2) {
        // 检查2个channel是否在2个die上
        uint32_t dieId0 = 0;
        uint32_t dieId1 = 0;
        GetChannelDieId(comm, myRank_, firstVector[0], dieId0);
        GetChannelDieId(comm, myRank_, firstVector[1], dieId1);
        if (dieId0 == dieId1) {
            dieNum = LINK_NUM_1;
        } else {
            dieNum = LINK_NUM_2;
        }
        return HCCL_SUCCESS;
    } else {
        HCCL_ERROR("[CcuTempBroadcastNHR1DMem2Mem::CalcRes] get channelDescs fail: there are [] link to rank []",
                   firstVector.size(), firstElement->first);
        return HCCL_E_INTERNAL;
    }
}

HcclResult CcuTempBroadcastNHR1DMem2Mem::ProcessNHRStepInfo(HcclComm comm, std::vector<NHRStepInfo>& stepInfoVector,
                                                            std::map<u32, u32>& rank2ChannelIdx, u32 enableDieNum,
                                                            std::vector<std::vector<HcclChannelDesc>>& channelsPerDie)
{
    u32 nSteps = GetNHRStepNum(templateRankSize_) * 2; // 分为Scatter和AG两次NHR
    for (u32 step = 0; step < nSteps; step++) {
        NHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));
        HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][ProcessNHRStepInfo] myRank[%u] rank[%u] step[%u] Slices[%u] toRank[%u] fromRank[%u]",
                    myRank_, stepInfo.myRank, stepInfo.step, stepInfo.nSlices, stepInfo.toRank, stepInfo.fromRank);
        stepInfoVector.push_back(stepInfo);
        if (rank2ChannelIdx.count(stepInfo.fromRank) == 0 && stepInfo.rxSliceIdxs.size() != 0) {
            // 存储 rankid → channelIdx 的索引
            u32 curChannelIdx = channelsPerDie[0].size();
            rank2ChannelIdx[stepInfo.fromRank] = curChannelIdx;

            for (HcclChannelDesc channel: rankIdToChannelDesc_.at(stepInfo.fromRank)) {
                HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][ProcessNHRStepInfo]rx remoteRank[%u] rankIdToChannelDesc size[%zu]",
                            channel.remoteRank, rankIdToChannelDesc_.size());
                uint32_t dieId = 0;
                CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
                // 如果是2个die的算法，则分别加入到2个vector中，否则只加入到1个vector
                uint32_t vecIdx = dieId % enableDieNum;
                HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][ProcessNHRStepInfo]rx vecIdx[%u] curChannelIdx[%u]", vecIdx, curChannelIdx);
                // 限制只加入一个channel
                if (channelsPerDie[vecIdx].size() == curChannelIdx) {
                    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][ProcessNHRStepInfo]rx pushback");
                    channelsPerDie[vecIdx].push_back(channel);
                }
            }
        }
        if (rank2ChannelIdx.count(stepInfo.toRank) == 0 && stepInfo.txSliceIdxs.size() != 0) {
            u32 curChannelIdx = channelsPerDie[0].size();
            rank2ChannelIdx[stepInfo.toRank] = curChannelIdx;

            for (HcclChannelDesc channel: rankIdToChannelDesc_.at(stepInfo.toRank)) {
                HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][ProcessNHRStepInfo]tx remoteRank[%u] rankIdToChannelDesc size[%zu]",
                            channel.remoteRank, rankIdToChannelDesc_.size());
                u32 dieId = 0;
                CHK_RET(GetChannelDieId(comm, myRank_, channel, dieId));
                u32 vecIdx = dieId % enableDieNum;
                HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][ProcessNHRStepInfo]tx vecIdx[%u] curChannelIdx[%u]", vecIdx, curChannelIdx);
                if (channelsPerDie[vecIdx].size() == curChannelIdx) {
                    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][ProcessNHRStepInfo]tx pushback");
                    channelsPerDie[vecIdx].push_back(channel);
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastNHR1DMem2Mem::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                         AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, channelDescs));
    CHK_RET(RestoreChannelMap(channelDescs, rankIdToChannelDesc_));

    // 1.从获得的channelDesc，判断kernel发送到几个die上
    uint32_t enableDieNum = 0;
    CHK_RET(GetDieNumFromChannelDescs(comm, enableDieNum));

    constexpr uint32_t NUMONE = 1;
    constexpr uint32_t NUMTWO = 2;
    if (enableDieNum < NUMONE || enableDieNum > NUMTWO) { // 目前只支持1个或2个die
        HCCL_ERROR("[CcuTempBroadcastNHR1DMem2Mem::CalcRes] get channelDescs fail");
        return HCCL_E_INTERNAL;
    }

    uint32_t kernelNum = enableDieNum;
    resourceRequest.notifyNumOnMainThread = 1;
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.ccuKernelNum.push_back(kernelNum);
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG("[CcuTempBroadcastNHR1DMem2Mem::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 2.将channelDescs分到2个die
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    std::map<u32, u32> rank2ChannelIdx;
    std::vector<NHRStepInfo> stepInfoVector;
    channelsPerDie.resize(enableDieNum);

    CHK_RET(ProcessNHRStepInfo(comm, stepInfoVector, rank2ChannelIdx, enableDieNum, channelsPerDie));

    // 3.构造kernelInfo
    for (uint32_t kernelIdx = 0; kernelIdx < kernelNum; kernelIdx++) {
        // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
        CcuKernelInfo kernelInfo;

        kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                                return std::make_unique<CcuKernelBroadcastNhr1DMem2Mem>(arg);
                            };
        kernelInfo.kernelArg = std::make_shared<CcuKernelArgBroadcastNhr1DMem2Mem>(myRank_,  // 通信域全局rankid
                                                                                kernelIdx,
                                                                                enableDieNum,
                                                                                subCommRanks_[0],
                                                                                stepInfoVector, rank2ChannelIdx,
                                                                                param, subCommRanks_);
        kernelInfo.channels = channelsPerDie[kernelIdx];
        resourceRequest.ccuKernelInfos.push_back(kernelInfo);
        HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem::CalcRes] kernelIdx[%u] channels Size[%zu]", kernelIdx, kernelInfo.channels.size());
    }

    HCCL_DEBUG("[CcuTempBroadcastNHR1DMem2Mem::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastNHR1DMem2Mem::SplitDataFor2Dies(const OpParam& param,
                                                           const TemplateDataParams& templateDataParams,
                                                           uint64_t& die0Size, uint64_t& die1Size) const
{
    constexpr uint64_t MULTIPLIER = 4;
    uint64_t typeSize = DataTypeSizeGet(param.DataDes.dataType);
    uint64_t dataCount = (templateDataParams.sliceSize / typeSize);

    if (dataCount <= templateRankSize_ * MULTIPLIER) {   // 数据量极小，不划分die
        die0Size = dataCount * typeSize;
        die1Size = 0;
        return HCCL_SUCCESS;
    }
    u8 die0BWcoeff = 1;
    u8 die1BWcoeff = 1;

    // 查询带宽系数，按比例切分数据

    die0Size = (dataCount * die0BWcoeff / (die0BWcoeff + die1BWcoeff)) * typeSize;
    die1Size = templateDataParams.sliceSize - die0Size;
    return HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastNHR1DMem2Mem::KernelRun(const OpParam& param,
                                                       const TemplateDataParams& templateDataParams,
                                                       const TemplateResource& templateResource)
{
    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem] Template KernelRun start.");
    opMode_ = param.opMode;
    buffInfo_ = templateDataParams.buffInfo;
    u32 kernelNum = templateResource.ccuKernels.size();

    if (templateDataParams.sliceSize == 0) {
        HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem] sliceSize is 0, no need do, just success.");
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
    uint64_t token = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));
    uint64_t die0SliceSize = die0Size / templateRankSize_;
    uint64_t die0LastSliceSize   = die0Size % templateRankSize_ + die0SliceSize;
    uint64_t die1SliceSize = die1Size / templateRankSize_;
    uint64_t die1LastSliceSize   = die1Size % templateRankSize_ + die1SliceSize;
    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem] die0Size[%llu], die1Size[%llu], inputAddr[%llu],"\
        "outputAddr[%llu], die0SliceSize[%llu], die0LastSliceSize[%llu], die1SliceSize[%llu],"\
        "die1LastSliceSize[%llu]",
        die0Size, die1Size, inputAddr, outputAddr, die0SliceSize, die0LastSliceSize,
        die1SliceSize, die1LastSliceSize);

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
        std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgBroadcastNhr1DMem2Mem>(
            inputAddr, outputAddr, token, die0Size, die1Size, die0SliceSize, die1SliceSize, die0LastSliceSize,
            die1LastSliceSize);

        void* taskArgPtr = static_cast<void*>(taskArg.get());

        CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[axisId], templateResource.ccuKernels[axisId], taskArgPtr));
    }
    if (kernelNum > 1) {
        // 后流同步
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        std::vector<u32> notifyIdxSubToMain(1, 0);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain));
    }

    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem] Template Run for all steps Ends.");
    return HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastNHR1DMem2Mem::GetStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
{
    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][GetStepInfo] step[%u] nSteps[%u]", step, nSteps);
    u32 nStepsNHR = nSteps / 2;
    u32 realStep = step;
    if (realStep < nStepsNHR) {
        CHK_RET(GetScatterStepInfo(realStep, nStepsNHR, stepInfo));
    } else {
        realStep = step % nStepsNHR;
        CHK_RET(GetAllGatherStepInfo(realStep, nStepsNHR, stepInfo));
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastNHR1DMem2Mem::GetScatterStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
{
    u32 rankIdx = mySubCommRank_;
    std::vector<u32> ranks = subCommRanks_[0];
    u32 rankSize = templateRankSize_;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.step = step;
    stepInfo.myRank = mySubCommRank_;

    uint32_t rootId = subCommRootId_;
    u32 deltaRoot = (rootId + rankSize - rankIdx) % rankSize;
    u32 deltaRankPair = 1 << step;

    // 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    // 判断是否是2的幂
    u32 nRanks = 0; // 本步需要进行收/发的rank数
    bool isPerfect = (rankSize & (rankSize - 1)) == 0;
    if (!isPerfect && step == nSteps - 1) {
        nRanks = rankSize - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) { // 需要发
        u32 sendTo = (rankIdx + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.toRank = ranks[sendTo];
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (rankIdx + deltaRankPair) % rankSize;
        u32 rxSliceIdx = rankIdx;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.fromRank = ranks[recvFrom];
        stepInfo.nSlices = nSlices;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuTempBroadcastNHR1DMem2Mem::GetAllGatherStepInfo(u32 step, u32 nSteps, NHRStepInfo &stepInfo)
{
    u32 rankIdx = mySubCommRank_;
    std::vector<u32> ranks = subCommRanks_[0];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = mySubCommRank_;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = rankIdx;
    u32 rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = ranks[sendTo];
    stepInfo.fromRank = ranks[recvFrom];

    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][GetAllGatherStepInfo] rankIdx[%u] templateRankSize[%u] nSlices[%u] toRank[%u] sendTo[%u] fromRank[%u] recvFrom[%u]",
                rankIdx, templateRankSize_, nSlices, stepInfo.toRank, sendTo, stepInfo.fromRank, recvFrom);

    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[CcuTempBroadcastNHR1DMem2Mem][GetAllGatherStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HCCL_SUCCESS;
}

void CcuTempBroadcastNHR1DMem2Mem::SetRoot(u32 root)
{
    std::vector<u32> ranks = subCommRanks_[0];
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    HCCL_INFO("[CcuTempBroadcastNHR1DMem2Mem][SetRoot] myRank_ [%u], set root_ [%u] subCommRootId[%u]", myRank_, root, subCommRootId_);
}

u64 CcuTempBroadcastNHR1DMem2Mem::GetThreadNum() const
{
    u64 twoNum = 2;
    return twoNum;
}

HcclResult CcuTempBroadcastNHR1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = 1;

    return HCCL_SUCCESS;
}
} // namespace ops_hccl