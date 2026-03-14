/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_broadcast_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr uint16_t OUTPUT_XN_ID     = 0;
constexpr uint16_t TOKEN_XN_ID      = 1;
constexpr uint16_t POST_SYNC_ID     = 2;
constexpr uint16_t STEP_PRE_SYNC_ID = 3;
constexpr uint16_t STEP_POST_SYNC_ID= 4;

constexpr uint16_t CKE_IDX_0        = 0;
constexpr uint16_t RANK_NUM_PER_CKE = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

CcuKernelBroadcastNhr1DMem2Mem::CcuKernelBroadcastNhr1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgBroadcastNhr1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgBroadcastNhr1DMem2Mem *>(&arg);
    rankId_         = kernelArg->rankId_;
    axisId_         = kernelArg->axisId_;
    axisSize_       = kernelArg->axisSize_;
    dimSize_        = kernelArg->dimSize_.size();
    stepInfoVector_ = kernelArg->stepInfoVector_;
    rank2ChannelIdx_= kernelArg->rank2ChannelIdx_;
    localSize_      = rank2ChannelIdx_.size();
    myRankIdx_      = rank2ChannelIdx_.size();
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] CtxArg: rankId_[%u], axisId_[%u], axisSize_[%u], dimSize_[%u], stepInfoVectorSize[%zu], "
              "localSize_[%zu], dataType[%u] channelsSize[%zu]",
              rankId_, axisId_, axisSize_, dimSize_, stepInfoVector_.size(), localSize_, dataType_, channels_.size());
}

HcclResult CcuKernelBroadcastNhr1DMem2Mem::InitResources()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelBroadcastNhr1DMem2Mem] channels is empty!");
        return HCCL_E_INTERNAL;
    }

    input_ = CreateVariable();
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        HCCL_DEBUG("[CcuKernelBroadcastNhr1DMem2Mem] MyRank[%u], ChannelId[%u]", rankId_, channelIdx);
        CcuRep::Variable outputVar, tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
        output_.push_back(outputVar);
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);
    }
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    die0Size_           = CreateVariable();
    die1Size_           = CreateVariable();
    die0SliceSize_      = CreateVariable();
    die1SliceSize_      = CreateVariable();
    die0LastSliceSize_  = CreateVariable();
    die1LastSliceSize_  = CreateVariable();

    localSrc_ = CreateLocalAddr();
    localDst_   = CreateLocalAddr();
    remoteDst_  = CreateRemoteAddr();
    CcuRep::Variable tmpSliceOffset   = CreateVariable();
    tmpSliceOffset                           = 0;
    for (u64 i = 0; i < dimSize_; i++) {
        sliceOffset_.push_back(CreateVariable());
        sliceOffset_[i] = tmpSliceOffset;
        tmpSliceOffset += axisId_ == 0? die0SliceSize_: die1SliceSize_;
    }

    event_ = CreateCompletedEvent();
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem][InitResources] InitResources end");
    return HCCL_SUCCESS;
}

void CcuKernelBroadcastNhr1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[myRankIdx_]);
    Load(token_[myRankIdx_]);
    Load(die0Size_);
    Load(die1Size_);
    Load(die0SliceSize_);
    Load(die1SliceSize_);
    Load(die0LastSliceSize_);
    Load(die1LastSliceSize_);
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] LoadArgs run finished");
    return;
}

void CcuKernelBroadcastNhr1DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[myRankIdx_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[myRankIdx_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] BroadcastNhr1D wait all end");
    return;
}

void CcuKernelBroadcastNhr1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] BroadcastNhr1D groupwait end");
    return;
}

void CcuKernelBroadcastNhr1DMem2Mem::DoScatterNHR()
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = 0; i < stepInfoVector_.size() / NHR_NUM; i++) {
        const NHRStepInfo &nhrStepInfo = stepInfoVector_[i];
        DoScatterNHRSingleStep(nhrStepInfo);
    }
}

void CcuKernelBroadcastNhr1DMem2Mem::DoScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo)
{
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;
    const std::vector<u32> &recvSliceIdxList  = nhrStepInfo.rxSliceIdxs;
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem][DoScatterNHRSingleStep] sendSliceIdxListSize[%zu] recvSliceIdxList[%zu] "
                "step[%u] myRank[%u] nSlices[%u] toRank[%u] fromRank[%u]", sendSliceIdxList.size(), recvSliceIdxList.size(),
                nhrStepInfo.step, nhrStepInfo.myRank, nhrStepInfo.nSlices, nhrStepInfo.toRank, nhrStepInfo.fromRank);
    // 只需要发
    if(sendSliceIdxList.size() != 0){
        u32& toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank];
        u32  sendSliceIdx = 0;
        ChannelHandle sendChannel             = channels_[toRankIdx];
        localSrc_.token                       = token_[myRankIdx_];
        remoteDst_.token                       = token_[toRankIdx];
        localDst_.token                        = token_[toRankIdx];
        for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
            sendSliceIdx = sendSliceIdxList[i];
            HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem][DoScatterNHRSingleStep] sendSliceIdx[%u]", sendSliceIdx);
            if (i != 0) {
                if (i % RANK_NUM_PER_CKE == 0) {
                    event_.SetMask((1 << RANK_NUM_PER_CKE) - 1);
                    WaitEvent(event_);
                }
            }
            if (nhrStepInfo.step == 0) {
            // 只有第0步的源数据从input中取
                localSrc_.addr = input_;
                localSrc_.addr += sliceOffset_[sendSliceIdx];
            } else {
                localSrc_.addr = output_[myRankIdx_];
                localSrc_.addr += sliceOffset_[sendSliceIdx];
            }
            remoteDst_.addr = output_[toRankIdx];
            remoteDst_.addr += sliceOffset_[sendSliceIdx];
            localDst_.addr = output_[toRankIdx];
            localDst_.addr += sliceOffset_[sendSliceIdx];
            DoSendRecvSlice(nhrStepInfo.toRank, localSrc_, remoteDst_, sendSliceIdx, i % RANK_NUM_PER_CKE);
        }
        event_.SetMask((1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
        WaitEvent(event_);
        // 通知toRank数据写入完毕
        NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);
    }
    //只需要收
    if(recvSliceIdxList.size() != 0){
        u32& fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
        ChannelHandle recvChannel = channels_[fromRankIdx];
        NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);
    }
}

void CcuKernelBroadcastNhr1DMem2Mem::DoSendRecvSlice(const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst,
                                                     const u32 &sendSliceIdx, u32 signalIndex)
{
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem][DoSendRecvSlice] toRank[%u] sendSliceIdx[%u] signalIndex[%u]",
                toRank, sendSliceIdx, signalIndex);
    ChannelHandle sendChannel = channels_[rank2ChannelIdx_[toRank]];
    bool          islastSlice;

    // 添加 die1 偏移
    if (axisId_ == 1) {
        src.addr += die0Size_;
        dst.addr += die0Size_;
        localDst_.addr += die0Size_;
    }

    islastSlice = (sendSliceIdx + 1 == dimSize_);
    const CcuRep::Variable &sliceSize = axisId_ == 0? (islastSlice? die0LastSliceSize_ : die0SliceSize_)
                                                    : (islastSlice? die1LastSliceSize_ : die1SliceSize_);
    event_.SetMask(1 << signalIndex);
    CCU_IF(sliceSize != 0)
    {
        WriteNb(sendChannel, dst, src, sliceSize, event_);
    }
    CCU_IF(sliceSize == 0)
    {
        RecordEvent(event_);
    }
}

void CcuKernelBroadcastNhr1DMem2Mem::DoAllGatherNHR()
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = stepInfoVector_.size() / NHR_NUM; i < stepInfoVector_.size(); i++) {
        const NHRStepInfo &nhrStepInfo = stepInfoVector_[i];
        DoAllGatherNHRSingleStep(nhrStepInfo);
    }
}

void CcuKernelBroadcastNhr1DMem2Mem::DoAllGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo)
{
    u32& toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32& fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
    u32  sendSliceIdx = 0;
    ChannelHandle sendChannel = channels_[toRankIdx];
    ChannelHandle recvChannel = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;
    localSrc_.token                        = token_[myRankIdx_];
    remoteDst_.token                       = token_[toRankIdx];
    localDst_.token                        = token_[toRankIdx];

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % RANK_NUM_PER_CKE == 0) {
                event_.SetMask((1 << RANK_NUM_PER_CKE) - 1);
                WaitEvent(event_);
            }
        }

        localSrc_.addr = output_[myRankIdx_];
        localSrc_.addr += sliceOffset_[sendSliceIdx];

        remoteDst_.addr = output_[toRankIdx];
        remoteDst_.addr += sliceOffset_[sendSliceIdx];
        localDst_.addr  = output_[toRankIdx];
        localDst_.addr  += sliceOffset_[sendSliceIdx];
        DoSendRecvSlice(nhrStepInfo.toRank, localSrc_, remoteDst_, sendSliceIdx, i % RANK_NUM_PER_CKE);
    }

    event_.SetMask((1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
    WaitEvent(event_);

    if (nhrStepInfo.step + 1 != stepInfoVector_.size()) {   // 最后一步不需要同步
        // 通知toRank，写入完毕
        NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
        // 等待fromRank通知写入完毕
        NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
    }

    HCCL_DEBUG("[CcuKernelBroadcastNhr1DMem2Mem][DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu toRankIdx=%u, fromRankIdx=%u",
                rankId_, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size(), toRankIdx, fromRankIdx);
}

HcclResult CcuKernelBroadcastNhr1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] BroadcastNHR1D run");

    CHK_RET(InitResources());
    LoadArgs();

    PreSync();
    DoScatterNHR();
    DoAllGatherNHR();
    PostSync();

    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] BroadcastNHR1D end");
    return HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelBroadcastNhr1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgBroadcastNhr1DMem2Mem *taskArg = dynamic_cast<const CcuTaskArgBroadcastNhr1DMem2Mem *>(&arg);
    uint64_t inputAddr          = taskArg->inputAddr_;
    uint64_t outputAddr         = taskArg->outputAddr_;
    uint64_t token              = taskArg->token_;
    uint64_t die0Size           = taskArg->die0Size_;
    uint64_t die1Size           = taskArg->die1Size_;
    uint64_t die0SliceSize      = taskArg->die0SliceSize_;
    uint64_t die1SliceSize      = taskArg->die1SliceSize_;
    uint64_t die0LastSliceSize  = taskArg->die0LastSliceSize_;
    uint64_t die1LastSliceSize  = taskArg->die1LastSliceSize_;
    std::vector<uint64_t> taskArgs                     = {
        inputAddr,
        outputAddr,
        token,
        die0Size,
        die1Size,
        die0SliceSize,
        die1SliceSize,
        die0LastSliceSize,
        die1LastSliceSize,
    };

    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "die0Size[%llu], die1Size[%llu], die0SliceSize[%llu], die1SliceSize[%llu],"
              "die0LastSliceSize[%llu], die1LastSliceSize[%llu]",
              inputAddr, outputAddr, die0Size, die1Size, die0SliceSize, die1SliceSize,
              die0LastSliceSize, die1LastSliceSize);
    return taskArgs;
}

} // namespace ops_hccl