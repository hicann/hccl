/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_nhr1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

constexpr uint16_t OUTPUT_XN_ID    = 0;
constexpr uint16_t TOKEN_XN_ID     = 1;
constexpr uint16_t POST_SYNC_ID     = 2;
constexpr uint16_t STEP_PRE_SYNC_ID = 3;
constexpr uint16_t STEP_POST_SYNC_ID= 4;

constexpr uint16_t CKE_IDX_0        = 0;

constexpr uint16_t LINK_SIZE        = 2;

constexpr uint16_t BIT_NUM_PER_CKE = 16;

CcuKernelAllGatherNHR1DMem2Mem::CcuKernelAllGatherNHR1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernel(arg)
{
    const CcuKernelArgAllGatherNHR1D *kernelArg = dynamic_cast<const CcuKernelArgAllGatherNHR1D *>(&arg);
    mySubCommRankId_                      = kernelArg->mySubCommRankId_;
    axisId_                               = kernelArg->axisId_;
    axisSize_                             = kernelArg->axisSize_;
    channels_                             = kernelArg->channels;
    dimSize_                              = kernelArg->dimSize_;
    stepInfoVector_                       = kernelArg->stepInfoVector_;
    rank2ChannelIdx_                      = kernelArg->rank2ChannelIdx_;
    localSize_                            = rank2ChannelIdx_.size();
    myRankIdx_                            = rank2ChannelIdx_.size();
    eventNum_                             = (dimSize_ + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE; // 每个CKE有16个bit

    HCCL_INFO(
        "[CcuKernelAllGatherNHR1DMem2Mem] Init, KernelArgs are mySubCommRankId[%u], axisId_[%u], axisSize_[%u], stepInfoVector_.size[%u], myRankIdx_[%u]",
        mySubCommRankId_, axisId_, axisSize_, stepInfoVector_.size(), myRankIdx_);
}

HcclResult CcuKernelAllGatherNHR1DMem2Mem::InitResource()
{
    die0Size_               = CreateVariable();
    die1Size_               = CreateVariable();
    inputSliceStride_       = CreateVariable();
    outputSliceStride_      = CreateVariable();
    inputRepeatStride_      = CreateVariable();
    outputRepeatStride_     = CreateVariable();
    repeatNum_              = CreateVariable();
    tmpCopyRepeatNum_       = CreateVariable();
    repeatTimeflag_         = CreateVariable();
    isInputOutputEqual_     = CreateVariable();
    myrankInputSliceOffset_ = CreateVariable();
    tmpSliceOffset_         = CreateVariable();
    for (u64 i = 0; i < dimSize_; i++) {
        outputSliceOffset_.push_back(CreateVariable());
    }
    constVar1_ = CreateVariable();
    constVar1_ = 1;

    localEvent_     = CreateCompletedEvent();
    localAxisEvent_ = CreateCompletedEvent();

    input_ = CreateVariable();
    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] mySubCommRankId[%u], channelId[%u]", mySubCommRankId_, channelIdx);
        CcuRep::Variable inputVar, scratchVar, tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &inputVar));
        output_.push_back(inputVar); // 获取channel中id=0的Var来传递output
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);
        channelIdx++;
    }
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    srcMem_ = CreateLocalAddr();
    dstMem_ = CreateRemoteAddr();
    localDst_ = CreateLocalAddr();
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMem2Mem] InitResources finished");

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherNHR1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[myRankIdx_]);
    Load(token_[myRankIdx_]);
    Load(die0Size_);
    Load(die1Size_);
    Load(repeatNum_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(isInputOutputEqual_);

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMem2Mem] LoadArgs run finished");
}

void CcuKernelAllGatherNHR1DMem2Mem::PreSync()
{
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PreSync start");
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[myRankIdx_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[myRankIdx_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PreSync end");
    return;
}

void CcuKernelAllGatherNHR1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);     
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PostSync run finished");
}

void CcuKernelAllGatherNHR1DMem2Mem::DoRepeatAllGatherNHR()
{
    tmpSliceOffset_         = 0;
    myrankInputSliceOffset_ = 0;
    for (u64 i = 0; i < mySubCommRankId_; i++) {
        myrankInputSliceOffset_ += inputSliceStride_;
    }
    for (u64 i = 0; i < dimSize_; i++) {
        outputSliceOffset_[i] = tmpSliceOffset_;
        tmpSliceOffset_ += outputSliceStride_;
    }
    srcMem_.addr = input_;
    srcMem_.addr += myrankInputSliceOffset_;
    srcMem_.token = token_[myRankIdx_];
    dstMem_.addr = output_[myRankIdx_];
    dstMem_.addr += outputSliceOffset_[mySubCommRankId_];
    dstMem_.token = token_[myRankIdx_];
    localDst_.addr  = output_[myRankIdx_];
    localDst_.addr += outputSliceOffset_[mySubCommRankId_];
    localDst_.token = token_[myRankIdx_];
    tmpCopyRepeatNum_ = repeatNum_;
    repeatTimeflag_   = 0;
    CCU_WHILE(tmpCopyRepeatNum_ != UINT64_MAX)
    {
        tmpCopyRepeatNum_ += constVar1_;
        CCU_IF(repeatTimeflag_ != 0)
        {
            srcMem_.addr += inputRepeatStride_;
            dstMem_.addr += outputRepeatStride_;
            localDst_.addr += outputRepeatStride_;
        }
        CCU_IF(repeatTimeflag_ == 0)
        {
            if (axisId_ == 1) {
                srcMem_.addr += die0Size_;
                dstMem_.addr += die0Size_;
                localDst_.addr += die0Size_;
            }
        }
        CCU_IF(isInputOutputEqual_ == 0)
        {
            localEvent_.SetMask(1 << mySubCommRankId_);
            LocalCopyNb(localDst_, srcMem_, axisId_ == 0 ? die0Size_ : die1Size_, localEvent_);
        }
        CCU_IF(isInputOutputEqual_ != 0)
        {
            localEvent_.SetMask(1 << mySubCommRankId_);
            RecordEvent(localEvent_);
        }
        localEvent_.SetMask(1 << mySubCommRankId_);
        WaitEvent(localEvent_);
        repeatTimeflag_ = 1;
    }

    for (auto &nhrStepInfo : stepInfoVector_) {
        DoRepeatAllGatherNHRSingleStep(nhrStepInfo);
    }
}

void CcuKernelAllGatherNHR1DMem2Mem::DoRepeatAllGatherNHRSingleStep(const NHRStepInfo                   &nhrStepInfo)
{
    u32                    &toRankIdx        = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32                    &fromRankIdx      = rank2ChannelIdx_[nhrStepInfo.fromRank];
    u32                     sendSliceIdx     = 0;
    ChannelHandle           sendChannel    = channels_[toRankIdx];
    ChannelHandle           recvChannel    = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    HCCL_INFO("sendSliceIdxList.size()[%zu]", sendSliceIdxList.size());
    srcMem_.token                            = token_[myRankIdx_];
    dstMem_.token                            = token_[toRankIdx];
    for (u32 i = 0; i < sendSliceIdxList.size(); i++) { ////这里写的可能有问题
        sendSliceIdx = sendSliceIdxList[i];
        if (i != 0) {
            if (i % BIT_NUM_PER_CKE == 0) {
                localEvent_.SetMask((1 << BIT_NUM_PER_CKE) - 1);
                WaitEvent(localEvent_);
            }
        }
        if (nhrStepInfo.step == 0) {
            srcMem_.addr = input_;
            srcMem_.addr += myrankInputSliceOffset_;
        } else {
            srcMem_.addr = output_[myRankIdx_];
            srcMem_.addr += outputSliceOffset_[sendSliceIdx];
        }
        dstMem_.addr = output_[toRankIdx];
        dstMem_.addr += outputSliceOffset_[sendSliceIdx];
        HCCL_INFO("mySubCommRankId[%zu], myRankIdx[%zu], toRankIdx[%zu], sendSliceIdx[%zu]", mySubCommRankId_, myRankIdx_, toRankIdx, sendSliceIdx);
        DoRepeatSendRecvSlices(nhrStepInfo.toRank, srcMem_, dstMem_, i % BIT_NUM_PER_CKE);
    }
    localEvent_.SetMask((1 << (sendSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1);
    WaitEvent(localEvent_);

    if (nhrStepInfo.step + 1 != stepInfoVector_.size()){
        uint16_t selfEventId = mySubCommRankId_ / BIT_NUM_PER_CKE;
        uint16_t selfBit      = 1 << (mySubCommRankId_ % BIT_NUM_PER_CKE);
        NotifyRecord(sendChannel, selfEventId + eventNum_ * CKE_IDX_0, selfBit);
        uint16_t recvEventId = nhrStepInfo.fromRank / BIT_NUM_PER_CKE;
        uint16_t recvBit      = 1 << (nhrStepInfo.fromRank % BIT_NUM_PER_CKE);
        NotifyRecord(recvChannel, recvEventId + eventNum_ * CKE_IDX_0, recvBit);
    }
}

void CcuKernelAllGatherNHR1DMem2Mem::DoRepeatSendRecvSlices(const u32 &toRank, hcomm::CcuRep::LocalAddr &src, hcomm::CcuRep::RemoteAddr &dst,
                                                      u32 signalIndex)
{
    ChannelHandle           sendChannel = channels_[rank2ChannelIdx_[toRank]];
    const CcuRep::Variable &sliceSize     = axisId_ == 0 ? die0Size_ : die1Size_;
    repeatTimeflag_                       = 0;

    CCU_WHILE(repeatNum_ != UINT64_MAX)
    {
        repeatNum_ += constVar1_;
        CCU_IF(repeatTimeflag_ == 1)
        {
            src.addr += inputRepeatStride_;
            dst.addr += outputRepeatStride_;
        }
        CCU_IF(repeatTimeflag_ == 0)
        {
            if (axisId_ == 1) {
                src.addr += die0Size_;
                dst.addr += die0Size_;
            }
        }
        localEvent_.SetMask(1 << signalIndex);
        WriteNb(sendChannel, dst, src, sliceSize, localEvent_);
        repeatTimeflag_ = 1;
    }
}

HcclResult CcuKernelAllGatherNHR1DMem2Mem::Algorithm()
{
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMem2Mem] AllgatherNHR1D run");
    InitResource();
    LoadArgs();
    PreSync();
    DoRepeatAllGatherNHR();
    PostSync();
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMem2Mem] AllgatherNHR1D end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherNHR1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherNHR1D *taskArg = dynamic_cast<const CcuTaskArgAllGatherNHR1D *>(&arg);
    uint64_t inputAddr          = taskArg->inputAddr_;
    uint64_t outputAddr         = taskArg->outputAddr_;
    uint64_t token              = taskArg->token_;
    uint64_t die0Size           = taskArg->die0Size_;
    uint64_t die1Size           = taskArg->die1Size_;
    uint64_t repeatNum          = UINT64_MAX - taskArg->repeatNum_;
    uint64_t inputSliceStride   = taskArg->inputSliceStride_;
    uint64_t outputSliceStride  = taskArg->outputSliceStride_;
    uint64_t inputRepeatStride  = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t isInputOutputEqual = taskArg->isInputOutputEqual_;

    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "die0Size[%llu], die1Size[%llu], repeatNum[%llu]"
              "inputSliceStride[%llu], outputSliceStride[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu]",
              inputAddr, outputAddr, die0Size, die1Size, repeatNum, inputSliceStride, outputSliceStride,
              inputRepeatStride, outputRepeatStride);
    return {inputAddr,          outputAddr,        token,
            die0Size,           die1Size,          repeatNum,
            inputSliceStride,   outputSliceStride, inputRepeatStride,
            outputRepeatStride, isInputOutputEqual};
}

} // namespace ops_hccl