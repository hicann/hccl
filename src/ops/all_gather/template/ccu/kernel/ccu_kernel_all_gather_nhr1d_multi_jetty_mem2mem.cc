/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_nhr1d_multi_jetty_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID       = 1;
constexpr uint16_t TOKEN_XN_ID        = 2;
constexpr uint16_t POST_SYNC_ID       = 3;
constexpr uint16_t STEP_PRE_SYNC_ID   = 4;
constexpr uint16_t STEP_POST_SYNC_ID  = 5;
constexpr uint16_t CKE_IDX_0          = 0; // 每个channel有单独cke(16个bit)，只需一个cke
constexpr uint16_t BIT_NUM_PER_CKE    = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

CcuKernelAllGatherNHR1DMultiJettyMem2Mem::CcuKernelAllGatherNHR1DMultiJettyMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem *kernelArg = dynamic_cast<const CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem *>(&arg);
    rankId_                               = kernelArg->rankId_;
    rankSize_                             = kernelArg->rankSize_; // 子通信域rank数
    channels_                             = kernelArg->channels;
    jettyNum_                             = kernelArg->jettyNum_;
    stepInfoVector_                       = kernelArg->stepInfoVector_;
    rank2ChannelIdx_                      = kernelArg->rank2ChannelIdx_;
    localSize_                            = rank2ChannelIdx_.size(); // nhr 算法通信rank数
    myRankIdx_                            = rank2ChannelIdx_.size();

    HCCL_INFO("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] kernelArg: rankId_[%u], rankSize_[%u], localSize_[%u], jettyNum_[%u]",
              rankId_, rankSize_, localSize_, jettyNum_);
}

HcclResult CcuKernelAllGatherNHR1DMultiJettyMem2Mem::InitResources()
{
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    sliceSize_              = CreateVariable();
    sliceSizePerJetty_      = CreateVariable();
    lastSliceSizePerJetty_  = CreateVariable();
    repeatNumInv_           = CreateVariable();
    inputSliceStride_       = CreateVariable();
    outputSliceStride_      = CreateVariable();
    inputRepeatStride_      = CreateVariable();
    outputRepeatStride_     = CreateVariable();
    tmpCopyRepeatNumInv_    = CreateVariable();
    repeatTimeflag_         = CreateVariable();
    isInputOutputEqual_     = CreateVariable();
    myrankInputSliceOffset_ = CreateVariable();
    tmpSliceOffset_         = CreateVariable();
    groupOpSize_            = CreateGroupOpSize();

    for (u64 i = 0; i < rankSize_; i++) {
        outputSliceOffset_.push_back(CreateVariable());
    }
    constVar1_ = CreateVariable();
    constVar1_ = 1;

    selfBit_     = 1 << rankId_;

    input_       = CreateVariable();
    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] MyRank[%u], channelIdx[%u]", rankId_, channelIdx);
        CcuRep::Variable outputVar;
        CcuRep::Variable tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
        output_.push_back(outputVar);
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);
    }
    output_.push_back(CreateVariable()); // 将本端数据加在末尾
    token_.push_back(CreateVariable());

    srcMem_    = CreateLocalAddr();
    dstMem_    = CreateRemoteAddr();
    srcMemTmp_ = CreateLocalAddr();
    dstMemTmp_ = CreateRemoteAddr();
    myDstMem_  = CreateLocalAddr(); // 用于本地拷贝
    event_     = CreateCompletedEvent();

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] InitResources finished");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherNHR1DMultiJettyMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[myRankIdx_]);
    Load(token_[myRankIdx_]);
    Load(sliceSize_);
    Load(sliceSizePerJetty_);
    Load(lastSliceSizePerJetty_);
    Load(repeatNumInv_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(isInputOutputEqual_);
    Load(groupOpSize_);
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] LoadArgs run finished");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherNHR1DMultiJettyMem2Mem::PreSync()
{
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PreSync start");
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[localSize_], 1 << OUTPUT_XN_ID)); // 同步并置位远端
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[localSize_], 1 << TOKEN_XN_ID));
    }
    
    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyWait(channel, CKE_IDX_0, allBit));
    }
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PreSync end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherNHR1DMultiJettyMem2Mem::PostSync()
{
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PostSync start");
    for (auto &ch : channels_) {
        CHK_RET(NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }

    for (auto &ch : channels_) {
        CHK_RET(NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PostSync end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherNHR1DMultiJettyMem2Mem::DoRepeatAllGatherNHR()
{
    tmpSliceOffset_         = 0;
    myrankInputSliceOffset_ = 0;
    for (u64 i = 0; i < rankId_; i++) {
        myrankInputSliceOffset_ += inputSliceStride_;
    }
    for (u64 i = 0; i < rankSize_; i++) {
        outputSliceOffset_[i] = tmpSliceOffset_;
        tmpSliceOffset_ += outputSliceStride_;
    }
    srcMem_.addr         = input_;
    srcMem_.addr        += myrankInputSliceOffset_;
    myDstMem_.addr       = output_[myRankIdx_];
    myDstMem_.addr      += outputSliceOffset_[rankId_];
    srcMem_.token        = token_[myRankIdx_];
    myDstMem_.token      = token_[myRankIdx_];
    tmpCopyRepeatNumInv_ = repeatNumInv_;
    repeatTimeflag_      = 0;

    CCU_WHILE(tmpCopyRepeatNumInv_ != UINT64_MAX)
    {
        tmpCopyRepeatNumInv_ += constVar1_;
        CCU_IF(repeatTimeflag_ != 0)
        {
            srcMem_.addr    += inputRepeatStride_;
            myDstMem_.addr  += outputRepeatStride_;
        }
        event_.SetMask(1 << rankId_);
        CCU_IF(isInputOutputEqual_ == 0)
        {
            CHK_RET(GroupCopy(myDstMem_, srcMem_, groupOpSize_));
            CHK_RET(RecordEvent(event_));
        }
        CCU_IF(isInputOutputEqual_ != 0)
        {
            CHK_RET(RecordEvent(event_));
        }
        event_.SetMask(1 << rankId_);
        CHK_RET(WaitEvent(event_));
        repeatTimeflag_ = 1;
    }

    for (auto &nhrStepInfo : stepInfoVector_) {
        CHK_RET(DoRepeatAllGatherNHRSingleStep(nhrStepInfo));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherNHR1DMultiJettyMem2Mem::DoRepeatAllGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo)
{
    u32                    &toRankIdx        = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32                    &fromRankIdx      = rank2ChannelIdx_[nhrStepInfo.fromRank];
    u32                    sendSliceIdx      = 0;
    ChannelHandle          &sendChannel      = channels_[toRankIdx];
    ChannelHandle          &recvChannel      = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    srcMem_.token                            = token_[myRankIdx_];
    dstMem_.token                            = token_[toRankIdx];

    CHK_RET(NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID)); // 通知fromrank可以写入
    CHK_RET(NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID)); // 等待torank准备好

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];
        srcMem_.addr = output_[myRankIdx_];
        srcMem_.addr += outputSliceOffset_[sendSliceIdx];
        dstMem_.addr = output_[toRankIdx];
        dstMem_.addr += outputSliceOffset_[sendSliceIdx];
        
        repeatTimeflag_ = 0;
        tmpCopyRepeatNumInv_ = repeatNumInv_;
        CCU_WHILE(tmpCopyRepeatNumInv_ != UINT64_MAX)
        {
            tmpCopyRepeatNumInv_ += constVar1_;
            CCU_IF(repeatTimeflag_ == 1)
            {
                srcMem_.addr += inputRepeatStride_;
                dstMem_.addr += outputRepeatStride_;
            }
            CHK_RET(DoSendRecvSlices(nhrStepInfo.toRank, srcMem_, dstMem_));
            repeatTimeflag_ = 1;
        }
    }
    CHK_RET(NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID)); // 通知torank已写完
    CHK_RET(NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID)); // 等待fromrank写完
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherNHR1DMultiJettyMem2Mem::DoSendRecvSlices(const uint32_t &toRank, const CcuRep::LocalAddr &srcMem,
                                                                const CcuRep::RemoteAddr &dstMem)
{
    ChannelHandle      &sendChannel     = channels_[rank2ChannelIdx_[toRank]];
    srcMemTmp_             = srcMem;
    dstMemTmp_             = dstMem;

    CCU_IF(sliceSizePerJetty_ != 0)
    {
        for (uint32_t i = 0; i < jettyNum_ - 1; ++i) {
            event_.SetMask(1 << i);
            CHK_RET(WriteNb(sendChannel, dstMemTmp_, srcMemTmp_, sliceSizePerJetty_, event_));
            srcMemTmp_.addr += sliceSizePerJetty_;
            dstMemTmp_.addr += sliceSizePerJetty_;
        }
    }
    CCU_IF(sliceSizePerJetty_ == 0)
    {
        for (uint32_t i = 0; i < jettyNum_ - 1; ++i) {
            event_.SetMask(1 << i);
            CHK_RET(RecordEvent(event_));
        }
    }
    CCU_IF(lastSliceSizePerJetty_ != 0)
    {
        event_.SetMask(1 << (jettyNum_ - 1));
        CHK_RET(WriteNb(sendChannel, dstMemTmp_, srcMemTmp_, lastSliceSizePerJetty_, event_));
    }
    CCU_IF(lastSliceSizePerJetty_ == 0)
    {
        event_.SetMask(1 << (jettyNum_ - 1));
        CHK_RET(RecordEvent(event_));
    }

    uint16_t sendBit = (1 << jettyNum_) - 1;
    event_.SetMask(sendBit);
    CHK_RET(WaitEvent(event_));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllGatherNHR1DMultiJettyMem2Mem::Algorithm()
{
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] AllGatherNHR1DMultiJettyMem2Mem run");

    InitResources();
    LoadArgs();
    PreSync();
    DoRepeatAllGatherNHR();
    PostSync();

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] AllGatherNHR1DMultiJettyMem2Mem end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherNHR1DMultiJettyMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherNHR1DMultiJettyMem2Mem *taskArg = dynamic_cast<const CcuTaskArgAllGatherNHR1DMultiJettyMem2Mem *>(&arg);

    uint64_t inputAddr               = taskArg->inputAddr_;
    uint64_t outputAddr              = taskArg->outputAddr_;
    uint64_t token                   = taskArg->token_;
    uint64_t sliceSize               = taskArg->sliceSize_;
    uint64_t sliceSizePerJetty       = taskArg->sliceSizePerJetty_;
    uint64_t lastSliceSizePerJetty   = taskArg->lastSliceSizePerJetty_;
    uint64_t repeatNumInv            = taskArg->repeatNumInv_;
    uint64_t inputSliceStride        = taskArg->inputSliceStride_;
    uint64_t outputSliceStride       = taskArg->outputSliceStride_;
    uint64_t inputRepeatStride       = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride      = taskArg->outputRepeatStride_;
    uint64_t isInputOutputEqual      = taskArg->isInputOutputEqual_;
    auto     goSize                  = CalGoSize(sliceSize);

    HCCL_INFO("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
    "token[%llu], sliceSize[%llu], sliceSizePerJetty[%llu], lastSliceSizePerJetty[%llu], repeatNumInv[%llu],"
    "inputSliceStride[%llu], outputSliceStride[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], isInputOutputEqual[%llu]",
    inputAddr, outputAddr, token, sliceSize, sliceSizePerJetty, lastSliceSizePerJetty, repeatNumInv, inputSliceStride, 
    outputSliceStride, inputRepeatStride, outputRepeatStride, isInputOutputEqual);

    return {inputAddr,          outputAddr,          token,
            sliceSize,          sliceSizePerJetty,   lastSliceSizePerJetty,
            repeatNumInv,       inputSliceStride,    outputSliceStride,
            inputRepeatStride,  outputRepeatStride,  isInputOutputEqual,
            goSize[0],          goSize[1],           goSize[2],
            goSize[3]};
}
} // namespace Hccl
