/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

// 按信号功能划分notify的bit
constexpr uint16_t INPUT_XN_ID      = 0;
constexpr uint16_t TOKEN_XN_ID      = 1;
constexpr uint16_t POST_SYNC_ID     = 2;
constexpr uint16_t STEP_PRE_SYNC_ID = 3;
constexpr uint16_t STEP_POST_SYNC_ID= 4;

constexpr uint16_t CKE_IDX_0        = 0;

constexpr uint16_t LINK_SIZE        = 2;

CcuKernelReduceScatterNHR1DMem2Mem::CcuKernelReduceScatterNHR1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterNHR1D *kernelArg = dynamic_cast<const CcuKernelArgReduceScatterNHR1D *>(&arg);
    mySubCommRankId_= kernelArg->mySubCommRankId_;   // 虚拟rankid，用于获取本rank对应的输入偏移
    axisId_         = kernelArg->axisId_;
    channels_       = kernelArg->channels;
    dimSize_        = kernelArg->dimSize_;
    stepInfoVector_ = kernelArg->stepInfoVector_;
    rank2ChannelIdx_= kernelArg->rank2ChannelIdx_;
    localSize_      = rank2ChannelIdx_.size();
    myRankIdx_      = rank2ChannelIdx_.size();
    reduceOp_       = kernelArg->opParam_.reduceType;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    axisSize_       = kernelArg->axisSize_;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG("[CcuKernelReduceScatterMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
                   outputDataType_);
    }
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] KernelArg: mySubCommRankId_[%u], myRankIdx_[%d], axisId_[%u], dimSize_[%u], localSize_[%u], "
              "dataType[%d], outputDataType[%d], reduceOp[%d]",
              mySubCommRankId_, myRankIdx_, axisId_, dimSize_, localSize_, dataType_, outputDataType_, reduceOp_);
}

void CcuKernelReduceScatterNHR1DMem2Mem::LoadArgs()
{
    Load(input_[myRankIdx_]);
    Load(output_);
    Load(token_[myRankIdx_]);
    Load(die0Size_);
    Load(die1Size_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(repeatNumVar_);
    repeatNumVarTemp_ = repeatNumVar_;
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] LoadArgs run finished");
}

HcclResult CcuKernelReduceScatterNHR1DMem2Mem::InitResources()
{
    die0Size_           = CreateVariable();
    die1Size_           = CreateVariable();
    sliceSize_          = CreateVariable();
    inputSliceStride_   = CreateVariable();
    outputSliceStride_  = CreateVariable();
    inputRepeatStride_  = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    event_              = CreateCompletedEvent();
    repeatNumVar_       = CreateVariable();
    repeatNumVarTemp_   = CreateVariable();

    output_ = CreateVariable();
    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] mySubCommRankId[%u], channelId[%u]", mySubCommRankId_, channelIdx);
        
        CcuRep::Variable inputVar, tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
        input_.push_back(inputVar); // 获取channel中id=0的Var来传递output
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);
    }
    input_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    repeatInputOffset_      = CreateVariable();
    repeatOutputOffset_     = CreateVariable();
    myrankInputSliceOffset_ = CreateVariable();

    localSrc_     = CreateLocalAddr();
    localDst_     = CreateLocalAddr();
    remoteDst_    = CreateRemoteAddr();
    isRepeatIter_ = CreateVariable();
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] InitResources finished");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceScatterNHR1DMem2Mem::PreSync()
{
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] PreSync start");
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[myRankIdx_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[myRankIdx_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] PreSync end");
    return;
}

void CcuKernelReduceScatterNHR1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);     
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] PostSync run finished");
}

void CcuKernelReduceScatterNHR1DMem2Mem::DoRepeatReduceScatterNHR()
{
    CcuRep::Variable tmpSliceOffset   = CreateVariable();
    tmpSliceOffset                    = 0;
    // 用来记录每个rank要读取的rank的sliceIdx的偏移
    // 后面会用inputAddr来加上这个偏移获取sliceIdx的地址
    std::vector<CcuRep::Variable> inputSliceOffset;
    for (u64 i = 0; i < dimSize_; i++) {
        inputSliceOffset.push_back(CreateVariable());
        inputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += inputSliceStride_;
    }

    for (auto &nhrStepInfo : stepInfoVector_) {
        DoRepeatReduceScatterNHRSingleStep(nhrStepInfo, inputSliceOffset);
    }
    // 因为所有的修改都是在input上进行的，所以最后需要把input上的数据搬到output上
    localSrc_.addr  = input_[myRankIdx_];
    localSrc_.addr += inputSliceOffset[mySubCommRankId_];
    localSrc_.token = token_[myRankIdx_];
    localDst_.addr  = output_;
    localDst_.token = token_[myRankIdx_];

    CcuRep::Variable repeatNumAdd2 = CreateVariable();
    repeatNumAdd2  = 1;
    CCU_WHILE(repeatNumVar_ != UINT64_MAX) {
        repeatNumVar_ += repeatNumAdd2;
        CCU_IF(isRepeatIter_ == 1) {
            localSrc_.addr += inputRepeatStride_;
            localDst_.addr += outputRepeatStride_;
        }
        CCU_IF(isRepeatIter_ == 0) {
            if (axisId_ == 1) {
                localSrc_.addr += die0Size_;
                localDst_.addr += die0Size_;
            }
        }
        CcuRep::Variable &localSliceSize = (axisId_ == 0) ? die0Size_ : die1Size_;
        event_.SetMask(1);
        LocalCopyNb(localDst_, localSrc_, localSliceSize, event_);
        WaitEvent(event_);
        isRepeatIter_ = 1;
    }
}

void CcuKernelReduceScatterNHR1DMem2Mem::DoRepeatReduceScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo,
    const std::vector<CcuRep::Variable> &inputSliceOffset)
{
    u32& toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32& fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
    ChannelHandle sendChannel = channels_[toRankIdx];
    ChannelHandle recvChannel = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    remoteDst_.token = token_[toRankIdx];
    localSrc_.token = token_[myRankIdx_];

    // 通知fromRank，可以写入
    NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);

    // 等待toRank通知其可以写入
    NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);

    for (const u32 &sendSliceIdx : sendSliceIdxList) {
        remoteDst_.addr = input_[toRankIdx];
        remoteDst_.addr += inputSliceOffset[sendSliceIdx];
        localSrc_.addr = input_[myRankIdx_];
        localSrc_.addr += inputSliceOffset[sendSliceIdx];
        DoRepeatWriteReduceSlices(nhrStepInfo.toRank, localSrc_, remoteDst_);
    }

    // 通知toRank数据写入完毕
    NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);

    // 等待fromRank通知数据写入完毕
    NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
}

void CcuKernelReduceScatterNHR1DMem2Mem::DoRepeatWriteReduceSlices(const u32 &toRank, CcuRep::LocalAddr &src,
                                                                 CcuRep::RemoteAddr &dst)
{
    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd  = 1;
    isRepeatIter_ = 0;
    ChannelHandle sendChannel = channels_[rank2ChannelIdx_[toRank]];
    repeatNumVarTemp_ = repeatNumVar_;
    CCU_WHILE(repeatNumVarTemp_ != UINT64_MAX) {
        CCU_IF(repeatNumVarTemp_ != UINT64_MAX) {
            repeatNumVarTemp_ += repeatNumAdd;
        }
        
        CCU_IF(isRepeatIter_ == 1) {
            src.addr += inputRepeatStride_;
            dst.addr += inputRepeatStride_;
        }
        CCU_IF(isRepeatIter_ == 0) {
            if (axisId_ == 1) {
                src.addr += die0Size_;
                dst.addr += die0Size_;
            }
        }
        sliceSize_ =  (axisId_ == 0) ? die0Size_ : die1Size_;
        event_.SetMask(1);
        WriteReduceNb(sendChannel, dst, src, sliceSize_, dataType_,
                    reduceOp_, event_);
        WaitEvent(event_);
        isRepeatIter_ = 1;
    }
    isRepeatIter_ = 0;
}

HcclResult CcuKernelReduceScatterNHR1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] CcuKernelReduceScatterNHR1DMem2Mem run.");
    CHK_RET(InitResources());
    LoadArgs();
    PreSync();
    DoRepeatReduceScatterNHR();
    PostSync();

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] CcuKernelReduceScatterNHR1DMem2Mem end.");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterNHR1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterNHR1D *taskArg = dynamic_cast<const CcuTaskArgReduceScatterNHR1D *>(&arg);
    // input & output & buffer地址
    uint64_t inputAddr          = taskArg->inputAddr_;
    uint64_t outputAddr         = taskArg->outputAddr_;
    uint64_t token              = taskArg->token_;
    uint64_t die0Size           = taskArg->die0Size_;
    uint64_t die1Size           = taskArg->die1Size_;
    uint64_t inputSliceStride   = taskArg->inputSliceStride_;
    uint64_t outputSliceStride  = taskArg->outputSliceStride_;
    uint64_t inputRepeatStride  = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t repeatNumVar       = UINT64_MAX - taskArg->repeatNum_;

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu],"
              "die0Size[%llu], die1Size[%llu],"
              "inputSliceStride[%llu], outputSliceStride[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu]",
              inputAddr, outputAddr, die0Size, die1Size,
              inputSliceStride, outputSliceStride, inputRepeatStride, outputRepeatStride);

    return {inputAddr,          outputAddr,        token,
            die0Size,           die1Size,          inputSliceStride,
            outputSliceStride,  inputRepeatStride, outputRepeatStride,
            repeatNumVar};
}
} // namespace ops_hccl