/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_nhr1d_mem2mem.h"

namespace ops_hccl {
using namespace hcomm;

constexpr uint16_t OUTPUT_XN_ID                   = 1;
constexpr uint16_t TOKEN_XN_ID                    = 2;
constexpr uint16_t POST_SYNC_ID                   = 3;
constexpr uint16_t REDUCE_SCATTER_PRE_SYNC_ID     = 4;
constexpr uint16_t REDUCE_SCATTER_POST_SYNC_ID    = 5;
constexpr uint16_t GATHER_SYNC_ID                 = 6;
constexpr uint16_t CKE_IDX_0                      = 0; // 前后同步
constexpr uint16_t RANK_NUM_PER_CKE               = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

CcuKernelReduceNHR1DMem2Mem::CcuKernelReduceNHR1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{   
    const CcuKernelArgReduceNHR1D *kernelArg
        = dynamic_cast<const CcuKernelArgReduceNHR1D *>(&arg);
    rankId_                          = kernelArg->rankId_;
    rootId_                          = kernelArg->rootId_;
    axisId_                          = kernelArg->axisId_;
    axisSize_                        = kernelArg->axisSize_;
    dimSize_                         = kernelArg->dimSize_;
    channels_                        = kernelArg->channels;
    stepInfoVector_                  = kernelArg->stepInfoVector_;
    rank2ChannelIdx_                 = kernelArg->rank2ChannelIdx_;
    localSize_                       = rank2ChannelIdx_.size();
    myRankIdx_                       = rank2ChannelIdx_.size();
    dataType_                        = kernelArg->opParam_.DataDes.dataType;
    reduceOp_                        = kernelArg->opParam_.reduceType;
    HCCL_INFO("[CcuKernelReduceNHR1DMem2Mem] kernelArg: rankId_[%u], axisId_[%u], dimSize_[%u], localSize_[%u], "
              "dataType[%d], reduceOp[%d], channels_[%d]",
              rankId_, axisId_, dimSize_, localSize_, dataType_, reduceOp_, channels_.size());
}

void CcuKernelReduceNHR1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[myRankIdx_]);
    Load(token_[myRankIdx_]);
    Load(isInputOutputEqual_);
    Load(die0Size_);
    Load(die1Size_);
    Load(die0SliceSize_);
    Load(die1SliceSize_);
    Load(die0LastSliceSize_);
    Load(die1LastSliceSize_);
    HCCL_DEBUG("[CcuKernelReduceNHR1DMem2Mem] LoadArgs run finished");
}

HcclResult CcuKernelReduceNHR1DMem2Mem::InitResources()
{
    die0Size_           = CreateVariable();
    die1Size_           = CreateVariable();
    die0SliceSize_      = CreateVariable();
    die1SliceSize_      = CreateVariable();
    die0LastSliceSize_  = CreateVariable();
    die1LastSliceSize_  = CreateVariable();
    isInputOutputEqual_ = CreateVariable();
    event_              = CreateCompletedEvent();

    input_ = CreateVariable();
    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        HCCL_DEBUG("[CcuKernelReduceNHR1DMem2Mem] MyRank[%u], TransportId[%u]", rankId_, channelIdx);
        
        CcuRep::Variable outputVar, tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
        output_.push_back(outputVar);
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);
    }
    output_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    localSrc_     = CreateLocalAddr();
    localDst_     = CreateLocalAddr();
    remoteDst_    = CreateRemoteAddr();
    HCCL_DEBUG("[CcuKernelReduceNHR1DMem2Mem] InitResources finished");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceNHR1DMem2Mem::PreSync()
{
    HCCL_DEBUG("[CcuKernelReduceNHR1DMem2Mem] PreSync start");
    for (auto ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, OUTPUT_XN_ID, output_[localSize_], 1 << OUTPUT_XN_ID); // index = 1，传递input信息
        NotifyRecord(ch, CKE_IDX_0, TOKEN_XN_ID, token_[localSize_], 1 << TOKEN_XN_ID);
    }
    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, allBit);
    }
    HCCL_DEBUG("[CcuKernelReduceNHR1DMem2Mem] PreSync end");
}

void CcuKernelReduceNHR1DMem2Mem::PostSync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_DEBUG("[CcuKernelReduceNHR1DMem2Mem] PostSync run finished");
}

void CcuKernelReduceNHR1DMem2Mem::DoReduceScatterNHR()
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = 0; i < stepInfoVector_.size() / NHR_NUM; i++) {
        const NHRStepInfo &nhrStepInfo = stepInfoVector_[i];
        DoReduceScatterNHRSingleStep(nhrStepInfo);
    }
}

void CcuKernelReduceNHR1DMem2Mem::DoReduceScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo)
{
    u32& toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32& fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
    u32  sendSliceIdx = 0;
    ChannelHandle sendChannel = channels_[toRankIdx];
    ChannelHandle recvChannel = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;
    localSrc_.token = token_[myRankIdx_];
    remoteDst_.token = token_[toRankIdx];

    if (nhrStepInfo.step != 0) {
        // 通知fromRank，可以写入
        NotifyRecord(recvChannel, CKE_IDX_0, 1 << REDUCE_SCATTER_PRE_SYNC_ID);

        // 等待toRank通知其可以写入
        NotifyWait(sendChannel, CKE_IDX_0, 1 << REDUCE_SCATTER_PRE_SYNC_ID);
    }

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        // cke用完了，等待上一轮结束在使用
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

        DoWriteReduceSlice(nhrStepInfo.toRank, localSrc_, remoteDst_, sendSliceIdx, i % RANK_NUM_PER_CKE);
    }
    // 等待上面的DoWriteReduceSlice方法传完
    event_.SetMask((1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
    WaitEvent(event_);

    // 通知toRank数据写入完毕
    NotifyRecord(sendChannel, CKE_IDX_0, 1 << REDUCE_SCATTER_POST_SYNC_ID);
    // 等待fromRank通知数据写入完毕
    NotifyWait(recvChannel, CKE_IDX_0, 1 << REDUCE_SCATTER_POST_SYNC_ID);

    HCCL_DEBUG("[DoReduceScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu", 
                rankId_, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
}

void CcuKernelReduceNHR1DMem2Mem::DoWriteReduceSlice(const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst, 
                                                  const u32 &sendSliceIdx, u32 signalIndex)
{
    ChannelHandle sendChannel = channels_[rank2ChannelIdx_[toRank]];
    bool          islastSlice;
    
    // 添加 die1 偏移
    if (axisId_ == 1) {
        src.addr += die0Size_;
        dst.addr += die0Size_;
    }

    // allreduce切片的最后一块slice，大小可能不一致
    islastSlice = (sendSliceIdx + 1 == dimSize_);
    const CcuRep::Variable &sliceSize = axisId_ == 0? (islastSlice? die0LastSliceSize_ : die0SliceSize_)
                                                    : (islastSlice? die1LastSliceSize_ : die1SliceSize_);

    CCU_IF(sliceSize != 0) {
        event_.SetMask(1 << signalIndex);
        WriteReduceNb(sendChannel, dst, src, sliceSize, dataType_, reduceOp_, event_);
    }
    CCU_IF(sliceSize == 0) {
        event_.SetMask(1 << signalIndex);
        RecordEvent(event_);
    }
}

void CcuKernelReduceNHR1DMem2Mem::DoGatherNHR()
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = stepInfoVector_.size() / NHR_NUM; i < stepInfoVector_.size(); i++) {
        const NHRStepInfo &nhrStepInfo = stepInfoVector_[i];
        DoGatherNHRSingleStep(nhrStepInfo);
    }
}

void CcuKernelReduceNHR1DMem2Mem::DoGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo)
{
    u32& toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank];
    u32& fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
    u32  sendSliceIdx = 0;
    ChannelHandle           sendChannel = channels_[toRankIdx];
    ChannelHandle           recvChannel = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;
    localSrc_.token                     = token_[myRankIdx_];
    remoteDst_.token                    = token_[toRankIdx];
    
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
        DoSendRecvSlice(nhrStepInfo.toRank, localSrc_, remoteDst_, sendSliceIdx, i % RANK_NUM_PER_CKE);
    }
    event_.SetMask((1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
    WaitEvent(event_);

    if (nhrStepInfo.step + 1 != stepInfoVector_.size()) {   // 最后一步不需要同步
        // 通知toRank，写入完毕
        NotifyRecord(sendChannel, CKE_IDX_0, 1 << GATHER_SYNC_ID);
        // 等待fromRank通知写入完毕
        NotifyWait(recvChannel, CKE_IDX_0, 1 << GATHER_SYNC_ID);
    }

    HCCL_DEBUG("[DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu", 
                rankId_, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
}

void CcuKernelReduceNHR1DMem2Mem::DoSendRecvSlice(const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst,
                                               const u32 &sendSliceIdx, u32 signalIndex)
{
    ChannelHandle sendChannel = channels_[rank2ChannelIdx_[toRank]];
    bool          islastSlice;
    
    // 添加 die1 偏移
    if (axisId_ == 1) {
        src.addr += die0Size_;
        dst.addr += die0Size_;
    }

    islastSlice = (sendSliceIdx + 1 == dimSize_);
    const CcuRep::Variable &sliceSize = axisId_ == 0? (islastSlice? die0LastSliceSize_ : die0SliceSize_)
                                                    : (islastSlice? die1LastSliceSize_ : die1SliceSize_);
    
    CCU_IF(sliceSize != 0) {
        event_.SetMask(1 << signalIndex);
        WriteNb(sendChannel, dst, src, sliceSize, event_);
    }
    CCU_IF(sliceSize == 0) {
        event_.SetMask(1 << signalIndex);
        RecordEvent(event_);
    }
}

void CcuKernelReduceNHR1DMem2Mem::LocalCopySlices()
{
    u32              nonTxSliceIdx    = 0;
    CcuRep::Variable tmpSliceOffset   = CreateVariable();
    tmpSliceOffset                    = 0;

    for (u64 i = 0; i < dimSize_; i++) {
        sliceOffset_.push_back(CreateVariable());
        sliceOffset_[i] = tmpSliceOffset;
        tmpSliceOffset += axisId_ == 0? die0SliceSize_: die1SliceSize_;
    }
    
    // 当input == output时，不需要拷贝
    CCU_IF(isInputOutputEqual_ == 0)
    {
        // 将step0中不需要写的slice，拷贝到本rank的output中
        const NHRStepInfo &nhrStepInfo = stepInfoVector_[0];
        const std::vector<u32> &nonTxSliceIdxList = GetNonTxSliceIdxs(nhrStepInfo.txSliceIdxs);
        for (u32 i = 0; i < nonTxSliceIdxList.size(); i++) {
            nonTxSliceIdx = nonTxSliceIdxList[i];

            if (i != 0) {
                if (i % RANK_NUM_PER_CKE == 0) {
                    event_.SetMask((1 << RANK_NUM_PER_CKE) - 1);
                    WaitEvent(event_);
                }
            }

            localSrc_.addr  = input_;
            localSrc_.addr += sliceOffset_[nonTxSliceIdx];
            localSrc_.token = token_[myRankIdx_];

            localDst_.addr  = output_[myRankIdx_];
            localDst_.addr += sliceOffset_[nonTxSliceIdx];
            localDst_.token = token_[myRankIdx_];
            DoLocalCopySlice(localSrc_, localDst_, nonTxSliceIdx, i);
        }
        event_.SetMask((1 << (nonTxSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
        WaitEvent(event_);
    } 
}

std::vector<u32> CcuKernelReduceNHR1DMem2Mem::GetNonTxSliceIdxs(const std::vector<u32> &txSliceIdxs) const
{
    std::vector<bool> isTx(dimSize_, false);
    for (u32 idx : txSliceIdxs) {
        if (idx < dimSize_) {
            isTx[idx] = true;
        }
    }

    std::vector<u32> nonTxSliceIdxs;
    for (u32 idx = 0; idx < dimSize_; ++idx) {
        if (!isTx[idx]) {
            nonTxSliceIdxs.push_back(idx);
        }
    }

    return nonTxSliceIdxs;
}

void CcuKernelReduceNHR1DMem2Mem::DoLocalCopySlice(CcuRep::LocalAddr &src, CcuRep::LocalAddr &dst,
                                                const u32 &copySliceIdx, u32 signalIndex)
{
    bool islastSlice;
    // 添加 die1 偏移
    if (axisId_ == 1) {
        src.addr += die0Size_;
        dst.addr += die0Size_;
    }

    islastSlice = (copySliceIdx + 1 == dimSize_);
    const CcuRep::Variable &sliceSize = axisId_ == 0? (islastSlice? die0LastSliceSize_ : die0SliceSize_)
                                                    : (islastSlice? die1LastSliceSize_ : die1SliceSize_);

    CCU_IF(sliceSize != 0) {
        event_.SetMask(1 << signalIndex);
        LocalCopyNb(dst, src, sliceSize, event_);
    }
    CCU_IF(sliceSize == 0) {
        event_.SetMask(1 << signalIndex);
        RecordEvent(event_);
    }
}

HcclResult CcuKernelReduceNHR1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceNHR1DMem2Mem] ReduceNhr1DMem2Mem run");

    CHK_RET(InitResources());
    LoadArgs();
    LocalCopySlices();
    PreSync();
    DoReduceScatterNHR();
    DoGatherNHR();
    PostSync();

    HCCL_DEBUG("[CcuKernelReduceNHR1DMem2Mem] AllReduceNHR1D end");
    return HcclResult::HCCL_SUCCESS;;
}

std::vector<uint64_t> CcuKernelReduceNHR1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceNHR1D *taskArg = dynamic_cast<const CcuTaskArgReduceNHR1D *>(&arg);
    // input&output&buffer地址
    uint64_t inputAddr          = taskArg->inputAddr_;
    uint64_t outputAddr         = taskArg->outputAddr_;
    uint64_t token              = taskArg->token_;
    uint64_t isInputOutputEqual = taskArg->isInputOutputEqual_;
    uint64_t die0Size           = taskArg->die0Size_;
    uint64_t die1Size           = taskArg->die1Size_;
    uint64_t die0SliceSize      = taskArg->die0SliceSize_;
    uint64_t die1SliceSize      = taskArg->die1SliceSize_;
    uint64_t die0LastSliceSize  = taskArg->die0LastSliceSize_;
    uint64_t die1LastSliceSize  = taskArg->die1LastSliceSize_;

    HCCL_INFO("[CcuContextReduceNHR1DMem2mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "die0Size[%llu], die1Size[%llu], die0SliceSize[%llu], die1SliceSize[%llu],"
              "die0LastSliceSize[%llu], die1LastSliceSize[%llu], isInputOutputEqual is [%lu]",
              inputAddr, outputAddr, die0Size, die1Size, die0SliceSize, die1SliceSize,
              die0LastSliceSize, die1LastSliceSize, isInputOutputEqual);

    return {inputAddr,          outputAddr,         token,
            isInputOutputEqual, die0Size,           die1Size,
            die0SliceSize,      die1SliceSize,      die0LastSliceSize,
            die1LastSliceSize};
}

} // namespace ops_hccl