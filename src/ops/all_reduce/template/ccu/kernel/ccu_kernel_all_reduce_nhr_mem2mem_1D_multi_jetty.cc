/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_nhr_mem2mem_1D_multi_jetty.h"

#include <set>

#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int BIT_NUM_PER_CKE = 16; // 一个CKE可以处理16个信号类型

CcuKernelAllReduceNhr1DMem2MemMultiJetty::CcuKernelAllReduceNhr1DMem2MemMultiJetty(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty *kernelArg
        = dynamic_cast<const CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty *>(&arg);

    rankId_ = kernelArg->rankId_;
    rankSize_ = kernelArg->rankSize_;
    portNum_ = kernelArg->portNum_;
    channels_ = kernelArg->channels;
    reduceOp_ = kernelArg->opParam_.reduceType;
    dataType_ = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    algStepInfoList_ = kernelArg->algStepInfoList_;
    channelIdxMap_ = kernelArg->channelIdxMap_;

    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelAllReduceNhr1DMem2MemMultiJetty] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    HCCL_INFO("[CcuKernelAllReduceNhr1DMem2MemMultiJetty] Init, KernelArgs are rankId[%u], rankSize_[%u], portSize[%u],"
        " dataType[%d], outputDataType[%d], reduceOp[%d]",
        rankId_, rankSize_, portNum_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::InitResource()
{
    CHK_PRT_RET(channels_.empty(), HCCL_ERROR("[CcuKernelAllReduceNhr1DMem2MemMultiJetty] channels is empty!"),
                HcclResult::HCCL_E_INTERNAL);

    // Xns
    inputAddr_ = CreateVariable();
    isInplace_ = CreateVariable();
    dataSizePerRank_ = CreateVariable();
    dataSizePerPort_ = CreateVariable();
    lastRankSliceSize_ = CreateVariable();
    lastPortSliceSize_ = CreateVariable();

    // LocalAddr & RemoteAddr
    localInput_ = CreateLocalAddr();
    localOutput_ = CreateLocalAddr();
    remoteOutput_ = CreateRemoteAddr();

    for (uint32_t peerRankId = 0; peerRankId < rankSize_; peerRankId++) {
        if (peerRankId == rankId_) {
            // 本rank
            outputAddrs_.push_back(CreateVariable());
            outputTokens_.push_back(CreateVariable());
        } else if (channelIdxMap_.find(peerRankId) != channelIdxMap_.end()) {
            // 需要通信的对端
            const u32 channelIdx = channelIdxMap_[peerRankId];
            HCCL_DEBUG("[CcuKernelAllReduceNhr1DMem2MemMultiJetty] MyRank[%u], peerRankId[%u], ChannelId[%u]", rankId_,
                       peerRankId, channelIdx);
            CcuRep::Variable addrVar;
            CHK_RET(CreateVariable(channels_[channelIdx], static_cast<int>(XnId::OUTPUT), &addrVar));
            outputAddrs_.push_back(addrVar); 
            CcuRep::Variable tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], static_cast<int>(XnId::TOKEN), &tokenVar));
            outputTokens_.push_back(tokenVar);
        } else {
            // 不需要通信的对端，填空
            outputAddrs_.push_back(hcomm::CcuRep::Variable());
            outputTokens_.push_back(hcomm::CcuRep::Variable());
        }

        sliceOffset_.push_back(CreateVariable());
    }

    // 创建goSize的Variables
    localCopyGoSize_ = CreateGroupOpSize();
    localCopyGoSizeLastSlice_ = CreateGroupOpSize();

    // 需要portNum_个event
    for (auto i = 0; i < portNum_; ++i) {
        events_.push_back(CreateCompletedEvent());
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::LoadArgs()
{
    Load(inputAddr_);
    Load(outputAddrs_[rankId_]);
    Load(outputTokens_[rankId_]);
    Load(isInplace_);
    Load(dataSizePerRank_);
    Load(dataSizePerPort_);
    Load(lastRankSliceSize_);
    Load(lastPortSliceSize_);
    Load(localCopyGoSize_);
    Load(localCopyGoSizeLastSlice_);
    return HcclResult::HCCL_SUCCESS;
}

static uint32_t GetSignalIndex(const SignalBit signalBit)
{
    // 一个CKE有16位，可以处理16个用途
    return static_cast<uint32_t>(signalBit) / BIT_NUM_PER_CKE;
}

static uint16_t GetSignalMask(const SignalBit signalBit)
{
    return (1 << (static_cast<uint32_t>(signalBit) % BIT_NUM_PER_CKE));
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::LocalWaitAllEvent(const uint16_t mask)
{
    for (auto &event : events_) {
        event.SetMask(mask);
        CHK_RET(WaitEvent(event));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::PreSync()
{
    HCCL_DEBUG("[CcuKernelAllReduceNhr1DMem2MemMultiJetty] PreSync start");

    const uint16_t signalBitOutput = GetSignalMask(SignalBit::PRE_SYNC_OUTPUT);
    const uint16_t signalBitToken = GetSignalMask(SignalBit::PRE_SYNC_TOKEN);
    const uint32_t signalIndexOutput = GetSignalIndex(SignalBit::PRE_SYNC_OUTPUT);
    const uint32_t signalIndexToken = GetSignalIndex(SignalBit::PRE_SYNC_TOKEN);

    // 通知所有对端，同时写output和token信息
    for (const auto &channel : channels_) {
        CHK_RET(NotifyRecord(channel, signalIndexOutput, static_cast<int>(XnId::OUTPUT), outputAddrs_[rankId_], signalBitOutput));
        CHK_RET(NotifyRecord(channel, signalIndexToken, static_cast<int>(XnId::TOKEN), outputTokens_[rankId_], signalBitToken));
    }

    // 等待所有需要通信的对端
    const uint16_t waitMask = signalBitOutput | signalBitToken;
    std::set<uint32_t> signalIdxes{signalIndexOutput, signalIndexToken};
    for (const auto &channel : channels_) {
        for (const auto signalIdx : signalIdxes) {
            CHK_RET(NotifyWait(channel, signalIdx, waitMask));
        }
    }

    HCCL_DEBUG("[CcuKernelAllReduceNhr1DMem2MemMultiJetty] PreSync end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<u32> CcuKernelAllReduceNhr1DMem2MemMultiJetty::GetNonTxSliceIdxs(const std::vector<u32> &txSliceIdxs) const
{
    std::vector<bool> isTx(rankSize_, false);
    for (u32 idx : txSliceIdxs) {
        if (idx < rankSize_) {
            isTx[idx] = true;
        }
    }

    std::vector<u32> nonTxSliceIdxs;
    for (u32 idx = 0; idx < rankSize_; ++idx) {
        if (!isTx[idx]) {
            nonTxSliceIdxs.push_back(idx);
        }
    }

    return nonTxSliceIdxs;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::DoLocalCopySlice(hcomm::CcuRep::LocalAddr &src,
                                                                hcomm::CcuRep::LocalAddr &dst, const u32 &copySliceIdx,
                                                                hcomm::CcuRep::CompletedEvent &event)
{
    const bool islastSlice = copySliceIdx + 1 == rankSize_;
    const auto &sliceSize = islastSlice ? lastRankSliceSize_ : dataSizePerRank_;
    const auto &goSize = islastSlice ? localCopyGoSize_ : localCopyGoSizeLastSlice_;

    CCU_IF(sliceSize != 0)
    {
        CHK_RET(LocalCopyNb(dst, src, sliceSize, event));
    }
    CCU_IF(sliceSize == 0)
    {
        CHK_RET(RecordEvent(event));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::LocalCopySlices()
{
    u32 nonTxSliceIdx = 0;
    hcomm::CcuRep::Variable tmpSliceOffset = CreateVariable();
    tmpSliceOffset = 0;

    for (u64 i = 0; i < rankSize_; i++) {
        sliceOffset_.push_back(CreateVariable());
        sliceOffset_[i] = tmpSliceOffset;
        tmpSliceOffset += dataSizePerRank_;
    }

    // 使用一个event
    hcomm::CcuRep::CompletedEvent &event = events_[0];

    // 原地操作时不需要拷贝
    CCU_IF(isInplace_ == 0)
    {
        // 将step0中不需要写的slice，拷贝到本rank的output中
        const NHRStepInfo &nhrStepInfo = algStepInfoList_[0];
        const std::vector<u32> &nonTxSliceIdxList = GetNonTxSliceIdxs(nhrStepInfo.txSliceIdxs);
        for (u32 i = 0; i < nonTxSliceIdxList.size(); i++) {
            nonTxSliceIdx = nonTxSliceIdxList[i];

            if (i != 0) {  // 每拷贝16块等一次
                if (i % BIT_NUM_PER_CKE == 0) {
                    event.SetMask((1 << BIT_NUM_PER_CKE) - 1);
                    WaitEvent(event);
                }
            }

            localInput_.addr = inputAddr_;
            localInput_.addr += sliceOffset_[nonTxSliceIdx];
            localInput_.token = outputTokens_[rankId_];

            localOutput_.addr = outputAddrs_[rankId_];
            localOutput_.addr += sliceOffset_[nonTxSliceIdx];
            localOutput_.token = outputTokens_[rankId_];
            event.SetMask(1 << i);
            CHK_RET(DoLocalCopySlice(localInput_, localOutput_, nonTxSliceIdx, event));
        }
        event.SetMask((1 << (nonTxSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1);
        CHK_RET(WaitEvent(event));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::DoReduceScatterNHR()
{
    constexpr uint32_t nhrNum = 2;
    for (u64 i = 0; i < algStepInfoList_.size() / nhrNum; i++) {
        const NHRStepInfo &nhrStepInfo = algStepInfoList_[i];
        CHK_RET(DoReduceScatterNHRSingleStep(nhrStepInfo));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::DoReduceScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo)
{
    const u32 toRankIdx = channelIdxMap_[nhrStepInfo.toRank];
    const u32 fromRankIdx = channelIdxMap_[nhrStepInfo.fromRank];
    u32 sendSliceIdx = 0;
    ChannelHandle &sendChannel = channels_[toRankIdx];
    ChannelHandle &recvChannel = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;

    localInput_.token = outputTokens_[rankId_];
    remoteOutput_.token = outputTokens_[nhrStepInfo.toRank];

    HCCL_DEBUG("[%s] nhrStepInfo{step[%u], myRank[%u], toRank[%u], fromRank[%u]}, toRankIdx[%u], fromRankIdx[%u]",
               __func__, nhrStepInfo.step, nhrStepInfo.myRank, nhrStepInfo.toRank, nhrStepInfo.fromRank, toRankIdx,
               fromRankIdx);

    const uint32_t signalIdReady = GetSignalIndex(SignalBit::READY_TO_RECV_RS);
    const uint32_t signalIdDone = GetSignalIndex(SignalBit::SEND_DONE_RS);
    const uint16_t signalBitReady = GetSignalMask(SignalBit::READY_TO_RECV_RS);
    const uint16_t signalBitDone = GetSignalMask(SignalBit::SEND_DONE_RS);

    if (nhrStepInfo.step != 0) {
        // 通知fromRank，可以写入
        CHK_RET(NotifyRecord(recvChannel, signalIdReady, signalBitReady));

        // 等待toRank通知其可以写入
        CHK_RET(NotifyWait(sendChannel, signalIdReady, signalBitReady));
    }

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % BIT_NUM_PER_CKE == 0) {
                CHK_RET(LocalWaitAllEvent((1 << BIT_NUM_PER_CKE) - 1));
            }
        }

        if (nhrStepInfo.step == 0) {
            // 只有第0步的源数据从input中取
            localInput_.addr = inputAddr_;
            localInput_.addr += sliceOffset_[sendSliceIdx];
        } else {
            localInput_.addr = outputAddrs_[rankId_];
            localInput_.addr += sliceOffset_[sendSliceIdx];
        }
        
        remoteOutput_.addr = outputAddrs_[nhrStepInfo.toRank];
        remoteOutput_.addr += sliceOffset_[sendSliceIdx];

        CHK_RET(DoWriteReduceSlice(nhrStepInfo.toRank, localInput_, remoteOutput_, sendSliceIdx, i % BIT_NUM_PER_CKE));
    }
    CHK_RET(LocalWaitAllEvent((1 << (sendSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1));

    // 通知toRank数据写入完毕
    CHK_RET(NotifyRecord(sendChannel, signalIdDone, signalBitDone));

    // 等待fromRank通知数据写入完毕
    CHK_RET(NotifyWait(recvChannel, signalIdDone, signalBitDone));
    HCCL_DEBUG("[DoReduceScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
                rankId_, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::DoWriteReduceSlice(const u32 toRank, hcomm::CcuRep::LocalAddr &src,
                                                                  hcomm::CcuRep::RemoteAddr &dst,
                                                                  const u32 sendSliceIdx, const u32 signalIndex)
{
    const u32 toRankIdx = channelIdxMap_[toRank];
    ChannelHandle &sendChannel = channels_[toRankIdx];

    // allreduce切片的最后一块slice，大小可能不一致
    const bool islastSlice = sendSliceIdx + 1 == rankSize_;
    const hcomm::CcuRep::Variable &lastSliceSize = islastSlice ? lastPortSliceSize_ : dataSizePerPort_;

    // 统一设置一下mask
    for (auto &event : events_) {
        event.SetMask(1 << signalIndex);
    }

    CCU_IF(dataSizePerPort_ != 0)
    {
        for (uint32_t i = 0; i < portNum_ - 1; ++i) {
            CHK_RET(WriteReduceNb(sendChannel, dst, src, dataSizePerPort_, dataType_, reduceOp_, events_[i]));
            src.addr += dataSizePerPort_;
            dst.addr += dataSizePerPort_;
        }
    }
    CCU_IF(dataSizePerPort_ == 0)
    {
        // 无数据时，直接record event
        for (uint32_t i = 0; i < portNum_ - 1; ++i) {
            CHK_RET(RecordEvent(events_[i]));
        }
    }
    CCU_IF(lastSliceSize != 0)
    {
        CHK_RET(WriteReduceNb(sendChannel, dst, src, lastSliceSize, dataType_, reduceOp_, events_[events_.size() - 1]));
    }
    CCU_IF(lastSliceSize == 0)
    {
        CHK_RET(RecordEvent(events_[events_.size() - 1]));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::DoAllGatherNHR()
{
    constexpr uint32_t nhrNum = 2;
    for (u64 i = algStepInfoList_.size() / nhrNum; i < algStepInfoList_.size(); i++) {
        const NHRStepInfo &nhrStepInfo = algStepInfoList_[i];
        CHK_RET(DoAllGatherNHRSingleStep(nhrStepInfo));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::DoAllGatherNHRSingleStep(const NHRStepInfo &nhrStepInfo)
{
    const u32 toRankIdx = channelIdxMap_[nhrStepInfo.toRank];
    const u32 fromRankIdx = channelIdxMap_[nhrStepInfo.fromRank];
    u32 sendSliceIdx = 0;
    ChannelHandle &sendChannel = channels_[toRankIdx];
    ChannelHandle &recvChannel = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    localInput_.token = outputTokens_[rankId_];
    remoteOutput_.token = outputTokens_[nhrStepInfo.toRank];
    
    const uint32_t signalIdReady = GetSignalIndex(SignalBit::READY_TO_RECV_AG);
    const uint32_t signalIdDone = GetSignalIndex(SignalBit::SEND_DONE_AG);
    const uint16_t signalBitReady = GetSignalMask(SignalBit::READY_TO_RECV_AG);
    const uint16_t signalBitDone = GetSignalMask(SignalBit::SEND_DONE_AG);

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % BIT_NUM_PER_CKE == 0) {
                CHK_RET(LocalWaitAllEvent((1 << BIT_NUM_PER_CKE) - 1));
            }
        }

        localInput_.addr = outputAddrs_[rankId_];
        localInput_.addr += sliceOffset_[sendSliceIdx];

        remoteOutput_.addr = outputAddrs_[nhrStepInfo.toRank];
        remoteOutput_.addr += sliceOffset_[sendSliceIdx];
        CHK_RET(DoSendRecvSlice(nhrStepInfo.toRank, localInput_, remoteOutput_, sendSliceIdx, i % BIT_NUM_PER_CKE));
    }
    CHK_RET(LocalWaitAllEvent((1 << (sendSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1));

    // 通知toRank，写入完毕
    CHK_RET(NotifyRecord(sendChannel, signalIdDone, signalBitDone));

    // 等待fromRank通知写入完毕
    CHK_RET(NotifyWait(recvChannel, signalIdDone, signalBitDone));

    HCCL_DEBUG("[DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
                rankId_, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
                
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::DoSendRecvSlice(const u32 toRank, hcomm::CcuRep::LocalAddr &src,
                                                               hcomm::CcuRep::RemoteAddr &dst, const u32 &sendSliceIdx,
                                                               u32 signalIndex)
{
    const u32 toRankIdx = channelIdxMap_[toRank];
    ChannelHandle &sendChannel = channels_[toRankIdx];

    // allreduce切片的最后一块slice，大小可能不一致
    const bool islastSlice = sendSliceIdx + 1 == rankSize_;
    const hcomm::CcuRep::Variable &lastSliceSize = islastSlice ? lastPortSliceSize_ : dataSizePerPort_;

    // 统一设置一下mask
    for (auto &event : events_) {
        event.SetMask(1 << signalIndex);
    }
    CCU_IF(dataSizePerPort_ != 0)
    {
        for (uint32_t i = 0; i < portNum_ - 1; ++i) {
            CHK_RET(WriteNb(sendChannel, dst, src, dataSizePerPort_, events_[i]));
            src.addr += dataSizePerPort_;
            dst.addr += dataSizePerPort_;
        }
    }
    CCU_IF(dataSizePerPort_ == 0)
    {
        // 无数据时，直接record event
        for (uint32_t i = 0; i < portNum_ - 1; ++i) {
            CHK_RET(RecordEvent(events_[i]));
        }
    }
    CCU_IF(lastSliceSize != 0)
    {
        CHK_RET(WriteNb(sendChannel, dst, src, lastSliceSize, events_[events_.size() - 1]));
    }
    CCU_IF(lastSliceSize == 0)
    {
        CHK_RET(RecordEvent(events_[events_.size() - 1]));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceNhr1DMem2MemMultiJetty::Algorithm()
{
    HCCL_INFO("[CcuKernelAllReduceNhr1DMem2MemMultiJetty] Algorithm run");

    CHK_RET(InitResource());
    CHK_RET(LoadArgs());
    CHK_RET(LocalCopySlices());
    CHK_RET(PreSync());
    CHK_RET(DoReduceScatterNHR());
    CHK_RET(DoAllGatherNHR());

    HCCL_INFO("[CcuKernelAllReduceNhr1DMem2MemMultiJetty] Algorithm end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllReduceNhr1DMem2MemMultiJetty::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllReduceNhrMem2Mem1DMultiJetty *taskArg =
        dynamic_cast<const CcuTaskArgAllReduceNhrMem2Mem1DMultiJetty *>(&arg);
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t outputToken = taskArg->outputToken_;
    uint64_t isInplace = taskArg->isInplace_;
    uint64_t dataSizePerRank = taskArg->dataSizePerRank_;
    uint64_t dataSizePerPort = taskArg->dataSizePerPort_;
    uint64_t lastRankSliceSize = taskArg->lastRankSliceSize_;
    uint64_t lastPortSliceSize = taskArg->lastPortSliceSize_;

    std::vector<uint64_t> taskArgs = {inputAddr,       outputAddr,      outputToken,       isInplace,
                                      dataSizePerRank, dataSizePerPort, lastRankSliceSize, lastPortSliceSize};

    // go size of group operations
    const auto localCopyGoSize = CalGoSize(dataSizePerRank);
    taskArgs.insert(taskArgs.end(), localCopyGoSize.cbegin(), localCopyGoSize.cend());
    const auto localCopyGoSizeLastSlice = CalGoSize(lastRankSliceSize);
    taskArgs.insert(taskArgs.end(), localCopyGoSizeLastSlice.cbegin(), localCopyGoSizeLastSlice.cend());

    HCCL_INFO("[CcuKernelAllReduceNhr1DMem2MemMultiJetty] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "outputToken[%llu], isInplace[%llu], dataSizePerRank[%llu], dataSizePerPort[%llu], "
              "lastRankSliceSize[%llu], lastPortSliceSize[%llu]",
              inputAddr, outputAddr, outputToken, isInplace, dataSizePerRank, dataSizePerPort,
              lastRankSliceSize, lastPortSliceSize);

    return taskArgs;
}

} // namespace ops_hccl