/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

// XN ID定义：用于标识不同类型的变量
constexpr uint16_t SCRATCH_XN_ID = 1;  // Scratch缓冲区变量ID
constexpr uint16_t TOKEN_XN_ID = 2;    // Token变量ID
constexpr uint16_t STEP_POST_SYNC_ID = 3;

// CKE索引定义：用于标识不同类型的同步信号
constexpr uint16_t CKE_IDX_0 = 0;  // 后同步

// 轴ID定义
constexpr uint16_t FST_AXIS_ID = 0;  // 第一个轴（die0）
constexpr uint16_t SEC_AXIS_ID = 1;  // 第二个轴（die1）

// 每个CKE可以表示的rank数量（16个bit，每个bit代表一个rank）
constexpr uint16_t RANK_NUM_PER_CKE = 16;

CcuKernelScatterNHR1DMem2Mem::CcuKernelScatterNHR1DMem2Mem(const CcuKernelArg &arg) : CcuKernelAlgBase(arg)
{
    const CcuKernelArgScatterNHRMem2Mem1D *kernelArg = dynamic_cast<const CcuKernelArgScatterNHRMem2Mem1D *>(&arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("[CcuKernelScatterNHR1DMem2Mem] kernelArg is null");
        return;
    }

    rankId_ = kernelArg->rankId_;
    dimSize_ = kernelArg->dimSize_;
    rootId_ = kernelArg->rootId_;
    axisId_ = kernelArg->axisId_;
    axisSize_ = kernelArg->axisSize_;
    stepInfoVector_ = kernelArg->stepInfoVector_;
    rank2ChannelIdx_ = kernelArg->rank2ChannelIdx_;
    channels_ = kernelArg->channels;
    dataType_ = kernelArg->opParam_.DataDes.dataType;

    localSize_ = rank2ChannelIdx_.size();
    myRankIdx_ = rank2ChannelIdx_.size();  // 本rank在rank2ChannelIdx中的索引（放在最后）

    // 计算signal数量：每个CKE有16个bit，可以表示16个rank
    signalNum_ = (dimSize_ + RANK_NUM_PER_CKE - 1) / RANK_NUM_PER_CKE;

    HCCL_INFO("[CcuKernelScatterNHR1DMem2Mem] Init, KernelArgs are rankId[%u], dimSize_[%u], rootId[%u], axisId[%u], "
              "axisSize[%u], dataType[%d], localSize_[%u], signalNum_[%u], stepInfoVector.size()[%lu], "
              "rank2ChannelIdx.size()[%lu]",
        rankId_,
        dimSize_,
        rootId_,
        axisId_,
        axisSize_,
        dataType_,
        localSize_,
        signalNum_,
        stepInfoVector_.size(),
        rank2ChannelIdx_.size());
}

HcclResult CcuKernelScatterNHR1DMem2Mem::InitResource()
{
    // 创建基础变量
    die0Size_ = CreateVariable();
    die1Size_ = CreateVariable();
    inputSliceStride_ = CreateVariable();
    curScratchStride_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    repeatNumVar_ = CreateVariable();
    repeatNumVarTemp_ = CreateVariable();
    repeatTimeflag_ = CreateVariable();
    curInputOffset_ = CreateVariable();
    curScratchOffset_ = CreateVariable();
    cursliceSize_ = CreateVariable();
    isOutputScratch_ = CreateVariable();

    // 创建输入和输出变量
    input_ = CreateVariable();
    output_ = CreateVariable();

    // 为每个需要通信的rank创建scratch和token变量
    uint16_t channelIdx = 0;
    for (uint32_t i = 0; i < localSize_; i++) {
        if (channels_.size() > channelIdx) {
            HCCL_DEBUG(
                "[CcuKernelScatterNHR1DMem2Mem] MyRank[%u], TransportId[%u], ChannelId[%u]", rankId_, i, channelIdx);
            CcuRep::Variable scratchVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], SCRATCH_XN_ID, &scratchVar));
            scratch_.push_back(scratchVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        } else {
            HCCL_ERROR(
                "[CcuKernelScatterNHR1DMem2Mem] channels size[%lu] is less than expected[%u]", channels_.size(), i);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    // 本端的scratch和token放在最后
    scratch_.push_back(CreateVariable());
    token_.push_back(CreateVariable());

    // 创建内存对象
    srcMem_ = CreateLocalAddr();
    dstMem_ = CreateLocalAddr();
    dstRemoteMem_ = CreateRemoteAddr();

    event_ = CreateCompletedEvent();  // 创建完成事件

    HCCL_INFO("[CcuKernelScatterNHR1DMem2Mem] InitResource finished");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelScatterNHR1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_);
    Load(token_[myRankIdx_]);
    Load(scratch_[myRankIdx_]);
    Load(die0Size_);
    Load(die1Size_);
    Load(inputSliceStride_);
    Load(curScratchStride_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(repeatNumVar_);
    Load(isOutputScratch_);
    HCCL_INFO("[CcuKernelScatterNHR1DMem2Mem] LoadArgs run finished");
}

void CcuKernelScatterNHR1DMem2Mem::PreSync()
{
    HCCL_INFO("[CcuKernelScatterNHR1DMem2Mem] PreSync start");

    uint32_t allBit = 1 << SCRATCH_XN_ID | 1 << TOKEN_XN_ID;

    // 向所有channel发送scratch和token信息（前同步）
    for (auto channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, SCRATCH_XN_ID, scratch_[myRankIdx_], 1 << SCRATCH_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[myRankIdx_], 1 << TOKEN_XN_ID);
    }

    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelScatterNHR1DMem2Mem] PreSync end");
}

void CcuKernelScatterNHR1DMem2Mem::DoScatterNHR()
{
    // 初始化偏移量
    curInputOffset_ = 0;    // input偏移
    curScratchOffset_ = 0;  // scratch偏移

    // 为每个rank计算input偏移量
    // inputOffset_[i]存储rank i的input地址偏移量
    for (u64 i = 0; i < dimSize_; i++) {
        inputOffset_.push_back(CreateVariable());
        inputOffset_[i] = curInputOffset_;
        curInputOffset_ += inputSliceStride_;
    }

    // 为每个rank计算scratch偏移量
    for (u64 i = 0; i < dimSize_; i++) {
        scratchOffset_.push_back(CreateVariable());
        scratchOffset_[i] = curScratchOffset_;
        curScratchOffset_ += curScratchStride_;
    }

    // 执行NHR算法的每个步骤
    for (auto &nhrStepInfo : stepInfoVector_) {
        DoScatterNHRSingleStep(nhrStepInfo);
    }

    // 将数据从scratch复制到output（或从input复制到output，如果是root）
    if (rankId_ == rootId_) {
        srcMem_.addr = input_;
        srcMem_.addr += inputOffset_[rankId_];
    } else {
        srcMem_.addr = scratch_[myRankIdx_];
        srcMem_.addr += scratchOffset_[rankId_];
    }
    dstMem_.addr = output_;
    srcMem_.token = token_[myRankIdx_];
    dstMem_.token = token_[myRankIdx_];

    // 处理重复操作
    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd = 1;
    repeatTimeflag_ = 0;
    CCU_WHILE(repeatNumVar_ != UINT64_MAX)
    {
        repeatNumVar_ += repeatNumAdd;
        CCU_IF(repeatTimeflag_ != 0)
        {
            // 非第一轮执行时，需要添加偏移量
            if (rankId_ == rootId_) {
                srcMem_.addr += inputRepeatStride_;
            } else {
                srcMem_.addr += outputRepeatStride_;
            }
            dstMem_.addr += outputRepeatStride_;
        }
        CCU_IF(repeatTimeflag_ == 0)
        {
            // 第一轮执行时，如果是die1，需要跳过die0的数据
            if (axisId_ == 1) {
                srcMem_.addr += die0Size_;
                dstMem_.addr += die0Size_;
            }
        }
        cursliceSize_ = (axisId_ == 0) ? die0Size_ : die1Size_;
        {
            event_.SetMask(1 << rankId_);
            CCU_IF(isOutputScratch_ == 1)
            {
                // 如果输出使用scratch buffer，需要特殊处理
                if (rootId_ != 0 && rankId_ == 0) {
                    RecordEvent(event_);
                } else {
                    LocalCopyNb(dstMem_, srcMem_, cursliceSize_, event_);
                }
            }
            CCU_IF(isOutputScratch_ != 1)
            {
                // 正常情况：从srcMem复制到dstMem
                LocalCopyNb(dstMem_, srcMem_, cursliceSize_, event_);
            }
            WaitEvent(event_);
        }
        repeatTimeflag_ = 1;
    }
}

void CcuKernelScatterNHR1DMem2Mem::DoScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo)
{
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const std::vector<u32> &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;

    // 接收端：等待发送方发送数据
    if (recvSliceIdxList.size() != 0) {
        if (rank2ChannelIdx_.count(nhrStepInfo.fromRank) == 0) {
            HCCL_ERROR(
                "[CcuKernelScatterNHR1DMem2Mem] fromRank[%u] not found in rank2ChannelIdx", nhrStepInfo.fromRank);
            return;
        }
        u32 fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
        if (fromRankIdx >= channels_.size()) {
            HCCL_ERROR("[CcuKernelScatterNHR1DMem2Mem] fromRankIdx[%u] >= channels.size()[%lu]",
                fromRankIdx,
                channels_.size());
            return;
        }
        ChannelHandle recvChannel = channels_[fromRankIdx];
        // 后同步：等待发送方通知写入完毕
        NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
    }

    // 发送端：向接收方发送数据，然后发送一个同步信号
    if (sendSliceIdxList.size() != 0) {
        if (rank2ChannelIdx_.count(nhrStepInfo.toRank) == 0) {
            HCCL_ERROR("[CcuKernelScatterNHR1DMem2Mem] toRank[%u] not found in rank2ChannelIdx", nhrStepInfo.toRank);
            return;
        }
        u32 toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank];
        if (toRankIdx >= channels_.size()) {
            HCCL_ERROR(
                "[CcuKernelScatterNHR1DMem2Mem] toRankIdx[%u] >= channels.size()[%lu]", toRankIdx, channels_.size());
            return;
        }
        ChannelHandle sendChannel = channels_[toRankIdx];

        // 发送每个slice的数据
        for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
            u32 sendSliceIdx = sendSliceIdxList[i];
            // 每16个slice需要等待一次（RANK_NUM_PER_CKE = 16）
            if (i != 0 && i % RANK_NUM_PER_CKE == 0) {
                event_.SetMask((1 << RANK_NUM_PER_CKE) - 1);
                WaitEvent(event_);
            }

            // 设置源地址：如果是root，从input中取；否则从scratch中取
            if (rankId_ == rootId_) {
                srcMem_.addr = input_;
                srcMem_.addr += inputOffset_[sendSliceIdx];
            } else {
                srcMem_.addr = scratch_[myRankIdx_];
                srcMem_.addr += scratchOffset_[sendSliceIdx];
            }

            // 设置目标地址：写入到接收方的scratch buffer
            srcMem_.token = token_[myRankIdx_];
            dstRemoteMem_.token = token_[toRankIdx];
            dstRemoteMem_.addr = scratch_[toRankIdx];
            dstRemoteMem_.addr += scratchOffset_[sendSliceIdx];

            // 执行发送接收操作
            DoSendRecvSlice(nhrStepInfo.toRank, srcMem_, dstRemoteMem_, i % RANK_NUM_PER_CKE);
        }

        // 后同步：发送一个同步信号，通知接收方写入完毕
        NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
    }

    HCCL_INFO("[DoScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
        rankId_,
        nhrStepInfo.step,
        nhrStepInfo.toRank,
        nhrStepInfo.fromRank,
        sendSliceIdxList.size());
}

void CcuKernelScatterNHR1DMem2Mem::DoSendRecvSlice(
    const u32 &toRank, CcuRep::LocalAddr &src, CcuRep::RemoteAddr &dst, u32 signalIndex)
{
    if (rank2ChannelIdx_.count(toRank) == 0) {
        HCCL_ERROR("[CcuKernelScatterNHR1DMem2Mem] toRank[%u] not found in rank2ChannelIdx", toRank);
        return;
    }
    u32 toRankIdx = rank2ChannelIdx_[toRank];
    if (toRankIdx >= channels_.size()) {
        HCCL_ERROR("[CcuKernelScatterNHR1DMem2Mem] toRankIdx[%u] >= channels.size()[%lu]", toRankIdx, channels_.size());
        return;
    }
    ChannelHandle sendChannel = channels_[toRankIdx];

    // 处理重复操作
    CcuRep::Variable repeatNumAdd2 = CreateVariable();
    repeatNumAdd2 = 1;
    repeatTimeflag_ = 0;
    repeatNumVarTemp_ = repeatNumVar_;
    CCU_WHILE(repeatNumVarTemp_ != UINT64_MAX)
    {
        repeatNumVarTemp_ += repeatNumAdd2;
        CCU_IF(repeatTimeflag_ == 1)
        {
            // 非第一轮执行时，需要添加偏移量
            if (rankId_ == rootId_) {
                src.addr += inputRepeatStride_;
            } else {
                src.addr += outputRepeatStride_;
            }
            dst.addr += outputRepeatStride_;
        }
        CCU_IF(repeatTimeflag_ == 0)
        {
            // 第一轮执行时，如果是die1，需要跳过die0的数据
            if (axisId_ == 1) {
                src.addr += die0Size_;
                dst.addr += die0Size_;
            }
        }
        cursliceSize_ = (axisId_ == 0) ? die0Size_ : die1Size_;
        // 执行写入操作：从本地地址src写入到远程地址dst
        event_.SetMask(1 << signalIndex);
        WriteNb(sendChannel, dst, src, cursliceSize_, event_);
        WaitEvent(event_);
        repeatTimeflag_ = 1;
    }
}

HcclResult CcuKernelScatterNHR1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelScatterNHR1DMem2Mem] ScatterNHR1D run");

    CHK_RET(InitResource());
    LoadArgs();

    PreSync();
    DoScatterNHR();

    HCCL_INFO("[CcuKernelScatterNHR1DMem2Mem] ScatterNHR1D end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelScatterNHR1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgScatterNHRMem2Mem1D *taskArg = dynamic_cast<const CcuTaskArgScatterNHRMem2Mem1D *>(&arg);
    if (taskArg == nullptr) {
        HCCL_ERROR("[CcuKernelScatterNHR1DMem2Mem] taskArg is null");
        return {};
    }

    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t token = taskArg->token_;
    uint64_t scratchAddr = taskArg->scratchAddr_;
    uint64_t die0Size = taskArg->die0Size_;
    uint64_t die1Size = taskArg->die1Size_;
    uint64_t inputSliceStride = taskArg->inputSliceStride_;
    uint64_t curScratchStride = taskArg->sliceSize_ * taskArg->repeatNum_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t repeatNumVar = UINT64_MAX - taskArg->repeatNum_;
    uint64_t isOutputScratch = taskArg->isOutputScratch_;

    HCCL_INFO(
        "[CcuKernelScatterNHR1DMem2Mem] TaskArgs: rankId_[%u], inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu], "
        "die0Size[%llu], die1Size[%llu], inputSliceStride[%llu], curScratchStride[%llu], "
        "inputRepeatStride[%llu], outputRepeatStride[%llu], repeatNumVar[%llu], isOutputScratch[%llu]",
        rankId_,
        inputAddr,
        outputAddr,
        scratchAddr,
        die0Size,
        die1Size,
        inputSliceStride,
        curScratchStride,
        inputRepeatStride,
        outputRepeatStride,
        repeatNumVar,
        isOutputScratch);

    return {inputAddr,
        outputAddr,
        token,
        scratchAddr,
        die0Size,
        die1Size,
        inputSliceStride,
        curScratchStride,
        inputRepeatStride,
        outputRepeatStride,
        repeatNumVar,
        isOutputScratch};
}

}  // namespace ops_hccl
