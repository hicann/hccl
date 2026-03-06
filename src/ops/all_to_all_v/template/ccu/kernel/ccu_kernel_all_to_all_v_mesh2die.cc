/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_v_mesh2die.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;

constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;

CcuKernelAllToAllVMesh2Die::CcuKernelAllToAllVMesh2Die(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllToAllVMesh2Die *kernelArg = dynamic_cast<const CcuKernelArgAllToAllVMesh2Die *>(&arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("[CcuKernelAllToAllVMesh2Die] kernelArg ptr is null.");
        return;
    }

    channels_ = kernelArg->channels;

    rankId_ = kernelArg->rankId_;
    withMyRank_ = kernelArg->withMyRank_;
    rankGroup_ = kernelArg->rankGroup_;

    localSize_ = channels_.size() + 1;
    localId_ = localSize_ - 1;  // 本rank所在DIE的编号，固定放在末尾

    peerSize_ = channels_.size() + (withMyRank_ ? 1 : 0);
    logicId_ = rankId_ % peerSize_;

    selfBit_ = 1 << logicId_;
    allBit_  = ((1 << peerSize_) - 1) & (~(withMyRank_ ? selfBit_ : 0));

    HCCL_INFO("[CcuKernelAllToAllVMesh2Die] RankId[%u], localSize[%u], peerSize[%u], withMyRank[%u]", rankId_,
        localSize_, peerSize_, withMyRank_);
}

HcclResult CcuKernelAllToAllVMesh2Die::InitResources()
{
    CHK_PRT_RET(channels_.empty(),
        HCCL_ERROR("[CcuKernelAllToAllVMesh2Die] RankId[%u] channels is empty!", rankId_),
        HcclResult::HCCL_E_INTERNAL);

    // 由于要计算xnMaxTransportGoSize_，需要提前调用AllocGoResource
    AllocGoResource(CCU_MS_LOCAL_COPY_LOOP_COUNT, LOCAL_COPY_MS_PER_LOOP);
    xnMaxTransportSize_ = CreateVariable();
    xnMaxTransportGoSize_ = CreateGroupOpSize();
    // 每个SQE只能携带13个参数，当参数过多时会导致拆分为多个SQE，会需要多次CcuLaunch，导致下发耗时会增加；
    // 为尽可能减少SQE数量，常量直接计算，不通过Load加载；
    xnMaxTransportSize_ = MAX_TRANSPORT_SIZE;
    auto xnMaxTransportGoSize = CalGoSize(MAX_TRANSPORT_SIZE);
    xnMaxTransportGoSize_.addrOffset = xnMaxTransportGoSize[GO_ADDR_OFFSET_IDX];
    xnMaxTransportGoSize_.loopParam = xnMaxTransportGoSize[GO_LOOP_PARAM_IDX];
    xnMaxTransportGoSize_.parallelParam = xnMaxTransportGoSize[GO_PARALLEL_PARAM_IDX];
    xnMaxTransportGoSize_.residual = xnMaxTransportGoSize[GO_RESIDUAL_IDX];

    input_ = CreateVariable();

    for (uint32_t peerId = 0; peerId < channels_.size(); peerId++) {
        HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] RankId[%u], PeerId[%u]", rankId_, peerId);
        hcomm::CcuRep::Variable output;
        CHK_RET(CreateVariable(channels_[peerId], OUTPUT_XN_ID, &output));
        output_.emplace_back(output);
        hcomm::CcuRep::Variable token;
        CHK_RET(CreateVariable(channels_[peerId], TOKEN_XN_ID, &token));
        token_.emplace_back(token);
    }
    // 本rank固定放在末尾
    output_.emplace_back(CreateVariable());
    token_.emplace_back(CreateVariable());

    sendRecvInfo_.resize(peerSize_);
    for (uint64_t peerId = 0; peerId < peerSize_; peerId++) {
        sendRecvInfo_[peerId].sendOffset = CreateVariable();
        sendRecvInfo_[peerId].recvOffset = CreateVariable();
        sendRecvInfo_[peerId].sendTailSize = CreateVariable();
        sendRecvInfo_[peerId].sendTailGoSize = CreateGroupOpSize();
        sendRecvInfo_[peerId].sendLoopNum = CreateVariable();
    }

    for (uint16_t i = 0; i < channels_.size(); i++) {
        src_.emplace_back(CreateLocalAddr());
        dst_.emplace_back(CreateRemoteAddr());
    }

    localSrc_ = CreateLocalAddr();
    localDst_ = CreateLocalAddr();

    curSendTailSize_ = CreateVariable();
    curSendTailGoSize_ = CreateGroupOpSize();

    xnConst1_ = CreateVariable();
    completedRankCount_ = CreateVariable();

    event_ = CreateCompletedEvent();

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllToAllVMesh2Die::LoadArgs()
{
    Load(input_);
    Load(output_[localId_]);
    Load(token_[localId_]);

    for (uint64_t peerId = 0; peerId < peerSize_; peerId++) {
        Load(sendRecvInfo_[peerId].sendOffset);
        Load(sendRecvInfo_[peerId].recvOffset);
        Load(sendRecvInfo_[peerId].sendTailSize);
        Load(sendRecvInfo_[peerId].sendTailGoSize);
        Load(sendRecvInfo_[peerId].sendLoopNum);
    }
}

void CcuKernelAllToAllVMesh2Die::ExchangeInfoSync()
{
    // 交换信息并做前同步
    hcomm::CcuRep::Variable tempDst = CreateVariable();
    for (u32 peerId = 0; peerId < channels_.size(); peerId++) {
        tempDst = output_[localId_];
        tempDst += sendRecvInfo_[peerId].recvOffset;
        NotifyRecord(channels_[peerId], CKE_IDX_1, OUTPUT_XN_ID, tempDst, selfBit_);
        NotifyRecord(channels_[peerId], CKE_IDX_2, TOKEN_XN_ID, token_[localId_], selfBit_);
    }
    uint32_t channelIdx = 0;
    for (u32 peerId = 0; peerId < peerSize_; peerId++) {
        if (withMyRank_ && (peerId == logicId_)) {
            continue;
        }
        NotifyWait(channels_[channelIdx], CKE_IDX_1, 1 << peerId);
        NotifyWait(channels_[channelIdx], CKE_IDX_2, 1 << peerId);
        channelIdx++;
    }
}

void CcuKernelAllToAllVMesh2Die::PostSync()
{
    for (const auto &channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, selfBit_);
    }
    uint32_t channelIdx = 0;
    for (u32 peerId = 0; peerId < peerSize_; peerId++) {
        if (withMyRank_ && (peerId == logicId_)) {
            continue;
        }
        NotifyWait(channels_[channelIdx], CKE_IDX_0, 1 << peerId);
        channelIdx++;
    }
}

void CcuKernelAllToAllVMesh2Die::DoAll2AllVMultiLoop()
{
    completedRankCount_ = 0;
    xnConst1_ = 1;
    CCU_WHILE(completedRankCount_ != peerSize_) {
        HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] Algorithm loops[%u].", peerSize_);
        LoopStep();
    }
}

void CcuKernelAllToAllVMesh2Die::WriteToDstOutput(uint32_t peerId)
{
    HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] WriteToDstOutput Start. RankId[%u] peerId[%u]", rankId_, peerId);

    event_.SetMask(1 << peerId);

    CCU_IF(sendRecvInfo_[peerId].sendLoopNum == UINT64_MAX)    // 已经搬完了，仅同步
    {
        RecordEvent(event_);
    }

    CCU_IF(sendRecvInfo_[peerId].sendLoopNum != UINT64_MAX)    // 还没有搬完
    {
        CCU_IF(sendRecvInfo_[peerId].sendLoopNum == UINT64_MAX - 1)    // 最后一次搬运, 发送尾块数据
        {
            curSendTailSize_ = sendRecvInfo_[peerId].sendTailSize;
            CCU_IF(curSendTailSize_ == 0)
            {
                RecordEvent(event_);
            }
            CCU_IF(curSendTailSize_ != 0)
            {
                WriteNb(channels_[peerId], dst_[peerId], src_[peerId], curSendTailSize_, event_);
            }
            completedRankCount_ += xnConst1_;
        }
        CCU_IF(sendRecvInfo_[peerId].sendLoopNum != UINT64_MAX - 1)    // 正常搬运
        {
            WriteNb(channels_[peerId], dst_[peerId], src_[peerId], xnMaxTransportSize_, event_);
            dst_[peerId].addr += xnMaxTransportSize_;
            src_[peerId].addr += xnMaxTransportSize_;
        }
        sendRecvInfo_[peerId].sendLoopNum += xnConst1_;
    }

    HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] WriteToDstOutput end. RankId[%u] peerId[%u]", rankId_, peerId);
}

void CcuKernelAllToAllVMesh2Die::GroupCopyToDstOutput(uint32_t peerId)
{
    HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] GroupCopyToDstOutput Start. RankId[%u] peerId[%u]", rankId_, peerId);

    event_.SetMask(1 << peerId);

    CCU_IF(sendRecvInfo_[peerId].sendLoopNum == UINT64_MAX)    // 已经搬完了，仅同步
    {
        RecordEvent(event_);
    }

    CCU_IF(sendRecvInfo_[peerId].sendLoopNum != UINT64_MAX)    // 还没有搬完
    {
        CCU_IF(sendRecvInfo_[peerId].sendLoopNum == UINT64_MAX - 1)    // 最后一次搬运, 发送尾块数据
        {
            curSendTailSize_ = sendRecvInfo_[peerId].sendTailSize;
            curSendTailGoSize_ = sendRecvInfo_[peerId].sendTailGoSize;
            CCU_IF(curSendTailSize_ == 0)
            {
                RecordEvent(event_);
            }
            CCU_IF(curSendTailSize_ != 0)
            {
                GroupCopy(localDst_, localSrc_, curSendTailGoSize_);
                RecordEvent(event_);
            }
            completedRankCount_ += xnConst1_;
        }
        CCU_IF(sendRecvInfo_[peerId].sendLoopNum != UINT64_MAX - 1)    // 正常搬运
        {
            GroupCopy(localDst_, localSrc_, xnMaxTransportGoSize_);
            RecordEvent(event_);
            localDst_.addr += xnMaxTransportSize_;
            localSrc_.addr += xnMaxTransportSize_;
        }
        sendRecvInfo_[peerId].sendLoopNum += xnConst1_;
    }

    HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] GroupCopyToDstOutput end. RankId[%u] peerId[%u]", rankId_, peerId);
}

void CcuKernelAllToAllVMesh2Die::CalcGroupSrcDst()
{
    for (uint32_t peerId = 0; peerId < channels_.size(); peerId++) {
        src_[peerId].addr = input_;
        src_[peerId].addr += sendRecvInfo_[peerId].sendOffset;
        src_[peerId].token = token_[peerId];
        dst_[peerId].addr = output_[peerId];    // recvOffset在前同步时已经计算
        dst_[peerId].token = token_[peerId];
    }

    if (withMyRank_) {
        localSrc_.addr = input_;
        localSrc_.addr += sendRecvInfo_[localId_].sendOffset;
        localSrc_.token = token_[localId_];
        localDst_.addr = output_[localId_];
        localDst_.addr += sendRecvInfo_[localId_].recvOffset;
        localDst_.token = token_[localId_];
    }
}

void CcuKernelAllToAllVMesh2Die::LoopStep()
{
    for (uint32_t peerId = 0; peerId < channels_.size(); peerId++) {
        WriteToDstOutput(peerId);
    }

    if (withMyRank_) {
        GroupCopyToDstOutput(localId_);
    }

    event_.SetMask((1 << peerSize_) - 1);
    WaitEvent(event_);
}

HcclResult CcuKernelAllToAllVMesh2Die::Algorithm()
{
    // 初始化寄存器资源 & 加载外部输入参数
    HCCL_INFO("[CcuKernelAllToAllVMesh2Die] Algorithm Init Begins. RankId[%u]", rankId_);
    CHK_RET(InitResources());
    LoadArgs();

    HCCL_INFO("[CcuKernelAllToAllVMesh2Die] Algorithm Begins. RankId[%u]", rankId_);

    // 框架已经默认做了前后轴同步，算法不需要再重复做
    ExchangeInfoSync();

    CalcGroupSrcDst();
    DoAll2AllVMultiLoop();

    PostSync();

    HCCL_INFO("[CcuKernelAllToAllVMesh2Die] Algorithm Ends. RankId[%u]", rankId_);

    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllToAllVMesh2Die::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllToAllVMesh2Die *taskArg = dynamic_cast<const CcuTaskArgAllToAllVMesh2Die *>(&arg);
    if (taskArg == nullptr) {
        HCCL_ERROR("[CcuKernelAllToAllVMesh2Die] taskArg ptr is null. RankId[%u]", rankId_);
        return {};
    }

    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;

    std::vector<uint64_t> taskParams = {inputAddr, outputAddr, tokenInfo};  // 不需要ScratchMem

    const auto &sendRecvInfo = taskArg->localSendRecvInfo_;
    for (auto peerId : rankGroup_) {
        const uint64_t sendSize = sendRecvInfo.sendLength[peerId];
        const uint64_t floorLoopNum = sendSize / MAX_TRANSPORT_SIZE;
        uint64_t sendLoopNum = UINT64_MAX - 1 - floorLoopNum;
        uint64_t sendTailSize = sendSize - floorLoopNum * MAX_TRANSPORT_SIZE;
        auto sendTailGoSize = CalGoSize(sendTailSize);
        uint64_t sendOffset = sendRecvInfo.sendOffset[peerId];
        uint64_t recvOffset = sendRecvInfo.recvOffset[peerId];
        taskParams.push_back(sendOffset);
        taskParams.push_back(recvOffset);
        taskParams.push_back(sendTailSize);
        taskParams.insert(taskParams.cend(), sendTailGoSize.cbegin(), sendTailGoSize.cend());
        taskParams.push_back(sendLoopNum);
        HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die][GeneArgs][sliceInfo] RankId[%u], dstRank[%u]: sendOffset[%llu], "
            "recvOffset[%llu], sendSize[%llu], sendTailSize[%llu], sendLoopNum[%llu]", rankId_, peerId, sendOffset,
            recvOffset, sendSize, sendTailSize, sendLoopNum);
    }

    HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die][GeneArgs] RankId[%u], inputAddr[%#llx], outputAddr[%#llx], "
        "xnMaxTransportSize[%llu], args[%u]", rankId_, inputAddr, outputAddr, MAX_TRANSPORT_SIZE, taskParams.size());

    return taskParams;
}

}
