/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ccu_kernel_reduce_scatter_nhr1d_multi_jetty_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID   = 0;
constexpr int TOKEN_XN_ID   = 1;

constexpr int CKE_IDX_INPUT = 0;
constexpr int CKE_IDX_TOKEN = 1;
constexpr int CKE_IDX_READY = 2;
constexpr int CKE_IDX_DONE  = 3;
constexpr int POST_XN_ID    = 4;
constexpr uint16_t BIT_NUM_PER_CKE = 16; // CKE的位数，一个CKE可以处理16种信号

CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::CcuKernelReduceScatterNhrMutilJettyMem2Mem1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D *kernelArg
        = dynamic_cast<const CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D *>(&arg);
    rankId_          = kernelArg->rankId_;
    dimSize_         = kernelArg->dimSize_;
    channels_        = kernelArg->channels;
    dataType_        = kernelArg->opParam_.DataDes.dataType;
    outputDataType_  = kernelArg->opParam_.DataDes.outputType;
    stepInfoVector_  = kernelArg->stepInfoVector_;
    rank2ChannelIdx_ = kernelArg->rank2ChannelIdx_;
    localSize_       = rank2ChannelIdx_.size(); // 有多少个对端rank
    myRankIdx_       = rank2ChannelIdx_.size(); // 本rank的id写成最后一个
    portNum_         = kernelArg->portNum_;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
    HCCL_INFO(
        "[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] Init, KernelArgs are rankId[%u], dimSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d], portNum[%d]",
        rankId_, dimSize_, dataType_, outputDataType_, reduceOp_, portNum_);
}

HcclResult CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::InitResource()
{
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    output_             = CreateVariable();//0
    sliceSize_          = CreateVariable();//1
    inputSliceStride_   = CreateVariable();//2
    outputSliceStride_  = CreateVariable();//3
    inputRepeatStride_  = CreateVariable();//4
    outputRepeatStride_ = CreateVariable();//5
    sliceOneJettySize_  = CreateVariable();//6
    sliceLastJettySize_ = CreateVariable();//7
    repeatNumVar_       = CreateVariable();//8
    repeatNumVarTemp_   = CreateVariable();//9

    for (uint32_t channelIdx = 0; channelIdx < localSize_; channelIdx++) {
        CcuRep::Variable inputVar, tokenVar;
        CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
        input_.push_back(inputVar); // 获取channel中id=0的Var来传递output
        CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
        token_.push_back(tokenVar);
    }
    input_.push_back(CreateVariable()); // 自己的放在最后
    token_.push_back(CreateVariable());
    localSrc_ = CreateLocalAddr();
    localDst_ = CreateLocalAddr();
    remoteDst_ = CreateRemoteAddr();
    flag_ = CreateVariable();
    event_ = CreateCompletedEvent();
    for (uint32_t jettyId = 0; jettyId < portNum_; jettyId++) {
        jettyEvent_.push_back(CreateCompletedEvent());
    }
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] InitResource success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::LoadArgs()
{
    Load(input_[myRankIdx_]); // init的时候初始化的东西
    Load(output_);
    Load(token_[myRankIdx_]);
    Load(sliceSize_);
    Load(inputSliceStride_);
    Load(outputSliceStride_);
    Load(sliceOneJettySize_);
    Load(sliceLastJettySize_);
    Load(repeatNumVar_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    repeatNumVarTemp_ = repeatNumVar_;
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] LoadArgs success!");
    return HcclResult::HCCL_SUCCESS;
}

static uint32_t GetSignalIndex(const int signalBit)
{
    // 一个CKE有16位，可以处理16个用途
    return static_cast<uint32_t>(signalBit) / BIT_NUM_PER_CKE;
}

static uint16_t GetSignalMask(const int signalBit)
{
    return (1 << (static_cast<uint32_t>(signalBit) % BIT_NUM_PER_CKE));
}

HcclResult CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::PreSync()
{
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] PreSync start");

    const uint16_t signalBitInput = GetSignalMask(CKE_IDX_INPUT);
    const uint16_t signalBitToken = GetSignalMask(CKE_IDX_TOKEN);
    const uint32_t signalIndexInput = GetSignalIndex(CKE_IDX_INPUT);
    const uint32_t signalIndexToken = GetSignalIndex(CKE_IDX_TOKEN);
    
    for (ChannelHandle &channel : channels_) {
        CHK_RET(NotifyRecord(channel, signalIndexInput, CKE_IDX_INPUT, input_[myRankIdx_], signalBitInput));
        CHK_RET(NotifyRecord(channel, signalIndexToken, CKE_IDX_TOKEN, token_[myRankIdx_], signalBitToken));
    }
    // 等所有对端
    const uint16_t waitMask = signalBitInput | signalBitToken; // 组合一下mask
    std::set<uint32_t> signalIdxes{signalIndexInput, signalIndexToken}; // 0
    for (ChannelHandle &channel : channels_) {
        for (uint32_t signalIdx : signalIdxes) {
            CHK_RET(NotifyWait(channel, signalIdx, waitMask));
        }
    }
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] PreSync end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::PostSync()
{
    // 后同步的逻辑和前同步类似
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] PreSync start");
    const uint16_t selfBitInput = GetSignalMask(POST_XN_ID);
    const uint32_t signalIndexInput = GetSignalIndex(POST_XN_ID);

    // 通知所有对端
    for (ChannelHandle &channel : channels_) {
        CHK_RET(NotifyRecord(channel, signalIndexInput, selfBitInput));
    }

    // 等待所有需要的对端
    for (ChannelHandle &channel : channels_) {
        CHK_RET(NotifyWait(channel, signalIndexInput, selfBitInput));
    }
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] PreSync end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::DoRepeatReduceScatter()
{
    CcuRep::Variable tmpSliceOffset = CreateVariable();
    tmpSliceOffset = 0;
    // 用来记录每个rank要读取的rank的sliceIdx的偏移
    // 后面会用inputAddr来加上这个偏移获取sliceIdx的地址
    std::vector<CcuRep::Variable> inputSliceOffset;
    for (u64 i = 0; i < dimSize_; i++) {
        inputSliceOffset.push_back(CreateVariable());
        inputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += inputSliceStride_;
    }

    for (auto &nhrStepInfo : stepInfoVector_) {
        CHK_RET(DoRepeatReduceScatterNHRSingleStep(nhrStepInfo, inputSliceOffset));
    }
    // 因为所有的修改都是在input上进行的，所以最后需要把input上的数据搬到output上
    localDst_.addr = output_;
    localDst_.token = token_[myRankIdx_];
    localSrc_.addr = input_[myRankIdx_];
    localSrc_.addr += inputSliceOffset[rankId_];
    localSrc_.token = token_[myRankIdx_];

    // 不需要重复，所以直接拷贝
    CcuRep::Variable repeatNumAdd2 = CreateVariable();
    repeatNumAdd2  = 1;
    CCU_WHILE(repeatNumVar_ != UINT64_MAX) {
        repeatNumVar_ += repeatNumAdd2;
        CCU_IF(flag_ == 1) {
            localSrc_.addr += inputRepeatStride_;
            localDst_.addr += outputRepeatStride_;
        }
        event_.SetMask(1);
        CCU_IF(sliceSize_ == 0) {
            CHK_RET(RecordEvent(event_));
        }
        CCU_IF(sliceSize_ != 0) {
            CHK_RET(LocalCopyNb(localDst_, localSrc_, sliceSize_, event_));
        }
        CHK_RET(WaitEvent(event_)); // 等待拷贝完成，会将Mask清零
        flag_ = 1;
    }
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] DoRepeatReduceScatter success");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::DoRepeatReduceScatterNHRSingleStep(const NHRStepInfo &nhrStepInfo,
    const std::vector<CcuRep::Variable> &inputSliceOffset)
{
    u32& toRankIdx = rank2ChannelIdx_[nhrStepInfo.toRank]; // 和channel形成映射
    u32& fromRankIdx = rank2ChannelIdx_[nhrStepInfo.fromRank];
    ChannelHandle &sendChannel = channels_[toRankIdx];
    ChannelHandle &recvChannel = channels_[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs; // 发送到那张卡
    remoteDst_.token = token_[toRankIdx];
    localSrc_.token = token_[myRankIdx_];

    const uint32_t signalIdxReady = GetSignalIndex(CKE_IDX_READY); // 0
    const uint32_t signalIdxDone = GetSignalIndex(CKE_IDX_DONE); // 0
    const uint16_t signalBitReady = GetSignalMask(CKE_IDX_READY); // 准备好的信号
    const uint16_t signalBitDone = GetSignalMask(CKE_IDX_DONE); // 写完的信号

    // // 被写之前告诉fromRank自己准备好了-前同步 第一步时可以跳过
    if (nhrStepInfo.step != 0) {
        CHK_RET(NotifyRecord(recvChannel, signalIdxReady, signalBitReady));
        // 等待toRank准备好写入数据
        CHK_RET(NotifyWait(sendChannel, signalIdxReady, signalBitReady));
    }
    
    u32 sendSliceIdx = 0;
    // do srs
    for (const u32 &sendSliceIdx : sendSliceIdxList) {
        remoteDst_.addr = input_[toRankIdx];
        remoteDst_.addr += inputSliceOffset[sendSliceIdx];
        localSrc_.addr = input_[myRankIdx_];
        localSrc_.addr += inputSliceOffset[sendSliceIdx];
        CHK_RET(DoRepeatSendRecvSlices(nhrStepInfo.toRank, localSrc_, remoteDst_));
        HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] DoRepeatSendRecvSlices success");
    }
    // 写完后告知对端，做尾同步
    // 告诉toRank数据写完了
    CHK_RET(NotifyRecord(sendChannel, signalIdxDone, signalBitDone));
    // 等待fromRank写完数据
    CHK_RET(NotifyWait(recvChannel, signalIdxDone, signalBitDone));
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] DoRepeatReduceScatterNHRSingleStep success");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::DoRepeatSendRecvSlices(const u32 &toRank, CcuRep::LocalAddr &src,
                                                                 CcuRep::RemoteAddr &dst)
{
    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd  = 1;
    flag_ = 0;
    repeatNumVarTemp_ = repeatNumVar_;
    CCU_WHILE(repeatNumVarTemp_ != UINT64_MAX) {
        CCU_IF(repeatNumVarTemp_ != UINT64_MAX) {
            repeatNumVarTemp_ += repeatNumAdd;
        }
        CCU_IF(flag_ == 1) {
            src.addr += inputRepeatStride_;
            dst.addr += inputRepeatStride_;
        }
        CcuRep::LocalAddr tempSrc = CreateLocalAddr();
        CcuRep::RemoteAddr tempDst = CreateRemoteAddr();
        tempSrc.addr = src.addr;
        tempSrc.token = src.token;
        tempDst.addr = dst.addr;
        tempDst.token = dst.token;
        CCU_IF(sliceOneJettySize_ == 0) {
            for (u32 jettyId = 0; jettyId < portNum_ - 1; jettyId++) {
                jettyEvent_[jettyId].SetMask(1);
                CHK_RET(RecordEvent(jettyEvent_[jettyId]));
            }
        }
        CCU_IF(sliceOneJettySize_ != 0) {
            for (u32 jettyId = 0; jettyId < portNum_ - 1; jettyId++) {
                // 用一个tempdst和src
                jettyEvent_[jettyId].SetMask(1);
                CHK_RET(WriteReduceNb(channels_[rank2ChannelIdx_[toRank]], tempDst, tempSrc, sliceOneJettySize_,
                    dataType_, reduceOp_, jettyEvent_[jettyId]));
                tempDst.addr += sliceOneJettySize_;
                tempSrc.addr += sliceOneJettySize_;
            }
        }
        CCU_IF(sliceLastJettySize_ == 0) {
            jettyEvent_[portNum_ - 1].SetMask(1);
            CHK_RET(RecordEvent(jettyEvent_[portNum_ - 1]));
        }
        CCU_IF(sliceLastJettySize_ != 0) {
            u32 jettyId = portNum_ - 1;
            jettyEvent_[jettyId].SetMask(1);
            CHK_RET(WriteReduceNb(channels_[rank2ChannelIdx_[toRank]], tempDst, tempSrc, sliceLastJettySize_,
                    dataType_, reduceOp_, jettyEvent_[jettyId]));
        }
        for (u32 jettyId = 0; jettyId < portNum_; jettyId++) {
            CHK_RET(WaitEvent(jettyEvent_[jettyId]));
        }
        flag_ = 1;
    }
    flag_ = 0;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] CcuKernelReduceScatterNhrMutilJettyMem2Mem1D run");

    CHK_RET(InitResource());

    CHK_RET(LoadArgs());

    CHK_RET(PreSync());

    CHK_RET(DoRepeatReduceScatter());

    CHK_RET(PostSync());

    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] CcuKernelReduceScatterNhrMutilJettyMem2Mem1D end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterNhrMutilJettyMem2Mem1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterNhrMutilJettyMem2Mem1D *taskArg
        = dynamic_cast<const CcuTaskArgReduceScatterNhrMutilJettyMem2Mem1D *>(&arg);
    uint64_t inputAddr = taskArg->inputAddr_; // 输入地址
    uint64_t outputAddr = taskArg->outputAddr_; // 输出地址
    uint64_t token = taskArg->token_;
    uint64_t slice = taskArg->sliceSize_;
    uint64_t inputSliceStride = taskArg->inputSliceStride_;
    uint64_t outputSliceStride = taskArg->outputSliceStride_;
    uint64_t sliceOneJettySize = taskArg->sliceOneJettySize_; // 单个jetty发送的数据长度
    uint64_t sliceLastJettySize = taskArg->sliceLastJettySize_; // 最后一个jetty发送的数据长度
    uint64_t repeatNum = taskArg->repeatNum_;
    uint64_t inputRepeatStride  = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride  = taskArg->outputRepeatStride_;

    std::vector<uint64_t> taskArgs = {
        inputAddr, outputAddr, token, slice, inputSliceStride,  outputSliceStride, sliceOneJettySize, sliceLastJettySize,
        repeatNum, inputRepeatStride, outputRepeatStride};

    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "slice[%llu], inputSliceStride[%llu], outputSliceStride[%llu], sliceOneJettySize[%llu],"
               "sliceLastJettySize[%llu], repeatNum[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu]",
               inputAddr, outputAddr, slice, inputSliceStride, outputSliceStride, sliceOneJettySize, sliceLastJettySize,
               repeatNum, inputRepeatStride, outputRepeatStride);
    return taskArgs;
}

} // namespace ops_hccl