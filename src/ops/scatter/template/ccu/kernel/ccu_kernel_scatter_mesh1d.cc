/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_mesh1d.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr uint16_t OUTPUT_XN_ID = 1;
constexpr uint16_t TOKEN_XN_ID = 2;
constexpr uint16_t POST_SYNC_ID = 3;
constexpr uint16_t CKE_IDX_0 = 0;

CcuKernelScatterMesh1D::CcuKernelScatterMesh1D(const CcuKernelArg &arg) : CcuKernelAlgBase(arg)
{
    const CcuKernelArgScatterMesh1D *kernelArg = dynamic_cast<const CcuKernelArgScatterMesh1D *>(&arg);
    rankId_ = kernelArg->rankId_;
    rankSize_ = kernelArg->dimSize_;
    channels_ = kernelArg->channels;
    rootId_ = kernelArg->rootId_;
    dataType_ = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG("[CcuKernelScatterMesh1D] outputDataType is [INVALID], set outputDataType to[%d]", outputDataType_);
    }
    HCCL_INFO("[CcuKernelScatterMesh1D] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
              "outputDataType[%d], rootId[%u]",
        rankId_,
        rankSize_,
        dataType_,
        outputDataType_,
        rootId_);
}

HcclResult CcuKernelScatterMesh1D::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelScatterMesh1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelScatterMesh1D] MyRank[%u], PeerId[%u], ChannelId[%u]", rankId_, peerId, channelIdx);
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));  // 获取channel中id=0的Var来传递output
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    input_ = CreateVariable();
    currentRankSliceInputOffset_ = CreateVariable();
    outputSliceStride_ = CreateVariable();
    normalSliceSize_ = CreateVariable();
    lastSliceSize_ = CreateVariable();
    inputRepeatStride_ = CreateVariable();
    outputRepeatStride_ = CreateVariable();
    repeatNum_ = CreateVariable();
    flag_ = CreateVariable();
    flag_ = 0;

    selfBit_ = 1 << rankId_;  // 仅rankid位为1，其他位为0，代表本端准备好了
    allBit_ = ((1 << rankSize_) - 1) & (~(1 << rankId_));  // 仅rankid位为0，其他位为1，代表远端准备好了
    outputMem_.reserve(rankSize_);
    inputMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        inputMem_.push_back(CreateLocalAddr());
        outputMem_.push_back(CreateRemoteAddr());
    }

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelScatterMesh1D::LoadArgs()
{
    Load(input_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(outputSliceStride_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(repeatNum_);
    return;
}

void CcuKernelScatterMesh1D::PreSync()
{
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto channel : channels_) {                                                           // 遍历所有通道
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);  // 广播output地址
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);     // 广播token
    }
    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    return;
}

void CcuKernelScatterMesh1D::PostSync()
{
    for (auto channel : channels_) {  // 通知所有通道操作完成
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return;
}

void CcuKernelScatterMesh1D::DoRepeatScatter()
{
    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd = 1;
    // 设置每张卡输入输出的起始地址
    for (uint64_t curId = 0; curId < rankSize_; curId++) {
        inputMem_[curId].token = token_[curId];   // 设置每张卡的输入token
        outputMem_[curId].token = token_[curId];  // 设置每张卡的输出token

        inputMem_[curId].addr = input_;  // 设置每张卡的输入地址，以root的起始地址为基准
        outputMem_[curId].addr = output_[curId];  // 设置每张卡的输出地址
        for (uint64_t i = 0; i < curId; i++) {
            inputMem_[curId].addr += currentRankSliceInputOffset_;  // 每张卡加上偏移量
            outputMem_[curId].addr += outputSliceStride_;
        }
    }
    if (rankId_ == rootId_) {
        CCU_WHILE(repeatNum_ != UINT64_MAX)
        {  // 循环UINT64_MAX - repeatNum_次
            DoScatter();
            repeatNum_ += repeatNumAdd;
            flag_ = 1;
        }
    } else {
        HCCL_INFO(
            "[CcuContextScatterMesh1D] RunRecvScatter local rank[%u], root rank[%u], do nothing", rankId_, rootId_);
    }
}

void CcuKernelScatterMesh1D::DoScatter()
{
    HCCL_INFO(
        "[CcuContextScatterMesh1D] RunSendScatter local rank[%u], root rank[%u], start send data", rankId_, rootId_);

    CCU_IF(flag_ != 0)
    {
        // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        for (auto &i : inputMem_) {
            i.addr += inputRepeatStride_;
        }
        for (auto &r : outputMem_) {
            r.addr += outputRepeatStride_;
        }
    }

    uint32_t channelId = 0;

    // 为root写到自己的地址专门创建一个LocalAddr变量
    CcuRep::LocalAddr myOutput = CreateLocalAddr();
    myOutput.addr = outputMem_[rankId_].addr;
    myOutput.token = outputMem_[rankId_].token;

    CcuRep::Variable sliceSize = CreateVariable();
    // root卡的数据发送到所有卡
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        event_.SetMask(1 << rankIdx);
        sliceSize = rankIdx == rankSize_ - 1 ? lastSliceSize_ : normalSliceSize_;
        CCU_IF(sliceSize != 0) {
            if (rankIdx == rankId_) {
                LocalCopyNb(myOutput, inputMem_[rankIdx], sliceSize, event_);
            } else {
                WriteNb(channels_[channelId], outputMem_[rankIdx], inputMem_[rankIdx], sliceSize, event_);
                channelId++;
            }
        }
        CCU_IF(sliceSize == 0) {
            RecordEvent(event_);
        }
    }

    // 等待数据传输完成
    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
    return;
}

HcclResult CcuKernelScatterMesh1D::Algorithm()
{
    HCCL_INFO("[CcuKernelScatterMesh1D] ScatterMesh1D run");

    CHK_RET(InitResource());

    LoadArgs();

    PreSync();

    DoRepeatScatter();

    PostSync();

    HCCL_INFO("[CcuKernelScatterMesh1D] ScatterMesh1D end");

    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelScatterMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgScatterMesh1D *taskArg = dynamic_cast<const CcuTaskArgScatterMesh1D *>(&arg);
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo = taskArg->token_;
    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_;
    uint64_t outputSliceStride = taskArg->outputSliceStride_;
    uint64_t inputRepeatStride = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize = taskArg->normalSliceSize_;
    uint64_t lastSliceSize = taskArg->lastSliceSize_;
    uint64_t repeatNum = taskArg->repeatNum_;

    std::vector<uint64_t> taskArgs = {inputAddr,
        outputAddr,
        tokenInfo,
        currentRankSliceInputOffset,
        outputSliceStride,
        inputRepeatStride,
        outputRepeatStride,
        normalSliceSize,
        lastSliceSize,
        repeatNum};

    HCCL_INFO("[CcuKernelScatterMesh1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
              "currentRankSliceInputOffset[%llu], outputSliceStride[%llu], inputRepeatStride[%llu],"
              "outputRepeatStride[%llu], normalSliceSize[%llu], lastSliceSize[%llu],"
              "repeatNum[%llu]",
        inputAddr,
        outputAddr,
        currentRankSliceInputOffset,
        outputSliceStride,
        inputRepeatStride,
        outputRepeatStride,
        normalSliceSize,
        lastSliceSize,
        repeatNum);
    return taskArgs;
}

}  // namespace ops_hccl