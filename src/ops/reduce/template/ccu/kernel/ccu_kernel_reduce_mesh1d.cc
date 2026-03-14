/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh1d.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0    = 0;

CcuKernelReduceMesh1D::CcuKernelReduceMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{   
    const CcuKernelArgReduceMesh1D *kernelArg
        = dynamic_cast<const CcuKernelArgReduceMesh1D *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
    rootId_         = kernelArg->rootId_;
    HCCL_INFO(
        "[CcuKernelReduceMesh1D] Init, KernelArgs are rankId[%u], rootId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]",
        rankId_, rootId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelReduceMesh1D::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelReduceMesh1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelReduceMesh1D]channels.size: [%u]", channels_.size());
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelReduceMesh1D] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    groupOpSize_ = CreateGroupOpSize();

    currentRankSliceInputOffset_  = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    repeatNum_                    = CreateVariable();
    inputRepeatStride_            = CreateVariable();
    outputRepeatStride_           = CreateVariable();

    normalSliceSize_ = CreateVariable();
    lastSliceSize_   = CreateVariable();
    repeatNumVar_    = CreateVariable();
    flag_            = CreateVariable();

    dstAddr_ = CreateLocalAddr();
    inputAddr_.reserve(rankSize_);
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        inputAddr_.push_back(CreateRemoteAddr());
    }

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;;
}

void CcuKernelReduceMesh1D::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(repeatNum_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(repeatNumVar_);
    Load(groupOpSize_);
    return;
}

void CcuKernelReduceMesh1D::PreSync()
{
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D LocalPost begin");
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID); // index = 1，传递input信息
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D wait all end");
}

void CcuKernelReduceMesh1D::PostSync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D Reduce post sync end");
}

void CcuKernelReduceMesh1D::DoRepeatReduce()
{
    std::vector<CcuRep::RemoteAddr> &src = inputAddr_;
    CcuRep::LocalAddr &dst = dstAddr_;
    
    dst.addr = output_[rankId_];
    dst.token = token_[rankId_];
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rootId_) {
            src[curId].addr  = input_[rankIdx];
            src[curId].token = token_[rankIdx];
            curId++;
        } else {
            continue;
        }
    }
    src[rankSize_ - 1].addr = input_[rankId_];
    src[rankSize_ - 1].token = token_[rankId_];

    CCU_IF (flag_ != 0) {
        // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        dst.addr += outputRepeatStride_;
        for (auto &s : src) {
            s.addr += inputRepeatStride_;
        }
    }
    GroupReduce(channels_, dst, src, groupOpSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelReduceMesh1D::Algorithm()
{
    HCCL_INFO("[CcuContextReduceMesh1D] ReduceMesh1D run");
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();
    if (rankId_ == rootId_) {
        CcuRep::Variable repeatNumAdd = CreateVariable();
        repeatNumAdd  = 1;
        flag_ = 0;
        CCU_WHILE(repeatNumVar_ != UINT64_MAX) { // 循环repeatNum_次
            DoRepeatReduce();
            repeatNumVar_ += repeatNumAdd;
            flag_ = 1;
        }
    }
    PostSync();
    HCCL_INFO("[CcuContextReduceMesh1D] ReduceMesh1D end");
    return HcclResult::HCCL_SUCCESS;;
}

std::vector<uint64_t> CcuKernelReduceMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceMesh1D *taskArg    = dynamic_cast<const CcuTaskArgReduceMesh1D *>(&arg);
    // 空指针校验
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;

    uint64_t currentRankSliceInputOffset  = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t repeatNum                    = taskArg->repeatNum_;
    uint64_t inputRepeatStride            = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride           = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize              = taskArg->normalSliceSize_;
    uint64_t lastSliceSize                = taskArg->lastSliceSize_;
    uint64_t repeatNumVar                 = taskArg->repeatNumVar_;

    auto     goSize     = CalGoSize(normalSliceSize);

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        tokenInfo,
        currentRankSliceInputOffset,
        currentRankSliceOutputOffset,
        repeatNum,
        inputRepeatStride,
        outputRepeatStride,
        normalSliceSize,
        lastSliceSize,
        repeatNumVar,
        goSize[0],
        goSize[1],
        goSize[2],
        goSize[3],
    };

    HCCL_INFO("[CcuContextReduceMesh1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], currentRankSliceInputOffset[%llu], "
        "currentRankSliceOutputOffset[%llu], repeatNum[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], "
        "normalSliceSize[%llu], lastSliceSize[%llu], repeatNumVar[%llu], goSize[0][%llu], goSize[1][%llu], goSize[2][%llu], goSize[3][%llu], ",
        inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, repeatNum, inputRepeatStride, 
        outputRepeatStride, normalSliceSize, lastSliceSize, repeatNumVar, goSize[0], goSize[1], goSize[2], goSize[3]);

    return taskArgs;
}

} // namespace ops_hccl