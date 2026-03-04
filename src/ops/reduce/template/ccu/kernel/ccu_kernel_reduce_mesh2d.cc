/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh2d.h"


namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID  = 0;
constexpr int TOKEN_XN_ID  = 1;
constexpr int POST_SYNC_ID = 2;
constexpr int CKE_IDX_0    = 0; // 前后同步
constexpr int CKE_IDX_1    = 1; // 轴同步

CcuKernelReduceMesh2D::CcuKernelReduceMesh2D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceMesh2D *kernelArg
        = dynamic_cast<const CcuKernelArgReduceMesh2D *>(&arg);
    rankId_          = kernelArg->rankId_;  // 全局rankid
    dimSize_         = kernelArg->dimSize_; // dimSize_.size = 2,
    axisId_          = kernelArg->axisId_; // 要进行操作的是 行或列
    dimId_.emplace_back(rankId_ % dimSize_[0]);  // rank的x
    dimId_.emplace_back(rankId_ / dimSize_[0]);  // rank的y
    localId_         = dimId_[axisId_]; // 本rank所在的行/列 
    localSize_ =       dimSize_[axisId_]; // 本rank所在的行/列的总数
    rootDimId_.emplace_back(rootId_ % dimSize_[0]); // root的x
    rootDimId_.emplace_back(rootId_ / dimSize_[0]); // root的y

    channels_        = kernelArg->channels;
    
    HCCL_INFO("[CcuKernelReduceMesh2D] RankId[%u], DimSize0[%u], DimSize1[%u], localId[%u], lcoalSize[%u]",
        rankId_, dimSize_[0], dimSize_[1], localId_, localSize_);
    
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceMesh2D] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
    HCCL_INFO("[CcuKernelReduceMesh2D] init end, kernelArg->dimSize size[%zu] localSize_[%u]", 
              dimSize_.size(), localSize_);
}

HcclResult CcuKernelReduceMesh2D::InitResources()
{
    output_.push_back(CreateVariable());
    if (channels_.size() == 0) {
        HCCL_ERROR("CcuKernelReduceMesh2D channels is empty");
        return HcclResult::HCCL_E_INTERNAL;
    }
    uint16_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < localSize_; peerId++) {
        if (peerId == localId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO("[CcuKernelReduceMesh2D] MyRank[%u], PeerId[%u], ChannelId[%u]", localId_, peerId,
                      channelIdx);
            CcuRep::Variable inputVar, outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    
    xAxisGroupOpSize_     = CreateGroupOpSize();
    yAxisGroupOpSize_     = CreateGroupOpSize();
    HCCL_INFO("[CcuKernelReduceMesh2D] InitResources finished");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceMesh2D::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[localId_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[localId_], 1 << TOKEN_XN_ID);
    }
    uint16_t allBit  = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    
    HCCL_INFO("[CcuKernelReduceMesh2D] PreSync run finished");
}

void CcuKernelReduceMesh2D::PostSync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceMesh2D] PostSync run finished");
}

void CcuKernelReduceMesh2D::AxisSync(uint32_t signalIndex) // 轴间同步
{
    const uint32_t DIE_NUM = 2;
    LocalNotifyRecord(1 - axisId_, CKE_IDX_1, 1 << (axisId_ + signalIndex * DIE_NUM));
    LocalNotifyWait(axisId_, CKE_IDX_1, 1 << (1 - axisId_ + signalIndex * DIE_NUM));
    HCCL_INFO("[CcuKernelReduceMesh2D] AxisSync run finished");
    return;
}

void CcuKernelReduceMesh2D::LoadArgs()
{
    Load(input_[localId_]);
    Load(output_[0]);
    Load(token_[localId_]);
    Load(offSet_);
    Load(xAxisGroupOpSize_);
    Load(yAxisGroupOpSize_);
    HCCL_INFO("[CcuKernelReduceMesh2D] LoadArgs run finished");
}

void CcuKernelReduceMesh2D::Step1Reduce()
{
    // 只有与 root 同列的 rank 的 die0 进行第一步 reduce
    if(dimId_[0] != rootDimId_[0] || axisId_ != 0) {
        HCCL_INFO("[CcuKernelReduceMesh2D] rankId [%u], axisId [%u], skip Step1Reduce", rankId_, axisId_);
        return;
    }
    HCCL_INFO("[CcuKernelReduceMesh2D] rankId [%u], axisId [%u], run Step1Reduce", rankId_, axisId_);

    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst.addr  = input_[localId_];
    dst.token = token_[localId_];

    std::vector<CcuRep::RemoteAddr> src;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        src.push_back(CreateRemoteAddr());
    }
    uint32_t curId = 0;
    uint32_t dstId = 0;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        if (rankIdx != localId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = localSize_ - 1;
        }
        src[curId].addr  = input_[rankIdx];
        src[curId].token = token_[rankIdx];
    }

    GroupReduce(channels_, dst, src, xAxisGroupOpSize_, dataType_, outputDataType_, reduceOp_);
}

void CcuKernelReduceMesh2D::Step2ReduceForRoot()
{
    if (rankId_ != rootId_ || axisId_ != 1) {
        HCCL_INFO("[CcuKernelReduceMesh2D] rankId [%u], axisId [%u], skip Step2Reduce", 
                  rankId_, axisId_);
        return;
    }
    HCCL_INFO("[CcuKernelReduceMesh2D] rankId [%u], axisId [%u], skip Step2Reduce", 
              rankId_, axisId_);

    std::vector<CcuRep::RemoteAddr> src;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        src.push_back(CreateRemoteAddr());
    }
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst.addr = output_[0]; // 第二步reduce是从input reduce到root的output
    dst.token = token_[localId_];
    uint32_t curId = 0;
    uint32_t dstId = 0;
    for (uint32_t rankIdx = 0; rankIdx < localSize_; rankIdx++) {
        if (rankIdx != localId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = localSize_ - 1; // 最后一个位置放root的input
        }
        src[curId].addr  = input_[rankIdx];
        src[curId].token = token_[rankIdx];
    }
    GroupReduce(channels_, dst, src, xAxisGroupOpSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelReduceMesh2D::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceMesh2D] ReduceMesh2D run");
    CHK_RET(InitResources());
    LoadArgs();
    HCCL_INFO("[CcuKernelReduceMesh2D] Algorithm first step begins.");
    PreSync(); // 前同步
    Step1Reduce();
    AxisSync(0);
    PostSync();
    AxisSync(1);
    Step2ReduceForRoot();
    AxisSync(0);
    PostSync();
    AxisSync(1);
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceMesh2D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceMesh2D *taskArg = dynamic_cast<const CcuTaskArgReduceMesh2D *>(&arg);
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;
    uint64_t offset     = taskArg->offSet_;
    uint64_t xAxisSize = taskArg->xAxisSize_;
    uint64_t yAxisSize = taskArg->yAxisSize_;
    auto     xAxisGoSize = CalGoSize(xAxisSize);
    auto     yAxisGoSize = CalGoSize(yAxisSize);

    HCCL_INFO("[CcuKernelReduceMesh2D] ReduceMesh2D inputAddr [%llu] outputAddr [%llu] offset [%llu]"
     "xAxisSize [%llu] yAxisSize [%llu]", inputAddr, outputAddr, offset, xAxisSize, yAxisSize);

    return {inputAddr, outputAddr, tokenInfo, offset, xAxisGoSize[0], xAxisGoSize[1], xAxisGoSize[2], xAxisGoSize[3],
        yAxisGoSize[0], yAxisGoSize[1], yAxisGoSize[2], yAxisGoSize[3]};
}
} // namespace ops_hccl