/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_mesh1d.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID  = 0;
constexpr int TOKEN_XN_ID  = 1;
constexpr int POST_SYNC_ID = 2;
constexpr int CKE_IDX_0    = 0;

CcuKernelReduceScatterMesh1D::CcuKernelReduceScatterMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterMesh1D *kernelArg = dynamic_cast<const CcuKernelArgReduceScatterMesh1D *>(&arg);
    rankId_        = kernelArg->rankId_;
    rankSize_      = kernelArg->dimSize_;
    channels_      = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG("[CcuKernelReduceScatterMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_ = kernelArg->opParam_.reduceType;
    HCCL_INFO(
        "[CcuKernelArgReduceScatterMesh1D] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]",
        rankId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelReduceScatterMesh1D::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceScatterMesh1D] ReduceScatterMesh1D run");
    uint16_t selfBit = 1 << rankId_;
    output_.push_back(CreateVariable());
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelReduceScatterMesh1D] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    offset_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();

    Load(input_[rankId_]);
    Load(output_[0]);
    Load(token_[rankId_]);
    Load(offset_);
    Load(groupOpSize_);
    for (auto ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(ch, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    
    uint32_t allBit = 1 << INPUT_XN_ID  | 1 << TOKEN_XN_ID;
    for (auto ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, allBit);
    }

    std::vector<CcuRep::RemoteAddr> src;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        src.push_back(CreateRemoteAddr());
    }
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst.addr  = output_[0];
    dst.token = token_[rankId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            // 本端input放在数组末尾
            curId = rankSize_ - 1;
        }
        // 其中本端input为LocalAddr，但适配接口，写入RemoteAddr中
        src[curId].addr = input_[rankIdx];
        src[curId].addr += offset_;
        src[curId].token = token_[rankIdx];
    }

    GroupReduce(channels_, dst, src, groupOpSize_, dataType_, outputDataType_, reduceOp_);

    for (auto ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceScatterMesh1D] ReduceScatterMesh1D end");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterMesh1D* taskArg = dynamic_cast<const CcuTaskArgReduceScatterMesh1D*>(&arg);
    
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;
    uint64_t offset     = taskArg->offset_;
    uint64_t sliceSize  = taskArg->sliceSize_;
    auto     goSize     = CalGoSize(sliceSize);
    
    HCCL_INFO("[CcuKernelReduceScatterMesh1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "offset[%llu], sliceSize[%llu]",
               inputAddr, outputAddr, offset, sliceSize);
    return {inputAddr, outputAddr, tokenInfo, offset, goSize[0], goSize[1], goSize[2], goSize[3]};
}
} // namespace ops_hccl
