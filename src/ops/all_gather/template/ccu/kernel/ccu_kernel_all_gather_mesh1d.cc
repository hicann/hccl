/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int POST_SYNC_ID = 3;  

CcuKernelAllGatherMesh1D::CcuKernelAllGatherMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherMesh1D *kernelArg
        = dynamic_cast<const CcuKernelArgAllGatherMesh1D *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;

    HCCL_INFO(
        "[CcuKernelAllGatherMesh1D] Init, KernelArgs are rankId[%u], rankSize_[%u]",
        rankId_, rankSize_);
}

HcclResult CcuKernelAllGatherMesh1D::InitResource()
{
    input_.push_back(CreateVariable());
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelAllGatherMesh1D] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, scratchVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &inputVar));
            output_.push_back(inputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    offset_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();;

    src = CreateLocalAddr();
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        dst.push_back(CreateRemoteAddr());
    }

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherMesh1D::LoadArgs()
{
    Load(input_[0]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(offset_);
    Load(groupOpSize_);
    return;
}

void CcuKernelAllGatherMesh1D::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }

    uint16_t allBit  = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    return;
}

void CcuKernelAllGatherMesh1D::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);         // bit index = 4, 用作后同步。cke都可以用同一个，所以都是CKE_IDX_0
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGatherMesh1D::DoAllGather()
{
    src.addr  = input_[0];
    src.token = token_[rankId_];
    uint32_t dstId = 0;
    uint32_t curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        dst[curId].addr = output_[rankIdx];
        dst[curId].addr += offset_;
        dst[curId].token = token_[rankIdx];
    }
    GroupBroadcast(channels_, dst, src, groupOpSize_);
}

HcclResult CcuKernelAllGatherMesh1D::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGatherMesh1D] AllGatherMesh1D run");

    CHK_RET(InitResource());

    LoadArgs();

    PreSync();

    DoAllGather();

    PostSync();

    HCCL_INFO("[CcuKernelAllGatherMesh1D] AllGatherMesh1D end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherMesh1D *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherMesh1D *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t token                       = taskArg->token_;
    uint64_t offset                      = taskArg->offset_;
    uint64_t sliceSize                   = taskArg->sliceSize_;
    auto     goSize     = CalGoSize(sliceSize);
    HCCL_INFO("[CcuKernelAllGatherMesh1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "offset[%llu], sliceSize[%llu]",
               inputAddr, outputAddr, offset, sliceSize);
    return {inputAddr, outputAddr, token, offset, goSize[0], goSize[1], goSize[2], goSize[3]};
}

} // namespace ops_hccl