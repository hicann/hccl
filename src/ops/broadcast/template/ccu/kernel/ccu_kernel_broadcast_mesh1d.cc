/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_broadcast_mesh1d.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

CcuKernelBroadcastMesh1D::CcuKernelBroadcastMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgBroadcastMesh1D *kernelArg
        = dynamic_cast<const CcuKernelArgBroadcastMesh1D *>(&arg);
    rankId_         = kernelArg->rankId_;
    rootId_         = kernelArg->rootId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelBroadcastMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    HCCL_INFO("[CcuKernelBroadcastMesh1D] Init, KernelArgs are rootId[%u] rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d]",
        rootId_, rankId_, rankSize_, dataType_, outputDataType_);
}

HcclResult CcuKernelBroadcastMesh1D::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelBroadcastMesh1D] channels is empty!");
        return HCCL_E_INTERNAL;
    }
    input_ = CreateVariable();
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelBroadcastMesh1D] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    offSet_                 = CreateVariable();
    slicesize_              = CreateVariable();
    groupOpSize_            = CreateGroupOpSize();

    return HCCL_SUCCESS;
}

void CcuKernelBroadcastMesh1D::LoadArgs()
{
    Load(input_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(offSet_);
    Load(slicesize_);
    Load(groupOpSize_);
    return;
}

void CcuKernelBroadcastMesh1D::PreSync()
{
    for (ChannelHandle channel : channels_) {
        HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D LocalPost begin");
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D wait all end");
    return;
}

void CcuKernelBroadcastMesh1D::PostSync()
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D groupwait end");
    return;
}

void CcuKernelBroadcastMesh1D::BroadcastFromRootToAll()
{
    std::vector<CcuRep::RemoteAddr> dst;
    for (uint32_t index = 0; index < rankSize_; index++) {
        dst.emplace_back(CreateRemoteAddr());
    }
    CcuRep::LocalAddr src = CreateLocalAddr();
    src.addr = input_;
    src.token = token_[rankId_];
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rootId_) {
            dst[curId].addr  = output_[rankIdx];
            dst[curId].token = token_[rankIdx];
            curId++;
        }
    }
    dst[rankSize_ - 1].addr = output_[rankId_];
    dst[rankSize_ - 1].token = token_[rankId_];
    GroupBroadcast(channels_, dst, src, groupOpSize_);
    HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D GroupBroadcast end");
    return;
}

HcclResult CcuKernelBroadcastMesh1D::Algorithm()
{
    HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D run");
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();

    if (rankId_ == rootId_) {
        BroadcastFromRootToAll();
    }
    PostSync();

    HCCL_INFO("[CcuKernelBroadcastMesh1D] BroadcastMesh1D end");
    return HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelBroadcastMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgBroadcastMesh1D *taskArg = dynamic_cast<const CcuTaskArgBroadcastMesh1D *>(&arg);
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;
    uint64_t offset     = taskArg->offSet_;
    uint64_t sliceSize  = taskArg->sliceSize_;
    auto     goSize     = CalGoSize(sliceSize);
    HCCL_INFO("[CcuKernelBroadcastMesh1D][GeneArgs] inputAddr[%p] outputAddr[%p] offset[%llu] sliceSize[%llu]",
                inputAddr, outputAddr, offset, sliceSize);
    std::vector<uint64_t> taskArgs                     = {
        inputAddr,
        outputAddr,
        tokenInfo,
        offset,
        sliceSize,
        goSize[0],
        goSize[1],
        goSize[2],
        goSize[3],
    };
    return taskArgs;
}

} // namespace ops_hccl