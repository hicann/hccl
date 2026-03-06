/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_mesh2die.h"

namespace ops_hccl {

constexpr int CKE_IDX_0   = 0;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;

CcuKernelAllToAllMesh2Die::CcuKernelAllToAllMesh2Die(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllToAllMesh2Die *kernelArg = dynamic_cast<const CcuKernelArgAllToAllMesh2Die *>(&arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("[CcuKernelAllToAllMesh2Die] kernelArg ptr is null.");
        return;
    }

    channels_ = kernelArg->channels;

    rankSize_ = kernelArg->rankSize_;
    rankId_ = kernelArg->rankId_;
    withMyRank_ = kernelArg->withMyRank_;
    rankGroup_ = kernelArg->rankGroup_;
}

HcclResult CcuKernelAllToAllMesh2Die::InitResources()
{
    // 创建Variable，用于交换地址及token
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllToAllMesh2Die] RankId[%u] channels_ is empty", rankId_);
    }
    virRankSize = channels_.size() + 1;

    for (u64 id = 0; id < channels_.size(); id++) {
        // 非本地，使用远端Variable
        HCCL_DEBUG("[CcuKernelAllToAllMesh2Die] RankId[%u], Id[%u]", rankId_, id);
        hcomm::CcuRep::Variable output, token;
        CHK_RET(CreateVariable(channels_[id], OUTPUT_XN_ID, &output));
        output_.emplace_back(output);
        CHK_RET(CreateVariable(channels_[id], TOKEN_XN_ID, &token));
        token_.emplace_back(token);
    }
    input_ = CreateVariable();
    // 最后一个位置放自己地址
    output_.emplace_back(CreateVariable());
    token_.emplace_back(CreateVariable());

    sliceSize_         = CreateVariable();
    inputSliceStride_  = CreateVariable();
    outputoffset_ = CreateVariable();
    groupOpSize_       = CreateGroupOpSize();

    logicRankSize = withMyRank_ ? channels_.size() + 1 : channels_.size();
    uint16_t logicId       = rankId_ % logicRankSize; // topo为 2 * n
    event_ = CreateCompletedEvent();

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllToAllMesh2Die::LoadArgs()
{
    // 从SQE load args，本rank需要的input、output地址等信息
    Load(input_);
    Load(output_[virRankSize - 1]);
    Load(token_[virRankSize - 1]);
    Load(sliceSize_); // 本轮传输的分片大小
    Load(inputSliceStride_);
    Load(outputoffset_);
    Load(groupOpSize_);
    return;
}

void CcuKernelAllToAllMesh2Die::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[virRankSize - 1], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[virRankSize - 1], 1 << TOKEN_XN_ID);
    }
    uint32_t waitBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (const ChannelHandle &channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, waitBits);
    }
    return;
}

void CcuKernelAllToAllMesh2Die::PostSync()
{
    for (const auto &channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (const auto &channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return;
}

uint32_t CcuKernelAllToAllMesh2Die::CalcDstRank(uint32_t peerId) const
{
    if (peerId > rankGroup_.size()) {
        HCCL_ERROR("[CcuKernelAllToAllMesh2Die][CalcDstRank] Unexpected peerId[%u]", peerId);
    }
    return rankGroup_[peerId];
}

void CcuKernelAllToAllMesh2Die::DoRepeatAllToAll()
{
    // 创建GSA， src为本地的各片HBM地址GSA列表，dst为所有对端的HBM地址GSA列表
    std::vector<hcomm::CcuRep::LocalAddr> src;
    for (uint64_t rankIdx = 0; rankIdx < logicRankSize; rankIdx++) {
        src.emplace_back(CreateLocalAddr());
    }
    std::vector<hcomm::CcuRep::RemoteAddr> dst;
    for (uint64_t rankIdx = 0; rankIdx < logicRankSize; rankIdx++) {
        dst.emplace_back(CreateRemoteAddr());
    }

    // 考虑stride信息
    for (uint64_t r = 0; r < logicRankSize; r++) {
        const u32 dstRank = CalcDstRank(r);

        src[r].token = token_[r];
        dst[r].token = token_[r];

        src[r].addr = input_;

        dst[r].addr = output_[r];
        dst[r].addr += outputoffset_;
        for(uint64_t i = 0; i < dstRank; i++){
            src[r].addr += inputSliceStride_;
        }
    }

    hcomm::CcuRep::LocalAddr localSrc_ = CreateLocalAddr(); // for LocalCopy
    hcomm::CcuRep::LocalAddr localDst_ = CreateLocalAddr();
    if(withMyRank_){
        const u32 with_dstRank = CalcDstRank(logicRankSize - 1);

        localSrc_.token = token_[logicRankSize - 1];
        localSrc_.addr = input_;

        localDst_.token = token_[logicRankSize - 1];
        localDst_.addr = output_[logicRankSize - 1];
        localDst_.addr += outputoffset_;
        for(uint64_t i = 0; i < with_dstRank; i++){
            localSrc_.addr += inputSliceStride_;
        }
    }

    //  all2all 数据搬运
    u32 channelsIdx = 0;
    for (uint64_t r = 0; r < logicRankSize; r++) {
        if (withMyRank_ && r == logicRankSize - 1) {
            GroupCopy(localDst_, localSrc_, groupOpSize_);
            continue;
        }
        event_.SetMask(1 << r);
        WriteNb(channels_[channelsIdx], dst[r], src[r], sliceSize_, event_);
        channelsIdx++;
    }
    event_.SetMask(withMyRank_ ? ((1 << logicRankSize) - 1) & (~(1 << channels_.size())) : (1 << logicRankSize) - 1);
    WaitEvent(event_);
}

HcclResult CcuKernelAllToAllMesh2Die::Algorithm()
{
    HCCL_INFO("[ccuAllToAllMesh2Die_kernel] AllToAllMesh2Die run.");
    InitResources();

    LoadArgs();

    PreSync();

    DoRepeatAllToAll();

    PostSync();
    HCCL_INFO("[ccuAllToAllMesh2Die_kernel] AllToAllMesh2Die end.");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllToAllMesh2Die::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllToAllMesh2Die *taskArg = dynamic_cast<const CcuTaskArgAllToAllMesh2Die *>(&arg);
    if (taskArg == nullptr) {
        HCCL_ERROR("[CcuKernelAllToAllMesh2Die] taskArg ptr is null. RankId[%u]", rankId_);
    }
    uint64_t inputAddr         = taskArg->inputAddr_;
    uint64_t outputAddr        = taskArg->outputAddr_;
    uint64_t tokenInfo         = taskArg->token_;
    uint64_t sliceSize         = taskArg->sliceSize_;
    uint64_t inputSliceStride  = taskArg->inputSliceStride_;
    uint64_t outputSliceStride = taskArg->outputSliceStride_ * rankId_;
    auto goSize = CalGoSize(sliceSize);
    std::vector<uint64_t> taskArgs ={inputAddr, outputAddr, tokenInfo, sliceSize, inputSliceStride, outputSliceStride,
            goSize[0],  goSize[1], goSize[2], goSize[3]};
    HCCL_INFO("[CcuKernelAllToAllMesh2Die] inputAddr[%llu], outputAddr[%llu], sliceSize[%llu], "
              "inputSliceStride[%llu], outputSliceStride[%llu].",
              inputAddr, outputAddr, sliceSize, inputSliceStride, outputSliceStride);

    return taskArgs;
}

}// namespace ops_hccl