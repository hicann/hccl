/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "ccu_kernel_all_gather_2dies_mesh_1D.h"
#include "ccu_kernel_alg_base.h"
namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr int CKE_IDX_2 = 2;
constexpr int CKE_IDX_3 = 3;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;

CcuKernelAllGather2DiesMesh1D::CcuKernelAllGather2DiesMesh1D(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGather2DiesMesh1D *kernelArg
        = dynamic_cast<const CcuKernelArgAllGather2DiesMesh1D*>(&arg);
    rankId_ = kernelArg->rankId_;//本rankId
    rankSize_ = kernelArg->dimSize_; //未划分的全部的dimSize
    rankIdGroup_ = kernelArg->rankIdGroup_; //划分后本kernel需要处理的rankIdGroup，序号存放需要是递增的，不包括自身rankId
    ifHandleSelfRank_ = kernelArg->ifHandleSelfRank_;//是否需要处理本rank的数据搬运
    channels_ = kernelArg->channels;//划分后的channel，长度和位置都与rankIdGroup_对齐
}

HcclResult CcuKernelAllGather2DiesMesh1D::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGather2DiesMesh1D] CcuKernelAllGather2DiesMesh1D run");
    
    if (rankIdGroup_.size() == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(InitResource());//资源初始化
 
    LoadArgs();//参数载入
 
    PreSync();//前同步
 
    DoAllGather();//AllGather计算
 
    PostSync();//后同步
 
    HCCL_INFO("[CcuKernelAllGather2DiesMesh1D] CcuKernelAllGather2DiesMesh1D end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGather2DiesMesh1D::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const CcuTaskArgAllGather2DiesMesh1D *taskArg
        = dynamic_cast<const CcuTaskArgAllGather2DiesMesh1D *>(&arg);
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t sliceSize = taskArg->sliceSize_;
    uint64_t offSet = taskArg->offSet_;
    uint64_t token = taskArg->token_;
    auto groupOpSize = CalGoSize(sliceSize);

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        sliceSize,
        offSet,
        token,
        groupOpSize[0],
        groupOpSize[1],
        groupOpSize[2],
        groupOpSize[3]
    };
    HCCL_INFO("[CcuKernelAllGather2DiesMesh1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], sliceSize[%llu], offSet[%llu]",
               inputAddr, outputAddr, sliceSize, offSet);
    return taskArgs;
}

HcclResult CcuKernelAllGather2DiesMesh1D::InitResource()
{
    input_.push_back(CreateVariable());//两个kernel共用一个input
    uint16_t channelIdx = 0;
    for (uint32_t peerId = 0; peerId < channels_.size(); peerId++) {
        HCCL_DEBUG("[CcuKernelAllGather2DiesMesh1D] RankId[%u], PeerId[%u]", rankId_, peerId);
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

    offSet_ = CreateVariable();
    sliceSize_ = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGather2DiesMesh1D::LoadArgs()
{
    Load(input_[0]);
    Load(output_[channels_.size()]);  // 最后一个是自己的地址
    Load(sliceSize_);
    Load(offSet_);
    Load(token_[channels_.size()]);
    Load(groupOpSize_);
}

void CcuKernelAllGather2DiesMesh1D::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[channels_.size()], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[channels_.size()], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = ((1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID));
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelAllGather2DiesMesh1D::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGather2DiesMesh1D::DoAllGather()
{
    //load src addr
    hcomm::CcuRep::LocalAddr src = CreateLocalAddr();
    src.addr = input_[0];
    src.token = token_[channels_.size()];

    //dst structure for remote write init
    std::vector<hcomm::CcuRep::RemoteAddr> dst;
    for (uint64_t rankIdx = 0; rankIdx < rankIdGroup_.size() + 1; rankIdx++) {
        dst.push_back(CreateRemoteAddr());
    }

    for (uint64_t rankIdx = 0; rankIdx < rankIdGroup_.size() + 1; rankIdx++) {
        dst[rankIdx].addr = output_[rankIdx];
        dst[rankIdx].addr += offSet_;
        dst[rankIdx].token = token_[rankIdx];
    }

    //self defined broadcast
    if (ifHandleSelfRank_) {
        GroupBroadcast(channels_, dst, src, groupOpSize_);
    } else {
        GroupBroadcastWithoutMyRank(channels_, dst, src, groupOpSize_);
    }
}
}
