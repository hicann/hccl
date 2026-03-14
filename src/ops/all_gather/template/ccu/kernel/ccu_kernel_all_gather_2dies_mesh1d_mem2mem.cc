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
#include "ccu_kernel_all_gather_2dies_mesh1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"
namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;

CcuKernelAllGather2DiesMeshMem2Mem1D::CcuKernelAllGather2DiesMeshMem2Mem1D(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGather2DiesMeshMem2Mem1D *kernelArg
        = dynamic_cast<const CcuKernelArgAllGather2DiesMeshMem2Mem1D*>(&arg);
    rankId_ = kernelArg->rankId_;
    rankSize_ = kernelArg->dimSize_;
    rankIdGroup_ = kernelArg->rankIdGroup_;
    ifHandleSelfRank_ = kernelArg->ifHandleSelfRank_;
    channels_ = kernelArg->channels;
}

HcclResult CcuKernelAllGather2DiesMeshMem2Mem1D::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGather2DiesMeshMem2Mem1D] CcuKernelAllGather2DiesMeshMem2Mem1D run");
    
    if (rankIdGroup_.size() == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(InitResource());
 
    LoadArgs();
 
    PreSync();
 
    DoAllGather();
 
    PostSync();
 
    HCCL_INFO("[CcuKernelAllGather2DiesMeshMem2Mem1D] CcuKernelAllGather2DiesMeshMem2Mem1D end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGather2DiesMeshMem2Mem1D::GeneArgs(const hcomm::CcuTaskArg &arg)
{
    const CcuTaskArgAllGather2DiesMeshMem2Mem1D *taskArg
        = dynamic_cast<const CcuTaskArgAllGather2DiesMeshMem2Mem1D *>(&arg);
    uint64_t inputAddr = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t sliceSize = taskArg->sliceSize_;
    uint64_t offSet = taskArg->offSet_;
    uint64_t token = taskArg->token_;
    auto localGoSize = CalGoSize(sliceSize);
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        sliceSize,
        offSet,
        token,
        localGoSize[0],
        localGoSize[1],
        localGoSize[2],
        localGoSize[3]
    };
    HCCL_INFO("[CcuKernelAllGather2DiesMeshMem2Mem1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], sliceSize[%llu], offSet[%llu]",
               inputAddr, outputAddr, sliceSize, offSet);
    return taskArgs;
}

HcclResult CcuKernelAllGather2DiesMeshMem2Mem1D::InitResource()
{
    localGoSize_ = CreateGroupOpSize();
    localCopyEvent_ = CreateCompletedEvent();
    event_ = CreateCompletedEvent();
    input_.push_back(CreateVariable());//两个kernel共用一个input
    uint16_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else if (peerId != rankIdGroup_[channelIdx]) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {//rankId == rankIdGroup_[channelIdx]
            HCCL_INFO("[CcuKernelArgAllGather2DiesMeshMem2Mem1D] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                rankId_, peerId, channelIdx);

            CcuRep::Variable outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            output_.push_back(outputVar);
            token_.push_back(tokenVar);
        }
        if (peerId >= rankIdGroup_[channelIdx] && channelIdx < rankIdGroup_.size() - 1) {
            channelIdx++;
        }
    }
    offSet_ = CreateVariable();
    sliceSize_ = CreateVariable();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGather2DiesMeshMem2Mem1D::LoadArgs()
{
    Load(input_[0]);
    Load(output_[rankId_]);
    Load(sliceSize_);
    Load(offSet_);
    Load(token_[rankId_]);
    Load(localGoSize_);
}

void CcuKernelAllGather2DiesMeshMem2Mem1D::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = ((1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID));
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
}

void CcuKernelAllGather2DiesMeshMem2Mem1D::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGather2DiesMeshMem2Mem1D::DoAllGather()
{    
    hcomm::CcuRep::LocalAddr src = CreateLocalAddr();
    src.addr = input_[0];
    src.token = token_[rankId_];
    std::vector<hcomm::CcuRep::RemoteAddr> remoteDst;
    for (uint64_t rankIdx = 0; rankIdx < rankIdGroup_.size(); rankIdx++) {
        remoteDst.push_back(CreateRemoteAddr());
    }
    for (uint64_t rankIdx = 0; rankIdx < rankIdGroup_.size(); rankIdx++) { 
        event_.SetMask(1 << rankIdGroup_[rankIdx]); 
        remoteDst[rankIdx].addr = output_[rankIdGroup_[rankIdx]]; 
        remoteDst[rankIdx].addr += offSet_;
        remoteDst[rankIdx].token = token_[rankIdGroup_[rankIdx]];

        CCU_IF (sliceSize_ != 0) {
            WriteNb(channels_[rankIdx], remoteDst[rankIdx], src, sliceSize_, event_);//write to the remote dst
        }
        CCU_IF (sliceSize_ == 0) {
            RecordEvent(event_);
        }
    }
    localCopyEvent_.SetMask(1 << rankId_);
    if (ifHandleSelfRank_) {
        hcomm::CcuRep::LocalAddr localDst = CreateLocalAddr();
        localDst.addr = output_[rankId_];
        localDst.token = token_[rankId_];
        localDst.addr += offSet_;
        GroupCopy(localDst, src, localGoSize_);
    }
    RecordEvent(localCopyEvent_);
    uint16_t rankMask = 0x0000;
    for (uint64_t rankIdx = 0; rankIdx < rankIdGroup_.size(); rankIdx++) {
        rankMask |= (1 << rankIdGroup_[rankIdx]);
    }
    event_.SetMask(rankMask);
    WaitEvent(event_);
}
}
