/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_v_mesh1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"


namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;
constexpr int POST_SYNC_ID = 3;  

CcuKernelAllGatherVMesh1DMem2Mem::CcuKernelAllGatherVMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherVMesh1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgAllGatherVMesh1DMem2Mem *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;

    HCCL_INFO(
        "[CcuKernelAllGatherVMesh1DMem2Mem] Init, KernelArgs are rankId[%u], rankSize_[%u]",
        rankId_, rankSize_);
}

HcclResult CcuKernelAllGatherVMesh1DMem2Mem::InitResource()
{
    input_  = CreateVariable();
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllGatherVMesh1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelAllGatherVMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable outputVar, scratchVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    mySliceSize_ = CreateVariable();
    mySliceSizeOutputOffset_ = CreateVariable();

    src = CreateLocalAddr();
    localDst = CreateLocalAddr();
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            dst.push_back({});
        }
        else {
            dst.push_back(CreateRemoteAddr());
        }
    }
    event_ = CreateCompletedEvent();
    localGoSize_ = CreateGroupOpSize();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherVMesh1DMem2Mem::LoadArgs()
{
    Load(input_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(mySliceSize_);
    Load(mySliceSizeOutputOffset_);
    Load(localGoSize_);
    return;
}

void CcuKernelAllGatherVMesh1DMem2Mem::PreSync()
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

void CcuKernelAllGatherVMesh1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);         // bit index = 4, 用作后同步。cke都可以用同一个，所以都是CKE_IDX_0
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGatherVMesh1DMem2Mem::DoAllGatherV()
{
    
    uint32_t channelId = 0;
    CCU_IF(mySliceSize_ != 0) {
        for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            src.addr = input_;
            src.token = token_[rankId_];
            if (rankIdx == rankId_) {
                localDst.addr = output_[rankId_];
                localDst.addr += mySliceSizeOutputOffset_;
                localDst.token = token_[rankId_];
                event_.SetMask(1 << rankIdx);
                GroupCopy(localDst, src, localGoSize_);
                RecordEvent(event_);
            } else {
                dst[rankIdx].addr = output_[rankIdx];
                dst[rankIdx].addr += mySliceSizeOutputOffset_;
                dst[rankIdx].token = token_[rankIdx];
                CCU_IF(mySliceSize_ != 0)
                {
                    event_.SetMask(1 << rankIdx);
                    WriteNb(channels_[channelId], dst[rankIdx], src, mySliceSize_, event_);
                }
                channelId++;
            }
        }
        event_.SetMask((1 << rankSize_) - 1);
        WaitEvent(event_);
    }
}

HcclResult CcuKernelAllGatherVMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] AllgatheVrMesh1D run.");
    InitResource();
    LoadArgs();
    PreSync();
    DoAllGatherV();
    PostSync();
    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] AllgatheVrMesh1D end.");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherVMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherVMesh1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherVMesh1DMem2Mem *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t token                       = taskArg->token_;
    uint64_t mySliceSize                 = taskArg->mySliceSize_;
    uint64_t mySliceSizeOutputOffset     = taskArg->mySliceSizeOutputOffset_;
    auto goSize                          = CalGoSize(mySliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, mySliceSize, mySliceSizeOutputOffset,
        goSize[0], goSize[1], goSize[2], goSize[3]};

    HCCL_INFO("[CcuKernelAllGatherVMesh1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "mySliceSize[%llu], mySliceSizeOutputOffset[%llu], ",
        inputAddr, outputAddr, mySliceSize, mySliceSizeOutputOffset);
    return taskArgs;
}

} // namespace ops_hccl