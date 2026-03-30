/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_all_to_all_mesh1d.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0;
constexpr int CKE_IDX_1    = 1;
constexpr int CKE_IDX_2    = 2;

constexpr uint64_t CCU_MS_SIZE   = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;

CcuKernelAlltoAllMesh1D::CcuKernelAlltoAllMesh1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAlltoAllMesh1D *kernelArg
        = dynamic_cast<const CcuKernelArgAlltoAllMesh1D *>(&arg);

    rankId_ = kernelArg->rankId_;
    channels_ = kernelArg->channels;
    rankSize_ = kernelArg->dimSize_;
    loadFromMem_ = kernelArg->loadFromMem_;
}

HcclResult CcuKernelAlltoAllMesh1D::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAlltoAllMesh1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelAlltoAllMesh1D] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    sliceSize_   = CreateVariable();
    srcStride_   = CreateVariable();
    srcOffset_   = CreateVariable();
    dstOffset_   = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();

    // 创建GSA， src为本地的各片HBM地址GSA列表，dst为所有对端的HBM地址GSA列表
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        srcAddr_.push_back(CreateLocalAddr());
    }
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            myDst_ = CreateLocalAddr();
            dstAddr_.push_back({});
        } else {
            dstAddr_.push_back(CreateRemoteAddr());
        }
    }

    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAlltoAllMesh1D::LoadArgs()
{
    // 从SQE load args，本rank需要的input、output地址等信息
    // inputAddr, outputAddr, tokenInfo, srcStride, srcOffset, dstOffset, groupOpSize
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(sliceSize_);  // 本轮传输的分片大小
    Load(srcStride_); // 单片数据大小
    Load(srcOffset_);
    Load(dstOffset_);
    Load(groupOpSize_);

    srcOffset_ += input_[rankId_];
    return;
}

void CcuKernelAlltoAllMesh1D::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID); // index = 1，传递input信息
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    
    uint16_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (auto &ch: channels_) {
        NotifyWait(ch, CKE_IDX_0, allBit);
    }
    return;
}

void CcuKernelAlltoAllMesh1D::PostSync()
{
    uint16_t postBit = 1 << 5;
    for (auto &ch : channels_) {
        NotifyRecord(ch, CKE_IDX_1, postBit);
    }

    for (auto &ch : channels_) {
        NotifyWait(ch, CKE_IDX_1, postBit);
    }
}

void CcuKernelAlltoAllMesh1D::DoAlltoAll()
{
    HCCL_INFO("DoAlltoAll Start.");
    // 考虑stride信息，即是每片数据大小
    for (uint64_t r = 0; r < rankSize_; r++) {
        // src[r] = srcOffset + r*srcStride, 每块数据进行了分片
        // dst[r] = recvBuf[r] + dstOffset
        if (r == rankId_) {
            srcAddr_[r].token = token_[r];
            myDst_.token = token_[r];
            myDst_.addr = output_[r];
            myDst_.addr += dstOffset_;
        } else {
            srcAddr_[r].token = token_[r];
            dstAddr_[r].token = token_[r];
            dstAddr_[r].addr = output_[r];
            dstAddr_[r].addr += dstOffset_;
        }

        srcAddr_[r].addr = srcOffset_;
        for (uint64_t i = 0; i < r; i++) {
            srcAddr_[r].addr += srcStride_;
        }
    }

    uint32_t channelId = 0;
    uint16_t allBit = ((1 << rankSize_) - 1) & (~(1 << rankId_)); // 仅rankid位为0，其他位为1，代表远端准备好了

    if (loadFromMem_) {
        for(uint64_t r = 0; r < rankSize_; r++) {
            event_.SetMask(1 << r);
            if (r == rankId_) {
                LocalCopyNb(myDst_, srcAddr_[r], sliceSize_, event_);
            }
            else {
                WriteNb(channels_[channelId], dstAddr_[r], srcAddr_[r], sliceSize_, event_);
                channelId++;
            }
        }
        // 等读完所有对端
        event_.SetMask((1 << rankSize_) - 1);
        WaitEvent(event_);
    } else {
        for(uint64_t r = 0; r < rankSize_; r++) {
            event_.SetMask(1 << r);
            if (r != rankId_) {
                WriteNb(channels_[channelId], dstAddr_[r], srcAddr_[r], sliceSize_, event_);
                channelId++;
            }
        }
        GroupCopy(myDst_, srcAddr_[rankId_], groupOpSize_);
        event_.SetMask(allBit);
        WaitEvent(event_);
    }
}

HcclResult CcuKernelAlltoAllMesh1D::Algorithm()
{
    HCCL_INFO("[ccuAllToAllMesh1D_context] AllToAllMesh1D run.");

    CHK_RET(InitResource());

    LoadArgs();

    PreSync();

    DoAlltoAll();

    PostSync();

    HCCL_INFO("[AllToAllAlgo] AllToAllMesh1D end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAlltoAllMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAlltoAllMesh1D *taskArg
        = dynamic_cast<const CcuTaskArgAlltoAllMesh1D *>(&arg);
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;

    uint64_t srcStride = taskArg->srcStride_;
    uint64_t srcOffset = taskArg->srcOffset_;
    uint64_t dstOffset = taskArg->dstOffset_;

    uint64_t sliceSize  = taskArg->sliceSize_;
    auto     goSize     = CalGoSize(sliceSize);

    std::vector<uint64_t> taskArgs = {
        inputAddr,  outputAddr,  tokenInfo, sliceSize, 
        srcStride,  srcOffset,
        dstOffset, goSize[0], goSize[1], goSize[2], goSize[3]
    };

    HCCL_INFO("[CcuKernelAlltoAllMesh1D] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
               "srcStride[%llu], srcOffset[%llu],"
               "dstOffset[%llu], sliceSize[%llu], goSize: [%llu], [%llu], [%llu], [%llu]",
               inputAddr, outputAddr, srcStride, srcOffset,
               dstOffset, sliceSize, goSize[0], goSize[1], goSize[2], goSize[3]);
    return taskArgs;
}

}// namespace ops_hccl