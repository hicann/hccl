/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_mesh1d_multi_jetty.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID   = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0     = 0;

CcuKernelAllToAllMesh1DMultiJetty::CcuKernelAllToAllMesh1DMultiJetty(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllToAllMesh1DMultiJetty *kernelArg
        = dynamic_cast<const CcuKernelArgAllToAllMesh1DMultiJetty *>(&arg);
    rankId_         = kernelArg->rankId_;
    channels_       = kernelArg->channels;
    jettyNums_      = kernelArg->jettyNums_;
    rankSize_       = kernelArg->rankSize_;
    HCCL_INFO(
        "[CcuKernelAllToAllMesh1DMultiJetty] Init, KernelArgs are rankId[%u], rankSize_[%u]",
        rankId_, rankSize_);
}

std::vector<uint64_t> CcuKernelAllToAllMesh1DMultiJetty::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllToAllMesh1DMultiJetty *taskArg
        = dynamic_cast<const CcuTaskArgAllToAllMesh1DMultiJetty *>(&arg);
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;
    uint64_t srcStride = taskArg->srcStride_;
    uint64_t srcOffset = taskArg->srcOffset_;
    uint64_t dstOffset = taskArg->dstOffset_;
    uint64_t sliceSize  = taskArg->sliceSize_;
    std::vector<uint64_t> jettySlice = taskArg->jettySlice_;
    std::vector<uint64_t> jettySliceTail = taskArg->jettySliceTail_;
    auto     goSize     = CalGoSize(sliceSize);
    HCCL_INFO("[%s] inputAddr[%llu], outputAddr[%llu], sliceSize[%llu], srcStride[%llu], srcOffset[%llu], "
              "dstOffset[%llu].", __FUNCTION__,
              inputAddr, outputAddr, sliceSize, srcStride, srcOffset, dstOffset);
    for (u32 i = 0; i < jettySlice.size(); i++) {
        HCCL_INFO("[%s]jettySlice[%u]=%lu, jettySliceTail[%u]=%lu", __FUNCTION__, 
                i, jettySlice[i], i, jettySliceTail[i]);
    }
    std::vector<uint64_t> res = {inputAddr, outputAddr, tokenInfo, sliceSize, srcStride, srcOffset, dstOffset,
                                goSize[0], goSize[1], goSize[2], goSize[3]};
    res.insert(res.end(), jettySlice.begin(), jettySlice.end());
    res.insert(res.end(), jettySliceTail.begin(), jettySliceTail.end());
    return res;
}

HcclResult CcuKernelAllToAllMesh1DMultiJetty::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllToAllMesh1DMultiJetty] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelAllToAllMesh1DMultiJetty] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, scratchVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &scratchVar));
            output_.push_back(scratchVar);
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

    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        eventList_.push_back(CreateCompletedEvent());
        jettySlice_.push_back(CreateVariable());
    }

    for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
        jettySliceTail_.push_back(CreateVariable());
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] InitResource success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllMesh1DMultiJetty::PreSync()
{
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID));
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID));
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyWait(channel, CKE_IDX_0, allBit));
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] PreSync success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllMesh1DMultiJetty::PostSync()
{
    for (auto &ch : channels_) {
        CHK_RET(NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (auto &ch : channels_) {
        CHK_RET(NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] PostSync success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllMesh1DMultiJetty::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(sliceSize_);
    Load(srcStride_);
    Load(srcOffset_);
    Load(dstOffset_);
    Load(groupOpSize_);
    for (int i = 0; i < rankSize_; i++) {
        Load(jettySlice_[i]);
    }
    for (int i = 0; i < rankSize_; i++) {
        Load(jettySliceTail_[i]);
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] LoadArgs success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllMesh1DMultiJetty::CalcAddrs()
{
    srcOffset_ += input_[rankId_];
    remoteSrc_.resize(rankSize_);
    remoteDst_.resize(rankSize_);

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            localSrc_ = CreateLocalAddr();
            localDst_ = CreateLocalAddr();
            localSrc_.addr = srcOffset_;
        for (uint32_t i = 0; i < rankIdx; i++) {
                localSrc_.addr += srcStride_;
            }
            localSrc_.token = token_[rankIdx];
            localDst_.addr = output_[rankIdx];
            localDst_.token = token_[rankIdx];
            localDst_.addr += dstOffset_;
        } else {
            remoteSrc_[rankIdx] = CreateLocalAddr();
            remoteDst_[rankIdx] = CreateRemoteAddr();
            remoteSrc_[rankIdx].addr = srcOffset_;
            for (uint32_t i = 0; i < rankIdx; i++) {
                remoteSrc_[rankIdx].addr += srcStride_;
            }
            remoteSrc_[rankIdx].token = token_[rankIdx];
            remoteDst_[rankIdx].addr = output_[rankIdx];
            remoteDst_[rankIdx].token = token_[rankIdx];
            remoteDst_[rankIdx].addr += dstOffset_;
        }
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] CalcAddrs success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllMesh1DMultiJetty::SendData(ChannelHandle channel, hcomm::CcuRep::RemoteAddr remoteDst,
    hcomm::CcuRep::LocalAddr remoteSrc, hcomm::CcuRep::Variable sliceLength, hcomm::CcuRep::CompletedEvent event)
{
    CCU_IF(sliceLength != 0) {
        CHK_RET(WriteNb(channel, remoteDst, remoteSrc, sliceLength, event));
    }
    CCU_IF(sliceLength == 0) {
        CHK_RET(RecordEvent(event));
    }
    return HcclResult::HCCL_SUCCESS;;
}

HcclResult CcuKernelAllToAllMesh1DMultiJetty::DoAllToAll()
{
    uint32_t channelId = 0;
    for (uint64_t r = 0; r < rankSize_; r++) {
        if (r == rankId_) {
            continue;
        }
        for (uint32_t jettyIdx = 0; jettyIdx < jettyNums_[r]; jettyIdx++) {
            eventList_[r].SetMask(1 << jettyIdx);
            if (jettyIdx == (jettyNums_[r] - 1)) {
                CHK_RET(SendData(channels_[channelId], remoteDst_[r], remoteSrc_[r], jettySliceTail_[r], eventList_[r]));
            } else {
                CHK_RET(SendData(channels_[channelId], remoteDst_[r], remoteSrc_[r], jettySlice_[r], eventList_[r]));
            }
            remoteDst_[r].addr += jettySlice_[r];
            remoteSrc_[r].addr += jettySlice_[r];
        }
        channelId++;
    }

    // 本地拷贝以及同步等待
    CHK_RET(GroupCopy(localDst_, localSrc_, groupOpSize_));
    eventList_[rankId_].SetMask(1);
    CHK_RET(RecordEvent(eventList_[rankId_]));

    for (uint64_t r = 0; r < rankSize_; r++) {
        if (r == rankId_) {
            eventList_[r].SetMask(1);
        } else {
            eventList_[r].SetMask((1 << jettyNums_[r]) - 1);
        }
        CHK_RET(WaitEvent(eventList_[r]));
    }

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] DoAllToAll success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllToAllMesh1DMultiJetty::Algorithm()
{
    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] AllToAllMesh1DMultiJetty run");

    CHK_RET(InitResource());

    CHK_RET(LoadArgs());

    CHK_RET(PreSync());

    CHK_RET(CalcAddrs());

    CHK_RET(DoAllToAll());

    CHK_RET(PostSync());

    HCCL_INFO("[CcuKernelAllToAllMesh1DMultiJetty] AllToAllMesh1DMultiJetty end");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl