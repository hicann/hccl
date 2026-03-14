/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_broadcast_mesh1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0;
constexpr int CKE_IDX_1    = 1;
constexpr int CKE_IDX_3    = 3;
constexpr int CKE_IDX_4    = 4;

CcuKernelBroadcastMesh1DMem2Mem::CcuKernelBroadcastMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgBroadcastMesh1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgBroadcastMesh1DMem2Mem *>(&arg);
    rankId_         = kernelArg->rankId_;
    rootId_         = kernelArg->rootId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelBroadcastMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] Init, KernelArgs are rootId[%d] rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d] channelsSize[%zu]",
        rootId_, rankId_, rankSize_, dataType_, outputDataType_, channels_.size());
}

HcclResult CcuKernelBroadcastMesh1DMem2Mem::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelBroadcastMesh1DMem2Mem] channels is empty!");
        return HCCL_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelBroadcastMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, outputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], OUTPUT_XN_ID, &outputVar));
            output_.push_back(outputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    currentRankSliceInputOffset_  = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    inputRepeatStride_            = CreateVariable();
    outputRepeatStride_           = CreateVariable();
    normalSliceSize_              = CreateVariable();
    lastSliceSize_                = CreateVariable();
    allgatherOffset_              = CreateVariable();
    repeatNumVar_                 = CreateVariable();
    flag_                         = CreateVariable();
    SliceOffset_                  = CreateVariable();

    scatterdstMem_.reserve(rankSize_);
    allgatherdstMem_.reserve(rankSize_);
    scattersrcMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        scattersrcMem_.push_back(CreateLocalAddr());
        if (rankIdx == rankId_) {
            myScatterDst_ = CreateLocalAddr();
            scatterdstMem_.push_back({});
            allgatherdstMem_.push_back({});
        } else {
            allgatherdstMem_.push_back(CreateRemoteAddr());
            scatterdstMem_.push_back(CreateRemoteAddr());
        }
    }

    event_ = CreateCompletedEvent();
    return HCCL_SUCCESS;
}

void CcuKernelBroadcastMesh1DMem2Mem::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(allgatherOffset_);
    Load(repeatNumVar_);
    return;
}

void CcuKernelBroadcastMesh1DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] BroadcastMesh1D LocalPost begin");
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID); // index = 1，传递input信息
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID); // index = 0，传递output信息
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID); // index = 2，传递token信息
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] BroadcastMesh1D wait all end");
    return;
}

void CcuKernelBroadcastMesh1DMem2Mem::PostSync(int CKE_id)
{
    for (auto &ch : channels_) {
        NotifyRecord(ch, CKE_IDX_0, 1 << CKE_id);
    }
    for (auto &ch : channels_) {
        NotifyWait(ch, CKE_IDX_0, 1 << CKE_id);
    }
    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] BroadcastMesh1D PostSync end");
    return;
}

void CcuKernelBroadcastMesh1DMem2Mem::DoRepeaScatterMem2Mem()
{
    if (rankId_ != rootId_) {
        return;
    }
    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem][DoRepeaScatterMem2Mem]rankId[%u] rankSize[%u]",rankId_ , rankSize_);
    std::vector<CcuRep::RemoteAddr> &dst = scatterdstMem_;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == 0) {
            SliceOffset_ = 0;
        } else {
            SliceOffset_ += normalSliceSize_;
        }
        scattersrcMem_[rankIdx].addr = input_[rankId_];
        scattersrcMem_[rankIdx].addr += currentRankSliceInputOffset_;
        scattersrcMem_[rankIdx].addr += SliceOffset_;
        scattersrcMem_[rankIdx].token = token_[rankId_];
        if (rankIdx == rankId_) {
            myScatterDst_.addr = output_[rankIdx];
            myScatterDst_.addr += currentRankSliceOutputOffset_;
            myScatterDst_.addr += SliceOffset_;
            myScatterDst_.token = token_[rankIdx];
        } else {
            dst[rankIdx].addr = output_[rankIdx];
            dst[rankIdx].addr += currentRankSliceOutputOffset_;
            dst[rankIdx].addr += SliceOffset_;
            dst[rankIdx].token = token_[rankIdx];
        }

        CCU_IF(flag_ == 1)
        {
            // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
            myScatterDst_.addr += outputRepeatStride_;
            for (uint32_t curId = 0; curId < rankSize_; curId++) {
                scattersrcMem_[curId].addr += inputRepeatStride_;
                if (curId != rankId_) {
                    dst[curId].addr += outputRepeatStride_;
                }
            }
        }
    }
    DoScatter(dst);
    return;
}

void CcuKernelBroadcastMesh1DMem2Mem::DoScatter(const std::vector<CcuRep::RemoteAddr> &dst)
{
    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem][DoScatter] DoScatter");
    uint64_t channelId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        auto &sliceSize = (rankIdx + 1 == rankSize_) ? lastSliceSize_ : normalSliceSize_;
        event_.SetMask(1 << rankIdx);
        CCU_IF(sliceSize != 0)
        {
            //rootid
            if (rankIdx == rankId_) {
                RecordEvent(event_);
            } else {
                WriteNb(channels_[channelId], dst[rankIdx], scattersrcMem_[rankIdx], sliceSize, event_);
                HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem][DoScatter] channelsId[%llu] rankIdx[%u]",
                            channelId, rankIdx);
                channelId++;
            }
        }
        CCU_IF(sliceSize == 0)
        {
            RecordEvent(event_);
        }
    }

    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
    return;
}

void CcuKernelBroadcastMesh1DMem2Mem::DoRepeatAllGatherMem2Mem()
{
    CcuRep::LocalAddr              &src = myScatterDst_;
    std::vector<CcuRep::RemoteAddr> &dst = allgatherdstMem_;
    src.addr                            = output_[rankId_];
    src.addr                           += currentRankSliceOutputOffset_;
    src.addr                           += allgatherOffset_;
    src.token                           = token_[rankId_];

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            dst[rankIdx].addr = output_[rankIdx];
            dst[rankIdx].addr += currentRankSliceOutputOffset_;
            dst[rankIdx].addr += allgatherOffset_;
            dst[rankIdx].token = token_[rankIdx];
        }
    }
    CCU_IF(flag_ == 1)
    {
        //  非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        src.addr += inputRepeatStride_;
        for (uint32_t curId = 0; curId < rankSize_; curId++) {
            if (curId != rankId_) {
                dst[curId].addr += outputRepeatStride_;
            }
        }
    }

    DoAllGather(src, dst);
    return;
}

void CcuKernelBroadcastMesh1DMem2Mem::DoAllGather(const CcuRep::LocalAddr             &src,
                                                             const std::vector<CcuRep::RemoteAddr> &dst)
{
    uint64_t channelId = 0;
    auto    &sliceSize   = (rankId_ + 1 == rankSize_) ? lastSliceSize_ : normalSliceSize_;
    CCU_IF(sliceSize != 0)
    {
        for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
            event_.SetMask(1 << rankIdx);
            if (rankIdx == rankId_) {
                RecordEvent(event_);
            } else {
                WriteNb(channels_[channelId], dst[rankIdx], src, sliceSize, event_);
                channelId++;
            }
        }
        event_.SetMask((1 << rankSize_) - 1);
        WaitEvent(event_);
    }
    return;
}

HcclResult CcuKernelBroadcastMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] BroadcastMesh1D run");
    CHK_RET(InitResource());
    LoadArgs();
    PreSync();

    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd                  = 1;
    flag_ = 0;
    CCU_WHILE(repeatNumVar_ != UINT64_MAX)
    {
        // 循环repeatNum_次
        DoRepeaScatterMem2Mem();
        PostSync(CKE_IDX_3);
        DoRepeatAllGatherMem2Mem();
        PostSync(CKE_IDX_4);
        repeatNumVar_ += repeatNumAdd;
        flag_ = 1;
    }

    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] BroadcastMesh1D end");
    return HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelBroadcastMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgBroadcastMesh1DMem2Mem *taskArg = dynamic_cast<const CcuTaskArgBroadcastMesh1DMem2Mem *>(&arg);
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;

    uint64_t              currentRankSliceInputOffset  = taskArg->inputSliceStride_ * rankId_;
    uint64_t              currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t              inputRepeatStride            = taskArg->inputRepeatStride_;
    uint64_t              outputRepeatStride           = taskArg->outputRepeatStride_;
    uint64_t              normalSliceSize              = taskArg->normalSliceSize_;
    uint64_t              lastSliceSize                = taskArg->lastSliceSize_;
    uint64_t              allgatherOffset              = taskArg->normalSliceSize_ * rankId_;
    uint64_t              repeatNumVar                 = taskArg->repeatNumVar_;
    std::vector<uint64_t> taskArgs                     = {
        inputAddr,
        outputAddr,
        tokenInfo,
        currentRankSliceInputOffset,
        currentRankSliceOutputOffset,
        inputRepeatStride,
        outputRepeatStride,
        normalSliceSize,
        lastSliceSize,
        allgatherOffset,
        repeatNumVar,
    };

    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] TaskArgs: inputAddr[%llx], outputAddr[%llx], "
              "currentRankSliceInputOffset[%llu], "
              "currentRankSliceOutputOffset[%llu], inputRepeatStride[%llu], outputRepeatStride[%llu], "
              "normalSliceSize[%llu], lastSliceSize[%llu], allgatherSliceSize[%llu], repeatNumVar[%llu] ",
              inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, inputRepeatStride,
              outputRepeatStride, normalSliceSize, lastSliceSize, allgatherOffset, repeatNumVar);
    return taskArgs;
}

} // namespace ops_hccl