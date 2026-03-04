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
#include "ccu_kernel_all_reduce_mesh1d.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0    = 0;

CcuKernelAllReduceMesh1D::CcuKernelAllReduceMesh1D(const CcuKernelArg &arg)
                                                    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllReduceMesh1D *kernelArg = dynamic_cast<const CcuKernelArgAllReduceMesh1D *>(&arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("CcuKernelAllReduceMesh1D::kernelArg ptr is null");
    }
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_[0];
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    channels_       = kernelArg->channels;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_INFO("[CcuKernelAllReduceMesh1D] outputDataType is [HCCL_DATA_TYPE_RESERVED], set outputDataType to[%d]",
            outputDataType_);
    }

    reduceOp_ = kernelArg->opParam_.reduceType;
    HCCL_DEBUG("[CcuKernelAllReduceMesh1D] Init, CtxArgs are rankId[%u], rankSize[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]", rankId_, rankSize_, dataType_,
        outputDataType_, reduceOp_);
}

void CcuKernelAllReduceMesh1D::RunBroadcast(std::vector<CcuRep::RemoteAddr> &dst, CcuRep::LocalAddr &src)
{
    GroupBroadcast(channels_, dst, src, groupOpSize_);
}

void CcuKernelAllReduceMesh1D::RunReduce(CcuRep::LocalAddr &dst, std::vector<CcuRep::RemoteAddr> &src)
{
    GroupReduce(channels_, dst, src, groupOpSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelAllReduceMesh1D::Algorithm()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1D] AllReduceMesh1D run");
    // 初始化资源
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("CcuKernelAllReduceMesh1D channels_ is empty");
    }
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO("[CcuKernelAllReduceMesh1D] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, tokenVar, outputVar;
            CreateVariable((channels_[channelIdx]), INPUT_XN_ID, &inputVar);
            CreateVariable((channels_[channelIdx]), OUTPUT_XN_ID, &outputVar);
            CreateVariable((channels_[channelIdx]), TOKEN_XN_ID, &tokenVar);
            input_.push_back(inputVar);
            output_.push_back(outputVar);
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    offSet_      = CreateVariable();
    groupOpSize_ = CreateGroupOpSize();

    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(offSet_);

    Load(groupOpSize_);

    for (auto t : channels_) {
        NotifyRecord(t, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(t, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(t, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint16_t syncBit = 1 << INPUT_XN_ID | 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto t : channels_) {
        NotifyWait(t, CKE_IDX_0, syncBit);
    }

    std::vector<CcuRep::RemoteAddr> reduceScatterSrc;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterSrc.push_back(CreateRemoteAddr());
    }
    CcuRep::LocalAddr reduceScatterDst = CreateLocalAddr();
    // DST
    reduceScatterDst.addr  = output_[rankId_];
    reduceScatterDst.addr += offSet_;
    reduceScatterDst.token = token_[rankId_];

    uint32_t dstId = 0;
    uint32_t curId = 0;
    // SRC
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        reduceScatterSrc[curId].addr = input_[rankIdx];
        reduceScatterSrc[curId].addr += offSet_;
        reduceScatterSrc[curId].token = token_[rankIdx];
    }
    RunReduce(reduceScatterDst, reduceScatterSrc);

    CcuRep::LocalAddr allGatherSrc = CreateLocalAddr();
    std::vector<CcuRep::RemoteAddr> allGatherDst;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        allGatherDst.push_back(CreateRemoteAddr());
    }
    // allGather 的输入就是 reduceScatter 的输出
    allGatherSrc.addr  = output_[rankId_];
    allGatherSrc.addr  += offSet_;
    allGatherSrc.token = token_[rankId_];

    dstId = 0;
    curId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx != rankId_) {
            curId = dstId;
            dstId++;
        } else {
            curId = rankSize_ - 1;
        }
        allGatherDst[curId].addr = output_[rankIdx];
        allGatherDst[curId].addr += offSet_;
        allGatherDst[curId].token = token_[rankIdx];
    }
    RunBroadcast(allGatherDst, allGatherSrc);

    for (auto t : channels_) {
        NotifyRecord(t, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto t : channels_) {
        NotifyWait(t, CKE_IDX_0, 1 << POST_SYNC_ID);
    }

    HCCL_INFO("[CcuKernelAllReduceMesh1D] AllReduceMesh1D end");
    return HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllReduceMesh1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllReduceMesh1D *taskArg = dynamic_cast<const CcuTaskArgAllReduceMesh1D *>(&arg);
    if (taskArg == nullptr) {
        HCCL_ERROR("CcuKernelAllReduceMesh1D::taskArg ptr is null");
    }
    uint64_t inputAddr  = taskArg->inputAddr_;
    uint64_t outputAddr = taskArg->outputAddr_;
    uint64_t tokenInfo  = taskArg->token_;
    uint64_t sliceSize  = taskArg->sliceSize_;
    uint64_t offset     = taskArg->offSet_;

    auto goSize = CalGoSize(sliceSize);

    HCCL_INFO("[CcuKernelAllReduceMesh1D] GeneArgs, taskArg are inputAddr[%llu], outputAddr[%llu], "
        "offset[%llu], sliceSize[%llu]", inputAddr, outputAddr, offset, sliceSize);
    return {inputAddr, outputAddr, tokenInfo, offset, goSize[0], goSize[1], goSize[2], goSize[3]};
}
}