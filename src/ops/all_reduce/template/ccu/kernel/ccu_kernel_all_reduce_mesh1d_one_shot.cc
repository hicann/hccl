/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d_one_shot.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_CKE_IDX   = 0;
constexpr int PRE_SYNC_CKE_IDX    = 1;
constexpr uint16_t POST_CKE_BIT0  = 0;

CcuKernelAllReduceMesh1DOneShot::CcuKernelAllReduceMesh1DOneShot(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllReduceMesh1DOneShot *kernelArg = dynamic_cast<const CcuKernelArgAllReduceMesh1DOneShot *>(&arg);
    rankId_ = kernelArg->rankId_;
    rankSize_ = kernelArg->dimSize_;
    dataType_ = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    reduceOp_ = kernelArg->opParam_.reduceType;
    channels_ = kernelArg->channels;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] outputDataType is [INVALID], set outputDataType to[%u]",
            outputDataType_);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] Init, CtxArgs are rankId[%u], rankSize[%u], dataType[%u], "
        "outputDataType[%u], reduceOp[%u]", rankId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelAllReduceMesh1DOneShot::Algorithm()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] AllReduceMesh1DOneShot start");
    CHK_RET(InitResource());
    LoadArgs();  // 加载 taskArg 参数
    Presync();  // 跨卡前同步，交换参数信息

    DoGroupReduce();

    Postsync();  // 所有搬运任务结束后，跨卡后同步

    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] AllReduceMesh1DOneShot end");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceMesh1DOneShot::InitResource()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] InitResource start");
    // 初始化资源
    output_ = CreateVariable();
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        // THROW<NullPtrException>(StringFormat("CcuKernelAllReduceMesh1DOneShot channels is empty")); //打印
        return HcclResult::HCCL_E_INTERNAL; 
    }
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    groupOpSize_ = CreateGroupOpSize(); 
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] InitResource end");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllReduceMesh1DOneShot::LoadArgs()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] LoadArgs start");
    Load(input_[rankId_]);
    Load(output_);
    Load(token_[rankId_]);
    Load(groupOpSize_);
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] LoadArgs end");
}

void CcuKernelAllReduceMesh1DOneShot::Presync()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] Presync start");
    for (auto &channel : channels_) {
        NotifyRecord(channel, PRE_SYNC_CKE_IDX, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(channel, PRE_SYNC_CKE_IDX, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto &chn : channels_) {
        NotifyWait(chn, PRE_SYNC_CKE_IDX, allBit);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] Presync end");
}

void CcuKernelAllReduceMesh1DOneShot::Postsync()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] Postsync start");
    uint16_t postCkeBit = 1 << POST_CKE_BIT0;
    for (auto &ch : channels_) {
        NotifyRecord(ch, POST_SYNC_CKE_IDX, postCkeBit);
    }
    for (auto &channel : channels_) {
        NotifyWait(channel, POST_SYNC_CKE_IDX, postCkeBit);
    }
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] Postsync end");
}

void CcuKernelAllReduceMesh1DOneShot::DoGroupReduce()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] DoGroupReduce start");
    // 初始化地址寄存器
    std::vector<CcuRep::RemoteAddr> reduceSrc;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceSrc.push_back(CreateRemoteAddr());
    }
    CcuRep::LocalAddr reduceDst = CreateLocalAddr();

    // 填充地址
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
        reduceSrc[curId].addr = input_[rankIdx];
        reduceSrc[curId].token = token_[rankIdx];
    }

    // DST
    reduceDst.addr  = output_;
    reduceDst.token = token_[rankId_];

    // 执行 reduce 操作
    GroupReduce(channels_, reduceDst, reduceSrc, groupOpSize_, dataType_, outputDataType_, reduceOp_);
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] DoGroupReduce end");
    return;
}

std::vector<uint64_t> CcuKernelAllReduceMesh1DOneShot::GeneArgs(const CcuTaskArg &arg)
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] GeneArgs start");
    const CcuTaskArgAllReduceMesh1DOneShot *taskArg    = dynamic_cast<const CcuTaskArgAllReduceMesh1DOneShot *>(&arg);
    uint64_t                                inputAddr  = taskArg->inputAddr_;
    uint64_t                                outputAddr = taskArg->outputAddr_;
    uint64_t                                tokenInfo  = taskArg->token_;
    uint64_t                                sliceSize  = taskArg->sliceSize_;

    auto blockGoSize = CalGoSize(sliceSize);
 
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] GeneArgs, taskArg are inputAddr[%llu], outputAddr[%llu], "
        "sliceSize[%llu]", inputAddr, outputAddr, sliceSize);
 
    std::vector<uint64_t> taskArgList{inputAddr, outputAddr, tokenInfo};

    for (auto val : blockGoSize) {
        taskArgList.push_back(val);
    }
    
    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] GeneArgs end");
    return taskArgList;
}
}