/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "ccu_kernel_reduce_scatter_mesh2die.h"

 namespace ops_hccl {

constexpr int CKE_IDX_0   = 0;
constexpr int INPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int LOOP_NUM    = 128;

constexpr int     MISSION_NUM = 2;

CcuKernelReduceScatterMesh2Die::CcuKernelReduceScatterMesh2Die(const hcomm::CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterMesh2Die *kernelArg = dynamic_cast<const CcuKernelArgReduceScatterMesh2Die *>(&arg);

    channels_ = kernelArg->channels;
    rmtReduceWithMyRank_ = kernelArg->rmtReduceWithMyRank_;
    myRankId_            = kernelArg->rankId_;
    rankSize_ = kernelArg->rankSize_;

     // 数据类型处理
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelReduceScatterMesh2Die] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
}

HcclResult CcuKernelReduceScatterMesh2Die::InitResources()
{
    myInput_   = CreateVariable();
    myOutput_  = CreateVariable();
    myScratch_ = CreateVariable();
    myToken_   = CreateVariable();

    for (u64 id = 0; id < channels_.size(); id++) {
        hcomm::CcuRep::Variable input, token;
        CHK_RET(CreateVariable(channels_[id], INPUT_XN_ID, &input));
        peerInput_.emplace_back(input);
        CHK_RET(CreateVariable(channels_[id], TOKEN_XN_ID, &token));
        peerToken_.emplace_back(token);
    }

    sliceSize_ = CreateVariable();
    rmtReduceSliceOffset_ = CreateVariable();
    rmtReduceGoSize_    = CreateGroupOpSize();
    rmtReduceRankNum_ = channels_.size() + (rmtReduceWithMyRank_ == true ? 1 : 0);
    rmtSyncMyBit_ = 1 << (myRankId_ % rmtReduceRankNum_);
    rmtSyncWaitBit_
        = rmtReduceWithMyRank_ ? ((1 << rmtReduceRankNum_) - 1) & (~rmtSyncMyBit_) : (1 << rmtReduceRankNum_) - 1;

    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelReduceScatterMesh2Die::LoadArgs()
{
    Load(myInput_);
    Load(myOutput_);
    Load(myToken_);
    Load(myScratch_);
    Load(sliceSize_);
    Load(rmtReduceSliceOffset_);
    Load(rmtReduceGoSize_);
}

void CcuKernelReduceScatterMesh2Die::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, myInput_, 1<<INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, myToken_, 1 << TOKEN_XN_ID);
    }
    uint32_t waitBits = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (const ChannelHandle &channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, waitBits);
    }
    return;
}

void CcuKernelReduceScatterMesh2Die::PostSync()
{
    for (const auto &channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (const auto &channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return;
}

void CcuKernelReduceScatterMesh2Die::RmtReduce()
{
    //读操作，本端地址为dst，远端地址为src
    std::vector<hcomm::CcuRep::RemoteAddr> src;
    src.reserve(rmtReduceRankNum_);
    for (uint32_t peerIdx = 0; peerIdx < channels_.size(); peerIdx++){
        src.emplace_back(CreateRemoteAddr());
        src.back().token = peerToken_[peerIdx];
        src.back().addr  = peerInput_[peerIdx];
        src.back().addr += rmtReduceSliceOffset_;
    }
    if (rmtReduceWithMyRank_){
        src.emplace_back(CreateRemoteAddr());
        src.back().token = myToken_;
        src.back().addr  = myInput_;
        src.back().addr += rmtReduceSliceOffset_;
    }

    hcomm::CcuRep::LocalAddr dst = CreateLocalAddr();
    dst.token          = myToken_;
    dst.addr           = rmtReduceWithMyRank_ ? myOutput_ : myScratch_;

     if (rmtReduceWithMyRank_) {
        GroupReduce(channels_, dst, src, rmtReduceGoSize_, dataType_, outputDataType_, reduceOp_);
    } else {
        GroupReduceWithoutMyRank(channels_, dst, src, rmtReduceGoSize_, dataType_, outputDataType_, reduceOp_);
    }
}

HcclResult CcuKernelReduceScatterMesh2Die::Algorithm()
{
    HCCL_INFO("[ccuReduceScatterMesh2Die_kernel] ReduceScatterMesh2Die run.");
    InitResources();
    LoadArgs();
    PreSync();
    RmtReduce();
    PostSync();
    HCCL_INFO("[ccuReduceScatterMesh2Die_kernel] ReduceScatterMesh2Die end.");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelReduceScatterMesh2Die::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterMesh2Die *taskArg = dynamic_cast<const CcuTaskArgReduceScatterMesh2Die *>(&arg);
    moConfig.loopCount = LOOP_NUM;
    uint64_t myInput   = taskArg->inputAddr_;
    uint64_t myOutput  = taskArg->outputAddr_;
    uint64_t myToken   = taskArg->token_;
    uint64_t offsetSliceSize = taskArg->sliceSize_;
    uint64_t myScratch = taskArg->scratchAddr_ + offsetSliceSize;

    uint64_t sliceSize = taskArg->sliceSize_;
    uint64_t inputSliceStride = taskArg->inputSliceStride_;
    uint64_t rmtReduceSliceOffset = inputSliceStride * myRankId_;

    u32 dataTypeSize = DataTypeSizeGet(dataType_);

    uint64_t localRedcueSize0 = (sliceSize / dataTypeSize) / MISSION_NUM * dataTypeSize;
    uint64_t localRedcueSize1 = sliceSize - localRedcueSize0;

    auto rmtReduceGoSize    = CalGoSize(sliceSize);
    auto localReduceGoSize0 = CalGoSize(localRedcueSize0);
    auto localReduceGoSize1 = CalGoSize(localRedcueSize1);

    HCCL_INFO("[CcuKernelReduceScatterMesh2Die][GeneArgs] myInput[%llu], myOutput[%llu], myScratch[%llu]"
              "rmtReduceSliceOffset[%llu], sliceSize[%llu], localRedcueSize0[%llu], localRedcueSize1[%llu], offsetSliceSize[%llu]",
              myInput, myOutput, myScratch, rmtReduceSliceOffset, sliceSize, localRedcueSize0, localRedcueSize1, offsetSliceSize);

    std::vector<uint64_t> taskArgs = {myInput,
                                      myOutput,
                                      myToken,
                                      myScratch,
                                      sliceSize,
                                      rmtReduceSliceOffset};

    for (auto &goSize : {rmtReduceGoSize}) {
        for (auto &element : goSize) {
            taskArgs.push_back(element);
        }
    }
    return taskArgs;
}
}// namespace ops_hccl