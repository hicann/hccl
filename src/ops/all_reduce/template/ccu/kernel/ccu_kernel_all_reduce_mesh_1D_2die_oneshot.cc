/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh_1D_2die_oneshot.h"
#include "ccu_kernel_alg_base.h"
#include "ccu_kernel.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int INPUT_XN_ID    = 0;
constexpr int TOKEN_XN_ID    = 1;
constexpr int POST_SYNC_ID   = 2;
constexpr int MISSION_SYNC_ID_0 = 3;
constexpr int MISSION_SYNC_ID_1 = 4;

constexpr int CKE_IDX_0     = 0;
constexpr int CKE_IDX_1     = 1;
constexpr int CKE_IDX_2     = 2;

constexpr int LOOP_NUM = 64;
constexpr int DIE_WORK = 2;
const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"_local_reduce_loop_"};

CcuKernelAllreduceMesh1D2DieOneShot::CcuKernelAllreduceMesh1D2DieOneShot(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllreduceMesh1D2DieOneShot *kernelArg
        = dynamic_cast<const CcuKernelArgAllreduceMesh1D2DieOneShot *>(&arg);
        
    myRankId_         = kernelArg->rankId_;
    rankSize_         = kernelArg->dimSize_; // 两个kernel传不一样的
    channels_         = kernelArg->channels;
    rmtReduceWithMyRank_ = kernelArg->rmtReduceWithMyRank_;
    rmtReduceRankNum_ = channels_.size() + (rmtReduceWithMyRank_ == true ? 1 : 0);
 
    rmtSyncMyBit_ = 1 << (myRankId_ % rmtReduceRankNum_);
    rmtSyncWaitBit_ = 
        rmtReduceWithMyRank_ ? ((1 << rmtReduceRankNum_) - 1) & (~rmtSyncMyBit_) : (1 << rmtReduceRankNum_) - 1;
  
    missionSyncMybit_   = 1 << (rmtReduceWithMyRank_ ? 1 : 0);
    missionSyncWaitBit_ = 1 << (!rmtReduceWithMyRank_ ? 1 : 0);

    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;

    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelAllreduceMesh1D2DieOneShot] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_          = kernelArg->opParam_.reduceType;
    moConfig.loopCount = LOOP_NUM;
    HCCL_INFO(
        "[CcuKernelAllreduceMesh1D2DieOneShot] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d], rmtReduceWithMyRank[%d], rmtReduceRankNum[%d]",
        myRankId_, rankSize_, dataType_, outputDataType_, reduceOp_, rmtReduceWithMyRank_, rmtReduceRankNum_);
}

HcclResult CcuKernelAllreduceMesh1D2DieOneShot::InitResource()
{
    uint64_t channelIdx = 0;
    myInput_ = CreateVariable();
    myOutput_ = CreateVariable();
    myScratch_ = CreateVariable();
    myToken_ = CreateVariable();

    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllreduceMesh1D2DieOneShot] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] myRankId_[%llu], channel[%llu]", myRankId_, channels_.size());
    for (auto channel : channels_) {
        HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] myRankId_[%llu] channelId[%llu]", myRankId_, channelIdx);
        CcuRep::Variable inputVar, tokenVar;
        CHK_RET(CreateVariable(channel, INPUT_XN_ID, &inputVar));
        input_.push_back(inputVar); // 获取channel中id=0的Var来传递output
        CHK_RET(CreateVariable(channel, TOKEN_XN_ID, &tokenVar));
        remoteToken_.push_back(tokenVar);
        channelIdx = channelIdx + 1;
    }

    scratchBaseOffset0_ = CreateVariable();
    scratchBaseOffset1_ = CreateVariable();
  
    localReduceSliceOffset0_ = CreateVariable();
    localReduceSliceOffset1_ = CreateVariable();
 
    rmtReduceGoSize_    = CreateGroupOpSize();
    localReduceGoSize0_ = CreateGroupOpSize();
    localReduceGoSize1_ = CreateGroupOpSize();

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] InitResources finished");
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllreduceMesh1D2DieOneShot::LoadArgs()
{
    Load(myInput_);
    Load(myOutput_);
    Load(myToken_);
    Load(myScratch_);
    Load(scratchBaseOffset0_);
    Load(scratchBaseOffset1_);
    
    Load(localReduceSliceOffset0_);
    Load(localReduceSliceOffset1_);

    Load(rmtReduceGoSize_);
    Load(localReduceGoSize0_);
    Load(localReduceGoSize1_);

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] LoadArgs run finished");
}

void CcuKernelAllreduceMesh1D2DieOneShot::PreSync()
{
    uint32_t channelIdx = 0;
    for (auto channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, myInput_, 1 << INPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, myToken_, 1 << TOKEN_XN_ID);
        channelIdx = channelIdx +1 ;
    }
    
    channelIdx = 0;
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
        channelIdx = channelIdx +1 ;
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] PreSync run finished");
}

void CcuKernelAllreduceMesh1D2DieOneShot::PostSync(uint32_t signalIndex)
{
    for (auto channel : channels_) {
        NotifyRecord(channel, CKE_IDX_1, 1 << signalIndex);
    }

    for (auto channel : channels_) {
        NotifyWait(channel, CKE_IDX_1, 1 << signalIndex);
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] PostSync run finished");
}

void CcuKernelAllreduceMesh1D2DieOneShot::DoRmtReduce()
{
    std::vector<CcuRep::RemoteAddr> src;
    src.reserve(rmtReduceRankNum_);
    for (uint32_t peerIdx = 0; peerIdx < channels_.size(); peerIdx++) {
        HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] DoRmtReduce myRankId_[%llu] peerIdx[%llu]", myRankId_, peerIdx);
        src.push_back(CreateRemoteAddr());
        src.back().token = remoteToken_[peerIdx];
        src.back().addr  = input_[peerIdx];
    }
    if (rmtReduceWithMyRank_) {
        src.push_back(CreateRemoteAddr());
        src.back().token = myToken_;
        src.back().addr  = myInput_;
    }
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst.token             = myToken_;
    dst.addr              = myScratch_;
    dst.addr += rmtReduceWithMyRank_ ? scratchBaseOffset0_ : scratchBaseOffset1_;
    if (rmtReduceWithMyRank_) {
        GroupReduce(channels_, dst, src, rmtReduceGoSize_, dataType_, outputDataType_, reduceOp_);
    } else {
        GroupReduceWithoutMyRank(channels_, dst, src, rmtReduceGoSize_, dataType_, outputDataType_, reduceOp_);
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Step1 RmtReduce run finished");
}

std::string CcuKernelAllreduceMesh1D2DieOneShot::GetLoopBlockTag(std::string loopType, int32_t index) const
{
    return loopType + LOCAL_REDUCE_LOOP_BLOCK_TAG + std::to_string(index);
}

void CcuKernelAllreduceMesh1D2DieOneShot::CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
                                                         HcclReduceOp opType)
{
    AllocGoResource();
    
    std::string loopType = GetReduceTypeStr(dataType, opType);
    loopType = "local_reduce_" + loopType;
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }
 
    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;
 
    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        CcuRep::LocalAddr              dst = CreateLocalAddr();
        std::vector<CcuRep::LocalAddr> src;
        src.reserve(size);
        for (uint32_t i = 0; i < size; i++) {
            src.emplace_back(CreateLocalAddr());
        }
        CcuRep::Variable  len             = CreateVariable();
        CcuRep::Variable  lenForExpansion = CreateVariable();
        CcuRep::LoopBlock lb(this, GetLoopBlockTag(loopType, index));
        lb(dst, src, len, lenForExpansion);
 
        std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
                                               moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::CompletedEvent &event = moRes.completedEvent[index];
 
        for (uint32_t i = 0; i < size; i++) {
            event.SetMask(1 << i);
            LocalCopyNb(bufs[i], src[i], len, event);
        }
        event.SetMask((1 << size) - 1);
        WaitEvent(event);
 
        if (size > 1) {
            event.SetMask(1);
            LocalReduceNb(bufs, size, dataType, outputDataType, opType, len, event);
            WaitEvent(event);
        }

        event.SetMask(1);
        LocalCopyNb(dst, bufs[0], lenForExpansion, event);
        WaitEvent(event);
    }
 
    registeredLoop.insert(loopType);
}

void CcuKernelAllreduceMesh1D2DieOneShot::ReduceLoopGroup(CcuRep::LocalAddr &outDstOrg, std::vector<CcuRep::LocalAddr> &srcOrg,
                                                        GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType,
                                                        HcclReduceOp opType)
{
    const uint32_t size = srcOrg.size();
 
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst                = outDstOrg;
 
    std::vector<CcuRep::LocalAddr> src;
    src.reserve(size);
    for (uint32_t idx = 0; idx < size; idx++) {
        src.push_back(CreateLocalAddr());
        src[idx] = srcOrg[idx];
    }
 
    CreateReduceLoop(size, dataType, outputDataType, opType);
 
    std::string      loopType           = GetReduceTypeStr(dataType, opType);
    loopType = "local_reduce_" + loopType;
    uint32_t         expansionNum       = GetReduceExpansionNum(opType, dataType, outputDataType);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();
 
    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp                  = GetExpansionParam(expansionNum);
        dst.token += tmp;
    }
 
    // m部分
    CCU_IF(goSize.loopParam != 0) // goSize1
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam                  = GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;
 
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize                  = moConfig.memSlice;
        sliceSizeExpansion         = moConfig.memSlice * expansionNum;
        auto lc = Loop(GetLoopBlockTag(loopType, 0))(dst, src, sliceSize, sliceSizeExpansion);
 
        CcuRep::Variable paraCfg   = CreateVariable();
        paraCfg                    = GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
 
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
 
    CCU_IF(goSize.parallelParam != 0) // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }
 
        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual; // goSize3
        }
 
        auto lc0 = Loop(GetLoopBlockTag(loopType, 0))(dst, src, goSize.residual, sliceSizeExpansion);
 
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr += goSize.residual;
        }
 
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }
 
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize                  = moConfig.memSlice;
        sliceSizeExpansion         = moConfig.memSlice * expansionNum;
 
        auto lc1 = Loop(GetLoopBlockTag(loopType, 1))(dst, src, sliceSize, sliceSizeExpansion);
 
        CcuRep::Variable loopCfg0  = CreateVariable();
        loopCfg0                   = GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1  = CreateVariable();
        loopCfg1                   = GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg                  = GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
 
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
}

void CcuKernelAllreduceMesh1D2DieOneShot::DoLocalReduce()
{
    std::vector<CcuRep::LocalAddr> src;
    src.reserve(DIE_WORK);
    for (uint32_t i = 0; i < DIE_WORK; i++) {
        src.push_back(CreateLocalAddr());
        src.back().token = myToken_;
        src.back().addr  = myScratch_;
        // 添加两个mission第一步跨卡reduce结果的偏移
        src.back().addr += i == 0 ? scratchBaseOffset0_ : scratchBaseOffset1_;
        // 添加两个mission各自第二步的需要本地reduce偏移
        src.back().addr += rmtReduceWithMyRank_ ? localReduceSliceOffset0_ : localReduceSliceOffset1_;
    }
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst.token          = myToken_;
    dst.addr           = myOutput_;
    dst.addr          += rmtReduceWithMyRank_ ? localReduceSliceOffset0_ : localReduceSliceOffset1_;

    if (rmtReduceWithMyRank_) {
        CCU_IF(localReduceSliceOffset1_ != 0)
        {
            ReduceLoopGroup(dst, src, localReduceGoSize0_, dataType_, outputDataType_, reduceOp_);
        }
    } else {
        ReduceLoopGroup(dst, src, localReduceGoSize1_, dataType_, outputDataType_, reduceOp_);
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] DoLocalReduce run finished");
}

void CcuKernelAllreduceMesh1D2DieOneShot::MissionSync(uint32_t maskIndex)
{
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] MissionSync, missionSyncMybit_[%u], missionSyncWaitBit_[%u]",
             missionSyncMybit_, missionSyncWaitBit_);
    uint32_t coreIdx = rmtReduceWithMyRank_ ? 1 : 0;
    LocalNotifyRecord(1 - coreIdx, CKE_IDX_2, missionSyncMybit_ << (DIE_WORK * maskIndex));
    LocalNotifyWait(coreIdx, CKE_IDX_2, missionSyncWaitBit_ << (DIE_WORK * maskIndex));
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] MissionSync run finished");
}

HcclResult CcuKernelAllreduceMesh1D2DieOneShot::Algorithm()
{
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] AllreduceMesh1D2DieOneShot run");
    CHK_RET(InitResource());
    
    LoadArgs();

    PreSync();

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Algorithm second step rmtReduce begins.");

    DoRmtReduce();

    PostSync(POST_SYNC_ID);

    MissionSync(0);

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Algorithm second step localreduce begins.");
    
    DoLocalReduce();

    MissionSync(1);

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] AllreduceMesh1D2Die end");
    
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllreduceMesh1D2DieOneShot::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllreduceMesh1D2DieOneShot *taskArg = dynamic_cast<const CcuTaskArgAllreduceMesh1D2DieOneShot *>(&arg);

    moConfig.loopCount = LOOP_NUM;
    uint64_t myInput   = taskArg->inputAddr_;
    uint64_t myOutput  = taskArg->outputAddr_;
    uint64_t myToken   = taskArg->token_;
    uint64_t myScratch = taskArg->scratchAddr_;
 
    uint64_t sliceSize = taskArg->sliceSize_;
 
    uint64_t scratchBaseOffset0 = 0;
    uint64_t scratchBaseOffset1 = sliceSize;
  
    uint32_t dataTypeSize = DataTypeSizeGet(dataType_);
 
    uint64_t localRedcueSize0 = ((sliceSize / dataTypeSize) / DIE_WORK) * dataTypeSize;
    uint64_t localRedcueSize1 = sliceSize - localRedcueSize0;
 
    uint64_t localReduceSliceOffset0 = 0;
    uint64_t localReduceSliceOffset1 = localRedcueSize0;
    auto rmtReduceGoSize    = CalGoSize(sliceSize);
    auto localReduceGoSize0 = CalGoSize(localRedcueSize0);
    auto localReduceGoSize1 = CalGoSize(localRedcueSize1);

    std::vector<uint64_t> taskArgs = {myInput,
                                      myOutput,
                                      myToken,
                                      myScratch,
                                      scratchBaseOffset0,
                                      scratchBaseOffset1,
                                      localReduceSliceOffset0,
                                      localReduceSliceOffset1};
 
    for (auto &goSize : {rmtReduceGoSize, localReduceGoSize0, localReduceGoSize1}) {
        for (auto &element : goSize) {
            taskArgs.push_back(element);
        }
    }

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot][GeneArgs]: inputAddr[%llu], outputAddr[%llu],"
        "sliceSize[%llu], scratchBaseOffset0[%llu], scratchBaseOffset1[%llu],localReduceSliceOffset0[%llu],localReduceSliceOffset1[%llu]",
        myInput, myOutput, sliceSize, scratchBaseOffset0, scratchBaseOffset1,localReduceSliceOffset0,localReduceSliceOffset1);

    return taskArgs;
}
} // namespace ops_hccl
