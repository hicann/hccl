/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_all_reduce_mesh1d_mem2mem_2die_oneshot.h"

namespace ops_hccl {
using namespace hcomm;
 
constexpr int INPUT_XN_ID        = 0;
constexpr int TOKEN_XN_ID        = 1;
constexpr int POST_SYNC_ID       = 2;

constexpr int CKE_IDX_0          = 0;
constexpr int CKE_IDX_1          = 1;

constexpr int MISSION_NUM        = 2;

constexpr uint32_t LOCAL_REDUCE_LOOP_COUNT  = 8;
constexpr uint32_t LOCAL_REDUCE_MS_PER_LOOP = 8;
const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"_local_reduce_loop_"};
 
CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::CcuKernelAllReduceMesh1DMem2Mem2DieOneShot(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot *kernelArg
        = dynamic_cast<const CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot *>(&arg);
    rankId_         = kernelArg->rankId_;  // 通信域rankId
    rankSize_       = kernelArg->dimSize_; // 两个die包含的所有ranksize
    channels_       = kernelArg->channels;
    
    rmtReduceWithMyRank_ = kernelArg->rmtReduceWithMyRank_;
    rmtReduceRankNum_ = channels_.size() + (rmtReduceWithMyRank_ == true ? 1 : 0); // 这里的rankNum才是单个die的通信范围
 
    // 数据类型处理
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;

    missionSyncMybit_   = 1 << (rmtReduceWithMyRank_ ? 1 : 0);
    missionSyncWaitBit_ = 1 << (!rmtReduceWithMyRank_ ? 1 : 0);
 
    HCCL_INFO(
        "[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d], rmtReduceWithMyRank_[%d]",
        rankId_, rankSize_, dataType_, outputDataType_, reduceOp_, rmtReduceWithMyRank_);
}
 
HcclResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::InitResource()
{
    myInput_   = CreateVariable();
    myOutput_  = CreateVariable();
    myScratch_ = CreateVariable();
    myToken_   = CreateVariable();
    
    for (auto channel : channels_) {
        CcuRep::Variable inputVar, tokenVar;
        CHK_RET(CreateVariable(channel, INPUT_XN_ID, &inputVar));
        peerInput_.push_back(inputVar);
        CHK_RET(CreateVariable(channel, TOKEN_XN_ID, &tokenVar));
        peerToken_.push_back(tokenVar);
    }
 
    scratchBaseOffset0_ = CreateVariable();
    scratchBaseOffset1_ = CreateVariable();
 
    normalSliceSize_ = CreateVariable();
 
    localReduceSliceOffset0_ = CreateVariable();
    localReduceSliceOffset1_ = CreateVariable();
 
    localReduceGoSize_ = CreateGroupOpSize();
    localReduceGoSize0_ = CreateGroupOpSize();
    localReduceGoSize1_ = CreateGroupOpSize();
 
    event_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}
 
void CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::LoadArgs()
{
    Load(myInput_);
    Load(myOutput_);
    Load(myToken_);
    Load(myScratch_);

    Load(normalSliceSize_);
 
    Load(scratchBaseOffset0_);
    Load(scratchBaseOffset1_);
 
    Load(localReduceSliceOffset0_);
    Load(localReduceSliceOffset1_);
 
    Load(localReduceGoSize_);
    Load(localReduceGoSize0_);
    Load(localReduceGoSize1_);
}
 
HcclResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::PreSync()
{
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, myInput_, 1 << INPUT_XN_ID));
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, myToken_, 1 << TOKEN_XN_ID));
    }
    
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyWait(channel, CKE_IDX_0, allBit));
    }
    return HCCL_SUCCESS;
}
 
HcclResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::PostSync()
{
    for (auto &ch : channels_) {
        CHK_RET(NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (auto &ch : channels_) {
        CHK_RET(NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return HCCL_SUCCESS;
}
 
HcclResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::RmtReduce()
{
    // 准备输入地址
    std::vector<CcuRep::RemoteAddr> remoteInput;
    uint32_t channelIdx = 0;
    remoteInput.reserve(rmtReduceRankNum_);
    for (uint32_t rankIdx = 0; rankIdx < rmtReduceRankNum_; rankIdx++) {
        if (rmtReduceWithMyRank_ && rankIdx == rankId_ % rmtReduceRankNum_) {
            remoteInput.push_back({});
        } else {
            remoteInput.push_back(CreateRemoteAddr());
            remoteInput.back().addr = peerInput_[channelIdx];
            remoteInput.back().token = peerToken_[channelIdx];
            channelIdx++;
        }
    }
 
    // 准备输出地址
    std::vector<CcuRep::LocalAddr> scratchDst;
    scratchDst.reserve(rmtReduceRankNum_);
    CcuRep::Variable scratchOffset = CreateVariable();
    scratchOffset = 0;
    for (uint32_t rankIdx = 0; rankIdx < rmtReduceRankNum_; rankIdx++) {
        scratchDst.push_back(CreateLocalAddr());
        scratchDst.back().addr  = myScratch_;
        scratchDst.back().addr += rmtReduceWithMyRank_ ? scratchBaseOffset0_ : scratchBaseOffset1_;
        scratchDst.back().addr += scratchOffset;
        scratchOffset += normalSliceSize_;
        scratchDst.back().token = myToken_;
    }
 
    // 拉取所有对端数据到本端scratch上
    uint32_t channelId = 0;
    for (uint32_t rankIdx = 0; rankIdx < rmtReduceRankNum_; rankIdx++) {
        event_.SetMask(1 << rankIdx);
        if (rmtReduceWithMyRank_ && rankIdx == rankId_ % rmtReduceRankNum_) {
            CHK_RET(RecordEvent(event_));
        } else {
            CHK_RET(ReadNb(channels_[channelId], scratchDst[rankIdx], remoteInput[rankIdx], normalSliceSize_, event_));
            channelId++;
        }
    }
 
    // 等读完所有对端
    event_.SetMask((1 << rmtReduceRankNum_) - 1);
    CHK_RET(WaitEvent(event_));
 
    // 用ccu_ms进行reduce
    if (rmtReduceWithMyRank_) {
        CcuRep::LocalAddr output = CreateLocalAddr();
        output.token          = myToken_;
        output.addr           = myOutput_;
        scratchDst[rankId_ % rmtReduceRankNum_].addr = myInput_;
        scratchDst[rankId_ % rmtReduceRankNum_].token = myToken_;
        ReduceLoopGroup(output, scratchDst, localReduceGoSize_, dataType_, outputDataType_, reduceOp_, "all_reduce_2die_localreduce1_");
    } else {
        ReduceLoopGroup(scratchDst[0], scratchDst, localReduceGoSize_, dataType_, outputDataType_, reduceOp_, "all_reduce_2die_localreduce1_");
    }
    
    return HCCL_SUCCESS;
}

void CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::DoLocalReduce()
{
    std::vector<CcuRep::LocalAddr> src;
    src.reserve(MISSION_NUM);
    for (uint32_t i = 0; i < MISSION_NUM; i++) {
        src.push_back(CreateLocalAddr());
        src.back().token = myToken_;
        src.back().addr  = i == 0 ? myOutput_ : myScratch_;
        src.back().addr += i == 0 ? scratchBaseOffset0_ : scratchBaseOffset1_;
        // 添加两个mission各自第二步的需要本地reduce偏移
        src.back().addr += rmtReduceWithMyRank_ ? localReduceSliceOffset0_ : localReduceSliceOffset1_;
    }
 
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst.token          = myToken_;
    dst.addr           = myOutput_;
    dst.addr += rmtReduceWithMyRank_ ? localReduceSliceOffset0_ : localReduceSliceOffset1_;
    if (rmtReduceWithMyRank_) {
        ReduceLoopGroup(dst, src, localReduceGoSize0_, dataType_, outputDataType_, reduceOp_, "all_reduce_2die_localreduce2_");
    } else {
        ReduceLoopGroup(dst, src, localReduceGoSize1_, dataType_, outputDataType_, reduceOp_, "all_reduce_2die_localreduce2_");
    }
}
 
void CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::ReduceLoopGroup(CcuRep::LocalAddr &outDstOrg, std::vector<CcuRep::LocalAddr> &srcOrg,
                                                        GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType,
                                                        HcclReduceOp opType, std::string loopName)
{
    const uint32_t size = srcOrg.size();
 
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst                = outDstOrg;
 
    std::vector<CcuRep::LocalAddr> src;
    for (uint32_t idx = 0; idx < size; idx++) {
        src.push_back(CreateLocalAddr());
        src[idx] = srcOrg[idx];
    }
 
    CreateReduceLoop(size, dataType, outputDataType, opType, loopName);
 
    std::string      loopType           = GetReduceTypeStr(dataType, opType);
    loopType = loopName + loopType;
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
 
void CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
                                                         HcclReduceOp opType, std::string loopName)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(LOOP_NUM);
    
    std::string loopType = GetReduceTypeStr(dataType, opType);
    loopType = loopName + loopType;
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }
 
    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;
 
    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        CcuRep::LocalAddr              dst = CreateLocalAddr();
        std::vector<CcuRep::LocalAddr> src;
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
 
std::string CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::GetLoopBlockTag(std::string loopType, int32_t index) const
{
    return loopType + LOCAL_REDUCE_LOOP_BLOCK_TAG + std::to_string(index);
}
 
HcclResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::MissionSync(uint32_t maskIndex)
{   
    uint32_t coreIdx = rmtReduceWithMyRank_ ? 1 : 0;
    CHK_RET(LocalNotifyRecord(1 - coreIdx, CKE_IDX_1, missionSyncMybit_ << (MISSION_NUM * maskIndex)));
    CHK_RET(LocalNotifyWait(coreIdx, CKE_IDX_1, missionSyncWaitBit_ << (MISSION_NUM * maskIndex)));
    return HCCL_SUCCESS;
}

HcclResult CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::Algorithm()
{
    HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot]AllReduceMesh1DMem2Mem2DieOneShot run");
 
    CHK_RET(InitResource());
 
    LoadArgs();
 
    CHK_RET(PreSync());
 
    CHK_RET(RmtReduce());

    CHK_RET(PostSync());
    
    CHK_RET(MissionSync(0));
 
    DoLocalReduce();
 
    CHK_RET(MissionSync(1));
 
    HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] AllReduceMesh1DMem2Mem2DieOneShot end");

    return HcclResult::HCCL_SUCCESS;
}
 
std::vector<uint64_t> CcuKernelAllReduceMesh1DMem2Mem2DieOneShot::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllReduceMesh1DMem2Mem2DieOneShot *taskArg = dynamic_cast<const CcuTaskArgAllReduceMesh1DMem2Mem2DieOneShot *>(&arg);
 
    uint64_t myInput   = taskArg->inputAddr_;
    uint64_t myOutput  = taskArg->outputAddr_;
    uint64_t myToken   = taskArg->token_;
    uint64_t myScratch = taskArg->scratchAddr_;
 
    uint64_t sliceSize = taskArg->normalSliceSize_;
 
    uint64_t scratchBaseOffset0 = 0;
    uint64_t scratchBaseOffset1 = sliceSize * rmtReduceRankNum_;
 
    u32 dataTypeSize = DataTypeSizeGet(dataType_);
    uint64_t localRedcueSize0 = (sliceSize / dataTypeSize) / MISSION_NUM * dataTypeSize;
    uint64_t localRedcueSize1 = sliceSize - localRedcueSize0;
 
    uint64_t localReduceSliceOffset0 = 0;
    uint64_t localReduceSliceOffset1 = localRedcueSize0;
 
    auto localReduceGoSize = CalGoSize(sliceSize);
    auto localReduceGoSize0 = CalGoSize(localRedcueSize0);
    auto localReduceGoSize1 = CalGoSize(localRedcueSize1);
 
    std::vector<uint64_t> taskArgs = {myInput,
                                      myOutput,
                                      myToken,
                                      myScratch,
                                      sliceSize,
                                      scratchBaseOffset0,
                                      scratchBaseOffset1,
                                      localReduceSliceOffset0,
                                      localReduceSliceOffset1};
 
    for (auto &goSize : {localReduceGoSize, localReduceGoSize0, localReduceGoSize1}) {
        for (auto &element : goSize) {
            taskArgs.push_back(element);
        }
    }
 
    HCCL_INFO("[CcuKernelAllReduceMesh1DMem2Mem2DieOneShot] TaskArgs: myInput[%llu], myOutput[%llu], "
               "myScratch[%llu], sliceSize[%llu], scratchBaseOffset0[%llu], scratchBaseOffset1[%llu], "
               "localReduceSliceOffset0[%llu], localReduceSliceOffset1[%llu]",
               myInput, myOutput, myScratch, sliceSize, scratchBaseOffset0, scratchBaseOffset1,
               localReduceSliceOffset0, localReduceSliceOffset1);
    return taskArgs;
}
} // namespace ops_hccl
