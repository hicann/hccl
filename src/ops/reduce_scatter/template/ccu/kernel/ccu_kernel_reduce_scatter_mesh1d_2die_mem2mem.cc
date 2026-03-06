/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "ccu_kernel_reduce_scatter_mesh1d_2die_mem2mem.h"
#include "ccu_kernel_alg_base.h"
 
namespace ops_hccl {
using namespace hcomm;
 
constexpr int INPUT_XN_ID        = 0;
constexpr int SCRATCH_XN_ID      = 1;
constexpr int TOKEN_XN_ID        = 2;
constexpr int POST_SYNC_ID       = 3;
constexpr int CKE_IDX_0          = 0;
 
const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"_local_reduce_loop_"};
 
CcuKernelReduceScatterMesh1D2DieMem2Mem::CcuKernelReduceScatterMesh1D2DieMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgReduceScatterMeshMem2Mem1D2Die *kernelArg = 
                        dynamic_cast<const CcuKernelArgReduceScatterMeshMem2Mem1D2Die *>(&arg);
    rankId_              = kernelArg->rankId_;
    gRankSize_           = kernelArg->gRankSize_;
    rankSize_            = kernelArg->rankSize_;
    channels_            = kernelArg->channels;
    subRankGroup_        = kernelArg->subRankGroup_;
    isReduceToOutput_     = kernelArg->isReduceToOutput_;
 
    // 数据类型处理
    dataType_            = kernelArg->opParam_.DataDes.dataType;
    outputDataType_      = kernelArg->opParam_.DataDes.outputType;
 
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_INFO("[CcuKernelReduceScatterMesh1D2DieMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_            = kernelArg->opParam_.reduceType;
}
 
HcclResult CcuKernelReduceScatterMesh1D2DieMem2Mem::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1D2DieMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
 
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (subRankGroup_[rankIdx] == rankId_) {
            input_.push_back(CreateVariable());
            scratch_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            CcuRep::Variable inputVar, scratchVar, tokenVar;
            CHK_RET(CreateVariable(channels_[channelIdx], INPUT_XN_ID, &inputVar));
            input_.push_back(inputVar);
            CHK_RET(CreateVariable(channels_[channelIdx], SCRATCH_XN_ID, &scratchVar));
            scratch_.push_back(scratchVar);
            CHK_RET(CreateVariable(channels_[channelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    if (!isReduceToOutput_) {
        input_.push_back(CreateVariable());
        scratch_.push_back(CreateVariable());
        token_.push_back(CreateVariable());
    }
    myRankIdx_ = input_.size() - 1;
 
    output_                      = CreateVariable();
    currentRankSliceInputOffset_ = CreateVariable();
    sliceSize_                   = CreateVariable();
    inputRepeatStride_           = CreateVariable();
    outputRepeatStride_          = CreateVariable();
    repeatNum_                   = CreateVariable();
    flag_                        = CreateVariable();
 
    sliceGoSize_                 = CreateGroupOpSize();
 
    remoteInput_.reserve(rankSize_);
    scratchMem_.reserve(rankSize_);
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        scratchMem_.push_back(CreateLocalAddr());
        if (subRankGroup_[rankIdx] == rankId_) {
            myInput_ = CreateLocalAddr();
            remoteInput_.push_back({});
        } else {
            remoteInput_.push_back(CreateRemoteAddr());
        }
    }
 
    event_ = CreateCompletedEvent();

    return HcclResult::HCCL_SUCCESS;
}
 
void CcuKernelReduceScatterMesh1D2DieMem2Mem::LoadArgs()
{
    Load(input_[myRankIdx_]); 
    Load(output_);
    Load(token_[myRankIdx_]);
    Load(scratch_[myRankIdx_]);
    Load(currentRankSliceInputOffset_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(sliceSize_);
    Load(repeatNum_);
    
    Load(sliceGoSize_);

    return ;
}
 
HcclResult CcuKernelReduceScatterMesh1D2DieMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, INPUT_XN_ID, input_[myRankIdx_], 1 << INPUT_XN_ID));
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, SCRATCH_XN_ID, scratch_[myRankIdx_], 1 << SCRATCH_XN_ID));
        CHK_RET(NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[myRankIdx_], 1 << TOKEN_XN_ID));
    }
 
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << SCRATCH_XN_ID | 1 << TOKEN_XN_ID;
    
    for (ChannelHandle channel : channels_) {
        CHK_RET(NotifyWait(channel, CKE_IDX_0, allBit));
    }
    return HCCL_SUCCESS;
}
 
HcclResult CcuKernelReduceScatterMesh1D2DieMem2Mem::PostSync()
{
    for (auto &ch : channels_) {
        CHK_RET(NotifyRecord(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    
    for (auto &ch : channels_) {
        CHK_RET(NotifyWait(ch, CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return HCCL_SUCCESS;
}
 
HcclResult CcuKernelReduceScatterMesh1D2DieMem2Mem::RmtReduce()
{
    CcuRep::Variable scratchOffset = CreateVariable();
    scratchOffset                  = 0;
    std::vector<CcuRep::Variable> scratchOffsetVec;
 
    for (uint32_t gRankIdx = 0; gRankIdx < gRankSize_; gRankIdx++) {
        CcuRep::Variable scratchOffTmp = CreateVariable();
        scratchOffTmp = scratchOffset;
        scratchOffsetVec.push_back(scratchOffTmp);
        scratchOffset += sliceSize_;
    }

    outputTmp_ = CreateLocalAddr();
    outputTmp_.addr = scratch_[myRankIdx_];
    outputTmp_.addr += scratchOffsetVec[gRankSize_ / 2];
    outputTmp_.token = token_[myRankIdx_];
 
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (subRankGroup_[rankIdx] != rankId_) {
            remoteInput_[rankIdx].addr = input_[rankIdx];
            remoteInput_[rankIdx].addr += currentRankSliceInputOffset_;
            remoteInput_[rankIdx].token = token_[rankIdx];
        }
        scratchMem_[rankIdx].addr = scratch_[myRankIdx_];
        scratchMem_[rankIdx].addr += scratchOffsetVec[subRankGroup_[rankIdx]];
        scratchMem_[rankIdx].token = token_[myRankIdx_];
    }
 
    CcuRep::Variable repeatNumAdd = CreateVariable();
    repeatNumAdd  = 1;
    flag_ = 0;
    CCU_WHILE(repeatNum_ != UINT64_MAX) {
        repeatNum_ += repeatNumAdd;
        CCU_IF(flag_ == 1) {
            //  非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
            for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                if (subRankGroup_[rankIdx] == rankId_) {
                    myInput_.addr += inputRepeatStride_;
                } else {
                    remoteInput_[rankIdx].addr += inputRepeatStride_;
                }
            }
            output_ += outputRepeatStride_;
        }
        CCU_IF(sliceSize_ != 0)
        {
            CHK_RET(DoReduceScatter());
        }
        flag_ = 1;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernelReduceScatterMesh1D2DieMem2Mem::DoReduceScatter()
{
    CcuRep::LocalAddr myOutput = CreateLocalAddr();
    myOutput.addr                     = output_;
    myOutput.token                    = token_[myRankIdx_];
    
    if (rankId_ < rankSize_) {
        if (!isReduceToOutput_) {
            myOutput.addr = outputTmp_.addr;
        }
    } else {
        if (isReduceToOutput_) {
            myOutput.addr = outputTmp_.addr;
        }
    }
    
    uint32_t channelId = 0;
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        event_.SetMask(1 << rankIdx);
        if (subRankGroup_[rankIdx] == rankId_) {
            CHK_RET(RecordEvent(event_));
            scratchMem_[rankIdx].addr = input_[myRankIdx_];
            scratchMem_[rankIdx].addr += currentRankSliceInputOffset_;
            scratchMem_[rankIdx].token = token_[myRankIdx_];
        } else {
            CHK_RET(ReadNb(channels_[channelId], scratchMem_[rankIdx], remoteInput_[rankIdx], sliceSize_, event_));
            channelId++;
        }
    }
 
    // 等读完所有对端
    event_.SetMask((1 << rankSize_) - 1);
    CHK_RET(WaitEvent(event_));
 
    // 做reduce
    ReduceLoopGroup(myOutput, scratchMem_);
    return HCCL_SUCCESS;
}
 
std::string CcuKernelReduceScatterMesh1D2DieMem2Mem::GetLoopBlockTag(std::string loopType, int32_t index)
{
    return loopType + LOCAL_REDUCE_LOOP_BLOCK_TAG + std::to_string(index);
}
 
void CcuKernelReduceScatterMesh1D2DieMem2Mem::CreateReduceLoop(uint32_t size)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(LOOP_NUM);
 
    std::string loopType = ops_hccl::GetReduceTypeStr(dataType_, reduceOp_);
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }
 
    uint32_t expansionNum = ops_hccl::GetReduceExpansionNum(reduceOp_, dataType_, outputDataType_);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;
 
    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        CcuRep::LocalAddr dst = CreateLocalAddr();
        std::vector<CcuRep::LocalAddr> scratch;
        for (uint32_t i = 0; i < size; i++) {
            scratch.emplace_back(CreateLocalAddr());
        }
        CcuRep::Variable            len = CreateVariable();
        CcuRep::Variable            lenForExpansion = CreateVariable();
        CcuRep::LoopBlock           lb(this, GetLoopBlockTag(loopType, index));
        lb(dst, scratch, len, lenForExpansion);
 
        std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
                                               moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::CompletedEvent sem = moRes.completedEvent[index];
 
        for (uint32_t i = 0; i < size; i++) {
            LocalCopyNb(bufs[i], scratch[i], len, sem);
        }
        WaitEvent(sem);
 
        if (size > 1) {
            LocalReduceNb(bufs, size, dataType_, outputDataType_, reduceOp_, len, sem);
            WaitEvent(sem);
        }
 
        LocalCopyNb(dst, bufs[0], lenForExpansion, sem);
        WaitEvent(sem);
    }
 
    registeredLoop.insert(loopType);
}
 
void CcuKernelReduceScatterMesh1D2DieMem2Mem::ReduceLoopGroup(CcuRep::LocalAddr outDstOrg,
        std::vector<CcuRep::LocalAddr> &scratchOrg)
{
    const uint32_t size = scratchOrg.size();
 
    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst = outDstOrg;
 
    std::vector<CcuRep::LocalAddr> scratch;
    for (uint32_t idx = 0; idx < size; idx++) {
        scratch.push_back(CreateLocalAddr());
        scratch[idx] = scratchOrg[idx];
    }
 
    CreateReduceLoop(size);
 
    std::string loopType = ops_hccl::GetReduceTypeStr(dataType_, reduceOp_);
    uint32_t         expansionNum = ops_hccl::GetReduceExpansionNum(reduceOp_, dataType_, outputDataType_);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();
 
    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp = ops_hccl::GetExpansionParam(expansionNum);
        dst.token += tmp;
    }
 
    // m部分
    CCU_IF(sliceGoSize_.loopParam != 0)                   // goSize1
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = ops_hccl::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += sliceGoSize_.loopParam;
 
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize          = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;
 
        auto lc = Loop(GetLoopBlockTag(loopType, 0))(dst, scratch, sliceSize, sliceSizeExpansion);
 
        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = ops_hccl::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = ops_hccl::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
 
        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }
 
    CCU_IF(sliceGoSize_.parallelParam != 0)               // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += sliceGoSize_.addrOffset;
        }
 
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += sliceGoSize_.addrOffset;
        }
 
        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += sliceGoSize_.residual;  // goSize3
        }
 
        auto lc0 = Loop(GetLoopBlockTag(loopType, 0))(dst, scratch, sliceGoSize_.residual, sliceSizeExpansion);
 
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += sliceGoSize_.residual;
        }
 
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += sliceGoSize_.residual;
        }
 
        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize          = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;
 
        auto lc1 = Loop(GetLoopBlockTag(loopType, 1))(dst, scratch, sliceSize, sliceSizeExpansion);
 
        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = ops_hccl::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = ops_hccl::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = ops_hccl::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);
 
        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, sliceGoSize_.parallelParam, offsetCfg);
    }
}
 
HcclResult CcuKernelReduceScatterMesh1D2DieMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelReduceScatterMesh1D2DieMem2Mem] Algorithm start");
    CHK_RET(InitResource());
    LoadArgs();
    CHK_RET(PreSync());
    CHK_RET(RmtReduce());
    CHK_RET(PostSync());
    HCCL_INFO("[CcuKernelReduceScatterMesh1D2DieMem2Mem] Algorithm end");
    return HcclResult::HCCL_SUCCESS;
}
 
std::vector<uint64_t> CcuKernelReduceScatterMesh1D2DieMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgReduceScatterMeshMem2Mem1D2Die *taskArg =
                        dynamic_cast<const CcuTaskArgReduceScatterMeshMem2Mem1D2Die *>(&arg);
    uint64_t myInput   = taskArg->inputAddr_;
    uint64_t myOutput  = taskArg->outputAddr_;
    uint64_t myToken   = taskArg->token_;
    uint64_t myScratch = taskArg->scratchAddr_;
    uint64_t sliceSize = taskArg->sliceSize_;
 
    uint64_t inputRepeatStride           = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride          = taskArg->outputRepeatStride_;
    uint64_t repeatNum                   = taskArg->repeatNum_;
 
    uint64_t currentRankSliceInputOffset = taskArg->inputSliceStride_ * rankId_;
 
    u32 dataTypeSize = DataTypeSizeGet(dataType_);
 
    auto rmtReduceGoSize    = CalGoSize(sliceSize);

    uint64_t scratchOffset0 = 0;
    uint64_t scratchOffset1 = sliceSize * rankSize_;
 
    std::vector<uint64_t> taskArgs = {myInput,
                                      myOutput,
                                      myToken,
                                      myScratch,
                                      currentRankSliceInputOffset,
                                      inputRepeatStride,
                                      outputRepeatStride,
                                      sliceSize,
                                      repeatNum};
 
    for (auto &goSize : {rmtReduceGoSize}) {
        for (auto &element : goSize) {
            taskArgs.push_back(element);
        }
    }
    
    HCCL_INFO("[CcuKernelReduceScatterMesh1D2DieMem2Mem::GeneArgs]currentRankSliceInputOffset[%d], sliceSize[%d], rankSize_[%d], scratchBaseOffset0[%d], scratchBaseOffset1[%d]",
        currentRankSliceInputOffset, sliceSize, rankSize_, scratchOffset0, scratchOffset1);
 
    return taskArgs;
}
} // namespace Hccl