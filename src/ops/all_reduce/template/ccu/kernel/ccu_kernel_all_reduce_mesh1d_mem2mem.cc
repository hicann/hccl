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
#include "ccu_kernel_all_reduce_mesh1d_mem2mem.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel.h"

namespace ops_hccl {
using namespace hcomm;
constexpr int INPUT_XN_ID   = 0;
constexpr int OUTPUT_XN_ID  = 1;
constexpr int SCRATCH_XN_ID = 2;
constexpr int TOKEN_XN_ID   = 3;
constexpr int POST_SYNC_ID   = 4;
constexpr int CKE_IDX_0     = 0;

CcuKernelAllReduceMeshMem2Mem1D::CcuKernelAllReduceMeshMem2Mem1D(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllReduceMeshMem2Mem1D *kernelArg
        = dynamic_cast<const CcuKernelArgAllReduceMeshMem2Mem1D *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    dataType_       = kernelArg->opParam_.DataDes.dataType;
    outputDataType_ = kernelArg->opParam_.DataDes.outputType;
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
        HCCL_DEBUG(
            "[CcuKernelAllReduceMeshMem2Mem1D] outputDataType is [HCCL_DATA_TYPE_RESERVED], set outputDataType to[%d]",
            outputDataType_);
    }
    reduceOp_       = kernelArg->opParam_.reduceType;
    HCCL_INFO(
        "[CcuKernelAllReduceMeshMem2Mem1D] Init, KernelArgs are rankId[%u], rankSize_[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d]",
        rankId_, rankSize_, dataType_, outputDataType_, reduceOp_);
}

HcclResult CcuKernelAllReduceMeshMem2Mem1D::InitResource()
{
    uint16_t channelIdx = 0;
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllReduceMeshMem2Mem1D] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    // 按照rank号从小到大遍历channels_，遇到本rank就填充本地资源，否则依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            scratch_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        } else {
            HCCL_DEBUG("[CcuKernelAllReduceMeshMem2Mem1D] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                       rankId_, peerId, channelIdx);
            CcuRep::Variable inputVar, scratchVar, tokenVar, outputVar;
            CreateVariable((channels_[channelIdx]), INPUT_XN_ID, &inputVar);
            CreateVariable((channels_[channelIdx]), OUTPUT_XN_ID, &outputVar);
            CreateVariable((channels_[channelIdx]), SCRATCH_XN_ID, &scratchVar);
            CreateVariable((channels_[channelIdx]), TOKEN_XN_ID, &tokenVar);
            input_.push_back(inputVar);
            output_.push_back(outputVar);
            scratch_.push_back(scratchVar);
            token_.push_back(tokenVar);
            channelIdx++;
        }
    }
    currentRankSliceInputOffset_  = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    normalSliceSize_              = CreateVariable();
    lastSliceSize_                = CreateVariable();
    mySliceSize_                  = CreateVariable();
    sliceOffset_                  = CreateVariable();
    isInputOutputEqual_           = CreateVariable();
    locEvent_                     = CreateCompletedEvent();
    srcMem_                       = CreateLocalAddr();
    localDstMem_                  = CreateLocalAddr();
    remoteDstMem_                 = CreateRemoteAddr();
    reduceScatterSrc_.reserve(rankSize_);
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterSrc_.push_back(CreateRemoteAddr());
    }
    reduceScatterDst_.reserve(rankSize_);
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterDst_.push_back(CreateLocalAddr());
    }
    sliceSize_ = CreateVariable();
    localGoSize_ = CreateGroupOpSize();
    return HCCL_SUCCESS;
}

std::string CcuKernelAllReduceMeshMem2Mem1D::GetLoopBlockTag(std::string loopType, int32_t index)
{
    return loopType + LOOP_BLOCK_TAG + std::to_string(index);
}
 
void CcuKernelAllReduceMeshMem2Mem1D::CreateReduceLoop(uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(LOOP_NUM);

    std::string loopType = GetReduceTypeStr(dataType, opType);
    if (registeredLoop.find(loopType) != registeredLoop.end()) {
        return;
    }

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        CcuRep::LocalAddr dst = CreateLocalAddr();
        CcuRep::LocalAddr src = CreateLocalAddr();
        std::vector<CcuRep::LocalAddr> scratch;
        for (uint32_t i = 0; i < size; i++) {
            scratch.emplace_back(CreateLocalAddr());
        }
        CcuRep::Variable            len = CreateVariable();
        CcuRep::Variable            lenForExpansion = CreateVariable();
        CcuRep::LoopBlock           lb(this, GetLoopBlockTag(loopType, index));
        lb(dst, src, scratch, len, lenForExpansion);

        std::vector<CcuRep::CcuBuf> bufs = {moRes.ccuBuf.begin() + index * moConfig.msInterleave,
                                               moRes.ccuBuf.begin() + index * moConfig.msInterleave + usedBufNum};
        CcuRep::CompletedEvent             e  = moRes.completedEvent[index];

        for (uint32_t i = 0; i < size; i++) {
            e.SetMask(1 << i);
            if (i == rankId_) {
                CcuKernel::LocalCopyNb(bufs[i], src, len, e);
            } else {
                CcuKernel::LocalCopyNb(bufs[i], scratch[i], len, e);
            }
        }
        e.SetMask((1 << size) - 1);
        WaitEvent(e);
        e.SetMask(1);
        if (size > 1) {
            LocalReduceNb(bufs, size, dataType, outputDataType, opType, len, e); 
            WaitEvent(e);
        }

        CcuKernel::LocalCopyNb(dst, bufs[0], lenForExpansion, e);
        WaitEvent(e);
    }

    registeredLoop.insert(loopType);
}

void CcuKernelAllReduceMeshMem2Mem1D::ReduceLoopGroup(CcuRep::LocalAddr outDstOrg, CcuRep::LocalAddr srcOrg,
    std::vector<CcuRep::LocalAddr> &scratchOrg, GroupOpSize goSize, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType)
{
    const uint32_t size = scratchOrg.size();

    CcuRep::LocalAddr dst = CreateLocalAddr();
    dst = outDstOrg;

    CcuRep::LocalAddr src = CreateLocalAddr();
    src = srcOrg;

    std::vector<CcuRep::LocalAddr> scratch;
    for (uint32_t idx = 0; idx < size; idx++) {
        scratch.push_back(CreateLocalAddr());
        scratch[idx] = scratchOrg[idx];
    }

    CreateReduceLoop(size, dataType, outputDataType, opType);

    std::string loopType = GetReduceTypeStr(dataType, opType);
    uint32_t         expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    CcuRep::Variable sliceSizeExpansion = CreateVariable();

    if (expansionNum != 1) {
        CcuRep::Variable tmp = CreateVariable();
        tmp = ops_hccl::GetExpansionParam(expansionNum);
        dst.token += tmp;
    }

    // m部分
    CCU_IF(goSize.loopParam != 0)                   // goSize1
    {
        CcuRep::Variable loopParam = CreateVariable();
        loopParam = ops_hccl::GetLoopParam(0, moConfig.memSlice * moConfig.loopCount, 0);
        loopParam += goSize.loopParam;

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize          = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc = Loop(GetLoopBlockTag(loopType, 0))(dst, src, scratch, sliceSize, sliceSizeExpansion);

        CcuRep::Variable paraCfg = CreateVariable();
        paraCfg = ops_hccl::GetParallelParam(moConfig.loopCount - 1, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = ops_hccl::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc}, {loopParam}, paraCfg, offsetCfg);
    }

    CCU_IF(goSize.parallelParam != 0)               // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.addrOffset;
        }
        src.addr += goSize.addrOffset;              // goSize0
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += goSize.residual;  // goSize3
        }

        auto lc0 = Loop(GetLoopBlockTag(loopType, 0))(dst, src, scratch, goSize.residual, sliceSizeExpansion);

        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += goSize.residual;
        }
        src.addr += goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += goSize.residual;
        }

        CcuRep::Variable sliceSize = CreateVariable();
        sliceSize          = moConfig.memSlice;
        sliceSizeExpansion = moConfig.memSlice * expansionNum;

        auto lc1 = Loop(GetLoopBlockTag(loopType, 1))(dst, src, scratch, sliceSize, sliceSizeExpansion);

        CcuRep::Variable loopCfg0 = CreateVariable();
        loopCfg0 = ops_hccl::GetLoopParam(0, 0, 1);
        CcuRep::Variable loopCfg1 = CreateVariable();
        loopCfg1 = ops_hccl::GetLoopParam(0, 0, 1);
        CcuRep::Variable offsetCfg = CreateVariable();
        offsetCfg = ops_hccl::GetOffsetParam(moConfig.memSlice, moConfig.msInterleave, 1);

        LoopGroup({lc0, lc1}, {loopCfg0, loopCfg1}, goSize.parallelParam, offsetCfg);
    }
}

void CcuKernelAllReduceMeshMem2Mem1D::LoadArgs()
{
    Load(input_[rankId_]);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(scratch_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(mySliceSize_);
    Load(sliceOffset_);
    Load(isInputOutputEqual_);
    Load(localGoSize_);
    return;
}

void CcuKernelAllReduceMeshMem2Mem1D::PreSync()
{   
    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D LocalPost begin");
    // 互换内存信息
    for (auto t : channels_) {
        NotifyRecord(t, CKE_IDX_0, INPUT_XN_ID, input_[rankId_], 1 << INPUT_XN_ID);
        NotifyRecord(t, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(t, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }
    uint16_t allBit = 1 << INPUT_XN_ID | 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (auto t : channels_) {
        NotifyWait(t, CKE_IDX_0, allBit);
    }

    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D wait all end");
    return;
}

void CcuKernelAllReduceMeshMem2Mem1D::PostSync()
{
    for (auto t : channels_) {
        NotifyRecord(t, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (auto t : channels_) {
        NotifyWait(t, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
 
    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D AllReduce GroupWait end");
}

void CcuKernelAllReduceMeshMem2Mem1D::BcastLocToRmt(const CcuRep::Variable              &srcAddr,
                                                               const std::vector<CcuRep::Variable> &dstAddr)
{
    if (dstAddr.size() != channels_.size() + 1) {
         HCCL_ERROR("[ReduceRmtToLoc] srcAddr.size[%zu] != channels_ size[%zu] + 1", dstAddr.size(), channels_.size()); 
         return;
    }
    srcMem_.addr = srcAddr;
    srcMem_.addr += sliceOffset_;
    srcMem_.token = token_[rankId_];

    uint32_t channelIdx = 0;
    for (uint32_t rmtId = 0; rmtId < dstAddr.size(); rmtId++) {
        locEvent_.SetMask(1 << rmtId);
        if (rmtId == rankId_) {
            RecordEvent(locEvent_);
            continue;
        }
        remoteDstMem_.addr = dstAddr[rmtId];
        remoteDstMem_.addr += sliceOffset_;
        remoteDstMem_.token = token_[rmtId];

        WriteNb(channels_[channelIdx], remoteDstMem_, srcMem_, sliceSize_, locEvent_);
        channelIdx++;
    }
    locEvent_.SetMask((1 << rankSize_) - 1);
    WaitEvent(locEvent_);
}

void CcuKernelAllReduceMeshMem2Mem1D::ReduceRmtToLoc(const std::vector<CcuRep::Variable> &srcAddr,
                                                                const CcuRep::Variable              &dstAddr)
{
    if (srcAddr.size() != channels_.size() + 1) {
        HCCL_ERROR("[ReduceRmtToLoc] srcAddr.size[%zu] != channels_ size[%zu] +1", srcAddr.size(), channels_.size());
        return;
    }

    localDstMem_.addr = dstAddr;
    localDstMem_.addr += sliceOffset_;
    localDstMem_.token = token_[rankId_];

    CcuRep::Variable scratchOffset = CreateVariable();
    scratchOffset                  = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        reduceScatterSrc_[rankIdx].addr = srcAddr[rankIdx];
        reduceScatterSrc_[rankIdx].addr += sliceOffset_;
        reduceScatterSrc_[rankIdx].token = token_[rankIdx];

        reduceScatterDst_[rankIdx].addr = scratch_[rankId_];
        reduceScatterDst_[rankIdx].addr += scratchOffset;
        scratchOffset += normalSliceSize_;
        reduceScatterDst_[rankIdx].token = token_[rankId_];
    }

    uint32_t channelIdx = 0;
    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        locEvent_.SetMask(1 << rankIdx);
        if (rankIdx == rankId_) {
            RecordEvent(locEvent_);
        } else {
            ReadNb(channels_[channelIdx], reduceScatterDst_[rankIdx], reduceScatterSrc_[rankIdx], sliceSize_, locEvent_);
            channelIdx++;
        }
    }
    locEvent_.SetMask((1 << rankSize_) - 1);
    WaitEvent(locEvent_);
    CcuRep::LocalAddr  srcLoc = CreateLocalAddr();
    srcLoc.addr = reduceScatterSrc_[rankId_].addr;
    srcLoc.token = reduceScatterSrc_[rankId_].token;
    ReduceLoopGroup(localDstMem_, srcLoc, reduceScatterDst_,  localGoSize_, dataType_, outputDataType_, reduceOp_);
}

void CcuKernelAllReduceMeshMem2Mem1D::DoRepeatAllReduce()
{
    if (rankId_ != rankSize_ - 1) {
        sliceSize_ = normalSliceSize_;
    } else {
        sliceSize_ = lastSliceSize_;
    }
    ReduceRmtToLoc(input_, output_[rankId_]);
    BcastLocToRmt(output_[rankId_], output_);
}

HcclResult CcuKernelAllReduceMeshMem2Mem1D::Algorithm()
{
    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D run");
    InitResource();
    LoadArgs();
    PreSync();

    CCU_IF(mySliceSize_ != 0)
    {
        DoRepeatAllReduce();
    }
    PostSync();
    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D end");
    return HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllReduceMeshMem2Mem1D::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllReduceMeshMem2Mem1D *taskArg
        = dynamic_cast<const CcuTaskArgAllReduceMeshMem2Mem1D *>(&arg);
    // 空指针校验
    if (taskArg == nullptr) {
        HCCL_ERROR("CcuKernelAllReduceMeshMem2Mem1D::taskArg ptr is null");
    }
    uint64_t inputAddr                    = taskArg->inputAddr_;
    uint64_t outputAddr                   = taskArg->outputAddr_;
    uint64_t tokenInfo                    = taskArg->token_;
    uint64_t scratchAddr                  = taskArg->scratchAddr_;
    uint64_t currentRankSliceInputOffset  = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t normalSliceSize              = taskArg->normalSliceSize_;
    uint64_t lastSliceSize                = taskArg->lastSliceSize_;
    uint64_t mySliceSize                  = taskArg->mySliceSize_;
    uint64_t sliceOffset                  = taskArg->normalSliceSize_ * rankId_;
    uint64_t isInputOutputEqual           = taskArg->isInputOutputEqual_;

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        tokenInfo,
        scratchAddr,
        currentRankSliceInputOffset,
        currentRankSliceOutputOffset,
        normalSliceSize,
        lastSliceSize,
        mySliceSize,
        sliceOffset,
        isInputOutputEqual,
    };

    auto normalGoSize = CalGoSize(normalSliceSize);
    auto lastGoSize = CalGoSize(lastSliceSize);

    if (rankId_ != rankSize_ - 1 ) {
        taskArgs.insert(taskArgs.end(), normalGoSize.begin(), normalGoSize.end());
    } else {
        taskArgs.insert(taskArgs.end(), lastGoSize.begin(), lastGoSize.end());
    }

    HCCL_INFO("[CcuContextAllReduce1DMesh] TaskArgs: inputAddr[%llu], outputAddr[%llu], scratchAddr[%llu], "
              "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], normalSliceSize[%llu], "
              "lastSliceSize[%llu], mySliceSize[%llu], sliceOffset[%llu], isInputOutputEqual[%llu]",
              inputAddr, outputAddr, scratchAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset,
              normalSliceSize, lastSliceSize, mySliceSize, sliceOffset, isInputOutputEqual);

    return taskArgs;
}
} // namespace ops_hccl