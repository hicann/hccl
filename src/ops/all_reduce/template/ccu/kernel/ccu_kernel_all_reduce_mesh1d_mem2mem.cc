/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d_mem2mem.h"

namespace ops_hccl {
#define MAX_LOOP_NUM 2
constexpr int INPUT_XN_ID   = 0;
constexpr int OUTPUT_XN_ID  = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID   = 3;
constexpr int CKE_IDX_0     = 0;
constexpr uint16_t BIT_NUM_PER_CKE = 16;
constexpr uint16_t GROUP_REDUCE_MAX_PIECE_CNT = 8;

struct GroupReduceMem2MemVar {
    ccu::LocalAddr loopDst[MAX_LOOP_NUM];
    ccu::LocalAddr loopSrc[MAX_LOOP_NUM];
    std::array<std::vector<ccu::LocalAddr>, MAX_LOOP_NUM> loopScratch;
    ccu::Variable  loopLen[MAX_LOOP_NUM];
    ccu::Variable  loopLenExp[MAX_LOOP_NUM];
};

static CcuResult ParseKernelArg(AllReduceMeshMem2Mem1DContext &ctx, CcuKernelArgAllReduceMeshMem2Mem1D *kernelArg)
{
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuAllReduceMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllReduceMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuAllReduceMeshMem2Mem1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuAllReduceMeshMem2Mem1D] channels.size: [%u]", arg->channelCount);

    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.reduceScatterSrc.resize(arg->rankSize);
    ctx.reduceScatterDst.resize(arg->rankSize);
    ctx.events.resize(arg->rankSize);

    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult PairwiseLocalReduce(AllReduceMeshMem2Mem1DContext &ctx, ccu::LocalAddr myOutput, 
    std::vector<ccu::LocalAddr> &inputVec, ccu::Variable sliceSize, 
    HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType)
{
    const auto *arg = ctx.arg;
    ccu::Variable len;

    uint32_t remainPieces = arg->rankSize;
    while (remainPieces > 1) {
        uint32_t reducePieces = remainPieces / 2;
        uint32_t srcIdx = remainPieces - reducePieces;
        
        len = sliceSize;
        for (uint32_t i = 0; i < reducePieces - 1; i++) {
            len += sliceSize;
        }

        ccu::LocalReduce(inputVec[0], inputVec[srcIdx], len, dataType, opType, ctx.events[0], 1);
        ccu::EventWait(ctx.events[0], 1);

        remainPieces -= reducePieces;
    }

    ccu::LocalCopy(myOutput, inputVec[0], sliceSize, ctx.events[0], 1);
    ccu::EventWait(ctx.events[0], 1);
    return CCU_SUCCESS;
}
 
static CcuResult CreateReduceLoop(AllReduceMeshMem2Mem1DContext &ctx, 
    GroupReduceMem2MemVar &var, uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, LOOP_NUM);
    if (ctx.IsLoopEntityRegistered("reduce_mesh1d_mem2mem")) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity("reduce_mesh1d_mem2mem");
    auto &loops = ctx.loopMap["reduce_mesh1d_mem2mem"];

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < MAX_LOOP_NUM; index++) { // 需要实例化2个Loop
        var.loopScratch[index].resize(size);
        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event             e  = ctx.moRes.completedEvent[index];
        loops.body[index].reset(new ccu::Func(
            [&ctx, index, bufBase, e, size, &var, dataType, outputDataType, opType]() {
            for (uint32_t i = 0; i < size; i++) {
                if (i == ctx.arg->rankId) {
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], var.loopSrc[index], var.loopLen[index], e, 1 << i);
                } else {
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], var.loopScratch[index][i], var.loopLen[index], e, 1 << i);
                }
            }
            ccu::EventWait(e, (1 << size) - 1);
            if (size > 1) {
                ccu::LocalReduce(&ctx.moRes.ccuBuf[bufBase], size, dataType, outputDataType, opType, var.loopLen[index], e, 1); 
                ccu::EventWait(e, 1);
            }

            ccu::LocalCopy(var.loopDst[index], ctx.moRes.ccuBuf[bufBase], var.loopLenExp[index], e, 1);
            ccu::EventWait(e, 1);
        }));
        
        loops.loops[index].reset(
            new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }
    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(AllReduceMeshMem2Mem1DContext &ctx, ccu::LocalAddr srcOrg)
{
    const uint32_t size = ctx.reduceScatterDst.size();

    ccu::LocalAddr dst;
    dst.addr = ctx.localDstMem.addr;
    dst.token = ctx.localDstMem.token;

    ccu::LocalAddr src;
    src.addr = srcOrg.addr;
    src.token = srcOrg.token;

    std::vector<ccu::LocalAddr> scratch;
    scratch.resize(size);
    for (uint32_t idx = 0; idx < size; idx++) {
        scratch[idx].addr = ctx.reduceScatterDst[idx].addr;
        scratch[idx].token = ctx.token[ctx.arg->rankId];
    }
    GroupReduceMem2MemVar var;
    CCU_CHK_RET(CreateReduceLoop(ctx, var, size, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    auto &loops = ctx.loopMap["reduce_mesh1d_mem2mem"];

    uint32_t         expansionNum = GetReduceExpansionNum(ctx.reduceOp, ctx.dataType, ctx.outputDataType);
    ccu::Variable sliceSizeExpansion;

    if (expansionNum != 1) {
        ccu::Variable tmp;
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    // m部分
    CCU_IF(ctx.goSize.loopParam != 0)                   // goSize1
    {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + ctx.goSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        for (uint32_t i = 0; i < size; ++i) {
            var.loopScratch[0][i].addr = scratch[i].addr;
            var.loopScratch[0][i].token = scratch[i].token;
        }
        var.loopSrc[0].addr = src.addr;
        var.loopSrc[0].token = src.token;
        var.loopDst[0].addr  = dst.addr;
        var.loopDst[0].token = dst.token;
        var.loopLen[0]       = sliceSize;
        var.loopLenExp[0]    = sliceSizeExpansion;

        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
        ccu::LoopGroup group(paraCfg, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    CCU_IF(ctx.goSize.parallelParam != 0)               // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.addrOffset;
        }
        src.addr += ctx.goSize.addrOffset;              // goSize0
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion += ctx.goSize.residual;  // goSize3
        }

        for (uint32_t i = 0; i < size; ++i) {
            var.loopScratch[0][i].addr = scratch[i].addr;
            var.loopScratch[0][i].token = scratch[i].token;
        }
        var.loopSrc[0].addr = src.addr;
        var.loopSrc[0].token = src.token;
        var.loopDst[0].addr  = dst.addr;
        var.loopDst[0].token = dst.token;
        var.loopLen[0]       = ctx.goSize.residual;
        var.loopLenExp[0]    = sliceSizeExpansion;
        
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.residual;
        }
        src.addr += ctx.goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.residual;
        }

        ccu::Variable sliceSize;
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        for (uint32_t i = 0; i < size; ++i) {
            var.loopScratch[1][i].addr = scratch[i].addr;
            var.loopScratch[1][i].token = scratch[i].token;
        }
        var.loopSrc[1].addr = src.addr;
        var.loopSrc[1].token = src.token;
        var.loopDst[1].addr  = dst.addr;
        var.loopDst[1].token = dst.token;
        var.loopLen[1]       = sliceSize;
        var.loopLenExp[1]    = sliceSizeExpansion;

        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);

        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);

        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(ctx.goSize.parallelParam, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllReduceMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myScratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.mySliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllReduceMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllReduceMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D AllReduce GroupWait end");
    return CCU_SUCCESS;
}

static CcuResult BcastLocToRmt(AllReduceMeshMem2Mem1DContext &ctx, const ccu::Variable &srcAddr,
                   const std::vector<ccu::Variable> &dstAddr)
{
    const auto *arg = ctx.arg;
    if (dstAddr.size() != arg->channelCount + 1) {
         HCCL_ERROR("[BcastLocToRmt] dstAddr.size[%zu] != channels_ size[%zu] + 1", dstAddr.size(), arg->channelCount); 
         return CCU_SUCCESS;
    }
    ctx.srcMem.addr = srcAddr;
    ctx.srcMem.addr += ctx.sliceOffset;
    ctx.srcMem.token = ctx.token[arg->rankId];

    uint32_t channelIdx = 0;
    for (uint32_t rmtId = 0; rmtId < dstAddr.size(); rmtId++) {
        uint32_t eventIdx = rmtId / BIT_NUM_PER_CKE;
        if (rmtId == arg->rankId) {
            ccu::EventRecord(ctx.events[eventIdx], 1 << (rmtId % BIT_NUM_PER_CKE));
            continue;
        }
        ctx.remoteDstMem.addr = dstAddr[rmtId];
        ctx.remoteDstMem.addr += ctx.sliceOffset;
        ctx.remoteDstMem.token = ctx.token[rmtId];

        ccu::Write(arg->channels[channelIdx], ctx.remoteDstMem, ctx.srcMem, ctx.sliceSize, ctx.events[eventIdx], 1 << (rmtId % BIT_NUM_PER_CKE));
        channelIdx++;
    }
    uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint32_t eventIdx = 0; eventIdx < eventNum; eventIdx++) {
        uint32_t sigNum = BIT_NUM_PER_CKE;
        if (arg->rankSize % BIT_NUM_PER_CKE != 0 && eventIdx == (eventNum - 1)) {
            sigNum = arg->rankSize % BIT_NUM_PER_CKE;
        }
        ccu::EventWait(ctx.events[eventIdx], (1 << sigNum) - 1);
    }
    return CCU_SUCCESS;
}

static CcuResult DoLocalReduce(AllReduceMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankSize <= GROUP_REDUCE_MAX_PIECE_CNT) {
        ccu::LocalAddr  srcLoc;
        srcLoc.addr = ctx.reduceScatterSrc[arg->rankId].addr;
        srcLoc.token = ctx.reduceScatterSrc[arg->rankId].token;
        CCU_CHK_RET(ReduceLoopGroup(ctx, srcLoc));
    } else {
        CCU_CHK_RET(PairwiseLocalReduce(ctx, ctx.localDstMem, ctx.reduceScatterDst, ctx.sliceSize, 
            ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    }
    return CCU_SUCCESS;
}

static CcuResult ReduceRmtToLoc(AllReduceMeshMem2Mem1DContext &ctx, const std::vector<ccu::Variable> &srcAddr,
                    const ccu::Variable              &dstAddr)
{
    const auto *arg = ctx.arg;
    if (srcAddr.size() != arg->channelCount + 1) {
        HCCL_ERROR("[ReduceRmtToLoc] srcAddr.size[%zu] != channels_ size[%zu] +1", srcAddr.size(), arg->channelCount);
        return CCU_SUCCESS;
    }

    ctx.localDstMem.addr = dstAddr;
    ctx.localDstMem.addr += ctx.sliceOffset;
    ctx.localDstMem.token = ctx.token[arg->rankId];

    ccu::Variable scratchOffset;
    scratchOffset                  = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {      
        ctx.reduceScatterSrc[rankIdx].addr = srcAddr[rankIdx];
        ctx.reduceScatterSrc[rankIdx].addr += ctx.sliceOffset;
        ctx.reduceScatterSrc[rankIdx].token = ctx.token[rankIdx];

        ctx.reduceScatterDst[rankIdx].addr = ctx.myScratch;
        ctx.reduceScatterDst[rankIdx].addr += scratchOffset;
        scratchOffset += ctx.sliceSize;
        ctx.reduceScatterDst[rankIdx].token = ctx.token[ctx.arg->rankId];
    }

    uint32_t channelIdx = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        uint32_t eventIdx = rankIdx / BIT_NUM_PER_CKE;
        if (rankIdx == arg->rankId) {
            if (arg->rankSize <= GROUP_REDUCE_MAX_PIECE_CNT) {
                ccu::EventRecord(ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
            } else {
                ccu::LocalAddr src;
                src.addr = ctx.reduceScatterSrc[rankIdx].addr;
                src.token = ctx.reduceScatterSrc[rankIdx].token;
                ccu::LocalCopy(ctx.reduceScatterDst[rankIdx], src, ctx.sliceSize, ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
            }
        } else {    
            ccu::Read(arg->channels[channelIdx], ctx.reduceScatterDst[rankIdx], ctx.reduceScatterSrc[rankIdx], ctx.sliceSize, ctx.events[eventIdx], 1 << (rankIdx % BIT_NUM_PER_CKE));
            channelIdx++;
        }
    }
    uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint32_t i = 0; i < eventNum; i++) {
        uint32_t sigNum = BIT_NUM_PER_CKE;
        if (arg->rankSize % BIT_NUM_PER_CKE != 0 && i == (eventNum - 1)) {
            sigNum = arg->rankSize % BIT_NUM_PER_CKE;
        }
        ccu::EventWait(ctx.events[i], (1 << sigNum) - 1);
    }
    CCU_CHK_RET(DoLocalReduce(ctx));
    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllReduce(AllReduceMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId != arg->rankSize - 1) {
        ctx.sliceSize = ctx.normalSliceSize;
    } else {
        ctx.sliceSize = ctx.lastSliceSize;
    }
    CCU_CHK_RET(ReduceRmtToLoc(ctx, ctx.input, ctx.output[arg->rankId]));
    CCU_CHK_RET(BcastLocToRmt(ctx, ctx.output[arg->rankId], ctx.output));
    return CCU_SUCCESS;
}

CcuResult CcuAllReduceMeshMem2Mem1DKernel(CcuKernelArg arg)
{
 auto *kernelArg = static_cast<CcuKernelArgAllReduceMeshMem2Mem1D *>(arg);

    AllReduceMeshMem2Mem1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_IF(ctx.mySliceSize != 0)
    {
        CCU_CHK_RET(DoRepeatAllReduce(ctx));
    }

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllReduceMeshMem2Mem1D] AllReduceMeshMem2Mem1D end");

    return CCU_SUCCESS;
}
} // namespace ops_hccl