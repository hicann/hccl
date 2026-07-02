/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_mesh1d_mem2mem.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

// bit序号，每种信号用一个bit
constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID   = 3;  
// cke序号
constexpr int CKE_IDX_0     = 0;

constexpr uint16_t BIT_NUM_PER_CKE = 16;

// 前向声明
static CcuResult ReduceLoopGroup(ReduceScatterMesh1DMem2MemContext &ctx, ccu::LocalAddr outDstOrg, 
    ccu::LocalAddr srcOrg, std::vector<ccu::LocalAddr> &scratchOrg);
static CcuResult PairwiseLocalReduce(ReduceScatterMesh1DMem2MemContext &ctx, ccu::LocalAddr myOutput, 
    std::vector<ccu::LocalAddr> &inputVec, ccu::Variable sliceSize);

static CcuResult ParseKernelArg(ReduceScatterMesh1DMem2MemContext &ctx, CcuKernelArgReduceScatterMesh1DMem2Mem *kernelArg)
{
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源
    ctx.input.resize(arg->rankSize);
    ctx.scratch.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    ctx.remoteInput.resize(arg->rankSize);
    ctx.scratchMem.resize(arg->rankSize);
    
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            // 本地资源，后续创建
        } else {
            HCCL_DEBUG("[CcuKernelReduceScatterMesh1DMem2Mem] MyRank[%u], PeerId[%u], ChannelId[%u]",
                       arg->rankId, peerId, channelIdx);
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.scratch[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], SCRATCH_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.moConfig.loopCount = REDUCE_SCATTER_LOOP_COUNT;
    ctx.moConfig.msInterleave = REDUCE_MS_CNT;
    ctx.moConfig.memSlice = CCU_MS_SIZE;

    ctx.resourceAllocated = false;

    // 创建events数组，每个CKE对应一个event
    uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    ctx.events.resize(RS_UNROLL_NUM * eventNum);

    ctx.constVar1 = 1;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, cnt++));
    return CCU_SUCCESS;
}

static void PreSync(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.scratch[arg->rankId], SCRATCH_XN_ID, CKE_IDX_0, 1 << SCRATCH_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << SCRATCH_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
}

static void PostSync(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

static void DoReduceScatterRead(ReduceScatterMesh1DMem2MemContext &ctx, uint32_t unrollIdx)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;
    uint32_t numEventsPerIter = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        uint32_t eventIdx = unrollIdx * numEventsPerIter + rankIdx / BIT_NUM_PER_CKE;
        uint32_t rankMask = 1 << (rankIdx % BIT_NUM_PER_CKE);

        if (rankIdx == arg->rankId) {
            if (arg->rankSize <= REDUCE_SCATTER_GROUP_REDUCE_MAX_PIECE_CNT) {
                ccu::EventRecord(ctx.events[eventIdx], rankMask);
            } else {
                ccu::LocalCopy(ctx.scratchMem[rankIdx], ctx.myInput, ctx.sliceSize, ctx.events[eventIdx], rankMask);
            }
        } else {
            ccu::Read(arg->channels[channelId], ctx.scratchMem[rankIdx], ctx.remoteInput[rankIdx], ctx.sliceSize, ctx.events[eventIdx], rankMask);
            channelId++;
        }
    }
}

static void DoReduceScatterWait(ReduceScatterMesh1DMem2MemContext &ctx, uint32_t unrollIdx)
{
    const auto *arg = ctx.arg;
    uint32_t numEventsPerIter = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint32_t i = 0; i < numEventsPerIter; i++) {
        uint32_t eventIdx = unrollIdx * numEventsPerIter + i;
        uint32_t sigNum = BIT_NUM_PER_CKE;
        if (arg->rankSize % BIT_NUM_PER_CKE != 0 && i == (numEventsPerIter - 1)) {
            sigNum = arg->rankSize % BIT_NUM_PER_CKE;
        }
        uint32_t allBit = (1 << sigNum) - 1;
        ccu::EventWait(ctx.events[eventIdx], allBit);
    }
}

static CcuResult DoReduceScatter(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    ccu::LocalAddr myOutput;
    myOutput.addr   = ctx.output;
    myOutput.addr  += ctx.currentRankSliceOutputOffset;
    myOutput.token  = ctx.token[arg->rankId];

    if (arg->rankSize <= REDUCE_SCATTER_GROUP_REDUCE_MAX_PIECE_CNT) {
        CCU_CHK_RET(ReduceLoopGroup(ctx, myOutput, ctx.myInput, ctx.scratchMem));
    } else {
        CCU_CHK_RET(PairwiseLocalReduce(ctx, myOutput, ctx.scratchMem, ctx.sliceSize));
    }
    return CCU_SUCCESS;
}

static CcuResult PairwiseLocalReduce(ReduceScatterMesh1DMem2MemContext &ctx, ccu::LocalAddr myOutput, 
    std::vector<ccu::LocalAddr> &inputVec, ccu::Variable sliceSize)
{
    const auto *arg = ctx.arg;
    ccu::Variable len;

    // 每轮将数据划分为2组做规约，总规约次数log2(n)
    uint32_t remainPieces = arg->rankSize;
    while (remainPieces > 1) {
        // 每轮将最后remain/2块，reduce到最前remian/2块
        uint32_t reducePieces = remainPieces / 2;
        uint32_t srcIdx = remainPieces - reducePieces;
        uint32_t dstIdx = 0;
        
        len = sliceSize;
        for (uint32_t i = 0; i < reducePieces - 1; i++) {
            len += sliceSize;
        }

        ccu::LocalReduce(inputVec[dstIdx], inputVec[srcIdx], len, ctx.dataType, ctx.reduceOp, ctx.events[0]);
        ccu::EventWait(ctx.events[0]);

        remainPieces -= reducePieces;
    }

    ccu::LocalCopy(myOutput, inputVec[0], sliceSize, ctx.events[0]);
    ccu::EventWait(ctx.events[0]);
    
    return CCU_SUCCESS;
}

static void InitReduceScatterAddr(ReduceScatterMesh1DMem2MemContext &ctx)
{
    ccu::Variable scratchOffset;
    const auto *arg = ctx.arg;
    scratchOffset = 0;

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            ctx.myInput.addr = ctx.input[rankIdx];
            ctx.myInput.addr += ctx.currentRankSliceInputOffset;
            ctx.myInput.token = ctx.token[rankIdx];
        } else {
            ctx.remoteInput[rankIdx].addr = ctx.input[rankIdx];
            ctx.remoteInput[rankIdx].addr += ctx.currentRankSliceInputOffset;
            ctx.remoteInput[rankIdx].token = ctx.token[rankIdx];
        }

        ctx.scratchMem[rankIdx].addr = ctx.scratch[arg->rankId];
        ctx.scratchMem[rankIdx].addr += scratchOffset;
        scratchOffset += ctx.sliceSize;
        ctx.scratchMem[rankIdx].token = ctx.token[arg->rankId];
    }
}

static void ResetReduceScatterAddr(ReduceScatterMesh1DMem2MemContext &ctx)
{
    ccu::Variable scratchOffset;
    const auto *arg = ctx.arg;
    scratchOffset = 0;
    ctx.myInput.addr = ctx.input[arg->rankId];
    ctx.myInput.addr += ctx.currentRankSliceInputOffset;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        ctx.scratchMem[rankIdx].addr = ctx.scratch[arg->rankId];
        ctx.scratchMem[rankIdx].addr += scratchOffset;
        scratchOffset += ctx.sliceSize;
    }
}

static CcuResult DoReduceScatterReduce(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    CCU_IF(ctx.readRepeatNum != UINT64_MAX)
    {
        ctx.flag = 0;
        CCU_WHILE(ctx.readRepeatNum != UINT64_MAX)
        {
            ctx.readRepeatNum += ctx.constVar1;
            CCU_IF(ctx.flag != 0)
            {
                for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                    ctx.scratchMem[rankIdx].addr += ctx.scratchRepeatStride;
                }
                ctx.myInput.addr += ctx.inputRepeatStride;
                ctx.output += ctx.outputRepeatStride;
            }
            CCU_CHK_RET(DoReduceScatter(ctx));
            ctx.flag = 1;
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatter(ReduceScatterMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    InitReduceScatterAddr(ctx);

    // 软件展开三阶段设计（仅支持repeatNum <= RS_UNROLL_NUM）:
    // Phase 1: 先下发所有ReadNb（非阻塞，event错开），步进scratch和input地址
    // Phase 2: 批量WaitEvent，等所有远端ReadNb数据到齐
    // Phase 3: Reduce串行（CCU_WHILE），恢复scratch/input地址后从头步进
    ctx.waitRepeatNum = ctx.repeatNum;
    ctx.readRepeatNum = ctx.repeatNum;

    // Phase 1: 第1轮迭代（不需要地址步进）
    CCU_IF(ctx.repeatNum != UINT64_MAX)
    {
        ctx.repeatNum += ctx.constVar1;
        DoReduceScatterRead(ctx, 0);
    }

    // 第2~RS_UNROLL_NUM轮迭代（需要地址步进）
    for (uint32_t i = 1; i < RS_UNROLL_NUM; i++) {
        CCU_IF(ctx.repeatNum != UINT64_MAX)
        {
            ctx.repeatNum += ctx.constVar1;
            for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                if (rankIdx == arg->rankId) {
                    ctx.myInput.addr += ctx.inputRepeatStride;
                } else {
                    ctx.remoteInput[rankIdx].addr += ctx.inputRepeatStride;
                }
                ctx.scratchMem[rankIdx].addr += ctx.scratchRepeatStride;
            }
            DoReduceScatterRead(ctx, i);
        }
    }

    // Phase 2: 批量WaitEvent，等所有远端ReadNb数据到齐
    for (uint32_t i = 0; i < RS_UNROLL_NUM; i++) {
        CCU_IF(ctx.waitRepeatNum != UINT64_MAX)
        {
            ctx.waitRepeatNum += ctx.constVar1;
            DoReduceScatterWait(ctx, i);
        }
    }

    // 恢复地址，为Phase 3 Reset起始位置
    ResetReduceScatterAddr(ctx);

    // Phase 3: Reduce串行
    CCU_CHK_RET(DoReduceScatterReduce(ctx));

    return CCU_SUCCESS;
}

static CcuResult CreateReduceLoop(ReduceScatterMesh1DMem2MemContext &ctx)
{
    constexpr uint32_t LOOP_NUM_16 = 16;
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, LOOP_NUM_16);

    if (ctx.IsLoopEntityRegistered("reduceScatterLocalReduce")) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity("reduceScatterLocalReduce");
    auto &loops = ctx.loopMap["reduceScatterLocalReduce"];

    const auto *arg = ctx.arg;
    uint32_t size = arg->rankSize;
    uint32_t expansionNum = GetReduceExpansionNum(arg->reduceOp, ctx.dataType, ctx.outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;
    constexpr uint32_t LOOP_NUM_2 = 2;
    for (int32_t index = 0; index < LOOP_NUM_2; index++) {
        ctx.loopScratch[index].resize(size);
        
        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event loopEvt = ctx.moRes.completedEvent[index];

        loops.body[index].reset(new ccu::Func(
            [&ctx, index, bufBase, loopEvt, size, expansionNum, usedBufNum]() {
            // Step 1: 将数据copy到ccuBuf（区分本rank和其他rank）
            for (uint32_t i = 0; i < size; i++) {
                if (i == ctx.arg->rankId) {
                    // 本rank数据从src copy
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], ctx.loopSrc[index], ctx.loopLen[index], loopEvt, 1 << i);
                } else {
                    // 其他rank数据从scratch copy
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], ctx.loopScratch[index][i], ctx.loopLen[index], loopEvt, 1 << i);
                }
            }
            ccu::EventWait(loopEvt, (1 << size) - 1);

            // Step 2: LocalReduce
            if (size > 1) {
                ccu::LocalReduce(&ctx.moRes.ccuBuf[bufBase], size, ctx.dataType, ctx.outputDataType, ctx.reduceOp, ctx.loopLen[index], loopEvt, 1);
                ccu::EventWait(loopEvt, 1);
            }

            // Step 3: Copy结果到dst
            ccu::LocalCopy(ctx.loopDst[index], ctx.moRes.ccuBuf[bufBase], ctx.loopLenExp[index], loopEvt, 1);
            ccu::EventWait(loopEvt, 1);
        }));

        loops.loops[index].reset(
            new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }

    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(ReduceScatterMesh1DMem2MemContext &ctx, ccu::LocalAddr outDstOrg, 
    ccu::LocalAddr srcOrg, std::vector<ccu::LocalAddr> &scratchOrg)
{
    const auto *arg = ctx.arg;
    const uint32_t size = scratchOrg.size();

    ccu::LocalAddr dst;
    dst.addr  = outDstOrg.addr;
    dst.token = outDstOrg.token;

    ccu::LocalAddr src;
    src.addr  = srcOrg.addr;
    src.token = srcOrg.token;

    std::vector<ccu::LocalAddr> scratch;
    for (uint32_t idx = 0; idx < size; idx++) {
        ccu::LocalAddr scratchAddr;
        scratchAddr.addr = scratchOrg[idx].addr;
        scratchAddr.token = scratchOrg[idx].token;
        scratch.push_back(scratchAddr);
    }

    CCU_CHK_RET(CreateReduceLoop(ctx));
    auto &loops = ctx.loopMap["reduceScatterLocalReduce"];

    uint32_t expansionNum = GetReduceExpansionNum(arg->reduceOp, ctx.dataType, ctx.outputDataType);
    ccu::Variable sliceSizeExpansion;
    ccu::Variable loopCfg0;
    ccu::Variable loopCfg1;
    ccu::Variable tmp;
    ccu::Variable loopParam;
    ccu::Variable paraCfg;
    ccu::Variable offsetCfg;

    if (expansionNum != 1) {
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    // 第一个loopgroup，处理m部分数据
    CCU_IF(ctx.goSize.loopParam != 0)
    {
        ccu::Variable sliceSize;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + ctx.goSize.loopParam;

        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop0的外部LocalAddr和Variable
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr  = src.addr;
        ctx.loopSrc[0].token = src.token;
        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0]       = sliceSize;
        ctx.loopLenExp[0]    = sliceSizeExpansion;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
        ccu::LoopGroup group(paraCfg, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    // 第二个loopgroup，处理n和p部分数据
    CCU_IF(ctx.goSize.parallelParam != 0)
    {
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.addrOffset;
        }
        src.addr += ctx.goSize.addrOffset;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + ctx.goSize.residual;
        }

        // 绑定loop0参数 (p部分)
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr  = src.addr;
        ctx.loopSrc[0].token = src.token;
        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0]    = ctx.goSize.residual;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.residual;
        }
        ccu::Variable sliceSize;
        src.addr += ctx.goSize.residual;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.residual;
        }

        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop1参数 (n部分)
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[1][i].addr = scratch[i].addr;
            ctx.loopScratch[1][i].token = scratch[i].token;
        }
        ctx.loopSrc[1].addr  = src.addr;
        ctx.loopSrc[1].token = src.token;
        ctx.loopDst[1].addr  = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopLen[1]    = sliceSize;
        ctx.loopLenExp[1] = sliceSizeExpansion;
        loopCfg0 = GetLoopParam(0, 0, 1);
        loopCfg1 = GetLoopParam(0, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(ctx.goSize.parallelParam, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceScatterMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterMesh1DMem2Mem *>(arg);

    ReduceScatterMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceScatterMesh1DMem2Mem] ReduceScatterMesh1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    ctx.sliceSize = (kernelArg->rankId == (kernelArg->rankSize - 1)) ? ctx.lastSliceSize : ctx.normalSliceSize;
    // sliceSize == 0时不需要read/wait/reduce，只需前后同步
    CCU_IF(ctx.sliceSize != 0)
    {
        CCU_CHK_RET(DoRepeatReduceScatter(ctx));
    }

    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceScatterMesh1DMem2Mem] ReduceScatterMesh1DMem2Mem end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
