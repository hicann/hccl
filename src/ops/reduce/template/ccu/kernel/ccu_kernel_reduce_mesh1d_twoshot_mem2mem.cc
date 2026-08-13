/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh1d_twoshot_mem2mem.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

static CcuResult
ParseKernelArg(ReduceMesh1DTwoShotMem2MemContext& ctx, CcuKernelArgReduceMesh1DTwoShotMem2Mem* kernelArg)
{
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceMesh1DTwoShotMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    ctx.input.resize(arg->rankSize);
    ctx.scratch.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    ctx.scratchMem.resize(arg->rankSize);
    ctx.remoteInput.resize(arg->rankSize);
    ctx.constVar1 = 1;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceMesh1DTwoShotMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelReduceMesh1DTwoShotMem2Mem] channels.size: [%u]", arg->channelCount);

    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            continue;
        }
        ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
        ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
        channelIdx++;
    }
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceMesh1DTwoShotMem2MemContext& ctx)
{
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[ctx.arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, cnt++));
    return CCU_SUCCESS;
}

static void PreSync(ReduceMesh1DTwoShotMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.input[arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.output[arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
}

static void PostSync(ReduceMesh1DTwoShotMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

static void InitSliceInfo(ReduceMesh1DTwoShotMem2MemContext& ctx)
{
    if (ctx.arg->rankId == ctx.arg->rankSize - 1) {
        ctx.mySliceSize = ctx.lastSliceSize;
    } else {
        ctx.mySliceSize = ctx.normalSliceSize;
    }
    ctx.sliceSize = ctx.mySliceSize;

    ccu::Variable scratchOffset;
    ctx.myScratchOffset = 0;
    for (uint32_t k = 0; k < ctx.arg->rankId; k++) {
        ctx.myScratchOffset += ctx.normalSliceSize;
    }

    scratchOffset = 0;
    for (uint32_t k = 0; k < ctx.arg->rankSize; k++) {
        ctx.scratchMem[k].addr = ctx.scratch[ctx.arg->rankId];
        ctx.scratchMem[k].addr += scratchOffset;
        ctx.scratchMem[k].token = ctx.token[ctx.arg->rankId];
        scratchOffset += ctx.sliceSize;
    }
}

// ============================================
// CreateReduceLoop + ReduceLoopGroup
// ============================================
static CcuResult CreateReduceLoop(ReduceMesh1DTwoShotMem2MemContext& ctx)
{
    constexpr uint32_t LOOP_NUM = 16;
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, LOOP_NUM);

    if (ctx.IsLoopEntityRegistered("twoshotReduce")) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity("twoshotReduce");
    auto& loops = ctx.loopMap["twoshotReduce"];

    const auto* arg = ctx.arg;
    uint32_t size = arg->rankSize;
    uint32_t expansionNum = GetReduceExpansionNum(ctx.reduceOp, ctx.dataType, ctx.outputDataType);

    for (int32_t index = 0; index < LOOP_GROUP_NUM; index++) {
        ctx.loopScratch[index].resize(size);
        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event loopEvt = ctx.moRes.completedEvent[index];

        loops.body[index].reset(new ccu::Func([&ctx, index, bufBase, loopEvt, size, expansionNum]() {
            for (uint32_t i = 0; i < size; i++) {
                if (i == ctx.arg->rankId) {
                    ccu::LocalCopy(
                        ctx.moRes.ccuBuf[bufBase + i], ctx.loopSrc[index], ctx.loopLen[index], loopEvt, 1 << i);
                } else {
                    ccu::LocalCopy(
                        ctx.moRes.ccuBuf[bufBase + i], ctx.loopScratch[index][i], ctx.loopLen[index], loopEvt, 1 << i);
                }
            }
            ccu::EventWait(loopEvt, (1 << size) - 1);

            if (size > 1) {
                ccu::LocalReduce(
                    &ctx.moRes.ccuBuf[bufBase], size, ctx.dataType, ctx.outputDataType, ctx.reduceOp,
                    ctx.loopLen[index], loopEvt, 1);
                ccu::EventWait(loopEvt, 1);
            }

            ccu::LocalCopy(ctx.loopDst[index], ctx.moRes.ccuBuf[bufBase], ctx.loopLenExp[index], loopEvt, 1);
            ccu::EventWait(loopEvt, 1);
        }));

        loops.loops[index].reset(new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }
    return CCU_SUCCESS;
}

static CcuResult DoReduceLoopM(
    ReduceMesh1DTwoShotMem2MemContext& ctx, uint32_t size, uint32_t expansionNum, const ccu::LocalAddr& src,
    const ccu::LocalAddr& dst, const std::vector<ccu::LocalAddr>& scratch)
{
    auto& loops = ctx.loopMap["twoshotReduce"];
    CCU_IF(ctx.goSize.loopParam != 0)
    {
        ccu::Variable loopParam;
        ccu::Variable sliceSize;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + ctx.goSize.loopParam;
        ccu::Variable sliceSizeExpansion;
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        ccu::Variable paraCfg;
        ccu::Variable offsetCfg;
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr = src.addr;
        ctx.loopSrc[0].token = src.token;
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0] = sliceSize;
        ctx.loopLenExp[0] = sliceSizeExpansion;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{*loops.loops[0]};
        ccu::LoopGroup group(paraCfg, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }
    return CCU_SUCCESS;
}

static CcuResult DoReduceLoopP(
    ReduceMesh1DTwoShotMem2MemContext& ctx, uint32_t size, uint32_t expansionNum, ccu::LocalAddr& src,
    ccu::LocalAddr& dst, std::vector<ccu::LocalAddr>& scratch)
{
    CCU_IF(ctx.goSize.parallelParam != 0)
    {
        ccu::Variable sliceSizeExpansion;
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
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr = src.addr;
        ctx.loopSrc[0].token = src.token;
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0] = ctx.goSize.residual;
        ctx.loopLenExp[0] = sliceSizeExpansion;
    }
    return CCU_SUCCESS;
}

static CcuResult DoReduceLoopN(
    ReduceMesh1DTwoShotMem2MemContext& ctx, uint32_t size, uint32_t expansionNum, ccu::LocalAddr& src,
    ccu::LocalAddr& dst, std::vector<ccu::LocalAddr>& scratch)
{
    auto& loops = ctx.loopMap["twoshotReduce"];
    CCU_IF(ctx.goSize.parallelParam != 0)
    {
        ccu::Variable sliceSizeExpansion;
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.residual;
        }
        src.addr += ctx.goSize.residual;
        ccu::Variable sliceSize;
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.residual;
        }
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;
        ccu::Variable loopCfg0;
        ccu::Variable loopCfg1;
        ccu::Variable offsetCfg;
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[1][i].addr = scratch[i].addr;
            ctx.loopScratch[1][i].token = scratch[i].token;
        }
        ctx.loopSrc[1].addr = src.addr;
        ctx.loopSrc[1].token = src.token;
        ctx.loopDst[1].addr = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopLen[1] = sliceSize;
        ctx.loopLenExp[1] = sliceSizeExpansion;
        loopCfg0 = GetLoopParam(0, 0, 1);
        loopCfg1 = GetLoopParam(0, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);
        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{*loops.loops[0], *loops.loops[1]};
        ccu::LoopGroup group(ctx.goSize.parallelParam, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }
    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(
    ReduceMesh1DTwoShotMem2MemContext& ctx, ccu::LocalAddr& outDstOrg, ccu::LocalAddr& srcOrg,
    std::vector<ccu::LocalAddr>& scratchOrg)
{
    const uint32_t size = scratchOrg.size();
    ccu::LocalAddr dst;
    dst.addr = outDstOrg.addr;
    dst.token = outDstOrg.token;
    ccu::LocalAddr src;
    src.addr = srcOrg.addr;
    src.token = srcOrg.token;
    std::vector<ccu::LocalAddr> scratch;
    scratch.resize(size);
    for (uint32_t idx = 0; idx < size; idx++) {
        scratch[idx].addr = scratchOrg[idx].addr;
        scratch[idx].token = scratchOrg[idx].token;
    }

    CCU_CHK_RET(CreateReduceLoop(ctx));
    uint32_t expansionNum = GetReduceExpansionNum(ctx.reduceOp, ctx.dataType, ctx.outputDataType);
    ccu::Variable tmp;
    if (expansionNum != 1) {
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    CCU_CHK_RET(DoReduceLoopM(ctx, size, expansionNum, src, dst, scratch));
    CCU_CHK_RET(DoReduceLoopP(ctx, size, expansionNum, src, dst, scratch));
    CCU_CHK_RET(DoReduceLoopN(ctx, size, expansionNum, src, dst, scratch));
    return CCU_SUCCESS;
}

// ============================================
// ReduceScatter 阶段
// 先 Read 远端 rank 对应数据片到本地 scratch（scratch 按 sliceSize 间距）
// 后 ReduceLoopGroup 分块流水线归约到本 rank scratch
// ============================================
static CcuResult DoReduceScatter(ReduceMesh1DTwoShotMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;

    // 本 rank 的 input 地址（self 数据来源，传给 ReduceLoopGroup 作为 src）
    ctx.myInput.addr = ctx.input[arg->rankId];
    ctx.myInput.addr += ctx.myScratchOffset; // myScratchOffset = rankId * normalSliceSize，是 input 中的偏移
    ctx.myInput.token = ctx.token[arg->rankId];

    uint32_t channelIdx = 0;
    for (uint32_t peerId = 0; peerId < arg->rankSize; peerId++) {
        uint16_t rankMask = 1 << peerId;
        if (peerId == arg->rankId) {
            ccu::EventRecord(ctx.event, rankMask);
        } else {
            // 远端 input 地址：远端 input 基址 + myScratchOffset（本 rank slice 在 input 中的偏移）
            ctx.remoteInput[peerId].addr = ctx.input[peerId];
            ctx.remoteInput[peerId].addr += ctx.myScratchOffset;
            ctx.remoteInput[peerId].token = ctx.token[peerId];

            // Read：远端 input → 本地 scratchMem[peerId]（按 sliceSize 间距），大小为 sliceSize
            CCU_IF(ctx.mySliceSize != 0)
            {
                ccu::Read(
                    arg->channels[channelIdx], ctx.scratchMem[peerId], ctx.remoteInput[peerId], ctx.sliceSize,
                    ctx.event, rankMask);
            }
            CCU_ELSE { ccu::EventRecord(ctx.event, rankMask); }
            channelIdx++;
        }
    }

    uint16_t allBit = (1 << arg->rankSize) - 1;
    ccu::EventWait(ctx.event, allBit);

    // ReduceLoopGroup：self input + scratch[0..N-1] → dst
    // root 直接写到 output，non-root 写到 scratchMem[rankId]
    CCU_IF(ctx.mySliceSize != 0)
    {
        if (arg->rankId == arg->rootId) {
            ctx.myOutput.addr = ctx.output[arg->rankId];
            ctx.myOutput.addr += ctx.myScratchOffset;
            ctx.myOutput.token = ctx.token[arg->rankId];
            CCU_CHK_RET(ReduceLoopGroup(ctx, ctx.myOutput, ctx.myInput, ctx.scratchMem));
        } else {
            CCU_CHK_RET(ReduceLoopGroup(ctx, ctx.scratchMem[arg->rankId], ctx.myInput, ctx.scratchMem));
        }
    }

    return CCU_SUCCESS;
}

// ============================================
// Gather 阶段
// root: reduce 结果已直接写入 output，无需 GroupCopy
// non-root: Write scratch 归约结果到 root 的 output
// ============================================
static CcuResult DoGather(ReduceMesh1DTwoShotMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;

    CCU_IF(ctx.mySliceSize != 0)
    {
        if (arg->rankId != arg->rootId) {
            uint16_t channelToRoot = (arg->rootId < arg->rankId) ? arg->rootId : arg->rootId - 1;

            ctx.remoteOutput.addr = ctx.output[arg->rootId];
            ctx.remoteOutput.addr += ctx.myScratchOffset;
            ctx.remoteOutput.token = ctx.token[arg->rootId];

            ccu::Write(
                arg->channels[channelToRoot], ctx.remoteOutput, ctx.scratchMem[arg->rankId], ctx.mySliceSize, ctx.event,
                1);
            ccu::EventWait(ctx.event, 1);
        }
    }
    return CCU_SUCCESS;
}

// ============================================
// 主循环：重复执行 ReduceScatter + Gather
// ============================================
static CcuResult DoRepeatTwoshot(ReduceMesh1DTwoShotMem2MemContext& ctx)
{
    ctx.flag = 0;

    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX)
    {
        CCU_IF(ctx.flag != 0)
        {
            ctx.input[ctx.arg->rankId] += ctx.inputRepeatStride;
            ctx.output[ctx.arg->rootId] += ctx.outputRepeatStride;
        }
        CCU_CHK_RET(DoReduceScatter(ctx));
        CCU_CHK_RET(DoGather(ctx));
        ctx.repeatNumVar += ctx.constVar1;
        ctx.flag = 1;
    }
    return CCU_SUCCESS;
}

// ============================================
// Kernel 主入口：PreSync → DoRepeatTwoshot → PostSync
// ============================================
CcuResult CcuReduceMesh1DTwoShotMem2MemKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<CcuKernelArgReduceMesh1DTwoShotMem2Mem*>(arg);
    ReduceMesh1DTwoShotMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.enginePool = 0;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO(
        "[CcuKernelReduceMesh1DTwoShotMem2Mem] run, rankId[%u], rankSize[%llu], rootId[%u]", kernelArg->rankId,
        kernelArg->rankSize, kernelArg->rootId);
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    PreSync(ctx);

    InitSliceInfo(ctx);

    CCU_IF(ctx.mySliceSize != 0) { CCU_CHK_RET(DoRepeatTwoshot(ctx)); }

    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceMesh1DTwoShotMem2Mem] end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl
