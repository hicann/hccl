/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_kernel_utils.h"


namespace ops_hccl {

constexpr int CKE_IDX_0   = 0;
constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;

static CcuResult ParseKernelArg(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx, CcuKernelArgReduceScatterOmniPipeMesh1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankId = kernelArg->rankId;
    ctx.rankSize = kernelArg->rankSize;
    ctx.userRank = kernelArg->subCommRanks[0][ctx.rankId];

    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterOmniPipeMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeMesh1DMem2Mem] userRank[%u] rankId[%u], rankSize[%u], "
                "dataType[%d], outputDataType[%d], reduceOp[%d]", ctx.userRank, ctx.rankId, ctx.rankSize,
                ctx.dataType, ctx.outputDataType, ctx.reduceOp);
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterOmniPipeMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源
    ctx.input.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            // 本地资源，默认构造
            continue;
        } else {
            HCCL_DEBUG("[CcuKernelReduceScatterOmniPipeMesh1DMem2Mem] rankId[%u], peerId[%u], channelId[%u]",
                       arg->rankId, peerId, channelIdx);
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.moConfig.msInterleave = CCU_MS_INTERLEAVE;
    ctx.moConfig.loopCount = CCU_M2M_LOCAL_COPY_LOOP_COUNT;
    ctx.moConfig.memSlice = CCU_MS_SIZE;

    ctx.resourceAllocated = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, cnt++));
    HCCL_DEBUG("[%s] end", __func__);
    return CCU_SUCCESS;
}

static CcuResult PreSync(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i],
                    ctx.input[arg->rankId],INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i],
                    ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    return CcuResult::CCU_SUCCESS;
}

static CcuResult PostSync(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_DEBUG("[%s] end");
    return CcuResult::CCU_SUCCESS;
}

static CcuResult CreateReduceLoop(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx, uint32_t size)
{
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, CCU_M2M_LOCAL_COPY_LOOP_COUNT);
 
    if (ctx.IsLoopEntityRegistered("reduceScatterOmniLocalReduce")) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity("reduceScatterOmniLocalReduce");
    auto &loops = ctx.loopMap["reduceScatterOmniLocalReduce"];
    
    uint32_t expansionNum = GetReduceExpansionNum(ctx.reduceOp, ctx.dataType, ctx.outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;
 
    for (int32_t index = 0; index < 2; index++) { // 需要实例化2个Loop
        ctx.loopScratch[index].resize(size);
        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event loopEvt = ctx.moRes.completedEvent[index];

        loops.body[index].reset(new ccu::Func(
            [&ctx, index, bufBase, loopEvt, size, expansionNum, usedBufNum]() {
                // Step 1: 将数据copy到ccuBuf
                for (uint32_t i = 0; i < size; ++i) {
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], ctx.loopScratch[index][i], ctx.loopLen[index], loopEvt, 1 << i);
                }
                ccu::EventWait(loopEvt, (1 << size) - 1);

                // Step 2: LocalReduce
                if (size > 1) {
                    ccu::LocalReduce(&ctx.moRes.ccuBuf[bufBase], size, ctx.dataType, ctx.outputDataType, ctx.reduceOp, ctx.loopLen[index], loopEvt, 1);
                    ccu::EventWait(loopEvt, 1);
                }

                // Step3: Copy结果到dst
                ccu::LocalCopy(ctx.loopDst[index], ctx.moRes.ccuBuf[bufBase], ctx.loopLenExp[index], loopEvt, 1);
                ccu::EventWait(loopEvt, 1);
            }
        ));
        
        loops.loops[index].reset(new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }
 
    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx, ccu::LocalAddr outDstOrg, std::vector<ccu::LocalAddr> &scratchOrg)
{
    const auto *arg = ctx.arg;
    const uint32_t size = scratchOrg.size();
 
    ccu::LocalAddr dst;
    dst.addr = outDstOrg.addr;
    dst.token = outDstOrg.token;
 
    std::vector<ccu::LocalAddr> scratch;
    for (uint32_t idx = 0; idx < size; idx++) {
        ccu::LocalAddr scratchAddr;
        scratchAddr.addr = scratchOrg[idx].addr;
        scratchAddr.token = scratchOrg[idx].token;
        scratch.push_back(scratchAddr);
    }
 
    CCU_CHK_RET(CreateReduceLoop(ctx, size));
    auto &loops = ctx.loopMap["reduceScatterOmniLocalReduce"];
 
    uint32_t expansionNum = GetReduceExpansionNum(ctx.reduceOp, ctx.dataType, ctx.outputDataType);
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

        // 绑定loop0的参数（m部分）
        for (uint32_t i = 0; i < size; ++i) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0] = sliceSize;
        ctx.loopLenExp[0] = sliceSizeExpansion;
 
        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);
 
        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
        ccu::LoopGroup group(paraCfg, offsetCfg, 1, grpLoops);
    }
 
    CCU_IF(ctx.goSize.parallelParam != 0)               // goSize2
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.addrOffset;
        }
 
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.addrOffset;
        }
 
        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + ctx.goSize.residual;  // goSize3
        }

        // 绑定loop0的参数（p部分）
        for (uint32_t i = 0; i < size; ++i) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopDst[0].addr = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0] = ctx.goSize.residual;
        ctx.loopLenExp[0] = sliceSizeExpansion;
 
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.goSize.residual;
        }
 
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.goSize.residual;
        }
 
        ccu::Variable sliceSize;
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop1的参数（n部分）
        for (uint32_t i = 0; i < size; ++i) {
            ctx.loopScratch[1][i].addr = scratch[i].addr;
            ctx.loopScratch[1][i].token = scratch[i].token;
        }
        ctx.loopDst[1].addr = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopLen[1] = sliceSize;
        ctx.loopLenExp[1] = sliceSizeExpansion;
 
        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);
        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);
 
        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(ctx.goSize.parallelParam, offsetCfg, NUM_TWO, grpLoops);
    }

    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatter(ReduceScatterOmniPipeMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
        
    ccu::LocalAddr dst;
    std::vector<ccu::RemoteAddr> src;

    src.resize(arg->rankSize);
    dst.addr = ctx.input[arg->rankId];
    dst.addr += ctx.inputSliceStride;
    dst.addr += ctx.inputOmniPipeSliceStride;
    dst.token = ctx.token[arg->rankId];
    src[arg->rankSize - 1].addr = dst.addr;
    src[arg->rankSize - 1].token = dst.token;

    // 准备源地址
    uint32_t idx = 0;
    for (auto i = 0; i < ctx.rankSize; ++i) {
        if (i == arg->rankId) { continue; }
        src[idx].addr = ctx.input[i];
        src[idx].addr += ctx.inputSliceStride;
        src[idx].addr += ctx.inputOmniPipeSliceStride;
        src[idx].token = ctx.token[i];
        idx++;
    }

    // 准备目的地址
    std::vector<ccu::LocalAddr> scratchMem;
    scratchMem.resize(arg->rankSize);
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (auto i = 0; i < ctx.rankSize; ++i) {
        scratchMem[i].addr = ctx.scratch;
        scratchMem[i].addr += scratchOffset;
        scratchMem[i].token = ctx.token[arg->rankId];
        scratchOffset += ctx.sliceSize;
    }

    // 从远端读
    uint32_t channelId = 0;
    for (auto i = 0; i < ctx.rankSize; ++i) {
        uint32_t rankMask = 1 << i;
        if (i == ctx.rankId) {
            ccu::EventRecord(ctx.event, rankMask);
            continue;
        }
        CCU_IF(ctx.sliceSize != 0) { ccu::Read(arg->channels[channelId], scratchMem[i], src[channelId], ctx.sliceSize, ctx.event, rankMask); }
        CCU_IF(ctx.sliceSize == 0) { ccu::EventRecord(ctx.event, rankMask); }
        channelId++;
    }
    // 等读完所有对端
    uint32_t allBit = (1 << ctx.rankSize) - 1;
    ccu::EventWait(ctx.event, allBit);

    // 做reduce
    scratchMem[ctx.rankId].addr = dst.addr;
    scratchMem[ctx.rankId].token = dst.token;
    CCU_IF(ctx.sliceSize != 0) { ReduceLoopGroup(ctx, dst, scratchMem); }
    HCCL_DEBUG("[DoRepeatReduceScatter] userRank[%u] rankId[%u] do repeat ReduceScatter success", ctx.userRank, ctx.rankId);

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceScatterOmniPipeMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterOmniPipeMesh1DMem2Mem *>(arg);

    ReduceScatterOmniPipeMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceScatterOmniPipeMesh1DMem2Mem] ReduceScatterOmniPipeMesh1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    PreSync(ctx);
    CCU_CHK_RET(DoRepeatReduceScatter(ctx));
    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeMesh1DMem2Mem] ReduceScatterOmniPipeMesh1DMem2Mem end");

    return CCU_SUCCESS;
}
} // namespace ops_hccl