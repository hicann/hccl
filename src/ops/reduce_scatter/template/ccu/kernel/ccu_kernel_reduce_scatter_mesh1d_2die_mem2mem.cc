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

const std::string LOCAL_REDUCE_LOOP_BLOCK_TAG{"_local_reduce_loop_"};

static CcuResult ParseKernelArg(ReduceScatterMesh1D2DieMem2MemContext &ctx, CcuKernelArgReduceScatterMesh1D2DieMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankId = kernelArg->rankId;
    ctx.gRankSize = kernelArg->gRankSize;
    ctx.rankSize = kernelArg->rankSize;
    ctx.subRankGroup = kernelArg->subRankGroup;
    ctx.isReduceToOutput = kernelArg->isReduceToOutput;

    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_INFO("[CcuKernelReduceScatterMesh1D2DieMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint16_t channelIdx = 0;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterMesh1D2DieMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.input.resize(ctx.rankSize);
    ctx.scratch.resize(ctx.rankSize);
    ctx.token.resize(ctx.rankSize);

    for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        if (ctx.subRankGroup[rankIdx] == ctx.rankId) {
            // 本地资源
        } else {
            ctx.input[rankIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.scratch[rankIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], SCRATCH_XN_ID);
            ctx.token[rankIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    if (!ctx.isReduceToOutput) {
        ctx.input.resize(ctx.rankSize + 1);
        ctx.scratch.resize(ctx.rankSize + 1);
        ctx.token.resize(ctx.rankSize + 1);
    }
    ctx.myRankIdx = ctx.input.size() - 1;

    ctx.output = ccu::Variable();
    ctx.currentRankSliceInputOffset = ccu::Variable();
    ctx.sliceSize = ccu::Variable();
    ctx.inputRepeatStride = ccu::Variable();
    ctx.outputRepeatStride = ccu::Variable();
    ctx.repeatNum = ccu::Variable();
    ctx.flag = ccu::Variable();

    ctx.remoteInput.resize(ctx.rankSize);
    ctx.scratchMem.resize(ctx.rankSize);

    ctx.moConfig.loopCount = REDUCE_SCATTER_LOOP_COUNT;
    ctx.moConfig.msInterleave = REDUCE_MS_CNT;
    ctx.moConfig.memSlice = CCU_MS_SIZE;

    ctx.resourceAllocated = false;
    ctx.event = ccu::Event();

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.myRankIdx], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[ctx.myRankIdx], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceGoSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceGoSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceGoSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceGoSize.residual, cnt++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[ctx.myRankIdx], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.scratch[ctx.myRankIdx], SCRATCH_XN_ID, CKE_IDX_0, 1 << SCRATCH_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << SCRATCH_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

static std::string GetLoopBlockTag(std::string loopType, int32_t index)
{
    return loopType + LOCAL_REDUCE_LOOP_BLOCK_TAG + std::to_string(index);
}

static CcuResult CreateReduceLoop(ReduceScatterMesh1D2DieMem2MemContext &ctx, uint32_t size)
{
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated, REDUCE_SCATTER_LOOP_COUNT);

    std::string loopType = ops_hccl::GetReduceTypeStr(ctx.dataType, ctx.reduceOp);
    if (ctx.IsLoopEntityRegistered(loopType)) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity(loopType);
    auto &loops = ctx.loopMap[loopType];

    uint32_t expansionNum = GetReduceExpansionNum(ctx.reduceOp, ctx.dataType, ctx.outputDataType);
    uint32_t usedBufNum   = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < LOOP_NUM; index++) {
        ctx.loopScratch[index].resize(size);

        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event loopEvt = ctx.moRes.completedEvent[index];

        loops.body[index].reset(new ccu::Func(
            [&ctx, index, bufBase, loopEvt, size, expansionNum, usedBufNum]() {
                for (uint32_t i = 0; i < size; i++) {
                    ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], ctx.loopScratch[index][i], ctx.loopLen[index], loopEvt, 1 << i);
                }
                ccu::EventWait(loopEvt, (1 << size) - 1);

                if (size > 1) {
                    ccu::LocalReduce(&ctx.moRes.ccuBuf[bufBase], size, ctx.dataType, ctx.outputDataType, ctx.reduceOp, ctx.loopLen[index], loopEvt, 1);
                    ccu::EventWait(loopEvt, 1);
                }

                ccu::LocalCopy(ctx.loopDst[index], ctx.moRes.ccuBuf[bufBase], ctx.loopLenExp[index], loopEvt, 1);
                ccu::EventWait(loopEvt, 1);
        }));

        loops.loops[index].reset(
            new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }

    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(ReduceScatterMesh1D2DieMem2MemContext &ctx, ccu::LocalAddr outDstOrg,
        std::vector<ccu::LocalAddr> &scratchOrg)
{
    const uint32_t size = scratchOrg.size();

    ccu::LocalAddr dst;
    dst.addr  = outDstOrg.addr;
    dst.token = outDstOrg.token;

    std::vector<ccu::LocalAddr> scratch;
    for (uint32_t idx = 0; idx < size; idx++) {
        ccu::LocalAddr scratchAddr;
        scratchAddr.addr = scratchOrg[idx].addr;
        scratchAddr.token = scratchOrg[idx].token;
        scratch.push_back(scratchAddr);
    }

    CCU_CHK_RET(CreateReduceLoop(ctx, size));
    auto &loops = ctx.loopMap[ops_hccl::GetReduceTypeStr(ctx.dataType, ctx.reduceOp)];

    uint32_t expansionNum = GetReduceExpansionNum(ctx.reduceOp, ctx.dataType, ctx.outputDataType);
    ccu::Variable sliceSizeExpansion;
    ccu::Variable tmp;
    if (expansionNum != 1) {
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    // m部分
    CCU_IF(ctx.sliceGoSize.loopParam != 0)
    {
        ccu::Variable loopParam;
        ccu::Variable sliceSize;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + ctx.sliceGoSize.loopParam;
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop0的外部LocalAddr和Variable
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr  = dst.addr;
        ctx.loopSrc[0].token = dst.token;
        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0]       = sliceSize;
        ctx.loopLenExp[0]    = sliceSizeExpansion;

        ccu::Variable paraCfg;
        ccu::Variable offsetCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
        ccu::LoopGroup group(paraCfg, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    CCU_IF(ctx.sliceGoSize.parallelParam != 0)
    {
        // p部分，加m的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.sliceGoSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.sliceGoSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + ctx.sliceGoSize.residual;
        }

        // 绑定loop0参数 (p部分)
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[0][i].addr = scratch[i].addr;
            ctx.loopScratch[0][i].token = scratch[i].token;
        }
        ctx.loopSrc[0].addr  = dst.addr;
        ctx.loopSrc[0].token = dst.token;
        ctx.loopDst[0].addr  = dst.addr;
        ctx.loopDst[0].token = dst.token;
        ctx.loopLen[0]    = ctx.sliceGoSize.residual;
        ctx.loopLenExp[0] = sliceSizeExpansion;

        ccu::Variable sliceSize;
        // n部分，再加p的偏移
        for (uint32_t i = 0; i < size; i++) {
            scratch[i].addr += ctx.sliceGoSize.residual;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr += ctx.sliceGoSize.residual;
        }
        sliceSize          = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        // 绑定loop1参数 (n部分)
        for (uint32_t i = 0; i < size; i++) {
            ctx.loopScratch[1][i].addr = scratch[i].addr;
            ctx.loopScratch[1][i].token = scratch[i].token;
        }
        ctx.loopSrc[1].addr  = dst.addr;
        ctx.loopSrc[1].token = dst.token;
        ctx.loopDst[1].addr  = dst.addr;
        ctx.loopDst[1].token = dst.token;
        ctx.loopLen[1]    = sliceSize;
        ctx.loopLenExp[1] = sliceSizeExpansion;

        ccu::Variable loopCfg0;
        ccu::Variable loopCfg1;
        ccu::Variable offsetCfg;
        loopCfg0 = GetLoopParam(0, 0, 1);
        loopCfg1 = GetLoopParam(0, 0, 1);
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(ctx.sliceGoSize.parallelParam, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    return CCU_SUCCESS;
}

static CcuResult DoReduceScatter(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    ccu::LocalAddr myOutput;
    myOutput.addr   = ctx.output;
    myOutput.token  = ctx.token[ctx.myRankIdx];

    if (ctx.rankId < ctx.rankSize) {
        if (!ctx.isReduceToOutput) {
            myOutput.addr = ctx.outputTmp.addr;
        }
    } else {
        if (ctx.isReduceToOutput) {
            myOutput.addr = ctx.outputTmp.addr;
        }
    }

    uint32_t channelId = 0;
    for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        if (ctx.subRankGroup[rankIdx] == ctx.rankId) {
            ccu::EventRecord(ctx.event, 1 << rankIdx);
            ctx.scratchMem[rankIdx].addr = ctx.input[ctx.myRankIdx];
            ctx.scratchMem[rankIdx].addr += ctx.currentRankSliceInputOffset;
            ctx.scratchMem[rankIdx].token = ctx.token[ctx.myRankIdx];
        } else {
            CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelId], ctx.scratchMem[rankIdx], ctx.remoteInput[rankIdx], ctx.sliceSize, ctx.event, 1 << rankIdx));
            channelId++;
        }
    }

    // 等读完所有对端
    ccu::EventWait(ctx.event, (1 << ctx.rankSize) - 1);

    // 做reduce
    CCU_CHK_RET(ReduceLoopGroup(ctx, myOutput, ctx.scratchMem));
    return CCU_SUCCESS;
}

static CcuResult RmtReduce(ReduceScatterMesh1D2DieMem2MemContext &ctx)
{
    ccu::Variable scratchOffset;
    std::vector<ccu::Variable> scratchOffsetVec;
    scratchOffset = 0;

    for (uint32_t gRankIdx = 0; gRankIdx < ctx.gRankSize; gRankIdx++) {
        ccu::Variable scratchOffTmp;
        scratchOffTmp = scratchOffset;
        scratchOffsetVec.push_back(scratchOffTmp);
        scratchOffset = scratchOffset + ctx.sliceSize;
    }

    ctx.outputTmp.addr = ctx.scratch[ctx.myRankIdx];
    ctx.outputTmp.addr += scratchOffsetVec[ctx.gRankSize / 2];
    ctx.outputTmp.token = ctx.token[ctx.myRankIdx];

    for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        if (ctx.subRankGroup[rankIdx] != ctx.rankId) {
            ctx.remoteInput[rankIdx].addr = ctx.input[rankIdx];
            ctx.remoteInput[rankIdx].addr += ctx.currentRankSliceInputOffset;
            ctx.remoteInput[rankIdx].token = ctx.token[rankIdx];
        }
        ctx.scratchMem[rankIdx].addr = ctx.scratch[ctx.myRankIdx];
        ctx.scratchMem[rankIdx].addr += scratchOffsetVec[ctx.subRankGroup[rankIdx]];
        ctx.scratchMem[rankIdx].token = ctx.token[ctx.myRankIdx];
    }

    for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        if (ctx.subRankGroup[rankIdx] == ctx.rankId) {
            ctx.myInput.addr = ctx.input[rankIdx];
            ctx.myInput.addr += ctx.currentRankSliceInputOffset;
            ctx.myInput.token = ctx.token[rankIdx];
        }
    }

    ccu::Variable repeatNumAdd;
    ctx.flag = 0;
    repeatNumAdd  = 1;
    CCU_WHILE(ctx.repeatNum != UINT64_MAX) {
        ctx.repeatNum += repeatNumAdd;
        CCU_IF(ctx.flag == 1) {
            for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
                if (ctx.subRankGroup[rankIdx] == ctx.rankId) {
                    ctx.myInput.addr += ctx.inputRepeatStride;
                } else {
                    ctx.remoteInput[rankIdx].addr += ctx.inputRepeatStride;
                }
            }
            ctx.output += ctx.outputRepeatStride;
        }
        CCU_IF(ctx.sliceSize != 0)
        {
            CCU_CHK_RET(DoReduceScatter(ctx));
        }
        ctx.flag = 1;
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMesh1D2DieMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterMesh1D2DieMem2Mem *>(arg);

    ReduceScatterMesh1D2DieMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceScatterMesh1D2DieMem2Mem] Algorithm start");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(RmtReduce(ctx));
    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelReduceScatterMesh1D2DieMem2Mem] Algorithm end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl