/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static CcuResult ParseKernelArg(AllGatherMesh1DMem2MemContext &ctx, CcuKernelArgAllGatherMesh1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] channels.size: [%u]", arg->channelCount);

    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);

    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    
    const uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    ctx.events.resize(AG_UNROLL_NUM * eventNum);

    ctx.resourceAllocated = false;

    ctx.constVar1 = 1;
    ctx.repeatTimeflag = 0;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.tmpRepeatNum, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++)); 
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));

    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllGatherMesh1DMem2MemContext &ctx)
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

static CcuResult InitAllGatherAddr(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.src.addr = ctx.input;
    ctx.src.addr += ctx.currentRankSliceInputOffset;
    ctx.src.token = ctx.token[arg->rankId];

    ctx.src_loccopy.addr = ctx.input;
    ctx.src_loccopy.addr += ctx.currentRankSliceInputOffset;
    ctx.src_loccopy.token = ctx.token[arg->rankId];

    ctx.dst.resize(arg->rankSize);
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            ctx.localDst.addr = ctx.output[arg->rankId];
            ctx.localDst.addr += ctx.currentRankSliceOutputOffset;
            ctx.localDst.token = ctx.token[arg->rankId];
        } else {
            ctx.dst[rankIdx].addr = ctx.output[rankIdx];
            ctx.dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            ctx.dst[rankIdx].token = ctx.token[rankIdx];
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherWrite(AllGatherMesh1DMem2MemContext &ctx, const ccu::LocalAddr &src,
    const std::vector<ccu::RemoteAddr> &dst, const ccu::Variable &sliceSize, uint32_t unrollIdx)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;
    uint32_t numEventsPerIter = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;

    for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        uint32_t eventIdx = unrollIdx * numEventsPerIter + rankIdx / BIT_NUM_PER_CKE;
        uint16_t rankMask = 1 << (rankIdx % BIT_NUM_PER_CKE);
        if (rankIdx == arg->rankId) {
            CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask));
        } else {
            CCU_CHK_RET(ccu::Write(arg->channels[channelId], dst[rankIdx],
                src, sliceSize, ctx.events[eventIdx], rankMask));
            channelId++;
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherWait(AllGatherMesh1DMem2MemContext &ctx, uint32_t unrollIdx)
{
    const auto *arg = ctx.arg;
    uint32_t numEventsPerIter = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;

    for (uint32_t i = 0; i < numEventsPerIter; i++) {
        uint32_t eventIdx = unrollIdx * numEventsPerIter + i;
        uint16_t eventMask;
        if (i == numEventsPerIter - 1) {
            if (arg->rankSize % BIT_NUM_PER_CKE == 0) {
                eventMask = (1 << BIT_NUM_PER_CKE) - 1;
            } else {
                eventMask = (1 << (arg->rankSize % BIT_NUM_PER_CKE)) - 1;
            }
        } else {
            eventMask = (1 << BIT_NUM_PER_CKE) - 1;
        }
        CCU_CHK_RET(ccu::EventWait(ctx.events[eventIdx], eventMask));
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherGroupCopy(AllGatherMesh1DMem2MemContext &ctx)
{
    CCU_IF(ctx.isInputOutputEqual == 0)
    {
        CCU_IF(ctx.groupCopyRepeatNum != UINT64_MAX)
        {
            ctx.repeatTimeflag = 0;
            CCU_WHILE(ctx.groupCopyRepeatNum != UINT64_MAX)
            {
                ctx.groupCopyRepeatNum += ctx.constVar1;
                CCU_IF(ctx.repeatTimeflag != 0)
                {
                    ctx.localDst.addr += ctx.outputRepeatStride;
                    ctx.src_loccopy.addr += ctx.inputRepeatStride;
                }
                CCU_CHK_RET(GroupCopy(ctx, ctx.localDst, ctx.src_loccopy, ctx.goSize));
                ctx.repeatTimeflag = 1;
            }
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGather(AllGatherMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    CCU_CHK_RET(InitAllGatherAddr(ctx));
    ctx.waitRepeatNum = ctx.tmpRepeatNum;
    ctx.groupCopyRepeatNum = ctx.tmpRepeatNum;

    // Phase 1: 先下发所有WriteNb（非阻塞，event错开），不包含GroupCopy
    CCU_IF(ctx.tmpRepeatNum != UINT64_MAX)
    {
        ctx.tmpRepeatNum += ctx.constVar1;
        CCU_CHK_RET(DoAllGatherWrite(ctx, ctx.src, ctx.dst, ctx.normalSliceSize, 0));
    }

    for (uint32_t i = 1; i < AG_UNROLL_NUM; i++) {
        CCU_IF(ctx.tmpRepeatNum != UINT64_MAX)
        {
            ctx.tmpRepeatNum += ctx.constVar1;
            ctx.src.addr += ctx.inputRepeatStride;
            for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
                if (rankIdx != arg->rankId) {
                    ctx.dst[rankIdx].addr += ctx.outputRepeatStride;
                }
            }
            CCU_CHK_RET(DoAllGatherWrite(ctx, ctx.src, ctx.dst, ctx.normalSliceSize, i));
        }
    }

    // Phase 2: GroupCopy使用CCU_WHILE
    CCU_CHK_RET(DoAllGatherGroupCopy(ctx));

    // Phase 3: 批量WaitEvent
    for (uint32_t i = 0; i < AG_UNROLL_NUM; i++) {
        CCU_IF(ctx.waitRepeatNum != UINT64_MAX)
        {
            ctx.waitRepeatNum += ctx.constVar1;
            CCU_CHK_RET(DoAllGatherWait(ctx, i));
        }
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherMesh1DMem2Mem *>(arg);

    AllGatherMesh1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllGatherMesh1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    ctx.sliceSize = (kernelArg->rankId == (kernelArg->rankSize - 1)) ? ctx.lastSliceSize : ctx.normalSliceSize;
    // sliceSize == 0时不需要执行AllGather，只需前后同步
    CCU_IF(ctx.sliceSize != 0) {
        CCU_CHK_RET(DoRepeatAllGather(ctx));
    }

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllGatherMesh1DMem2Mem end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
