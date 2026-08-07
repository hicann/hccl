/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID = 1;
constexpr uint16_t TOKEN_XN_ID = 2;
constexpr uint16_t POST_SYNC_ID = 3;
constexpr uint16_t CKE_IDX_0 = 0;

static CcuResult
ParseKernelArg(ScatterOmniPipeMesh1DMem2MemContext& ctx, CcuKernelArgScatterOmniPipeMesh1DMem2Mem* kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.ifRealRoot = kernelArg->ifRealRoot;
    ctx.myrealrank = kernelArg->myrealrank;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ScatterOmniPipeMesh1DMem2MemContext& ctx)
{
    uint16_t channelIdx = 0;
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuScatterOmniPipeMesh1DMem2MemKernel] channels is empty!");
        return CCU_E_INTERNAL;
    }

    ctx.output.resize(ctx.rankSize);
    ctx.token.resize(ctx.rankSize);
    for (uint64_t peerId = 0; peerId < ctx.rankSize; peerId++) {
        if (peerId != ctx.rankId) {
            if (channelIdx >= ctx.arg->channelCount) {
                HCCL_ERROR("[CcuScatterOmniPipeMesh1DMem2MemKernel] channelIdx out of range!");
                return CCU_E_INTERNAL;
            }
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.inputMem.resize(ctx.rankSize);
    ctx.outputMem.resize(ctx.rankSize);
    ctx.outputOmniSliceStrideVec.resize(ctx.rankSize);
    ctx.inputOmniSliceStrideVec.resize(ctx.rankSize);
    ctx.inputOmniSliceSizeVec.resize(ctx.rankSize);
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ScatterOmniPipeMesh1DMem2MemContext& ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isStepOne, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isLastStep, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ifNewRoot, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isFirstPiece, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isLastPiece, argId++));
    for (uint64_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceStrideVec[i], argId++));
    }
    for (uint64_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.outputOmniSliceStrideVec[i], argId++));
    }
    for (uint64_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceSizeVec[i], argId++));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(ScatterOmniPipeMesh1DMem2MemContext& ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult PreSync(ScatterOmniPipeMesh1DMem2MemContext& ctx)
{
    HCCL_DEBUG(
        "[CcuScatterOmniPipeMesh1DMem2MemKernel] PreSync realRank[%u] ctx.arg->channelCount[%u]", ctx.myrealrank,
        ctx.arg->channelCount);

    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[i], ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[i], ctx.token[ctx.arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatter(ScatterOmniPipeMesh1DMem2MemContext& ctx)
{
    uint32_t channelId = 0;

    for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
        uint64_t mask = 1ULL << rankIdx;
        ctx.sliceSize = ctx.inputOmniSliceSizeVec[rankIdx];

        if (rankIdx == ctx.rankId) {
            CCU_CHK_RET(ccu::EventRecord(ctx.event, mask));
        } else {
            CCU_IF(ctx.sliceSize != 0)
            {
                CCU_CHK_RET(ccu::Write(
                    ctx.arg->channels[channelId], ctx.outputMem[rankIdx], ctx.inputMem[rankIdx], ctx.sliceSize,
                    ctx.event, mask));
            }
            CCU_IF(ctx.sliceSize == 0) { CCU_CHK_RET(ccu::EventRecord(ctx.event, mask)); }
            channelId++;
        }
    }

    CCU_CHK_RET(ccu::EventWait(ctx.event, (1ULL << ctx.rankSize) - 1));
    return CCU_SUCCESS;
}

static CcuResult DoRepeatScatter(ScatterOmniPipeMesh1DMem2MemContext& ctx)
{
    for (uint64_t curId = 0; curId < ctx.rankSize; curId++) {
        if (curId == ctx.rankId) {
            continue;
        }
        ctx.inputMem[curId].token = ctx.token[curId];
        ctx.inputMem[curId].addr = ctx.input;
        ctx.inputMem[curId].addr += ctx.inputOmniSliceStrideVec[curId];

        ctx.outputMem[curId].token = ctx.token[curId];
        ctx.outputMem[curId].addr = ctx.output[curId];
        ctx.outputMem[curId].addr += ctx.outputOmniSliceStrideVec[curId];
    }

    CCU_IF(ctx.ifNewRoot == true) { CCU_CHK_RET(DoScatter(ctx)); }
    return CCU_SUCCESS;
}

CcuResult CcuScatterOmniPipeMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<CcuKernelArgScatterOmniPipeMesh1DMem2Mem*>(arg);
    ScatterOmniPipeMesh1DMem2MemContext ctx;

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_IF(ctx.isFirstPiece == true) { CCU_CHK_RET(PreSync(ctx)); }

    CCU_CHK_RET(DoRepeatScatter(ctx));
    CCU_IF(ctx.isLastPiece == true) { CCU_CHK_RET(PostSync(ctx)); }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
