/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID = 0;
constexpr uint16_t TOKEN_XN_ID = 1;
constexpr uint16_t POST_SYNC_ID = 2;
constexpr uint16_t STEP_SYNC_ID = 3;
constexpr uint16_t CKE_IDX_0 = 0;

static CcuResult
ParseKernelArg(ScatterOmniPipeNHR1DMem2MemContext& ctx, CcuKernelArgScatterOmniPipeNHR1DMem2Mem* kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.ifRealRoot = kernelArg->ifRealRoot;
    ctx.myrealrank = kernelArg->myrealrank;
    ctx.stepInfoVector = kernelArg->stepInfoVector;
    ctx.rank2ChannelIdx = kernelArg->rank2ChannelIdx;
    ctx.localSize = static_cast<uint32_t>(ctx.rank2ChannelIdx.size());
    ctx.myRankIdx = ctx.localSize;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ScatterOmniPipeNHR1DMem2MemContext& ctx)
{
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuScatterOmniPipeNHR1DMem2MemKernel] channels is empty!");
        return CCU_E_INTERNAL;
    }

    ctx.output.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);
    for (uint64_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        if (ctx.arg->channelCount <= channelIdx) {
            HCCL_ERROR(
                "[CcuScatterOmniPipeNHR1DMem2MemKernel] channels size[%u] < localSize[%u]", ctx.arg->channelCount,
                ctx.localSize);
            return CCU_E_INTERNAL;
        }
        ctx.output[channelIdx] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }

    ctx.outputOmniSliceStrideVec.resize(ctx.rankSize);
    ctx.inputOmniSliceStrideVec.resize(ctx.rankSize);
    ctx.inputOmniSliceSizeVec.resize(ctx.rankSize);
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ScatterOmniPipeNHR1DMem2MemContext& ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isStepOne, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isLastStep, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.ifNewRoot, argId++));
    for (uint64_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.outputOmniSliceStrideVec[i], argId++));
    }
    for (uint64_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceStrideVec[i], argId++));
    }
    for (uint64_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceSizeVec[i], argId++));
    }
    return CCU_SUCCESS;
}

static CcuResult PreSync(ScatterOmniPipeNHR1DMem2MemContext& ctx)
{
    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[i], ctx.output[ctx.myRankIdx], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[i], ctx.token[ctx.myRankIdx], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(ScatterOmniPipeNHR1DMem2MemContext& ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterOmniPipeNHRSend(ScatterOmniPipeNHR1DMem2MemContext& ctx, const NHRStepInfo& nhrStepInfo)
{
    if (ctx.rank2ChannelIdx.count(nhrStepInfo.toRank) == 0) {
        return CCU_E_INTERNAL;
    }
    u32 toRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.toRank];
    if (toRankIdx >= ctx.arg->channelCount) {
        return CCU_E_INTERNAL;
    }

    ChannelHandle sendChannel = ctx.arg->channels[toRankIdx];
    ccu::LocalAddr src;
    ccu::RemoteAddr dst;
    src.token = ctx.token[ctx.myRankIdx];
    dst.token = ctx.token[toRankIdx];

    for (u32 i = 0; i < nhrStepInfo.txSliceIdxs.size(); i++) {
        u32 sendSliceIdx = nhrStepInfo.txSliceIdxs[i];
        ctx.sliceSize = ctx.inputOmniSliceSizeVec[sendSliceIdx];
        if (ctx.ifRealRoot) {
            src.addr = ctx.input;
            src.addr += ctx.inputOmniSliceStrideVec[sendSliceIdx];
        } else {
            src.addr = ctx.output[ctx.myRankIdx];
            src.addr += ctx.outputOmniSliceStrideVec[sendSliceIdx];
        }
        dst.addr = ctx.output[toRankIdx];
        dst.addr += ctx.outputOmniSliceStrideVec[sendSliceIdx];
        uint16_t mask = 1 << i;
        CCU_IF(ctx.sliceSize != 0) { CCU_CHK_RET(ccu::Write(sendChannel, dst, src, ctx.sliceSize, ctx.event, mask)); }
        CCU_IF(ctx.sliceSize == 0) { CCU_CHK_RET(ccu::EventRecord(ctx.event, mask)); }
    }
    uint16_t sendBit = (1 << nhrStepInfo.txSliceIdxs.size()) - 1;
    CCU_CHK_RET(ccu::EventWait(ctx.event, sendBit));
    CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[toRankIdx], CKE_IDX_0, 1 << STEP_SYNC_ID));
    return CCU_SUCCESS;
}

static CcuResult DoScatterOmniPipeNHRRecv(ScatterOmniPipeNHR1DMem2MemContext& ctx, const NHRStepInfo& nhrStepInfo)
{
    if (ctx.rank2ChannelIdx.count(nhrStepInfo.fromRank) != 0) {
        u32 fromRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.fromRank];
        if (fromRankIdx < ctx.arg->channelCount) {
            ChannelHandle recvChannel = ctx.arg->channels[fromRankIdx];
            CCU_CHK_RET(ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_SYNC_ID));
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterOmniPipeNHRSingleStep(ScatterOmniPipeNHR1DMem2MemContext& ctx, const NHRStepInfo& nhrStepInfo)
{
    if (!nhrStepInfo.txSliceIdxs.empty()) {
        CCU_CHK_RET(DoScatterOmniPipeNHRSend(ctx, nhrStepInfo));
    }
    if (!nhrStepInfo.rxSliceIdxs.empty()) {
        CCU_CHK_RET(DoScatterOmniPipeNHRRecv(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterOmniPipeNHR(ScatterOmniPipeNHR1DMem2MemContext& ctx)
{
    for (auto& step : ctx.stepInfoVector) {
        CCU_CHK_RET(DoScatterOmniPipeNHRSingleStep(ctx, step));
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterOmniPipeNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<CcuKernelArgScatterOmniPipeNHR1DMem2Mem*>(arg);
    ScatterOmniPipeNHR1DMem2MemContext ctx;

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoScatterOmniPipeNHR(ctx));
    CCU_CHK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
