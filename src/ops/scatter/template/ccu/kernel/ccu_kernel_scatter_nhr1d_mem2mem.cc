/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t SCRATCH_XN_ID       = 1;
constexpr uint16_t TOKEN_XN_ID         = 2;
constexpr uint16_t POST_SYNC_ID        = 3;
constexpr uint16_t STEP_POST_SYNC_ID   = 4;
constexpr uint16_t CKE_IDX_0           = 0;
constexpr uint16_t RANK_NUM_PER_CKE    = 16;

static CcuResult ParseKernelArg(ScatterNHR1DContext &ctx, CcuKernelArgScatterNHRMem2Mem1D *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.axisId = kernelArg->axisId;
    ctx.axisSize = kernelArg->axisSize;
    ctx.stepInfoVector = kernelArg->stepInfoVector;
    ctx.rank2ChannelIdx = kernelArg->rank2ChannelIdx;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;

    ctx.localSize = static_cast<uint32_t>(ctx.rank2ChannelIdx.size());
    ctx.myRankIdx = ctx.localSize;
    ctx.signalNum = static_cast<uint32_t>((ctx.rankSize + RANK_NUM_PER_CKE - 1) / RANK_NUM_PER_CKE);
    return CCU_SUCCESS;
}

static CcuResult InitResource(ScatterNHR1DContext &ctx)
{
    // remote ranks scratch/token
    ctx.scratch.clear();
    ctx.token.clear();
    ctx.scratch.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);

    for (uint64_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        if (ctx.arg->channelCount <= channelIdx) {
            HCCL_ERROR("[CcuScatterNHR1DMem2MemKernel] arg->channels size[%llu] < localSize[%u]", ctx.arg->channelCount, ctx.localSize);
            return CCU_E_INTERNAL;
        }
		ctx.scratch[channelIdx] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], SCRATCH_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ScatterNHR1DContext &ctx)
{
	uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.curScratchStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isOutputScratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0TailSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1TailSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isSliceSizeZero, argId++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(ScatterNHR1DContext &ctx)
{
    uint32_t allBit = (1 << SCRATCH_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.scratch[ctx.myRankIdx], SCRATCH_XN_ID, CKE_IDX_0, 1 << SCRATCH_XN_ID);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.myRankIdx], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit);
    }
    return CCU_SUCCESS;
}

static void DoLocalCopy(ccu::LocalAddr &dst, ccu::LocalAddr &src, ccu::Variable &sliceSize, ccu::Event &event,
                          uint16_t mask)
{
    CCU_IF(sliceSize == 0)
    {
        ccu::EventRecord(event, mask);
    }
    CCU_IF(sliceSize != 0)
    {
        ccu::LocalCopy(dst, src, sliceSize, event, mask);
    }
}

static void DoWrite(ChannelHandle sendChannel, ccu::RemoteAddr &dst, ccu::LocalAddr &src,
                      ccu::Variable &sliceSize, ccu::Event &event, uint16_t mask)
{
    CCU_IF(sliceSize == 0)
    {
        ccu::EventRecord(event, mask);
    }
    CCU_IF(sliceSize != 0)
    {
        ccu::Write(sendChannel, dst, src, sliceSize, event, mask);
    }
}

static CcuResult DoSendRecvSlice(ScatterNHR1DContext &ctx, const u32 &toRank, ccu::LocalAddr &src, ccu::RemoteAddr &dst,
                                 u32 signalIndex, bool isLastSlice)
{
    if (ctx.rank2ChannelIdx.count(toRank) == 0) {
        return CCU_SUCCESS;
    }
    u32 toRankIdx = ctx.rank2ChannelIdx[toRank];
    if (toRankIdx >= ctx.arg->channelCount) {
        return CCU_SUCCESS;
    }
    ChannelHandle sendChannel = ctx.arg->channels[toRankIdx];

    ccu::Variable repeatNumAdd;
    repeatNumAdd = 1;
    ctx.repeatTimeFlag = 0;
    ctx.repeatNumVarTemp = ctx.repeatNumVar;

    CCU_WHILE(ctx.repeatNumVarTemp != UINT64_MAX)
    {
        ctx.repeatNumVarTemp += repeatNumAdd;
        CCU_IF(ctx.repeatTimeFlag == 1)
        {
            if (ctx.rankId == ctx.rootId) {
                src.addr += ctx.inputRepeatStride;
            } else {
                src.addr += ctx.outputRepeatStride;
            }
            dst.addr += ctx.outputRepeatStride;
        }
        CCU_IF(ctx.repeatTimeFlag == 0)
        {
            if(ctx.axisId == 1)
            {
                if (isLastSlice) {
                    src.addr += ctx.die0TailSize;
                    dst.addr += ctx.die0TailSize;
                } else {
                    src.addr += ctx.die0Size;
                    dst.addr += ctx.die0Size;
                }
            }
        }

        if (isLastSlice) {
            ctx.curSliceSize = (ctx.axisId == 0) ? ctx.die0TailSize : ctx.die1TailSize;
        } else {
            ctx.curSliceSize = (ctx.axisId == 0) ? ctx.die0Size : ctx.die1Size;
        }
		uint16_t mask = 1 << signalIndex;
        DoWrite(sendChannel, dst, src, ctx.curSliceSize, ctx.event, mask);
        ccu::EventWait(ctx.event, mask);
        ctx.repeatTimeFlag = 1;
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterNHRSingleStep(ScatterNHR1DContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const auto &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;

    if (!recvSliceIdxList.empty()) {
        if (ctx.rank2ChannelIdx.count(nhrStepInfo.fromRank) != 0) {
            u32 fromRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.fromRank];
            if (fromRankIdx < ctx.arg->channelCount) {
                ChannelHandle recvChannel = ctx.arg->channels[fromRankIdx];
                ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
            }
        }
    }

    if (!sendSliceIdxList.empty()) {
        if (ctx.rank2ChannelIdx.count(nhrStepInfo.toRank) == 0) {
            return CCU_E_INTERNAL;
        }
        u32 toRankIdx = ctx.rank2ChannelIdx[nhrStepInfo.toRank];
        if (toRankIdx >= ctx.arg->channelCount) {
            return CCU_E_INTERNAL;
        }
        ChannelHandle sendChannel = ctx.arg->channels[toRankIdx];

        for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
            u32 sendSliceIdx = sendSliceIdxList[i];
            bool isLastSlice = (sendSliceIdx == ctx.rankSize - 1);

            if (i != 0 && i % RANK_NUM_PER_CKE == 0) {
                ccu::EventWait(ctx.event, (1 << RANK_NUM_PER_CKE) - 1);
            }

            if (ctx.rankId == ctx.rootId) {
                ctx.srcMem.addr = ctx.input;
                ctx.srcMem.addr += ctx.inputOffset[sendSliceIdx];
            } else {
                ctx.srcMem.addr = ctx.scratch[ctx.myRankIdx];
                ctx.srcMem.addr += ctx.scratchOffset[sendSliceIdx];
            }
            ctx.srcMem.token = ctx.token[ctx.myRankIdx];

            ctx.dstRemoteMem.token = ctx.token[toRankIdx];
            ctx.dstRemoteMem.addr = ctx.scratch[toRankIdx];
            ctx.dstRemoteMem.addr += ctx.scratchOffset[sendSliceIdx];

            CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.dstRemoteMem,
                                        i % RANK_NUM_PER_CKE, isLastSlice));
        }

        ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterNHR(ScatterNHR1DContext &ctx)
{
    ctx.curInputOffset = 0;
    ctx.curScratchOffset = 0;

    ctx.inputOffset.resize(ctx.rankSize);
    ctx.scratchOffset.resize(ctx.rankSize);
    for (u64 i = 0; i < ctx.rankSize; i++) {
        ctx.inputOffset[i] = ctx.curInputOffset;
        ctx.curInputOffset += ctx.inputSliceStride;
    }
    for (u64 i = 0; i < ctx.rankSize; i++) {
        ctx.scratchOffset[i] = ctx.curScratchOffset;
        ctx.curScratchOffset += ctx.curScratchStride;
    }

    for (auto &step : ctx.stepInfoVector) {
        CCU_CHK_RET(DoScatterNHRSingleStep(ctx, step));
    }

    // final local copy to output
    if (ctx.rankId == ctx.rootId) {
        ctx.srcMem.addr = ctx.input;
        ctx.srcMem.addr += ctx.inputOffset[ctx.rankId];
    } else {
        ctx.srcMem.addr = ctx.scratch[ctx.myRankIdx];
        ctx.srcMem.addr += ctx.scratchOffset[ctx.rankId];
    }
    ctx.dstMem.addr = ctx.output;
    ctx.srcMem.token = ctx.token[ctx.myRankIdx];
    ctx.dstMem.token = ctx.token[ctx.myRankIdx];

    ccu::Variable repeatNumAdd;
    repeatNumAdd = 1;
    ctx.repeatTimeFlag = 0;

    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX)
    {
        ctx.repeatNumVar += repeatNumAdd;
        CCU_IF(ctx.repeatTimeFlag != 0)
        {
            if (ctx.rankId == ctx.rootId) {
                ctx.srcMem.addr += ctx.inputRepeatStride;
            } else {
                ctx.srcMem.addr += ctx.outputRepeatStride;
            }
            ctx.dstMem.addr += ctx.outputRepeatStride;
        }
        CCU_IF(ctx.repeatTimeFlag == 0)
        {
            if(ctx.axisId == 1)
            {
                if (ctx.rankId != ctx.rankSize - 1) {
                    ctx.srcMem.addr += ctx.die0Size;
                    ctx.dstMem.addr += ctx.die0Size;
                } else {
                    ctx.srcMem.addr += ctx.die0TailSize;
                    ctx.dstMem.addr += ctx.die0TailSize;
                }
            }
        }

        if (ctx.rankId != ctx.rankSize - 1) {
            ctx.curSliceSize = (ctx.axisId == 0) ? ctx.die0Size : ctx.die1Size;
        } else {
            ctx.curSliceSize = (ctx.axisId == 0) ? ctx.die0TailSize : ctx.die1TailSize;
        }
		uint16_t mask = 1 << ctx.rankId;
        CCU_IF(ctx.isOutputScratch == 1)
        {
            CCU_IF(ctx.outputSliceStride == 0)
            {
                // special-case: rootId != 0 && rankId == 0
                if (ctx.rootId != 0 && ctx.rankId == 0) {
                    ccu::EventRecord(ctx.event, mask);
                } else {
                    CCU_IF(ctx.isInputOutputEqual != 1)
                    {
                        if (ctx.rankId == ctx.rootId) {
                            DoLocalCopy(ctx.dstMem, ctx.srcMem, ctx.curSliceSize, ctx.event, mask);
                        } else {
                            CCU_IF(ctx.isSliceSizeZero != 1)
                            {
                                DoLocalCopy(ctx.dstMem, ctx.srcMem, ctx.curSliceSize, ctx.event, mask);
                            }
                            CCU_IF(ctx.isSliceSizeZero == 1)
                            {
                                ccu::EventRecord(ctx.event, mask);
                            }
                        }
                    }
                    CCU_IF(ctx.isInputOutputEqual == 1)
                    {
                        ccu::EventRecord(ctx.event, mask);
                    }
                }
            }
            CCU_IF(ctx.outputSliceStride != 0)
            {
                if (ctx.rankId == ctx.rootId) {
                    CCU_IF(ctx.isInputOutputEqual != 1)
                    {
                        for (uint32_t i = 0; i < ctx.rootId; i++) {
                            ctx.dstMem.addr = ctx.dstMem.addr + ctx.outputSliceStride;
                        }
                        DoLocalCopy(ctx.dstMem, ctx.srcMem, ctx.curSliceSize, ctx.event, mask);
                    }
                    CCU_IF(ctx.isInputOutputEqual == 1)
                    {
                        ccu::EventRecord(ctx.event, mask);
                    }
                } else {
                    ccu::EventRecord(ctx.event, mask);
                }
            }
        }
        CCU_IF(ctx.isOutputScratch != 1)
        {
            DoLocalCopy(ctx.dstMem, ctx.srcMem, ctx.curSliceSize, ctx.event, mask);
        }
        ccu::EventWait(ctx.event, mask);
        ctx.repeatTimeFlag = 1;
    }

    return CCU_SUCCESS;
}

static CcuResult PostSync(ScatterNHR1DContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatterNHRMem2Mem1D *>(arg);
    ScatterNHR1DContext ctx;

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoScatterNHR(ctx));
    CCU_CHK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
