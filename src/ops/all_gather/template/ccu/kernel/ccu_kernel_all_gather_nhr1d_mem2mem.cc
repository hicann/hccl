/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_nhr1d_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID    = 1;
constexpr uint16_t TOKEN_XN_ID     = 2;
constexpr uint16_t POST_SYNC_ID    = 3;
constexpr uint16_t STEP_PRE_SYNC_ID = 4;
constexpr uint16_t STEP_POST_SYNC_ID = 5;
constexpr uint16_t CKE_IDX_0        = 0;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static CcuResult ParseKernelArg(AllGatherNHR1DMem2MemContext &ctx, CcuKernelArgAllGatherNHR1D *kernelArg)
{
    ctx.arg          = kernelArg;
    ctx.localSize    = kernelArg->rank2ChannelIdx.size();
    ctx.myRankIdx    = kernelArg->rank2ChannelIdx.size();
    HCCL_DEBUG(
        "[CcuKernelAllGatherNHR1DMem2Mem] Init, KernelArgs are mySubCommRankId[%u], axisId[%u], axisSize[%u], stepInfoVector.size[%u], myRankIdx[%u] localSize[%u]",
        kernelArg->mySubCommRankId, kernelArg->axisId, kernelArg->axisSize, kernelArg->stepInfoVector.size(), ctx.myRankIdx, ctx.localSize);
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherNHR1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] channels.size: [%u]", arg->channelCount);

    ctx.output.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; channelIdx++) {
        HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] mySubCommRankId[%u], channelId[%u] localSize[%u]",
            arg->mySubCommRankId, channelIdx, arg->channelCount);
        ctx.output[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }

    ctx.outputSliceOffset.resize(arg->dimSize);
    ctx.myrankInputSliceOffset = 0;
    ctx.repeatTimeflag = 0;
    ctx.constVar1 = 1;
    ctx.resourceAllocated = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0LastSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1LastSize, argId++));

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMem2Mem] LoadArgs run finished");
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PreSync start");
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[ctx.myRankIdx],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PreSync end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllGatherNHR1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PostSync start");
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PostSync finished");
    return CCU_SUCCESS;
}

static CcuResult DoRepeatSendRecvSlices(AllGatherNHR1DMem2MemContext &ctx, const u32 &toRank,
    ccu::LocalAddr &src, ccu::RemoteAddr &dst, u32 signalIndex, bool islastSlice)
{
    const auto *arg = ctx.arg;
    ChannelHandle sendChannel = arg->channels[arg->rank2ChannelIdx.at(toRank)];
    ccu::Variable tmpRepeatNum;
    tmpRepeatNum = ctx.repeatNum;
    ctx.repeatTimeflag = 0;

    CCU_WHILE(tmpRepeatNum != UINT64_MAX)
    {
        tmpRepeatNum += ctx.constVar1;
        CCU_IF(ctx.repeatTimeflag == 1)
        {
            src.addr += ctx.inputRepeatStride;
            dst.addr += ctx.outputRepeatStride;
        } CCU_ELSE {
            if (arg->axisId == 1) {
                src.addr += (islastSlice ? ctx.die0LastSize : ctx.die0Size);
                dst.addr += (islastSlice ? ctx.die0LastSize : ctx.die0Size);
            }
        }
        ccu::Variable &sliceSize = (arg->axisId == 0) ? (islastSlice ? ctx.die0LastSize : ctx.die0Size)
                                    : (islastSlice ? ctx.die1LastSize : ctx.die1Size);

        const uint16_t signalMask = 1 << signalIndex;
        CCU_IF(sliceSize != 0) {
            CCU_CHK_RET(ccu::Write(sendChannel, dst, src, sliceSize, ctx.localEvent, signalMask));
            CCU_CHK_RET(ccu::EventWait(ctx.localEvent, signalMask));
        }
        ctx.repeatTimeflag = 1;
    }

    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherNHRSingleStep(AllGatherNHR1DMem2MemContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    const u32 &toRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.toRank);
    const u32 &fromRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.fromRank);
    u32 sendSliceIdx = 0;
    ChannelHandle sendChannel = arg->channels[toRankIdx];
    ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    HCCL_INFO("sendSliceIdxList.size()[%zu]", sendSliceIdxList.size());
    ctx.srcMem.token = ctx.token[ctx.myRankIdx];
    ctx.dstMem.token = ctx.token[toRankIdx];

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];
        if (nhrStepInfo.step == 0) {
            ctx.srcMem.addr = ctx.input;
            ctx.srcMem.addr += ctx.myrankInputSliceOffset;
        } else {
            ctx.srcMem.addr = ctx.output[ctx.myRankIdx];
            ctx.srcMem.addr += ctx.outputSliceOffset[sendSliceIdx];
        }
        ctx.dstMem.addr = ctx.output[toRankIdx];
        ctx.dstMem.addr += ctx.outputSliceOffset[sendSliceIdx];
        bool islastSlice = false;
        islastSlice = (sendSliceIdx + 1 == arg->dimSize);
        HCCL_INFO("mySubCommRankId[%zu], rankId[%zu], subCommToRankId[%zu], sendSliceIdx[%zu]",
            arg->mySubCommRankId, ctx.myRankIdx, nhrStepInfo.toRank, sendSliceIdx);
        CCU_CHK_RET(DoRepeatSendRecvSlices(ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.dstMem, i % BIT_NUM_PER_CKE, islastSlice));
    }

    if (nhrStepInfo.step + 1 != arg->stepInfoVector.size()) {
        CCU_CHK_RET(ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
        CCU_CHK_RET(ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
    }

    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherNHR(AllGatherNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable localSliceSize;
    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;

    for (u64 i = 0; i < arg->mySubCommRankId; i++) {
        ctx.myrankInputSliceOffset += ctx.inputSliceStride;
    }

    for (u64 i = 0; i < arg->dimSize; i++) {
        ctx.outputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.outputSliceStride;
    }

    ctx.srcMem.addr = ctx.input;
    ctx.srcMem.addr += ctx.myrankInputSliceOffset;
    ctx.srcMem.token = ctx.token[ctx.myRankIdx];
    ctx.dstMem.addr = ctx.output[ctx.myRankIdx];
    ctx.dstMem.addr += ctx.outputSliceOffset[arg->mySubCommRankId];
    ctx.dstMem.token = ctx.token[ctx.myRankIdx];
    ctx.localDst.addr = ctx.output[ctx.myRankIdx];
    ctx.localDst.addr += ctx.outputSliceOffset[arg->mySubCommRankId];
    ctx.localDst.token = ctx.token[ctx.myRankIdx];

    ccu::Variable tmpCopyRepeatNum;
    tmpCopyRepeatNum = ctx.repeatNum;

    bool islastSlice = (arg->mySubCommRankId + 1 == arg->dimSize);

    CCU_WHILE(tmpCopyRepeatNum != UINT64_MAX)
    {
        localSliceSize = (arg->axisId == 0) ? (islastSlice ? ctx.die0LastSize : ctx.die0Size)
                : (islastSlice ? ctx.die1LastSize : ctx.die1Size);
        tmpCopyRepeatNum += ctx.constVar1;
        CCU_IF(ctx.repeatTimeflag != 0)
        {
            ctx.srcMem.addr += ctx.inputRepeatStride;
            ctx.dstMem.addr += ctx.outputRepeatStride;
            ctx.localDst.addr += ctx.outputRepeatStride;
        } CCU_ELSE {
            if (arg->axisId == 1) {
                ctx.srcMem.addr += (islastSlice ? ctx.die0LastSize : ctx.die0Size);
                ctx.dstMem.addr += (islastSlice ? ctx.die0LastSize : ctx.die0Size);
                ctx.localDst.addr += (islastSlice ? ctx.die0LastSize : ctx.die0Size);
            }
        }

        const uint16_t localMask = 1;
        CCU_IF(ctx.isInputOutputEqual == 0)
        {
            CCU_IF(localSliceSize != 0) {
                CCU_CHK_RET(ccu::LocalCopy(ctx.localDst, ctx.srcMem, localSliceSize, ctx.localEvent, localMask));
            } CCU_ELSE {
                CCU_CHK_RET(ccu::EventRecord(ctx.localEvent, localMask));
            }
        } CCU_ELSE {
            CCU_CHK_RET(ccu::EventRecord(ctx.localEvent, localMask));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.localEvent, localMask));
        ctx.repeatTimeflag = 1;
    }

    for (auto &nhrStepInfo : arg->stepInfoVector) {
        CCU_CHK_RET(DoRepeatAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherNHR1D *>(arg);

    AllGatherNHR1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] AllGatherNHR1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoRepeatAllGatherNHR(ctx));

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] AllGatherNHR1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
