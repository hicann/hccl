/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_broadcast_nhr1d_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID      = 0;
constexpr uint16_t TOKEN_XN_ID       = 1;
constexpr uint16_t POST_SYNC_ID      = 3;
constexpr uint16_t STEP_PRE_SYNC_ID  = 4;
constexpr uint16_t STEP_POST_SYNC_ID = 5;

constexpr uint16_t CKE_IDX_0         = 0;
constexpr uint16_t RANK_NUM_PER_CKE  = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

static CcuResult ParseKernelArg(BroadcastNhr1DMem2MemContext &ctx, CcuKernelArgBroadcastNhr1DMem2Mem *kernelArg)
{
    ctx.arg       = kernelArg;
    ctx.localSize = kernelArg->rank2ChannelIdx.size();
    ctx.myRankIdx = kernelArg->rank2ChannelIdx.size();
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] CtxArg: rankId[%u], axisId[%u], axisSize[%u], dimSize[%llu], "
              "stepInfoVectorSize[%zu], localSize[%llu], channelCount[%u]",
              kernelArg->rankId, kernelArg->axisId, kernelArg->axisSize, kernelArg->dimSize,
              kernelArg->stepInfoVector.size(), ctx.localSize, kernelArg->channelCount);
    return CCU_SUCCESS;
}

static CcuResult InitResource(BroadcastNhr1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelBroadcastNhr1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] channels.size: [%u]", arg->channelCount);

    ctx.output.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);

    // 按照rank号从小到大遍历channels，依次取远端资源，要求给框架返回的Link同样是按顺序排列的
    for (uint32_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        HCCL_DEBUG("[CcuKernelBroadcastNhr1DMem2Mem] rankId[%u], channelId[%u]", arg->rankId, channelIdx);
        ctx.output[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }

    ctx.sliceOffset.resize(arg->dimSize);
    ctx.resourceAllocated = false;
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem][InitResource] InitResource end");
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(BroadcastNhr1DMem2MemContext &ctx)
{
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0SliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1SliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0LastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1LastSliceSize, argId++));
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] LoadArgs run finished");
    return CCU_SUCCESS;
}

static CcuResult CalcSliceOffset(BroadcastNhr1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;
    for (uint64_t i = 0; i < arg->dimSize; i++) {
        ctx.sliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += (arg->axisId == 0) ? ctx.die0SliceSize : ctx.die1SliceSize;
    }
    return CCU_SUCCESS;
}

static CcuResult PreSync(BroadcastNhr1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[ctx.myRankIdx],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] BroadcastNhr1D wait all end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(BroadcastNhr1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] BroadcastNhr1D groupwait end");
    return CCU_SUCCESS;
}

static CcuResult DoSendRecvSlice(BroadcastNhr1DMem2MemContext &ctx, const u32 &toRank, ccu::LocalAddr &src,
                                 ccu::RemoteAddr &dst, const u32 &sendSliceIdx, u32 signalIndex)
{
    const auto *arg = ctx.arg;
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem][DoSendRecvSlice] toRank[%u] sendSliceIdx[%u] signalIndex[%u]",
              toRank, sendSliceIdx, signalIndex);
    ChannelHandle sendChannel = arg->channels[arg->rank2ChannelIdx.at(toRank)];

    // 添加 die1 偏移
    if (arg->axisId == 1) {
        src.addr += ctx.die0Size;
        dst.addr += ctx.die0Size;
        ctx.localDst.addr += ctx.die0Size;
    }

    bool islastSlice = (sendSliceIdx + 1 == arg->dimSize);
    ccu::Variable &sliceSize = (arg->axisId == 0) ? (islastSlice ? ctx.die0LastSliceSize : ctx.die0SliceSize)
                                                  : (islastSlice ? ctx.die1LastSliceSize : ctx.die1SliceSize);
    const uint16_t signalMask = 1 << signalIndex;
    CCU_IF(sliceSize != 0)
    {
        CCU_CHK_RET(ccu::Write(sendChannel, dst, src, sliceSize, ctx.event, signalMask));
    }
    CCU_IF(sliceSize == 0)
    {
        CCU_CHK_RET(ccu::EventRecord(ctx.event, signalMask));
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterNHRSingleStep(BroadcastNhr1DMem2MemContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const std::vector<u32> &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;
    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem][DoScatterNHRSingleStep] sendSliceIdxListSize[%zu] recvSliceIdxList[%zu] "
              "step[%u] myRank[%u] nSlices[%u] toRank[%u] fromRank[%u]", sendSliceIdxList.size(), recvSliceIdxList.size(),
              nhrStepInfo.step, nhrStepInfo.myRank, nhrStepInfo.nSlices, nhrStepInfo.toRank, nhrStepInfo.fromRank);
    // 只需要发
    if (sendSliceIdxList.size() != 0) {
        const u32 &toRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.toRank);
        u32 sendSliceIdx = 0;
        ChannelHandle sendChannel = arg->channels[toRankIdx];
        ctx.localSrc.token  = ctx.token[ctx.myRankIdx];
        ctx.remoteDst.token = ctx.token[toRankIdx];
        ctx.localDst.token  = ctx.token[toRankIdx];
        for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
            sendSliceIdx = sendSliceIdxList[i];
            HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem][DoScatterNHRSingleStep] sendSliceIdx[%u]", sendSliceIdx);
            if (i != 0) {
                if (i % RANK_NUM_PER_CKE == 0) {
                    CCU_CHK_RET(ccu::EventWait(ctx.event, (1 << RANK_NUM_PER_CKE) - 1));
                }
            }
            if (nhrStepInfo.step == 0) {
                // 只有第0步的源数据从input中取
                ctx.localSrc.addr = ctx.input;
                ctx.localSrc.addr += ctx.sliceOffset[sendSliceIdx];
            } else {
                ctx.localSrc.addr = ctx.output[ctx.myRankIdx];
                ctx.localSrc.addr += ctx.sliceOffset[sendSliceIdx];
            }
            ctx.remoteDst.addr = ctx.output[toRankIdx];
            ctx.remoteDst.addr += ctx.sliceOffset[sendSliceIdx];
            ctx.localDst.addr = ctx.output[toRankIdx];
            ctx.localDst.addr += ctx.sliceOffset[sendSliceIdx];
            CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, ctx.localSrc, ctx.remoteDst, sendSliceIdx,
                                        i % RANK_NUM_PER_CKE));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1));
        // 通知toRank数据写入完毕
        CCU_CHK_RET(ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID));
    }
    // 只需要收
    if (recvSliceIdxList.size() != 0) {
        const u32 &fromRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.fromRank);
        ChannelHandle recvChannel = arg->channels[fromRankIdx];
        CCU_CHK_RET(ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterNHR(BroadcastNhr1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint32_t NHR_NUM = 2;
    for (uint64_t i = 0; i < arg->stepInfoVector.size() / NHR_NUM; i++) {
        const NHRStepInfo &nhrStepInfo = arg->stepInfoVector[i];
        CCU_CHK_RET(DoScatterNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHRSingleStep(BroadcastNhr1DMem2MemContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    const u32 &toRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.toRank);
    const u32 &fromRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.fromRank);
    u32 sendSliceIdx = 0;
    ChannelHandle sendChannel = arg->channels[toRankIdx];
    ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    ctx.localSrc.token  = ctx.token[ctx.myRankIdx];
    ctx.remoteDst.token = ctx.token[toRankIdx];
    ctx.localDst.token  = ctx.token[toRankIdx];

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % RANK_NUM_PER_CKE == 0) {
                CCU_CHK_RET(ccu::EventWait(ctx.event, (1 << RANK_NUM_PER_CKE) - 1));
            }
        }

        ctx.localSrc.addr = ctx.output[ctx.myRankIdx];
        ctx.localSrc.addr += ctx.sliceOffset[sendSliceIdx];

        ctx.remoteDst.addr = ctx.output[toRankIdx];
        ctx.remoteDst.addr += ctx.sliceOffset[sendSliceIdx];
        ctx.localDst.addr = ctx.output[toRankIdx];
        ctx.localDst.addr += ctx.sliceOffset[sendSliceIdx];
        CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, ctx.localSrc, ctx.remoteDst, sendSliceIdx,
                                    i % RANK_NUM_PER_CKE));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.event, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1));

    if (nhrStepInfo.step + 1 != arg->stepInfoVector.size()) {   // 最后一步不需要同步
        // 通知toRank，写入完毕
        CCU_CHK_RET(ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
        // 等待fromRank通知写入完毕
        CCU_CHK_RET(ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
    }

    HCCL_DEBUG("[CcuKernelBroadcastNhr1DMem2Mem][DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, "
               "nSlice=%zu toRankIdx=%u, fromRankIdx=%u",
               arg->rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size(),
               toRankIdx, fromRankIdx);
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHR(BroadcastNhr1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint32_t NHR_NUM = 2;
    for (uint64_t i = arg->stepInfoVector.size() / NHR_NUM; i < arg->stepInfoVector.size(); i++) {
        const NHRStepInfo &nhrStepInfo = arg->stepInfoVector[i];
        CCU_CHK_RET(DoAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastNhr1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcastNhr1DMem2Mem *>(arg);

    BroadcastNhr1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] BroadcastNHR1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(CalcSliceOffset(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoScatterNHR(ctx));
    CCU_CHK_RET(DoAllGatherNHR(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[CcuKernelBroadcastNhr1DMem2Mem] BroadcastNHR1D end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl
