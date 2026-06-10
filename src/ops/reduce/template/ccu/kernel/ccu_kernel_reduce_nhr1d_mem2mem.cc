/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_nhr1d_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID                   = 1;
constexpr uint16_t TOKEN_XN_ID                    = 2;
constexpr uint16_t POST_SYNC_ID                   = 3;
constexpr uint16_t REDUCE_SCATTER_PRE_SYNC_ID     = 4;
constexpr uint16_t REDUCE_SCATTER_POST_SYNC_ID    = 5;
constexpr uint16_t GATHER_SYNC_ID                 = 6;
constexpr uint16_t CKE_IDX_0                      = 0; // 前后同步
constexpr uint16_t RANK_NUM_PER_CKE               = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

static CcuResult ParseKernelArg(ReduceNHR1DMem2MemContext &ctx, CcuKernelArgReduceNHR1DMem2Mem *kernelArg)

{
    ctx.localSize = kernelArg->rank2ChannelIdx.size();
    ctx.myRankIdx = kernelArg->rank2ChannelIdx.size();
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceNHR1DMem2MemContext &ctx)
{
    HCCL_DEBUG("[CcuKernelReduceNHR1DMem2Mem] LoadArgs run start");
    uint32_t argId = 0;
    const auto *arg = ctx.arg;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0SliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1SliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0LastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1LastSliceSize, argId++));
    HCCL_DEBUG("[CcuKernelReduceNHR1DMem2Mem] LoadArgs run end");
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceNHR1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelReduceNHR1DMem2Mem] channels.size: [%u]", arg->channelCount);

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.output.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);
    for (uint64_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        ctx.output[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }

    ctx.resourceAllocated = false;

    return CCU_SUCCESS;
}

static void PreSync(ReduceNHR1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceNHR1DMem2Mem] PreSync begin");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[ctx.localSize], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.localSize], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelReduceNHR1DMem2Mem] PreSync end");
}

static void PostSync(ReduceNHR1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceNHR1DMem2Mem] post sync start");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceNHR1DMem2Mem] post sync end");
}

static CcuResult DoWriteReduceSlice(ReduceNHR1DMem2MemContext &ctx, const u32 &toRank, const u32 &sendSliceIdx, u32 signalIndex)
{
    const auto *arg = ctx.arg;
    auto toRankIt = arg->rank2ChannelIdx.find(toRank);
    if (toRankIt == arg->rank2ChannelIdx.end()) {
        HCCL_ERROR("[CcuKernelReduceNHR1DMem2Mem] rank2ChannelIdx not find toRank key [%u]", toRank);
        return CCU_E_PARA;
    }
    const u32& toRankIdx = toRankIt->second;
    ChannelHandle sendChannel = arg->channels[toRankIdx];

    bool          islastSlice;

    // 添加 die1 偏移
    if (arg->axisId == 1) {
        ctx.localSrc.addr += ctx.die0Size;
        ctx.remoteDst.addr += ctx.die0Size;
    }

    // allreduce切片的最后一块slice，大小可能不一致
    islastSlice = (sendSliceIdx + 1 == arg->rankSize);
    ccu::Variable &sliceSize = arg->axisId == 0 ? (islastSlice ? ctx.die0LastSliceSize : ctx.die0SliceSize)
                                                  : (islastSlice ? ctx.die1LastSliceSize : ctx.die1SliceSize);

    CCU_IF(sliceSize != 0) {
        ccu::WriteReduce(sendChannel, ctx.remoteDst, ctx.localSrc, sliceSize, ctx.dataType, ctx.reduceOp, ctx.event, 1 << signalIndex);
    }
    CCU_IF(sliceSize == 0) {
		const uint32_t rankMask = 1 << signalIndex;
        ccu::EventRecord(ctx.event, rankMask);
    }
    return CCU_SUCCESS;
}

static CcuResult DoReduceScatterNHRSingleStep(ReduceNHR1DMem2MemContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    auto toRankIt = arg->rank2ChannelIdx.find(nhrStepInfo.toRank);
    if (toRankIt == arg->rank2ChannelIdx.end()) {
        HCCL_ERROR("[CcuKernelReduceNHR1DMem2Mem] rank2ChannelIdx not find toRank key [%u]", nhrStepInfo.toRank);
        return CCU_E_PARA;
    }
    const u32& toRankIdx = toRankIt->second;

    auto fromRankIt = arg->rank2ChannelIdx.find(nhrStepInfo.fromRank);
    if (fromRankIt == arg->rank2ChannelIdx.end()) {
        HCCL_ERROR("[CcuKernelReduceNHR1DMem2Mem] rank2ChannelIdx not find fromRank key [%u]", nhrStepInfo.toRank);
        return CCU_E_PARA;
    }
    const u32& fromRankIdx = fromRankIt->second;

    u32  sendSliceIdx = 0;
    ChannelHandle sendChannel = arg->channels[toRankIdx];
    ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;
    ctx.localSrc.token = ctx.token[ctx.myRankIdx];
    ctx.remoteDst.token = ctx.token[toRankIdx];

    if (nhrStepInfo.step != 0) {
        // 通知fromRank，可以写入
        ccu::NotifyRecord(recvChannel, CKE_IDX_0, 1 << REDUCE_SCATTER_PRE_SYNC_ID);

        // 等待toRank通知其可以写入
        ccu::NotifyWait(sendChannel, CKE_IDX_0, 1 << REDUCE_SCATTER_PRE_SYNC_ID);
    }

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        // cke用完了，等待上一轮结束在使用
        if (i != 0) {
            if (i % RANK_NUM_PER_CKE == 0) {
                ccu::EventWait(ctx.event, (1 << RANK_NUM_PER_CKE) - 1);
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

        CCU_CHK_RET(DoWriteReduceSlice(ctx, nhrStepInfo.toRank, sendSliceIdx, i % RANK_NUM_PER_CKE));
    }
    // 等待上面的DoWriteReduceSlice方法传完
    ccu::EventWait(ctx.event, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);

    // 通知toRank数据写入完毕
    ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << REDUCE_SCATTER_POST_SYNC_ID);
    // 等待fromRank通知数据写入完毕
    ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << REDUCE_SCATTER_POST_SYNC_ID);

    HCCL_DEBUG("[DoReduceScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
                arg->rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return CCU_SUCCESS;
}

static CcuResult DoReduceScatterNHR(ReduceNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint32_t NHR_NUM = 2;
    for (u64 i = 0; i < arg->stepInfoVector.size() / NHR_NUM; i++) {
        const NHRStepInfo &nhrStepInfo = arg->stepInfoVector[i];
        CCU_CHK_RET(DoReduceScatterNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static CcuResult DoSendRecvSlice(ReduceNHR1DMem2MemContext &ctx, const u32 &toRank, const u32 &sendSliceIdx, u32 signalIndex)
{
    const auto *arg = ctx.arg;
    auto toRankIt = arg->rank2ChannelIdx.find(toRank);
    if (toRankIt == arg->rank2ChannelIdx.end()) {
        HCCL_ERROR("[CcuKernelReduceNHR1DMem2Mem] rank2ChannelIdx not find toRank key [%u]", toRank);
        return CCU_E_PARA;
    }
    const u32& toRankIdx = toRankIt->second;
    ChannelHandle sendChannel = arg->channels[toRankIdx];

    bool islastSlice;

    // 添加 die1 偏移
    if (arg->axisId == 1) {
        ctx.localSrc.addr += ctx.die0Size;
        ctx.remoteDst.addr += ctx.die0Size;
    }

    islastSlice = (sendSliceIdx + 1 == arg->rankSize);
    ccu::Variable &sliceSize = arg->axisId == 0 ? (islastSlice ? ctx.die0LastSliceSize : ctx.die0SliceSize)
                                                  : (islastSlice ? ctx.die1LastSliceSize : ctx.die1SliceSize);

    CCU_IF(sliceSize != 0) {
        ccu::Write(sendChannel, ctx.remoteDst, ctx.localSrc, sliceSize, ctx.event, 1 << signalIndex);
    }
    CCU_IF(sliceSize == 0) {
		const uint32_t rankMask = 1 << signalIndex;
        ccu::EventRecord(ctx.event, rankMask);
    }
    return CCU_SUCCESS;
}

static CcuResult DoGatherNHRSingleStep(ReduceNHR1DMem2MemContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    auto toRankIt = arg->rank2ChannelIdx.find(nhrStepInfo.toRank);
    if (toRankIt == arg->rank2ChannelIdx.end()) {
        HCCL_ERROR("[CcuKernelReduceNHR1DMem2Mem] rank2ChannelIdx not find toRank key [%u]", nhrStepInfo.toRank);
        return CCU_E_PARA;
    }
    const u32& toRankIdx = toRankIt->second;

    auto fromRankIt = arg->rank2ChannelIdx.find(nhrStepInfo.fromRank);
    if (fromRankIt == arg->rank2ChannelIdx.end()) {
        HCCL_ERROR("[CcuKernelReduceNHR1DMem2Mem] rank2ChannelIdx not find fromRank key [%u]", nhrStepInfo.fromRank);
        return CCU_E_PARA;
    }
    const u32& fromRankIdx = fromRankIt->second;

    u32  sendSliceIdx = 0;
    ChannelHandle sendChannel = arg->channels[toRankIdx];
    ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;
    ctx.localSrc.token = ctx.token[ctx.myRankIdx];
    ctx.remoteDst.token = ctx.token[toRankIdx];

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % RANK_NUM_PER_CKE == 0) {
                ccu::EventWait(ctx.event, (1 << RANK_NUM_PER_CKE) - 1);
            }
        }

        ctx.localSrc.addr = ctx.output[ctx.myRankIdx];
        ctx.localSrc.addr += ctx.sliceOffset[sendSliceIdx];

        ctx.remoteDst.addr = ctx.output[toRankIdx];
        ctx.remoteDst.addr += ctx.sliceOffset[sendSliceIdx];
        CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, sendSliceIdx, i % RANK_NUM_PER_CKE));
    }
    ccu::EventWait(ctx.event, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);

    if (nhrStepInfo.step + 1 != arg->stepInfoVector.size()) {   // 最后一步不需要同步
        // 通知toRank，写入完毕
        ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << GATHER_SYNC_ID);
        // 等待fromRank通知写入完毕
        ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << GATHER_SYNC_ID);
    }

    HCCL_DEBUG("[DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
                arg->rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return CCU_SUCCESS;
}

static CcuResult DoGatherNHR(ReduceNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint32_t NHR_NUM = 2;
    for (u64 i = arg->stepInfoVector.size() / NHR_NUM; i < arg->stepInfoVector.size(); i++) {
        const NHRStepInfo &nhrStepInfo = arg->stepInfoVector[i];
        CCU_CHK_RET(DoGatherNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static CcuResult DoLocalCopySlice(ReduceNHR1DMem2MemContext &ctx, const u32 &copySliceIdx, u32 signalIndex)
{
    const auto *arg = ctx.arg;
    bool islastSlice;
    // 添加 die1 偏移
    if (arg->axisId == 1) {
        ctx.localSrc.addr += ctx.die0Size;
        ctx.localDst.addr += ctx.die0Size;
    }

    islastSlice = (copySliceIdx + 1 == arg->rankSize);
    ccu::Variable &sliceSize = arg->axisId == 0 ? (islastSlice ? ctx.die0LastSliceSize : ctx.die0SliceSize)
                                                  : (islastSlice ? ctx.die1LastSliceSize : ctx.die1SliceSize);

    CCU_IF(sliceSize != 0) {
        ccu::LocalCopy(ctx.localDst, ctx.localSrc, sliceSize, ctx.event, 1 << signalIndex);
    }
    CCU_IF(sliceSize == 0) {
		const uint32_t rankMask = 1 << signalIndex;
        ccu::EventRecord(ctx.event, rankMask);
    }
    return CCU_SUCCESS;
}

static std::vector<u32> GetNonTxSliceIdxs(ReduceNHR1DMem2MemContext &ctx, const std::vector<u32> &txSliceIdxs)
{
    const auto *arg = ctx.arg;
    std::vector<bool> isTx(arg->rankSize, false);
    for (u32 idx : txSliceIdxs) {
        if (idx < arg->rankSize) {
            isTx[idx] = true;
        }
    }

    std::vector<u32> nonTxSliceIdxs;
    for (u32 idx = 0; idx < arg->rankSize; ++idx) {
        if (!isTx[idx]) {
            nonTxSliceIdxs.push_back(idx);
        }
    }

    return nonTxSliceIdxs;
}

static CcuResult LocalCopySlices(ReduceNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    u32 nonTxSliceIdx = 0;
    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;

    ctx.sliceOffset.resize(arg->rankSize);
    for (u64 i = 0; i < arg->rankSize; i++) {
        ctx.sliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += arg->axisId == 0 ? ctx.die0SliceSize : ctx.die1SliceSize;
    }

    // 当input == output时，不需要拷贝
    CCU_IF(ctx.isInputOutputEqual == 0)
    {
        // 将step0中不需要写的slice，拷贝到本rank的output中
        const NHRStepInfo &nhrStepInfo = arg->stepInfoVector[0];
        const std::vector<u32> &nonTxSliceIdxList = GetNonTxSliceIdxs(ctx, nhrStepInfo.txSliceIdxs);
        for (u32 i = 0; i < nonTxSliceIdxList.size(); i++) {
            nonTxSliceIdx = nonTxSliceIdxList[i];

            if (i != 0) {
                if (i % RANK_NUM_PER_CKE == 0) {
                    ccu::EventWait(ctx.event, (1 << RANK_NUM_PER_CKE) - 1);
                }
            }

            ctx.localSrc.addr  = ctx.input;
            ctx.localSrc.addr += ctx.sliceOffset[nonTxSliceIdx];
            ctx.localSrc.token = ctx.token[ctx.myRankIdx];

            ctx.localDst.addr  = ctx.output[ctx.myRankIdx];
            ctx.localDst.addr += ctx.sliceOffset[nonTxSliceIdx];
            ctx.localDst.token = ctx.token[ctx.myRankIdx];
            CCU_CHK_RET(DoLocalCopySlice(ctx, nonTxSliceIdx, i));
        }
        ccu::EventWait(ctx.event, (1 << (nonTxSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
    }
    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceNHR1DMem2Mem *>(arg);

    ReduceNHR1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuContextReduceNHR1DMem2mem] ReduceNHR1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(LocalCopySlices(ctx));
    PreSync(ctx);
    CCU_CHK_RET(DoReduceScatterNHR(ctx));
    CCU_CHK_RET(DoGatherNHR(ctx));
    PostSync(ctx);
    HCCL_INFO("[CcuContextReduceNHR1DMem2mem] ReduceNHR1DMem2Mem end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl