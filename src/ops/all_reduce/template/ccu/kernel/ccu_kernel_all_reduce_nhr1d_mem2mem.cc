/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_nhr1d_mem2mem.h"

namespace ops_hccl {
constexpr uint16_t OUTPUT_XN_ID     = 1;
constexpr uint16_t TOKEN_XN_ID      = 2;
constexpr uint16_t POST_SYNC_ID     = 3;
constexpr uint16_t STEP0_PRE_SYNC_ID = 4;
constexpr uint16_t STEP0_POST_SYNC_ID = 5;
constexpr uint16_t STEP1_POST_SYNC_ID = 6;
constexpr uint16_t CKE_IDX_0        = 0;    
constexpr uint16_t FST_AXIS_ID      = 0;
constexpr uint16_t SEC_AXIS_ID      = 1;
constexpr uint16_t RANK_NUM_PER_CKE = 16; // 本rank给远端置位时应当写的CKE，16个对端一个CKE

static CcuResult ParseKernelArg(AllReduceNHR1DContext &ctx, CcuKernelArgAllReduceNHR1D *kernelArg)
{
    ctx.rankId          = kernelArg->rankId;
    ctx.rankSize        = kernelArg->rankSize;
    ctx.axisId          = kernelArg->axisId;
    ctx.axisSize        = kernelArg->axisSize;
    ctx.stepInfoVector  = kernelArg->stepInfoVector;
    ctx.indexMap        = kernelArg->indexMap;
    ctx.localSize       = ctx.indexMap.size();
    ctx.myRankIdx       = ctx.indexMap.size();
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.reduceOp        = kernelArg->opParam.reduceType;

    HCCL_INFO("[CcuKernelAllReduceNHR1D] kernelArg: rankId_[%u], axisId_[%u], axisSize_[%u], rankSize_[%u], localSize_[%u], "
              "dataType[%d], reduceOp[%d]",
              ctx.rankId, ctx.axisId, ctx.axisSize, ctx.rankSize, ctx.localSize, ctx.dataType, ctx.reduceOp);
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllReduceNHR1DContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllReduceNHR1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllReduceNHR1D] channels.size: [%u]", arg->channelCount);

    ctx.output.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);
    for (uint64_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        ctx.output[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }

    ctx.resourceAllocated = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllReduceNHR1DContext &ctx)
{
    HCCL_DEBUG("[CcuKernelAllReduceNHR1D] LoadArgs run start");
    uint32_t argId = 0;
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
    HCCL_DEBUG("[CcuKernelAllReduceNHR1D] LoadArgs run end");
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllReduceNHR1DContext &ctx)
{
    const auto *arg = ctx.arg;
    HCCL_DEBUG("[CcuKernelAllReduceNHR1D] PreSync start");
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[ctx.localSize],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.localSize],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_DEBUG("[CcuKernelAllReduceNHR1D] PreSync end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllReduceNHR1DContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_DEBUG("[CcuKernelAllReduceNHR1D] PostSync run finished");
    return CCU_SUCCESS;
}

static CcuResult DoWriteReduceSlice(AllReduceNHR1DContext &ctx, const u32 &toRank, ccu::LocalAddr &src, ccu::RemoteAddr &dst, 
                               const u32 &sendSliceIdx, u32 signalIndex)
{
    const auto *arg = ctx.arg;
    ChannelHandle sendChannel = arg->channels[ctx.indexMap[toRank]];
    bool          islastSlice;
    
    // 添加 die1 偏移
    if (ctx.axisId == 1) {
        src.addr += ctx.die0Size;
        dst.addr += ctx.die0Size;
    }

    // allreduce切片的最后一块slice，大小可能不一致
    islastSlice = (sendSliceIdx + 1 == ctx.rankSize);
    ccu::Variable &sliceSize = ctx.axisId == 0? (islastSlice? ctx.die0LastSliceSize : ctx.die0SliceSize)
                                                    : (islastSlice? ctx.die1LastSliceSize : ctx.die1SliceSize);
    CCU_IF(sliceSize != 0)
    {   
        ccu::WriteReduce(sendChannel, dst, src, sliceSize, ctx.dataType, ctx.reduceOp, ctx.localEvent, 1 << signalIndex);
    }
    CCU_IF(sliceSize == 0)
    {
        ccu::EventRecord(ctx.localEvent, 1 << signalIndex);
    }
    return CCU_SUCCESS;
}

static CcuResult DoReduceScatterNHRSingleStep(AllReduceNHR1DContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    u32& toRankIdx = ctx.indexMap[nhrStepInfo.toRank];
    u32& fromRankIdx = ctx.indexMap[nhrStepInfo.fromRank];
    u32  sendSliceIdx = 0;
    ChannelHandle           sendChannel = arg->channels[toRankIdx];
    ChannelHandle           recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;
    ctx.srcMem.token                         = ctx.token[ctx.myRankIdx];
    ctx.rmtDstMem.token                      = ctx.token[toRankIdx];

    if (nhrStepInfo.step != 0) {
        // 通知fromRank，可以写入
        ccu::NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP0_PRE_SYNC_ID);

        // 等待toRank通知其可以写入
        ccu::NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP0_PRE_SYNC_ID);
    }

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % RANK_NUM_PER_CKE == 0) {
                ccu::EventWait(ctx.localEvent, (1 << RANK_NUM_PER_CKE) - 1);
            }
        }

        if (nhrStepInfo.step == 0) {
            ctx.srcMem.addr = ctx.input;
            ctx.srcMem.addr += ctx.sliceOffset[sendSliceIdx];
        } else {
            ctx.srcMem.addr = ctx.output[ctx.myRankIdx];
            ctx.srcMem.addr += ctx.sliceOffset[sendSliceIdx];
        }
        
        ctx.rmtDstMem.addr = ctx.output[toRankIdx];
        ctx.rmtDstMem.addr += ctx.sliceOffset[sendSliceIdx];

        CCU_CHK_RET(DoWriteReduceSlice(ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.rmtDstMem, sendSliceIdx, i % RANK_NUM_PER_CKE));
    }
    ccu::EventWait(ctx.localEvent, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);

    // 通知toRank数据写入完毕
    ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP0_POST_SYNC_ID);
    // 等待fromRank通知数据写入完毕
    ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP0_POST_SYNC_ID);
    HCCL_DEBUG("[DoReduceScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
                ctx.rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return CCU_SUCCESS;
}

static CcuResult DoReduceScatterNHR(AllReduceNHR1DContext &ctx)
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = 0; i < ctx.stepInfoVector.size() / NHR_NUM; i++) {
        const NHRStepInfo &nhrStepInfo = ctx.stepInfoVector[i];
        CCU_CHK_RET(DoReduceScatterNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static CcuResult DoSendRecvSlice(AllReduceNHR1DContext &ctx, const u32 &toRank, ccu::LocalAddr &src, ccu::RemoteAddr &dst,
                            const u32 &sendSliceIdx, u32 signalIndex)
{
    const auto *arg = ctx.arg;
    ChannelHandle sendChannel = arg->channels[ctx.indexMap[toRank]];
    bool          islastSlice;
    
    // 添加 die1 偏移
    if (ctx.axisId == 1) {
        src.addr += ctx.die0Size;
        dst.addr += ctx.die0Size;
    }

    islastSlice = (sendSliceIdx + 1 == ctx.rankSize);
    ccu::Variable &sliceSize = ctx.axisId == 0? (islastSlice? ctx.die0LastSliceSize : ctx.die0SliceSize)
                                                    : (islastSlice? ctx.die1LastSliceSize : ctx.die1SliceSize);
    CCU_IF(sliceSize != 0)
    {   
        ccu::Write(sendChannel, dst, src, sliceSize, ctx.localEvent, 1 << signalIndex);
    }
    CCU_IF(sliceSize == 0)
    {   
        ccu::EventRecord(ctx.localEvent, 1 << signalIndex);
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHRSingleStep(AllReduceNHR1DContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    u32& toRankIdx = ctx.indexMap[nhrStepInfo.toRank];
    u32& fromRankIdx = ctx.indexMap[nhrStepInfo.fromRank];
    u32  sendSliceIdx = 0;
    ChannelHandle           sendChannel = arg->channels[toRankIdx];
    ChannelHandle           recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList  = nhrStepInfo.txSliceIdxs;
    ctx.srcMem.token                         = ctx.token[ctx.myRankIdx];
    ctx.rmtDstMem.token                      = ctx.token[toRankIdx];
    
    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % RANK_NUM_PER_CKE == 0) {
                ccu::EventWait(ctx.localEvent, (1 << RANK_NUM_PER_CKE) - 1);
            }
        }

        ctx.srcMem.addr = ctx.output[ctx.myRankIdx];
        ctx.srcMem.addr += ctx.sliceOffset[sendSliceIdx];

        ctx.rmtDstMem.addr = ctx.output[toRankIdx];
        ctx.rmtDstMem.addr += ctx.sliceOffset[sendSliceIdx];
        CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.rmtDstMem, sendSliceIdx, i % RANK_NUM_PER_CKE));
    }
    ccu::EventWait(ctx.localEvent, (1 << (sendSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);

    if (nhrStepInfo.step + 1 != ctx.stepInfoVector.size()) {   // 最后一步不需要同步
        // 通知toRank，写入完毕
        ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP1_POST_SYNC_ID);
        // 等待fromRank通知写入完毕
        ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP1_POST_SYNC_ID);
    }

    HCCL_DEBUG("[DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
                ctx.rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHR(AllReduceNHR1DContext &ctx)
{
    const uint32_t NHR_NUM = 2;
    for (u64 i = ctx.stepInfoVector.size() / NHR_NUM; i < ctx.stepInfoVector.size(); i++) {
        const NHRStepInfo &nhrStepInfo = ctx.stepInfoVector[i];
        CCU_CHK_RET(DoAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static std::vector<u32> GetNonTxSliceIdxs(AllReduceNHR1DContext &ctx, const std::vector<u32> &txSliceIdxs)
{
    std::vector<bool> isTx(ctx.rankSize, false);
    for (u32 idx : txSliceIdxs) {
        if (idx < ctx.rankSize) {
            isTx[idx] = true;
        }
    }

    std::vector<u32> nonTxSliceIdxs;
    for (u32 idx = 0; idx < ctx.rankSize; ++idx) {
        if (!isTx[idx]) {
            nonTxSliceIdxs.push_back(idx);
        }
    }

    return nonTxSliceIdxs;
}

static CcuResult DoLocalCopySlice(AllReduceNHR1DContext &ctx, ccu::LocalAddr &src, ccu::LocalAddr &dst,
                             const u32 &copySliceIdx, u32 signalIndex)
{
    bool islastSlice;
    // 添加 die1 偏移
    if (ctx.axisId == 1) {
        src.addr += ctx.die0Size;
        dst.addr += ctx.die0Size;
    }

    islastSlice = (copySliceIdx + 1 == ctx.rankSize);
    ccu::Variable &sliceSize = ctx.axisId == 0? (islastSlice? ctx.die0LastSliceSize : ctx.die0SliceSize)
                                                    : (islastSlice? ctx.die1LastSliceSize : ctx.die1SliceSize);
    CCU_IF(sliceSize != 0)
    {   
        ccu::LocalCopy(dst, src, sliceSize, ctx.localEvent, 1 << signalIndex);
    }
    CCU_IF(sliceSize == 0)
    {   
        ccu::EventRecord(ctx.localEvent, 1 << signalIndex);
    }
    return CCU_SUCCESS;
}

static CcuResult LocalCopySlices(AllReduceNHR1DContext &ctx)
{
    u32              nonTxSliceIdx    = 0;
    ccu::Variable tmpSliceOffset;
    tmpSliceOffset                    = 0;

    for (u64 i = 0; i < ctx.rankSize; i++) {
        ccu::Variable offset;
        ctx.sliceOffset.push_back(offset);
        ctx.sliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.axisId == 0? ctx.die0SliceSize: ctx.die1SliceSize;
    }
    
    // 当input == output时，不需要拷贝
    CCU_IF(ctx.isInputOutputEqual == 0)
    {
        // 将step0中不需要写的slice，拷贝到本rank的output中
        const NHRStepInfo &nhrStepInfo = ctx.stepInfoVector[0];
        const std::vector<u32> &nonTxSliceIdxList = GetNonTxSliceIdxs(ctx, nhrStepInfo.txSliceIdxs);
        for (u32 i = 0; i < nonTxSliceIdxList.size(); i++) {
            nonTxSliceIdx = nonTxSliceIdxList[i];

            if (i != 0) {
                if (i % RANK_NUM_PER_CKE == 0) {
                    ccu::EventWait(ctx.localEvent, (1 << RANK_NUM_PER_CKE) - 1);
                }
            }

            ctx.srcMem.addr  = ctx.input;
            ctx.srcMem.addr += ctx.sliceOffset[nonTxSliceIdx];
            ctx.srcMem.token = ctx.token[ctx.myRankIdx];

            ctx.locDstMem.addr  = ctx.output[ctx.myRankIdx];
            ctx.locDstMem.addr += ctx.sliceOffset[nonTxSliceIdx];
            ctx.locDstMem.token = ctx.token[ctx.myRankIdx];
            CCU_CHK_RET(DoLocalCopySlice(ctx, ctx.srcMem, ctx.locDstMem, nonTxSliceIdx, i));
        }
        ccu::EventWait(ctx.localEvent, (1 << (nonTxSliceIdxList.size() % RANK_NUM_PER_CKE)) - 1);
    } 
    return CCU_SUCCESS;
}

CcuResult CcuAllReduceNHR1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduceNHR1D *>(arg);

    AllReduceNHR1DContext ctx;
    ctx.arg = kernelArg;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_DEBUG("[CcuKernelAllReduceNHR1D] AllReduceNHR1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(LocalCopySlices(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoReduceScatterNHR(ctx));
    CCU_CHK_RET(DoAllGatherNHR(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_DEBUG("[CcuKernelAllReduceNHR1D] AllReduceNHR1D end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl