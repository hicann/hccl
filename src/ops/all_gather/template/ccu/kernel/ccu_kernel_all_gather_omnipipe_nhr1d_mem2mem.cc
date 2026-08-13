/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_omnipipe_nhr1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID = 0;
constexpr uint16_t TOKEN_XN_ID = 1;
constexpr uint16_t POST_SYNC_ID = 2;
constexpr uint16_t STEP_PRE_SYNC_ID = 4;
constexpr uint16_t STEP_POST_SYNC_ID = 5;

constexpr uint16_t CKE_IDX_0 = 0;

constexpr uint16_t CKE_IDX_OUTPUT = 0;
constexpr uint16_t CKE_IDX_TOKEN = 1;
constexpr uint16_t CKE_IDX_READY = 2;
constexpr uint16_t CKE_IDX_DONE = 3;
constexpr uint16_t POST_XN_ID = 4;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static CcuResult
ParseKernelArg(AllGatherOmniPipeNHR1DMem2MemContext& ctx, CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem* kernelArg)
{
    ctx.arg = kernelArg;
    kernelArg->localSize = kernelArg->rank2ChannelIdx.size(); // nhr算法通信rank数
    kernelArg->myRankIdx = kernelArg->rank2ChannelIdx.size(); // InitResources中将本端放在末尾 此处为对应的idx
    if (kernelArg->subCommRanks.empty() || kernelArg->subCommRanks[0].empty()
        || kernelArg->rankId >= kernelArg->subCommRanks[0].size()) {
        HCCL_ERROR("[%s] invalid subCommRanks or rankId[%u] out of range.", __func__, kernelArg->rankId);
        return CcuResult::CCU_E_INTERNAL;
    }
    kernelArg->userRank = kernelArg->subCommRanks[0][kernelArg->rankId];
    return CcuResult::CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherOmniPipeNHR1DMem2MemContext& ctx)
{
    HCCL_INFO("[%s] start", __func__);
    const auto* arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[%s] channels is empty!", __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[%s] channels.size[%u]", __func__, arg->channelCount);

    ctx.output.resize(arg->localSize + 1);
    ctx.token.resize(arg->localSize + 1);

    for (uint32_t channelIdx = 0; channelIdx < arg->localSize; channelIdx++) {
        HCCL_DEBUG("[%s] MyRank[%u], channelIdx[%u]", __func__, arg->rankId, channelIdx);
        ctx.output[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }

    ctx.inputOmniSliceStrideVec.resize(arg->rankSize);
    ctx.inputOmniSliceSizeVec.resize(arg->rankSize);

    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherOmniPipeNHR1DMem2MemContext& ctx)
{
    HCCL_INFO("[%s] start", __func__);
    const auto* arg = ctx.arg;
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, argId++));
    for (uint32_t i = 0; i < arg->rankSize; ++i) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceStrideVec[i], argId++));
    }
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    for (uint32_t i = 0; i < arg->rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceSizeVec[i], argId++));
    }

    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherOmniPipeNHR1DMem2MemContext& ctx)
{
    HCCL_INFO("[%s] start", __func__);
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.output[arg->myRankIdx], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[arg->myRankIdx], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult PostSync(AllGatherOmniPipeNHR1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

static CcuResult
DoRepeatAllGatherNHRSingleStep(AllGatherOmniPipeNHR1DMem2MemContext& ctx, const NHRStepInfo& nhrStepInfo)
{
    const auto* arg = ctx.arg;
    const u32& toRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.toRank);
    const u32& fromRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.fromRank);
    const ChannelHandle& sendChannel = arg->channels[toRankIdx];
    const ChannelHandle& recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    const std::vector<u32>& recvSliceIdxList = nhrStepInfo.rxSliceIdxs;
    HCCL_DEBUG(
        "[%s] myRank[%u] rankId[%u] step[%u] toRank[%u](channelIdx[%u]) fromRank[%u](channelIdx[%u]) SliceSize[%u]",
        __func__, arg->userRank, arg->rankId, nhrStepInfo.step, nhrStepInfo.toRank, toRankIdx, nhrStepInfo.fromRank,
        fromRankIdx, recvSliceIdxList.size());

    ccu::LocalAddr src;
    ccu::RemoteAddr dst;
    src.token = ctx.token[arg->myRankIdx];
    dst.token = ctx.token[toRankIdx];

    // 通知对端rank自己准备好了-前同步
    CCU_CHK_RET(ccu::NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID));
    CCU_CHK_RET(ccu::NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID));

    for (uint32_t idx = 0; idx < sendSliceIdxList.size(); idx++) {
        u32 sendSliceIdx = sendSliceIdxList[idx];
        src.addr = ctx.output[arg->myRankIdx];
        dst.addr = ctx.output[toRankIdx];
        if (sendSliceIdx == arg->rankId) {
            src.addr += ctx.sliceStride;
            src.addr += ctx.inputOmniPipeSliceStride;

            dst.addr += ctx.sliceStride;
            dst.addr += ctx.inputOmniPipeSliceStride;
            ctx.sliceSize = ctx.inputOmniSliceSizeVec[sendSliceIdx];
        } else {
            src.addr += ctx.inputOmniSliceStrideVec[sendSliceIdx];
            dst.addr += ctx.inputOmniSliceStrideVec[sendSliceIdx];
            ctx.sliceSize = ctx.inputOmniSliceSizeVec[sendSliceIdx];
        }

        uint16_t mask = 1 << idx;
        CCU_IF(ctx.sliceSize != 0) { CCU_CHK_RET(ccu::Write(sendChannel, dst, src, ctx.sliceSize, ctx.event, mask)); }
        CCU_IF(ctx.sliceSize == 0) { CCU_CHK_RET(ccu::EventRecord(ctx.event, mask)); }
    }
    uint16_t sendBit = (1 << sendSliceIdxList.size()) - 1;
    CCU_CHK_RET(ccu::EventWait(ctx.event, sendBit));

    // 写之后告诉对端写完了-后同步
    // 告诉toRank数据写完了
    CCU_CHK_RET(ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
    // 等待fromRank写完数据
    CCU_CHK_RET(ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
    return CcuResult::CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherNHR(AllGatherOmniPipeNHR1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    for (auto& nhrStepInfo : arg->stepInfoVector) {
        CCU_CHK_RET(DoRepeatAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAllGatherOmniPipeNHR1DMem2MemKernel(CcuKernelArg arg)
{
    HCCL_INFO("[%s] start", __func__);
    auto* kernelArg = static_cast<CcuKernelArgAllGatherOmniPipeNHR1DMem2Mem*>(arg);

    AllGatherOmniPipeNHR1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoRepeatAllGatherNHR(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

} // namespace ops_hccl
