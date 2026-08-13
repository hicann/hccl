/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_omnipipe_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

constexpr uint16_t BIT_NUM_PER_CKE = 16;

static CcuResult
ParseKernelArg(AllGatherOmniPipeMesh1DMem2MemContext& ctx, CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem* kernelArg)
{
    ctx.arg = kernelArg;
    if (kernelArg->subCommRanks.empty() || kernelArg->subCommRanks[0].empty()
        || kernelArg->rankId >= kernelArg->subCommRanks[0].size()) {
        HCCL_ERROR("[%s] invalid subCommRanks or rankId[%u] out of range.", __func__, kernelArg->rankId);
        return CcuResult::CCU_E_INTERNAL;
    }
    kernelArg->userRank = kernelArg->subCommRanks[0][kernelArg->rankId];

    HCCL_INFO(
        "[%s] myRank[%u] rankId[%u] rankSize[%u]", __func__, kernelArg->userRank, kernelArg->rankId,
        kernelArg->rankSize);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherOmniPipeMesh1DMem2MemContext& ctx)
{
    HCCL_INFO("[%s] start", __func__);
    const auto* arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[%s] channels is empty!", __func__);
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[%s] channels.size: [%u]", __func__, arg->channelCount);

    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    const uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    ctx.events.resize(eventNum);

    ctx.resourceAllocated = false;
    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherOmniPipeMesh1DMem2MemContext& ctx)
{
    HCCL_INFO("[%s] start", __func__);
    const auto* arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual, argId++));

    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherOmniPipeMesh1DMem2MemContext& ctx)
{
    HCCL_INFO("[%s] start", __func__);
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.output[arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult PostSync(AllGatherOmniPipeMesh1DMem2MemContext& ctx)
{
    HCCL_INFO("[%s] start", __func__);
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult DoAllGather(AllGatherOmniPipeMesh1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    ccu::LocalAddr src;
    std::vector<ccu::RemoteAddr> dst;
    dst.resize(arg->rankSize);

    src.addr = ctx.output[arg->rankId];
    src.addr += ctx.sliceStride;
    src.addr += ctx.inputOmniPipeSliceStride;
    src.token = ctx.token[arg->rankId];

    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            continue;
        }
        dst[peerId].addr = ctx.output[peerId];
        dst[peerId].addr += ctx.sliceStride;
        dst[peerId].addr += ctx.inputOmniPipeSliceStride;
        dst[peerId].token = ctx.token[peerId];
    }

    uint16_t channelId = 0;
    const uint32_t eventNum = (arg->rankSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        const uint16_t eventIdx = peerId / BIT_NUM_PER_CKE;
        const uint16_t rankMask = 1 << (peerId % BIT_NUM_PER_CKE);
        if (peerId == arg->rankId) {
            CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask));
        } else {
            CCU_IF(ctx.sliceSize != 0)
            {
                ccu::Write(arg->channels[channelId], dst[peerId], src, ctx.sliceSize, ctx.events[eventIdx], rankMask);
            }
            CCU_IF(ctx.sliceSize == 0) { CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask)); }
            channelId++;
        }
    }
    for (uint32_t i = 0; i < eventNum; i++) {
        uint16_t eventMask;
        if (i == eventNum - 1) {
            if (arg->rankSize % BIT_NUM_PER_CKE == 0) {
                eventMask = (1 << BIT_NUM_PER_CKE) - 1;
            } else {
                eventMask = (1 << (arg->rankSize % BIT_NUM_PER_CKE)) - 1;
            }
        } else {
            eventMask = (1 << BIT_NUM_PER_CKE) - 1;
        }
        CCU_CHK_RET(ccu::EventWait(ctx.events[i], eventMask));
    }
    return CcuResult::CCU_SUCCESS;
}

static CcuResult DoRepeatAllGather(AllGatherOmniPipeMesh1DMem2MemContext& ctx)
{
    HCCL_INFO("[%s] start", __func__);
    const auto* arg = ctx.arg;
    // 本地拷贝
    CCU_IF(ctx.localCopyFlag == 1)
    {
        HCCL_INFO("[%s] rankIdx[%u] userRank[%u] localcopy begin", __func__, arg->rankId, arg->userRank);
        ccu::LocalAddr myOutput;
        ccu::LocalAddr myInput;

        ccu::Variable outSliceStride;
        outSliceStride = 0;
        for (uint32_t i = 0; i < arg->userRank; i++) {
            outSliceStride += ctx.sliceStride;
        }

        myInput.addr = ctx.input;
        myInput.token = ctx.token[arg->rankId];
        myOutput.addr = ctx.output[arg->rankId];
        myOutput.addr += outSliceStride;
        myOutput.token = ctx.token[arg->rankId];

        uint16_t mask = 1 << arg->rankId;
        CCU_IF(ctx.sliceSize != 0) { CCU_CHK_RET(GroupCopy(ctx, myOutput, myInput, ctx.groupOpSize, GetCcuVersion())); }
        HCCL_INFO("[%s] rankId[%u] userRank[%u] localcopy end", __func__, arg->rankId, arg->userRank);
    }

    CCU_IF(ctx.localCopyFlag == 0) { CCU_CHK_RET(DoAllGather(ctx)); }
    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuAllGatherOmniPipeMesh1DMem2MemKernel(CcuKernelArg arg)
{
    HCCL_INFO("[%s] start", __func__);
    auto* kernelArg = static_cast<CcuKernelArgAllGatherOmniPipeMesh1DMem2Mem*>(arg);

    AllGatherOmniPipeMesh1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoRepeatAllGather(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

} // namespace ops_hccl
