/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh1d_one_shot.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_CKE_IDX = 0;
constexpr int PRE_SYNC_CKE_IDX = 1;
constexpr uint16_t POST_CKE_BIT0 = 0;

static CcuResult ParseKernelArg(AllReduceMesh1DOneShotContext &ctx,
    CcuKernelArgAllReduceMesh1DOneShot *kernelArg)
{
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] outputDataType is [INVALID], set outputDataType to[%u]",
            ctx.outputDataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllReduceMesh1DOneShotContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DOneShot] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    if (arg->rankSize != static_cast<uint64_t>(arg->channelCount) + 1) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DOneShot] rankSize[%llu] and channelCount[%u] mismatch.",
            arg->rankSize, arg->channelCount);
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.input.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);

    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            if (channelIdx >= arg->channelCount) {
                HCCL_ERROR("[CcuKernelAllReduceMesh1DOneShot] channelIdx[%u] >= channelCount[%u].",
                    channelIdx, arg->channelCount);
                return CcuResult::CCU_E_INTERNAL;
            }
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllReduceMesh1DOneShotContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual, argId++));
    return CCU_SUCCESS;
}

static void PreSync(AllReduceMesh1DOneShotContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            INPUT_XN_ID, PRE_SYNC_CKE_IDX, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, PRE_SYNC_CKE_IDX, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], PRE_SYNC_CKE_IDX, allBit);
    }
}

static void PostSync(AllReduceMesh1DOneShotContext &ctx)
{
    const auto *arg = ctx.arg;
    uint16_t postCkeBit = 1 << POST_CKE_BIT0;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], POST_SYNC_CKE_IDX, postCkeBit);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], POST_SYNC_CKE_IDX, postCkeBit);
    }
}

static CcuResult DoGroupReduce(AllReduceMesh1DOneShotContext &ctx)
{
    const auto *arg = ctx.arg;
    std::vector<ccu::RemoteAddr> remoteSrc(arg->channelCount);
    ccu::LocalAddr localSrc;
    ccu::LocalAddr reduceDst;

    reduceDst.addr = ctx.output;
    reduceDst.token = ctx.token[arg->rankId];
    localSrc.addr = ctx.input[arg->rankId];
    localSrc.token = ctx.token[arg->rankId];

    uint32_t srcIdx = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            continue;
        }
        if (srcIdx >= remoteSrc.size()) {
            HCCL_ERROR("[CcuKernelAllReduceMesh1DOneShot] srcIdx[%u] >= remoteSrc.size[%llu].",
                srcIdx, static_cast<unsigned long long>(remoteSrc.size()));
            return CcuResult::CCU_E_INTERNAL;
        }
        remoteSrc[srcIdx].addr = ctx.input[rankIdx];
        remoteSrc[srcIdx].token = ctx.token[rankIdx];
        srcIdx++;
    }

    CCU_CHK_RET(GroupReduce(ctx, arg->channels, arg->channelCount, reduceDst, remoteSrc, localSrc,
        ctx.groupOpSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    return CCU_SUCCESS;
}

CcuResult CcuAllReduceMesh1DOneShotKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduceMesh1DOneShot *>(arg);
    if (kernelArg == nullptr) {
        HCCL_ERROR("[CcuKernelAllReduceMesh1DOneShot] kernelArg is null.");
        return CcuResult::CCU_E_PTR;
    }

    AllReduceMesh1DOneShotContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] AllReduceMesh1DOneShot start");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);
    CCU_CHK_RET(DoGroupReduce(ctx));
    PostSync(ctx);

    HCCL_INFO("[CcuKernelAllReduceMesh1DOneShot] AllReduceMesh1DOneShot end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl
