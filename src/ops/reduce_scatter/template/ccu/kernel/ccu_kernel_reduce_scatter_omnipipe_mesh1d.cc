/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_omnipipe_mesh1d.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

constexpr int CKE_IDX_0 = 0;
constexpr int INPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;

constexpr uint64_t LOCAL_COPY_MS = 8;

static CcuResult
ParseKernelArg(ReduceScatterOmniPipeMesh1DContext& ctx, CcuKernelArgReduceScatterOmniPipeMesh1D* kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankId = kernelArg->rankId;
    ctx.rankSize = kernelArg->rankSize;
    ctx.userRank = kernelArg->subCommRanks[0][ctx.rankId];

    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG(
            "[CcuKernelReduceScatterOmniPipeMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    HCCL_INFO(
        "[CcuKernelReduceScatterOmniPipeMesh1D] userRank[%u] rankId[%u], rankSize[%u], "
        "dataType[%d], outputDataType[%d], reduceOp[%d]",
        ctx.userRank, ctx.rankId, ctx.rankSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterOmniPipeMesh1DContext& ctx)
{
    const auto* arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterOmniPipeMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源
    ctx.input.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId) {
            // 本地资源，默认构造
            continue;
        } else {
            HCCL_DEBUG(
                "[CcuKernelReduceScatterOmniPipeMesh1D] rankId[%u], peerId[%u], channelId[%u]", arg->rankId, peerId,
                channelIdx);
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }

    ctx.resourceAllocated = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterOmniPipeMesh1DContext& ctx)
{
    const auto* arg = ctx.arg;
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.offSet, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, cnt++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(ReduceScatterOmniPipeMesh1DContext& ctx)
{
    const auto* arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.input[arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    return CcuResult::CCU_SUCCESS;
}

static CcuResult PostSync(ReduceScatterOmniPipeMesh1DContext& ctx)
{
    const auto* arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_DEBUG("[%s] end", __func__);
    return CcuResult::CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatter(ReduceScatterOmniPipeMesh1DContext& ctx)
{
    const auto* arg = ctx.arg;
    HCCL_INFO("[DoRepeatReduceScatter] userRank[%u] rankId[%u] do repeat ReduceScatter", ctx.userRank, ctx.rankId);

    ccu::LocalAddr dst;
    std::vector<ccu::RemoteAddr> src;

    src.resize(arg->rankSize - 1);
    dst.addr = ctx.input[arg->rankId];
    dst.addr += ctx.inputSliceStride;
    dst.addr += ctx.inputOmniPipeSliceStride;
    dst.token = ctx.token[arg->rankId];

    // 准备源地址
    uint32_t idx = 0;
    for (auto i = 0; i < ctx.rankSize; ++i) {
        if (i == arg->rankId) {
            continue;
        }
        HCCL_DEBUG("[DoRepeatReduceScatter] myRank[%u] mySubCommRank[%u] current i[%d]", ctx.userRank, ctx.rankId, i);
        src[idx].addr = ctx.input[i];
        src[idx].addr += ctx.inputSliceStride;
        src[idx].addr += ctx.inputOmniPipeSliceStride;
        src[idx].token = ctx.token[i];
        idx++;
    }
    ccu::LocalAddr tmp;
    tmp.addr = dst.addr;
    tmp.token = dst.token;
    CCU_CHK_RET(GroupReduce(
        ctx, arg->channels, arg->channelCount, dst, src, tmp, ctx.goSize, ctx.dataType, ctx.outputDataType,
        ctx.reduceOp, GetCcuVersion()));

    HCCL_INFO(
        "[DoRepeatReduceScatter] userRank[%u] rankId[%u] do repeat ReduceScatter success", ctx.userRank, ctx.rankId);

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceScatterOmniPipeMesh1DKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<CcuKernelArgReduceScatterOmniPipeMesh1D*>(arg);

    ReduceScatterOmniPipeMesh1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceScatterOmniPipeMesh1D] ReduceScatterOmniPipeMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    PreSync(ctx);
    CCU_CHK_RET(DoRepeatReduceScatter(ctx));
    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeMesh1D] ReduceScatterOmniPipeMesh1D end");

    return CCU_SUCCESS;
}
} // namespace ops_hccl
