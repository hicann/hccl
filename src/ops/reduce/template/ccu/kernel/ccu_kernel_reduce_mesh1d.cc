/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh1d.h"
#include "ccu_launch_dl.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int POST_SYNC_ID  = 3;
constexpr int CKE_IDX_0    = 0;

static CcuResult ParseKernelArg(ReduceMesh1DContext &ctx, CcuKernelArgReduceMesh1D *kernelArg)

{
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;

        HCCL_DEBUG("[CcuKernelReduceMesh1D] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelReduceMesh1D] channels.size: [%u]", arg->channelCount);

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.input.resize(arg->rankSize);
    ctx.output.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
			ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
             channelIdx++;
        }
    }

    ctx.resourceAllocated = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
	uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual, argId++));
    return CCU_SUCCESS;
}

static void PreSync(ReduceMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D PreSync begin");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D PreSync end");
}

static void PostSync(ReduceMesh1DContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D post sync start");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D post sync end");
}

static CcuResult DoRepeatReduce(ReduceMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    std::vector<ccu::RemoteAddr> remoteSrc(arg->rankSize);

    ccu::LocalAddr localSrc;
    ccu::LocalAddr dst;

    dst.addr = ctx.output[arg->rankId];
    dst.token = ctx.token[arg->rankId];
    uint32_t curId = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rootId) {
            remoteSrc[curId].addr  = ctx.input[rankIdx]; // GSA[400] + Xn[400] to GSA[0]
            remoteSrc[curId].token = ctx.token[rankIdx];
            curId++;
        } else {
            continue;
        }
    }
    localSrc.addr = ctx.input[arg->rankId];
    localSrc.token = ctx.token[arg->rankId];

    CCU_IF (ctx.flag != 0) {
        // 非第一轮执行时，remoteSrc 和 dst 已经初始化，需要添加偏移量
        dst.addr += ctx.outputRepeatStride;
        for (auto &s : remoteSrc) {
            s.addr += ctx.inputRepeatStride;
        }
    }
    GroupReduce(ctx, arg->channels, arg->channelCount, dst, remoteSrc, localSrc,
                ctx.groupOpSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);

    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceMesh1D *>(arg);

    ReduceMesh1DContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    if (kernelArg->rankId == kernelArg->rootId) {
        ccu::Variable repeatNumAdd;
        repeatNumAdd  = 1;
        ctx.flag = 0;
        CCU_WHILE(ctx.repeatNumVar != UINT64_MAX) { // 循环repeatNum_次
            CCU_CHK_RET(DoRepeatReduce(ctx));
            ctx.repeatNumVar += repeatNumAdd;
            ctx.flag = 1;
        }
    }

    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceMesh1D] ReduceMesh1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl