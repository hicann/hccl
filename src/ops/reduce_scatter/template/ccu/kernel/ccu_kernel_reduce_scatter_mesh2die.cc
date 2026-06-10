/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_mesh2die.h"

namespace ops_hccl {

constexpr int CKE_IDX_0   = 0;
constexpr int INPUT_XN_ID = 0;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int LOOP_NUM    = 128;

constexpr int MISSION_NUM = 2;

static CcuResult InitResources(ReduceScatterMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.peerInput.resize(arg->channelCount);
    ctx.peerToken.resize(arg->channelCount);
    for (uint64_t id = 0; id < arg->channelCount; id++) {
        ctx.peerInput[id] = ccu::GetResByChannel<ccu::Variable>(arg->channels[id], INPUT_XN_ID);
        ctx.peerToken[id] = ccu::GetResByChannel<ccu::Variable>(arg->channels[id], TOKEN_XN_ID);
    }

    ctx.rmtReduceRankNum = arg->channelCount + (ctx.rmtReduceWithMyRank == true ? 1 : 0);
    ctx.rmtSyncMyBit = 1 << (ctx.myRankId % ctx.rmtReduceRankNum);
    ctx.rmtSyncWaitBit = ctx.rmtReduceWithMyRank ? ((1 << ctx.rmtReduceRankNum) - 1) & (~ctx.rmtSyncMyBit) 
                                                  : (1 << ctx.rmtReduceRankNum) - 1;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterMesh2DieContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.myInput, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myOutput, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myScratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rmtReduceSliceOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rmtReduceGoSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rmtReduceGoSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rmtReduceGoSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rmtReduceGoSize.residual, argId++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(ReduceScatterMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.myInput, INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.myToken, TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t waitBits = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, waitBits);
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(ReduceScatterMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult RmtReduce(ReduceScatterMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    std::vector<ccu::RemoteAddr> src;
    src.resize(arg->channelCount);
    for (uint32_t peerIdx = 0; peerIdx < arg->channelCount; peerIdx++) {
        ccu::RemoteAddr addr;
        addr.token = ctx.peerToken[peerIdx];
        addr.addr = ctx.peerInput[peerIdx];
        addr.addr += ctx.rmtReduceSliceOffset;
        src[peerIdx] = addr;
    }
    ccu::LocalAddr localSrc;
    if (ctx.rmtReduceWithMyRank) {
        localSrc.token = ctx.myToken;
        localSrc.addr = ctx.myInput;
        localSrc.addr += ctx.rmtReduceSliceOffset;
    }

    ccu::LocalAddr dst;
    dst.token = ctx.myToken;
    dst.addr = ctx.rmtReduceWithMyRank ? ctx.myOutput : ctx.myScratch;

    if (ctx.rmtReduceWithMyRank) {
        CCU_CHK_RET(GroupReduce(ctx, arg->channels, arg->channelCount, dst, src, localSrc, ctx.rmtReduceGoSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    } else {
        CCU_CHK_RET(GroupReduceWithoutMyRank(ctx, arg->channels, arg->channelCount, dst, src, ctx.rmtReduceGoSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMesh2DieKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterMesh2Die *>(arg);
    
    ReduceScatterMesh2DieContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    HCCL_INFO("[ccuReduceScatterMesh2Die_kernel] ReduceScatterMesh2Die run.");

    ctx.rmtReduceWithMyRank = kernelArg->rmtReduceWithMyRank;
    ctx.myRankId = kernelArg->rankId;
    ctx.rankSize = kernelArg->rankSize;

    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterMesh2Die] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;

    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(RmtReduce(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[ccuReduceScatterMesh2Die_kernel] ReduceScatterMesh2Die end.");
    return CCU_SUCCESS;
}

}// namespace ops_hccl