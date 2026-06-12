/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_mesh_1D_2die_oneshot.h"

namespace ops_hccl {
#define MAX_LOOP_NUM 2

constexpr int INPUT_XN_ID    = 0;
constexpr int TOKEN_XN_ID    = 1;
constexpr int POST_SYNC_ID   = 3;
constexpr int MISSION_SYNC_ID_0 = 4;
constexpr int MISSION_SYNC_ID_1 = 5;

constexpr int CKE_IDX_0     = 0;
constexpr int CKE_IDX_1     = 1;

constexpr int DIE_WORK = 2;

struct LocalReduceVar {
    ccu::LocalAddr loopDst[MAX_LOOP_NUM];
    std::array<std::vector<ccu::LocalAddr>, MAX_LOOP_NUM> loopSrc;
    ccu::Variable loopLen[MAX_LOOP_NUM];
    ccu::Variable loopLenExp[MAX_LOOP_NUM];
};

static CcuResult ParseKernelArg(AllreduceMesh1D2DieOneShotContext &ctx)
{
    const auto *arg = ctx.arg;

    ctx.rankId = arg->rankId;
    ctx.rankSize = arg->rankSize;
    ctx.rmtReduceWithMyRank = arg->rmtReduceWithMyRank;
    ctx.rmtReduceRankNum = arg->channelCount + (ctx.rmtReduceWithMyRank ? 1 : 0);

    ctx.rmtSyncMyBit = 1 << (ctx.rankId % ctx.rmtReduceRankNum);
    ctx.rmtSyncWaitBit = ctx.rmtReduceWithMyRank ? ((1 << ctx.rmtReduceRankNum) - 1) & (~ctx.rmtSyncMyBit) : (1 << ctx.rmtReduceRankNum) - 1;

    ctx.missionSyncMybit = 1 << (ctx.rmtReduceWithMyRank ? 1 : 0);
    ctx.missionSyncWaitBit = 1 << (!ctx.rmtReduceWithMyRank ? 1 : 0);

    ctx.dataType = arg->opParam.DataDes.dataType;
    ctx.outputDataType = arg->opParam.DataDes.outputType;

    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelAllreduceMesh1D2DieOneShot] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp = arg->opParam.reduceType;

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Init, KernelArgs are rankId[%u], rankSize[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d], rmtReduceWithMyRank[%d], rmtReduceRankNum[%d]",
        ctx.rankId, ctx.rankSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp, ctx.rmtReduceWithMyRank, ctx.rmtReduceRankNum);

    return CCU_SUCCESS;
}

static CcuResult InitResource(AllreduceMesh1D2DieOneShotContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllreduceMesh1D2DieOneShot] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.input.resize(arg->channelCount);
    ctx.remoteToken.resize(arg->channelCount);
    for (size_t i = 0; i < arg->channelCount; i++) {
        ctx.input[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], INPUT_XN_ID);
        ctx.remoteToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], TOKEN_XN_ID);
    }

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] InitResources finished");
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllreduceMesh1D2DieOneShotContext &ctx)
{
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.myInput, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myOutput, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myScratch, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchBaseOffset0, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchBaseOffset1, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceSliceOffset0, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceSliceOffset1, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.rmtReduceGoSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rmtReduceGoSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rmtReduceGoSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rmtReduceGoSize.residual, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize0.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize0.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize0.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize0.residual, argId++));

    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize1.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize1.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize1.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localReduceGoSize1.residual, argId++));

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] LoadArgs run finished");
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllreduceMesh1D2DieOneShotContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.myInput, INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.myToken, TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] PreSync run finished");
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllreduceMesh1D2DieOneShotContext &ctx, uint32_t signalIndex)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_1, 1 << signalIndex);
    }

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_1, 1 << signalIndex);
    }
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] PostSync run finished");
    return CCU_SUCCESS;
}

static CcuResult CreateReduceLoop(AllreduceMesh1D2DieOneShotContext &ctx,
    LocalReduceVar &var, uint32_t size, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType)
{
    AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated);

    const std::string loopType = "allreduce_mesh_1d_2die_oneshot_local_reduce";

    if (ctx.IsLoopEntityRegistered(loopType)) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity(loopType);
    auto &loops = ctx.loopMap[loopType];

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    uint32_t usedBufNum = size > expansionNum ? size : expansionNum;

    for (int32_t index = 0; index < MAX_LOOP_NUM; index++) {
        var.loopSrc[index].resize(size);
        uint32_t bufBase = index * ctx.moConfig.msInterleave;
        ccu::Event e = ctx.moRes.completedEvent[index];

        loops.body[index].reset(new ccu::Func(
            [&ctx, index, bufBase, e, size, &var, dataType, outputDataType, opType]() {
            for (uint32_t i = 0; i < size; i++) {
                ccu::LocalCopy(ctx.moRes.ccuBuf[bufBase + i], var.loopSrc[index][i], var.loopLen[index], e, 1 << i);
            }
            ccu::EventWait(e, (1 << size) - 1);

            if (size > 1) {
                ccu::LocalReduce(&ctx.moRes.ccuBuf[bufBase], size, dataType, outputDataType, opType, var.loopLen[index], e, 1);
                ccu::EventWait(e, 1);
            }

            ccu::LocalCopy(var.loopDst[index], ctx.moRes.ccuBuf[bufBase], var.loopLenExp[index], e, 1);
            ccu::EventWait(e, 1);
        }));

        loops.loops[index].reset(new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }

    return CCU_SUCCESS;
}

static CcuResult ReduceLoopGroup(AllreduceMesh1D2DieOneShotContext &ctx,
    ccu::LocalAddr &outDstOrg, std::vector<ccu::LocalAddr> &srcOrg,
    GroupOpSizeVars goSize, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType)
{
    const uint32_t size = srcOrg.size();

    ccu::LocalAddr dst;
    dst.addr = outDstOrg.addr;
    dst.token = outDstOrg.token;

    std::vector<ccu::LocalAddr> src;
    src.resize(size);
    for (uint32_t idx = 0; idx < size; idx++) {
        src[idx].addr = srcOrg[idx].addr;
        src[idx].token = srcOrg[idx].token;
    }

    LocalReduceVar var;
    CCU_CHK_RET(CreateReduceLoop(ctx, var, size, dataType, outputDataType, opType));

    const std::string loopType = "allreduce_mesh_1d_2die_oneshot_local_reduce";
    auto &loops = ctx.loopMap[loopType];

    uint32_t expansionNum = GetReduceExpansionNum(opType, dataType, outputDataType);
    ccu::Variable sliceSizeExpansion;

    if (expansionNum != 1) {
        ccu::Variable tmp;
        tmp = GetExpansionParam(expansionNum);
        dst.token = dst.token + tmp;
    }

    CCU_IF(goSize.loopParam != 0) {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + goSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        for (uint32_t i = 0; i < size; ++i) {
            var.loopSrc[0][i].addr = src[i].addr;
            var.loopSrc[0][i].token = src[i].token;
        }
        var.loopDst[0].addr = dst.addr;
        var.loopDst[0].token = dst.token;
        var.loopLen[0] = sliceSize;
        var.loopLenExp[0] = sliceSizeExpansion;

        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopParam;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
        ccu::LoopGroup group(paraCfg, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    CCU_IF(goSize.parallelParam != 0) {
        for (uint32_t i = 0; i < size; i++) {
            src[i].addr = src[i].addr + goSize.addrOffset;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr = dst.addr + goSize.addrOffset;
        }

        sliceSizeExpansion = 0;
        for (uint32_t i = 0; i < expansionNum; i++) {
            sliceSizeExpansion = sliceSizeExpansion + goSize.residual;
        }

        for (uint32_t i = 0; i < size; ++i) {
            var.loopSrc[0][i].addr = src[i].addr;
            var.loopSrc[0][i].token = src[i].token;
        }
        var.loopDst[0].addr = dst.addr;
        var.loopDst[0].token = dst.token;
        var.loopLen[0] = goSize.residual;
        var.loopLenExp[0] = sliceSizeExpansion;

        for (uint32_t i = 0; i < size; i++) {
            src[i].addr = src[i].addr + goSize.residual;
        }
        for (uint32_t i = 0; i < expansionNum; i++) {
            dst.addr = dst.addr + goSize.residual;
        }

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;
        sliceSizeExpansion = ctx.moConfig.memSlice * expansionNum;

        for (uint32_t i = 0; i < size; ++i) {
            var.loopSrc[1][i].addr = src[i].addr;
            var.loopSrc[1][i].token = src[i].token;
        }
        var.loopDst[1].addr = dst.addr;
        var.loopDst[1].token = dst.token;
        var.loopLen[1] = sliceSize;
        var.loopLenExp[1] = sliceSizeExpansion;

        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);
        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

        loops.loopParam[0] = loopCfg0;
        loops.loopParam[1] = loopCfg1;
        std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
        ccu::LoopGroup group(goSize.parallelParam, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    return CCU_SUCCESS;
}

static CcuResult DoRmtReduce(AllreduceMesh1D2DieOneShotContext &ctx)
{
    const auto *arg = ctx.arg;

    std::vector<ccu::RemoteAddr> src;
    src.resize(ctx.rmtReduceRankNum);
    for (uint32_t peerIdx = 0; peerIdx < arg->channelCount; peerIdx++) {
        src[peerIdx].token = ctx.remoteToken[peerIdx];
        src[peerIdx].addr = ctx.input[peerIdx];
    }
    ccu::LocalAddr localSrc;
    if (ctx.rmtReduceWithMyRank) {
        localSrc.token = ctx.myToken;
        localSrc.addr = ctx.myInput;
    }

    ccu::LocalAddr dst;
    dst.token = ctx.myToken;
    dst.addr = ctx.myScratch;
    dst.addr = dst.addr + (ctx.rmtReduceWithMyRank ? ctx.scratchBaseOffset0 : ctx.scratchBaseOffset1);

    if (ctx.rmtReduceWithMyRank) {
        CCU_CHK_RET(GroupReduce(ctx, arg->channels, arg->channelCount, dst, src, localSrc, ctx.rmtReduceGoSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    } else {
        CCU_CHK_RET(GroupReduceWithoutMyRank(ctx, arg->channels, arg->channelCount, dst, src, ctx.rmtReduceGoSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    }

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Step1 RmtReduce run finished");
    return CCU_SUCCESS;
}

static CcuResult DoLocalReduce(AllreduceMesh1D2DieOneShotContext &ctx)
{
    std::vector<ccu::LocalAddr> src;
    src.resize(DIE_WORK);
    for (uint32_t i = 0; i < DIE_WORK; i++) {
        src[i].token = ctx.myToken;
        src[i].addr = ctx.myScratch;
        src[i].addr = src[i].addr + (i == 0 ? ctx.scratchBaseOffset0 : ctx.scratchBaseOffset1);
        src[i].addr = src[i].addr + (ctx.rmtReduceWithMyRank ? ctx.localReduceSliceOffset0 : ctx.localReduceSliceOffset1);
    }

    ccu::LocalAddr dst;
    dst.token = ctx.myToken;
    dst.addr = ctx.myOutput;
    dst.addr = dst.addr + (ctx.rmtReduceWithMyRank ? ctx.localReduceSliceOffset0 : ctx.localReduceSliceOffset1);

    if (ctx.rmtReduceWithMyRank) {
        CCU_IF(ctx.localReduceSliceOffset1 != 0) {
            CCU_CHK_RET(ReduceLoopGroup(ctx, dst, src, ctx.localReduceGoSize0, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
        }
    } else {
        CCU_CHK_RET(ReduceLoopGroup(ctx, dst, src, ctx.localReduceGoSize1, ctx.dataType, ctx.outputDataType, ctx.reduceOp));
    }

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] DoLocalReduce run finished");
    return CCU_SUCCESS;
}

static CcuResult MissionSync(AllreduceMesh1D2DieOneShotContext &ctx, uint32_t maskIndex)
{
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] MissionSync, missionSyncMybit[%u], missionSyncWaitBit[%u]",
        ctx.missionSyncMybit, ctx.missionSyncWaitBit);
    uint32_t coreIdx = ctx.rmtReduceWithMyRank ? 1 : 0;
    uint16_t mask = static_cast<uint16_t>(ctx.missionSyncMybit << (DIE_WORK * maskIndex));
    ccu::EventRecord((coreIdx == 0) ? "core1_mission_sync" : "core0_mission_sync", mask);
    mask = static_cast<uint16_t>(ctx.missionSyncWaitBit << (DIE_WORK * maskIndex));
    ccu::EventWait((coreIdx == 0) ? "core0_mission_sync" : "core1_mission_sync", mask);
    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] MissionSync run finished");
    return CCU_SUCCESS;
}

CcuResult CcuAllreduceMesh1D2DieOneShotKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllreduceMesh1D2DieOneShot *>(arg);

    AllreduceMesh1D2DieOneShotContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] AllreduceMesh1D2DieOneShot run");

    CCU_CHK_RET(ParseKernelArg(ctx));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Algorithm second step rmtReduce begins.");

    CCU_CHK_RET(DoRmtReduce(ctx));

    CCU_CHK_RET(PostSync(ctx, POST_SYNC_ID));

    CCU_CHK_RET(MissionSync(ctx, 0));

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] Algorithm second step localreduce begins.");

    CCU_CHK_RET(DoLocalReduce(ctx));

    CCU_CHK_RET(MissionSync(ctx, 1));

    HCCL_INFO("[CcuKernelAllreduceMesh1D2DieOneShot] AllreduceMesh1D2Die end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
