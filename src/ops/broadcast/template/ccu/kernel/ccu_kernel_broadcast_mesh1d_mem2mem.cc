/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_broadcast_mesh1d_mem2mem.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID  = 0;
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0;
constexpr int CKE_IDX_3    = 3;
constexpr int CKE_IDX_4    = 4;

static CcuResult ParseKernelArg(BroadcastMesh1DMem2MemContext &ctx, CcuKernelArgBroadcastMesh1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    return CCU_SUCCESS;
}

static CcuResult InitResource(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelBroadcastMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] channels.size: [%u]", arg->channelCount);

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

    ctx.scattersrcMem.resize(arg->rankSize);
    ctx.scatterdstMem.resize(arg->rankSize);
    ctx.allgatherdstMem.resize(arg->rankSize);

    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.allgatherOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, argId++));

    return CCU_SUCCESS;
}

static CcuResult PreSync(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId],
            INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(BroadcastMesh1DMem2MemContext &ctx, int ckeId)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << ckeId));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << ckeId));
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatter(BroadcastMesh1DMem2MemContext &ctx, std::vector<ccu::RemoteAddr> &dst)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        auto &sliceSize = (rankIdx + 1 == arg->rankSize) ? ctx.lastSliceSize : ctx.normalSliceSize;
        const uint16_t rankMask = 1 << rankIdx;
        CCU_IF(sliceSize != 0)
        {
            if (rankIdx == arg->rankId) {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
            } else {
                CCU_CHK_RET(ccu::Write(arg->channels[channelId], dst[rankIdx],
                    ctx.scattersrcMem[rankIdx], sliceSize, ctx.event, rankMask));
                HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem][DoScatter] channelsId[%u] rankIdx[%u]",
                          channelId, rankIdx);
                channelId++;
            }
        }
        CCU_IF(sliceSize == 0)
        {
            CCU_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
        }
    }

    const uint16_t allRankMask = (1 << arg->rankSize) - 1;
    CCU_CHK_RET(ccu::EventWait(ctx.event, allRankMask));
    return CCU_SUCCESS;
}

static CcuResult DoRepeaScatterMem2Mem(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId != arg->rootId) {
        return CCU_SUCCESS;
    }
    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem][DoRepeaScatterMem2Mem] rankId[%u] rankSize[%llu]",
              arg->rankId, arg->rankSize);

    std::vector<ccu::RemoteAddr> &dst = ctx.scatterdstMem;
    ccu::Variable sliceOffset;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == 0) {
            sliceOffset = 0;
        } else {
            sliceOffset += ctx.normalSliceSize;
        }
        ctx.scattersrcMem[rankIdx].addr = ctx.input[arg->rankId];
        ctx.scattersrcMem[rankIdx].addr += ctx.currentRankSliceInputOffset;
        ctx.scattersrcMem[rankIdx].addr += sliceOffset;
        ctx.scattersrcMem[rankIdx].token = ctx.token[arg->rankId];
        if (rankIdx == arg->rankId) {
            ctx.myScatterDst.addr = ctx.output[rankIdx];
            ctx.myScatterDst.addr += ctx.currentRankSliceOutputOffset;
            ctx.myScatterDst.addr += sliceOffset;
            ctx.myScatterDst.token = ctx.token[rankIdx];
        } else {
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            dst[rankIdx].addr += sliceOffset;
            dst[rankIdx].token = ctx.token[rankIdx];
        }

        CCU_IF(ctx.flag == 1)
        {
            // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
            ctx.myScatterDst.addr += ctx.outputRepeatStride;
            for (uint32_t curId = 0; curId < arg->rankSize; curId++) {
                ctx.scattersrcMem[curId].addr += ctx.inputRepeatStride;
                if (curId != arg->rankId) {
                    dst[curId].addr += ctx.outputRepeatStride;
                }
            }
        }
    }
    CCU_CHK_RET(DoScatter(ctx, dst));
    return CCU_SUCCESS;
}

static CcuResult DoAllGather(BroadcastMesh1DMem2MemContext &ctx, const ccu::LocalAddr &src,
                             const std::vector<ccu::RemoteAddr> &dst)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;
    auto &sliceSize = (arg->rankId + 1 == arg->rankSize) ? ctx.lastSliceSize : ctx.normalSliceSize;
    CCU_IF(sliceSize != 0)
    {
        for (uint64_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
            const uint16_t rankMask = 1 << rankIdx;
            if (rankIdx == arg->rankId) {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
            } else {
                CCU_CHK_RET(ccu::Write(arg->channels[channelId], dst[rankIdx],
                    src, sliceSize, ctx.event, rankMask));
                channelId++;
            }
        }
        const uint16_t allRankMask = (1 << arg->rankSize) - 1;
        CCU_CHK_RET(ccu::EventWait(ctx.event, allRankMask));
    }
    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherMem2Mem(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr &src = ctx.allgatherSrc;
    std::vector<ccu::RemoteAddr> &dst = ctx.allgatherdstMem;
    src.addr = ctx.output[arg->rankId];
    src.addr += ctx.currentRankSliceOutputOffset;
    src.addr += ctx.allgatherOffset;
    src.token = ctx.token[arg->rankId];

    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            dst[rankIdx].addr = ctx.output[rankIdx];
            dst[rankIdx].addr += ctx.currentRankSliceOutputOffset;
            dst[rankIdx].addr += ctx.allgatherOffset;
            dst[rankIdx].token = ctx.token[rankIdx];
        }
    }
    CCU_IF(ctx.flag == 1)
    {
        // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        src.addr += ctx.inputRepeatStride;
        for (uint32_t curId = 0; curId < arg->rankSize; curId++) {
            if (curId != arg->rankId) {
                dst[curId].addr += ctx.outputRepeatStride;
            }
        }
    }

    CCU_CHK_RET(DoAllGather(ctx, src, dst));
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcastMesh1DMem2Mem *>(arg);

    BroadcastMesh1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] BroadcastMesh1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    ccu::Variable repeatNumAdd;
    repeatNumAdd = 1;
    ctx.flag = 0;
    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX)
    {
        // 循环repeatNum次
        CCU_CHK_RET(DoRepeaScatterMem2Mem(ctx));
        CCU_CHK_RET(PostSync(ctx, CKE_IDX_3));
        CCU_CHK_RET(DoRepeatAllGatherMem2Mem(ctx));
        CCU_CHK_RET(PostSync(ctx, CKE_IDX_4));
        ctx.repeatNumVar += repeatNumAdd;
        ctx.flag = 1;
    }

    HCCL_INFO("[CcuKernelBroadcastMesh1DMem2Mem] BroadcastMesh1DMem2Mem end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl
