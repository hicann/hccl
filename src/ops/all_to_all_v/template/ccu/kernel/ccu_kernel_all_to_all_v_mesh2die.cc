/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_alg_base.h"
#include "ccu_kernel_all_to_all_v_mesh2die.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0;
constexpr int CKE_IDX_1    = 1;
constexpr int CKE_IDX_2    = 2;

static CcuResult ParseKernelArg(AllToAllVMesh2DieContext &ctx, CcuKernelArgAllToAllVMesh2Die *kernelArg)
{
    ctx.localSize = kernelArg->channelCount + 1;
    ctx.localId = ctx.localSize - 1;
    ctx.peerSize = kernelArg->channelCount + ( kernelArg->withMyRank ? 1 : 0);
    ctx.logicId =  kernelArg->rankId % ctx.peerSize;

    ctx.selfBit = 1 << ctx.logicId;
    ctx.allBit = ((1 << ctx.peerSize) - 1) & (~(kernelArg->withMyRank ? ctx.selfBit : 0));

    HCCL_INFO("[CcuKernelAllToAllVMesh2Die] RankId[%u], localSize[%u], peerSize[%u], withMyRank[%u]",
        kernelArg->rankId, ctx.localSize, ctx.peerSize, kernelArg->withMyRank);

    return CCU_SUCCESS;
}

static CcuResult InitResource(AllToAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllToAllVMesh2Die] RankId[%u] channels is empty!", arg->rankId);
        return CcuResult::CCU_E_INTERNAL;
    }

    CCU_CHK_RET(AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated,
        CCU_MS_LOCAL_COPY_LOOP_COUNT, LOCAL_COPY_MS_PER_LOOP));

    ctx.output.resize(arg->channelCount + 1);
    ctx.token.resize(arg->channelCount + 1);
    for (uint32_t peerId = 0; peerId < arg->channelCount; peerId++) {
        HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] RankId[%u], PeerId[%u]", arg->rankId, peerId);
        ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[peerId], OUTPUT_XN_ID);
        ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[peerId], TOKEN_XN_ID);
    }

    ctx.sendRecvInfo.resize(ctx.peerSize);

    ctx.src.resize(arg->channelCount);
    ctx.dst.resize(arg->channelCount);

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllToAllVMesh2DieContext &ctx)
{
    uint16_t index = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, index++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.localId], index++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.localId], index++));

    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.addrOffset, index++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.loopParam, index++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.parallelParam, index++));
    CCU_CHK_RET(ccu::LoadArg(ctx.xnMaxTransportGoSize.residual, index++));

    for (uint64_t peerId = 0; peerId < ctx.peerSize; peerId++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendOffset, index++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].recvOffset, index++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailSize, index++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailGoSize.addrOffset, index++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailGoSize.loopParam, index++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailGoSize.parallelParam, index++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendTailGoSize.residual, index++));
        CCU_CHK_RET(ccu::LoadArg(ctx.sendRecvInfo[peerId].sendLoopNum, index++));
    }

    return CCU_SUCCESS;
}

static CcuResult ExchangeInfoSync(AllToAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable tempDst;
    for (u32 peerId = 0; peerId < arg->channelCount; peerId++) {
        tempDst = ctx.output[ctx.localId];
        tempDst += ctx.sendRecvInfo[peerId].recvOffset;
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[peerId], tempDst, OUTPUT_XN_ID, CKE_IDX_1, ctx.selfBit));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[peerId], ctx.token[ctx.localId], TOKEN_XN_ID, CKE_IDX_2, ctx.selfBit));
    }
    uint32_t channelIdx = 0;
    for (u32 peerId = 0; peerId < ctx.peerSize; peerId++) {
        if (arg->withMyRank && (peerId == ctx.logicId)) {
            continue;
        }
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX_1, 1 << peerId));
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX_2, 1 << peerId));
        channelIdx++;
    }

    return CCU_SUCCESS;
}

static CcuResult PostSync(AllToAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, ctx.selfBit));
    }
    uint32_t channelIdx = 0;
    for (u32 peerId = 0; peerId < ctx.peerSize; peerId++) {
        if (arg->withMyRank && (peerId == ctx.logicId)) {
            continue;
        }
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX_0, 1 << peerId));
        channelIdx++;
    }
    return CCU_SUCCESS;
}

static CcuResult WriteToDstOutput(AllToAllVMesh2DieContext &ctx, uint32_t peerId)
{
    const auto *arg = ctx.arg;
    HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] WriteToDstOutput Start. RankId[%u] peerId[%u]", arg->rankId, peerId);

    CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum == UINT64_MAX)
    {
        CCU_CHK_RET(ccu::EventRecord(ctx.event, 1 << peerId));
    }

    CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum != UINT64_MAX)
    {
        CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum == UINT64_MAX - 1)
        {
            ctx.curSendTailSize = ctx.sendRecvInfo[peerId].sendTailSize;
            CCU_IF(ctx.curSendTailSize == 0)
            {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, 1 << peerId));
            }
            CCU_IF(ctx.curSendTailSize != 0)
            {
                CCU_CHK_RET(ccu::Write(arg->channels[peerId], ctx.dst[peerId], ctx.src[peerId],
                    ctx.curSendTailSize, ctx.event, 1 << peerId));
            }
            ctx.completedRankCount += ctx.xnConst1;
        }
        CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum != UINT64_MAX - 1)
        {
            CCU_CHK_RET(ccu::Write(arg->channels[peerId], ctx.dst[peerId], ctx.src[peerId],
                ctx.xnMaxTransportSize, ctx.event, 1 << peerId));
            ctx.dst[peerId].addr += ctx.xnMaxTransportSize;
            ctx.src[peerId].addr += ctx.xnMaxTransportSize;
        }
        ctx.sendRecvInfo[peerId].sendLoopNum += ctx.xnConst1;
    }

    HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] WriteToDstOutput end. RankId[%u] peerId[%u]", arg->rankId, peerId);
    return CCU_SUCCESS;
}

static CcuResult GroupCopyToDstOutput(AllToAllVMesh2DieContext &ctx, uint32_t peerId)
{
    const auto *arg = ctx.arg;
    HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] GroupCopyToDstOutput Start. RankId[%u] peerId[%u]", arg->rankId, peerId);

    CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum == UINT64_MAX)
    {
        CCU_CHK_RET(ccu::EventRecord(ctx.event, 1 << peerId));
    }

    CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum != UINT64_MAX)
    {
        CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum == UINT64_MAX - 1)
        {
            ctx.curSendTailSize = ctx.sendRecvInfo[peerId].sendTailSize;
            ctx.curSendTailGoSize = ctx.sendRecvInfo[peerId].sendTailGoSize;
            CCU_IF(ctx.curSendTailSize == 0)
            {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, 1 << peerId));
            }
            CCU_IF(ctx.curSendTailSize != 0)
            {
                GroupCopy(ctx, ctx.localDst, ctx.localSrc, ctx.curSendTailGoSize);
                CCU_CHK_RET(ccu::EventRecord(ctx.event, 1 << peerId));
            }
            ctx.completedRankCount += ctx.xnConst1;
        }
        CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum != UINT64_MAX - 1)
        {
            GroupCopy(ctx, ctx.localDst, ctx.localSrc, ctx.xnMaxTransportGoSize);
            CCU_CHK_RET(ccu::EventRecord(ctx.event, 1 << peerId));
            ctx.localDst.addr += ctx.xnMaxTransportSize;
            ctx.localSrc.addr += ctx.xnMaxTransportSize;
        }
        ctx.sendRecvInfo[peerId].sendLoopNum += ctx.xnConst1;
    }

    HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] GroupCopyToDstOutput end. RankId[%u] peerId[%u]", arg->rankId, peerId);
    return CCU_SUCCESS;
}

static void CalcGroupSrcDst(AllToAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t peerId = 0; peerId < arg->channelCount; peerId++) {
        ctx.src[peerId].addr = ctx.input;
        ctx.src[peerId].addr += ctx.sendRecvInfo[peerId].sendOffset;
        ctx.src[peerId].token = ctx.token[peerId];
        ctx.dst[peerId].addr = ctx.output[peerId];
        ctx.dst[peerId].token = ctx.token[peerId];
    }

    if (arg->withMyRank) {
        ctx.localSrc.addr = ctx.input;
        ctx.localSrc.addr += ctx.sendRecvInfo[ctx.localId].sendOffset;
        ctx.localSrc.token = ctx.token[ctx.localId];
        ctx.localDst.addr = ctx.output[ctx.localId];
        ctx.localDst.addr += ctx.sendRecvInfo[ctx.localId].recvOffset;
        ctx.localDst.token = ctx.token[ctx.localId];
    }
}

static CcuResult LoopStep(AllToAllVMesh2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t peerId = 0; peerId < arg->channelCount; peerId++) {
        CCU_CHK_RET(WriteToDstOutput(ctx, peerId));
    }

    if (arg->withMyRank) {
        CCU_CHK_RET(GroupCopyToDstOutput(ctx, ctx.localId));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.event, (1 << ctx.peerSize) - 1));
    return CCU_SUCCESS;
}

static CcuResult DoAll2AllVMultiLoop(AllToAllVMesh2DieContext &ctx)
{
    ctx.xnMaxTransportSize = ctx.MAX_TRANSPORT_SIZE;
    ctx.completedRankCount = 0;
    ctx.xnConst1 = 1;
    CCU_WHILE(ctx.completedRankCount != ctx.peerSize) {
        HCCL_DEBUG("[CcuKernelAllToAllVMesh2Die] Algorithm loops[%u].", ctx.peerSize);
        CCU_CHK_RET(LoopStep(ctx));
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllToAllVMesh2DieKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllToAllVMesh2Die *>(arg);
    AllToAllVMesh2DieContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllToAllVMesh2Die] Algorithm Init Begins. RankId[%u]", ctx.arg->rankId);
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));;
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    HCCL_INFO("[CcuKernelAllToAllVMesh2Die] Algorithm Begins. RankId[%u]", ctx.arg->rankId);

    CCU_CHK_RET(ExchangeInfoSync(ctx));
    CalcGroupSrcDst(ctx);
    CCU_CHK_RET(DoAll2AllVMultiLoop(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[CcuKernelAllToAllVMesh2Die] Algorithm Ends. RankId[%u]", ctx.arg->rankId);

    return CCU_SUCCESS;
}

}
