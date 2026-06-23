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
#include "ccu_kernel_all_to_all_v_mesh1d_2die.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID  = 2;
constexpr int CKE_IDX_0    = 0;
constexpr int POST_SYNC_ID = 3;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static CcuResult ParseKernelArg(AllToAllVMesh1D2DieContext &ctx, CcuKernelArgAllToAllVMesh1D2Die *kernelArg)
{
    ctx.peerSize = kernelArg->channelCount + (kernelArg->withMyRank ? 1 : 0);
    ctx.localId = kernelArg->channelCount; // 表示本卡的序号

    HCCL_INFO("[CcuKernelAllToAllVMesh1D2Die] rankId[%u], peerSize[%u], withMyRank[%u]",
        kernelArg->rankId, ctx.peerSize, kernelArg->withMyRank);

    return CCU_SUCCESS;
}

static CcuResult InitResource(AllToAllVMesh1D2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllToAllVMesh1D2Die] RankId[%u] channels is empty!", arg->rankId);
        return CcuResult::CCU_E_INTERNAL;
    }

    CCU_CHK_RET(AllocGoResource(ctx.moConfig, ctx.moRes, ctx.resourceAllocated,
        CCU_MS_LOCAL_COPY_LOOP_COUNT, LOCAL_COPY_MS_PER_LOOP));

    ctx.output.resize(arg->channelCount + 1);
    ctx.token.resize(arg->channelCount + 1);
    for (uint32_t peerId = 0; peerId < arg->channelCount; peerId++) {
        HCCL_DEBUG("[CcuKernelAllToAllVMesh1D2Die] RankId[%u], PeerId[%u]", arg->rankId, peerId);
        ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[peerId], OUTPUT_XN_ID);
        ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[peerId], TOKEN_XN_ID);
    }

    ctx.sendRecvInfo.resize(ctx.peerSize);

    ctx.src.resize(arg->channelCount);
    ctx.dst.resize(arg->channelCount);

    const uint32_t eventNum = (ctx.peerSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    ctx.events.resize(eventNum);

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllToAllVMesh1D2DieContext &ctx)
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

static CcuResult ExchangeInfoSync(AllToAllVMesh1D2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable tempDst;
    for (u32 peerId = 0; peerId < arg->channelCount; peerId++) {
        tempDst = ctx.output[ctx.localId];
        tempDst += ctx.sendRecvInfo[peerId].recvOffset;
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[peerId], tempDst, OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[peerId], ctx.token[ctx.localId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    uint32_t waitBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (u32 peerId = 0; peerId < arg->channelCount; peerId++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[peerId], CKE_IDX_0, waitBits));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllToAllVMesh1D2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t peerId = 0; peerId < arg->channelCount; peerId++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[peerId], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (u32 peerId = 0; peerId < arg->channelCount; peerId++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[peerId], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

static void CalcGroupSrcDst(AllToAllVMesh1D2DieContext &ctx)
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

static CcuResult LoopStep(AllToAllVMesh1D2DieContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t peerId = 0; peerId < ctx.peerSize; peerId++) {
        HCCL_DEBUG("[CcuKernelAllToAllVMesh1D2Die] LoopStep start, rankId[%u] peerId[%u]", arg->rankId, peerId);

        const uint16_t eventIdx = peerId / BIT_NUM_PER_CKE;
        const uint16_t rankMask = 1 << (peerId % BIT_NUM_PER_CKE);

        CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum == UINT64_MAX) {
            CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask));
            continue;
        }
        CCU_IF(ctx.sendRecvInfo[peerId].sendLoopNum == UINT64_MAX - 1) {
            ctx.curSendTailSize = ctx.sendRecvInfo[peerId].sendTailSize;
            ctx.curSendTailGoSize = ctx.sendRecvInfo[peerId].sendTailGoSize;
            CCU_IF(ctx.curSendTailSize == 0) {
                CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask));
            } CCU_ELSE {
                if (arg->withMyRank && peerId == ctx.localId) {
                    GroupCopy(ctx, ctx.localDst, ctx.localSrc, ctx.curSendTailGoSize);
                    CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask));
                } else {
                    CCU_CHK_RET(ccu::Write(arg->channels[peerId], ctx.dst[peerId], ctx.src[peerId],
                        ctx.curSendTailSize, ctx.events[eventIdx], rankMask));
                }
            }
            ctx.completedRankCount += ctx.xnConst1;
        } CCU_ELSE {
            if (arg->withMyRank && peerId == ctx.localId) {
                GroupCopy(ctx, ctx.localDst, ctx.localSrc, ctx.xnMaxTransportGoSize);
                CCU_CHK_RET(ccu::EventRecord(ctx.events[eventIdx], rankMask));
                ctx.localDst.addr += ctx.xnMaxTransportSize;
                ctx.localSrc.addr += ctx.xnMaxTransportSize;
            } else {
                CCU_CHK_RET(ccu::Write(arg->channels[peerId], ctx.dst[peerId], ctx.src[peerId],
                    ctx.xnMaxTransportSize, ctx.events[eventIdx], rankMask));
                ctx.dst[peerId].addr += ctx.xnMaxTransportSize;
                ctx.src[peerId].addr += ctx.xnMaxTransportSize;
            }
        }
        ctx.sendRecvInfo[peerId].sendLoopNum += ctx.xnConst1;

        HCCL_DEBUG("[CcuKernelAllToAllVMesh1D2Die] LoopStep end, RankId[%u] peerId[%u]", arg->rankId, peerId);
    }

    const uint32_t eventNum = (ctx.peerSize + BIT_NUM_PER_CKE - 1) / BIT_NUM_PER_CKE;
    for (uint32_t i = 0; i < eventNum; i++) {
        uint16_t eventMask;
        if (i == eventNum - 1) {
            if (ctx.peerSize % BIT_NUM_PER_CKE == 0) {
                eventMask = (1 << BIT_NUM_PER_CKE) - 1;
            } else {
                eventMask = (1 << (ctx.peerSize % BIT_NUM_PER_CKE)) - 1;
            }
        } else {
            eventMask = (1 << BIT_NUM_PER_CKE) - 1;
        }
        CCU_CHK_RET(ccu::EventWait(ctx.events[i], eventMask));
    }
    return CCU_SUCCESS;
}

static CcuResult DoAll2AllVMultiLoop(AllToAllVMesh1D2DieContext &ctx)
{
    ctx.xnMaxTransportSize = ctx.MAX_TRANSPORT_SIZE;
    ctx.completedRankCount = 0;
    ctx.xnConst1 = 1;
    CCU_WHILE(ctx.completedRankCount != ctx.peerSize) {
        HCCL_DEBUG("[CcuKernelAllToAllVMesh1D2Die] Algorithm loops[%u].", ctx.peerSize);
        CCU_CHK_RET(LoopStep(ctx));
    }

    return CCU_SUCCESS;
}

CcuResult CcuAlltoAllVMesh1D2DieKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllToAllVMesh1D2Die *>(arg);
    AllToAllVMesh1D2DieContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelAllToAllVMesh1D2Die] Algorithm Init Begins. RankId[%u]", ctx.arg->rankId);
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    HCCL_INFO("[CcuKernelAllToAllVMesh1D2Die] Algorithm Begins. RankId[%u]", ctx.arg->rankId);

    CCU_CHK_RET(ExchangeInfoSync(ctx));
    CalcGroupSrcDst(ctx);
    CCU_CHK_RET(DoAll2AllVMultiLoop(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[CcuKernelAllToAllVMesh1D2Die] Algorithm Ends. RankId[%u]", ctx.arg->rankId);

    return CCU_SUCCESS;
}

}
