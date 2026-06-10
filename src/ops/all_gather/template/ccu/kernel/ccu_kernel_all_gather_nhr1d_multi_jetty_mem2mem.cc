/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_nhr1d_multi_jetty_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID       = 1;
constexpr uint16_t TOKEN_XN_ID        = 2;
constexpr uint16_t POST_SYNC_ID       = 3;
constexpr uint16_t STEP_PRE_SYNC_ID   = 4;
constexpr uint16_t STEP_POST_SYNC_ID = 5;
constexpr uint16_t CKE_IDX_0          = 0;
constexpr uint16_t BIT_NUM_PER_CKE    = 16;

static CcuResult ParseKernelArg(AllGatherNHR1DMultiJettyMem2MemContext &ctx,
    CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.localSize = kernelArg->rank2ChannelIdx.size();
    ctx.myRankIdx = kernelArg->rank2ChannelIdx.size();
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] channels.size: [%u]", arg->channelCount);

    ctx.output.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; channelIdx++) {
        HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] MyRank[%u], channelIdx[%u]",
            arg->rankId, channelIdx);
        ctx.output[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }

    ctx.outputSliceOffset.resize(arg->rankSize);
    ctx.constVar1 = 1;
    ctx.repeatTimeflag = 0;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSizePerJetty, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSizePerJetty, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumInv, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual, argId++));

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] LoadArgs run finished");
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PreSync start");
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[ctx.myRankIdx],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint16_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PreSync end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PostSync start");
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] PostSync end");
    return CCU_SUCCESS;
}

static CcuResult DoSendRecvSlices(AllGatherNHR1DMultiJettyMem2MemContext &ctx,
    const uint32_t &toRank, ccu::LocalAddr &srcMem, ccu::RemoteAddr &dstMem)
{
    const auto *arg = ctx.arg;
    ChannelHandle sendChannel = arg->channels[arg->rank2ChannelIdx.at(toRank)];
    ccu::LocalAddr srcMemTmp;
    ccu::RemoteAddr dstMemTmp;
    srcMemTmp.addr = srcMem.addr;
    srcMemTmp.token = srcMem.token;
    dstMemTmp.addr = dstMem.addr;
    dstMemTmp.token = dstMem.token;

    CCU_IF(ctx.sliceSizePerJetty != 0)
    {
        for (uint32_t i = 0; i < arg->jettyNum - 1; ++i) {
            const uint16_t jettyMask = 1 << i;
            CCU_CHK_RET(ccu::Write(sendChannel, dstMemTmp, srcMemTmp, 
                ctx.sliceSizePerJetty, ctx.event, jettyMask));
            srcMemTmp.addr += ctx.sliceSizePerJetty;
            dstMemTmp.addr += ctx.sliceSizePerJetty;
        }
    } CCU_ELSE {
        for (uint32_t i = 0; i < arg->jettyNum - 1; ++i) {
            const uint16_t jettyMask = 1 << i;
            CCU_CHK_RET(ccu::EventRecord(ctx.event, jettyMask));
        }
    }
    CCU_IF(ctx.lastSliceSizePerJetty != 0)
    {
        const uint16_t lastJettyMask = 1 << (arg->jettyNum - 1);
        CCU_CHK_RET(ccu::Write(sendChannel, dstMemTmp, srcMemTmp, 
            ctx.lastSliceSizePerJetty, ctx.event, lastJettyMask));
    } CCU_ELSE {
        const uint16_t lastJettyMask = 1 << (arg->jettyNum - 1);
        CCU_CHK_RET(ccu::EventRecord(ctx.event, lastJettyMask));
    }

    uint16_t sendBit = (1 << arg->jettyNum) - 1;
    CCU_CHK_RET(ccu::EventWait(ctx.event, sendBit));

    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherNHRSingleStep(AllGatherNHR1DMultiJettyMem2MemContext &ctx,
    const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    const u32 &toRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.toRank);
    const u32 &fromRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.fromRank);
    u32 sendSliceIdx = 0;
    const ChannelHandle &sendChannel = arg->channels[toRankIdx];
    const ChannelHandle &recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    ctx.srcMem.token = ctx.token[ctx.myRankIdx];
    ctx.dstMem.token = ctx.token[toRankIdx];

    CCU_CHK_RET(ccu::NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID)); // 待修改
    CCU_CHK_RET(ccu::NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID));

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];
        ctx.srcMem.addr = ctx.output[ctx.myRankIdx];
        ctx.srcMem.addr += ctx.outputSliceOffset[sendSliceIdx];
        ctx.dstMem.addr = ctx.output[toRankIdx];
        ctx.dstMem.addr += ctx.outputSliceOffset[sendSliceIdx];

        ctx.repeatTimeflag = 0;
        ctx.tmpCopyRepeatNumInv = ctx.repeatNumInv;

        CCU_WHILE(ctx.tmpCopyRepeatNumInv != UINT64_MAX)
        {
            ctx.tmpCopyRepeatNumInv += ctx.constVar1;
            CCU_IF(ctx.repeatTimeflag == 1)
            {
                ctx.srcMem.addr += ctx.inputRepeatStride;
                ctx.dstMem.addr += ctx.outputRepeatStride;
            }
            CCU_CHK_RET(DoSendRecvSlices(ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.dstMem));
            ctx.repeatTimeflag = 1;
        }
    }
    CCU_CHK_RET(ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
    CCU_CHK_RET(ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));

    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherNHR(AllGatherNHR1DMultiJettyMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;

    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;

    ccu::Variable myrankInputSliceOffset;
    myrankInputSliceOffset = 0;

    for (u64 i = 0; i < arg->rankId; i++) {
        myrankInputSliceOffset += ctx.inputSliceStride;
    }
    for (u64 i = 0; i < arg->rankSize; i++) {
        ctx.outputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.outputSliceStride;
    }

    ctx.srcMem.addr = ctx.input;
    ctx.srcMem.addr += myrankInputSliceOffset;
    ctx.myDstMem.addr = ctx.output[ctx.myRankIdx];
    ctx.myDstMem.addr += ctx.outputSliceOffset[arg->rankId];
    ctx.srcMem.token = ctx.token[ctx.myRankIdx];
    ctx.myDstMem.token = ctx.token[ctx.myRankIdx];

    ctx.tmpCopyRepeatNumInv = ctx.repeatNumInv;

    CCU_WHILE(ctx.tmpCopyRepeatNumInv != UINT64_MAX)
    {
        ctx.tmpCopyRepeatNumInv += ctx.constVar1;
        CCU_IF(ctx.repeatTimeflag != 0)
        {
            ctx.srcMem.addr += ctx.inputRepeatStride;
            ctx.myDstMem.addr += ctx.outputRepeatStride;
        }
        const uint16_t rankMask = 1 << arg->rankId;
        CCU_IF(ctx.isInputOutputEqual == 0)
        {
            CCU_CHK_RET(GroupCopy(ctx, ctx.myDstMem, ctx.srcMem, ctx.groupOpSize));
            CCU_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
        } CCU_ELSE {
            CCU_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
        }
        
        CCU_CHK_RET(ccu::EventWait(ctx.event, rankMask));
        ctx.repeatTimeflag = 1;
    }

    for (auto &nhrStepInfo : arg->stepInfoVector) {
        CCU_CHK_RET(DoRepeatAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherNHR1DMultiJettyMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGatherNHR1DMultiJettyMem2Mem *>(arg);

    AllGatherNHR1DMultiJettyMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] AllGatherNHR1DMultiJettyMem2Mem start");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoRepeatAllGatherNHR(ctx));

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGatherNHR1DMultiJettyMem2Mem] AllGatherNHR1DMultiJettyMem2Mem end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
