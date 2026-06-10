/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_reduce_nhr_mem2mem_1D_multi_jetty.h"

#include <set>

namespace ops_hccl {
constexpr int BIT_NUM_PER_CKE = 16;
constexpr uint16_t OUTPUT_XN_ID = 0;
constexpr uint16_t TOKEN_XN_ID = 1;

static CcuResult ParseKernelArg(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, 
    CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.rankId = kernelArg->rankId;
    ctx.rankSize = kernelArg->rankSize;
    ctx.portNum = kernelArg->portNum;
    ctx.algStepInfoList = kernelArg->algStepInfoList;
    ctx.channelIdxMap = kernelArg->channelIdxMap;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.reduceOp = kernelArg->opParam.reduceType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;

    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[AllReduceNhrMem2Mem1DMultiJetty] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    HCCL_INFO("[AllReduceNhrMem2Mem1DMultiJetty] Init, KernelArgs are rankId[%u], rankSize[%u], portSize[%u],"
        " dataType[%d], outputDataType[%d], reduceOp[%d]",
        ctx.rankId, ctx.rankSize, ctx.portNum, ctx.dataType, ctx.outputDataType, ctx.reduceOp);
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;
    CHK_PRT_RET(arg->channelCount == 0, HCCL_ERROR("[AllReduceNhrMem2Mem1DMultiJetty] channels is empty!"),
        CCU_E_INTERNAL);

    ctx.outputAddrs.resize(ctx.rankSize);
    ctx.outputTokens.resize(ctx.rankSize);
    ctx.sliceOffset.resize(ctx.rankSize);

    for (uint32_t peerRankId = 0; peerRankId < ctx.rankSize; peerRankId++) {
        if (peerRankId == ctx.rankId) {
            // 本rank
        } else if (ctx.channelIdxMap.find(peerRankId) != ctx.channelIdxMap.end()) {
            const u32 channelIdx = ctx.channelIdxMap.at(peerRankId);
            HCCL_DEBUG("[AllReduceNhrMem2Mem1DMultiJetty] MyRank[%u], peerRankId[%u], ChannelId[%u]", ctx.rankId,
                peerRankId, channelIdx);
            ctx.outputAddrs[peerRankId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.outputTokens[peerRankId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
        }
    }

    ctx.events.resize(ctx.portNum);
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddrs[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputTokens[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInplace, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dataSizePerRank, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dataSizePerPort, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastRankSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastPortSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSize.residual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSizeLastSlice.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSizeLastSlice.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSizeLastSlice.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyGoSizeLastSlice.residual, argId++));
    return CCU_SUCCESS;
}

static uint32_t GetSignalIndex(const uint32_t signalBit)
{
    return signalBit / BIT_NUM_PER_CKE;
}

static uint16_t GetSignalMask(const uint32_t signalBit)
{
    return (1 << (signalBit % BIT_NUM_PER_CKE));
}

static CcuResult LocalWaitAllEvent(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const uint16_t mask)
{
    for (auto &event : ctx.events) {
        ccu::EventWait(event, mask);
    }
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;
    HCCL_DEBUG("[AllReduceNhrMem2Mem1DMultiJetty] PreSync start");

    const uint32_t signalBitOutput = 0;
    const uint32_t signalBitToken = 1;
    const uint32_t signalIndexOutput = GetSignalIndex(signalBitOutput);
    const uint32_t signalIndexToken = GetSignalIndex(signalBitToken);

    const uint16_t bitOutput = GetSignalMask(signalBitOutput);
    const uint16_t bitToken = GetSignalMask(signalBitToken);

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.outputAddrs[ctx.rankId],
            OUTPUT_XN_ID, signalIndexOutput, bitOutput);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.outputTokens[ctx.rankId],
            TOKEN_XN_ID, signalIndexToken, bitToken);
    }

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], signalIndexOutput, bitOutput);
        ccu::NotifyWait(arg->channels[i], signalIndexToken, bitToken);
    }

    HCCL_DEBUG("[AllReduceNhrMem2Mem1DMultiJetty] PreSync end");
    return CCU_SUCCESS;
}

static std::vector<u32> GetNonTxSliceIdxs(const std::vector<u32> &txSliceIdxs, uint32_t rankSize)
{
    std::vector<bool> isTx(rankSize, false);
    for (u32 idx : txSliceIdxs) {
        if (idx < rankSize) {
            isTx[idx] = true;
        }
    }

    std::vector<u32> nonTxSliceIdxs;
    for (u32 idx = 0; idx < rankSize; ++idx) {
        if (!isTx[idx]) {
            nonTxSliceIdxs.push_back(idx);
        }
    }
    return nonTxSliceIdxs;
}

static CcuResult DoLocalCopySlice(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, ccu::LocalAddr &src,
    ccu::LocalAddr &dst, u32 &copySliceIdx, ccu::Event &event, uint16_t mask)
{
    bool islastSlice = (copySliceIdx + 1 == ctx.rankSize);
    auto &sliceSize = islastSlice ? ctx.lastRankSliceSize : ctx.dataSizePerRank;

    CCU_IF (sliceSize != 0) {
        ccu::LocalCopy(dst, src, sliceSize, event, mask);
    } CCU_ELSE {
        ccu::EventRecord(event, mask);
    }
    return CCU_SUCCESS;
}

static CcuResult LocalCopySlices(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    u32 nonTxSliceIdx = 0;
    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;

    for (u64 i = 0; i < ctx.rankSize; i++) {
        ctx.sliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.dataSizePerRank;
    }

    ccu::Event &event = ctx.events[0];

    CCU_IF(ctx.isInplace == 0)
    {
        const NHRStepInfo &nhrStepInfo = ctx.algStepInfoList[0];
        const std::vector<u32> &nonTxSliceIdxList = GetNonTxSliceIdxs(nhrStepInfo.txSliceIdxs, ctx.rankSize);
        
        for (u32 i = 0; i < nonTxSliceIdxList.size(); i++) {
            nonTxSliceIdx = nonTxSliceIdxList[i];

            if (i != 0) {
                if (i % BIT_NUM_PER_CKE == 0) {
                    ccu::EventWait(event, (1 << BIT_NUM_PER_CKE) - 1);
                }
            }

            ctx.localInput.addr = ctx.inputAddr;
            ctx.localInput.addr += ctx.sliceOffset[nonTxSliceIdx];
            ctx.localInput.token = ctx.outputTokens[ctx.rankId];

            ctx.localOutput.addr = ctx.outputAddrs[ctx.rankId];
            ctx.localOutput.addr += ctx.sliceOffset[nonTxSliceIdx];
            ctx.localOutput.token = ctx.outputTokens[ctx.rankId];
            
            CCU_CHK_RET(DoLocalCopySlice(ctx, ctx.localInput, ctx.localOutput, nonTxSliceIdx, event, 1 << i));
        }
        ccu::EventWait(event, (1 << (nonTxSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1);
    }
    return CCU_SUCCESS;
}

static CcuResult DoReduceScatterNHR(AllReduceNhrMem2Mem1DMultiJettyContext &ctx);
static CcuResult DoReduceScatterNHRSingleStep(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const NHRStepInfo &nhrStepInfo);
static CcuResult DoWriteReduceSlice(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const u32 toRank, 
    ccu::LocalAddr &src, ccu::RemoteAddr &dst, const u32 sendSliceIdx, const u32 signalIndex);

static CcuResult DoReduceScatterNHR(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    constexpr uint32_t nhrNum = 2;
    for (u64 i = 0; i < ctx.algStepInfoList.size() / nhrNum; i++) {
        const NHRStepInfo &nhrStepInfo = ctx.algStepInfoList[i];
        CCU_CHK_RET(DoReduceScatterNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static CcuResult DoReduceScatterNHRSingleStep(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx = ctx.channelIdxMap.at(nhrStepInfo.toRank);
    const u32 fromRankIdx = ctx.channelIdxMap.at(nhrStepInfo.fromRank);
    u32 sendSliceIdx = 0;
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    ctx.localInput.token = ctx.outputTokens[ctx.rankId];
    ctx.remoteOutput.token = ctx.outputTokens[nhrStepInfo.toRank];

    HCCL_DEBUG("[%s] nhrStepInfo{step[%u], myRank[%u], toRank[%u], fromRank[%u]}, toRankIdx[%u], fromRankIdx[%u]",
        __func__, nhrStepInfo.step, nhrStepInfo.myRank, nhrStepInfo.toRank, nhrStepInfo.fromRank, toRankIdx,
        fromRankIdx);

    const uint32_t signalBitReady = 2;
    const uint32_t signalBitDone = 4;
    const uint32_t signalIdReady = GetSignalIndex(signalBitReady);
    const uint32_t signalIdDone = GetSignalIndex(signalBitDone);
    const uint16_t signalBitReadyMask = GetSignalMask(signalBitReady);
    const uint16_t signalBitDoneMask = GetSignalMask(signalBitDone);

    if (nhrStepInfo.step != 0) {
        ccu::NotifyRecord(arg->channels[fromRankIdx], signalIdReady, signalBitReadyMask);
        ccu::NotifyWait(arg->channels[toRankIdx], signalIdReady, signalBitReadyMask);
    }

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % BIT_NUM_PER_CKE == 0) {
                CCU_CHK_RET(LocalWaitAllEvent(ctx, (1 << BIT_NUM_PER_CKE) - 1));
            }
        }

        if (nhrStepInfo.step == 0) {
            ctx.localInput.addr = ctx.inputAddr;
            ctx.localInput.addr += ctx.sliceOffset[sendSliceIdx];
        } else {
            ctx.localInput.addr = ctx.outputAddrs[ctx.rankId];
            ctx.localInput.addr += ctx.sliceOffset[sendSliceIdx];
        }

        ctx.remoteOutput.addr = ctx.outputAddrs[nhrStepInfo.toRank];
        ctx.remoteOutput.addr += ctx.sliceOffset[sendSliceIdx];

        CCU_CHK_RET(DoWriteReduceSlice(ctx, nhrStepInfo.toRank, ctx.localInput, ctx.remoteOutput, 
            sendSliceIdx, i % BIT_NUM_PER_CKE));
    }
    CCU_CHK_RET(LocalWaitAllEvent(ctx, (1 << (sendSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1));

    ccu::NotifyRecord(arg->channels[toRankIdx], signalIdDone, signalBitDoneMask);
    ccu::NotifyWait(arg->channels[fromRankIdx], signalIdDone, signalBitDoneMask);
    
    HCCL_DEBUG("[DoReduceScatterNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
        ctx.rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return CCU_SUCCESS;
}

static CcuResult DoWriteReduceSlice(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const u32 toRank,
    ccu::LocalAddr &src, ccu::RemoteAddr &dst, const u32 sendSliceIdx, const u32 signalIndex)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx = ctx.channelIdxMap.at(toRank);

    const bool islastSlice = sendSliceIdx + 1 == ctx.rankSize;
    ccu::Variable lastSliceSize;
    lastSliceSize = islastSlice ? ctx.lastPortSliceSize : ctx.dataSizePerPort;

    uint16_t mask = 1 << signalIndex;

    CCU_IF(ctx.dataSizePerPort != 0)
    {
        for (uint32_t i = 0; i < ctx.portNum - 1; ++i) {
            ccu::WriteReduce(arg->channels[toRankIdx], dst, src, ctx.dataSizePerPort, 
                ctx.dataType, ctx.reduceOp, ctx.events[i], mask);
            src.addr += ctx.dataSizePerPort;
            dst.addr += ctx.dataSizePerPort;
        }
    }
    CCU_IF(ctx.dataSizePerPort == 0)
    {
        for (uint32_t i = 0; i < ctx.portNum - 1; ++i) {
            ccu::EventRecord(ctx.events[i], mask);
        }
    }
    CCU_IF(lastSliceSize != 0)
    {
        ccu::WriteReduce(arg->channels[toRankIdx], dst, src, lastSliceSize, 
            ctx.dataType, ctx.reduceOp, ctx.events[ctx.portNum - 1], mask);
    }
    CCU_IF(lastSliceSize == 0)
    {
        ccu::EventRecord(ctx.events[ctx.portNum - 1], mask);
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHR(AllReduceNhrMem2Mem1DMultiJettyContext &ctx);
static CcuResult DoAllGatherNHRSingleStep(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const NHRStepInfo &nhrStepInfo);
static CcuResult DoSendRecvSlice(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const u32 toRank,
    ccu::LocalAddr &src, ccu::RemoteAddr &dst, const u32 &sendSliceIdx, u32 signalIndex);

static CcuResult DoAllGatherNHR(AllReduceNhrMem2Mem1DMultiJettyContext &ctx)
{
    constexpr uint32_t nhrNum = 2;
    for (u64 i = ctx.algStepInfoList.size() / nhrNum; i < ctx.algStepInfoList.size(); i++) {
        const NHRStepInfo &nhrStepInfo = ctx.algStepInfoList[i];
        CCU_CHK_RET(DoAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGatherNHRSingleStep(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const NHRStepInfo &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx = ctx.channelIdxMap.at(nhrStepInfo.toRank);
    const u32 fromRankIdx = ctx.channelIdxMap.at(nhrStepInfo.fromRank);
    u32 sendSliceIdx = 0;
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    ctx.localInput.token = ctx.outputTokens[ctx.rankId];
    ctx.remoteOutput.token = ctx.outputTokens[nhrStepInfo.toRank];

    const uint32_t signalBitReady = 3;
    const uint32_t signalBitDone = 5;
    const uint32_t signalIdReady = GetSignalIndex(signalBitReady);
    const uint32_t signalIdDone = GetSignalIndex(signalBitDone);
    const uint16_t signalBitReadyMask = GetSignalMask(signalBitReady);
    const uint16_t signalBitDoneMask = GetSignalMask(signalBitDone);

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];

        if (i != 0) {
            if (i % BIT_NUM_PER_CKE == 0) {
                CCU_CHK_RET(LocalWaitAllEvent(ctx, (1 << BIT_NUM_PER_CKE) - 1));
            }
        }

        ctx.localInput.addr = ctx.outputAddrs[ctx.rankId];
        ctx.localInput.addr += ctx.sliceOffset[sendSliceIdx];

        ctx.remoteOutput.addr = ctx.outputAddrs[nhrStepInfo.toRank];
        ctx.remoteOutput.addr += ctx.sliceOffset[sendSliceIdx];
        
        CCU_CHK_RET(DoSendRecvSlice(ctx, nhrStepInfo.toRank, ctx.localInput, ctx.remoteOutput, 
            sendSliceIdx, i % BIT_NUM_PER_CKE));
    }
    CCU_CHK_RET(LocalWaitAllEvent(ctx, (1 << (sendSliceIdxList.size() % BIT_NUM_PER_CKE)) - 1));

    ccu::NotifyRecord(arg->channels[toRankIdx], signalIdDone, signalBitDoneMask);
    ccu::NotifyWait(arg->channels[fromRankIdx], signalIdDone, signalBitDoneMask);

    HCCL_DEBUG("[DoAllGatherNHRSingleStep] rank %u step %u, toRank=%u, fromRank=%u, nSlice=%lu",
        ctx.rankId, nhrStepInfo.step, nhrStepInfo.toRank, nhrStepInfo.fromRank, sendSliceIdxList.size());
    return CCU_SUCCESS;
}

static CcuResult DoSendRecvSlice(AllReduceNhrMem2Mem1DMultiJettyContext &ctx, const u32 toRank,
    ccu::LocalAddr &src, ccu::RemoteAddr &dst, const u32 &sendSliceIdx, u32 signalIndex)
{
    const auto *arg = ctx.arg;
    const u32 toRankIdx = ctx.channelIdxMap.at(toRank);

    const bool islastSlice = sendSliceIdx + 1 == ctx.rankSize;
    ccu::Variable lastSliceSize;
    lastSliceSize = islastSlice ? ctx.lastPortSliceSize : ctx.dataSizePerPort;

    uint16_t mask = 1 << signalIndex;

    CCU_IF(ctx.dataSizePerPort != 0)
    {
        for (uint32_t i = 0; i < ctx.portNum - 1; ++i) {
            ccu::Write(arg->channels[toRankIdx], dst, src, ctx.dataSizePerPort, ctx.events[i], mask);
            src.addr += ctx.dataSizePerPort;
            dst.addr += ctx.dataSizePerPort;
        }
    }
    CCU_IF(ctx.dataSizePerPort == 0)
    {
        for (uint32_t i = 0; i < ctx.portNum - 1; ++i) {
            ccu::EventRecord(ctx.events[i], mask);
        }
    }
    CCU_IF(lastSliceSize != 0)
    {
        ccu::Write(arg->channels[toRankIdx], dst, src, lastSliceSize, ctx.events[ctx.portNum - 1], mask);
    }
    CCU_IF(lastSliceSize == 0)
    {
        ccu::EventRecord(ctx.events[ctx.portNum - 1], mask);
    }
    return CCU_SUCCESS;
}

CcuResult CcuAllReduceNhrMem2Mem1DMultiJettyKernel(CcuKernelArg arg)
{
    HCCL_INFO("[AllReduceNhrMem2Mem1DMultiJetty] Algorithm run");

    AllReduceNhrMem2Mem1DMultiJettyContext ctx;
    auto *kernelArg = static_cast<CcuKernelArgAllReduceNhrMem2Mem1DMultiJetty *>(arg);
    CHK_PRT_RET(kernelArg == nullptr, HCCL_ERROR("[AllReduceNhrMem2Mem1DMultiJetty] kernelArg is null!"),
        CCU_E_INTERNAL);

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(LocalCopySlices(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoReduceScatterNHR(ctx));
    CCU_CHK_RET(DoAllGatherNHR(ctx));

    HCCL_INFO("[AllReduceNhrMem2Mem1DMultiJetty] Algorithm end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl