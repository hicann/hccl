/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ccu_kernel_reduce_scatter_nhr1d_multi_jetty_mem2mem.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID   = 0;
constexpr int TOKEN_XN_ID   = 1;

constexpr int CKE_IDX_INPUT = 0;
constexpr int CKE_IDX_TOKEN = 1;
constexpr int CKE_IDX_READY = 2;
constexpr int CKE_IDX_DONE  = 3;
constexpr int POST_XN_ID    = 4;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static uint32_t GetSignalIndex(const int signalBit)
{
    return static_cast<uint32_t>(signalBit) / BIT_NUM_PER_CKE;
}

static uint16_t GetSignalMask(const int signalBit)
{
    return (1 << (static_cast<uint32_t>(signalBit) % BIT_NUM_PER_CKE));
}

static CcuResult InitResource(ReduceScatterNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.input.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);
    for (uint32_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        ctx.input[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }
    
    ctx.jettyEvent.resize(ctx.portNum);
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] InitResource success!");
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterNhrMem2Mem1DMultiJettyContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceOneJettySize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceLastJettySize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    ctx.repeatNumVarTemp = ctx.repeatNumVar;
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] LoadArgs success!");
    return CCU_SUCCESS;
}

static CcuResult PreSync(ReduceScatterNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] PreSync start");

    const uint16_t signalBitInput = GetSignalMask(CKE_IDX_INPUT);
    const uint16_t signalBitToken = GetSignalMask(CKE_IDX_TOKEN);
    const uint32_t signalIndexInput = GetSignalIndex(CKE_IDX_INPUT);
    const uint32_t signalIndexToken = GetSignalIndex(CKE_IDX_TOKEN);
    
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[ctx.myRankIdx], CKE_IDX_INPUT, signalIndexInput, signalBitInput);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx], CKE_IDX_TOKEN, signalIndexToken, signalBitToken);
    }
    
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], signalIndexInput, signalBitInput);
        ccu::NotifyWait(arg->channels[i], signalIndexToken, signalBitToken);
    }
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] PreSync end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(ReduceScatterNhrMem2Mem1DMultiJettyContext &ctx)
{
    const auto *arg = ctx.arg;
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] PostSync start");
    const uint16_t selfBitInput = GetSignalMask(POST_XN_ID);
    const uint32_t signalIndexInput = GetSignalIndex(POST_XN_ID);

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], signalIndexInput, selfBitInput);
    }

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], signalIndexInput, selfBitInput);
    }
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] PostSync end");
    return CCU_SUCCESS;
}

static CcuResult DoRepeatSendRecvSlices(ReduceScatterNhrMem2Mem1DMultiJettyContext &ctx, const uint32_t &toRank, 
    ccu::LocalAddr &src, ccu::RemoteAddr &dst)
{
    const auto *arg = ctx.arg;
    ccu::Variable repeatNumAdd;
    repeatNumAdd = 1;
    ctx.flag = 0;
    ctx.repeatNumVarTemp = ctx.repeatNumVar;
    
    CCU_WHILE(ctx.repeatNumVarTemp != UINT64_MAX) {
        CCU_IF(ctx.repeatNumVarTemp != UINT64_MAX) {
            ctx.repeatNumVarTemp += repeatNumAdd;
        }
        CCU_IF(ctx.flag == 1) {
            src.addr += ctx.inputRepeatStride;
            dst.addr += ctx.inputRepeatStride;
        }
        
        ccu::LocalAddr tempSrc;
        ccu::RemoteAddr tempDst;
        tempSrc.addr = src.addr;
        tempSrc.token = src.token;
        tempDst.addr = dst.addr;
        tempDst.token = dst.token;
        
        CCU_IF(ctx.sliceOneJettySize == 0) {
            for (uint32_t jettyId = 0; jettyId < ctx.portNum - 1; jettyId++) {
                ccu::EventRecord(ctx.jettyEvent[jettyId], 1);
            }
        }
        CCU_IF(ctx.sliceOneJettySize != 0) {
            for (uint32_t jettyId = 0; jettyId < ctx.portNum - 1; jettyId++) {
                ccu::WriteReduce(arg->channels[ctx.rank2ChannelIdx.at(toRank)], tempDst, tempSrc, ctx.sliceOneJettySize,
                    ctx.dataType, ctx.reduceOp, ctx.jettyEvent[jettyId], 1);
                tempDst.addr += ctx.sliceOneJettySize;
                tempSrc.addr += ctx.sliceOneJettySize;
            }
        }
        CCU_IF(ctx.sliceLastJettySize == 0) {
            ccu::EventRecord(ctx.jettyEvent[ctx.portNum - 1], 1);
        }
        CCU_IF(ctx.sliceLastJettySize != 0) {
            uint32_t jettyId = ctx.portNum - 1;
            ccu::WriteReduce(arg->channels[ctx.rank2ChannelIdx.at(toRank)], tempDst, tempSrc, ctx.sliceLastJettySize,
                    ctx.dataType, ctx.reduceOp, ctx.jettyEvent[jettyId], 1);
        }
        for (uint32_t jettyId = 0; jettyId < ctx.portNum; jettyId++) {
            ccu::EventWait(ctx.jettyEvent[jettyId], 1);
        }
        ctx.flag = 1;
    }
    ctx.flag = 0;
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatterNHRSingleStep(ReduceScatterNhrMem2Mem1DMultiJettyContext &ctx, 
    const NHRStepInfo &nhrStepInfo, const std::vector<ccu::Variable> &inputSliceOffset)
{
    const auto *arg = ctx.arg;
    uint32_t& toRankIdx = ctx.rank2ChannelIdx.at(nhrStepInfo.toRank);
    uint32_t& fromRankIdx = ctx.rank2ChannelIdx.at(nhrStepInfo.fromRank);
    const auto &sendChannel = arg->channels[toRankIdx];
    const auto &recvChannel = arg->channels[fromRankIdx];
    const std::vector<uint32_t> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    
    ctx.remoteDst.token = ctx.token[toRankIdx];
    ctx.localSrc.token = ctx.token[ctx.myRankIdx];

    const uint32_t signalIdxReady = GetSignalIndex(CKE_IDX_READY);
    const uint32_t signalIdxDone = GetSignalIndex(CKE_IDX_DONE);
    const uint16_t signalBitReady = GetSignalMask(CKE_IDX_READY);
    const uint16_t signalBitDone = GetSignalMask(CKE_IDX_DONE);

    if (nhrStepInfo.step != 0) {
        ccu::NotifyRecord(recvChannel, signalIdxReady, signalBitReady);
        ccu::NotifyWait(sendChannel, signalIdxReady, signalBitReady);
    }
    u32 sendSliceIdx = 0;
    for (const uint32_t &sendSliceIdx : sendSliceIdxList) {
        ctx.remoteDst.addr = ctx.input[toRankIdx];
        ctx.remoteDst.addr += inputSliceOffset[sendSliceIdx];
        ctx.localSrc.addr = ctx.input[ctx.myRankIdx];
        ctx.localSrc.addr += inputSliceOffset[sendSliceIdx];
        CCU_CHK_RET(DoRepeatSendRecvSlices(ctx, nhrStepInfo.toRank, ctx.localSrc, ctx.remoteDst));
        HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] DoRepeatSendRecvSlices success");
    }
    
    ccu::NotifyRecord(sendChannel, signalIdxDone, signalBitDone);
    ccu::NotifyWait(recvChannel, signalIdxDone, signalBitDone);
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] DoRepeatReduceScatterNHRSingleStep success");
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatter(ReduceScatterNhrMem2Mem1DMultiJettyContext &ctx)
{
    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;
    
    std::vector<ccu::Variable> inputSliceOffset;
    inputSliceOffset.resize(ctx.dimSize);
    for (uint64_t i = 0; i < ctx.dimSize; i++) {
        inputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.inputSliceStride;
    }

    for (auto &nhrStepInfo : ctx.stepInfoVector) {
        CCU_CHK_RET(DoRepeatReduceScatterNHRSingleStep(ctx, nhrStepInfo, inputSliceOffset));
    }
    
    ctx.localDst.addr = ctx.output;
    ctx.localDst.token = ctx.token[ctx.myRankIdx];
    ctx.localSrc.addr = ctx.input[ctx.myRankIdx];
    ctx.localSrc.addr += inputSliceOffset[ctx.rankId];
    ctx.localSrc.token = ctx.token[ctx.myRankIdx];

    ccu::Variable repeatNumAdd2;
    repeatNumAdd2 = 1;
    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX) {
        ctx.repeatNumVar += repeatNumAdd2;
        CCU_IF(ctx.flag == 1) {
            ctx.localSrc.addr += ctx.inputRepeatStride;
            ctx.localDst.addr += ctx.outputRepeatStride;
        }
        CCU_IF(ctx.sliceSize == 0) {
            ccu::EventRecord(ctx.event, 1);
        }
        CCU_IF(ctx.sliceSize != 0) {
            ccu::LocalCopy(ctx.localDst, ctx.localSrc, ctx.sliceSize, ctx.event, 1);
        }
        ccu::EventWait(ctx.event, 1);
        ctx.flag = 1;
    }
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] DoRepeatReduceScatter success");
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterNhrMem2Mem1DMultiJettyKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterNhrMutilJettyMem2Mem1D *>(arg);
    
    ReduceScatterNhrMem2Mem1DMultiJettyContext ctx;
    ctx.arg = kernelArg;

    ctx.rankId = kernelArg->rankId;
    ctx.dimSize = kernelArg->dimSize;
    ctx.portNum = kernelArg->portNum;
    ctx.stepInfoVector = kernelArg->stepInfoVector;
    ctx.rank2ChannelIdx = kernelArg->rank2ChannelIdx;
    ctx.localSize = kernelArg->rank2ChannelIdx.size();
    ctx.myRankIdx = kernelArg->rank2ChannelIdx.size();
    
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] Init, KernelArgs are rankId[%u], dimSize[%u], dataType[%d], "
        "outputDataType[%d], reduceOp[%d], portNum[%d]",
        ctx.rankId, ctx.dimSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp, ctx.portNum);

    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] CcuKernelReduceScatterNhrMutilJettyMem2Mem1D run");

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoRepeatReduceScatter(ctx));
    CCU_CHK_RET(PostSync(ctx));

    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] CcuKernelReduceScatterNhrMutilJettyMem2Mem1D end");
    
    return CCU_SUCCESS;
}

} // namespace ops_hccl