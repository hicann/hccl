/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_omnipipe_nhr1d_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t INPUT_XN_ID   = 0;
constexpr uint16_t TOKEN_XN_ID   = 1;
constexpr uint16_t CKE_IDX_INPUT = 0;
constexpr uint16_t CKE_IDX_TOKEN = 1;
constexpr uint16_t CKE_IDX_READY = 2;
constexpr uint16_t CKE_IDX_DONE  = 3;
constexpr uint16_t POST_XN_ID    = 4;
constexpr uint16_t BIT_NUM_PER_CKE = 16; // CKE的位数，一个CKE可以处理16种信号

static CcuResult ParseKernelArg(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx, CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankId = kernelArg->rankId;
    ctx.rankSize = kernelArg->rankSize;
    ctx.userRank = kernelArg->subCommRanks[0][ctx.rankId];
    ctx.stepInfoVector = kernelArg->stepInfoVector;
    ctx.rank2ChannelIdx = kernelArg->rank2ChannelIdx;
    ctx.localSize = ctx.rank2ChannelIdx.size(); // nhr 算法通信rank数
    ctx.myRankIdx = ctx.rank2ChannelIdx.size(); // InitResources中将本端放在末尾 此处为对应的idx

    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.outputDataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] userRank[%u] rankId[%u], rankSize[%u], "
                "dataType[%d], outputDataType[%d], reduceOp[%d]", ctx.userRank, ctx.rankId, ctx.rankSize,
                ctx.dataType, ctx.outputDataType, ctx.reduceOp);
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.input.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);
    for (uint32_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        ctx.input[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }

    ctx.inputOmniSliceStrideVec.resize(ctx.rankSize);
    ctx.inputOmniSliceSizeVec.resize(ctx.rankSize);

    HCCL_INFO("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] InitResource success!");
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyFlag, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniPipeSliceStride, argId++));
    for (uint32_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceStrideVec[i], argId++));
    }
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    for (uint32_t i = 0; i < ctx.rankSize; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.inputOmniSliceSizeVec[i], argId++));
    }

    HCCL_INFO("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] LoadArgs success!");
    return CCU_SUCCESS;
}

static uint32_t GetSignalIndex(const int signalBit)
{
    // 一个CKE有16位，可以处理16个用途
    return static_cast<uint32_t>(signalBit) / BIT_NUM_PER_CKE;
}

static uint16_t GetSignalMask(const int signalBit)
{
    return (1 << (static_cast<uint32_t>(signalBit) % BIT_NUM_PER_CKE));
}

static CcuResult PreSync(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] PreSync start");

    const uint16_t signalBitInput = GetSignalMask(CKE_IDX_INPUT);
    const uint16_t signalBitToken = GetSignalMask(CKE_IDX_TOKEN);
    const uint32_t signalIndexInput = GetSignalIndex(CKE_IDX_INPUT);
    const uint32_t signalIndexToken = GetSignalIndex(CKE_IDX_TOKEN);
    
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[ctx.myRankIdx],
                        CKE_IDX_INPUT, signalIndexInput, signalBitInput));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx],
                        CKE_IDX_TOKEN, signalIndexToken, signalBitToken));
    }
    
    const uint16_t waitMask = signalBitInput | signalBitToken; // 组合一下mask
    std::set<uint32_t> signalIdxes{signalIndexInput, signalIndexToken};
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        for (uint32_t signalIdx : signalIdxes) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], signalIdx, waitMask));
        }
    }
    HCCL_INFO("[CcuKernelReduceScatterNhrMutilJettyMem2Mem1D] PreSync end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] PostSync start");
    const auto *arg = ctx.arg;
    const uint16_t selfBitInput = GetSignalMask(POST_XN_ID);
    const uint32_t signalIndexInput = GetSignalIndex(POST_XN_ID);

    // 通知所有对端
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], signalIndexInput, selfBitInput));
    }

    // 等待所有需要的对端
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], signalIndexInput, selfBitInput));
    }
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] PostSync end");
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatterNHRSingleStep(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx, const NHRStepInfoRS &nhrStepInfo)
{
    const auto *arg = ctx.arg;
    uint32_t& toRankIdx = ctx.rank2ChannelIdx.at(nhrStepInfo.toRank);
    uint32_t& fromRankIdx = ctx.rank2ChannelIdx.at(nhrStepInfo.fromRank);
    const auto &sendChannel = arg->channels[toRankIdx];
    const auto &recvChannel = arg->channels[fromRankIdx];
    const std::vector<uint32_t> &recvSliceIdxList = nhrStepInfo.rxSliceIdxs;
    HCCL_DEBUG(
        "[DoRepeatReduceScatterNHRSingleStep] myRank[%u] rankId[%u] step[%u] toRank[%u](channelIdx[%u]) fromRank[%u](channelIdx[%u]) SliceSize[%u]",
        ctx.userRank, ctx.rankId, nhrStepInfo.step, nhrStepInfo.toRank, toRankIdx, nhrStepInfo.fromRank,
        fromRankIdx, recvSliceIdxList.size());

    ccu::LocalAddr dst;
    ccu::RemoteAddr src;
    dst.token = ctx.token[ctx.myRankIdx];
    src.token = ctx.token[fromRankIdx];

    const uint32_t signalIdxReady = GetSignalIndex(CKE_IDX_READY); // 0
    const uint32_t signalIdxDone = GetSignalIndex(CKE_IDX_DONE);   // 0
    const uint16_t signalBitReady = GetSignalMask(CKE_IDX_READY);  // 准备好的信号
    const uint16_t signalBitDone = GetSignalMask(CKE_IDX_DONE);    // 写完的信号

    // 通知对端rank自己准备好了-前同步
    if (nhrStepInfo.step != 0) {
        CCU_CHK_RET(ccu::NotifyRecord(recvChannel, signalIdxReady, signalBitReady)); // 通知fromrank可以写入
        CCU_CHK_RET(ccu::NotifyWait(sendChannel, signalIdxReady, signalBitReady)); // 等待torank准备好
    }

    for (const uint32_t &recvSliceIdx : recvSliceIdxList) {
        HCCL_DEBUG("[DoRepeatReduceScatterNHRSingleStep] sliceIdx[%u]", recvSliceIdx);
        src.addr = ctx.input[fromRankIdx];
        dst.addr = ctx.input[ctx.myRankIdx];
        if (recvSliceIdx == ctx.rankId) {
            src.addr += ctx.sliceStride;
            src.addr += ctx.inputOmniPipeSliceStride;

            dst.addr += ctx.sliceStride;
            dst.addr += ctx.inputOmniPipeSliceStride;
            ctx.sliceSize = ctx.inputOmniSliceSizeVec[recvSliceIdx];
        } else {
            src.addr += ctx.inputOmniSliceStrideVec[recvSliceIdx];
            dst.addr += ctx.inputOmniSliceStrideVec[recvSliceIdx];
            ctx.sliceSize = ctx.inputOmniSliceSizeVec[recvSliceIdx];
        }

        CCU_IF(ctx.sliceSize != 0) {
            CCU_CHK_RET(ccu::ReadReduce(recvChannel, dst, src, ctx.sliceSize, ctx.dataType, ctx.reduceOp, ctx.event, 1));
        }
        CCU_IF(ctx.sliceSize == 0) {
            CCU_CHK_RET(ccu::EventRecord(ctx.event, 1));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    }
    
    // 写之后告诉对端写完了-后同步
    // 告诉toRank数据写完了
    CCU_CHK_RET(ccu::NotifyRecord(sendChannel, signalIdxDone, signalBitDone));
    // 等待fromRank写完数据
    CCU_CHK_RET(ccu::NotifyWait(recvChannel, signalIdxDone, signalBitDone));
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatterNHR(ReduceScatterOmniPipeNHR1DMem2MemContext &ctx)
{
    for (auto &nhrStepInfo : ctx.stepInfoVector) {
        CCU_CHK_RET(DoRepeatReduceScatterNHRSingleStep(ctx, nhrStepInfo));
    }
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] DoRepeatReduceScatterNHR success");
    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceScatterOmniPipeNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterOmniPipeNHR1DMem2Mem *>(arg);
    ReduceScatterOmniPipeNHR1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] ReduceScatterOmniPipeNHR1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoRepeatReduceScatterNHR(ctx));
    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelReduceScatterOmniPipeNHR1DMem2Mem] ReduceScatterOmniPipeNHR1DMem2Mem end");
    
    return CCU_SUCCESS;
}
} // namespace Hccl
