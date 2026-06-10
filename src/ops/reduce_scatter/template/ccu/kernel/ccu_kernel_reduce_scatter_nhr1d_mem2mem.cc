/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_scatter_nhr1d_mem2mem.h"

namespace ops_hccl {

// 按信号功能划分notify的bit
constexpr uint16_t INPUT_XN_ID      = 0;
constexpr uint16_t TOKEN_XN_ID      = 2;
constexpr uint16_t POST_SYNC_ID     = 3;
constexpr uint16_t STEP_PRE_SYNC_ID = 4;
constexpr uint16_t STEP_POST_SYNC_ID= 5;

constexpr uint16_t CKE_IDX_0        = 0;

constexpr uint16_t LINK_SIZE        = 2;

static CcuResult ParseKernelArg(ReduceScatterNHR1DMem2MemContext &ctx, CcuKernelArgReduceScatterNHR1D *kernelArg)
{
    ctx.mySubCommRankId = kernelArg->mySubCommRankId;   // 虚拟rankid，用于获取本rank对应的输入偏移
    ctx.axisId          = kernelArg->axisId;
    ctx.dimSize         = kernelArg->dimSize;
    ctx.axisSize        = kernelArg->axisSize;
    ctx.localSize       = kernelArg->rank2ChannelIdx.size();
    ctx.myRankIdx       = kernelArg->rank2ChannelIdx.size();
    ctx.reduceOp        = kernelArg->opParam.reduceType;
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterNHR1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
                   ctx.outputDataType);
    }
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] KernelArg: mySubCommRankId[%u], myRankIdx[%d], axisId[%u], dimSize[%llu], localSize[%u], "
              "dataType[%d], outputDataType[%d], reduceOp[%d]",
              ctx.mySubCommRankId, ctx.myRankIdx, ctx.axisId, ctx.dimSize, ctx.localSize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    
    ctx.output = ccu::Variable{};
    for (uint32_t channelIdx = 0; channelIdx < ctx.localSize; channelIdx++) {
        HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] mySubCommRankId[%u], channelId[%u]", ctx.mySubCommRankId, channelIdx);

        ctx.input.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID));
        ctx.token.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID));
    }
    ctx.input.push_back(ccu::Variable{});
    ctx.token.push_back(ccu::Variable{});

    ctx.resourceAllocated = false;
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] InitResource finished");
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[ctx.myRankIdx], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0Size, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1Size, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0LastSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1LastSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceOutputOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, cnt++));
    ctx.repeatNumVarTemp = ctx.repeatNumVar;
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] LoadArgs run finished");
    return CCU_SUCCESS;
}

static void PreSync(ReduceScatterNHR1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] PreSync start");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[ctx.myRankIdx], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[ctx.myRankIdx], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] PreSync end");
}

static void PostSync(ReduceScatterNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] PostSync run finished");
}

static CcuResult DoRepeatWriteReduceSlices(ReduceScatterNHR1DMem2MemContext &ctx, const u32 &toRank, 
    ccu::LocalAddr &src, ccu::RemoteAddr &dst, const bool islastSlice)
{
    const auto *arg = ctx.arg;
    ccu::Variable repeatNumAdd;
    repeatNumAdd  = 1;
    ctx.isRepeatIter = 0;
    
    auto toRankIt = arg->rank2ChannelIdx.find(toRank);
    if (toRankIt == arg->rank2ChannelIdx.end()) {
        HCCL_ERROR("[CcuKernelReduceScatterNHR1DMem2Mem] rank2ChannelIdx not find toRank key [%u]", toRank);
        return CCU_E_PARA;
    }
    const u32& toRankIdx = toRankIt->second;
    ChannelHandle sendChannel = arg->channels[toRankIdx];
    
    ctx.repeatNumVarTemp = ctx.repeatNumVar;
    CCU_WHILE(ctx.repeatNumVarTemp != UINT64_MAX) {
        CCU_IF(ctx.repeatNumVarTemp != UINT64_MAX) {
            ctx.repeatNumVarTemp += repeatNumAdd;
        }

        CCU_IF(ctx.isRepeatIter == 1) {
            src.addr += ctx.inputRepeatStride;
            dst.addr += ctx.inputRepeatStride;
        }
        CCU_IF(ctx.isRepeatIter == 0) {
            if (ctx.axisId == 1) {
                src.addr += ctx.die0Size;
                dst.addr += ctx.die0Size;
            }
        }
        ctx.sliceSize = (ctx.axisId == 0) ? (islastSlice? ctx.die0LastSliceSize : ctx.die0Size)
                                    : (islastSlice? ctx.die1LastSliceSize : ctx.die1Size);

        CCU_IF(ctx.sliceSize != 0) {
            ccu::WriteReduce(sendChannel, dst, src, ctx.sliceSize, ctx.dataType, ctx.reduceOp, ctx.event, 1);
        }
        CCU_IF(ctx.sliceSize == 0) {
            ccu::EventRecord(ctx.event, 1);
        }
        ccu::EventWait(ctx.event, 1);
        ctx.isRepeatIter = 1;
    }
    ctx.isRepeatIter = 0;
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatterNHRSingleStep(ReduceScatterNHR1DMem2MemContext &ctx, const NHRStepInfo &nhrStepInfo,
    const std::vector<ccu::Variable> &inputSliceOffset)
{
    const auto *arg = ctx.arg;
    auto toRankIt = arg->rank2ChannelIdx.find(nhrStepInfo.toRank);
    if (toRankIt == arg->rank2ChannelIdx.end()) {
        HCCL_ERROR("[CcuKernelReduceScatterNHR1DMem2Mem] rank2ChannelIdx not find toRank key [%u]", nhrStepInfo.toRank);
        return CCU_E_PARA;
    }
    const u32& toRankIdx = toRankIt->second;

    auto fromRankIt = arg->rank2ChannelIdx.find(nhrStepInfo.fromRank);
    if (fromRankIt == arg->rank2ChannelIdx.end()) {
        HCCL_ERROR("[CcuKernelReduceScatterNHR1DMem2Mem] rank2ChannelIdx not find fromRank key [%u]", nhrStepInfo.fromRank);
        return CCU_E_PARA;
    }
    const u32& fromRankIdx = fromRankIt->second;

    ChannelHandle sendChannel = arg->channels[toRankIdx];
    ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32> &sendSliceIdxList = nhrStepInfo.txSliceIdxs;
    ctx.remoteDst.token = ctx.token[toRankIdx];
    ctx.localSrc.token = ctx.token[ctx.myRankIdx];

    bool islastSlice = false;

    // 通知fromRank，可以写入
    ccu::NotifyRecord(recvChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);

    // 等待toRank通知其可以写入
    ccu::NotifyWait(sendChannel, CKE_IDX_0, 1 << STEP_PRE_SYNC_ID);

    for (const u32 &sendSliceIdx : sendSliceIdxList) {
        ctx.remoteDst.addr = ctx.input[toRankIdx];
        ctx.remoteDst.addr += inputSliceOffset[sendSliceIdx];
        ctx.localSrc.addr = ctx.input[ctx.myRankIdx];
        ctx.localSrc.addr += inputSliceOffset[sendSliceIdx];

        islastSlice = (sendSliceIdx + 1 == arg->dimSize);
        CCU_CHK_RET(DoRepeatWriteReduceSlices(ctx, nhrStepInfo.toRank, ctx.localSrc, ctx.remoteDst, islastSlice));
    }

    // 通知toRank数据写入完毕
    ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);

    // 等待fromRank通知数据写入完毕
    ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID);
    
    return CCU_SUCCESS;
}

static CcuResult DoRepeatReduceScatterNHR(ReduceScatterNHR1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;
    // 用来记录每个rank要读取的rank的sliceIdx的偏移
    // 后面会用inputAddr来加上这个偏移获取sliceIdx的地址
    std::vector<ccu::Variable> inputSliceOffset;
    for (u64 i = 0; i < arg->dimSize; i++) {
        inputSliceOffset.push_back(ccu::Variable{});
        inputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.inputSliceStride;
    }

    for (auto &nhrStepInfo : arg->stepInfoVector) {
        CCU_CHK_RET(DoRepeatReduceScatterNHRSingleStep(ctx, nhrStepInfo, inputSliceOffset));
    }
    // 因为所有的修改都是在input上进行的，所以最后需要把input上的数据搬到output上
    ctx.localSrc.addr  = ctx.input[ctx.myRankIdx];
    ctx.localSrc.addr += inputSliceOffset[ctx.mySubCommRankId];
    ctx.localSrc.token = ctx.token[ctx.myRankIdx];
    ctx.localDst.addr  = ctx.output;
    ctx.localDst.addr += ctx.currentRankSliceOutputOffset;
    ctx.localDst.token = ctx.token[ctx.myRankIdx];

    bool islastSlice = (ctx.mySubCommRankId + 1 == arg->dimSize);
    ccu::Variable repeatNumAdd2;
    repeatNumAdd2  = 1;
    CCU_WHILE(ctx.repeatNumVar != UINT64_MAX) {
        ctx.repeatNumVar += repeatNumAdd2;
        CCU_IF(ctx.isRepeatIter == 1) {
            ctx.localSrc.addr += ctx.inputRepeatStride;
            ctx.localDst.addr += ctx.outputRepeatStride;
        }
        CCU_IF(ctx.isRepeatIter == 0) {
            if (ctx.axisId == 1) {
                ctx.localSrc.addr += ctx.die0Size;
                ctx.localDst.addr += ctx.die0Size;
            }
        }
        ccu::Variable &localSliceSize = (ctx.axisId == 0) ? (islastSlice? ctx.die0LastSliceSize : ctx.die0Size)
                                                          : (islastSlice? ctx.die1LastSliceSize : ctx.die1Size);
        CCU_IF(localSliceSize != 0) {
            CCU_IF(ctx.isInputOutputEqual == 0) {
                ccu::LocalCopy(ctx.localDst, ctx.localSrc, localSliceSize, ctx.event, 1);
            }
            CCU_IF(ctx.isInputOutputEqual != 0) {
                ccu::EventRecord(ctx.event, 1);
            }
        }
        CCU_IF(localSliceSize == 0) {
            ccu::EventRecord(ctx.event, 1);
        }
        ccu::EventWait(ctx.event, 1);
        ctx.isRepeatIter = 1;
    }
    
    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceScatterNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterNHR1D *>(arg);

    ReduceScatterNHR1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] CcuKernelReduceScatterNHR1DMem2Mem run.");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    PreSync(ctx);
    CCU_CHK_RET(DoRepeatReduceScatterNHR(ctx));
    PostSync(ctx);

    HCCL_INFO("[CcuKernelReduceScatterNHR1DMem2Mem] CcuKernelReduceScatterNHR1DMem2Mem end.");
    return CCU_SUCCESS;
}

} // namespace ops_hccl
