/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_nhr1d_mem2mem.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID = 1;
constexpr uint16_t TOKEN_XN_ID = 2;
constexpr uint16_t POST_SYNC_ID = 3;
constexpr uint16_t STEP_PRE_SYNC_ID = 4;
constexpr uint16_t STEP_POST_SYNC_ID = 5;
constexpr uint16_t CKE_IDX_0 = 0;
constexpr uint16_t BIT_NUM_PER_CKE = 16;

static CcuResult ParseKernelArg(AllGatherNHR1DMem2MemContext& ctx, CcuKernelArgAllGatherNHR1D* kernelArg)
{
    ctx.arg = kernelArg;
    ctx.localSize = kernelArg->rank2ChannelIdx.size();
    ctx.myRankIdx = kernelArg->rank2ChannelIdx.size();
    HCCL_DEBUG(
        "[CcuKernelAllGatherNHR1DMem2Mem] Init, KernelArgs are mySubCommRankId[%u], axisId[%u], axisSize[%u], "
        "stepInfoVector.size[%u], myRankIdx[%u] localSize[%u]",
        kernelArg->mySubCommRankId, kernelArg->axisId, kernelArg->axisSize, kernelArg->stepInfoVector.size(),
        ctx.myRankIdx, ctx.localSize);
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGatherNHR1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGatherNHR1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] channels.size: [%u]", arg->channelCount);

    ctx.output.resize(ctx.localSize + 1);
    ctx.token.resize(ctx.localSize + 1);

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; channelIdx++) {
        HCCL_INFO(
            "[CcuKernelAllGatherNHR1DMem2Mem] mySubCommRankId[%u], channelId[%u] localSize[%u]", arg->mySubCommRankId,
            channelIdx, arg->channelCount);
        ctx.output[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }

    ctx.outputSliceOffset.resize(arg->dimSize);
    ctx.myrankInputSliceOffset = 0;
    ctx.repeatTimeflag = 0;
    ctx.constVar1 = 1;
    ctx.resourceAllocated = false;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGatherNHR1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    uint32_t argId = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.myRankIdx], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1Size, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die0LastSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.die1LastSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argId++));

    HCCL_DEBUG("[CcuKernelAllGatherNHR1DMem2Mem] LoadArgs run finished");
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGatherNHR1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;

    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PreSync start");
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.output[ctx.myRankIdx], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[ctx.myRankIdx], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    uint32_t allBit = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PreSync end");
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllGatherNHR1DMem2MemContext& ctx)
{
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PostSync start");
    const auto* arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] PostSync finished");
    return CCU_SUCCESS;
}

static CcuResult DoRepeatSendRecvSlices(
    AllGatherNHR1DMem2MemContext& ctx, const u32& toRank, ccu::LocalAddr& src, ccu::RemoteAddr& dst, u32 signalIndex,
    bool islastSlice)
{
    const auto* arg = ctx.arg;
    ccu::Variable tmpRepeatNum;
    ChannelHandle sendChannel = arg->channels[arg->rank2ChannelIdx.at(toRank)];
    tmpRepeatNum = ctx.repeatNum;
    ctx.repeatTimeflag = 0;
    const uint16_t signalMask = 1 << signalIndex;

    CCU_WHILE(tmpRepeatNum != UINT64_MAX)
    {
        tmpRepeatNum += ctx.constVar1;
        CCU_IF(ctx.repeatTimeflag == 1)
        {
            // 同一bit同时只能挂一笔在飞: repeat 多笔时发本笔前先等上一笔完成并清位,
            // 每笔完成信号恰好被消费一次(末笔完成信号留给外层批量Wait);
            // repeatNum==1(小数据)时本分支不执行, 16个bit并发挂16笔由外层批量等待收敛
            CCU_CHK_RET(ccu::EventWait(ctx.localEvent, signalMask));
            src.addr += ctx.inputRepeatStride;
            dst.addr += ctx.outputRepeatStride;
        }
        CCU_ELSE
        {
            if (arg->axisId == 1) {
                src.addr += (islastSlice ? ctx.die0LastSize : ctx.die0Size);
                dst.addr += (islastSlice ? ctx.die0LastSize : ctx.die0Size);
            }
        }
        ccu::Variable& sliceSize = (arg->axisId == 0) ? (islastSlice ? ctx.die0LastSize : ctx.die0Size) :
                                                        (islastSlice ? ctx.die1LastSize : ctx.die1Size);

        CCU_IF(sliceSize != 0)
        {
            CCU_CHK_RET(ccu::Write(sendChannel, dst, src, sliceSize, ctx.localEvent, signalMask));
        }
        CCU_ELSE
        {
            // 空片也置位完成信号, 保证外层批量Wait不悬挂
            CCU_CHK_RET(ccu::EventRecord(ctx.localEvent, signalMask));
        }
        ctx.repeatTimeflag = 1;
    }

    return CCU_SUCCESS;
}

static CcuResult DoAllGatherGroupCopy(AllGatherNHR1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    CCU_IF(ctx.isInputOutputEqual == 0)
    {
        CCU_IF(ctx.groupCopyRepeatNum != UINT64_MAX)
        {
            // src/dst 在此独立计算: GroupCopy 已移至最后一个 step 内提交,
            // 此时 ctx.srcMem 已被 Write 循环改写, 不能再依赖其在 DoRepeatAllGatherNHR 的初值
            ccu::Variable tmpSrc;
            ccu::Variable tmpDst;
            tmpSrc = ctx.input;
            tmpSrc += ctx.myrankInputSliceOffset;
            tmpDst = ctx.output[ctx.myRankIdx];
            tmpDst += ctx.outputSliceOffset[arg->mySubCommRankId];
            bool islastSlice = (arg->mySubCommRankId + 1 == arg->dimSize);
            if (arg->axisId == 1) {
                ccu::Variable die0Slice = islastSlice ? ctx.die0LastSize : ctx.die0Size;
                tmpSrc += die0Slice;
                tmpDst += die0Slice;
            }
            ctx.repeatTimeflag = 0;
            CCU_WHILE(ctx.groupCopyRepeatNum != UINT64_MAX)
            {
                ctx.groupCopyRepeatNum += ctx.constVar1;
                CCU_IF(ctx.repeatTimeflag != 0)
                {
                    tmpDst += ctx.outputRepeatStride;
                    tmpSrc += ctx.inputRepeatStride;
                }
                ccu::LocalAddr localDst;
                localDst.addr = tmpDst;
                localDst.token = ctx.token[ctx.myRankIdx];
                ccu::LocalAddr localSrc;
                localSrc.addr = tmpSrc;
                localSrc.token = ctx.token[ctx.myRankIdx];
                CCU_CHK_RET(GroupCopy(ctx, localDst, localSrc, ctx.goSize, GetCcuVersion()));
                ctx.repeatTimeflag = 1;
            }
        }
    }
    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherNHRSingleStep(AllGatherNHR1DMem2MemContext& ctx, const NHRStepInfo& nhrStepInfo)
{
    const auto* arg = ctx.arg;
    const u32& toRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.toRank);
    const u32& fromRankIdx = arg->rank2ChannelIdx.at(nhrStepInfo.fromRank);
    u32 sendSliceIdx = 0;
    ChannelHandle sendChannel = arg->channels[toRankIdx];
    ChannelHandle recvChannel = arg->channels[fromRankIdx];
    const std::vector<u32>& sendSliceIdxList = nhrStepInfo.txSliceIdxs;

    HCCL_INFO("sendSliceIdxList.size()[%zu]", sendSliceIdxList.size());
    ctx.srcMem.token = ctx.token[ctx.myRankIdx];
    ctx.dstMem.token = ctx.token[toRankIdx];

    for (u32 i = 0; i < sendSliceIdxList.size(); i++) {
        sendSliceIdx = sendSliceIdxList[i];
        if (i != 0 && i % BIT_NUM_PER_CKE == 0) {
            // 等满上一批16笔完成并清位, 释放bit给本批复用
            CCU_CHK_RET(ccu::EventWait(ctx.localEvent, (1 << BIT_NUM_PER_CKE) - 1));
        }
        if (nhrStepInfo.step == 0 || sendSliceIdx == arg->mySubCommRankId) {
            // 自己的贡献数据从 input 直发: txSliceIdxs 每个 step 均以自己的 slot 开头,
            // 若走 output 转发路径会依赖 GroupCopy 先完成; 直发后 GroupCopy 只承担
            // 最终结果落位, 可与通信并发(数据源头相同, 内容完全等价)
            ctx.srcMem.addr = ctx.input;
            ctx.srcMem.addr += ctx.myrankInputSliceOffset;
        } else {
            ctx.srcMem.addr = ctx.output[ctx.myRankIdx];
            ctx.srcMem.addr += ctx.outputSliceOffset[sendSliceIdx];
        }
        ctx.dstMem.addr = ctx.output[toRankIdx];
        ctx.dstMem.addr += ctx.outputSliceOffset[sendSliceIdx];
        bool islastSlice = false;
        islastSlice = (sendSliceIdx + 1 == arg->dimSize);
        HCCL_INFO(
            "mySubCommRankId[%zu], rankId[%zu], subCommToRankId[%zu], sendSliceIdx[%zu]", arg->mySubCommRankId,
            ctx.myRankIdx, nhrStepInfo.toRank, sendSliceIdx);
        CCU_CHK_RET(
            DoRepeatSendRecvSlices(ctx, nhrStepInfo.toRank, ctx.srcMem, ctx.dstMem, i % BIT_NUM_PER_CKE, islastSlice));
    }

    // 等最后一批(可能不足16笔)全部完成再发step同步通知, 避免接收方提前读数据造成竞争;
    // 批量为16时(1<<16)-1经int运算得65535, 截断到uint16_t恰为全1掩码
    u32 lastGroupSize = ((sendSliceIdxList.size() - 1) % BIT_NUM_PER_CKE) + 1;

    if (nhrStepInfo.step + 1 == arg->stepInfoVector.size()) {
        // 最后一个step发送slice最多: 此刻全部Write已在飞, 提交本rank贡献的GroupCopy,
        // local copy与通信并发执行, 由EventWait等待期间与PostSync掩盖其耗时;
        // 自己slot的转发已改为input直发, 无数据依赖; kernel结束前运行时保证copy完成
        CCU_CHK_RET(DoAllGatherGroupCopy(ctx));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.localEvent, (1 << lastGroupSize) - 1));

    if (nhrStepInfo.step + 1 != arg->stepInfoVector.size()) {
        CCU_CHK_RET(ccu::NotifyRecord(sendChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
        CCU_CHK_RET(ccu::NotifyWait(recvChannel, CKE_IDX_0, 1 << STEP_POST_SYNC_ID));
    }

    return CCU_SUCCESS;
}

static CcuResult DoRepeatAllGatherNHR(AllGatherNHR1DMem2MemContext& ctx)
{
    const auto* arg = ctx.arg;
    ccu::Variable tmpSliceOffset;
    tmpSliceOffset = 0;

    for (u64 i = 0; i < arg->mySubCommRankId; i++) {
        ctx.myrankInputSliceOffset += ctx.inputSliceStride;
    }

    for (u64 i = 0; i < arg->dimSize; i++) {
        ctx.outputSliceOffset[i] = tmpSliceOffset;
        tmpSliceOffset += ctx.outputSliceStride;
    }

    // GroupCopy 已移至最后一个 step 内与通信并发提交(见 DoRepeatAllGatherNHRSingleStep),
    // 本 rank 贡献的转发由 input 直发, 不再需要 step 循环前先完成落位
    ctx.groupCopyRepeatNum = ctx.repeatNum;

    for (auto& nhrStepInfo : arg->stepInfoVector) {
        CCU_CHK_RET(DoRepeatAllGatherNHRSingleStep(ctx, nhrStepInfo));
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGatherNHR1DMem2MemKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<CcuKernelArgAllGatherNHR1D*>(arg);

    AllGatherNHR1DMem2MemContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] AllGatherNHR1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoRepeatAllGatherNHR(ctx));

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGatherNHR1DMem2Mem] AllGatherNHR1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl
