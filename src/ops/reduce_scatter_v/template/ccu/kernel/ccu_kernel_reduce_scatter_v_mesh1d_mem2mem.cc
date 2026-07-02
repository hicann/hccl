/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ccu_kernel_reduce_scatter_v_mesh1d_mem2mem.h"

namespace ops_hccl {

// bit序号，每种信号用一个bit
constexpr int INPUT_XN_ID   = 0;
constexpr int SCRATCH_XN_ID = 1;
constexpr int TOKEN_XN_ID   = 2;
constexpr int POST_SYNC_ID  = 3;
// cke序号
constexpr int CKE_IDX_0     = 0;

static CcuResult ParseKernelArg(
    ReduceScatterVMesh1DMem2MemContext &ctx, CcuKernelArgReduceScatterVMesh1DMem2Mem *kernelArg)
{
    ctx.dataType = kernelArg->opParam.vDataDes.dataType;
    ctx.outputDataType = ctx.dataType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
        HCCL_DEBUG("[CcuKernelReduceScatterMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }

    ctx.reduceOp = kernelArg->opParam.reduceType;
    HCCL_INFO(
        "[CcuKernelReduceScatterVMesh1DMem2Mem] Init, KernelArgs are dataType[%d], outputDataType[%d], reduceOp[%d]",
        ctx.dataType, ctx.outputDataType, ctx.reduceOp);
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceScatterVMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;
    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceScatterVMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] channels.size: [%u]", arg->channelCount);
    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
    ctx.input.resize(arg->rankSize);
    ctx.token.resize(arg->rankSize);
    ctx.dst.resize(arg->rankSize);
    ctx.reduceScatterVSrc.resize(arg->rankSize);
    ctx.reduceScatterVDst.resize(arg->rankSize);
    for (uint64_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId != arg->rankId) {
            ctx.input[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], INPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
        }
    }
    ctx.moConfig.loopCount = CCU_M2M_LOCAL_COPY_LOOP_COUNT;
    ctx.moConfig.msInterleave = CCU_MS_INTERLEAVE;
    ctx.moConfig.memSlice = CCU_MS_SIZE;
    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceScatterVMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratch, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchInterval, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.offset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.reduceGosize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.reduceGosize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.reduceGosize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.reduceGosize.residual, cnt++));
    return CCU_SUCCESS;
}

static void PreSync(ReduceScatterVMesh1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] ReduceScatterVMem2Mem1D PreSync begin");
 	const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.input[arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t allBit = 1 << INPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
 	    ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
 	}
    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] ReduceScatterVMem2Mem1D PreSync end");
}

static void PostSync(ReduceScatterVMesh1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] ReduceScatterVMem2Mem1D post sync start");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] ReduceScatterVMem2Mem1D post sync end");
}

static void CollectAllRanksSlice(std::vector<ccu::RemoteAddr>& tmpSrc,
    std::vector<ccu::LocalAddr>& tmpDst, ReduceScatterVMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelId = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            // 跳过本卡
            ccu::EventRecord(ctx.event, 1 << rankIdx);
        } else {
            ccu::Read(arg->channels[channelId], tmpDst[rankIdx], tmpSrc[rankIdx], ctx.sliceSize, ctx.event, 1 << rankIdx);
            channelId++;
        }
    }
    // 等读完所有对端
    ccu::EventWait(ctx.event, (1 << arg->rankSize) - 1);
}

static void PrepareReduceScatterVData(std::vector<ccu::RemoteAddr>& reduceScatterVSrc,
    std::vector<ccu::LocalAddr>& reduceScatterVDst, ReduceScatterVMesh1DMem2MemContext &ctx)
{
    ccu::Variable scratchOffset;
    const auto *arg = ctx.arg;
    scratchOffset = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; rankIdx++) {
        if (rankIdx == arg->rankId) {
            // 本卡不使用 scratch 数据依然放在 input 上
            ctx.reduceScatterVDst[rankIdx].addr = ctx.input[arg->rankId];
            ctx.reduceScatterVDst[rankIdx].addr += ctx.offset;
            ctx.reduceScatterVDst[rankIdx].token = ctx.token[arg->rankId];
            continue;
        }
        // 从远端的 input 上，读取本卡所需的数据片
        ctx.reduceScatterVSrc[rankIdx].addr = ctx.input[rankIdx];
        ctx.reduceScatterVSrc[rankIdx].addr += ctx.offset;
        ctx.reduceScatterVSrc[rankIdx].token = ctx.token[rankIdx];

        // 将数据放到本卡的 scratch 上
        ctx.reduceScatterVDst[rankIdx].addr  = ctx.scratch;
        ctx.reduceScatterVDst[rankIdx].addr += scratchOffset;
        scratchOffset += ctx.scratchInterval;
        ctx.reduceScatterVDst[rankIdx].token = ctx.token[arg->rankId];
    }
}

static void DoReduceScatterV(ReduceScatterVMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    CCU_IF(ctx.sliceSize != 0) {
        PrepareReduceScatterVData(ctx.reduceScatterVSrc, ctx.reduceScatterVDst, ctx);
        CollectAllRanksSlice(ctx.reduceScatterVSrc, ctx.reduceScatterVDst, ctx);
        ccu::LocalAddr outDst;
        outDst.addr = ctx.output;
        outDst.token = ctx.token[arg->rankId];
        // 执行reduce操作
        GroupLocalReduce(ctx, outDst, ctx.reduceScatterVDst, ctx.reduceGosize, ctx.dataType, ctx.outputDataType, ctx.reduceOp);
    }
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================

CcuResult CcuReduceScatterVMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterVMesh1DMem2Mem *>(arg);
    ReduceScatterVMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] ReduceScatterVMesh1DMem2Mem run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    PreSync(ctx);
    DoReduceScatterV(ctx);
    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceScatterVMesh1DMem2Mem] ReduceScatterVMesh1DMem2Mem end");

    return CCU_SUCCESS;
}
} // namespace ops_hccl