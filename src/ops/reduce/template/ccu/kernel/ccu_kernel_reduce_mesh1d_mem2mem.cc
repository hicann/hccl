/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_reduce_mesh1d_mem2mem.h"
#include "ccu_kernel_utils.h"

namespace ops_hccl {

constexpr int INPUT_XN_ID    = 0;
constexpr int OUTPUT_XN_ID   = 1;
constexpr int TOKEN_XN_ID    = 2;
constexpr int POST_SYNC_ID   = 3;
constexpr int CKE_IDX_0      = 0;
constexpr int GROUP_NUM      = 2;
const std::string LOOP_NAME  = "reduceMesh1DMem2MemLoop";

static CcuResult ParseKernelArg(ReduceMesh1DMem2MemContext &ctx, CcuKernelArgReduceMesh1DMem2Mem *kernelArg)

{
    ctx.dataType        = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType  = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;

        HCCL_DEBUG("[CcuKernelReduceMesh1DMem2Mem] outputDataType is [INVALID], set outputDataType to[%d]",
            ctx.dataType);
    }
    ctx.reduceOp = kernelArg->opParam.reduceType;
    return CCU_SUCCESS;
}

static CcuResult CreateLocalCopyLoop(ReduceMesh1DMem2MemContext &ctx, GroupReduceMesh1DMem2MemVar &var)
{
    if (ctx.IsLoopEntityRegistered(LOOP_NAME)) {
        return CCU_SUCCESS;
    }
    ctx.CreateLoopEntity(LOOP_NAME);
    auto &loops = ctx.loopMap[LOOP_NAME];

    for (uint32_t index = 0; index < 2; index++) { // 需要2个Loop
        ccu::Event event = ctx.moRes.completedEvent[index];

		loops.body[index].reset(new ccu::Func([&ctx, index, event, &var]() {
            ccu::LocalCopy(ctx.moRes.ccuBuf[0], var.src[index], var.len[index], event);
            ccu::EventWait(event);
            ccu::LocalCopy(var.dst[index], ctx.moRes.ccuBuf[0], var.len[index], event);
            ccu::EventWait(event);
        }));
		loops.loops[index].reset(new ccu::Loop(loops.loopParam[index], *loops.body[index]));
    }
    return CCU_SUCCESS;
}

static CcuResult LocalCopyByLoopGroup(ReduceMesh1DMem2MemContext &ctx, ccu::LocalAddr dst, ccu::LocalAddr src)
{
    GroupReduceMesh1DMem2MemVar var;
    CreateLocalCopyLoop(ctx, var);
	auto &loops = ctx.loopMap[LOOP_NAME];

    CCU_IF(ctx.localGoSize.addrOffset != 0)
    {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.moConfig.memSlice * ctx.moConfig.loopCount, 0);
        loopParam = loopParam + ctx.localGoSize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = ctx.moConfig.memSlice;

        var.src[0].addr = src.addr;
        var.src[0].token = src.token;
        var.dst[0].addr = dst.addr;
        var.dst[0].token = dst.token;
        var.len[0] = sliceSize;

        ccu::Variable paraCfg;
        paraCfg = GetParallelParam(ctx.moConfig.loopCount - 1, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

		loops.loopParam[0] = loopParam;
 	    std::vector<ccu::Loop> grpLoops{ *loops.loops[0] };
 	    ccu::LoopGroup group(paraCfg, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }

    CCU_IF(ctx.localGoSize.parallelParam != 0)
    {
        src.addr += ctx.localGoSize.addrOffset;
        dst.addr += ctx.localGoSize.addrOffset;

        var.src[0].addr = src.addr;
        var.src[0].token = src.token;
        var.dst[0].addr = dst.addr;
        var.dst[0].token = dst.token;
        var.len[0] = ctx.localGoSize.residual;

        src.addr += ctx.localGoSize.residual;
        dst.addr += ctx.localGoSize.residual;
        ccu::Variable sliceSize;
        sliceSize                  = ctx.moConfig.memSlice;

        var.src[1].addr = src.addr;
        var.src[1].token = src.token;
        var.dst[1].addr = dst.addr;
        var.dst[1].token = dst.token;
        var.len[1] = sliceSize;

        ccu::Variable loopCfg0;
        loopCfg0 = GetLoopParam(0, 0, 1);
        ccu::Variable loopCfg1;
        loopCfg1 = GetLoopParam(0, 0, 1);
        ccu::Variable offsetCfg;
        offsetCfg = GetOffsetParam(ctx.moConfig.memSlice, ctx.moConfig.msInterleave, 1);

		loops.loopParam[0] = loopCfg0;
 	    loops.loopParam[1] = loopCfg1;
 	    std::vector<ccu::Loop> grpLoops{ *loops.loops[0], *loops.loops[1] };
 	    ccu::LoopGroup group(ctx.localGoSize.parallelParam, offsetCfg, ctx.moConfig.loopCount, grpLoops);
    }
    return CCU_SUCCESS;
}

static CcuResult InitResource(ReduceMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelReduceMesh1DMem2Mem] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] channels.size: [%u]", arg->channelCount);

    // 按照rank号从小到大遍历channels，遇到本rank就填充本地资源，否则依次取远端资源，要求算法返回的Link同样是按顺序排列的
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

    ctx.chunkSize.resize(arg->rankSize - 1);

    ctx.moConfig.loopCount = LOOP_NUM;
    ctx.moConfig.msInterleave = LOCAL_COPY_MS;
    ctx.moConfig.memSlice = LOCAL_COPY_MS * CCU_MS_SIZE;

    ctx.moRes.eventCount = ctx.moConfig.loopCount;
	ctx.moRes.completedEvent = ccu::Array<ccu::Event>(ctx.moRes.eventCount);

    ctx.moRes.bufCount = ctx.moConfig.loopCount * ctx.moConfig.msInterleave;
	ctx.moRes.ccuBuf = ccu::Array<ccu::CcuBuffer>(ctx.moRes.bufCount);

    ctx.resourceAllocated = true;

    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ReduceMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNumVar, cnt++));
    for (uint16_t i = 0; i < arg->rankSize - 1; i++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.chunkSize[i], cnt++));
    }
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.residual, cnt++));
    return CCU_SUCCESS;
}

static void PreSync(ReduceMesh1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D PreSync begin");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.input[arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }

    uint32_t allBit = (1 << INPUT_XN_ID) | (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit);
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D PreSync end");
}

static void PostSync(ReduceMesh1DMem2MemContext &ctx)
{
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D post sync start");
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D post sync end");
}

static CcuResult DoRepeatReduce(ReduceMesh1DMem2MemContext &ctx, const std::vector<ccu::Variable> &srcAddr, const ccu::Variable &dstAddr)
{
    const auto *arg = ctx.arg;

    ctx.dstAddr.addr  = dstAddr;
    ctx.dstAddr.token  = ctx.token[arg->rankId];

    ctx.myInputAddr.addr  = srcAddr[arg->rankId];
    ctx.myInputAddr.token = ctx.token[arg->rankId];

    CCU_IF (ctx.flag != 0) {
        // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
        ctx.dstAddr.addr += ctx.outputRepeatStride;
        ctx.myInputAddr.addr += ctx.inputRepeatStride;
    }
    // 若root节点的输入输出地址不一致，则本地拷贝
    CCU_IF (ctx.isInputOutputEqual == 0) {
        CCU_CHK_RET(LocalCopyByLoopGroup(ctx, ctx.dstAddr, ctx.myInputAddr));
    }

    for (uint16_t i = 0; i < arg->rankSize - 1; i++) { // 外层循环控制step
        // 读不同rank的不同chunk
        for (uint16_t rmtId = 0; rmtId < arg->rankSize; ++rmtId) {
            if (rmtId == arg->rootId) {
                continue;
            }
            ctx.chunkOffset  = 0;
            ctx.dstAddr.addr  = dstAddr;
            ctx.remoteInputAddr.addr  = srcAddr[rmtId];
            ctx.remoteInputAddr.token = ctx.token[rmtId];

            CCU_IF (ctx.flag != 0) {
                // 非第一轮执行时，src 和 dst 已经初始化，需要添加偏移量
                ctx.dstAddr.addr += ctx.outputRepeatStride;
                ctx.remoteInputAddr.addr += ctx.inputRepeatStride;
            }
            uint16_t chkId = 0;
            if (rmtId < arg->rankId) {
                chkId = (i + rmtId) % (arg->rankSize - 1);
            } else {
                chkId = (i + rmtId - 1) % (arg->rankSize - 1);
            }
            // 获取链接Id，跳过本端
            uint16_t channelId = rmtId < arg->rootId ? rmtId : rmtId - 1;
            HCCL_DEBUG(
                "[ReadReduceRmtToLoc] debug rankId[%llu], root[%llu] chkId[%llu], rmtId[%llu] channelId[%llu]",
                arg->rankId, arg->rootId, chkId, rmtId, channelId);

            // 计算一下offset 0~(chikd-1)
            for (uint16_t j = 0; j < chkId; ++j) {
                ctx.chunkOffset += ctx.chunkSize[j];
            }
            // 更新对应的addr
            ctx.remoteInputAddr.addr += ctx.chunkOffset;
            ctx.dstAddr.addr += ctx.chunkOffset;

            CCU_IF(ctx.chunkSize[chkId] == 0)
            {
				const uint32_t rankMask = 1 << rmtId;
                ccu::EventRecord(ctx.event, rankMask);
            }

            CCU_IF(ctx.chunkSize[chkId] != 0)
            {
                // 读远端内存并Reduce, 将远端内存中的数据，和本端内存中的数据进行Reduce操作，结果保存在本端内存中
                ccu::ReadReduce(arg->channels[channelId], ctx.dstAddr, ctx.remoteInputAddr, ctx.chunkSize[chkId], ctx.dataType, ctx.reduceOp, ctx.event, 1 << rmtId);
            }
        }
        uint16_t allBit = ((1 << arg->rankSize) - 1) & (~(1 << arg->rankId));
        ccu::EventWait(ctx.event, allBit);
    }
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMeshMem2Mem1D ReadReduce end");
    return CCU_SUCCESS;
}

// ============================================================================
// 主入口 Kernel 函数
// ============================================================================
CcuResult CcuReduceMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceMesh1DMem2Mem *>(arg);

    ReduceMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMesh1D run");
    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    PreSync(ctx);

    CCU_IF(ctx.normalSliceSize != 0)
    {
        if (kernelArg->rankId == kernelArg->rootId) {
            ccu::Variable repeatNumAdd;
            repeatNumAdd  = 1;
            ctx.flag = 0;
            CCU_WHILE(ctx.repeatNumVar != UINT64_MAX) { // 循环repeatNum_次
                // root要去读每个rank每个chunk的数据
                DoRepeatReduce(ctx, ctx.input, ctx.output[kernelArg->rankId]);
                ctx.repeatNumVar += repeatNumAdd;
                ctx.flag = 1;
            }
        }
    }

    PostSync(ctx);
    HCCL_INFO("[CcuKernelReduceMesh1DMem2Mem] ReduceMesh1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl