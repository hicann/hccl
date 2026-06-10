/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_scatter_mesh1d.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint16_t OUTPUT_XN_ID   = 0;
constexpr uint16_t TOKEN_XN_ID    = 1;
constexpr uint16_t POST_SYNC_ID   = 3;
constexpr uint16_t CKE_IDX_0      = 0;

static CcuResult ParseKernelArg(ScatterMesh1DContext &ctx, CcuKernelArgScatterMesh1D *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankSize = kernelArg->rankSize;
    ctx.rankId = kernelArg->rankId;
    ctx.rootId = kernelArg->rootId;
    ctx.dataType = kernelArg->opParam.DataDes.dataType;
    ctx.outputDataType = kernelArg->opParam.DataDes.outputType;
    if (ctx.outputDataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        ctx.outputDataType = ctx.dataType;
    }
    return CCU_SUCCESS;
}

static CcuResult InitResource(ScatterMesh1DContext &ctx)
{
    uint16_t channelIdx = 0;
    if (ctx.arg->channelCount == 0) {
        HCCL_ERROR("[CcuScatterMesh1DKernel] channels is empty!");
        return CCU_E_INTERNAL;
    }

    ctx.output.resize(ctx.rankSize);
    ctx.token.resize(ctx.rankSize);
    for (uint64_t peerId = 0; peerId < ctx.rankSize; peerId++) {
        if (peerId != ctx.rankId) {
			ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
            channelIdx++;
		}
    }

    ctx.flag = 0;
    ctx.inputMem.resize(ctx.rankSize);
    ctx.outputMem.resize(ctx.rankSize);
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(ScatterMesh1DContext &ctx)
{
	uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[ctx.rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankSliceInputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputSliceStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.isInputOutputEqual, argId++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(ScatterMesh1DContext &ctx)
{
    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.output[ctx.rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(ctx.arg->channels[i], ctx.token[ctx.rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit);
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(ScatterMesh1DContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult DoScatterOnce(ScatterMesh1DContext &ctx)
{
    uint32_t channelId = 0;

    ccu::LocalAddr myOutput;
    myOutput.addr = ctx.outputMem[ctx.rankId].addr;
    myOutput.token = ctx.outputMem[ctx.rankId].token;

    ccu::Variable sliceSize;

    for (uint64_t rankIdx = 0; rankIdx < ctx.rankSize; rankIdx++) {
		uint16_t mask = 1 << rankIdx;
        sliceSize = (rankIdx == ctx.rankSize - 1) ? ctx.lastSliceSize : ctx.normalSliceSize;
        CCU_IF(sliceSize != 0)
        {
            if (rankIdx == ctx.rankId) {
                CCU_IF(ctx.isInputOutputEqual == 0)
                {
                    ccu::LocalCopy(myOutput, ctx.inputMem[rankIdx], sliceSize, ctx.event, mask);
                }
                CCU_IF(ctx.isInputOutputEqual != 0)
                {
                    ccu::EventRecord(ctx.event, mask);
                }
            } else {
                ccu::Write(ctx.arg->channels[channelId], ctx.outputMem[rankIdx],
				             ctx.inputMem[rankIdx], sliceSize, ctx.event, mask);
                channelId++;
            }
        }
        CCU_IF(sliceSize == 0)
        {
            ccu::EventRecord(ctx.event, mask);
        }
    }

    ccu::EventWait(ctx.event, (1 << ctx.rankSize) - 1);
    return CCU_SUCCESS;
}

static CcuResult DoRepeatScatter(ScatterMesh1DContext &ctx)
{
    ccu::Variable repeatNumAdd;
    repeatNumAdd = 1;

    // 初始化每张卡 input/output 逻辑地址
    for (uint64_t curId = 0; curId < ctx.rankSize; curId++) {
        ctx.inputMem[curId].token = ctx.token[curId];
        ctx.outputMem[curId].token = ctx.token[curId];

        ctx.inputMem[curId].addr = ctx.input;
        ctx.outputMem[curId].addr = ctx.output[curId];
        for (uint64_t i = 0; i < curId; i++) {
            ctx.inputMem[curId].addr += ctx.currentRankSliceInputOffset;
            ctx.outputMem[curId].addr += ctx.outputSliceStride;
        }
    }

    if (ctx.rankId != ctx.rootId) {
        return CCU_SUCCESS;
    }

    CCU_WHILE(ctx.repeatNum != UINT64_MAX)
    {
        CCU_IF(ctx.flag != 0)
        {
            for (auto &i : ctx.inputMem) {
                i.addr += ctx.inputRepeatStride;
            }
            for (auto &r : ctx.outputMem) {
                r.addr += ctx.outputRepeatStride;
            }
        }
        CCU_CHK_RET(DoScatterOnce(ctx));
        ctx.repeatNum += repeatNumAdd;
        ctx.flag = 1;
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatterMesh1D *>(arg);
    ScatterMesh1DContext ctx;

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(DoRepeatScatter(ctx));
    CCU_CHK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl