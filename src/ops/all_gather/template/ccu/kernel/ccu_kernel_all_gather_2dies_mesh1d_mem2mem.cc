/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_2dies_mesh1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

static CcuResult ParseKernelArg(AllGather2DiesMeshMem2Mem1DContext &ctx, CcuKernelArgAllGather2DiesMeshMem2Mem1D *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankIdGroup = kernelArg->rankIdGroup;
    ctx.ifHandleSelfRank = kernelArg->ifHandleSelfRank;
    ctx.rankSize = kernelArg->dimSize;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGather2DiesMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGather2DiesMeshMem2Mem1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.input.push_back(ccu::Variable()); // 两个kernel共用一个input
    uint16_t channelIdx = 0;
    ctx.output.resize(arg->dimSize);
    ctx.token.resize(arg->dimSize);

    for (uint64_t peerId = 0; peerId < arg->dimSize; peerId++) {
        if (peerId == arg->rankId) {
            // 本地资源，默认构造
        } else if (peerId != ctx.rankIdGroup[channelIdx]) {
            // 非本die的rank，本地资源
        } else {
            HCCL_INFO("[CcuKernelArgAllGather2DiesMeshMem2Mem1D] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                arg->rankId, peerId, channelIdx);
            ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
            ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
        }
        if (peerId >= ctx.rankIdGroup[channelIdx] && channelIdx < ctx.rankIdGroup.size() - 1) {
            channelIdx++;
        }
    }
    ctx.offSet = ccu::Variable();
    ctx.sliceSize = ccu::Variable();
    ctx.event = ccu::Event();

    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGather2DiesMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input[0], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.offSet, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localGoSize.residual, cnt++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGather2DiesMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }
    uint32_t allBit = ((1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID));
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllGather2DiesMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult DoAllGather(AllGather2DiesMeshMem2Mem1DContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr src;
    src.addr = ctx.input[0];
    src.token = ctx.token[arg->rankId];

    std::vector<ccu::RemoteAddr> remoteDst;
    remoteDst.resize(ctx.rankIdGroup.size());

    for (uint64_t rankIdx = 0; rankIdx < ctx.rankIdGroup.size(); rankIdx++) {
        remoteDst[rankIdx].addr = ctx.output[ctx.rankIdGroup[rankIdx]]; 
        remoteDst[rankIdx].addr += ctx.offSet;
        remoteDst[rankIdx].token = ctx.token[ctx.rankIdGroup[rankIdx]]; 
    
        CCU_IF(ctx.sliceSize != 0) {
            CCU_CHK_RET(ccu::Write(arg->channels[rankIdx], remoteDst[rankIdx], src, ctx.sliceSize, ctx.event, 1 << ctx.rankIdGroup[rankIdx]));
        }
        CCU_IF(ctx.sliceSize == 0) {
            CCU_CHK_RET(ccu::EventRecord(ctx.event, 1 << ctx.rankIdGroup[rankIdx]));
        }
    }

    if (ctx.ifHandleSelfRank) {
        ccu::LocalAddr localDst;
        localDst.addr = ctx.output[arg->rankId];
        localDst.token = ctx.token[arg->rankId];
        localDst.addr += ctx.offSet;
        CCU_CHK_RET(GroupCopy(ctx, localDst, src, ctx.localGoSize));
    }

    uint16_t rankMask = 0x0000;
    for (uint64_t rankIdx = 0; rankIdx < ctx.rankIdGroup.size(); rankIdx++) {
        rankMask |= (1 << ctx.rankIdGroup[rankIdx]);
    }
    CCU_CHK_RET(ccu::EventWait(ctx.event, rankMask));

    return CCU_SUCCESS;
}

CcuResult CcuAllGather2DiesMeshMem2Mem1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGather2DiesMeshMem2Mem1D *>(arg);

    AllGather2DiesMeshMem2Mem1DContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelAllGather2DiesMeshMem2Mem1D] CcuKernelAllGather2DiesMeshMem2Mem1D run");

    if (kernelArg->rankIdGroup.size() == 0) {
        HCCL_INFO("[CcuKernelAllGather2DiesMeshMem2Mem1D] rankIdGroup empty, skip");
        return CCU_SUCCESS;
    }

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoAllGather(ctx));

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGather2DiesMeshMem2Mem1D] CcuKernelAllGather2DiesMeshMem2Mem1D end");

    return CCU_SUCCESS;
}

}