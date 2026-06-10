/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_2dies_mesh_1D.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

static CcuResult ParseKernelArg(AllGather2DiesMesh1DContext &ctx, CcuKernelArgAllGather2DiesMesh1D *kernelArg)
{
    ctx.arg = kernelArg;
    ctx.rankIdGroup = kernelArg->rankIdGroup;
    ctx.ifHandleSelfRank = kernelArg->ifHandleSelfRank;
    return CCU_SUCCESS;
}

static CcuResult InitResource(AllGather2DiesMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->channelCount == 0) {
        HCCL_ERROR("[CcuKernelAllGather2DiesMesh1D] channels is empty!");
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.input = ccu::Variable(); // 两个kernel共用一个input

    ctx.output.resize(arg->dimSize);
    ctx.token.resize(arg->dimSize);

    uint32_t channelIdx = 0;
    for (uint64_t peerId = 0; peerId < arg->dimSize; peerId++) {
        if (peerId != arg->rankId) {
            if (peerId == ctx.rankIdGroup[channelIdx]) {
                HCCL_DEBUG("[CcuKernelAllGather2DiesMesh1D] MyRank[%u], PeerId[%llu], ChannelId[%u]",
                    arg->rankId, peerId, channelIdx);
                ctx.output[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
                ctx.token[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
                if (channelIdx < ctx.rankIdGroup.size() - 1) {
                    channelIdx++;
                }
            } else {
                // 非本die的rank，本地资源默认构造即可
            }
        }
    }
    // 本rank固定放在末尾 (本地资源，默认构造)
    // offSet, sliceSize, goSize 默认构造
    ctx.offSet = ccu::Variable();
    ctx.sliceSize = ccu::Variable();

    ctx.resourceAllocated = false;
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllGather2DiesMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t cnt = 0;

    CCU_CHK_RET(ccu::LoadArg(ctx.input, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[arg->rankId], cnt++)); // 最后一个是自己的地址
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.offSet, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.token[arg->rankId], cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, cnt++));

    return CCU_SUCCESS;
}

static CcuResult PreSync(AllGather2DiesMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;

    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.output[arg->rankId],
            OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[i], ctx.token[arg->rankId],
            TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    uint32_t allBit = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(AllGather2DiesMesh1DContext &ctx)
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

static CcuResult DoAllGather(AllGather2DiesMesh1DContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr src;
    ccu::LocalAddr localDst;
    std::vector<ccu::RemoteAddr> dst;
    dst.resize(ctx.rankIdGroup.size());

    src.addr = ctx.input;
    src.token = ctx.token[arg->rankId];

    uint32_t dstId = 0;
    for (uint64_t rankIdx = 0; rankIdx < arg->dimSize; rankIdx++) {
        if (rankIdx != arg->rankId) {
            if (rankIdx == ctx.rankIdGroup[dstId]) {
                dst[dstId].addr = ctx.output[rankIdx];
                dst[dstId].addr += ctx.offSet;
                dst[dstId].token = ctx.token[rankIdx];
                dstId++;
            }
        } else {
            localDst.addr = ctx.output[rankIdx];
            localDst.addr += ctx.offSet;
            localDst.token = ctx.token[rankIdx];
        }
    }

    if (ctx.ifHandleSelfRank) {
        CCU_CHK_RET(GroupBroadcast(ctx, arg->channels, arg->channelCount,
            localDst, dst, src, ctx.goSize));
    } else {
        CCU_CHK_RET(GroupBroadcastWithoutMyRank(ctx, arg->channels, arg->channelCount,
            dst, src, ctx.goSize));
    }

    return CCU_SUCCESS;
}

CcuResult CcuAllGather2DiesMesh1DKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGather2DiesMesh1D *>(arg);

    if (kernelArg->rankIdGroup.size() == 0) {
        HCCL_INFO("[CcuKernelAllGather2DiesMesh1D] rankIdGroup empty, skip");
        return CCU_SUCCESS;
    }

    AllGather2DiesMesh1DContext ctx;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;

    HCCL_INFO("[CcuKernelAllGather2DiesMesh1D] CcuKernelAllGather2DiesMesh1D run");

    CCU_CHK_RET(ParseKernelArg(ctx, kernelArg));
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));

    CCU_CHK_RET(PreSync(ctx));

    CCU_CHK_RET(DoAllGather(ctx));

    CCU_CHK_RET(PostSync(ctx));
    HCCL_INFO("[CcuKernelAllGather2DiesMesh1D] CcuKernelAllGather2DiesMesh1D end");

    return CCU_SUCCESS;
}

} // namespace ops_hccl