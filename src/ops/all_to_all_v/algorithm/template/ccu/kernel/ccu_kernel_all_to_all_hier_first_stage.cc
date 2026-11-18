/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_to_all_hier_first_stage.h"

namespace ops_hccl {

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int POST_SYNC_ID = 3;
constexpr int CKE_IDX_0 = 0;

static CcuResult InitResource(AllToAllHierFirstStageContext& ctx)
{
    const auto* arg = ctx.arg;
    if (arg->groupSize == 0 || arg->peerSliceNum == 0) {
        HCCL_ERROR(
            "[CcuAllToAllHierFirstStageKernel][%s] invalid groupSize[%u] or peerSliceNum[%u].", __func__,
            arg->groupSize, arg->peerSliceNum);
        return CcuResult::CCU_E_INTERNAL;
    }
    if (arg->groupSize > 1 && arg->channelCount != arg->groupSize - 1) {
        HCCL_ERROR(
            "[CcuAllToAllHierFirstStageKernel][%s] channelCount[%u] != groupSize-1[%u].", __func__, arg->channelCount,
            arg->groupSize - 1);
        return CcuResult::CCU_E_INTERNAL;
    }
    if (arg->myAlgRank >= arg->groupSize) {
        HCCL_ERROR(
            "[CcuAllToAllHierFirstStageKernel][%s] myAlgRank[%u] >= groupSize[%u].", __func__, arg->myAlgRank,
            arg->groupSize);
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.remoteCclBuff.reserve(arg->channelCount);
    ctx.peerToken.reserve(arg->channelCount);
    for (u64 i = 0; i < arg->channelCount; i++) {
        ctx.remoteCclBuff.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[i], OUTPUT_XN_ID));
        ctx.peerToken.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[i], TOKEN_XN_ID));
    }
    ctx.eventGroup.Init((arg->groupSize - 1) * arg->peerSliceNum);
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(AllToAllHierFirstStageContext& ctx)
{
    uint32_t cnt = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCclBuff, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myToken, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.segSize, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inBuffBaseOff, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.hcclBuffBaseOff, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.blockStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.slotStride, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.addrOffset, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.loopParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.parallelParam, cnt++));
    CCU_CHK_RET(ccu::LoadArg(ctx.groupOpSize.residual, cnt++));
    return CCU_SUCCESS;
}

static void PreSync(AllToAllHierFirstStageContext& ctx)
{
    const auto* arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.localCclBuff, OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID);
        ccu::WriteVariableWithNotify(arg->channels[i], ctx.myToken, TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID);
    }
    uint32_t waitBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, waitBits);
    }
}

static void PostSync(AllToAllHierFirstStageContext& ctx)
{
    const auto* arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

template <typename DstAddr>
static void CalcColAddrs(
    AllToAllHierFirstStageContext& ctx, uint32_t srcColIdx, const ccu::Variable& dstBase, const ccu::Variable& dstToken,
    bool withSliceOffset, ccu::LocalAddr& src, DstAddr& dst)
{
    const auto* arg = ctx.arg;
    // src 列基址 = INPUT + inBuffBaseOff [+ sliceOffset] + srcColIdx*peerSliceNum*blockStride
    const uint64_t srcBlkIdx = static_cast<uint64_t>(srcColIdx) * arg->peerSliceNum;
    ctx.offScratch = 0;
    for (uint64_t i = 0; i < srcBlkIdx; i++) {
        ctx.offScratch += ctx.blockStride;
    }
    src.token = ctx.myToken;
    src.addr = ctx.input;
    src.addr += ctx.inBuffBaseOff;
    if (withSliceOffset) {
        src.addr += ctx.sliceOffset;
    }
    src.addr += ctx.offScratch;

    // dst 列基址 = dstBase + hcclBuffBaseOff [+ sliceOffset] + myAlgRank*slotStride
    ctx.offScratch = 0;
    for (uint32_t i = 0; i < arg->myAlgRank; i++) {
        ctx.offScratch += ctx.slotStride;
    }
    dst.token = dstToken;
    dst.addr = dstBase;
    dst.addr += ctx.hcclBuffBaseOff;
    if (withSliceOffset) {
        dst.addr += ctx.sliceOffset;
    }
    dst.addr += ctx.offScratch;
}

static void CalcSrcDstAddrs(
    AllToAllHierFirstStageContext& ctx, std::vector<ccu::LocalAddr>& src, std::vector<ccu::RemoteAddr>& dst,
    ccu::Variable& dstSlotAdv)
{
    const auto* arg = ctx.arg;
    uint32_t channelIdx = 0;
    for (uint32_t p = 0; p < arg->groupSize; p++) {
        if (p == arg->myAlgRank) {
            continue;
        }
        CalcColAddrs(
            ctx, p, ctx.remoteCclBuff[channelIdx], ctx.peerToken[channelIdx], true, src[channelIdx], dst[channelIdx]);
        channelIdx++;
    }

    // 槽位步长 = groupSize*slotStride（远端/本地共用）
    dstSlotAdv = 0;
    for (uint32_t i = 0; i < arg->groupSize; i++) {
        dstSlotAdv += ctx.slotStride;
    }
}

static CcuResult DoAlltoAll(AllToAllHierFirstStageContext& ctx)
{
    const auto* arg = ctx.arg;
    std::vector<ccu::LocalAddr> src(arg->channelCount);
    std::vector<ccu::RemoteAddr> dst(arg->channelCount);
    ccu::Variable dstSlotAdv;
    CalcSrcDstAddrs(ctx, src, dst, dstSlotAdv);

    uint32_t channelIdx = 0;
    uint32_t sigIdx = 0;
    for (uint32_t p = 0; p < arg->groupSize; p++) {
        if (p == arg->myAlgRank) {
            continue;
        }
        uint32_t batchStart = sigIdx;
        for (uint32_t i = 0; i < arg->peerSliceNum; i++) {
            if (i > 0) {
                src[channelIdx].addr += ctx.blockStride;
                dst[channelIdx].addr += dstSlotAdv;
            }
            CCU_IF(ctx.segSize != 0)
            {
                CCU_CHK_RET(ccu::Write(
                    arg->channels[channelIdx], dst[channelIdx], src[channelIdx], ctx.segSize,
                    ctx.eventGroup.GetEvent(sigIdx), ctx.eventGroup.GetMask(sigIdx)));
            }
            CCU_ELSE { CCU_CHK_RET(ctx.eventGroup.Record(sigIdx)); }
            sigIdx++;
        }
        CCU_CHK_RET(ctx.eventGroup.WaitRange(batchStart, sigIdx));
        channelIdx++;
    }

    ccu::LocalAddr localSrc;
    ccu::LocalAddr localDst;
    CalcColAddrs(ctx, arg->myAlgRank, ctx.localCclBuff, ctx.myToken, true, localSrc, localDst);
    for (uint32_t i = 0; i < arg->peerSliceNum; i++) {
        if (i > 0) {
            localSrc.addr += ctx.blockStride;
            localDst.addr += dstSlotAdv;
        }
        ctx.tmpSrc = localSrc;
        ctx.tmpDst = localDst;
        CCU_CHK_RET(GroupCopy(ctx, ctx.tmpDst, ctx.tmpSrc, ctx.groupOpSize, GetCcuVersion()));
    }
    return CCU_SUCCESS;
}

CcuResult CcuAllToAllHierFirstStageKernel(CcuKernelArg arg)
{
    auto* kernelArg = static_cast<CcuKernelArgAllToAllHierFirstStage*>(arg);
    AllToAllHierFirstStageContext ctx;
    ctx.arg = kernelArg;
    ctx.resourceAllocated = false;
    ctx.moConfig.msInterleave = 0;
    ctx.moConfig.loopCount = 0;
    ctx.moConfig.memSlice = 0;
    ctx.moRes.eventCount = 0;
    ctx.moRes.bufCount = 0;
    ctx.enginePool = 0;

    HCCL_INFO(
        "[%s] AllToAllHierFirstStage run. rankId[%u], myAlgRank[%u], groupSize[%u], peerSliceNum[%u]", __func__,
        kernelArg->rankId, kernelArg->myAlgRank, kernelArg->groupSize, kernelArg->peerSliceNum);
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    PreSync(ctx);
    CCU_CHK_RET(DoAlltoAll(ctx));
    PostSync(ctx);
    HCCL_INFO("[%s]AllToAllHierFirstStage end.", __func__);
    return CCU_SUCCESS;
}

} // namespace ops_hccl
