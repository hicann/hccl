/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_alg_nhr_opt_wrapper.h"
#include "hcomm_primitives.h"

namespace ops_hccl {

// ========== 内部辅助：构造 tx/rx slice 列表 ==========
namespace {

void BuildTxSlices(
    const AicpuNHRStepInfo &stepInfo,
    const TemplateDataParams &tempAlgParam,
    u32 repeat,
    u32 templateRankSize,
    void *sendCclBuffAddr,
    std::vector<DataSlice> &txSrcSlices,
    std::vector<DataSlice> &txDstSlices)
{
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs.at(i);
        u64 sliceSize = tempAlgParam.allRankSliceSize.at(txId);
        u64 sliceCount = tempAlgParam.allRankProcessedDataCount.at(txId);
        u64 sliceOffset = tempAlgParam.allRankDispls.at(txId);
        u64 srcDstOffset = repeat * templateRankSize * sliceSize +
            tempAlgParam.buffInfo.hcclBuffBaseOff + sliceOffset;

        if (sliceSize != 0) {
            txSrcSlices.push_back(
                DataSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, sliceSize, sliceCount));
            txDstSlices.push_back(
                DataSlice(sendCclBuffAddr, srcDstOffset, sliceSize, sliceCount));
        }
    }
}

void BuildRxSlices(
    const AicpuNHRStepInfo &stepInfo,
    const TemplateDataParams &tempAlgParam,
    u32 repeat,
    u32 templateRankSize,
    void *recvCclBuffAddr,
    std::vector<DataSlice> &rxSrcSlices,
    std::vector<DataSlice> &rxDstSlices)
{
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs.at(i);
        u64 sliceSize = tempAlgParam.allRankSliceSize.at(rxId);
        u64 sliceCount = tempAlgParam.allRankProcessedDataCount.at(rxId);
        u64 sliceOffset = tempAlgParam.allRankDispls.at(rxId);
        u64 srcDstOffset = repeat * templateRankSize * sliceSize +
            tempAlgParam.buffInfo.hcclBuffBaseOff + sliceOffset;

        if (sliceSize != 0) {
            rxSrcSlices.push_back(
                DataSlice(recvCclBuffAddr, srcDstOffset, sliceSize, sliceCount));
            rxDstSlices.push_back(
                DataSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, sliceSize, sliceCount));
        }
    }
}

}  // anonymous namespace

// ========== NHR 批量传输：覆盖只发 / 只收 / 同对端 / 不同对端四种场景 ==========

HcclResult BatchTransferNHR(
    const AicpuNHRStepInfo &stepInfo,
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const TemplateDataParams &tempAlgParam,
    u32 repeat,
    u32 myRank,
    u32 templateRankSize)
{
#ifndef AICPU_COMPILE
    bool hasTx = stepInfo.txSliceIdxs.size() > 0;
    bool hasRx = stepInfo.rxSliceIdxs.size() > 0;

    // 空闲 rank，直接跳过
    if (!hasTx && !hasRx) {
        return HCCL_SUCCESS;
    }

    auto txIt = hasTx ? channels.find(stepInfo.toRank) : channels.end();
    auto rxIt = hasRx ? channels.find(stepInfo.fromRank) : channels.end();
    if (hasTx && txIt == channels.end()) {
        HCCL_ERROR("[BatchTransferNHR] tx channel not found for rank=%u, channels size=%zu",
            stepInfo.toRank, channels.size());
        return HCCL_E_PARA;
    }
    if (hasRx && rxIt == channels.end()) {
        HCCL_ERROR("[BatchTransferNHR] rx channel not found for rank=%u, channels size=%zu",
            stepInfo.fromRank, channels.size());
        return HCCL_E_PARA;
    }
    const ChannelInfo *txCh = hasTx ? &txIt->second[0] : nullptr;
    const ChannelInfo *rxCh = hasRx ? &rxIt->second[0] : nullptr;

    // ====== Phase 1+2+3: 构造 slice 并委托 DpuBatchTransfer（内含 step sync）======
    void *sendCclBuffAddr = hasTx ? txCh->remoteCclMem.addr : nullptr;
    void *recvCclBuffAddr = hasRx ? rxCh->remoteCclMem.addr : nullptr;

    DpuTransferCtx ctx;
    ctx.txCh = txCh;
    ctx.rxCh = rxCh;
    if (hasTx) {
        BuildTxSlices(stepInfo, tempAlgParam, repeat, templateRankSize,
            sendCclBuffAddr, ctx.txSrcSlices, ctx.txDstSlices);
        for (u32 i = 0; i < ctx.txSrcSlices.size(); i++) {
            HCCL_INFO("[BatchTransferNHR][Tx] toRank=%u slice=%u/%zu size=%llu",
                stepInfo.toRank, i, ctx.txSrcSlices.size(), ctx.txSrcSlices[i].size_);
        }
    }
    if (hasRx) {
        BuildRxSlices(stepInfo, tempAlgParam, repeat, templateRankSize,
            recvCclBuffAddr, ctx.rxSrcSlices, ctx.rxDstSlices);
    }
    std::vector<DpuTransferCtx> pairs = {ctx};
    CHK_RET(DpuBatchTransfer(pairs));
#endif
    return HCCL_SUCCESS;
}

// ========== 三阶段批量传输 ==========

HcclResult DpuBatchTransfer(std::vector<DpuTransferCtx> &pairs)
{
#ifndef AICPU_COMPILE
    if (pairs.empty()) {
        return HCCL_SUCCESS;
    }

    // ====== Phase 1: 前同步 — 批量 rx Record + tx Wait ======
    for (auto &p : pairs) {
        if (p.hasRecv()) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(0, p.rxCh->handle, NOTIFY_IDX_STEP_SYNC)));
        }
    }
    for (auto &p : pairs) {
        if (p.hasSend()) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(0, p.txCh->handle, NOTIFY_IDX_STEP_SYNC, STEP_SYNC_TIMEOUT)));
        }
    }

    // ====== Phase 2: 批量写数据（HcommWriteNbiOnThread，不混入 Record）======
    for (auto &p : pairs) {
        if (p.hasSend()) {
            for (u32 i = 0; i < p.txSrcSlices.size(); i++) {
                void *dst = static_cast<s8 *>(p.txDstSlices[i].addr_) + p.txDstSlices[i].offset_;
                void *src = static_cast<s8 *>(p.txSrcSlices[i].addr_) + p.txSrcSlices[i].offset_;
                CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyNbiOnThread(
                    0, p.txCh->handle, dst, src, p.txSrcSlices[i].size_, NOTIFY_IDX_DATA_SIGNAL)));
            }
        }
    }

    // ====== Phase 3: 后同步 — 仅实际有数据时 Record/Wait/Fence ======
    for (auto &p : pairs) {
        if (p.hasRecv()) {
            for (u32 i = 0; i < p.rxSrcSlices.size(); i++) {
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(0, p.rxCh->handle, NOTIFY_IDX_DATA_SIGNAL, STEP_SYNC_TIMEOUT)));
            }
        }
    }
    // Fence：发送通道 + 接收通道（去重，samePeer 时仅一次）
    for (auto &p : pairs) {
        if (p.hasSend() && !p.txSrcSlices.empty()) {
            CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, p.txCh->handle)));
        }
        if (p.hasRecv() && !p.rxSrcSlices.empty() && p.rxCh != p.txCh) {
            CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, p.rxCh->handle)));
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
#endif
    return HCCL_SUCCESS;
}

}  // namespace ops_hccl
