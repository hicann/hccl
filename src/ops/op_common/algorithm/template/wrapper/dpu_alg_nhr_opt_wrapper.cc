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
#include "dpu_alg_data_trans_wrapper.h"
#include "exec_timeout_manager.h"
#include "hcomm_primitives.h"

namespace ops_hccl {

// ========== 内部辅助：参数校验与 tx/rx slice 构造 ==========
namespace {

    bool IsSupportedDataType(const HcclDataType dataType)
    {
        return dataType == HCCL_DATA_TYPE_INT8 || dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_INT32
               || dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_FP32
               || dataType == HCCL_DATA_TYPE_BFP16;
    }

    bool IsSupportedReduceOp(const HcclReduceOp reduceOp)
    {
        return reduceOp == HCCL_REDUCE_SUM || reduceOp == HCCL_REDUCE_MAX || reduceOp == HCCL_REDUCE_MIN;
    }

    HcclResult CheckWriteReduceChannel(const ChannelInfo* channel, const char* direction)
    {
        CHK_PRT_RET(
            channel == nullptr, HCCL_ERROR("[DpuWriteReduceTransfer] %s channel is null.", direction), HCCL_E_PTR);
        CHK_PRT_RET(
            !channel->isValid || channel->handle == 0
                || channel->locationType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST
                || channel->protocol != CommProtocol::COMM_PROTOCOL_ROCE,
            HCCL_ERROR("[DpuWriteReduceTransfer] %s channel is not a valid HOST/ROCE channel.", direction),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    HcclResult CheckReduceSlices(
        const std::vector<DataSlice>& srcSlices, const std::vector<DataSlice>& dstSlices, HcclDataType dataType,
        const char* direction)
    {
        CHK_PRT_RET(
            srcSlices.size() != dstSlices.size(),
            HCCL_ERROR(
                "[DpuWriteReduceTransfer] %s src slice num[%zu] does not match dst slice num[%zu].", direction,
                srcSlices.size(), dstSlices.size()),
            HCCL_E_PARA);
        const u64 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
        for (u32 i = 0; i < srcSlices.size(); i++) {
            const DataSlice& srcSlice = srcSlices[i];
            const DataSlice& dstSlice = dstSlices[i];
            CHK_PRT_RET(
                srcSlice.size_ != dstSlice.size_ || srcSlice.count_ != dstSlice.count_,
                HCCL_ERROR(
                    "[DpuWriteReduceTransfer] %s slice[%u] src and dst metadata mismatch: "
                    "srcSize[%llu], dstSize[%llu], srcCount[%llu], dstCount[%llu].",
                    direction, i, srcSlice.size_, dstSlice.size_, srcSlice.count_, dstSlice.count_),
                HCCL_E_PARA);
            if (srcSlice.size_ == 0) {
                CHK_PRT_RET(
                    srcSlice.count_ != 0,
                    HCCL_ERROR(
                        "[DpuWriteReduceTransfer] %s slice[%u] has zero size but non-zero count[%llu].", direction, i,
                        srcSlice.count_),
                    HCCL_E_PARA);
                continue;
            }
            CHK_PRT_RET(
                srcSlice.addr_ == nullptr || dstSlice.addr_ == nullptr,
                HCCL_ERROR("[DpuWriteReduceTransfer] %s slice[%u] has null address.", direction, i), HCCL_E_PTR);
            CHK_PRT_RET(
                srcSlice.count_ > UINT64_MAX / dataTypeSize || srcSlice.count_ * dataTypeSize != srcSlice.size_,
                HCCL_ERROR(
                    "[DpuWriteReduceTransfer] %s slice[%u] count[%llu] does not match size[%llu] for dataType[%d].",
                    direction, i, srcSlice.count_, srcSlice.size_, dataType),
                HCCL_E_PARA);
        }
        return HCCL_SUCCESS;
    }

    HcclResult CheckBufferRange(const HcclMem& buffer, u64 offset, u64 size, const char* bufferName)
    {
        CHK_PRT_RET(
            offset > UINT64_MAX - size, HCCL_ERROR("[CheckBufferRange] %s range overflow.", bufferName), HCCL_E_PARA);
        CHK_PRT_RET(
            size != 0 && offset + size > buffer.size,
            HCCL_ERROR(
                "[CheckBufferRange] %s range[%llu, %llu) exceeds buffer size[%llu].", bufferName, offset, offset + size,
                buffer.size),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    HcclResult CalcRepeatBase(u64 baseOffset, u64 repeatStride, u32 repeatIdx, u64& repeatBase)
    {
        CHK_PRT_RET(
            repeatIdx != 0 && repeatStride > UINT64_MAX / repeatIdx,
            HCCL_ERROR("[ValidateReduceScatterNhrParams] repeat offset overflow, repeatIdx[%u].", repeatIdx),
            HCCL_E_PARA);
        const u64 repeatOffset = repeatStride * repeatIdx;
        CHK_PRT_RET(
            baseOffset > UINT64_MAX - repeatOffset,
            HCCL_ERROR("[ValidateReduceScatterNhrParams] repeat base overflow."), HCCL_E_PARA);
        repeatBase = baseOffset + repeatOffset;
        return HCCL_SUCCESS;
    }

    HcclResult GetWriteReduceChannel(
        const std::map<u32, std::vector<ChannelInfo>>& channels, u32 rank, const ChannelInfo*& channel)
    {
        auto channelIt = channels.find(rank);
        CHK_PRT_RET(
            channelIt == channels.end() || channelIt->second.empty(),
            HCCL_ERROR("[DpuBatchWriteReduceTransferNHR] channel for rank[%u] is missing.", rank), HCCL_E_PARA);
        channel = &channelIt->second.front();
        CHK_PRT_RET(
            channel->remoteCclMem.addr == nullptr,
            HCCL_ERROR("[DpuBatchWriteReduceTransferNHR] remote CCL buffer for rank[%u] is null.", rank), HCCL_E_PTR);
        CHK_PRT_RET(
            channel->remoteCclMem.size == 0,
            HCCL_ERROR("[DpuBatchWriteReduceTransferNHR] remote CCL buffer for rank[%u] has zero capacity.", rank),
            HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    bool HasReduceSlice(const std::vector<u32>& sliceIdxs, const TemplateDataParams& params)
    {
        return std::any_of(sliceIdxs.begin(), sliceIdxs.end(), [&params](u32 sliceIdx) {
            return params.allRankSliceSize.at(sliceIdx) != 0;
        });
    }

    void BuildTxSlices(
        const AicpuNHRStepInfo& stepInfo, const TemplateDataParams& tempAlgParam, u32 repeat, u32 templateRankSize,
        void* sendCclBuffAddr, std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices)
    {
        for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
            u32 txId = stepInfo.txSliceIdxs.at(i);
            u64 sliceSize = tempAlgParam.allRankSliceSize.at(txId);
            u64 sliceCount = tempAlgParam.allRankProcessedDataCount.at(txId);
            u64 sliceOffset = tempAlgParam.allRankDispls.at(txId);
            u64 srcDstOffset
                = repeat * templateRankSize * sliceSize + tempAlgParam.buffInfo.hcclBuffBaseOff + sliceOffset;

            if (sliceSize != 0) {
                txSrcSlices.push_back(
                    DataSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, sliceSize, sliceCount));
                txDstSlices.push_back(DataSlice(sendCclBuffAddr, srcDstOffset, sliceSize, sliceCount));
            }
        }
    }

    void BuildRxSlices(
        const AicpuNHRStepInfo& stepInfo, const TemplateDataParams& tempAlgParam, u32 repeat, u32 templateRankSize,
        void* recvCclBuffAddr, std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices)
    {
        for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
            u32 rxId = stepInfo.rxSliceIdxs.at(i);
            u64 sliceSize = tempAlgParam.allRankSliceSize.at(rxId);
            u64 sliceCount = tempAlgParam.allRankProcessedDataCount.at(rxId);
            u64 sliceOffset = tempAlgParam.allRankDispls.at(rxId);
            u64 srcDstOffset
                = repeat * templateRankSize * sliceSize + tempAlgParam.buffInfo.hcclBuffBaseOff + sliceOffset;

            if (sliceSize != 0) {
                rxSrcSlices.push_back(DataSlice(recvCclBuffAddr, srcDstOffset, sliceSize, sliceCount));
                rxDstSlices.push_back(
                    DataSlice(tempAlgParam.buffInfo.hcclBuff.addr, srcDstOffset, sliceSize, sliceCount));
            }
        }
    }

} // anonymous namespace

HcclResult CheckWriteReduceTransfer(const DpuTransferCtx& transfer, HcclDataType dataType)
{
    CHK_PRT_RET(
        static_cast<u32>(dataType) >= HCCL_DATA_TYPE_RESERVED || DATATYPE_SIZE_TABLE[dataType] == 0,
        HCCL_ERROR("[DpuWriteReduceTransfer] invalid dataType[%d].", dataType), HCCL_E_PARA);
    CHK_RET(CheckReduceSlices(transfer.txSrcSlices, transfer.txDstSlices, dataType, "tx"));
    CHK_RET(CheckReduceSlices(transfer.rxSrcSlices, transfer.rxDstSlices, dataType, "rx"));

    if (transfer.hasSend()) {
        CHK_RET(CheckWriteReduceChannel(transfer.txCh, "tx"));
        for (const DataSlice& slice : transfer.txDstSlices) {
            CHK_RET(CheckBufferRange(transfer.txCh->remoteCclMem, slice.offset_, slice.size_, "remote CCL buffer"));
        }
    }
    if (transfer.hasRecv()) {
        CHK_RET(CheckWriteReduceChannel(transfer.rxCh, "rx"));
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateReduceScatterNhrParams(const TemplateDataParams& params, u32 rankSize)
{
    CHK_PRT_RET(
        !IsSupportedDataType(params.dataType),
        HCCL_ERROR("[RunReduceScatterNhrDpu] unsupported dataType[%d].", params.dataType), HCCL_E_PARA);
    CHK_PRT_RET(
        !IsSupportedReduceOp(params.reduceOp),
        HCCL_ERROR("[RunReduceScatterNhrDpu] unsupported reduceOp[%d].", params.reduceOp), HCCL_E_PARA);
    CHK_PRT_RET(
        params.allRankSliceSize.size() != rankSize || params.allRankDispls.size() != rankSize
            || params.allRankProcessedDataCount.size() != rankSize,
        HCCL_ERROR(
            "[RunReduceScatterNhrDpu] inconsistent slice metadata: rankSize[%u], sizes[%zu], displs[%zu], "
            "counts[%zu].",
            rankSize, params.allRankSliceSize.size(), params.allRankDispls.size(),
            params.allRankProcessedDataCount.size()),
        HCCL_E_PARA);
    const u64 dataTypeSize = DATATYPE_SIZE_TABLE[params.dataType];
    bool hasData = false;
    for (u32 sliceIdx = 0; sliceIdx < rankSize; ++sliceIdx) {
        const u64 size = params.allRankSliceSize[sliceIdx];
        hasData = hasData || size != 0;
        const u64 count = params.allRankProcessedDataCount[sliceIdx];
        CHK_PRT_RET(
            count > UINT64_MAX / dataTypeSize || count * dataTypeSize != size,
            HCCL_ERROR(
                "[RunReduceScatterNhrDpu] slice[%u] count[%llu] does not match size[%llu] for dataType[%d].", sliceIdx,
                count, size, params.dataType),
            HCCL_E_PARA);
        CHK_PRT_RET(
            params.allRankDispls[sliceIdx] > UINT64_MAX - size,
            HCCL_ERROR("[RunReduceScatterNhrDpu] slice[%u] offset overflow.", sliceIdx), HCCL_E_PARA);
        CHK_PRT_RET(
            size > UINT64_MAX / rankSize,
            HCCL_ERROR("[ValidateReduceScatterNhrParams] slice[%u] scratch repeat stride overflow.", sliceIdx),
            HCCL_E_PARA);
    }
    if (!hasData || params.repeatNum == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        params.buffInfo.hcclBuff.addr == nullptr, HCCL_ERROR("[RunReduceScatterNhrDpu] local CCL buffer is null."),
        HCCL_E_PTR);
    for (u32 repeatIdx = 0; repeatIdx < params.repeatNum; ++repeatIdx) {
        for (u32 sliceIdx = 0; sliceIdx < rankSize; ++sliceIdx) {
            const u64 scratchRepeatStride = rankSize * params.allRankSliceSize[sliceIdx];
            u64 scratchBase = 0;
            CHK_RET(CalcRepeatBase(params.buffInfo.hcclBuffBaseOff, scratchRepeatStride, repeatIdx, scratchBase));
            CHK_PRT_RET(
                scratchBase > UINT64_MAX - params.allRankDispls[sliceIdx],
                HCCL_ERROR("[ValidateReduceScatterNhrParams] slice[%u] scratch offset overflow.", sliceIdx),
                HCCL_E_PARA);
            CHK_RET(CheckBufferRange(
                params.buffInfo.hcclBuff, scratchBase + params.allRankDispls[sliceIdx],
                params.allRankSliceSize[sliceIdx], "local CCL buffer"));
        }
    }
    return HCCL_SUCCESS;
}

// ========== NHR 批量传输：覆盖只发 / 只收 / 同对端 / 不同对端四种场景 ==========

HcclResult BatchTransferNHR(
    const AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const TemplateDataParams& tempAlgParam, u32 repeat, u32 myRank, u32 templateRankSize)
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
        HCCL_ERROR(
            "[BatchTransferNHR] tx channel not found for rank=%u, channels size=%zu", stepInfo.toRank, channels.size());
        return HCCL_E_PARA;
    }
    if (hasRx && rxIt == channels.end()) {
        HCCL_ERROR(
            "[BatchTransferNHR] rx channel not found for rank=%u, channels size=%zu", stepInfo.fromRank,
            channels.size());
        return HCCL_E_PARA;
    }
    const ChannelInfo* txCh = hasTx ? &txIt->second[0] : nullptr;
    const ChannelInfo* rxCh = hasRx ? &rxIt->second[0] : nullptr;

    // ====== Phase 1+2+3: 构造 slice 并委托 DpuBatchTransfer（内含 step sync）======
    void* sendCclBuffAddr = hasTx ? txCh->remoteCclMem.addr : nullptr;
    void* recvCclBuffAddr = hasRx ? rxCh->remoteCclMem.addr : nullptr;

    DpuTransferCtx ctx;
    ctx.txCh = txCh;
    ctx.rxCh = rxCh;
    if (hasTx) {
        BuildTxSlices(
            stepInfo, tempAlgParam, repeat, templateRankSize, sendCclBuffAddr, ctx.txSrcSlices, ctx.txDstSlices);
        for (u32 i = 0; i < ctx.txSrcSlices.size(); i++) {
            HCCL_INFO(
                "[BatchTransferNHR][Tx] toRank=%u slice=%u/%zu size=%llu", stepInfo.toRank, i, ctx.txSrcSlices.size(),
                ctx.txSrcSlices[i].size_);
        }
    }
    if (hasRx) {
        BuildRxSlices(
            stepInfo, tempAlgParam, repeat, templateRankSize, recvCclBuffAddr, ctx.rxSrcSlices, ctx.rxDstSlices);
    }
    std::vector<DpuTransferCtx> pairs = {ctx};
    CHK_RET(DpuBatchTransfer(pairs));
#endif
    return HCCL_SUCCESS;
}

HcclResult DpuBatchWriteReduceTransferNHR(
    const AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const TemplateDataParams& tempAlgParam, u32 repeat, u32 templateRankSize)
{
#ifndef AICPU_COMPILE
    const auto validSliceIndex = [&](u32 sliceIdx) {
        return sliceIdx < templateRankSize && sliceIdx < tempAlgParam.allRankSliceSize.size()
               && sliceIdx < tempAlgParam.allRankDispls.size()
               && sliceIdx < tempAlgParam.allRankProcessedDataCount.size();
    };
    CHK_PRT_RET(
        !std::all_of(stepInfo.txSliceIdxs.begin(), stepInfo.txSliceIdxs.end(), validSliceIndex)
            || !std::all_of(stepInfo.rxSliceIdxs.begin(), stepInfo.rxSliceIdxs.end(), validSliceIndex),
        HCCL_ERROR("[DpuBatchWriteReduceTransferNHR] step slice index exceeds rank or metadata bounds."), HCCL_E_PARA);
    const bool hasTx = HasReduceSlice(stepInfo.txSliceIdxs, tempAlgParam);
    const bool hasRx = HasReduceSlice(stepInfo.rxSliceIdxs, tempAlgParam);
    if (!hasTx && !hasRx) {
        return HCCL_SUCCESS;
    }

    DpuTransferCtx ctx{};
    if (hasTx) {
        CHK_RET(GetWriteReduceChannel(channels, stepInfo.toRank, ctx.txCh));
    }
    if (hasRx) {
        CHK_RET(GetWriteReduceChannel(channels, stepInfo.fromRank, ctx.rxCh));
    }
    if (hasTx) {
        BuildTxSlices(
            stepInfo, tempAlgParam, repeat, templateRankSize, ctx.txCh->remoteCclMem.addr, ctx.txSrcSlices,
            ctx.txDstSlices);
        for (u32 i = 0; i < ctx.txSrcSlices.size(); i++) {
            HCCL_INFO(
                "[DpuBatchWriteReduceTransferNHR][Tx] toRank=%u slice=%u/%zu size=%llu", stepInfo.toRank, i,
                ctx.txSrcSlices.size(), ctx.txSrcSlices[i].size_);
        }
    }
    if (hasRx) {
        BuildRxSlices(
            stepInfo, tempAlgParam, repeat, templateRankSize, ctx.rxCh->remoteCclMem.addr, ctx.rxSrcSlices,
            ctx.rxDstSlices);
    }
    CHK_RET(CheckWriteReduceTransfer(ctx, tempAlgParam.dataType));
    CHK_RET(DpuWriteReduceTransfer(ctx, tempAlgParam.dataType, tempAlgParam.reduceOp));
#else
    (void)stepInfo;
    (void)channels;
    (void)tempAlgParam;
    (void)repeat;
    (void)templateRankSize;
#endif
    return HCCL_SUCCESS;
}

// ========== 三阶段批量传输 ==========

HcclResult DpuBatchTransfer(std::vector<DpuTransferCtx>& pairs)
{
#ifndef AICPU_COMPILE
    if (pairs.empty()) {
        return HCCL_SUCCESS;
    }
    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();

    // ====== Phase 1: 前同步 — 批量 rx Record + tx Wait ======
    for (auto& p : pairs) {
        if (p.hasRecv()) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, p.rxCh->handle, NOTIFY_IDX_STEP_SYNC)));
        }
    }
    for (auto& p : pairs) {
        if (p.hasSend()) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(0, p.txCh->handle, NOTIFY_IDX_STEP_SYNC, execTimeout)));
        }
    }

    // ====== Phase 2: 批量写数据（HcommWriteNbiOnThread，不混入 Record）======
    for (auto& p : pairs) {
        if (p.hasSend()) {
            for (u32 i = 0; i < p.txSrcSlices.size(); i++) {
                void* dst = static_cast<s8*>(p.txDstSlices[i].addr_) + p.txDstSlices[i].offset_;
                void* src = static_cast<s8*>(p.txSrcSlices[i].addr_) + p.txSrcSlices[i].offset_;
                CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyNbiOnThread(
                    0, p.txCh->handle, dst, src, p.txSrcSlices[i].size_, NOTIFY_IDX_DATA_SIGNAL)));
            }
        }
    }

    // ====== Phase 3: 后同步 — Drain/Fence → NotifyWait(DATA_SIGNAL) ======
    // Drain：发送通道 + 接收通道（去重，samePeer 时仅一次）
    for (auto& p : pairs) {
        bool txDrained = false;
        if (p.hasSend()) {
            CHK_RET(HcommChannelDrainOnThreadWithCompat(0, p.txCh->handle));
            txDrained = true;
        }
        if (p.hasRecv() && (!txDrained || p.rxCh->handle != p.txCh->handle)) {
            CHK_RET(HcommChannelDrainOnThreadWithCompat(0, p.rxCh->handle));
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
    for (auto& p : pairs) {
        if (p.hasRecv()) {
            for (u32 i = 0; i < p.rxSrcSlices.size(); i++) {
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(0, p.rxCh->handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
            }
        }
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult DpuWriteReduceTransfer(const DpuTransferCtx& transfer, HcclDataType dataType, HcclReduceOp reduceOp)
{
#ifndef AICPU_COMPILE
    const bool hasSend = transfer.hasSend();
    const bool hasRecv = transfer.hasRecv();
    if (!hasSend && !hasRecv) {
        return HCCL_SUCCESS;
    }

    const u32 execTimeout = ExecTimeoutManager::Instance().GetExecTimeout();
    if (hasRecv) {
        CHK_RET(
            static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, transfer.rxCh->handle, NOTIFY_IDX_STEP_SYNC)));
    }
    if (hasSend) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(0, transfer.txCh->handle, NOTIFY_IDX_STEP_SYNC, execTimeout)));
    }

    if (hasSend) {
        for (u32 i = 0; i < transfer.txSrcSlices.size(); i++) {
            const DataSlice& srcSlice = transfer.txSrcSlices[i];
            if (srcSlice.size_ == 0) {
                continue;
            }
            const DataSlice& dstSlice = transfer.txDstSlices[i];
            void* dst = static_cast<s8*>(dstSlice.addr_) + dstSlice.offset_;
            const void* src = static_cast<s8*>(srcSlice.addr_) + srcSlice.offset_;
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceWithNotifyOnThread(
                0, transfer.txCh->handle, dst, src, srcSlice.count_, static_cast<HcommDataType>(dataType),
                static_cast<HcommReduceOp>(reduceOp), NOTIFY_IDX_DATA_SIGNAL)));
        }
    }

    bool txDrained = false;
    if (hasSend) {
        CHK_RET(HcommChannelDrainOnThreadWithCompat(0, transfer.txCh->handle));
        txDrained = true;
    }
    if (hasRecv && (!txDrained || transfer.rxCh->handle != transfer.txCh->handle)) {
        CHK_RET(HcommChannelDrainOnThreadWithCompat(0, transfer.rxCh->handle));
    }
    CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));

    if (hasRecv) {
        for (const DataSlice& slice : transfer.rxSrcSlices) {
            if (slice.size_ == 0) {
                continue;
            }
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(0, transfer.rxCh->handle, NOTIFY_IDX_DATA_SIGNAL, execTimeout)));
        }
    }
#endif
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
