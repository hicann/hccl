/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "reduce_scatter_birs.h"

namespace ops_hccl_experimental {

ReduceScatterBIRS::ReduceScatterBIRS() : AlgTemplateBaseExperimental() {}

ReduceScatterBIRS::~ReduceScatterBIRS() {}

HcclResult ReduceScatterBIRS::Prepare(u32 interRank, u32 interRankSize)
{
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRS::Prepare(
    HcclMem& inputMem, HcclMem& outputMem, HcclMem& scratchMem, const u64 count, const HcclDataType dataType,
    ThreadHandle thread, const std::vector<ThreadHandle>& slaveThreads, const HcclReduceOp reductionOp, const u32 root,
    const std::vector<Slice>& slices, const u64 baseOffset, const bool disableDMAReduce)
{
    mainThread = thread;
    subThreads = slaveThreads;
    AlgTemplateBase::Prepare(
        inputMem, outputMem, scratchMem, count, dataType, thread, reductionOp, root, slices, baseOffset,
        disableDMAReduce);
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRS::LocalReduceCCLToCCL(u64 srcOffset, u64 dstOffset, u64 size, ThreadHandle thread)
{
    void* srcSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + srcOffset);
    void* dstSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + dstOffset);
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        thread, dstSlice, srcSlice, size / unitSize, static_cast<HcommDataType>(dataType_),
        static_cast<HcommReduceOp>(reductionOp_))));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRS::TreeLocalReduce(const std::vector<u32>& offsets, u64 unitSize, ThreadHandle thread)
{
    u32 ind = static_cast<u32>(offsets.size());
    for (u32 stride = 1; stride < ind; stride *= 2) {
        for (u32 i = stride; i < ind; i += stride * 2) {
            CHK_RET(LocalReduceCCLToCCL(offsets[i], offsets[i - stride], unitSize, thread));
        }
    }
    return HCCL_SUCCESS;
}

void ReduceScatterBIRS::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = 3;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void ReduceScatterBIRS::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = 3;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult ReduceScatterBIRS::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = totalCount * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG(" default slice[%u]: offset: [%llu] size[%llu]", i, i * sliceSize, sliceSize);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRS::Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    if (rankSize == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
        }
        return HCCL_SUCCESS;
    }

    if (channels.size() < rankSize) {
        HCCL_ERROR(
            "[ReduceScatterBIRS][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, channels.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[ReduceScatterBIRS][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, rankSize);
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRS::HCCSProcessMainLoop(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize)
{
    if (round != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 localOffsetByte = hccs_ranks[round - 1] / rankSizeX_ * localStrideSize;
        u64 remoteOffsetByte = ((rankSize / rankSizeX_) + rank / rankSizeX_) * localStrideSize;
        void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void*>(static_cast<u8*>(hccs_links[round - 1].remoteOutput.addr) + remoteOffsetByte);

        HcommWriteOnThread(subThreads[0], hccs_links[round - 1].handle, dst, src, sliceSize);

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRS::SIOProcessMainLoop(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

    u64 localOffsetByte
        = (round != hccs_ranks.size()) ? slices_[hccs_neighbour_rank[round]].offset : slices_[sio_rank].offset;
    u64 remoteOffsetByte = (round != hccs_ranks.size()) ? hccs_ranks[round] / rankSizeX_ * localStrideSize :
                                                          rank / rankSizeX_ * localStrideSize;
    void* src = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + localOffsetByte);
    void* dst = static_cast<void*>(static_cast<u8*>(sio_link.remoteOutput.addr) + remoteOffsetByte);

    HcommWriteReduceOnThread(
        mainThread, sio_link.handle, dst, src, sliceSize / unitSize, static_cast<HcommDataType>(dataType_),
        static_cast<HcommReduceOp>(reductionOp_));

    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRS::LocalCopyMainLoop(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize)
{
    if (round < hccs_ranks.size()) {
        u32 rank_idx = (round < hccs_ranks.size() - 1) ? hccs_ranks[round + 1] : rank;
        void* srcSlice = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + slices_[rank_idx].offset);
        void* dstSlice
            = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + rank_idx / rankSizeX_ * localStrideSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(subThreads[1], dstSlice, srcSlice, sliceSize)));
    }
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterBIRS::FinalStep(const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize)
{
    std::vector<u32> vec;
    for (u32 i = 0; i < (rankSize / rankSizeX_); i++) {
        if (i == (rank / rankSizeX_)) {
            vec.push_back((rank / rankSizeX_) * localStrideSize);
        } else {
            vec.push_back(((rankSize / rankSizeX_) + i) * localStrideSize);
        }
    }
    // Tree local reduce
    CHK_RET(TreeLocalReduce(vec, sliceSize, mainThread));
    // Local copy to output
    void* srcSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + vec[0]);
    void* dstSlice = static_cast<void*>(static_cast<u8*>(outputMem_.addr));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRS::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    HCCL_INFO(
        "ReduceScatterBIRS run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.addr, outputMem_.addr, count_);

    Preprocess(rank, rankSize, channels);

    u32 rankSizeX_ = 2;
    if (rankSize % rankSizeX_ != 0) {
        HCCL_ERROR(
            "[ReduceScatterBIRS][RunAsync]rankSize[%u] is not evenly divisible by rankSizeX_[%u]", rankSize,
            rankSizeX_);
        return HCCL_E_INTERNAL;
    }
    u32 rankSizeY_ = rankSize / rankSizeX_;

    sio_rank = rank ^ 1;
    sio_link = channels[sio_rank];

    for (u32 i = 1; i < rankSize / rankSizeX_; ++i) {
        u32 current_hccs_rank = (rank + rankSizeX_ * i) % (rankSizeX_ * rankSizeY_);
        hccs_ranks.push_back(current_hccs_rank);
        hccs_neighbour_rank.push_back(current_hccs_rank ^ 1);
        hccs_links.push_back(channels[hccs_ranks[i - 1]]);
    }
    hccs_links_reversed.assign(hccs_links.rbegin(), hccs_links.rend());

    u64 sliceSize = count_ * unitSize;
    u64 localStrideSize = RoundUpWithDivisor(sliceSize, HCCL_MIN_SLICE_ALIGN_910B);

    // MainRecordSub + SubWaitMain
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

    void* srcSlice = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + slices_[hccs_ranks[0]].offset);
    void* dstSlice
        = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + hccs_ranks[0] / rankSizeX_ * localStrideSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        // MainRecordSub + SubWaitMain
        CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

        HCCSProcessMainLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

        SIOProcessMainLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

        LocalCopyMainLoop(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

        // SubRecordMain + MainWaitSub
        CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    }

    // MainRecordSub + SubWaitMain
    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

    FinalStep(rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));

    HCCL_INFO("ReduceScatterBIRS finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TEMPLATE_REDUCE_SCATTER_BIRS, ReduceScatterBIRS);
} // namespace ops_hccl_experimental
