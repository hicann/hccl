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
#include "all_reduce_biar.h"

namespace ops_hccl_experimental {

AllReduceBIAR::AllReduceBIAR() : AlgTemplateBaseExperimental() {}

AllReduceBIAR::~AllReduceBIAR() {}

HcclResult AllReduceBIAR::Prepare(u32 interRank, u32 interRankSize)
{
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::Prepare(
    HcclMem& inputMem, HcclMem& outputMem, HcclMem& scratchMem, const u64 count, const HcclDataType dataType,
    ThreadHandle thread, const std::vector<ThreadHandle>& slaveThreads, const HcclReduceOp reductionOp, const u32 root,
    const std::vector<Slice>& slices, const u64 baseOffset, const bool disableDMAReduce)
{
    mainThread_ = thread;
    subThreads_ = slaveThreads;
    AlgTemplateBase::Prepare(
        inputMem, outputMem, scratchMem, count, dataType, thread, reductionOp, root, slices, baseOffset,
        disableDMAReduce);
    return HCCL_SUCCESS;
}

void AllReduceBIAR::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = BIAR_THREAD_NUM;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void AllReduceBIAR::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = BIAR_THREAD_NUM;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

HcclResult AllReduceBIAR::Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
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
            "[AllReduceBIAR][Preprocess]rank[%u] linksize[%llu] is less than rankSize[%u]", rank, channels.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    unitSize_ = DataUnitSize(dataType_);
    if (unitSize_ == 0) {
        HCCL_ERROR("[AllReduceBIAR][Preprocess]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::LocalReduceCCLToCCL(u64 srcOffset, u64 dstOffset, u64 size, ThreadHandle thread)
{
    void* srcSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + srcOffset);
    void* dstSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + dstOffset);
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        thread, dstSlice, srcSlice, size / unitSize_, static_cast<HcommDataType>(dataType_),
        static_cast<HcommReduceOp>(reductionOp_))));
    return HCCL_SUCCESS;
}

HcclResult
AllReduceBIAR::HCCSProcessRsLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX, u64 localStrideSize)
{
    if (round != 0 && round <= hccs_ranks.size()) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads_[0], hccs_links[round - 1].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads_[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 localOffsetByte = hccs_ranks[round - 1] / rankSizeX * localStrideSize;
        u64 remoteOffsetByte = ((rankSize / rankSizeX) + rank / rankSizeX) * localStrideSize;
        void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void*>(static_cast<u8*>(hccs_links[round - 1].remoteOutput.addr) + remoteOffsetByte);

        if (slices_[hccs_ranks[round - 1]].size > 0) {
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                subThreads_[0], hccs_links[round - 1].handle, dst, src, slices_[hccs_ranks[round - 1]].size)));
        }

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads_[0], hccs_links[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads_[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult
AllReduceBIAR::SIOProcessRsLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX, u64 localStrideSize)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread_, sio_link.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread_, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

    u64 localOffsetByte
        = (round < hccs_ranks.size()) ? slices_[hccs_neighbour_rank[round]].offset : slices_[sio_rank_].offset;
    u64 remoteOffsetByte = (round < hccs_ranks.size()) ? hccs_ranks[round] / rankSizeX * localStrideSize :
                                                         rank / rankSizeX * localStrideSize;
    u64 curSliceSize = (round < hccs_ranks.size()) ? slices_[hccs_neighbour_rank[round]].size : slices_[sio_rank_].size;
    void* src = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + localOffsetByte);
    void* dst = static_cast<void*>(static_cast<u8*>(sio_link.remoteOutput.addr) + remoteOffsetByte);

    if (curSliceSize > 0) {
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(
            mainThread_, sio_link.handle, dst, src, curSliceSize / unitSize_, static_cast<HcommDataType>(dataType_),
            static_cast<HcommReduceOp>(reductionOp_))));
    }

    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(mainThread_, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread_, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult
AllReduceBIAR::LocalCopyRsLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX, u64 localStrideSize)
{
    if (round < hccs_ranks.size()) {
        u32 rank_idx = (round + 1 < hccs_ranks.size()) ? hccs_ranks[round + 1] : rank;
        void* srcSlice = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + slices_[rank_idx].offset);
        void* dstSlice
            = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + rank_idx / rankSizeX * localStrideSize);
        if (slices_[rank_idx].size > 0) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(subThreads_[1], dstSlice, srcSlice, slices_[rank_idx].size)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::TreeReduceRs(const u32 rank, const u32 rankSize, u32 rankSizeX, u64 localStrideSize)
{
    std::vector<u64> vec;
    for (u32 i = 0; i < (rankSize / rankSizeX); i++) {
        if (i == (rank / rankSizeX)) {
            vec.push_back((rank / rankSizeX) * localStrideSize);
        } else {
            vec.push_back(((rankSize / rankSizeX) + i) * localStrideSize);
        }
    }
    // Tree local reduce
    auto ind = rankSize / rankSizeX;
    for (u32 stride = 1; stride < ind; stride *= 2) {
        for (u32 i = stride; i < ind; i += stride * 2) {
            if (slices_[rank].size > 0) {
                CHK_RET(LocalReduceCCLToCCL(vec[i], vec[i - stride], slices_[rank].size, mainThread_));
            }
        }
    }

    u64 allGatherOffset = rank / rankSizeX * localStrideSize;

    if (vec[0] != allGatherOffset) {
        void* srcSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + vec[0]);
        void* dstSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + allGatherOffset);
        if (slices_[rank].size > 0) {
            CHK_RET(
                static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread_, dstSlice, srcSlice, slices_[rank].size)));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::RunReduceScatter(
    const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels, u64 localStrideSize)
{
    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        // MainRecordSub + SubWaitMain
        CHK_RET(PreSyncInterThreads(mainThread_, subThreads_, notifyIdxMainToSub_));

        CHK_RET(HCCSProcessRsLoop(round, rank, rankSize, rankSizeX_, localStrideSize));
        CHK_RET(SIOProcessRsLoop(round, rank, rankSize, rankSizeX_, localStrideSize));
        CHK_RET(LocalCopyRsLoop(round, rank, rankSize, rankSizeX_, localStrideSize));

        // SubRecordMain + MainWaitSub
        CHK_RET(PostSyncInterThreads(mainThread_, subThreads_, notifyIdxSubToMain_));
    }

    CHK_RET(PreSyncInterThreads(mainThread_, subThreads_, notifyIdxMainToSub_));

    CHK_RET(TreeReduceRs(rank, rankSize, rankSizeX_, localStrideSize));

    CHK_RET(PostSyncInterThreads(mainThread_, subThreads_, notifyIdxSubToMain_));

    return HCCL_SUCCESS;
}

HcclResult
AllReduceBIAR::HCCSProcessAgLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX, u64 localStrideSize)
{
    u64 allGatherOffset = rank / rankSizeX * localStrideSize;
    if (round < hccs_ranks.size()) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads_[0], hccs_links[round].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads_[0], hccs_links_reversed[round].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 remoteOffsetByte = (rank / rankSizeX) * localStrideSize;
        void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + allGatherOffset);
        void* dst = static_cast<void*>(static_cast<u8*>(hccs_links[round].remoteOutput.addr) + remoteOffsetByte);

        if (slices_[rank].size > 0) {
            CHK_RET(static_cast<HcclResult>(
                HcommWriteOnThread(subThreads_[0], hccs_links[round].handle, dst, src, slices_[rank].size)));
        }

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads_[0], hccs_links[round].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads_[0], hccs_links_reversed[round].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult
AllReduceBIAR::SIOProcessAgLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX, u64 localStrideSize)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread_, sio_link.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread_, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

    u32 agRank = (round != 0 && round <= hccs_ranks.size()) ? (hccs_ranks_reversed[round - 1] ^ 1) : sio_rank_;
    u64 localOffsetByte = slices_[agRank].offset;
    u64 remoteOffsetByte = (round != 0 && round <= hccs_ranks.size()) ?
                               (hccs_ranks_reversed[round - 1] / rankSizeX) * localStrideSize :
                               (sio_rank_ / rankSizeX) * localStrideSize;
    void* src = static_cast<void*>(static_cast<u8*>(sio_link.remoteOutput.addr) + remoteOffsetByte);
    void* dst = static_cast<void*>(static_cast<u8*>(outputMem_.addr) + localOffsetByte);

    if (slices_[agRank].size > 0) {
        CHK_RET(
            static_cast<HcclResult>(HcommReadOnThread(mainThread_, sio_link.handle, dst, src, slices_[agRank].size)));
    }

    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(mainThread_, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread_, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult
AllReduceBIAR::LocalCopyAgLoop(u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX, u64 localStrideSize)
{
    if (round != 0 && round <= hccs_ranks.size()) {
        void* srcSlice = static_cast<void*>(
            static_cast<u8*>(scratchMem_.addr) + (hccs_ranks_reversed[round - 1] / rankSizeX) * localStrideSize);
        void* dstSlice
            = static_cast<void*>(static_cast<u8*>(outputMem_.addr) + slices_[hccs_ranks_reversed[round - 1]].offset);
        if (slices_[hccs_ranks_reversed[round - 1]].size > 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                subThreads_[1], dstSlice, srcSlice, slices_[hccs_ranks_reversed[round - 1]].size)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
AllReduceBIAR::RunAllGather(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels, u64 localStrideSize)
{
    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        // MainRecordSub + SubWaitMain
        CHK_RET(PreSyncInterThreads(mainThread_, subThreads_, notifyIdxMainToSub_));
        CHK_RET(HCCSProcessAgLoop(round, rank, rankSize, rankSizeX_, localStrideSize));
        CHK_RET(SIOProcessAgLoop(round, rank, rankSize, rankSizeX_, localStrideSize));
        CHK_RET(LocalCopyAgLoop(round, rank, rankSize, rankSizeX_, localStrideSize));

        // SubRecordMain + MainWaitSub
        CHK_RET(PostSyncInterThreads(mainThread_, subThreads_, notifyIdxSubToMain_));
    }
    CHK_RET(PreSyncInterThreads(mainThread_, subThreads_, notifyIdxMainToSub_));
    u64 allGatherOffset = rank / rankSizeX_ * localStrideSize;
    void* srcSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + allGatherOffset);
    void* dstSlice = static_cast<void*>(static_cast<u8*>(outputMem_.addr) + slices_[rank].offset);
    if (slices_[rank].size > 0) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread_, dstSlice, srcSlice, slices_[rank].size)));
    }

    CHK_RET(PostSyncInterThreads(mainThread_, subThreads_, notifyIdxSubToMain_));
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::CheckScratchCapacity(u32 rankSize, u64 localStrideSize)
{
    u64 scratchDemand = static_cast<u64>(rankSize) * localStrideSize;
    CHK_PRT_RET(
        scratchDemand > scratchMem_.size,
        HCCL_ERROR(
            "[AllReduceBIAR][CheckScratchCapacity]scratch demand[%llu] exceeds scratchMem size[%llu]", scratchDemand,
            scratchMem_.size),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult AllReduceBIAR::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    HCCL_INFO(
        "AllReduceBIAR run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.addr, outputMem_.addr, count_);

    CHK_RET(Preprocess(rank, rankSize, channels));

    rankSizeX_ = 2;
    if (rankSize % rankSizeX_ != 0) {
        HCCL_ERROR(
            "[AllReduceBIAR][RunAsync]rankSize[%u] is not evenly divisible by rankSizeX_[%u]", rankSize, rankSizeX_);
        return HCCL_E_INTERNAL;
    }
    rankSizeY_ = rankSize / rankSizeX_;

    sio_rank_ = rank ^ 1;
    sio_link = channels[sio_rank_];

    for (u32 i = 1; i < rankSize / rankSizeX_; ++i) {
        u32 current_hccs_rank = (rank + rankSizeX_ * i) % (rankSizeX_ * rankSizeY_);
        hccs_ranks.push_back(current_hccs_rank);
        hccs_neighbour_rank.push_back(current_hccs_rank ^ 1);
        hccs_links.push_back(channels[hccs_ranks[i - 1]]);
    }
    hccs_ranks_reversed.assign(hccs_ranks.rbegin(), hccs_ranks.rend());
    hccs_links_reversed.assign(hccs_links.rbegin(), hccs_links.rend());

    // slices_ prepared by Prepare() with V-style distribution (ReduceScatterV)
    u64 sliceSize = 0;
    for (u32 i = 0; i < rankSize; i++) {
        if (slices_[i].size > sliceSize) {
            sliceSize = slices_[i].size;
        }
        HCCL_DEBUG(" slice[%u]: offset: [%llu] size[%llu]", i, slices_[i].offset, slices_[i].size);
    }
    u64 localStrideSize = RoundUpWithDivisor(sliceSize, HCCL_MIN_SLICE_ALIGN_910B);

    CHK_RET(CheckScratchCapacity(rankSize, localStrideSize));

    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    CHK_RET(PreSyncInterThreads(mainThread_, subThreads_, notifyIdxMainToSub_));

    if (!hccs_ranks.empty() && slices_[hccs_ranks[0]].size > 0) {
        void* srcSlice = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + slices_[hccs_ranks[0]].offset);
        void* dstSlice
            = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + hccs_ranks[0] / rankSizeX_ * localStrideSize);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread_, dstSlice, srcSlice, slices_[hccs_ranks[0]].size)));
    }

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    CHK_RET(PostSyncInterThreads(mainThread_, subThreads_, notifyIdxSubToMain_));

    // MainRecordSub + SubWaitMain
    CHK_RET(RunReduceScatter(rank, rankSize, channels, localStrideSize));
    CHK_RET(RunAllGather(rank, rankSize, channels, localStrideSize));

    HCCL_INFO("AllReduceBIAR finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TEMPLATE_ALL_REDUCE_BIAR, AllReduceBIAR);
} // namespace ops_hccl_experimental
