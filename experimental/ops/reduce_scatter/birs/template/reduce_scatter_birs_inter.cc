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
#include "reduce_scatter_birs_inter.h"

namespace ops_hccl_experimental {

ReduceScatterBIRSInter::ReduceScatterBIRSInter() : ReduceScatterBIRS() {}

ReduceScatterBIRSInter::~ReduceScatterBIRSInter() {}

HcclResult ReduceScatterBIRSInter::Prepare(u32 serverNum, u32 interRankSize)
{
    serverNum_ = serverNum;
    intraRankSize_ = interRankSize / serverNum_;
    return HCCL_SUCCESS;
}

HcclResult
ReduceScatterBIRSInter::LocalCopyPreproc(ThreadHandle& thread, const u32 rank, u64 sliceSize, u64 localStrideSize)
{
    void* src;
    void* dst;
    u32 cnt = 0;
    for (u32 round = 0; round < intraRankSize_ / 2; round++) {
        for (u32 i = 0; i < serverNum_; i++) {
            src = static_cast<void*>(
                static_cast<u8*>(inputMem_.addr) + (2 * round + i * intraRankSize_ + rank % 2) * sliceSize);
            dst = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + cnt * localStrideSize);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, src, sliceSize)));
            cnt++;
        }
    }

    for (u32 cnt = 0; cnt < serverNum_; cnt++) {
        void* srcSlice = static_cast<void*>(
            static_cast<u8*>(inputMem_.addr)
            + (hccs_neighbour_rank[0] % intraRankSize_ + cnt * intraRankSize_) * sliceSize);
        void* dstSlice = static_cast<void*>(
            static_cast<u8*>(scratchMem_.addr) + (intraRankSize_)*localStrideSize * serverNum_ + cnt * localStrideSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::Preprocess(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    HCCL_INFO(
        "ReduceScatterBIRSInter run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", rank, rankSize,
        inputMem_.addr, outputMem_.addr, count_);
    return ReduceScatterBIRS::Preprocess(rank, rankSize, channels);
}

HcclResult ReduceScatterBIRSInter::HCCSIntraStep(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize)
{
    if (round != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        u64 localOffsetByte = hccs_ranks[round - 1] % intraRankSize_ / rankSizeX_ * localStrideSize * serverNum_;
        u64 remoteOffsetByte
            = ((intraRankSize_ / rankSizeX_) + rank % intraRankSize_ / rankSizeX_) * localStrideSize * serverNum_;
        void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void*>(static_cast<u8*>(hccs_links[round - 1].remoteOutput.addr) + remoteOffsetByte);

        HcommWriteOnThread(subThreads[0], hccs_links[round - 1].handle, dst, src, localStrideSize * serverNum_);

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(subThreads[0], hccs_links[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            subThreads[0], hccs_links_reversed[round - 1].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::SIOIntraStep(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

    u64 localOffsetByte = intraRankSize_ + (round % 2);
    u64 remoteOffsetByte = (round != hccs_ranks.size()) ? (hccs_ranks[round] % intraRankSize_) / rankSizeX_ :
                                                          rank % intraRankSize_ / rankSizeX_;
    localOffsetByte *= localStrideSize * serverNum_;
    remoteOffsetByte *= localStrideSize * serverNum_;
    void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte);
    void* dst = static_cast<void*>(static_cast<u8*>(sio_link.remoteOutput.addr) + remoteOffsetByte);

    HcommWriteReduceOnThread(
        mainThread, sio_link.handle, dst, src, localStrideSize * serverNum_ / unitSize,
        static_cast<HcommDataType>(dataType_), static_cast<HcommReduceOp>(reductionOp_));

    CHK_RET(
        static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, sio_link.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::LocalCopyIntraStep(
    u32 round, const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize)
{
    if (round < hccs_ranks.size()) {
        for (u32 cnt = 0; cnt < serverNum_; cnt++) {
            u64 srcOffsetByte
                = (round < hccs_ranks.size() - 1) ?
                      ((hccs_neighbour_rank[round + 1]) % intraRankSize_ + cnt * intraRankSize_) * sliceSize :
                      ((sio_rank) % intraRankSize_ + cnt * intraRankSize_) * sliceSize;
            u64 dstOffsetByte
                = (round < hccs_ranks.size() - 1) ?
                      (intraRankSize_ + ((round + 1) % 2)) * localStrideSize * serverNum_ + cnt * localStrideSize :
                      (intraRankSize_ + ((round + 1) % 2)) * localStrideSize * serverNum_ + cnt * localStrideSize;

            void* srcSlice = static_cast<void*>(static_cast<u8*>(inputMem_.addr) + srcOffsetByte);
            void* dstSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + dstOffsetByte);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(subThreads[1], dstSlice, srcSlice, sliceSize)));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::PreprocInterServer(
    const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize,
    std::vector<ChannelInfo>& channels)
{
    for (u32 i = 0; i < (intraRankSize_ / rankSizeX_); i++) {
        if (i == (rank % intraRankSize_ / rankSizeX_)) {
            vec_offsets.push_back((rank % intraRankSize_ / rankSizeX_) * localStrideSize * serverNum_);
        } else {
            vec_offsets.push_back(((intraRankSize_ / rankSizeX_) + i) * localStrideSize * serverNum_);
        }
    }

    CHK_RET(TreeLocalReduce(vec_offsets, localStrideSize * serverNum_, mainThread));

    u64 remoteOffsetByte = localStrideSize * serverNum_;
    u64 localOffsetByte = vec_offsets[0] + localStrideSize * (((rank) % rankSize) / intraRankSize_);
    void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte);
    void* dst = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + remoteOffsetByte);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dst, src, sliceSize)));
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterBIRSInter::InterServer(
    const u32 rank, const u32 rankSize, u32 rankSizeX_, u64 sliceSize, u64 localStrideSize,
    std::vector<ChannelInfo>& channels)
{
    for (u32 round = 1; round < serverNum_; round++) {
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            mainThread, channels[((serverNum_ - round) * intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            mainThread, channels[((round)*intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        void* remMemPtr = nullptr;
        u64 remoteOffsetByte = localStrideSize * serverNum_ + localStrideSize * round;
        u64 localOffsetByte
            = vec_offsets[0] + localStrideSize * ((((round)*intraRankSize_ + rank) % rankSize) / intraRankSize_);
        void* src = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + localOffsetByte);
        void* dst = static_cast<void*>(
            static_cast<u8*>(channels[((round)*intraRankSize_ + rank) % rankSize].remoteOutput.addr)
            + remoteOffsetByte);
        HcommWriteOnThread(
            mainThread, channels[((round)*intraRankSize_ + rank) % rankSize].handle, dst, src, sliceSize);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            mainThread, channels[((round)*intraRankSize_ + rank) % rankSize].handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            mainThread, channels[((serverNum_ - round) * intraRankSize_ + rank) % rankSize].handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    }
    u64 ptrSz = (serverNum_ == 1 ? vec_offsets[0] : localStrideSize * serverNum_);
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));
    for (u32 step = 1; step < serverNum_; step *= 2) {
        for (u32 i = 0; i + step < serverNum_; i += 2 * step) {
            LocalReduceCCLToCCL(ptrSz + sliceSize * (i + step), ptrSz + sliceSize * i, sliceSize, mainThread);
        }
    }
    void* srcSlice = static_cast<void*>(static_cast<u8*>(scratchMem_.addr) + ptrSz);
    void* dstSlice = static_cast<void*>(static_cast<u8*>(outputMem_.addr));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, dstSlice, srcSlice, sliceSize)));
    return HCCL_SUCCESS;
}

// scatter的入口函数
HcclResult ReduceScatterBIRSInter::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo>& channels)
{
    Preprocess(rank, rankSize, channels);
    u32 rankSizeX_ = 2;
    if (intraRankSize_ % rankSizeX_ != 0) {
        HCCL_ERROR(
            "[ReduceScatterBIRS][RunAsync]intraRankSize_[%u] is not evenly divisible by rankSizeX_[%u]", intraRankSize_,
            rankSizeX_);
        return HCCL_E_INTERNAL;
    }
    u32 rankSizeY_ = intraRankSize_ / rankSizeX_;

    sio_rank = rank ^ 1;
    sio_link = channels[sio_rank];

    for (u32 i = 1; i < rankSizeY_; ++i) {
        u32 current_hccs_rank
            = (rank % intraRankSize_ + rankSizeX_ * i) % (intraRankSize_) + rank / intraRankSize_ * intraRankSize_;
        hccs_ranks.push_back(current_hccs_rank);
        hccs_neighbour_rank.push_back(current_hccs_rank ^ 1);
        hccs_links.push_back(channels[hccs_ranks[i - 1]]);
    }
    hccs_links_reversed.assign(hccs_links.rbegin(), hccs_links.rend());

    u64 sliceSize = count_ * unitSize;
    u64 localStrideSize = sliceSize;

    // MainRecordSub + SubWaitMain
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

    LocalCopyPreproc(mainThread, rank, sliceSize, localStrideSize);

    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    for (u32 round = 0; round < hccs_ranks.size() + 1; round++) {
        // MainRecordSub + SubWaitMain
        CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

        HCCSIntraStep(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
        SIOIntraStep(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);
        LocalCopyIntraStep(round, rank, rankSize, rankSizeX_, sliceSize, localStrideSize);

        // SubRecordMain + MainWaitSub
        CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    }

    // MainRecordSub + SubWaitMain
    CHK_RET(PreSyncInterThreads(mainThread, subThreads, notifyIdxMainToSub_));

    PreprocInterServer(rank, rankSize, rankSizeX_, sliceSize, localStrideSize, channels);

    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));

    InterServer(rank, rankSize, rankSizeX_, sliceSize, localStrideSize, channels);

    CHK_RET(PostSyncInterThreads(mainThread, subThreads, notifyIdxSubToMain_));
    HCCL_INFO("ReduceScatterBIRS finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TEMPLATE_REDUCE_SCATTER_BIRS_INTER, ReduceScatterBIRSInter);
} // namespace ops_hccl_experimental
