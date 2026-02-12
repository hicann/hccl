/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_REDUCE_SCATTER_MESH_1D_CORE_CTRL_H
#define AIV_REDUCE_SCATTER_MESH_1D_CORE_CTRL_H

#include "aiv_communication_base_v2.h"
#include "kernel_common.h"

using namespace AscendC;

template<typename T>
class AivReduceScatterMesh1DCoreCtrl : public AivCommBase {
    constexpr static uint32_t STAGE_NUM = 2;
    constexpr static uint32_t MAX_RANK_SIZE = 8;
public:
    __aicore__ inline AivReduceScatterMesh1DCoreCtrl() {}

    // 0 < block_num < 2 * rankSize
    __aicore__ inline void InitCoreInfo(uint64_t len, uint64_t inputStride)
    {
        lenPerRank_ = len;
        inputStride_ = inputStride;
        rankSizeU32_ = static_cast<uint32_t>(rankSize_);
        blockIdx_  = static_cast<uint32_t>(GetBlockIdx());
        blockNum_ = static_cast<uint32_t>(GetBlockNum());
        if (rankSizeU32_ == 0 || blockNum_ == 0) {
            valid_ = false;
            return;
        }
        fullLogical_ = rankSizeU32_ * STAGE_NUM;
        if (blockNum_ >= fullLogical_) {
            valid_ = false;
            return;
        }

        processNum_ = lenPerRank_ / rankSizeU32_;
        const uint32_t baseCnt = fullLogical_ / blockNum_; // 每个核当几个核用 (2*4) / 5 = 1
        const uint32_t extra = fullLogical_ % blockNum_; // 剩下的核数 3
        const uint32_t myCnt = baseCnt + (blockIdx_ < extra ? 1u : 0u); // 1 + (1 1 1 0 0) = 2 2 2 1 1
        // 0 1 2 3 4 + (0 1 2 3 3) = 0 2 4 6 7
        const uint32_t myStart = baseCnt * blockIdx_ + (blockIdx_ < extra ? blockIdx_ : extra);
        const uint32_t myEnd = myStart + myCnt; // 2 4 6 7 8

        // producer 区间
        uint32_t p0 = myStart; // (0 2) (2 4) (4 6) (6 7) (7 8)
        uint32_t p1 = myEnd;
        if (p0 > rankSizeU32_) {
            p0 = rankSizeU32_;
        }
        if (p1 > rankSizeU32_) {
            p1 = rankSizeU32_;
        }
        producerBegin_ = p0; // (0 2) (2 4)
        producerEnd_   = p1;

        // consumer 区间
        uint32_t c0 = (myStart < rankSizeU32_) ? rankSizeU32_ : myStart; // (4 6) (6 7) (7 8)
        uint32_t c1 = (myEnd   < rankSizeU32_) ? rankSizeU32_ : myEnd;
        if (c0 > fullLogical_) {
            c0 = fullLogical_;
        }
        if (c1 > fullLogical_) {
            c1 = fullLogical_;
        }
        consumerBegin_ = (c0 > rankSizeU32_) ? (c0 - rankSizeU32_) : 0; // (0 2) (2 3) (3 4)
        consumerEnd_   = (c1 > rankSizeU32_) ? (c1 - rankSizeU32_) : 0;

        valid_ = true;
    }

    __aicore__ inline void Process(uint32_t tag)
    {
        if (!valid_) {
            return;
        }
        curTag_ = static_cast<int32_t>(tag);

        for (uint32_t p = producerBegin_; p < producerEnd_; ++p) {
            ProducerOne(p);
        }

        for (uint32_t c = consumerBegin_; c < consumerEnd_; ++c) {
            ConsumerOne(c);
        }
    }

private:
    bool valid_ = false;
    uint64_t lenPerRank_ = 0;
    uint64_t inputStride_ = 0;
    uint64_t processNum_ = 0;
    uint32_t rankSizeU32_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t fullLogical_ = 0;
    uint32_t producerBegin_ = 0;
    uint32_t producerEnd_ = 0;
    uint32_t consumerBegin_ = 0;
    uint32_t consumerEnd_ = 0;
    int32_t  curTag_ = 0;
    uint64_t outputOffset_ = 0;
    uint64_t inputOffVec_[MAX_RANK_SIZE];
    uint64_t consumProcessNum_ = 0;

    __aicore__ inline void ProducerOne(uint32_t producerId)
    {
        // producerId is targetRankId
        const uint64_t outerOffsetBytes = static_cast<uint64_t>(producerId) * inputStride_;
        const uint64_t ipcRankOffset = static_cast<uint64_t>(producerId) * lenPerRank_ * sizeof(T);

        __gm__ T *src = reinterpret_cast<__gm__ T *>(input_ + outerOffsetBytes);
        __gm__ T *dst = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank_]) + ipcRankOffset);

        CpGM2GM(dst, src, len_);

        pipe_barrier(PIPE_ALL);

        const uint64_t flagOffset = rank_;
        Record(producerId, flagOffset, curTag_);
    }

    __aicore__ inline void ConsumerOne(uint32_t consumerId)
    {
        const uint64_t chanIdx = static_cast<uint64_t>(consumerId);
        const uint64_t outerOffsetBytes = chanIdx * processNum_ * sizeof(T);

        outputOffset_ = output_ + outerOffsetBytes;
        consumProcessNum_ = processNum_;
        if (consumerId == (rankSizeU32_ - 1)) {
            consumProcessNum_ = lenPerRank_ - processNum_ * (rankSizeU32_ - 1);
        }

        for (uint32_t idx = 0; idx < rankSizeU32_; ++idx) {
            const uint32_t rankIdx = (idx + consumerId) % rankSizeU32_;
            const uint64_t baseIpc =
                reinterpret_cast<uint64_t>(GM_IN[rankIdx]) +
                lenPerRank_ * static_cast<uint64_t>(rank_) * sizeof(T);
            inputOffVec_[idx] = baseIpc + outerOffsetBytes;
        }

        for (uint32_t idx = 0; idx < rankSizeU32_; ++idx) {
            const uint32_t rankIdx = (idx + consumerId) % rankSizeU32_;
            WaitFlag(rank_, static_cast<uint64_t>(rankIdx), curTag_);
            __gm__ T *src = reinterpret_cast<__gm__ T *>(inputOffVec_[idx]);
            __gm__ T *dst = reinterpret_cast<__gm__ T *>(outputOffset_);
            if (idx == 0) {
                CpGM2GM(dst, src, consumProcessNum_);
            } else {
                CpGM2GM(dst, src, consumProcessNum_, reduceOp_);
            }

            pipe_barrier(PIPE_ALL);
        }
    }
};

template<typename T>
__aicore__ inline void AivReduceScatterV2Mesh1DCoreCtrl(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivReduceScatterMesh1DCoreCtrl<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.InitCoreInfo(len, inputSliceStride);

    SyncAll<true>();
    if (block_idx == 0 &&
        (tag >> AIV_TAG_MOVE_RIGHT_BITS) == 1 &&
        (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();

    op.Process(tag);
    op.BarrierAll();
}

#endif // AIV_REDUCE_SCATTER_MESH_1D_CORE_CTRL_H