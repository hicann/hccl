/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_REDUCE_SCATTER_LOCAL_TREE_CORECTRL_H
#define AIV_REDUCE_SCATTER_LOCAL_TREE_CORECTRL_H

#include "aiv_communication_base_v2.h"

using namespace AscendC;

template<typename T>
class AivReduceScatterLocalTreeCoreCtrl : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterLocalTreeCoreCtrl() {}

    __aicore__ inline void InitCoreInfo(uint64_t len, uint64_t inputStride)
    {
        lenPerRank_ = len;
        inputStride_ = inputStride;
        rankSizeU32_ = static_cast<uint32_t>(rankSize_);
        valid_ = rankSizeU32_ > 0 && numBlocks_ > 0 && numBlocks_ < rankSizeU32_;
        if (!valid_) {
            return;
        }

        SplitLogicalRange(rankSizeU32_, publishBegin_, publishEnd_);
        SplitLogicalRange(rankSizeU32_, fetchBegin_, fetchEnd_);
    }

    __aicore__ inline void Process(uint32_t sliceId)
    {
        if (!valid_) {
            return;
        }

        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);

        PublishLocalShardRange();
        FetchPeerShardRange();
        SyncAll<true>();

        LocalTreeReduceCoreCtrl();
        StoreResult();
        WaitAllFetchDone();
    }

private:
    bool valid_ = false;
    uint64_t lenPerRank_ = 0;
    uint64_t inputStride_ = 0;
    uint32_t rankSizeU32_ = 0;
    uint32_t publishBegin_ = 0;
    uint32_t publishEnd_ = 0;
    uint32_t fetchBegin_ = 0;
    uint32_t fetchEnd_ = 0;

    __aicore__ inline uint64_t LocalPublishOffset(uint32_t targetRank)
    {
        return static_cast<uint64_t>(targetRank) * lenPerRank_ * sizeof(T);
    }

    __aicore__ inline uint64_t LocalStageOffset(uint32_t peerRank)
    {
        return static_cast<uint64_t>(rankSizeU32_ + peerRank) * lenPerRank_ * sizeof(T);
    }

    __aicore__ inline uint64_t FetchDoneFlagOffset(uint32_t peerRank)
    {
        return static_cast<uint64_t>(rankSizeU32_ + peerRank);
    }

    __aicore__ inline uint64_t ReduceReadyFlagOffset(uint32_t offset)
    {
        return static_cast<uint64_t>(DOUBLE * rankSizeU32_ + offset);
    }

    __aicore__ inline void SplitLogicalRange(uint32_t total, uint32_t &begin, uint32_t &end)
    {
        const uint32_t baseCnt = total / numBlocks_;
        const uint32_t extra = total % numBlocks_;
        const uint32_t myCnt = baseCnt + (blockIdx_ < extra ? 1u : 0u);
        begin = baseCnt * blockIdx_ + (blockIdx_ < extra ? blockIdx_ : extra);
        end = begin + myCnt;
    }

    __aicore__ inline uint32_t CeilLog2(uint32_t n)
    {
        if (n <= 1) {
            return 0;
        }
        uint32_t result = 0;
        uint32_t value = 1;
        while (value < n) {
            value <<= 1;
            ++result;
        }
        return result;
    }

    __aicore__ inline uint32_t GetLargestPowerOf2(uint32_t n)
    {
        if (n <= 1) {
            return 0;
        }
        uint32_t result = 1;
        while ((result << 1) < n) {
            result <<= 1;
        }
        return result;
    }

    __aicore__ inline void PublishOne(uint32_t targetRank)
    {
        const uint64_t inputOffset = input_ + static_cast<uint64_t>(targetRank) * inputStride_;
        const uint64_t publishOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + LocalPublishOffset(targetRank);
        CpGM2GM(reinterpret_cast<__gm__ T *>(publishOffset), reinterpret_cast<__gm__ T *>(inputOffset), lenPerRank_);
        pipe_barrier(PIPE_ALL);
        Record(targetRank, rank_, curTag_);
    }

    __aicore__ inline void PublishLocalShardRange()
    {
        for (uint32_t targetRank = publishBegin_; targetRank < publishEnd_; ++targetRank) {
            PublishOne(targetRank);
        }
    }

    __aicore__ inline void FetchOne(uint32_t peerRank)
    {
        WaitFlag(rank_, peerRank, curTag_);
        const uint64_t peerOffset =
            reinterpret_cast<uint64_t>(GM_IN[peerRank]) + static_cast<uint64_t>(rank_) * lenPerRank_ * sizeof(T);
        const uint64_t stageOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + LocalStageOffset(peerRank);
        CpGM2GM(reinterpret_cast<__gm__ T *>(stageOffset), reinterpret_cast<__gm__ T *>(peerOffset), lenPerRank_);
        pipe_barrier(PIPE_ALL);
        Record(peerRank, FetchDoneFlagOffset(rank_), curTag_);
    }

    __aicore__ inline void FetchPeerShardRange()
    {
        for (uint32_t peerRank = fetchBegin_; peerRank < fetchEnd_; ++peerRank) {
            FetchOne(peerRank);
        }
    }

    __aicore__ inline void WaitAllFetchDone()
    {
        for (uint32_t peerRank = fetchBegin_; peerRank < fetchEnd_; ++peerRank) {
            WaitFlag(rank_, FetchDoneFlagOffset(peerRank), curTag_);
        }
    }

    __aicore__ inline void LocalTreeReduceCoreCtrl()
    {
        uint32_t curBlocks = rankSizeU32_;
        const uint32_t totalRounds = CeilLog2(rankSizeU32_);
        for (uint32_t round = 0; round < totalRounds; ++round) {
            const uint32_t powerOf2 = GetLargestPowerOf2(curBlocks);
            if (powerOf2 == 0) {
                break;
            }

            for (uint32_t offset = blockIdx_; offset < powerOf2; offset += numBlocks_) {
                const uint32_t backIdx = powerOf2 + offset;
                if (backIdx < curBlocks) {
                    if (round > 0) {
                        WaitFlag(rank_, ReduceReadyFlagOffset(offset), static_cast<int32_t>(curTag_ + round));
                        WaitFlag(rank_, ReduceReadyFlagOffset(backIdx), static_cast<int32_t>(curTag_ + round));
                    }
                    const uint64_t frontOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + LocalStageOffset(offset);
                    const uint64_t backOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + LocalStageOffset(backIdx);
                    CpGM2GM(reinterpret_cast<__gm__ T *>(frontOffset), reinterpret_cast<__gm__ T *>(backOffset),
                            lenPerRank_, reduceOp_);
                    pipe_barrier(PIPE_ALL);
                }
                Record(rank_, ReduceReadyFlagOffset(offset), static_cast<int32_t>(curTag_ + round + 1));
            }

            curBlocks = powerOf2;
        }
    }

    __aicore__ inline void StoreResult()
    {
        if (blockIdx_ != 0) {
            return;
        }

        const uint64_t resultOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + LocalStageOffset(0);
        CpGM2GM(reinterpret_cast<__gm__ T *>(output_), reinterpret_cast<__gm__ T *>(resultOffset), lenPerRank_);
        pipe_barrier(PIPE_ALL);
    }
};

template<typename T>
__aicore__ inline void AivReduceScatterV2LocalTreeCoreCtrl(KERNEL_ARGS_DEF)
{
    AivReduceScatterLocalTreeCoreCtrl<T> op;
    bool pingpong = false;
    if (len * sizeof(T) <= DATA_LIMIT) {
        pingpong = true;
    }
    op.Init(KERNEL_CLASS_INIT, true, pingpong);
    op.InitCoreInfo(len, inputSliceStride);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    op.Process(sliceId);
    if (!pingpong) {
        op.BarrierAll();
    }
}

#endif // AIV_REDUCE_SCATTER_LOCAL_TREE_CORECTRL_H
