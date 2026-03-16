/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base_v2.h"

using namespace AscendC;

template<typename T>
class AivAllGatherMesh1D : public AivCommBase {
public:
    __aicore__ inline AivAllGatherMesh1D() {}

    __aicore__ inline void InitCoreInfo(uint64_t len, int32_t tag)
    {
        curTag = tag;
        uint32_t coreId = GetBlockIdx();
        uint32_t coreCount = numBlocks_;

        uint64_t dataPerCore = len / coreCount;
        uint64_t remainder = len % coreCount;

        uint64_t innerOffset = 0;
        if (coreId < remainder) {
            innerOffset = coreId * (dataPerCore + 1) * sizeof(T);
            curCount = dataPerCore + 1;
        } else {
            innerOffset = remainder * (dataPerCore + 1) * sizeof(T) +
                        (coreId - remainder) * dataPerCore * sizeof(T);
            curCount = dataPerCore;
        }
        coreOffset = innerOffset;
    }

    __aicore__ inline void Run(uint64_t len, uint64_t stride)
    {
        if (curCount == 0) {
            return;
        }
        auto gmIn = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank_]) + coreOffset);
        auto input = reinterpret_cast<__gm__ T *>(input_ + coreOffset);

        CpGM2GM(gmIn, input, curCount);
        PipeBarrier<PIPE_ALL>();

        uint64_t flagOffset = numBlocks_ * rankSize_ * FLAG_SIZE;
        Record(rank_, GetBlockIdx() * FLAG_SIZE + flagOffset, curTag);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t rank = 0; rank < rankSize_; ++rank) {
            auto gmOthers = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank]) + coreOffset);
            auto output = reinterpret_cast<__gm__ T *>(output_ + rank * stride + coreOffset);
            PipeBarrier<PIPE_ALL>();

            WaitFlag(rank, GetBlockIdx() * FLAG_SIZE + flagOffset, curTag);
            CpGM2GM(output, gmOthers, curCount);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process(uint64_t count, uint64_t tag, uint64_t stride)
    {
        if (numBlocks_ >= rankSize_) {
            InitCoreInfo(count, tag);
            Run(count, stride);
        } else {
            RunCtrlCore(count, tag, stride);
        }
    }

    __aicore__ inline void RunCtrlCore(uint64_t count, uint64_t tag, uint64_t stride)
    {
        if (block_idx >= numBlocks_) {
            return;
        }
        auto input = reinterpret_cast<__gm__ T *>(input_);
        uint64_t dataTypeSize = sizeof(T);
        uint64_t countPerCore = count / numBlocks_;
        uint64_t curCountCore = block_idx == numBlocks_ - 1 ? count - countPerCore * (numBlocks_ - 1) : countPerCore;
        auto gmIn = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank_]) + block_idx * countPerCore * dataTypeSize);
        CpGM2GM(gmIn, input + block_idx * countPerCore * dataTypeSize, curCountCore);
        PipeBarrier<PIPE_ALL>();
        Record(rank_, block_idx, tag);
        for (uint32_t idx = 0; idx < numBlocks_; idx++) {
            WaitFlag(rank_, idx, tag);
            Record(rank_, idx, 0);
        }
        if (block_idx == 0) {
            Record(rank_, rank_, tag);
        }
        uint32_t perCoreRankNum = rankSize_ / numBlocks_;
        uint32_t curCoreRankNum = block_idx == numBlocks_ - 1 ? rankSize_ - perCoreRankNum * (numBlocks_ - 1) : perCoreRankNum;
        uint32_t startRank = block_idx * perCoreRankNum;
        for (uint32_t rank = startRank; rank < startRank + curCoreRankNum; rank++) {
            auto gmOthers = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank]));
            auto output = reinterpret_cast<__gm__ T *>(output_ + rank * stride);
            WaitFlag(rank, rank, tag);
            CpGM2GM(output, gmOthers, count);
            PipeBarrier<PIPE_ALL>();
        }
    }
    uint64_t coreOffset;
    int32_t curTag;
    uint64_t curCount;
};

template<typename T>
__aicore__ inline void AivAllGatherV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivAllGatherMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();

    op.Process(len, tag, outputSliceStride);
    op.BarrierAll();
}
