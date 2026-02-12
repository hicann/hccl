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
 
    __aicore__ inline void InitCoreInfo(uint64_t len, int32_t tag) //len is input length
    {
        curTag = tag;
        // 每个核处理 len/coreCount 个数据
        uint32_t coreId = GetBlockIdx();
        uint32_t coreCount = numBlocks_;
 
        uint64_t dataPerCore = len / coreCount;
        uint64_t remainder = len % coreCount;
 
        // 处理数据不对齐问题：确保每个核心处理的数据量是sizeof(T)的倍数
        uint64_t innerOffset = 0;
        if (coreId < remainder) {
            // 前remainder个核心多分配一个元素
            innerOffset = coreId * (dataPerCore + 1) * sizeof(T);
            curCount = dataPerCore + 1;
        } else {
            // 其余核心按正常分配
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
        // 步骤1: 分核执行LocalCopy，拷贝input到GM_IN
        // 每个核将自己负责的数据拷贝到 GM_IN 中对应位置
        auto input = reinterpret_cast<__gm__ T *>(input_ + coreOffset);
 
        CpGM2GM(gmIn, input, curCount);
        PipeBarrier<PIPE_ALL>();
 
        // 步骤2: 设置完成拷贝flag
        uint64_t flagOffset = numBlocks_ * rankSize_ * FLAG_SIZE;
        Record(rank_, GetBlockIdx() * FLAG_SIZE + flagOffset, curTag);
        PipeBarrier<PIPE_ALL>();
 
        // 步骤3: 循环对每一个rank执行CpGM2GM将数据写入output
        for (uint32_t rank = 0; rank < rankSize_; ++rank) {
            // 直接从 GM_IN 中读取数据到 output
            auto gmOthers = reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(GM_IN[rank]) + coreOffset);
            auto output = reinterpret_cast<__gm__ T *>(output_ + rank * stride + coreOffset);
            PipeBarrier<PIPE_ALL>();
 
            WaitFlag(rank, GetBlockIdx() * FLAG_SIZE + flagOffset, curTag);
            // 使用 CpGM2GM 直接拷贝数据
            CpGM2GM(output, gmOthers, curCount);
        }
        PipeBarrier<PIPE_ALL>();
    }
 
    __aicore__ inline void Process(uint64_t count, uint64_t tag, uint64_t stride)
    {
        if (numBlocks_ >= rankSize_) {
            // 核数大于等于ranksize
            InitCoreInfo(count, tag);
            Run(count, stride);
        } else {
            // 核数小于ranksize
            RunCtrlCore(count, tag, stride);
        }
    }
 
    __aicore__ inline void RunCtrlCore(uint64_t count, uint64_t tag, uint64_t stride)
    {
        // 核数小于ranksize
        if (block_idx >= numBlocks_) {
            return;
        }
        // 分核把数据从input搬到gm
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
        // 每个核分配多个rank搬运数据从gm到对端output
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
    // 执行barrier全同步
    op.BarrierAll();
}