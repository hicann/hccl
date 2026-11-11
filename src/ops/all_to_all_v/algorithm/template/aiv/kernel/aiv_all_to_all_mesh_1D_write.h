/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALL_TO_ALL_MESH_1D_WRITE_H
#define AIV_ALL_TO_ALL_MESH_1D_WRITE_H

#include "aiv_all_to_all_mesh_1D.h"

using namespace AscendC;

template <typename T>
class AivAlltoAllMesh1DWrite : public AivCommBase {
public:
    __aicore__ inline AivAlltoAllMesh1DWrite() {}

    __aicore__ inline void InitCommon(uint32_t sliceId)
    {
        dataSize_ = len_ * sizeof(T);
        coreIdx_ = blockIdx_;
        coreNum_ = numBlocks_;
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    }

    __aicore__ inline void Process()
    {
        if (dataSize_ <= DATA_LIMIT) {
            // 多核小数据量场景
            ProcessSmallData();
        } else {
            // 多核大数据量场景
            ProcessMultiCore();
        }
    }

private:
    __aicore__ inline void ProcessMultiCore()
    {
        // 实测整卡8个搬运核达到较优带宽，按rank均分预算，并保证每peer至少1核。
        constexpr uint32_t DATA_COPY_CORE_BUDGET = 8;
        uint32_t coreNumPerDstRank = DATA_COPY_CORE_BUDGET / rankSize_;
        coreNumPerDstRank = coreNumPerDstRank == 0 ? 1 : coreNumPerDstRank;
        uint32_t availableCoreNumPerDstRank = coreNum_ / rankSize_;
        coreNumPerDstRank
            = coreNumPerDstRank < availableCoreNumPerDstRank ? coreNumPerDstRank : availableCoreNumPerDstRank;
        // 仅跳过搬运，numBlocks_保持原值，所有已启动核仍参加外层首尾同步。
        if (coreIdx_ >= coreNumPerDstRank * rankSize_) {
            return;
        }
        uint32_t dstRank = coreIdx_ / coreNumPerDstRank;
        uint32_t coreIdxForDstRank = coreIdx_ % coreNumPerDstRank;

        // 按核切分数据
        uint64_t sliceOffset;
        uint64_t sliceCount;
        SplitData(len_, coreNumPerDstRank, coreIdxForDstRank, sliceOffset, sliceCount);
        uint64_t sliceOffsetSize = sliceOffset * sizeof(T);

        // PutRemote阶段（写语义）：input直写对端dstRank的cclBuff rank_槽位，无中转
        srcOffset_ = input_ + dstRank * inputSliceStride_ + sliceOffsetSize;
        dstOffset_ = reinterpret_cast<uint64_t>(GetGmIn(dstRank)) + rank_ * dataSize_ + sliceOffsetSize;
        CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, sliceCount);
        pipe_barrier(PIPE_ALL);

        // 通知对端dstRank：本rank的数据已直写其rank_槽位（读语义此flag记录在本端）
        uint64_t setFlagIdx = rank_ * coreNumPerDstRank + coreIdxForDstRank;
        Record(dstRank, setFlagIdx, curTag_);
        pipe_barrier(PIPE_ALL);

        // LocalRead阶段（写语义）：等dstRank直写本端的数据就绪，纯本地读写入output
        uint64_t waitFlagIdx = dstRank * coreNumPerDstRank + coreIdxForDstRank;
        WaitFlag(rank_, waitFlagIdx, curTag_);

        srcOffset_ = reinterpret_cast<uint64_t>(myGmIn_) + dstRank * dataSize_ + sliceOffsetSize;
        dstOffset_ = output_ + dstRank * outputSliceStride_ + sliceOffsetSize;
        CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, sliceCount);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessSmallData()
    {
        if (coreIdx_ >= rankSize_) {
            return;
        }

        uint32_t dstRank = coreIdx_;

        // PutRemote
        srcOffset_ = input_ + dstRank * inputSliceStride_;
        dstOffset_ = reinterpret_cast<uint64_t>(GetGmIn(dstRank)) + rank_ * dataSize_;
        CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, len_);
        pipe_barrier(PIPE_ALL);

        Record(dstRank, rank_, curTag_); // 按照数据源rank编排flag的偏移量

        // PostCopy
        WaitFlag(rank_, dstRank, curTag_);

        srcOffset_ = reinterpret_cast<uint64_t>(myGmIn_) + dstRank * dataSize_;
        dstOffset_ = output_ + dstRank * outputSliceStride_;
        CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, len_);
        // 小数据量可能走pingpong路径，无BarrierAll兜底，需在返回前确保搬运完成
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void
    SplitData(uint64_t dataCount, uint64_t splitNum, uint64_t idx, uint64_t& sliceOffset, uint64_t& sliceCount)
    {
        // 防止idx越界
        if (idx >= splitNum) {
            sliceOffset = 0;
            sliceCount = 0;
            return;
        }
        uint64_t baseSliceCount = dataCount / splitNum;
        uint64_t remainSize = dataCount % splitNum; // remainSize必然小于splitNum

        // 将remainSize均分给前remainSize个核，每个核多1
        if (idx < remainSize) {
            sliceCount = baseSliceCount + 1;
            sliceOffset = idx * (baseSliceCount + 1);
        } else {
            sliceCount = baseSliceCount;
            sliceOffset = remainSize * (baseSliceCount + 1) + (idx - remainSize) * baseSliceCount;
        }
    }

    uint32_t coreIdx_;
    uint32_t coreNum_;

    uint64_t dataSize_; // 要给每个rank搬运的数据大小

    uint64_t srcOffset_;
    uint64_t dstOffset_;
};

template <typename T>
__aicore__ inline void AivAlltoAllV2Mesh1DWrite(KERNEL_ARGS_DEF)
{
    // 少核统一使用读语义的预拷贝、远端读和同步流程。
    if (numBlocks < rankSize) {
        return AivAlltoAllV2Mesh1D<T>(KERNEL_ARGS_CALL);
    }

    bool pingpong = false;
    if (len * sizeof(T) <= DATA_LIMIT && rankSize * len * sizeof(T) <= PINGPONG_TOTAL_DATA_LIMIT) {
        pingpong = true;
    }

    AivAlltoAllMesh1DWrite<T> op;
    op.Init(KERNEL_CLASS_INIT, true, pingpong);
    op.InitCommon(sliceId);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    op.Process();
    if (!pingpong) {
        op.BarrierAll();
    }
}

#endif // AIV_ALL_TO_ALL_MESH_1D_WRITE_H
