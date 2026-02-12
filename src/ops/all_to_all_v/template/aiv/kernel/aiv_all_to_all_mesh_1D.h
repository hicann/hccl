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
// todo 简化参数
class AivAlltoAllMesh1D : public AivCommBase {
public:
    __aicore__ inline AivAlltoAllMesh1D() {}
 
    __aicore__ inline void InitCommon(uint32_t tag)
    {
        dataSize_ = len_ * sizeof(T);
        coreIdx_ = GetBlockIdx();
        coreNum_ = numBlocks_;
        opTag_ = tag;
    }
 
    __aicore__ inline void Process()
    {
        if (coreIdx_ >= coreNum_) {
            return;
        }
 
        if (coreNum_ >= rankSize_) {
            ProcessMultiCore();
        } else {
            ProcessMultiRank();
        }
    }
 
private:
    __aicore__ inline void ProcessMultiCore()
    {
        // 下发核数由上层框架保证符合控核公式，算子内不校验
        uint32_t coreNumPerDstRank = coreNum_ / rankSize_;
        uint32_t dstRank = coreIdx_ / coreNumPerDstRank;
        uint32_t coreIdxForDstRank = coreIdx_ % coreNumPerDstRank;
 
        // 按核切分数据
        uint64_t sliceOffset;
        uint64_t sliceCount;
        SplitData(len_, coreNumPerDstRank, coreIdxForDstRank, sliceOffset, sliceCount);
        uint64_t sliceOffsetSize = sliceOffset * sizeof(T);
        uint64_t sliceSize = sliceCount * sizeof(T);
 
        // PutRemote阶段
        srcOffset_ = input_ + dstRank * inputSliceStride_ + sliceOffsetSize;
        dstOffset_ = reinterpret_cast<uint64_t>(GM_IN[dstRank]) + rank_ * dataSize_ + sliceOffsetSize;
        CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, sliceCount);
        pipe_barrier(PIPE_ALL);
 
        uint64_t setFlagIdx = rank_ * coreNumPerDstRank + coreIdxForDstRank;
        Record(dstRank, setFlagIdx, opTag_);  // 按照数据源rank编排flag的偏移量
 
        // PostCopy阶段
        uint64_t waitFlagIdx = coreIdx_;
        WaitFlag(rank_, waitFlagIdx, opTag_);
 
        srcOffset_ = reinterpret_cast<uint64_t>(GM_IN[rank_]) + dstRank * dataSize_ + sliceOffsetSize;
        dstOffset_ = output_ + dstRank * outputSliceStride_ + sliceOffsetSize;
        CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, sliceCount);
        pipe_barrier(PIPE_ALL);
    }
 
    __aicore__ inline void ProcessMultiRank()
    {
        // 下发核数由上层框架保证符合控核公式，算子内不校验
        uint32_t rankNumPerCore = (rankSize_ + coreNum_ - 1) / coreNum_;  // 向上取整
 
        for (uint32_t idx = 0; idx < rankNumPerCore; ++idx) {
            uint32_t dstRank = coreIdx_ * rankNumPerCore + idx;
            if (dstRank >= rankSize_) {
                return;
            }
 
            // PutRemote阶段
            srcOffset_ = input_ + dstRank * inputSliceStride_;
            dstOffset_ = reinterpret_cast<uint64_t>(GM_IN[dstRank]) + rank_ * dataSize_;
            CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, len_);
            pipe_barrier(PIPE_ALL);
 
            uint64_t setFlagIdx = rank_;
            Record(dstRank, setFlagIdx, opTag_);  // 按照数据源rank编排flag的偏移量
        }
 
        for (uint32_t idx = 0; idx < rankNumPerCore; ++idx) {
            uint32_t dstRank = coreIdx_ * rankNumPerCore + idx;
            if (dstRank >= rankSize_) {
                return;
            }
 
            // PostCopy阶段
            uint64_t waitFlagIdx = dstRank;
            WaitFlag(rank_, waitFlagIdx, opTag_);
 
            srcOffset_ = reinterpret_cast<uint64_t>(GM_IN[rank_]) + dstRank * dataSize_;
            dstOffset_ = output_ + dstRank * outputSliceStride_;
            CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, len_);
            pipe_barrier(PIPE_ALL);
        }
    }
 
    __aicore__ inline void SplitData(uint64_t dataCount, uint64_t splitNum, uint64_t idx,
        uint64_t& sliceOffset, uint64_t& sliceCount)
    {
        // 防止idx越界
        if (idx >= splitNum) {
            sliceOffset = 0;
            sliceCount = 0;
            return;
        }
        uint64_t baseSliceCount = dataCount / splitNum;
        uint64_t remainSize = dataCount % splitNum;  // remainSize必然小于splitNum

        // 将remainSize均分给前remainSize个核，每个核多1
        if (idx < remainSize) {
            sliceCount = baseSliceCount + 1;
            sliceOffset = idx * (baseSliceCount + 1);
        } else {
            sliceCount = baseSliceCount;
            sliceOffset = remainSize * (baseSliceCount + 1) + (idx - remainSize) * baseSliceCount;
        }
    }
 
    uint32_t opTag_;
 
    uint32_t coreNum_;
    uint32_t coreIdx_;
 
    uint64_t dataSize_;  // 要给每个rank搬运的数据大小
 
    uint64_t srcOffset_;
    uint64_t dstOffset_;
};
 
template<typename T>
__aicore__ inline void AivAlltoAllV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivAlltoAllMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.InitCommon(tag);
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    op.Process();
    op.BarrierAll();
}