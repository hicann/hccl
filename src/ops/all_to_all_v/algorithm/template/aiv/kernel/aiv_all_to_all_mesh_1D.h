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

template <typename T>
class AivAlltoAllMesh1D : public AivCommBase {
public:
    __aicore__ inline AivAlltoAllMesh1D() {}

    __aicore__ inline void InitCommon(uint32_t sliceId)
    {
        dataSize_ = len_ * sizeof(T);
        coreIdx_ = blockIdx_;
        coreNum_ = numBlocks_;
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
    }

    __aicore__ inline void Process()
    {
        uint64_t smallDataSize = 512 * 1024;
        if (numBlocks_ < rankSize_) {
            // 少核场景
            ProcessMultiRank();
        } else if (dataSize_ <= smallDataSize) {
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

        // PreCopy阶段
        srcOffset_ = input_ + dstRank * inputSliceStride_ + sliceOffsetSize;
        dstOffset_ = reinterpret_cast<uint64_t>(myGmIn_) + dstRank * dataSize_ + sliceOffsetSize;
        CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, sliceCount);
        pipe_barrier(PIPE_ALL);

        uint64_t setFlagIdx = coreIdx_;
        Record(rank_, setFlagIdx, curTag_);

        // ReadRemote阶段
        uint64_t waitFlagIdx = rank_ * coreNumPerDstRank + coreIdxForDstRank;
        WaitFlag(dstRank, waitFlagIdx, curTag_);

        srcOffset_ = reinterpret_cast<uint64_t>(GetGmIn(dstRank)) + rank_ * dataSize_ + sliceOffsetSize;
        dstOffset_ = output_ + dstRank * outputSliceStride_ + sliceOffsetSize;
        CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, sliceCount);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessMultiRank()
    {
        // 下发核数由上层框架保证符合控核公式，算子内不校验
        uint32_t rankNumPerCore = (rankSize_ + coreNum_ - 1) / coreNum_; // 向上取整

        for (uint32_t idx = 0; idx < rankNumPerCore; ++idx) {
            uint32_t dstRank = coreIdx_ * rankNumPerCore + idx;
            if (dstRank >= rankSize_) {
                break;
            }

            // PreCopy阶段
            srcOffset_ = input_ + dstRank * inputSliceStride_;
            dstOffset_ = reinterpret_cast<uint64_t>(myGmIn_) + dstRank * dataSize_;
            CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, len_);
            pipe_barrier(PIPE_ALL);

            uint64_t setFlagIdx = dstRank;
            Record(rank_, setFlagIdx, curTag_); // 按照数据源rank编排flag的偏移量
        }

        for (uint32_t idx = 0; idx < rankNumPerCore; ++idx) {
            uint32_t dstRank = coreIdx_ * rankNumPerCore + idx;
            if (dstRank >= rankSize_) {
                break;
            }

            // ReadRemote阶段
            uint64_t waitFlagIdx = rank_;
            WaitFlag(dstRank, waitFlagIdx, curTag_);

            srcOffset_ = reinterpret_cast<uint64_t>(GetGmIn(dstRank)) + rank_ * dataSize_;
            dstOffset_ = output_ + dstRank * outputSliceStride_;
            CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, len_);
            pipe_barrier(PIPE_ALL);
        }
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

    uint32_t coreNum_;
    uint32_t coreIdx_;

    uint64_t dataSize_; // 要给每个rank搬运的数据大小

    uint64_t srcOffset_;
    uint64_t dstOffset_;
};

template <typename T>
__aicore__ inline void AivAlltoAllV2Mesh1D(KERNEL_ARGS_DEF)
{
    bool pingpong = false;
    if (len * sizeof(T) <= DATA_LIMIT && rankSize * len * sizeof(T) <= PINGPONG_TOTAL_DATA_LIMIT) {
        pingpong = true;
    }

    AivAlltoAllMesh1D<T> op;
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

template <typename T>
__aicore__ inline void AivAlltoAllV2Mesh1DSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    AivAlltoAllMesh1D<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT, false);
    uint64_t maxCountPerLoop = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / op.rankSize_ / sizeof(T);
    uint64_t countLeft = op.len_;

    int32_t loopTag = op.tag_;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        op.len_ = curCount;
        op.InitCommon(loopTag);
        op.Process();
        op.BarrierAll();

        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}

__aicore__ inline void sk_a2a_mesh_1d(SUPERKERNEL_ARGS_DEF)
{
#ifdef HCCL_DTYPE_INT8
    AivAlltoAllV2Mesh1DSuperKernel<int8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_UINT8
    AivAlltoAllV2Mesh1DSuperKernel<uint8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT16
    AivAlltoAllV2Mesh1DSuperKernel<int16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_UINT16
    AivAlltoAllV2Mesh1DSuperKernel<uint16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT32
    AivAlltoAllV2Mesh1DSuperKernel<int32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_UINT32
    AivAlltoAllV2Mesh1DSuperKernel<uint32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP16
    AivAlltoAllV2Mesh1DSuperKernel<half>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP32
    AivAlltoAllV2Mesh1DSuperKernel<float>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_BFP16
    AivAlltoAllV2Mesh1DSuperKernel<bfloat16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT64
    AivAlltoAllV2Mesh1DSuperKernel<int64_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_UINT64
    AivAlltoAllV2Mesh1DSuperKernel<uint64_t>(SUPERKERNEL_ARGS_CALL);
#else
#endif
}
