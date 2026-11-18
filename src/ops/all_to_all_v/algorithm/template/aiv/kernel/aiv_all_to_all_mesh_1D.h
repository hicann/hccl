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

// 对称直写路径的有效核数上限：单核需搬足够大的连续块才能摊薄 DMA 固定开销，故按 16 收敛；
// rankSize 超过 16 时按 rankSize 取（每个 dstRank 至少一个核）
constexpr uint32_t A2A_SYM_MAX_CORES = 16U;

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
        // 对端 output 对称 VA 只含「窗口基址 + 算子级 outputOffset」，不含 host 分循环偏移；
        // output_ 已前移 curSize，写对端必须补同样偏移，否则每个 loop 都写到对端第一个 loop 的区域。
        if (symOutEnable_ == 1) {
            const uint64_t mySymOut = reinterpret_cast<uint64_t>(GetGmSymOut(rank_));
            symLoopOffset_ = (output_ >= mySymOut) ? (output_ - mySymOut) : 0;
        }
        uint64_t smallDataSize = 512 * 1024;
        // 少核或小数据量场景：一核多 rank
        if (PathCoreNum() < rankSize_ || dataSize_ <= smallDataSize) {
            ProcessMultiRank();
        } else {
            // 多核大数据量场景
            ProcessMultiCore();
        }
    }

    // 参与计算的核数：对称路径按 A2A_SYM_MAX_CORES 收敛；非对称路径沿用全部 numBlocks_
    __aicore__ inline uint32_t PathCoreNum() const
    {
        return (symOutEnable_ == 1) ? EffectiveCoreNum() : static_cast<uint32_t>(numBlocks_);
    }

    // 对称路径的有效核数：min(numBlocks_, max(A2A_SYM_MAX_CORES, rankSize_))，多出的核自然空闲；
    // 收发两侧用同一公式，flag 编码因此一致。
    __aicore__ inline uint32_t EffectiveCoreNum() const
    {
        const uint32_t cap = (rankSize_ > A2A_SYM_MAX_CORES) ? static_cast<uint32_t>(rankSize_) : A2A_SYM_MAX_CORES;
        const uint32_t nb = static_cast<uint32_t>(numBlocks_);
        return (nb > cap) ? cap : nb;
    }

private:
    __aicore__ inline void ProcessMultiCore()
    {
        // 本路径无SyncAll，空闲核直接跳过不影响同步结构
        const bool symPath = (symOutEnable_ == 1);
        const uint32_t cores = PathCoreNum();
        if (coreIdx_ >= cores) {
            return;
        }
        uint32_t dstRank = 0;
        uint32_t coresForDstRank = 0;
        uint32_t coreIdxForDstRank = 0;
        const uint32_t rankNum = static_cast<uint32_t>(rankSize_);
        if (symPath) {
            // 对称路径：按比例分核，任意核数（含非 rankSize 整数倍）也不会算出越界 dstRank
            dstRank = static_cast<uint32_t>(static_cast<uint64_t>(coreIdx_) * rankNum / cores);
            const uint32_t coreStart = CoreStartOfRank(dstRank, cores, rankNum);
            coresForDstRank = CoreStartOfRank(dstRank + 1U, cores, rankNum) - coreStart;
            coreIdxForDstRank = coreIdx_ - coreStart;
        } else {
            // 下发核数由上层框架保证符合控核公式，算子内不校验
            uint32_t coreNumPerDstRank = coreNum_ / rankSize_;
            dstRank = coreIdx_ / coreNumPerDstRank;
            coreIdxForDstRank = coreIdx_ % coreNumPerDstRank;
            coresForDstRank = coreNumPerDstRank;
        }

        // 按核切分数据
        uint64_t sliceOffset;
        uint64_t sliceCount;
        SplitData(len_, coresForDstRank, coreIdxForDstRank, sliceOffset, sliceCount);
        uint64_t sliceOffsetSize = sliceOffset * sizeof(T);
        uint64_t sliceSize = sliceCount * sizeof(T);

        // 对称直写路径：本端input直接写对端output的对称VA，跳过PreCopy与本地staging
        if (symOutEnable_ == 1) {
            srcOffset_ = input_ + dstRank * inputSliceStride_ + sliceOffsetSize;
            dstOffset_ = reinterpret_cast<uint64_t>(GetGmSymOut(dstRank)) + symLoopOffset_ + rank_ * outputSliceStride_
                         + sliceOffsetSize;
            CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, sliceCount);
            pipe_barrier(PIPE_ALL);
            // 先写后通知：告知 dstRank 本切片已落入其 output
            Record(rank_, coreIdx_, curTag_);
            WaitPeerSlices(dstRank, symPath, cores, coresForDstRank, coreIdxForDstRank);
            return;
        }

        // PreCopy阶段
        srcOffset_ = input_ + dstRank * inputSliceStride_ + sliceOffsetSize;
        dstOffset_ = reinterpret_cast<uint64_t>(myGmIn_) + dstRank * dataSize_ + sliceOffsetSize;
        CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, sliceCount);
        pipe_barrier(PIPE_ALL);

        uint64_t setFlagIdx = coreIdx_;
        Record(rank_, setFlagIdx, curTag_);

        // ReadRemote阶段
        WaitPeerSlices(dstRank, symPath, cores, coresForDstRank, coreIdxForDstRank);

        srcOffset_ = reinterpret_cast<uint64_t>(GetGmIn(dstRank)) + rank_ * dataSize_ + sliceOffsetSize;
        dstOffset_ = output_ + dstRank * outputSliceStride_ + sliceOffsetSize;
        CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, sliceCount);
        pipe_barrier(PIPE_ALL);
    }

    // 第 r 个 dstRank 分到的第一个核 = ceil(r*cores/rankSize)；r == rankSize 时等于 cores（对称路径用）
    static __aicore__ inline uint32_t CoreStartOfRank(uint32_t r, uint32_t cores, uint32_t rankNum)
    {
        return static_cast<uint32_t>((static_cast<uint64_t>(r) * cores + rankNum - 1U) / rankNum);
    }

    // 等 dstRank 把属于本 rank 的切片写完：
    //   对称路径：发送核按 dstRank 分组，落进本 rank 组内的核区间 [CoreStartOfRank(rank),
    //   CoreStartOfRank(rank+1))，逐个等；非对称路径：单个 flag 下标 rank_*coresForDstRank + coreIdxForDstRank
    __aicore__ inline void
    WaitPeerSlices(uint32_t dstRank, bool symPath, uint32_t cores, uint32_t coresForDstRank, uint32_t coreIdxForDstRank)
    {
        if (!symPath) {
            WaitFlag(dstRank, rank_ * coresForDstRank + coreIdxForDstRank, curTag_);
            return;
        }
        const uint32_t rankNum = static_cast<uint32_t>(rankSize_);
        const uint32_t begin = CoreStartOfRank(static_cast<uint32_t>(rank_), cores, rankNum);
        const uint32_t end = CoreStartOfRank(static_cast<uint32_t>(rank_) + 1U, cores, rankNum);
        for (uint32_t c = begin; c < end; ++c) {
            WaitFlag(dstRank, c, curTag_);
        }
    }

    __aicore__ inline void ProcessMultiRank()
    {
        // 一核多 rank：核按 ceil(rankSize/cores) 个一组分摊 dstRank，核数多于 rankSize 时后面的核自然无事可做
        const uint32_t cores = PathCoreNum();
        uint32_t rankNumPerCore = (rankSize_ + cores - 1U) / cores; // 向上取整

        if (symOutEnable_ == 1) {
            for (uint32_t idx = 0; idx < rankNumPerCore; ++idx) {
                uint32_t dstRank = coreIdx_ * rankNumPerCore + idx;
                if (dstRank >= rankSize_) {
                    break;
                }
                srcOffset_ = input_ + dstRank * inputSliceStride_;
                dstOffset_
                    = reinterpret_cast<uint64_t>(GetGmSymOut(dstRank)) + symLoopOffset_ + rank_ * outputSliceStride_;
                CpGM2GM((__gm__ T*)dstOffset_, (__gm__ T*)srcOffset_, len_);
                pipe_barrier(PIPE_ALL);
                Record(rank_, dstRank, curTag_);
            }
            for (uint32_t idx = 0; idx < rankNumPerCore; ++idx) {
                uint32_t dstRank = coreIdx_ * rankNumPerCore + idx;
                if (dstRank >= rankSize_) {
                    break;
                }
                WaitFlag(dstRank, rank_, curTag_);
            }
            return;
        }

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
    uint64_t symLoopOffset_ = 0; // 对称路径：output_ 相对本 rank 对称窗口基址的偏移（host 分循环时需自己补）
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
    uint64_t countLeft = op.len_;
    uint64_t maxCountPerLoop = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / op.rankSize_ / sizeof(T);

    int32_t loopTag = op.tag_;
    uint32_t barrierSeq = 0;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        op.len_ = curCount;
        op.InitCommon(loopTag);
        op.Process();
        op.BarrierAll(barrierSeq);
        ++barrierSeq;

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
