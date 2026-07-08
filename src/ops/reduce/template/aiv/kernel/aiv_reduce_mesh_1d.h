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
class AivReduceMesh1DTwoShot : public AivCommBase {
public:
    __aicore__ inline AivReduceMesh1DTwoShot()
    {
    }

    __aicore__ inline void Process(int32_t sliceId)
    {
        maxCoreNum_ = static_cast<uint32_t>(numBlocks_);
        if (maxCoreNum_ >= rankSize_ + 1) {
            ProcessMultiCore(sliceId);
        } else if (maxCoreNum_ >= 1) {
            ProcessMultiRank(sliceId);
        } else {
            // 下发0核场景，算子不操作，由Template层处理异常场景
            return;
        }
    }

private:
    __aicore__ inline void ProcessMultiCore(int32_t sliceId)
    {
        InitCoreInfo(sliceId);
        ReduceScatter();
        GatherToRoot();
    }

    __aicore__ inline uint64_t RoundUp(uint64_t dividend, uint64_t divisor)
    {
        if (divisor == 0) {
            return dividend;
        }
        return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
    }

    __aicore__ inline void InitCoreInfo(int32_t sliceId)
    {
        curTag = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        uint64_t dataCount = len_;
        coreNumPerRank = numBlocks_ / (rankSize_ + 1);
        coreNumFirstStage = coreNumPerRank * rankSize_;
        coreNumTotal = coreNumPerRank * (rankSize_ + 1);

        innerChunkStride = RoundUp(dataCount, coreNumFirstStage);
        rankChunkStride = innerChunkStride * coreNumPerRank;
        if (blockIdx_ < coreNumFirstStage) {
            targetRank = blockIdx_ / coreNumPerRank;
            rankChunkSize = ((targetRank + 1) * rankChunkStride <= dataCount)
                ? rankChunkStride
                : (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));
        } else if (blockIdx_ < coreNumTotal) {
            targetRank = rank_;
            rankChunkSize = ((rank_ + 1) * rankChunkStride <= dataCount)
                ? rankChunkStride
                : (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
        }

        if (blockIdx_ < coreNumTotal) {
            innerId = blockIdx_ % coreNumPerRank;
            innerChunkSize = ((innerId + 1) * innerChunkStride <= rankChunkSize)
                ? innerChunkStride
                : (rankChunkSize <= innerId * innerChunkStride ? 0 : (rankChunkSize - innerId * innerChunkStride));
        }
        ipcReduceFlagOffset = coreNumFirstStage;
    }

    __aicore__ inline void ReduceScatter()
    {
        if (blockIdx_ < coreNumFirstStage) {
            if (innerChunkSize > 0) {
                uint64_t inputOffset =
                    input_ + (targetRank * rankChunkStride + innerId * innerChunkStride) * sizeof(T);
                uint64_t outputOffset =
                    reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (rank_ * rankChunkSize + innerId * innerChunkStride) * sizeof(T);
                CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, innerChunkSize);
                pipe_barrier(PIPE_ALL);
            }
            Record(targetRank, rank_ * coreNumPerRank + innerId, curTag);
        } else if (blockIdx_ < coreNumTotal) {
            if (innerChunkSize > 0) {
                for (uint32_t i = 0; i < rankSize_; i++) {
                    WaitFlag(rank_, i * coreNumPerRank + innerId, curTag);
                    if (i == 0) {
                        continue;
                    }
                    uint64_t inputOffset =
                        reinterpret_cast<uint64_t>(GM_IN[rank_]) + (i * rankChunkSize + innerId * innerChunkStride) * sizeof(T);
                    uint64_t outputOffset =
                        reinterpret_cast<uint64_t>(GM_IN[rank_]) + (innerId * innerChunkStride) * sizeof(T);
                    CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, innerChunkSize, reduceOp_);
                    pipe_barrier(PIPE_ALL);
                }
            } else {
                // 没有数据要处理，也需要接收同步信号保证时序，防止后续Gather数据异常
                for (uint32_t i = 0; i < rankSize_; i++) {
                    WaitFlag(rank_, i * coreNumPerRank + innerId, curTag);
                }
            }
            Record(rank_, ipcReduceFlagOffset + innerId, curTag);
        }
    }

    __aicore__ inline void GatherToRoot()
    {
        if (rank_ != root_ || blockIdx_ >= coreNumFirstStage) {
            return;
        }
        WaitFlag(targetRank, ipcReduceFlagOffset + innerId, curTag);
        if (innerChunkSize > 0) {
            uint64_t inputOffset =
                reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (innerId * innerChunkStride) * sizeof(T);
            uint64_t outputOffset = output_ + (targetRank * rankChunkStride + innerId * innerChunkStride) * sizeof(T);
            CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, innerChunkSize);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ProcessMultiRank(int32_t sliceId)
    {
        InitMultiRank(sliceId);
        ReduceScatterMultiRank();
        GatherMultiRank();
    }

    __aicore__ inline void InitMultiRank(int32_t sliceId)
    {
        useCoreNum_ = maxCoreNum_;  // 少核场景使用全核
        maxRankPerCore_ = (rankSize_ + maxCoreNum_ - 1) / maxCoreNum_;  // 向上取整
        coreIdx_ = static_cast<uint32_t>(blockIdx_);

        uint64_t dataCount = len_;
        sliceCount_ = dataCount / rankSize_;
        tailCount_ = dataCount - (rankSize_ - 1) * sliceCount_;

        syncTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        reduceTagOffset_ = rankSize_;  // reduce流程的tag起始偏移
    }

    __aicore__ inline void ReduceScatterMultiRank()
    {
        // 向远端写数据
        for (uint32_t i = 0; i < maxRankPerCore_; ++i) {
            // 每个core负责的rank间隔为useCoreNum_
            uint32_t targetRank = i * useCoreNum_ + coreIdx_;
            if (targetRank >= rankSize_) {
                break;
            }

            uint64_t processCount = (targetRank == rankSize_ - 1) ? tailCount_ : sliceCount_;
            if (processCount > 0) {
                uint64_t srcOffset = input_ + targetRank * sliceCount_ * sizeof(T);
                uint64_t dstOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + rank_ * processCount * sizeof(T);
                CpGM2GM((__gm__ T *)dstOffset, (__gm__ T *)srcOffset, processCount);
                pipe_barrier(PIPE_ALL);
            }
            Record(targetRank, rank_, syncTag_);
        }

        // 本地reduce
        if (coreIdx_ != useCoreNum_ - 1) {
            return;
        }
        // 用最后一个core做reduce，reduce至首片数据位置
        uint64_t reduceCount = sliceCount_;
        if (rank_ == rankSize_ - 1) {
            reduceCount = tailCount_;
        }
        WaitFlag(rank_, 0, syncTag_);
        for (uint32_t sliceIdx = 1; sliceIdx < rankSize_; ++sliceIdx) {
            WaitFlag(rank_, sliceIdx, syncTag_);
            if (reduceCount > 0) {
                uint64_t srcOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + sliceIdx * reduceCount * sizeof(T);
                uint64_t dstOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]);
                CpGM2GM((__gm__ T *)dstOffset, (__gm__ T *)srcOffset, reduceCount, reduceOp_);
                pipe_barrier(PIPE_ALL);
            }
        }
        Record(rank_, reduceTagOffset_, syncTag_);
    }

    __aicore__ inline void GatherMultiRank()
    {
        // root读远端数据
        if (rank_ != root_) {
            return;
        }
        for (uint32_t i = 0; i < maxRankPerCore_; ++i) {
            // 每个core负责的rank间隔为useCoreNum_
            uint32_t targetRank = i * useCoreNum_ + coreIdx_;
            if (targetRank >= rankSize_) {
                break;
            }

            uint64_t processCount = (targetRank == rankSize_ - 1) ? tailCount_ : sliceCount_;
            if (processCount > 0) {
                WaitFlag(targetRank, reduceTagOffset_, syncTag_);
                uint64_t srcOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]);
                uint64_t dstOffset = output_ + targetRank * sliceCount_ * sizeof(T);
                CpGM2GM((__gm__ T *)dstOffset, (__gm__ T *)srcOffset, processCount);
                pipe_barrier(PIPE_ALL);
            }
        }
    }

    uint32_t coreNumPerRank;
    uint32_t coreNumFirstStage;
    uint32_t coreNumTotal;
    uint32_t innerId;
    uint32_t targetRank;
    uint64_t rankChunkSize;
    uint64_t innerChunkSize;
    uint64_t rankChunkStride;
    uint64_t innerChunkStride;
    int32_t curTag;
    uint64_t ipcReduceFlagOffset{1024};

    uint32_t syncTag_{0};
    uint32_t maxCoreNum_{0};
    uint32_t useCoreNum_{0};
    uint32_t coreIdx_{0};
    uint32_t maxRankPerCore_{0};
    uint64_t sliceCount_{0};
    uint64_t tailCount_{0};
    uint64_t reduceTagOffset_{0};
};

template<typename T>
class AivReduceMesh1D : public AivCommBase {
    constexpr static uint64_t DATA_SLICE_NUM = 64 * 1024;
public:
    __aicore__ inline AivReduceMesh1D() {}
 
    __aicore__ inline void InitCoreInfo()
    {
        dataSize_ = len_ * sizeof(T);
        // 小数据量情况下，缩减实际使用核数
        useBlocks_ = (dataSize_ + DATA_SLICE_NUM - 1) / DATA_SLICE_NUM;
        if (useBlocks_ > rankSize_) {
            useBlocks_ = rankSize_;
        }
        if (blockIdx_ >= useBlocks_) {
            return;
        }
        SplitData(len_, sliceLen_, offsetLen_);
        offsetSize_ = offsetLen_ * sizeof(T);
        if (rank_ < root_) {
            srcOffset_ = input_ + offsetSize_;
            dstOffset_ = reinterpret_cast<uint64_t>(GM_IN[root_]) + offsetSize_ + rank_ * dataSize_;
        } else if (rank_ > root_) {
            srcOffset_ = input_ + offsetSize_;
            dstOffset_ = reinterpret_cast<uint64_t>(GM_IN[root_]) + offsetSize_ + (rank_ - 1) * dataSize_;
        } else {
            srcOffset_ = input_ + offsetSize_;
            dstOffset_ = output_ + offsetSize_;
        }
    }
 
    __aicore__ inline void Process(int32_t sliceId)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        if (blockIdx_ >= useBlocks_) {
            return;
        }
        if (rank_ != root_) {
            // 写远端：将自身core负责的Input数据搬运至root的Scratch上
            if (sliceLen_ > 0) {
                CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, sliceLen_);
                PipeBarrier<PIPE_ALL>();
            }
            // 写同步：将aivTag写入root上的数据同步标志位，表示数据搬运完成
            uint64_t flagOffset;
            if (rank_ < root_) {
                flagOffset = rank_ * useBlocks_ + blockIdx_;
            } else {
                flagOffset = (rank_ - 1) * useBlocks_ + blockIdx_;
            }
            Record(root_, flagOffset, curTag_);
        } else {
            // 本地拷贝：将自身core负责的Input数据搬运至本地Output上
            if (sliceLen_ > 0) {
                CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, sliceLen_);
                PipeBarrier<PIPE_ALL>();
            }
            uint32_t sliceIdx = 0;
            for (uint32_t dataRank = 0; dataRank < rankSize_; dataRank++) {
                if (dataRank == rank_) {
                    continue;
                }
                // 读同步：阻塞读取本地数据同步标志位，当前aivTag等于读取值时，继续步骤
                uint64_t flagOffset = sliceIdx * useBlocks_ + blockIdx_;
                WaitFlag(rank_, flagOffset, curTag_);
                // 本地规约：将本地ScratchBuffer上的数据Reduce到本地OutputBuffer上
                if (sliceLen_ > 0) {
                    srcOffset_ = reinterpret_cast<uint64_t>(GM_IN[root_]) + sliceIdx * dataSize_ + offsetSize_;
                    CpGM2GM((__gm__ T *)dstOffset_, (__gm__ T *)srcOffset_, sliceLen_, reduceOp_);
                    PipeBarrier<PIPE_ALL>();
                }
                sliceIdx++;
            }
        }
    }
 
    __aicore__ inline void SplitData(uint64_t dataLen, uint64_t& sliceLen, uint64_t& offsetLen)
    {
        uint64_t sliceLenMin = dataLen / useBlocks_;
        uint64_t remainLen = dataLen % useBlocks_;
        // remainLen必然小于dataLen，均分给前remainLen个aiv处理
        if (blockIdx_ < remainLen) {
            sliceLen = sliceLenMin + 1;
            offsetLen = blockIdx_ * (sliceLenMin + 1);
        } else {
            sliceLen = sliceLenMin;
            offsetLen = remainLen * (sliceLenMin + 1) + (blockIdx_ - remainLen) * sliceLenMin;
        }
    }

    uint64_t useBlocks_;
    uint64_t dataSize_;
    uint64_t sliceLen_;
    uint64_t offsetLen_;
    uint64_t offsetSize_;
    uint64_t srcOffset_;
    uint64_t dstOffset_;
};
 
template<typename T>
__aicore__ inline void AivReduceV2Mesh1D(KERNEL_ARGS_DEF)
{
    AivReduceMesh1DTwoShot<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    op.Process(sliceId);
    SyncAll<true>();
    op.BarrierAll();
}