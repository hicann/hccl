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
class AivAllReduceMesh1DTwoShot : public AivCommBase {
public:
    __aicore__ inline AivAllReduceMesh1DTwoShot()
    {
    }

    __aicore__ inline void Prepare(uint32_t sliceId)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        coreIdx = blockIdx_;
        coreNum = numBlocks_;
        count = len_;

        // 将core分组
        producerNum = rankSize_;
        groupSize = (coreNum - producerNum) / producerNum;  // 一个producer对应几个consumer，整数倍
        consumerNum = groupSize * producerNum;
        // 返回多余的核
        if (coreIdx >= producerNum + consumerNum) {
            return;
        }
        // 计算核所在组的索引，以及和在组内的索引
        if (coreIdx < producerNum) {
            groupIdx = coreIdx;
            idxInGroup = 0;
        } else {
            groupIdx = (coreIdx - producerNum) / groupSize;
            idxInGroup = (coreIdx - producerNum) % groupSize;
        }

        // 数据切分
        sliceNum = consumerNum;
        smallSliceCount = count / sliceNum;
        smallSliceSize = smallSliceCount * sizeof(T);
        bigSliceNum = count % sliceNum;  // 大片数据的数量
        bigSliceCount = smallSliceCount + 1;
        bigSliceSize = bigSliceCount * sizeof(T);
        // HcclBuffer上的数据按照chunkSize对齐存放，便于计算偏移
        if (bigSliceNum == 0) {
            chunkSize = smallSliceSize;
        } else {
            chunkSize = bigSliceSize;
        }

        // 基础数据偏移计算
        storeBuffOffset = 0;  // HcclBuffer上存储数据的Buffer偏移
        reduceBuffOffset = chunkSize * sliceNum;  // HcclBuffer上Reduce计算的Buffer偏移

        // 基础flag偏移计算
        preCopyFlagOffset = 0;
        getRemoteFlagOffset = preCopyFlagOffset + consumerNum;
        localReduceFlagOffset = getRemoteFlagOffset + consumerNum;
    }

    __aicore__ inline void Process()
    {
        ReduceScatter();
        AllGather();
    }

    __aicore__ inline void ReduceScatter()
    {
        if (coreIdx < producerNum) {
            ReduceScatterPreCopy();
        } else if (coreIdx < producerNum + consumerNum) {
            ReduceScatterGetRemote();
            ReduceScatterLocalReduce();
        } else {
            return;
        }
    }

    __aicore__ inline void ReduceScatterPreCopy()
    {
        uint64_t processCount;
        uint64_t sliceIdx;
        uint64_t srcOffset;
        uint64_t dstOffset;
        for (uint64_t idx = 0; idx < groupSize; ++idx) {
            sliceIdx = groupIdx * groupSize + idx;
            if (sliceIdx < bigSliceNum) {
                processCount = bigSliceCount;
                srcOffset = sliceIdx * bigSliceSize;
            } else {
                processCount = smallSliceCount;
                srcOffset = bigSliceNum * bigSliceSize + (sliceIdx - bigSliceNum) * smallSliceSize;
            }
            dstOffset = storeBuffOffset + sliceIdx * chunkSize;

            if (processCount > 0) {
                uint64_t src = input_ + srcOffset;
                uint64_t dst = reinterpret_cast<uint64_t>(GM_IN[rank_]) + dstOffset;
                CpGM2GM((__gm__ T *)dst, (__gm__ T *)src, processCount);
                pipe_barrier(PIPE_ALL);
            }

            // 通知数据从input搬运到output完成
            Record(rank_, preCopyFlagOffset + groupIdx * groupSize + idx, curTag_);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ReduceScatterGetRemote()
    {
        // 核按照远端rank分组，每组核负责读取一个远端rank的数据
        uint64_t rmtRank = groupIdx;
        // 数据源的索引和偏移
        uint64_t srcSliceIdx = rank_ * groupSize + idxInGroup;
        uint64_t processCount;
        if (srcSliceIdx < bigSliceNum) {
            processCount = bigSliceCount;
        } else {
            processCount = smallSliceCount;
        }
        uint64_t srcOffset = storeBuffOffset + srcSliceIdx * chunkSize;
        // 数据目的索引和偏移
        uint64_t dstSliceIdx = rmtRank * groupSize + idxInGroup;
        uint64_t dstOffset = reduceBuffOffset + dstSliceIdx * chunkSize;

        WaitFlag(rmtRank, preCopyFlagOffset + srcSliceIdx, curTag_);

        if (processCount > 0) {
            uint64_t src = reinterpret_cast<uint64_t>(GM_IN[rmtRank]) + srcOffset;
            uint64_t dst = reinterpret_cast<uint64_t>(GM_IN[rank_]) + dstOffset;
            CpGM2GM((__gm__ T *)dst, (__gm__ T *)src, processCount);
            pipe_barrier(PIPE_ALL);
        }

        Record(rank_, getRemoteFlagOffset + dstSliceIdx, curTag_);
    }

    __aicore__ inline void ReduceScatterLocalReduce()
    {
        // 确定性计算，只需要ratio个核进行LocalReduce
        // 负责从本端rank搬运至本端rank的一组核，理论效率比读取远端快，用这组核来做Reduce可以减少性能抖动
        if (groupIdx != rank_) {
            return;
        }

        // 计算要处理的数据片的大小
        uint64_t dstSliceIdx = rank_ * groupSize + idxInGroup;
        uint64_t processCount;
        if (dstSliceIdx < bigSliceNum) {
            processCount = bigSliceCount;
        } else {
            processCount = smallSliceCount;
        }

        // 统一Reduce到本端rank的数据上
        uint64_t dstOffset = reduceBuffOffset + dstSliceIdx * chunkSize;
        uint64_t dst = reinterpret_cast<uint64_t>(GM_IN[rank_]) + dstOffset;
        for (uint64_t dataIdx = 0; dataIdx < rankSize_; ++dataIdx) {
            if (dataIdx == rank_) {
                continue;
            }

            uint64_t srcSliceIdx = dataIdx * groupSize + idxInGroup;
            WaitFlag(rank_, getRemoteFlagOffset + srcSliceIdx, curTag_);
            if (processCount > 0) {
                uint64_t srcOffset = reduceBuffOffset + srcSliceIdx * chunkSize;
                uint64_t src = reinterpret_cast<uint64_t>(GM_IN[rank_]) + srcOffset;
                CpGM2GM((__gm__ T *)dst, (__gm__ T *)src, processCount, reduceOp_);
                pipe_barrier(PIPE_ALL);
            }
        }
        // 单核全部Reduce完成再Record
        Record(rank_, localReduceFlagOffset + idxInGroup, curTag_);
    }

    __aicore__ inline void AllGather()
    {
        if (coreIdx >= producerNum && coreIdx < producerNum + consumerNum) {
            AllGatherGetRemote();
        } else {
            return;
        }
    }

    __aicore__ inline void AllGatherGetRemote()
    {
        uint64_t rmtRank = groupIdx;

        // 远端rank是最后一个rank时，本端每组最后一个核搬运尾块
        uint64_t sliceIdx = rmtRank * groupSize + idxInGroup;
        uint64_t processCount;
        uint64_t dstOffset;
        if (sliceIdx < bigSliceNum) {
            processCount = bigSliceCount;
            dstOffset = sliceIdx * bigSliceSize;
        } else {
            processCount = smallSliceCount;
            dstOffset = bigSliceNum * bigSliceSize + (sliceIdx - bigSliceNum) * smallSliceSize;
        }
        uint64_t srcOffset = reduceBuffOffset + sliceIdx * chunkSize;

        WaitFlag(rmtRank, localReduceFlagOffset + idxInGroup, curTag_);

        if (processCount > 0) {
            uint64_t src = reinterpret_cast<uint64_t>(GM_IN[rmtRank]) + srcOffset;
            uint64_t dst = output_ + dstOffset;
            CpGM2GM((__gm__ T *)dst, (__gm__ T *)src, processCount);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline uint64_t RoundUPAiv(uint64_t dividend, uint64_t divisor)
    {
        return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
    }

    __aicore__ inline void SmallCoreReduceScatter(uint32_t sliceId)
    {
        uint64_t dataCount     = len_;
        rankChunkStride        = RoundUPAiv(dataCount, rankSize_);
        ipc_reduce_flag_offset = rankSize_;
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);

        // scatter
        for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
            targetRank = blockIdx_ + i * numBlocks_;
            rankChunkSize
                = ((targetRank + 1) * rankChunkStride <= dataCount)
                    ? rankChunkStride
                    : (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));
    
            if (rankChunkSize > 0) {
                uint64_t inputOffset = input_ + (targetRank * rankChunkStride) * sizeof(T);
                uint64_t outputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) +
                    (rank_ * rankChunkSize) * sizeof(T);
                CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, rankChunkSize);
                pipe_barrier(PIPE_ALL);
            }
            // set flag
            Record(targetRank, rank_, curTag_);
        }
        // reduce
        if (blockIdx_ == numBlocks_ - 1) {
            uint64_t myRankChuckSize
                = ((rank_ + 1) * rankChunkStride <= dataCount)
                    ? rankChunkStride
                    : (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
            ;
            if (myRankChuckSize > 0) {
                for (uint32_t i = 0; i < rankSize_; i++) {
                    WaitFlag(rank_, i, curTag_);
                    if (i == 0) { // 需要等待第0个rank的scatter数据到达，但是不能自身重复reduce
                        continue;
                    }
                    // reduce
                    uint64_t inputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) +
                        (i * myRankChuckSize) * sizeof(T);
                    uint64_t outputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]);
                    CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, myRankChuckSize, reduceOp_);
                    pipe_barrier(PIPE_ALL);
                }
            }
            Record(rank_, ipc_reduce_flag_offset + 1, curTag_);
        }
    }
    
    __aicore__ inline void SmallCoreAllgather()
    {
        uint64_t dataCount = len_;
        for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
            targetRank = blockIdx_ + i * numBlocks_;
            rankChunkSize
                = ((targetRank + 1) * rankChunkStride <= dataCount)
                    ? rankChunkStride
                    : (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));
    
            if (rankChunkSize <= 0) {
                return;
            }
    
            // waitflag
            WaitFlag(targetRank, ipc_reduce_flag_offset + 1, curTag_);
    
            // gather
            uint64_t inputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]);
            uint64_t outputOffset = output_ + (targetRank * rankChunkStride) * sizeof(T);
            CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, rankChunkSize);
            pipe_barrier(PIPE_ALL);
        }
    }

    // 基础信息
    uint64_t coreIdx{0};
    uint64_t coreNum{0};
    uint64_t count{0};
    // 分组信息
    uint64_t producerNum{0};
    uint64_t groupSize{0};
    uint64_t consumerNum{0};
    uint64_t groupIdx{0};
    uint64_t idxInGroup{0};
    // 数据切分信息
    uint64_t sliceNum{0};
    uint64_t bigSliceNum{0};
    uint64_t bigSliceCount{0};
    uint64_t bigSliceSize{0};
    uint64_t smallSliceCount{0};
    uint64_t smallSliceSize{0};
    uint64_t chunkSize{0};
    // HcclBuffer偏移信息
    uint64_t storeBuffOffset{0};
    uint64_t reduceBuffOffset{0};
    // flag偏移信息
    uint64_t preCopyFlagOffset{0};
    uint64_t getRemoteFlagOffset{0};
    uint64_t localReduceFlagOffset{0};
    // 少核场景
    uint32_t targetRank;
    uint64_t rankChunkSize;
    uint64_t rankChunkStride;
    uint64_t ipc_reduce_flag_offset{1024};
};

template<typename T>
__aicore__ inline void AivAllReduceV2Mesh1DTwoShot(KERNEL_ARGS_DEF)
{
    AivAllReduceMesh1DTwoShot<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    if (op.numBlocks_ >= rankSize * 2) {
        op.Prepare(sliceId);
        op.Process();
    } else {
        op.SmallCoreReduceScatter(sliceId);
        op.SmallCoreAllgather();
    }
    op.BarrierAll();
}

template<typename T>
__aicore__ inline void AivAllReduceV2Mesh1DTwoShotSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    // 参数中的input和output是分别加了步长inBuffBaseOff和outBuffBaseOff步长
    AivAllReduceMesh1DTwoShot<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT);

    uint64_t maxCountPerLoop = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / op.rankSize_ / sizeof(T);
    uint64_t countLeft = op.len_;

    int32_t loopTag = op.tag_;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);
        op.len_ = curCount;
        if (op.numBlocks_ >= op.rankSize_ * 2) {
            op.Prepare(loopTag);
            op.Process();
        } else {
            op.SmallCoreReduceScatter(loopTag);
            op.SmallCoreAllgather();
        }
        op.BarrierAll();

        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}

__aicore__ inline void sk_ar_mesh_1d_twoshot(SUPERKERNEL_ARGS_DEF)
{
    #ifdef HCCL_DTYPE_INT8
        AivAllReduceV2Mesh1DTwoShotSuperKernel<int8_t> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_INT16
        AivAllReduceV2Mesh1DTwoShotSuperKernel<int16_t> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_INT32
        AivAllReduceV2Mesh1DTwoShotSuperKernel<int32_t> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_FP16
        AivAllReduceV2Mesh1DTwoShotSuperKernel<half> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_FP32
        AivAllReduceV2Mesh1DTwoShotSuperKernel<float> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_BFP16
        AivAllReduceV2Mesh1DTwoShotSuperKernel<bfloat16_t> (SUPERKERNEL_ARGS_CALL);
    #else
    #endif
}
