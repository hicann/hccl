/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_all_reduce_mesh_1d_twoshot.h"

using namespace AscendC;

// 基于aiv_all_reduce_mesh_1d_twoshot.h，producer逐片直写对端归约区并发布就绪通知。
// RS根据发送字节数和归约粒度，为每个目的rank选择1个或2个producer。
// 每个原分片由一个consumer归约并直接发布完成通知，AG保留全部回拷核。少核复用读语义TwoShot。
// 本文件仅供单算子写语义入口使用，图模式仍使用原TwoShot内核。
template <typename T>
class AivAllReduceMesh1DTwoShotWrite : public AivCommBase {
public:
    __aicore__ inline AivAllReduceMesh1DTwoShotWrite() {}

    __aicore__ inline void Prepare(uint32_t sliceId)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        coreIdx = blockIdx_;
        coreNum = numBlocks_;
        count = len_;

        // 双producer仅用于发送量足够且剩余consumer仍有足量归约工作的场景。
        uint64_t effectiveCoreNum = coreNum / rankSize_ * rankSize_;
        producerNum = rankSize_;
        uint64_t candidateProducerNum = MAX_PRODUCER_NUM_PER_RANK * rankSize_;
        if (effectiveCoreNum >= candidateProducerNum + rankSize_) {
            uint64_t candidateGroupSize = (effectiveCoreNum - candidateProducerNum) / rankSize_;
            uint64_t candidateSliceNum = candidateGroupSize * rankSize_;
            uint64_t candidateSmallCount = count / candidateSliceNum;
            uint64_t candidateBigNum = count % candidateSliceNum;
            // 最后一个目的rank的1号producer工作量最少，计入其实际领取的大片。
            uint64_t lastRankSliceBegin = (rankSize_ - 1) * candidateGroupSize;
            uint64_t lastRankBigNum = candidateBigNum > lastRankSliceBegin ? candidateBigNum - lastRankSliceBegin : 0;
            uint64_t minProducerCount = (candidateGroupSize / MAX_PRODUCER_NUM_PER_RANK) * candidateSmallCount
                                        + lastRankBigNum / MAX_PRODUCER_NUM_PER_RANK;
            // 每个归约核负责一个完整分片。沿用原门槛，要求最小分片至少为4KiB乘以rank数。
            if (minProducerCount >= MIN_BYTES_PER_PRODUCER / sizeof(T)
                && candidateSmallCount / rankSize_ >= MIN_BYTES_PER_RANK_REDUCTION_BUDGET / sizeof(T)) {
                producerNum = candidateProducerNum;
            }
        }
        groupSize = (effectiveCoreNum - producerNum) / rankSize_;
        consumerNum = groupSize * rankSize_;
        // 返回多余的核
        if (coreIdx >= producerNum + consumerNum) {
            return;
        }
        // 计算核所在组的索引，以及和在组内的索引
        if (coreIdx < producerNum) {
            groupIdx = coreIdx % rankSize_;
            producerIdx_ = coreIdx / rankSize_;
            idxInGroup = 0;
        } else {
            groupIdx = (coreIdx - producerNum) / groupSize;
            idxInGroup = (coreIdx - producerNum) % groupSize;
        }

        // 数据切分
        sliceNum = consumerNum;
        smallSliceCount = count / sliceNum;
        smallSliceSize = smallSliceCount * sizeof(T);
        bigSliceNum = count % sliceNum; // 大片数据的数量
        bigSliceCount = smallSliceCount + 1;
        bigSliceSize = bigSliceCount * sizeof(T);
        // HcclBuffer上的数据按照chunkSize对齐存放，便于计算偏移
        if (bigSliceNum == 0) {
            chunkSize = smallSliceSize;
        } else {
            chunkSize = bigSliceSize;
        }

        // 基础数据偏移计算
        storeBuffOffset = 0;                     // HcclBuffer上存储数据的Buffer偏移
        reduceBuffOffset = chunkSize * sliceNum; // HcclBuffer上Reduce计算的Buffer偏移

        // 保留原consumerReduce区的空间；逐片归约直接发布localReduce通知。
        getRemoteFlagOffset = consumerNum;
        localReduceFlagOffset = getRemoteFlagOffset + consumerNum;
        allGatherFlagOffset_ = localReduceFlagOffset + groupSize;
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
        uint64_t producerNumPerRank = producerNum / rankSize_;
        for (uint64_t idx = producerIdx_; idx < groupSize; idx += producerNumPerRank) {
            sliceIdx = groupIdx * groupSize + idx;
            if (sliceIdx < bigSliceNum) {
                processCount = bigSliceCount;
                srcOffset = sliceIdx * bigSliceSize;
            } else {
                processCount = smallSliceCount;
                srcOffset = bigSliceNum * bigSliceSize + (sliceIdx - bigSliceNum) * smallSliceSize;
            }
            dstOffset = reduceBuffOffset + (rank_ * groupSize + idx) * chunkSize;

            if (processCount > 0) {
                uint64_t src = input_ + srcOffset;
                uint64_t dst = reinterpret_cast<uint64_t>(GetGmIn(groupIdx)) + dstOffset;
                CpGM2GM((__gm__ T*)dst, (__gm__ T*)src, processCount);
                pipe_barrier(PIPE_ALL);
            }

            // 直接通知对端归约核：本rank的数据片已直写其归约区，空片也发布就绪。
            Record(groupIdx, getRemoteFlagOffset + rank_ * groupSize + idx, curTag_);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ReduceScatterLocalReduce()
    {
        // 仅本rank对应的一组consumer参与归约，每核独立负责一个完整原分片。
        if (groupIdx != rank_) {
            return;
        }
        uint64_t dstSliceIdx = rank_ * groupSize + idxInGroup;
        uint64_t processCount = dstSliceIdx < bigSliceNum ? bigSliceCount : smallSliceCount;
        if (processCount == 0) {
            Record(rank_, localReduceFlagOffset + idxInGroup, curTag_);
            return;
        }
        // 多rank由树形折叠在首次使用槽位时等待；单rank无折叠，需单独确认输入就绪。
        if (rankSize_ == 1) {
            WaitFlag(rank_, getRemoteFlagOffset + dstSliceIdx, curTag_);
        }

        // 树形折叠（与 Aicpu 保序 RunLocalReduce 同序）：累加器 = rank0 的 slice 槽（Aicpu 基准），slot(dst) +=
        // slot(src) 零存储 flag 等待：源槽仅当 src >= k 时等待（[1,k) 已在作为目的槽被折入时等待过）；
        // 目的槽仅首次被折入时等待（新目的槽只出现在第 0/1 轮）
        uint64_t baseOffset = reduceBuffOffset + idxInGroup * chunkSize;
        uint64_t base = reinterpret_cast<uint64_t>(myGmIn_) + baseOffset;
        uint32_t curBlocks = static_cast<uint32_t>(rankSize_);
        uint32_t m0 = GetLargestPowerOf2LessThan(curBlocks);
        uint32_t firstRoundDstNum = curBlocks - m0;
        uint32_t k = (firstRoundDstNum > (m0 >> 1)) ? firstRoundDstNum : (m0 >> 1);
        uint32_t round = 0;
        while (curBlocks > 1) {
            uint32_t M = GetLargestPowerOf2LessThan(curBlocks);
            if (M == 0) {
                break;
            }
            for (uint32_t src = M; src < curBlocks; ++src) {
                uint32_t dst = src % M;
                if (src >= k) {
                    WaitFlag(rank_, getRemoteFlagOffset + src * groupSize + idxInGroup, curTag_);
                }
                if (round == 0 || (round == 1 && dst >= firstRoundDstNum)) {
                    WaitFlag(rank_, getRemoteFlagOffset + dst * groupSize + idxInGroup, curTag_);
                }
                uint64_t srcOffset = reduceBuffOffset + (src * groupSize + idxInGroup) * chunkSize;
                uint64_t srcAddr = reinterpret_cast<uint64_t>(myGmIn_) + srcOffset;
                uint64_t dstOffset = (dst == 0) ? base :
                                                  (reinterpret_cast<uint64_t>(myGmIn_) + reduceBuffOffset
                                                   + (dst * groupSize + idxInGroup) * chunkSize);
                CpGM2GM((__gm__ T*)dstOffset, (__gm__ T*)srcAddr, processCount, reduceOp_);
                pipe_barrier(PIPE_ALL);
            }
            curBlocks = M;
            ++round;
        }
        // 结果留在 rank0 槽（累加器），AllGather 直接读 rank0 槽，无需额外拷贝
        // 整片由本核完成，直接通知AG，消除子区间完成flag及其汇总等待。
        Record(rank_, localReduceFlagOffset + idxInGroup, curTag_);
    }

    __aicore__ inline void AllGather()
    {
        // AG沿用RS发送核数，发送与回拷分别按逻辑任务循环领取。
        if (coreIdx < producerNum) {
            AllGatherPutRemote();
        } else if (coreIdx < producerNum + consumerNum) {
            AllGatherCopyBackLocal();
        } else {
            return;
        }
    }

    __aicore__ inline void AllGatherPutRemote()
    {
        if (rankSize_ == 1) {
            return;
        }
        // 按“本地片序、远端peer”展开任务，producer以自身核号为起点循环领取。
        uint64_t taskCount = groupSize * (rankSize_ - 1);
        for (uint64_t task = coreIdx; task < taskCount; task += producerNum) {
            uint64_t idx = task / (rankSize_ - 1);
            uint64_t dstRank = (rank_ + 1 + task % (rankSize_ - 1)) % rankSize_;
            uint64_t sliceIdx = rank_ * groupSize + idx;
            uint64_t processCount = sliceIdx < bigSliceNum ? bigSliceCount : smallSliceCount;
            WaitFlag(rank_, localReduceFlagOffset + idx, curTag_);
            // 与读kernel一致，归约结果保存在rank0槽。
            uint64_t src = reinterpret_cast<uint64_t>(myGmIn_) + reduceBuffOffset + idx * chunkSize;
            if (processCount > 0) {
                // store区与归约区独立，接收结果不会覆盖尚未发完的本端归约结果。
                uint64_t dst = reinterpret_cast<uint64_t>(GetGmIn(dstRank)) + storeBuffOffset + sliceIdx * chunkSize;
                CpGM2GM((__gm__ T*)dst, (__gm__ T*)src, processCount);
                pipe_barrier(PIPE_ALL);
            }
            // 每个(本地片, 远端rank)只由一个核发送，空片也发布原AG就绪flag。
            Record(dstRank, allGatherFlagOffset_ + sliceIdx, curTag_);
        }
    }

    __aicore__ inline void AllGatherCopyBackLocal()
    {
        // 回拷核沿用RS consumer，按步长领取输出片，保持完整覆盖和唯一归属。
        for (uint64_t sliceIdx = coreIdx - producerNum; sliceIdx < sliceNum; sliceIdx += consumerNum) {
            uint64_t rmtRank = sliceIdx / groupSize;
            uint64_t idx = sliceIdx % groupSize;
            uint64_t processCount;
            uint64_t dstOffset;
            if (sliceIdx < bigSliceNum) {
                processCount = bigSliceCount;
                dstOffset = sliceIdx * bigSliceSize;
            } else {
                processCount = smallSliceCount;
                dstOffset = bigSliceNum * bigSliceSize + (sliceIdx - bigSliceNum) * smallSliceSize;
            }
            uint64_t srcOffset = storeBuffOffset + sliceIdx * chunkSize;

            if (rmtRank == rank_) {
                // 本rank结果直接从归约区回拷，不再经store区中转。
                WaitFlag(rank_, localReduceFlagOffset + idx, curTag_);
                srcOffset = reduceBuffOffset + idx * chunkSize;
            } else {
                WaitFlag(rank_, allGatherFlagOffset_ + sliceIdx, curTag_);
            }

            if (processCount > 0) {
                uint64_t src = reinterpret_cast<uint64_t>(myGmIn_) + srcOffset;
                uint64_t dst = output_ + dstOffset;
                CpGM2GM((__gm__ T*)dst, (__gm__ T*)src, processCount);
                pipe_barrier(PIPE_ALL);
            }
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
    uint64_t getRemoteFlagOffset{0};
    uint64_t localReduceFlagOffset{0};
    uint64_t allGatherFlagOffset_{0};

private:
    static constexpr uint64_t MAX_PRODUCER_NUM_PER_RANK = 2;
    // producer选择沿用原发送量与归约预算门槛，便于独立对比每片单核归约的性能。
    static constexpr uint64_t MIN_BYTES_PER_PRODUCER = UB_DB_DATA_BATCH_SIZE;
    // 沿用原归约预算：按源rank分摊分片字节数，每个源rank至少预留4KiB。
    static constexpr uint64_t MIN_BYTES_PER_RANK_REDUCTION_BUDGET = 4 * 1024;
    uint64_t producerIdx_{0};
};

template <typename T>
__aicore__ inline void AivAllReduceV2Mesh1DTwoShotWrite(KERNEL_ARGS_DEF)
{
    // 少核统一使用读语义TwoShot的搬运和同步流程。
    if (numBlocks < rankSize * 2) {
        return AivAllReduceV2Mesh1DTwoShot<T>(KERNEL_ARGS_CALL);
    }

    AivAllReduceMesh1DTwoShotWrite<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    op.Prepare(sliceId);
    op.Process();
    op.BarrierAll();
}
