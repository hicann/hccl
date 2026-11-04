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
class AivAllReduceMesh1DTwoShot : public AivCommBase {
public:
    __aicore__ inline AivAllReduceMesh1DTwoShot() {}

    __aicore__ inline void Prepare(uint32_t sliceId)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        coreIdx = blockIdx_;
        coreNum = numBlocks_;
        count = len_;

        // 将core分组
        producerNum = rankSize_;
        groupSize = (coreNum - producerNum) / producerNum; // 一个producer对应几个consumer，整数倍
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
                uint64_t dst = reinterpret_cast<uint64_t>(myGmIn_) + dstOffset;
                CpGM2GM((__gm__ T*)dst, (__gm__ T*)src, processCount);
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
            uint64_t src = reinterpret_cast<uint64_t>(GetGmIn(rmtRank)) + srcOffset;
            uint64_t dst = reinterpret_cast<uint64_t>(myGmIn_) + dstOffset;
            CpGM2GM((__gm__ T*)dst, (__gm__ T*)src, processCount);
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
                if (processCount > 0) {
                    uint64_t srcOffset = reduceBuffOffset + (src * groupSize + idxInGroup) * chunkSize;
                    uint64_t src = reinterpret_cast<uint64_t>(myGmIn_) + srcOffset;
                    uint64_t dstOffset = (dst == 0) ? base :
                                                      (reinterpret_cast<uint64_t>(myGmIn_) + reduceBuffOffset
                                                       + (dst * groupSize + idxInGroup) * chunkSize);
                    CpGM2GM((__gm__ T*)dstOffset, (__gm__ T*)src, processCount, reduceOp_);
                    pipe_barrier(PIPE_ALL);
                }
            }
            curBlocks = M;
            ++round;
        }
        // 结果留在 rank0 槽（累加器），AllGather 直接读 rank0 槽，无需额外拷贝
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
        // 归约结果落在各卡的 rank0 槽（reduceBuffOffset + idxInGroup*chunkSize），直接从该槽读取，
        // 消除归约段内的结果拷回本端槽
        uint64_t srcOffset = reduceBuffOffset + idxInGroup * chunkSize;

        WaitFlag(rmtRank, localReduceFlagOffset + idxInGroup, curTag_);

        if (processCount > 0) {
            uint64_t src = reinterpret_cast<uint64_t>(GetGmIn(rmtRank)) + srcOffset;
            uint64_t dst = output_ + dstOffset;
            CpGM2GM((__gm__ T*)dst, (__gm__ T*)src, processCount);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline uint64_t RoundUPAiv(uint64_t dividend, uint64_t divisor)
    {
        return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
    }

    __aicore__ inline void SmallCoreReduceScatter(uint32_t sliceId)
    {
        uint64_t dataCount = len_;
        rankChunkStride = RoundUPAiv(dataCount, rankSize_);
        ipc_reduce_flag_offset = rankSize_;
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        // 镜像区/槽区分界。只依赖 len_，各卡一致，对端可直接算出
        slotBase_ = ((len_ * sizeof(T) + UB_ALIGN_SIZE - 1) / UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        // step1: 本卡 input -> 本卡镜像区，然后写对端 flag。索引 rank_ 落在 targetRank 卡上，
        // 对给定 (源rank, 目标片) 只有唯一一个核，握手已精确到片。
        // SK空闲核跳过，避免与忙核重复写数据和flag。
        if (!IsIdleCore()) {
            for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
                targetRank = blockIdx_ + i * numBlocks_;
                rankChunkSize
                    = ((targetRank + 1) * rankChunkStride <= dataCount) ?
                          rankChunkStride :
                          (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));

                uint64_t chunkOffset = (targetRank * rankChunkStride) * sizeof(T);
                if (rankChunkSize > 0) {
                    uint64_t outputOffset = reinterpret_cast<uint64_t>(myGmIn_) + chunkOffset;
                    CpGM2GM((__gm__ T*)outputOffset, (__gm__ T*)(input_ + chunkOffset), rankChunkSize);
                    pipe_barrier(PIPE_ALL);
                }
                // set flag
                Record(targetRank, rank_, curTag_);
            }
            pipe_barrier(PIPE_ALL);
        }
        // step2: 对端镜像区 -> 本卡槽区。核 c 负责 peer r = c, c+numBlocks_, ...，
        // 每核 ceil(rankSize_/numBlocks_) 次跨卡读，
        // 各核写入的槽互不重叠，可完全并行。槽地址与改动前一致，折叠代码原样复用。
        if (!IsIdleCore()) {
            uint64_t myChunkSize
                = ((rank_ + 1) * rankChunkStride <= dataCount) ?
                      rankChunkStride :
                      (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
            uint64_t peerChunkOffset = (rank_ * rankChunkStride) * sizeof(T);
            for (uint32_t r = blockIdx_; r < rankSize_; r += static_cast<uint32_t>(numBlocks_)) {
                WaitFlag(rank_, r, curTag_);
                if (myChunkSize > 0) {
                    uint64_t srcOffset = reinterpret_cast<uint64_t>(GetGmIn(r)) + peerChunkOffset;
                    uint64_t dstOffset
                        = reinterpret_cast<uint64_t>(myGmIn_) + slotBase_ + (r * myChunkSize) * sizeof(T);
                    CpGM2GM((__gm__ T*)dstOffset, (__gm__ T*)srcOffset, myChunkSize);
                    pipe_barrier(PIPE_ALL);
                }
            }
        }
        // 卡内栅栏：槽区由多核分头填充，折叠核必须等全部就绪
        SyncAll<true>();
        pipe_barrier(PIPE_ALL);
        // step3: 树形折叠(与 Aicpu 保序 RunLocalReduce 同序)，仅槽基址平移到 slotBase_
        if (blockIdx_ == numBlocks_ - 1) {
            uint64_t myRankChuckSize
                = ((rank_ + 1) * rankChunkStride <= dataCount) ?
                      rankChunkStride :
                      (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
            ;
            if (myRankChuckSize > 0) {
                // 树形折叠（与 Aicpu 保序 RunLocalReduce 同序）：累加器 = rank0 槽（初值已是 rank0，
                // 由 step2 跨卡读回填充），slot(dst) += slot(src) 零存储 flag 等待：源槽仅当 src >= k 时
                // 等待（[1,k) 已在作为目的槽被折入时等待过）； 目的槽仅首次被折入时等待（新目的槽只在第 0/1 轮）
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
                            WaitFlag(rank_, src, curTag_);
                        }
                        if (round == 0 || (round == 1 && dst >= firstRoundDstNum)) {
                            WaitFlag(rank_, dst, curTag_);
                        }
                        // reduce
                        uint64_t inputOffset
                            = reinterpret_cast<uint64_t>(myGmIn_) + slotBase_ + (src * myRankChuckSize) * sizeof(T);
                        uint64_t outputOffset
                            = reinterpret_cast<uint64_t>(myGmIn_) + slotBase_ + (dst * myRankChuckSize) * sizeof(T);
                        CpGM2GM((__gm__ T*)outputOffset, (__gm__ T*)inputOffset, myRankChuckSize, reduceOp_);
                        pipe_barrier(PIPE_ALL);
                    }
                    curBlocks = M;
                    ++round;
                }
            }
            Record(rank_, ipc_reduce_flag_offset + 1, curTag_);
        }
    }

    __aicore__ inline void SmallCoreAllgather()
    {
        uint64_t dataCount = len_;
        // SK空闲核跳过gather，避免与忙核重复搬运
        if (IsIdleCore()) {
            return;
        }
        for (uint32_t i = 0; blockIdx_ + i * numBlocks_ < rankSize_; i++) {
            targetRank = blockIdx_ + i * numBlocks_;
            rankChunkSize
                = ((targetRank + 1) * rankChunkStride <= dataCount) ?
                      rankChunkStride :
                      (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));

            if (rankChunkSize <= 0) {
                return;
            }

            // waitflag
            WaitFlag(targetRank, ipc_reduce_flag_offset + 1, curTag_);

            // gather
            uint64_t inputOffset = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + slotBase_;
            uint64_t outputOffset = output_ + (targetRank * rankChunkStride) * sizeof(T);
            CpGM2GM((__gm__ T*)outputOffset, (__gm__ T*)inputOffset, rankChunkSize);
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
    uint64_t slotBase_{0}; // 镜像区/槽区分界(= align32(len_*sizeof(T)))，各卡一致
};

template <typename T>
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

template <typename T>
__aicore__ inline void AivAllReduceV2Mesh1DTwoShotSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    // 参数中的input和output是分别加了步长inBuffBaseOff和outBuffBaseOff步长
    AivAllReduceMesh1DTwoShot<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT);

    uint64_t maxCountPerLoop;
    if (op.numBlocks_ >= op.rankSize_ * 2) {
        uint64_t consumerNum = (op.numBlocks_ - op.rankSize_) / op.rankSize_ * op.rankSize_;
        uint64_t baseCount = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / 2 / sizeof(T);
        maxCountPerLoop = (baseCount > consumerNum) ? (baseCount - consumerNum) : 0;
    } else {
        // 少核路径 scratch = 镜像区 + 槽区，两段各约 curCount*sizeof(T)，需 curCount <= base/2/sizeof(T)。
        // 原式在 rankSize_>=2 时更紧(已覆盖)，取 min 后对 rankSize_>=2 行为不变，只堵住 rankSize_==1。
        uint64_t base = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
        uint64_t bySlotLayout = base / op.rankSize_ / sizeof(T);
        uint64_t byPullRegions = base / DOUBLE / sizeof(T);
        maxCountPerLoop = (bySlotLayout < byPullRegions) ? bySlotLayout : byPullRegions;
    }
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
    AivAllReduceV2Mesh1DTwoShotSuperKernel<int8_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT16
    AivAllReduceV2Mesh1DTwoShotSuperKernel<int16_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_INT32
    AivAllReduceV2Mesh1DTwoShotSuperKernel<int32_t>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP16
    AivAllReduceV2Mesh1DTwoShotSuperKernel<half>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_FP32
    AivAllReduceV2Mesh1DTwoShotSuperKernel<float>(SUPERKERNEL_ARGS_CALL);
#elif defined HCCL_DTYPE_BFP16
    AivAllReduceV2Mesh1DTwoShotSuperKernel<bfloat16_t>(SUPERKERNEL_ARGS_CALL);
#else
#endif
}
