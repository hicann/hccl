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
    constexpr static uint64_t TAG_FLAG_SIZE = 4;
 
public:
    __aicore__ inline AivAllReduceMesh1DTwoShot()
    {
    }
 
    uint32_t coreNumPerRank;
    uint32_t coreNumFirstStage;
    uint32_t coreNumTotal;
    uint32_t inner_id;
    uint32_t targetRank;
    uint64_t rankChunkSize;
    uint64_t innerChunkSize;
    uint64_t rankChunkStride;
    uint64_t innerChunkStride;
    int32_t  curTag;
 
    uint64_t ipc_reduce_flag_offset{1024};
 
/*
 @Desc:初始化aiv core 角色，数据chuck划分
 @param dataCount：输入数据大小
 @param stepTag： 迭代轮次标记
*/
__aicore__ inline void InitCoreInfo(uint32_t stepTag)
{
    curTag             = static_cast<int32_t>(stepTag);
    uint64_t dataCount = len_;
    // aiv core 划分
    // 1D rankSize_ <=16； coreNumPerRank*(ranksize_+1)一定比numBlocks_小，否则就需要一个aicore处理多个rank的数据
    coreNumPerRank    = numBlocks_ / (rankSize_ + 1);
    coreNumFirstStage = coreNumPerRank * rankSize_;
    coreNumTotal      = coreNumPerRank * (rankSize_ + 1);
 
    // 均分
    innerChunkStride = RoundUPAiv(dataCount, coreNumFirstStage);
    rankChunkStride  = innerChunkStride * coreNumPerRank;
    if (GetBlockIdx() < coreNumFirstStage) {
        // 负责对应rank数据在scatter和gather阶段的搬移
        targetRank = GetBlockIdx() / coreNumPerRank;
        rankChunkSize
            = ((targetRank + 1) * rankChunkStride <= dataCount)
                  ? rankChunkStride
                  : (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));
    } else if (GetBlockIdx() < coreNumTotal) {
        // 负责当前rank对应的data chuck reduce
        targetRank    = rank_;
        rankChunkSize = ((rank_ + 1) * rankChunkStride <= dataCount)
                            ? rankChunkStride
                            : (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
    }
 
    if (GetBlockIdx() < coreNumTotal) {
        inner_id = GetBlockIdx() % coreNumPerRank;
        innerChunkSize
            = ((inner_id + 1) * innerChunkStride <= rankChunkSize)
                  ? innerChunkStride
                  : (rankChunkSize <= inner_id * innerChunkStride ? 0 : (rankChunkSize - inner_id * innerChunkStride));
    }
    // 跳过前面的scatter标记位置
    ipc_reduce_flag_offset = coreNumFirstStage;
}
 
/*
  @Desc: allreduce two shot的ReduceScatter阶段，先scatter再reduce
*/
__aicore__ inline void ReduceScatter()
{
    // Scatter
    if (GetBlockIdx() < coreNumFirstStage) {
        if (innerChunkSize > 0) {
            uint64_t inputOffset  = input_ + (targetRank * rankChunkStride + inner_id * innerChunkStride) * sizeof(T);
            uint64_t outputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank])
                                    + (rank_ * rankChunkSize + inner_id * innerChunkStride) * sizeof(T);
            CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, innerChunkSize);
            pipe_barrier(PIPE_ALL);
        }
        // set flag
        Record(targetRank, rank_ * coreNumPerRank + inner_id, curTag);
    } else if (GetBlockIdx() < coreNumTotal) {
        if (innerChunkSize > 0) {
            // 等待当前slice的数据都已送到
            for (uint32_t i = 0; i < rankSize_; i++) {
                // check flag
                WaitFlag(rank_, i * coreNumPerRank + inner_id, curTag);
                if (i == 0) { // 需要等待第0个rank的scatter数据到达，但是不能自身重复reduce
                    continue;
                }
                // reduce
                uint64_t inputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_])
                                       + (i * rankChunkSize + inner_id * innerChunkStride) * sizeof(T);
                uint64_t outputOffset
                    = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (inner_id * innerChunkStride) * sizeof(T);
                CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, innerChunkSize, reduceOp_);
                pipe_barrier(PIPE_ALL);
            }
        }
        // set flag
        Record(rank_, ipc_reduce_flag_offset + inner_id, curTag);
    }
}
 
/*
  @Desc: allreduce two shot中的allgather阶段
*/
__aicore__ inline void AllGather()
{
    if (GetBlockIdx() < coreNumFirstStage) {
        if (innerChunkSize <= 0) {
            return;
        }
        // waitflag
        WaitFlag(targetRank, ipc_reduce_flag_offset + inner_id, curTag);
 
        // gather
        uint64_t inputOffset
            = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (inner_id * innerChunkStride) * sizeof(T);
        uint64_t outputOffset = output_ + (targetRank * rankChunkStride + inner_id * innerChunkStride) * sizeof(T);
        CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, innerChunkSize);
        pipe_barrier(PIPE_ALL);
    }
}
 
/*
  @desc: slice切分函数，尾部取0
*/
__aicore__ inline uint64_t RoundUPAiv(uint64_t dividend, uint64_t divisor)
{
    return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
}
 
/*
  @desc: 适配支持aiv 数目小于ranksize的情况
*/
__aicore__ inline void SmallCoreReduceScatter(uint32_t stepTag)
{
    uint64_t dataCount     = len_;
    rankChunkStride        = RoundUPAiv(dataCount, rankSize_);
    ipc_reduce_flag_offset = rankSize_;
    curTag                 = static_cast<int32_t>(stepTag);
 
    // scatter
    for (uint32_t i = 0; block_idx + i * numBlocks_ < rankSize_; i++) {
        targetRank = block_idx + i * numBlocks_;
        rankChunkSize
            = ((targetRank + 1) * rankChunkStride <= dataCount)
                  ? rankChunkStride
                  : (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));
 
        if (rankChunkSize > 0) {
            uint64_t inputOffset  = input_ + (targetRank * rankChunkStride) * sizeof(T);
            uint64_t outputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (rank_ * rankChunkSize) * sizeof(T);
            CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, rankChunkSize);
            pipe_barrier(PIPE_ALL);
        }
        // set flag
        Record(targetRank, rank_, curTag);
    }
    // reduce
    if (block_idx == numBlocks_ - 1) {
        uint64_t myRankChuckSize
            = ((rank_ + 1) * rankChunkStride <= dataCount)
                  ? rankChunkStride
                  : (dataCount <= rank_ * rankChunkStride ? 0 : (dataCount - rank_ * rankChunkStride));
        ;
        if (myRankChuckSize > 0) {
            for (uint32_t i = 0; i < rankSize_; i++) {
                WaitFlag(rank_, i, curTag);
                if (i == 0) { // 需要等待第0个rank的scatter数据到达，但是不能自身重复reduce
                    continue;
                }
                // reduce
                uint64_t inputOffset  = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (i * myRankChuckSize) * sizeof(T);
                uint64_t outputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]);
                CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, myRankChuckSize, reduceOp_);
                pipe_barrier(PIPE_ALL);
            }
        }
        Record(rank_, ipc_reduce_flag_offset + 1, curTag);
    }
}
 
/*
  @desc: 适配支持aiv 数目小于ranksize的情况
*/
__aicore__ inline void SmallCoreAllgather()
{
    uint64_t dataCount = len_;
    for (uint32_t i = 0; block_idx + i * numBlocks_ < rankSize_; i++) {
        targetRank = block_idx + i * numBlocks_;
        rankChunkSize
            = ((targetRank + 1) * rankChunkStride <= dataCount)
                  ? rankChunkStride
                  : (dataCount <= targetRank * rankChunkStride ? 0 : (dataCount - targetRank * rankChunkStride));
 
        if (rankChunkSize <= 0) {
            return;
        }
 
        // waitflag
        WaitFlag(targetRank, ipc_reduce_flag_offset + 1, curTag);
 
        // gather
        uint64_t inputOffset  = reinterpret_cast<uint64_t>(GM_IN[targetRank]);
        uint64_t outputOffset = output_ + (targetRank * rankChunkStride) * sizeof(T);
        CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, rankChunkSize);
        pipe_barrier(PIPE_ALL);
    }
}
 
};
 
/*
 @Desc:AivAllReduce91095Mesh1DTwoShot是allreduce aiv算子的入口
 @param EXTERN_KERNEL_ARGS_DEF_V2: 宏变量定义了所有算子需要的参数信息包括IPC buffer, input & output 位置，rank信息等
*/
 
template<typename T>
__aicore__ inline void AivAllReduceV2Mesh1DTwoShot(EXTERN_KERNEL_ARGS_DEF_V2)
{
    // 参数中的input和output是分别加了步长inBuffBaseOff和outBuffBaseOff步长
    AivAllReduceMesh1DTwoShot<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    if (rankSize + 1 <= block_num) {
        op.InitCoreInfo(tag);
        op.ReduceScatter();
        op.AllGather();
    } else {
        op.SmallCoreReduceScatter(tag);
        op.SmallCoreAllgather();
    }
    op.BarrierAll();
}