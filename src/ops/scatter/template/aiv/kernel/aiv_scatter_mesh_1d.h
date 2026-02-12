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
class AivScatterMesh1D : public AivCommBase {
    constexpr static uint64_t CORE_NUMS_PER_STAGE = 16;  // 每个阶段提供的最大核数
    constexpr static uint64_t STAGE_NUM = 2;  // 生产者 消费者
    constexpr static uint64_t TAG_FLAG_SIZE = 8;
    constexpr static uint64_t coreNumPerRank = 1;
 
public:
 
    __aicore__ inline AivScatterMesh1D() {
    }
 
    __aicore__ inline void InitCoreInfo(uint64_t len, uint64_t stride)
    {
        coreNumPerStage = coreNumPerRank * rankSize_;
        if(rank_ == root_){
            if(block_idx < coreNumPerStage){
                targetRank = block_idx / coreNumPerRank;
                uint64_t outerOffset = targetRank  * stride;
                uint64_t innerOffset = 0;
                inputOffset = input_ + innerOffset + outerOffset;
                outputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + innerOffset;
            } else if(block_idx < coreNumPerStage + coreNumPerRank){
                uint64_t innerOffset = 0;
                uint64_t outerOffset = 0;
                inputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + outerOffset + innerOffset;
                outputOffset = output_ + innerOffset;
            }
        } else {
            if (block_idx < coreNumPerRank){
                uint64_t innerOffset = 0;
                outputOffset = output_ + innerOffset;
            }
        }
    }
 
    __aicore__ inline void Producer()
    {
        CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, len_);
        pipe_barrier(PIPE_ALL);
        uint64_t flag_offset =  0;
        Record(targetRank, flag_offset, curTag);
    }
 
    __aicore__ inline void Consumer()
    {
        uint64_t flag_offset;
        if(rank_ == root_){
            flag_offset = 0;
        }else{
            flag_offset = 0;
        }
        WaitFlag(rank_, flag_offset, curTag);
        CpGM2GM((__gm__ T *)output_, (__gm__ T *)GM_IN[rank_], len_);
    }
 
    __aicore__ inline void FlagClear()
    {
        uint64_t flag_offset = 0;
        Record(rank_, flag_offset, 0);
    }
 
    __aicore__ inline void Process(uint64_t curCount, uint32_t curTag, uint64_t stride)
    {
        this->curTag = static_cast<int32_t>(curTag);
        this->curCount = curCount / coreNumPerRank;
        if(rank_ == root_){
            inputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputOffset));
            outputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outputOffset));
            if(block_idx < coreNumPerStage){
                Producer();
                WaitFlag(rank_, BARRIER_OFFSET / FLAG_SIZE, curTag);
            } else if(block_idx < coreNumPerStage + coreNumPerRank){
                Consumer();
                Record(rank_, BARRIER_OFFSET / FLAG_SIZE, curTag);
            }
        } else {
            outputGT.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outputOffset));
            if (block_idx < coreNumPerRank){
                Consumer();
            }
        }
    }
 
    uint32_t coreNumPerStage;
    uint32_t targetRank;
    GM_ADDR peerMemThisCore;
    uint64_t inputOffset;
    uint64_t outputOffset;
    int32_t curTag;
    uint64_t curCount;
    uint64_t dataBufferSize;
    GlobalTensor<T> inputGT;
    GlobalTensor<T> outputGT;
};
 
template<typename T>
__aicore__ inline void AivScatterV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivScatterMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.InitCoreInfo(len, inputSliceStride);
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    op.Process(len, tag, inputSliceStride);
    op.BarrierAll();
}