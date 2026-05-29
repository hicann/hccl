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

class AivAllReduceMesh1DOneShot : public AivCommBase {
    constexpr static uint64_t CORE_NUMS_PER_STAGE = 16;  // 每个阶段提供的最大核数
    constexpr static uint64_t STAGE_NUM = 2;  // 生产者 消费者
    constexpr static uint64_t TAG_FLAG_SIZE = 8;
    constexpr static uint64_t coreNumPerRank = 1;
 
public:
 
    __aicore__ inline AivAllReduceMesh1DOneShot() {
    }
 
    __aicore__ inline void Producer()
    {
        CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)input_, len_);
        pipe_barrier(PIPE_ALL);
        Record(targetRank, rank_, curTag_);
    }
 
    __aicore__ inline void Consumer()
    {
        uint32_t waitRank = 0;
        uint64_t outerOffset = waitRank  * len_ * sizeof(T); //rank_  * len;
        inputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + outerOffset;
        WaitFlag(rank_, waitRank, curTag_);
        CpGM2GM((__gm__ T *)output_, (__gm__ T *)inputOffset, len_);
        pipe_barrier(PIPE_ALL);
        for (waitRank = 1; waitRank < rankSize_; waitRank++) {
            outerOffset = waitRank  * len_ * sizeof(T); //rank_  * len;
            inputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + outerOffset;
            WaitFlag(rank_, waitRank, curTag_);
            CpGM2GM((__gm__ T *)output_, (__gm__ T *)inputOffset, len_, reduceOp_);
            pipe_barrier(PIPE_ALL);
        }
    }
 
    __aicore__ inline void FlagClear()
    {
        uint64_t flag_offset = 0; //(block_idx % coreNumPerRank) * TAG_FLAG_SIZE;
        Record(rank_, flag_offset, 0);
    }
 
    //aiv core数目大于ranksize+1
    __aicore__ inline void ProcessCoreLargeCase(uint64_t curCount, uint32_t sliceId, uint64_t stride)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        this->curCount = curCount / coreNumPerRank;
        coreNumPerStage = coreNumPerRank * rankSize_;
 
        if(GetBlockIdx() < coreNumPerStage){
            targetRank = GetBlockIdx();
            uint64_t outerOffset = rank_  * this->curCount * sizeof(T);
            outputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + outerOffset;
            Producer();
            SyncAll<true>();
        } else if(GetBlockIdx() < coreNumPerStage + coreNumPerRank){
            SyncAll<true>();
            Consumer();
        } else {
            SyncAll<true>();
        }
    }
 
    //aiv core数目小于ranksize
    __aicore__ inline void ProcessCoreSmallCase(uint64_t curCount, uint32_t sliceId, uint64_t stride)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        this->curCount = curCount;
 
        for(uint32_t i=0;block_idx+i*numBlocks_<rankSize_;i++){
            targetRank = block_idx+i*numBlocks_;
            uint64_t outerOffset = rank_  * this->curCount * sizeof(T);
            outputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + outerOffset;
            Producer();
        }
        SyncAll<true>();
        if(block_idx==numBlocks_-1){
          Consumer();
        }
    }
 
    uint32_t coreNumPerStage;
    uint32_t targetRank;
    GM_ADDR peerMemThisCore;
    uint64_t inputOffset;
    uint64_t outputOffset;
    uint64_t curCount;
    uint64_t dataBufferSize;
};
 
template<typename T>
__aicore__ inline void AivAllReduceV2Mesh1DOneShot(KERNEL_ARGS_DEF)
{
    AivAllReduceMesh1DOneShot<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    if(rankSize+1<=block_num){
      op.ProcessCoreLargeCase(len, sliceId, inputSliceStride);
    }else{
      op.ProcessCoreSmallCase(len, sliceId, inputSliceStride);
    }
    op.BarrierAll();
}

template<typename T>
__aicore__ inline void AivAllReduceV2Mesh1DOneShotSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    AivAllReduceMesh1DOneShot<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT);

    uint64_t maxCountPerLoop = op.cclBufferSize_ / UB_ALIGN_SIZE * UB_ALIGN_SIZE / op.rankSize_ / sizeof(T);
    uint64_t countLeft = op.len_;

    int32_t loopTag = op.tag_;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        op.ProcessCoreLargeCase(curCount, loopTag, 1);
        op.BarrierAll();

        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}

__aicore__ inline void sk_ar_mesh_1d_oneshot(SUPERKERNEL_ARGS_DEF)
{
    #ifdef HCCL_DTYPE_INT8
        AivAllReduceV2Mesh1DOneShotSuperKernel<int8_t> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_INT16
        AivAllReduceV2Mesh1DOneShotSuperKernel<int16_t> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_INT32
        AivAllReduceV2Mesh1DOneShotSuperKernel<int32_t> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_FP16
        AivAllReduceV2Mesh1DOneShotSuperKernel<half> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_FP32
        AivAllReduceV2Mesh1DOneShotSuperKernel<float> (SUPERKERNEL_ARGS_CALL);
    #elif defined HCCL_DTYPE_BFP16
        AivAllReduceV2Mesh1DOneShotSuperKernel<bfloat16_t> (SUPERKERNEL_ARGS_CALL);
    #else
    #endif
}