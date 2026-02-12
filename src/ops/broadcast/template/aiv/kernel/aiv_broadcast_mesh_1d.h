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
 
// todo 简化参数
 
class AivBroadcastMesh1D : public AivCommBase {
    constexpr static uint64_t CORE_NUMS_ALL = 2;
 
public:
    __aicore__ inline AivBroadcastMesh1D() {}
 
    template<typename T>
    __aicore__ inline void Process(uint64_t curCount, uint64_t curTag, uint64_t stride);
};
 
template<typename T>
__aicore__ inline void AivBroadcastMesh1D::Process(uint64_t curCount, uint64_t curTag, uint64_t stride)
{
    uint64_t dataTypeSize = sizeof(T);
    uint64_t curStageCoreNum = CORE_NUMS_ALL / rankSize_ * rankSize_;
    if (block_idx >= curStageCoreNum) {
        return;
    }
    uint32_t peerRank = block_idx / (curStageCoreNum / rankSize_);
    uint64_t offsetPerCore = curCount / curStageCoreNum * dataTypeSize;
    uint64_t dataOffset = offsetPerCore * block_idx;
    uint64_t countPerCore = block_idx == curStageCoreNum - 1 ? curCount - (curStageCoreNum - 1) * (curCount / curStageCoreNum)
                                    : curCount / curStageCoreNum;
    uint64_t flag_offset = block_idx;
    __gm__ T *inputGM = (__gm__ T *)(input_ + dataOffset);
    __gm__ T *cclGM = (__gm__ T *)(GM_IN[peerRank] + dataOffset);
    // scatter
    if (rank_ == root_) {
        CpGM2GM(cclGM, inputGM, countPerCore);
        PipeBarrier<PIPE_ALL>();
        Record(peerRank, flag_offset, curTag);
    }
 
    // allgather
    WaitFlag(peerRank, flag_offset, curTag);
    CpGM2GM(inputGM, cclGM, countPerCore);
    PipeBarrier<PIPE_ALL>();
}
 
template<typename T>
__aicore__ inline void AivBroadcastV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivBroadcastMesh1D op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    op.Process<T>(len, tag, inputSliceStride);
    op.BarrierAll();
}