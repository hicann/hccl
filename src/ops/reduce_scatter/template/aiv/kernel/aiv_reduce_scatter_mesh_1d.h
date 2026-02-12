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
class AivReduceScatterMesh1D : public AivCommBase {
    constexpr static uint64_t stageNum = 2;  // 生产者 消费者
    constexpr static uint64_t TAG_FLAG_SIZE = 8;
    constexpr static uint64_t maxRankSize = 8;

public:

    __aicore__ inline AivReduceScatterMesh1D() {
    }

    __aicore__ inline void InitCoreInfo(uint64_t len, uint64_t inputStride)
    {
        coreNumPerStage = rankSize_;  // 每个阶段提供的最大核数
        uint64_t processNum = len / rankSize_;
        if(GetBlockIdx() < coreNumPerStage) { // input->ipc,一个block负责一个rank input
            targetRank = GetBlockIdx();
            int64_t outerOffset = targetRank  * inputStride; // inputStride是整个算子的输入size
            int64_t ipcRankOffset = targetRank * len * sizeof(T);
            inputOffset = input_ + outerOffset; // 这里的input是已经偏移过前面处理完数据量的地址了
            outputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + ipcRankOffset;
        } else if (GetBlockIdx() < (coreNumPerStage * stageNum)) { // ipc->output,一个block负责一个卡的数据
            int64_t outerOffset = (GetBlockIdx() % coreNumPerStage) * processNum * sizeof(T);
            outputOffset = output_ + outerOffset;
            int64_t consumInOffset;
            for (int index = 0; index < rankSize_; index++) { // 轮询每个rank的数据，拉取过来，做顺序累加
                consumInOffset = reinterpret_cast<uint64_t>(GM_IN[(index + GetBlockIdx()) % rankSize_]) +
                                 len * rank_ * sizeof(T) + outerOffset; // 本卡的数据都在ipc
                inputOffVec[index] = consumInOffset;
            }
            consumProcessNum = processNum;
            if ((GetBlockIdx() % coreNumPerStage) == (rankSize_ - 1)) {
                consumProcessNum = len - processNum * (rankSize_ - 1);
            }
        }
    }

    __aicore__ inline void Producer()
    {
        CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffset, len_); // 本卡数据拷贝到ipc上
        pipe_barrier(PIPE_ALL);
        uint64_t flag_offset = rank_;
        Record(targetRank, flag_offset, curTag);
    }

    __aicore__ inline void Consumer()
    {
        uint64_t flag_offset;
        for (int index = 0; index < rankSize_; index++) {
            uint32_t rankIdx = (index + GetBlockIdx()) % rankSize_;
            flag_offset = rankIdx;
            WaitFlag(rank_, flag_offset, curTag);
            if (index == 0) { // 通过直接覆盖output把数据清一下
                CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffVec[index], consumProcessNum);
            } else { // 其他卡往output上做atomic add
                CpGM2GM((__gm__ T *)outputOffset, (__gm__ T *)inputOffVec[index], consumProcessNum, reduceOp_);
            }
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void Process(uint32_t curTag)
    {
        this->curTag = static_cast<int32_t>(curTag);
        if(GetBlockIdx() < coreNumPerStage){ // 0-1
            Producer();
        } else if(GetBlockIdx() < (coreNumPerStage * stageNum)) { // 2-3
            Consumer();
        }
    }

private:

    uint32_t targetRank;
    uint64_t inputOffset;
    uint64_t outputOffset;
    int32_t curTag;
    uint64_t consumProcessNum;
    int64_t inputOffVec[maxRankSize];
    uint32_t coreNumPerStage;
};

template<typename T>
__aicore__ inline void AivReduceScatterV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivReduceScatterMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.InitCoreInfo(len, inputSliceStride);
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    op.Process(tag);
    op.BarrierAll();
}
