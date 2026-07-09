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
class AivReduceScatterMesh1DBigData : public AivCommBase {
    constexpr static uint64_t stageNum = 2;  // 生产者 消费者
    constexpr static uint64_t TAG_FLAG_SIZE = 8;

public:

    __aicore__ inline AivReduceScatterMesh1DBigData() {
    }

    // 应该第一步要先用rankSize个核去切数据，一种方案是cutNum是多少就切多少份，另一个方案是切细一点，去多读几次，用足够多的流水去掩盖第一步和第三步的时间
    // 第二部是并行去读对端的数据
    // 第三步，之前把数据搬回来的那个卡，搬回来之后就没事干了，顺路就去把reduce做了

    __aicore__ inline void Producer(uint64_t len, uint64_t inputStride)
    {
        // 每个核负责len数据，一块数据切成cutNum块
        uint64_t targetRank = blockIdx_;

        uint64_t dataPerCore = len / cutNum;
        uint64_t remainder = len % cutNum;

        for (int64_t coreIndex = 0; coreIndex < cutNum; coreIndex++) {
            // 数据对不齐的情况
            uint64_t innerDispls = 0;
            uint64_t sendCurCount = 0;
            if (coreIndex < remainder) { // 这部分核需要多处理一个数据
                innerDispls = coreIndex * dataPerCore + coreIndex;
                sendCurCount = dataPerCore + 1;
            } else {
                innerDispls = coreIndex * dataPerCore + remainder;
                sendCurCount = dataPerCore;
            }

            if (sendCurCount > 0) {
                uint64_t usrInOffset = input_ + targetRank * inputStride + innerDispls * sizeof(T);
                uint64_t cclInOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (targetRank * len + innerDispls) * sizeof(T);
                CpGM2GM((__gm__ T *)cclInOffset, (__gm__ T *)usrInOffset, sendCurCount);
                PipeBarrier<PIPE_ALL>();
                // 每个核写flag
                Record(rank_, blockIdx_ * cutNum + coreIndex, curTag);
            }
        }
    }

    __aicore__ inline void Consumer(uint64_t len)
    {
        // stage2的核分成rankSize份，每一份有cutNum，并发去读其他所有卡的数据
        uint64_t blockInedx = blockIdx_ - coreNumStage1;
        uint32_t coreNumPerRank = coreNumStage2 / rankSize_;
        uint32_t targetRank = blockInedx / coreNumPerRank;
        uint32_t coreIndex = (blockInedx - (targetRank * coreNumPerRank))  % coreNumPerRank;
        // 每个核分数据
        uint64_t dataPerCore = len / coreNumPerRank;
        uint64_t remainder = len % coreNumPerRank;
        // 数据对不齐的情况
        uint64_t innerDispls = 0;
        uint64_t recvCurCount = 0;
        if (coreIndex < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex * dataPerCore + coreIndex;
            recvCurCount = dataPerCore + 1;
        } else {
            innerDispls = coreIndex * dataPerCore + remainder;
            recvCurCount = dataPerCore;
        }

        if (recvCurCount > 0) {
            // 先去wait flag，然后把数据拷贝过来
            WaitFlag(targetRank, rank_ * coreNumPerRank + coreIndex, curTag);

            uint64_t srcCclInOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (rank_ * len + innerDispls ) * sizeof(T);
            uint64_t dstCclOutOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + cclBufferStride + (targetRank * len + innerDispls) * sizeof(T);
            if (targetRank != 0) { // targetRank == 0 的那张卡，一会直接到output就可以了
                CpGM2GM((__gm__ T *)dstCclOutOffset, (__gm__ T *)srcCclInOffset, recvCurCount);
                PipeBarrier<PIPE_ALL>();
            }
            // 要写一个flag，后面做reduce需要用到
            Record(rank_, rankSize_ * coreNumPerRank + blockInedx, curTag);

            // 把所有数据放到cclBuffer之后，开始本地做reduce
            // 前面做过数据搬运的core现在是空闲状态，顺路帮忙做一下reduce
            if (targetRank == 0) {
                uint64_t usrOutOffset = output_ + innerDispls * sizeof(T);
                for (int index = 0; index < rankSize_; index++) {
                    // 按照顺序把数据reduce 到 output
                    WaitFlag(rank_, rankSize_ * coreNumPerRank + index * coreNumPerRank + coreIndex, curTag);
                    uint64_t cclOutOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + cclBufferStride + (index * len + innerDispls) * sizeof(T);
                    if (index == 0) { // 通过直接覆盖output把数据清一下
                        CpGM2GM((__gm__ T *)usrOutOffset, (__gm__ T *)srcCclInOffset, recvCurCount);
                    } else { // 其他卡往output上做atomic add
                        CpGM2GM((__gm__ T *)usrOutOffset, (__gm__ T *)cclOutOffset, recvCurCount, reduceOp_);
                    }
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

    __aicore__ inline void Process(uint32_t sliceId, uint64_t len, uint64_t inputStride)
    {
        // 核数大于等于2倍ranksize
        uint64_t rankMultiple = 2;
        curStageCoreNum = numBlocks_ / rankSize_ * rankSize_; // 总的核数
        if (curStageCoreNum < rankMultiple * rankSize_) {
            return;
        }
        coreNumStage1 = rankSize_;
        coreNumStage2 = curStageCoreNum - coreNumStage1;
        cutNum = coreNumStage2 / rankSize_;
        curTag = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        cclBufferStride = len * rankSize_ * sizeof(T); // cclIn和cclOut的分界
        if (blockIdx_ < coreNumStage1) {
            Producer(len, inputStride);
        } else if (blockIdx_ < curStageCoreNum) {
            Consumer(len);
        }
    }

    uint64_t curStageCoreNum;
    uint64_t coreNumStage1;
    uint64_t coreNumStage2;
    uint64_t cutNum;
    uint64_t curTag;
    uint64_t cclBufferStride;
};

template<typename T>
__aicore__ inline void AivReduceScatterV2Mesh1DBigData(KERNEL_ARGS_DEF)
{
    AivReduceScatterMesh1DBigData<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    op.Process(sliceId, len, inputSliceStride);
    op.BarrierAll();
}

template<typename T>
__aicore__ inline void AivReduceScatterV2Mesh1DBigDataSuperKernel(SUPERKERNEL_ARGS_DEF)
{
    AivReduceScatterMesh1DBigData<T> op;
    op.Init(SUPERKERNEL_CLASS_INIT);

    uint64_t maxCountPerLoop = op.cclBufferSize_ / (2 * op.rankSize_) / UB_ALIGN_SIZE * UB_ALIGN_SIZE / sizeof(T);
    uint64_t countLeft = op.len_;
    int32_t loopTag = op.tag_;

    while (countLeft > 0) {
        uint64_t curCount = (countLeft > maxCountPerLoop) ? maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * sizeof(T);

        op.Process(loopTag, curCount, op.inputSliceStride_);
        op.BarrierAll();

        countLeft -= curCount;
        op.input_ += curSize;
        op.output_ += curSize;
        loopTag += curSize / UB_DB_DATA_BATCH_SIZE + 1;
    }
}
