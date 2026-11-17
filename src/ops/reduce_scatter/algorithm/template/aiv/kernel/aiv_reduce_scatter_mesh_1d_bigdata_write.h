/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_REDUCE_SCATTER_MESH_1D_BIGDATA_WRITE_H
#define AIV_REDUCE_SCATTER_MESH_1D_BIGDATA_WRITE_H

#include "aiv_communication_base_v2.h"

using namespace AscendC;

template <typename T>
class AivReduceScatterMesh1DBigDataWrite : public AivCommBase {
public:
    __aicore__ inline AivReduceScatterMesh1DBigDataWrite() {}

    // 根据可用核数与非空分片数，为每个目的rank选择一个或两个producer，不限制rank数量。
    // 保持stage布局；全部consumer核按独立子区间执行原树形归约。
    // 数据就绪flag占用[0, coreNumStage2)。

    __aicore__ inline void Producer(uint64_t len, uint64_t inputStride)
    {
        uint64_t targetRank = blockIdx_ % rankSize_;
        uint64_t producerIndex = blockIdx_ / rankSize_;
        uint64_t producerNumPerRank = coreNumStage1 / rankSize_;

        uint64_t dataPerCore = len / cutNum;
        uint64_t remainder = len % cutNum;

        for (uint64_t coreIndex = producerIndex; coreIndex < cutNum; coreIndex += producerNumPerRank) {
            // 数据对不齐的情况
            uint64_t sendCurCount = 0;
            uint64_t innerDispls = 0;
            if (coreIndex < remainder) { // 这部分核需要多处理一个数据
                innerDispls = coreIndex * dataPerCore + coreIndex;
                sendCurCount = dataPerCore + 1;
            } else {
                innerDispls = coreIndex * dataPerCore + remainder;
                sendCurCount = dataPerCore;
            }

            if (sendCurCount > 0) {
                uint64_t usrInOffset = input_ + targetRank * inputStride + innerDispls * sizeof(T);
                uint64_t cclInOffset = reinterpret_cast<uint64_t>(GetGmIn(targetRank)) + cclBufferStride
                                       + (rank_ * len + innerDispls) * sizeof(T);
                CpGM2GM((__gm__ T*)cclInOffset, (__gm__ T*)usrInOffset, sendCurCount);
                PipeBarrier<PIPE_ALL>();
                // 每个核写flag
                Record(targetRank, rank_ * cutNum + coreIndex, curTag);
            }
        }
    }

    __aicore__ inline void Consumer(uint64_t len)
    {
        // 每个producer分片再交给rankSize_个consumer核，使用全部stage2核。
        // 子区间不跨producer分片，读取每个源rank时只需等待该片的一个就绪flag。
        uint64_t coreIndex = blockIdx_ - coreNumStage1;
        uint64_t pieceIndex = coreIndex / rankSize_;
        uint64_t coreInPiece = coreIndex % rankSize_;
        uint64_t dataPerPiece = len / cutNum;
        uint64_t pieceRemainder = len % cutNum;
        uint64_t pieceDispls = pieceIndex * dataPerPiece + (pieceIndex < pieceRemainder ? pieceIndex : pieceRemainder);
        uint64_t pieceCount = dataPerPiece + (pieceIndex < pieceRemainder ? 1 : 0);
        uint64_t dataPerCore = pieceCount / rankSize_;
        uint64_t remainder = pieceCount % rankSize_;
        uint64_t innerDispls
            = pieceDispls + coreInPiece * dataPerCore + (coreInPiece < remainder ? coreInPiece : remainder);
        uint64_t recvCurCount = dataPerCore + (coreInPiece < remainder ? 1 : 0);
        if (recvCurCount == 0) {
            return;
        }

        // rank0槽作为初值；其余源槽沿用读kernel的树形折叠顺序。
        WaitFlag(rank_, pieceIndex, curTag);
        uint64_t usrOutOffset = output_ + innerDispls * sizeof(T);
        uint64_t rankZeroOffset = reinterpret_cast<uint64_t>(myGmIn_) + cclBufferStride + innerDispls * sizeof(T);
        CpGM2GM((__gm__ T*)usrOutOffset, (__gm__ T*)rankZeroOffset, recvCurCount);
        PipeBarrier<PIPE_ALL>();
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
                    WaitFlag(rank_, src * cutNum + pieceIndex, curTag);
                }
                if (round == 0 || (round == 1 && dst >= firstRoundDstNum)) {
                    WaitFlag(rank_, dst * cutNum + pieceIndex, curTag);
                }
                uint64_t srcOffset
                    = reinterpret_cast<uint64_t>(myGmIn_) + cclBufferStride + (src * len + innerDispls) * sizeof(T);
                uint64_t dstOffset = (dst == 0) ? usrOutOffset :
                                                  reinterpret_cast<uint64_t>(myGmIn_) + cclBufferStride
                                                      + (dst * len + innerDispls) * sizeof(T);
                CpGM2GM((__gm__ T*)dstOffset, (__gm__ T*)srcOffset, recvCurCount, reduceOp_);
                PipeBarrier<PIPE_ALL>();
            }
            curBlocks = M;
            ++round;
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
        uint64_t candidateProducerNum = MAX_PRODUCER_NUM_PER_RANK * rankSize_;
        // 至少保留一组consumer；非空分片足以分给所有producer时才增加发送核。
        if (curStageCoreNum >= candidateProducerNum + rankSize_) {
            uint64_t candidateCutNum = (curStageCoreNum - candidateProducerNum) / rankSize_;
            uint64_t nonEmptyPieceNum = len < candidateCutNum ? len : candidateCutNum;
            if (nonEmptyPieceNum >= MAX_PRODUCER_NUM_PER_RANK) {
                coreNumStage1 = candidateProducerNum;
            }
        }
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

private:
    static constexpr uint64_t MAX_PRODUCER_NUM_PER_RANK = 2;
};

template <typename T>
__aicore__ inline void AivReduceScatterV2Mesh1DBigDataWrite(KERNEL_ARGS_DEF)
{
    AivReduceScatterMesh1DBigDataWrite<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    op.Process(sliceId, len, inputSliceStride);
    op.BarrierAll();
}

#endif // AIV_REDUCE_SCATTER_MESH_1D_BIGDATA_WRITE_H
