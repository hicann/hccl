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
class AivAlltoAllVMesh1D : public AivCommBase {
public:
    __aicore__ inline AivAlltoAllVMesh1D() {}

    __aicore__ inline void InitCoreInfo(ExtraArgs& extraArgs, uint64_t processedDataCount, uint64_t currDataCount)
    {
        // 发送数据的编排
        uint64_t sendCountThisLoop = 0;
        uint64_t sendDisplThisLoop = extraArgs.sendDispls[targetRank_] + extraArgs.sendCounts[targetRank_];
        if (extraArgs.sendCounts[targetRank_] > processedDataCount) {
            sendCountThisLoop = min(currDataCount, extraArgs.sendCounts[targetRank_] - processedDataCount);
            sendDisplThisLoop = extraArgs.sendDispls[targetRank_] + processedDataCount;
        }
        uint64_t dataPerCore = sendCountThisLoop / coreNumPerRank_; // 数据量很少的时候，dataPerCore为0
        uint64_t remainder = sendCountThisLoop % coreNumPerRank_;
        // 数据对不齐的情况
        uint64_t innerDispls = 0;
        if (coreIndex_ < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex_ * dataPerCore + coreIndex_;
            sendCurCount_ = dataPerCore + 1;
        } else {
            innerDispls = coreIndex_ * dataPerCore + remainder;
            sendCurCount_ = dataPerCore;
        }
        sendInputOffset_ = input_ + (sendDisplThisLoop + innerDispls) * sizeof(T);
        sendOutputOffset_ = reinterpret_cast<uint64_t>(GM_IN[rank_])
                            + (targetRank_ * cclBufferCountPerRank_ + innerDisplsForCcl_) * sizeof(T);

        // 接收数据的编排
        uint64_t recvCountThisLoop = 0;
        uint64_t recvDisplThisLoop = extraArgs.recvDispls[targetRank_] + extraArgs.recvCounts[targetRank_];
        if (extraArgs.recvCounts[targetRank_] > processedDataCount) {
            recvCountThisLoop = min(currDataCount, extraArgs.recvCounts[targetRank_] - processedDataCount);
            recvDisplThisLoop = extraArgs.recvDispls[targetRank_] + processedDataCount;
        }
        dataPerCore = recvCountThisLoop / coreNumPerRank_;
        remainder = recvCountThisLoop % coreNumPerRank_;
        if (coreIndex_ < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex_ * dataPerCore + coreIndex_;
            recvCurCount_ = dataPerCore + 1;
        } else {
            innerDispls = coreIndex_ * dataPerCore + remainder;
            recvCurCount_ = dataPerCore;
        }
        recvInputOffset_ = reinterpret_cast<uint64_t>(GM_IN[targetRank_])
                           + (rank_ * cclBufferCountPerRank_ + innerDisplsForCcl_) * sizeof(T);
        recvOutputOffset_ = output_ + (recvDisplThisLoop + innerDispls) * sizeof(T);
    }

    __aicore__ inline void Producer(uint64_t loop)
    {
        if (sendCurCount_ == 0) {
            return;
        }
        uint64_t flag_offset = blockIdx_;
        uint32_t tagTemp = loop;
        if (loop == 0) {
            tagTemp = curTag_;
        }
        WaitFlag(rank_, flag_offset, tagTemp);

        CpGM2GM((__gm__ T*)sendOutputOffset_, (__gm__ T*)sendInputOffset_, sendCurCount_);
        PipeBarrier<PIPE_ALL>();

        flag_offset = rank_ * coreNumPerRank_ + coreIndex_ + coreCount_;
        Record(targetRank_, flag_offset, loop);
    }

    __aicore__ inline void Consumer(uint64_t loop)
    {
        if (recvCurCount_ == 0) {
            return;
        }
        uint64_t flag_offset = blockIdx_ + coreCount_;
        WaitFlag(rank_, flag_offset, loop);

        CpGM2GM((__gm__ T*)recvOutputOffset_, (__gm__ T*)recvInputOffset_, recvCurCount_);
        PipeBarrier<PIPE_ALL>(); // 核内自己的同步

        flag_offset = rank_ * coreNumPerRank_ + coreIndex_;
        Record(targetRank_, flag_offset, loop + 1);
    }

    // 控核初始化：一核多 rank 映射 + per-rank 初始握手（参考多核 :117-125 的 2 Record + 1 Wait）
    // 顺序：先初始化对端 C 区(data-ready)为 curTag_ → 等对端初始化本 rank 的 C 区 → 再初始化对端 P 区(ack)。
    // C 区必须先就绪为非0：CONSUME 的 WaitFlag(tag=loop=0) 会被残留0误匹配。
    // P 区由 PRODUCE 的 WaitFlag(tag=curTag_, per-op唯一) 保证，无需单独 Wait。
    // P 区初始化必须在 Wait(C) 之后：Record(对端P) 会释放对端 PRODUCE(把本 rank 的 C 区写0)，
    // 若在 Wait(C) 之前，本 rank 的 C 区被写0，Wait(C,curTag_) 永远等不到 → 死锁。
    __aicore__ inline void InitCtrlCore()
    {
        rankNumPerCore_ = (rankSize_ + numBlocks_ - 1) / numBlocks_; // 一核负责的 rank 数(ceil)
        for (uint32_t idx = 0; idx < rankNumPerCore_; idx++) {
            uint32_t dstRank = blockIdx_ * rankNumPerCore_ + idx;
            if (dstRank >= rankSize_) {
                break;
            }
            Record(dstRank, rank_ + rankSize_, curTag_); // 1. 初始化对端 C 区 [dstRank][rank_+rankSize]
        }
        for (uint32_t idx = 0; idx < rankNumPerCore_; idx++) {
            uint32_t dstRank = blockIdx_ * rankNumPerCore_ + idx;
            if (dstRank >= rankSize_) {
                break;
            }
            WaitFlag(rank_, dstRank + rankSize_, curTag_); // 2. 等对端初始化本 rank 的 C 区为 curTag_
        }
        for (uint32_t idx = 0; idx < rankNumPerCore_; idx++) {
            uint32_t dstRank = blockIdx_ * rankNumPerCore_ + idx;
            if (dstRank >= rankSize_) {
                break;
            }
            Record(dstRank, rank_, curTag_); // 3. 初始化对端 P 区 [dstRank][rank_]，释放对端 PRODUCE
        }
    }

    // 控核每轮：两阶段（先全部 put，再全部 get），flag 用 per-rank 两区
    __aicore__ inline void
    ProduceConsumeCtrlCore(uint64_t loop, ExtraArgs& extraArgs, uint64_t processedDataCount, uint64_t currDataCount)
    {
        // PRODUCE：本核负责的每个 dstRank，input 本轮 chunk -> GM_IN[rank_][dstRank 区]
        for (uint32_t idx = 0; idx < rankNumPerCore_; idx++) {
            uint32_t dstRank = blockIdx_ * rankNumPerCore_ + idx;
            if (dstRank >= rankSize_) {
                break;
            }
            sendCurCount_ = 0;
            if (extraArgs.sendCounts[dstRank] > processedDataCount) {
                sendCurCount_ = min(currDataCount, extraArgs.sendCounts[dstRank] - processedDataCount);
            }
            if (sendCurCount_ == 0) { // 与多核 Producer 一致
                continue;
            }
            uint32_t ackTag = loop;
            if (loop == 0) {
                ackTag = curTag_; // 等对端上一轮 ack
            }
            WaitFlag(rank_, dstRank, ackTag); // P 区 [rank_][dstRank]
            uint64_t sendDisplThisLoop = extraArgs.sendDispls[dstRank] + processedDataCount;
            sendInputOffset_ = input_ + sendDisplThisLoop * sizeof(T);
            sendOutputOffset_ = reinterpret_cast<uint64_t>(GM_IN[rank_]) + dstRank * cclBufferCountPerRank_ * sizeof(T);
            CpGM2GM((__gm__ T*)sendOutputOffset_, (__gm__ T*)sendInputOffset_, sendCurCount_);
            PipeBarrier<PIPE_ALL>();
            Record(dstRank, rank_ + rankSize_, loop); // C 区 [dstRank][rank_+rankSize] = data-ready
        }
        // CONSUME：本核负责的每个 dstRank，GM_IN[dstRank][rank_ 区] -> output
        for (uint32_t idx = 0; idx < rankNumPerCore_; idx++) {
            uint32_t dstRank = blockIdx_ * rankNumPerCore_ + idx;
            if (dstRank >= rankSize_) {
                break;
            }
            recvCurCount_ = 0;
            if (extraArgs.recvCounts[dstRank] > processedDataCount) {
                recvCurCount_ = min(currDataCount, extraArgs.recvCounts[dstRank] - processedDataCount);
            }
            if (recvCurCount_ == 0) { // 与多核 Consumer 一致
                continue;
            }
            WaitFlag(rank_, dstRank + rankSize_, loop); // C 区 [rank_][dstRank+rankSize] = data-ready
            uint64_t recvDisplThisLoop = extraArgs.recvDispls[dstRank] + processedDataCount;
            recvInputOffset_ = reinterpret_cast<uint64_t>(GM_IN[dstRank]) + rank_ * cclBufferCountPerRank_ * sizeof(T);
            recvOutputOffset_ = output_ + recvDisplThisLoop * sizeof(T);
            CpGM2GM((__gm__ T*)recvOutputOffset_, (__gm__ T*)recvInputOffset_, recvCurCount_);
            PipeBarrier<PIPE_ALL>();
            Record(dstRank, rank_, loop + 1); // P 区 [dstRank][rank_] = ack
        }
    }

    __aicore__ inline void CalcLoopInfo(ExtraArgs& extraArgs, uint64_t& maxSendOrRecvDataCount, uint64_t& loopTimes)
    {
        maxSendOrRecvDataCount = 0;
        for (uint64_t i = 0; i < rankSize_; i++) {
            maxSendOrRecvDataCount = max(maxSendOrRecvDataCount, extraArgs.sendCounts[i]);
            maxSendOrRecvDataCount = max(maxSendOrRecvDataCount, extraArgs.recvCounts[i]);
        }
        // 每张卡的loopTimes可能是不一样的
        loopTimes = maxSendOrRecvDataCount / cclBufferCountPerRank_
                    + static_cast<uint64_t>(maxSendOrRecvDataCount % cclBufferCountPerRank_ != 0);
    }

    __aicore__ inline void ProcessCtrlCore(uint64_t len, uint32_t sliceId, ExtraArgs& extraArgs)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        cclBufferCountPerRank_ = len;

        InitCtrlCore();

        uint64_t maxSendOrRecvDataCount = 0;
        uint64_t loopTimes = 0;
        CalcLoopInfo(extraArgs, maxSendOrRecvDataCount, loopTimes);

        uint64_t processedDataCount = 0;
        for (uint64_t loop = 0; loop < loopTimes; loop++) {
            uint64_t currDataCount
                = (loop == loopTimes - 1) ? maxSendOrRecvDataCount - processedDataCount : cclBufferCountPerRank_;
            ProduceConsumeCtrlCore(loop, extraArgs, processedDataCount, currDataCount);
            SyncAll<true>();
            processedDataCount += currDataCount;
        }
    }

    __aicore__ inline void Process(uint64_t len, uint32_t sliceId, ExtraArgs& extraArgs)
    {
        curTag_ = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);
        cclBufferCountPerRank_ = len;

        // 多核一 rank 路径
        coreNumPerRank_ = numBlocks_ / rankSize_;
        coreCount_ = coreNumPerRank_ * rankSize_;
        if (blockIdx_ >= coreCount_) { // 负责每个rank的核数相同，方便读写都能并行
            return;
        }
        targetRank_ = blockIdx_ / coreNumPerRank_; // 每个核负责哪个rank的数据
        coreIndex_
            = (blockIdx_ - (targetRank_ * coreNumPerRank_)) % coreNumPerRank_; // 每个核在当前coreNumPerRank_里面的排序

        uint64_t dataPerCore = cclBufferCountPerRank_ / coreNumPerRank_;
        uint64_t remainder = cclBufferCountPerRank_ % coreNumPerRank_;
        if (coreIndex_ < remainder) { // 这部分核需要多处理一个数据
            innerDisplsForCcl_ = coreIndex_ * dataPerCore + coreIndex_;
        } else {
            innerDisplsForCcl_ = coreIndex_ * dataPerCore + remainder;
        }

        // 前面 coreCount_ 个位置给 Producer，后面 coreCount_ 个位置给 Consumer
        // 初始化的时候，先给对端一个flag
        uint64_t flag_offset = rank_ * coreNumPerRank_ + coreIndex_ + coreCount_;
        Record(targetRank_, flag_offset, curTag_);

        // 然后wait到Consumer位置的flag之后，写个flag 到对端Producer的位置，后续正常循环
        flag_offset = blockIdx_ + coreCount_;
        WaitFlag(rank_, flag_offset, curTag_);

        flag_offset = rank_ * coreNumPerRank_ + coreIndex_;
        Record(targetRank_, flag_offset, curTag_);

        uint64_t maxSendOrRecvDataCount = 0;
        uint64_t loopTimes = 0;
        CalcLoopInfo(extraArgs, maxSendOrRecvDataCount, loopTimes);

        uint64_t processedDataCount = 0;
        for (uint64_t loop = 0; loop < loopTimes; loop++) {
            uint64_t currDataCount
                = (loop == loopTimes - 1) ? maxSendOrRecvDataCount - processedDataCount : cclBufferCountPerRank_;
            InitCoreInfo(extraArgs, processedDataCount, currDataCount);
            Producer(loop); // 写数据
            Consumer(loop); // 读数据
            SyncAll<true>();
            processedDataCount += currDataCount;
        }
    }

    uint32_t coreCount_;
    uint32_t coreNumPerRank_;
    uint32_t targetRank_;
    uint32_t coreIndex_;
    uint32_t rankNumPerCore_;
    uint64_t sendInputOffset_;
    uint64_t sendOutputOffset_;
    uint64_t sendCurCount_;
    uint64_t recvInputOffset_;
    uint64_t recvOutputOffset_;
    uint64_t recvCurCount_;
    uint64_t cclBufferCountPerRank_;
    uint64_t innerDisplsForCcl_;
};

template <typename T>
__aicore__ inline void AivAlltoAllVV2Mesh1D(KERNEL_ARGS_DEF, ExtraArgs& extraArgs)
{
    AivAlltoAllVMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.BarrierForFirstOP();
    }
    if (numBlocks >= rankSize) {
        op.Process(len, sliceId, extraArgs);
    } else {
        op.ProcessCtrlCore(len, sliceId, extraArgs);
    }
    op.BarrierAll();
}
