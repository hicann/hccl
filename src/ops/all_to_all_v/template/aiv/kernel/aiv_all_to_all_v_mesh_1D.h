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
class AivAlltoAllVMesh1D : public AivCommBase {
public:
 
    __aicore__ inline AivAlltoAllVMesh1D() {
    }
 
    __aicore__ inline void InitCoreInfo(uint64_t len, ExtraArgs &extraArgsPerLoop)
    {
        targetRank = block_idx / coreNumPerRank; // 每个核负责哪个rank的数据
        coreIndex = block_idx % coreNumPerRank;  // 每个核在当前coreNumPerRank里面的排序
 
        // 发送数据的编排
        uint64_t dataPerCore = extraArgsPerLoop.sendCounts[targetRank] / coreNumPerRank; // 数据量很少的时候，dataPerCore为0
        uint64_t remainder = extraArgsPerLoop.sendCounts[targetRank] % coreNumPerRank;
        // 数据对不齐的情况
        uint64_t innerDispls = 0;
        if (coreIndex < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex * dataPerCore + coreIndex;
            sendCurCount = dataPerCore + 1;
        } else {
            innerDispls = coreIndex * dataPerCore + remainder;
            sendCurCount = dataPerCore;
        }
        sendInputOffset = input_ + (extraArgsPerLoop.sendDispls[targetRank] + innerDispls)  * sizeof(T);
        sendOutputOffset = reinterpret_cast<uint64_t>(GM_IN[rank_]) + (targetRank * len + innerDispls) * sizeof(T);
 
        //接收数据的编排
        dataPerCore = extraArgsPerLoop.recvCounts[targetRank] / coreNumPerRank;
        remainder = extraArgsPerLoop.recvCounts[targetRank] % coreNumPerRank;
        if (coreIndex < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex * dataPerCore + coreIndex;
            recvCurCount = dataPerCore + 1;
        } else {
            innerDispls = coreIndex * dataPerCore + remainder;
            recvCurCount = dataPerCore;
        }
        recvInputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + (rank_ * len + innerDispls) * sizeof(T);
        recvOutputOffset = output_ + (extraArgsPerLoop.recvDispls[targetRank] + innerDispls) * sizeof(T);
    }
 
    __aicore__ inline void Producer()
    {
        if (sendCurCount == 0) {
            return;
        }
        CpGM2GM((__gm__ T *)sendOutputOffset, (__gm__ T *)sendInputOffset, sendCurCount);
        PipeBarrier<PIPE_ALL>();
        uint64_t flag_offset = block_idx;
        WaitFlag(rank_, flag_offset, 0);
        Record(rank_, flag_offset, curTag);
    }
 
    __aicore__ inline void Consumer()
    {
        if (recvCurCount == 0) {
            return;
        }
        uint64_t flag_offset = rank_ * coreNumPerRank + coreIndex;
        WaitFlag(targetRank, flag_offset, curTag);
        Record(targetRank, flag_offset, 0);
        CpGM2GM((__gm__ T *)recvOutputOffset, (__gm__ T *)recvInputOffset, recvCurCount);
        PipeBarrier<PIPE_ALL>(); // 核内自己的同步
    }
 
    __aicore__ inline void Process(uint64_t len, uint32_t tag, ExtraArgs &extraArgs)
    {
        // 先看一个或者多个核处理一张卡数据的情况
        coreNumPerRank = numBlocks_ / rankSize_;
        if (coreNumPerRank < 1) { // 控核情况暂不处理
            return;
        }
 
        coreCount = coreNumPerRank * rankSize_;
        if (block_idx >= coreCount) { // 负责每个rank的核数相同，方便读写都能并行
            return;
        }
 
        curTag = static_cast<int32_t>(tag);
        cclBufferCountPerRank = len;
 
        // 运行过程中用到的所有flag，先置为0，后面会复用
        Record(rank_, block_idx, 0);
        PipeBarrier<PIPE_ALL>();
 
        // 这里根据ccl buffer的大小去做循环
        uint64_t maxSendOrRecvDataCount = 0;
        for (uint64_t i = 0; i < rankSize_; i++) {
            maxSendOrRecvDataCount = max(maxSendOrRecvDataCount, extraArgs.sendCounts[i]);
            maxSendOrRecvDataCount = max(maxSendOrRecvDataCount, extraArgs.recvCounts[i]);
        }
 
        uint64_t processedDataCount = 0;
        // 每张卡的loopTimes可能是不一样的
        uint64_t loopTimes = maxSendOrRecvDataCount / cclBufferCountPerRank +
            static_cast<uint64_t>(maxSendOrRecvDataCount % cclBufferCountPerRank != 0);
        for (uint64_t loop = 0; loop < loopTimes; loop++) {
            ExtraArgs extraArgsPerLoop;
            uint64_t currDataCount = (loop == loopTimes - 1) ? maxSendOrRecvDataCount - processedDataCount : cclBufferCountPerRank;
            for (uint64_t i = 0; i < rankSize_; i++) {
                if (extraArgs.sendCounts[i] > processedDataCount) {
                    extraArgsPerLoop.sendCounts[i] = min(currDataCount, extraArgs.sendCounts[i] - processedDataCount);
                    extraArgsPerLoop.sendDispls[i] = extraArgs.sendDispls[i] + processedDataCount;
                } else {
                    extraArgsPerLoop.sendCounts[i] = 0;
                    extraArgsPerLoop.sendDispls[i] = extraArgs.sendDispls[i] + extraArgs.sendCounts[i];
                }
 
                if (extraArgs.recvCounts[i] > processedDataCount) {
                    extraArgsPerLoop.recvCounts[i] = min(currDataCount, extraArgs.recvCounts[i] - processedDataCount);
                    extraArgsPerLoop.recvDispls[i] = extraArgs.recvDispls[i] + processedDataCount;
                } else {
                    extraArgsPerLoop.recvCounts[i] = 0;
                    extraArgsPerLoop.recvDispls[i] = extraArgs.recvDispls[i] + extraArgs.recvCounts[i];
                }
            }
 
            InitCoreInfo(currDataCount, extraArgsPerLoop);
            Producer(); // 写数据
            Consumer(); // 读数据
            processedDataCount += currDataCount;
        }
    }
 
    int32_t curTag;
    uint32_t coreCount;
    uint32_t coreNumPerRank;
    uint32_t targetRank;
    uint32_t coreIndex;
    uint64_t sendInputOffset;
    uint64_t sendOutputOffset;
    uint64_t sendCurCount;
    uint64_t recvInputOffset;
    uint64_t recvOutputOffset;
    uint64_t recvCurCount;
    uint64_t cclBufferCountPerRank;
};
 
template<typename T>
__aicore__ inline void AivAlltoAllVV2Mesh1D(EXTERN_KERNEL_ARGS_DEF_V2)
{
    AivAlltoAllVMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    SyncAll<true>();
    if (block_idx == 0 && tag >> AIV_TAG_MOVE_RIGHT_BITS == 1 && (tag & LOW_16_BITS) == 1) {
        op.BarrierForFirstOP();
    }
    SyncAll<true>();
    op.Process(len, tag, extraArgs);
    op.BarrierAll();
}