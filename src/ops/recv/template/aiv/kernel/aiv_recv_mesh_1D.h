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
class AivRecvMesh1D : public AivCommBase { //单核、多核都能运行
public:

    __aicore__ inline AivRecvMesh1D() {
    }

    __aicore__ inline void InitCoreInfo(uint64_t currDataCount)
    {
        coreIndex = blockIdx_;  // 每个核在当前coreNumPerRank里面的排序

        // 发送数据的编排
        uint64_t dataPerCore = currDataCount / coreNumPerRank; // 数据量很少的时候，dataPerCore为0
        uint64_t remainder = currDataCount % coreNumPerRank;
        // 数据对不齐的情况
        uint64_t innerDispls = 0;
        if (coreIndex < remainder) { // 这部分核需要多处理一个数据
            innerDispls = coreIndex * dataPerCore + coreIndex;
            recvCurCount = dataPerCore + 1;
        } else {
            innerDispls = coreIndex * dataPerCore + remainder;
            recvCurCount = dataPerCore;
        }
        recvInputOffset = reinterpret_cast<uint64_t>(GM_IN[targetRank]) + innerDispls * sizeof(T);
        recvOutputOffset = output_ + innerDispls * sizeof(T); // 能不能直接拿着这个地址就用
    }

    __aicore__ inline void Consumer()
    {
        if (recvCurCount == 0) {
            return;
        }
        uint64_t flag_offset = blockIdx_;
        WaitFlag(targetRank, flag_offset, curTag);

        CpGM2GM((__gm__ T *)recvOutputOffset, (__gm__ T *)recvInputOffset, recvCurCount);
        PipeBarrier<PIPE_ALL>(); // 核内自己的同步

        // 通知Send数据已经搬运完成
        Record(rank_, flag_offset, curTag);
    }

    __aicore__ inline void Process(uint64_t len, uint32_t sliceId)
    {
        // 目标卡只有一个，一个或多个核去处理
        coreNumPerRank = numBlocks_;
        if (coreNumPerRank < 1) {
            return;
        }

        coreCount = coreNumPerRank;
        if (blockIdx_ >= coreCount) { // 负责每个rank的核数相同，方便读写都能并行,这个校验好像有点鸡肋
            return;
        }

        targetRank = sendRecvRemoteRank_; // 每个核负责哪个rank的数据
        curTag = (static_cast<uint32_t>(tag_) << AIV_TAG_MOVE_RIGHT_BITS) | (sliceId & LOW_16_BITS);

        InitCoreInfo(len);
        Consumer(); // 读数据
    }

    int32_t curTag;
    uint32_t coreCount;
    uint32_t coreNumPerRank;
    uint32_t targetRank;
    uint32_t coreIndex;
    uint64_t recvInputOffset;
    uint64_t recvOutputOffset;
    uint64_t recvCurCount;
};

template<typename T>
__aicore__ inline void AivRecvV2Mesh1D(KERNEL_ARGS_DEF)
{
    AivRecvMesh1D<T> op;
    op.Init(KERNEL_CLASS_INIT, true);
    if (op.IsFirstOP(sliceId)) {
        op.SendRecvBarrierForFirstOP(rank, sendRecvRemoteRank);
    }
    op.Process(len, sliceId);
    op.SendRecvBarrierAll(rank, sendRecvRemoteRank);
}
