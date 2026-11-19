/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_URMA_AIV_COMM_BASE_STUB_H
#define ALLTOALL_URMA_AIV_COMM_BASE_STUB_H

#include <map>
#include <vector>
#include "aiv_defines.h"

// 在CPU上执行URMA kernel, 记录WRITE/COPY/WAIT/CLEAR事件序列, 由测试调度器离线重演校验
// 依赖协议(地址就绪1/数据到达2)与内存覆盖。真实GM读写由桩转成事件, WaitNotify不自旋。
namespace alltoall_urma_aiv_test {
enum class EventKind { WRITE, COPY, WAIT, CLEAR, READ, DRAIN, SYNC };
struct Event {
    EventKind kind;
    uint64_t a = 0; // WRITE/CLEAR/WAIT/READ: 地址; COPY: dst
    uint64_t b = 0; // WRITE: 值; WAIT: 期望值; COPY: src
    uint64_t c = 0; // COPY: bytes
};
struct Trace {
    uint32_t rank = 0;
    uint32_t block = 0;
    std::vector<Event> events;
};
inline Trace* currentTrace = nullptr;
// two-pass: pass1收集对端output地址slot写入值, pass2由ReadPeerOutputAddr查表返回真实值
inline std::map<uint64_t, uint64_t>* outputSlotValues = nullptr;

constexpr uint64_t RANK_STRIDE = 0x10000000ULL;
constexpr uint64_t BUFFIN_BASE = 0x100000000ULL;
constexpr uint64_t INPUT_BASE = 0x1100000000ULL;
constexpr uint64_t OUTPUT_BASE = 0x2100000000ULL;
inline uint64_t Address(uint64_t base, uint32_t rank) { return base + rank * RANK_STRIDE; }
} // namespace alltoall_urma_aiv_test

struct UrmaWqeEntry {
    uint32_t odr;
    uint32_t fence;
    uint32_t se;
    uint32_t cqe;
    uint32_t inlineEn;
};
using ChannelHandle = uint64_t;

class AivUrmaCommBase {
public:
    __aicore__ inline AivUrmaCommBase() = default;

    // 签名精确匹配 KERNEL_CLASS_INIT + subCommRankLevel + level0ChannelNumPerRank(23参)
    __aicore__ inline void Init(
        GM_ADDR buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t, uint32_t rankSize, uint64_t, uint32_t,
        uint32_t, uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, bool,
        uint32_t blocks, uint32_t, uint32_t)
    {
        buffIn_ = (uint64_t)buffIn;
        input_ = input;
        output_ = output;
        rank_ = rank;
        subCommRankSize_ = rankSize;
        numBlocks_ = blocks;
        blockIdx_ = GetBlockIdx();
        notifyBase_ = buffIn_ + FLAG1_OFFSET;
        outputAddrSlotPtr_ = buffIn_ + AIV_URMA_OUTPUT_ADDR_SLOT_OFFSET;
    }

    __aicore__ inline uint32_t GetBlockIdx() { return alltoall_urma_aiv_test::currentTrace->block; }

    __aicore__ inline uint64_t GetPeerCommInfoBase(uint32_t peerRank)
    {
        return alltoall_urma_aiv_test::Address(alltoall_urma_aiv_test::BUFFIN_BASE, peerRank);
    }

    __aicore__ inline uint64_t ReadPeerOutputAddr(uint32_t peerRank)
    {
        uint64_t slot = outputAddrSlotPtr_ + peerRank * sizeof(uint64_t);
        alltoall_urma_aiv_test::currentTrace->events.push_back({alltoall_urma_aiv_test::EventKind::READ, slot, 0, 0});
        if (alltoall_urma_aiv_test::outputSlotValues != nullptr) {
            auto it = alltoall_urma_aiv_test::outputSlotValues->find(slot);
            return it != alltoall_urma_aiv_test::outputSlotValues->end() ? it->second : 0;
        }
        return 0;
    }

    __aicore__ inline void SubmitDataOnBatch(
        uint64_t, uint64_t remoteDst, uint64_t localSrc, uint64_t dataLen, bool needNotify, uint64_t notifyAddr,
        uint64_t notifyVal)
    {
        alltoall_urma_aiv_test::currentTrace->events.push_back(
            {alltoall_urma_aiv_test::EventKind::COPY, remoteDst, localSrc, dataLen});
        if (needNotify) {
            alltoall_urma_aiv_test::currentTrace->events.push_back(
                {alltoall_urma_aiv_test::EventKind::WRITE, notifyAddr, notifyVal, 0});
        }
    }

    __aicore__ inline void FlushBatch() {}

    __aicore__ inline void WaitNotify(uint64_t notifySlot, uint32_t expectedVal, bool clearAfter)
    {
        alltoall_urma_aiv_test::currentTrace->events.push_back(
            {alltoall_urma_aiv_test::EventKind::WAIT, notifySlot, expectedVal, 0});
        if (clearAfter) {
            alltoall_urma_aiv_test::currentTrace->events.push_back(
                {alltoall_urma_aiv_test::EventKind::CLEAR, notifySlot, 0, 0});
        }
    }

    __aicore__ inline void LocalCopy(uint64_t dst, uint64_t src, uint64_t len)
    {
        alltoall_urma_aiv_test::currentTrace->events.push_back(
            {alltoall_urma_aiv_test::EventKind::COPY, dst, src, len});
    }

    template <bool = true>
    __aicore__ inline void SyncAll()
    {
        alltoall_urma_aiv_test::currentTrace->events.push_back({alltoall_urma_aiv_test::EventKind::SYNC, 0, 0, 0});
    }

    __aicore__ inline uint64_t GetChannelHandle(uint32_t peerRank, uint32_t ch)
    {
        return ((uint64_t)peerRank << 32) | ch;
    }

    __aicore__ inline uint32_t GetChannelsPerRank(uint32_t) { return channelCfg_.channelsPerRank; }

    __aicore__ inline uint32_t GetPortNum(uint32_t, uint32_t ch)
    {
        return (ch < channelCfg_.portsPerChannel) ? channelCfg_.portsPerChannel : 1;
    }

    struct HcommStub {
        // C++20: auto非类型参数包吸收核显式传入的<bool, PIPE_S, PIPE_MTE3, UrmaWqeEntry>模板实参
        template <typename T, auto...>
        __aicore__ inline void WriteValueNbi(ChannelHandle, GM_ADDR addr, T val)
        {
            auto* trace = alltoall_urma_aiv_test::currentTrace;
            trace->events.push_back({alltoall_urma_aiv_test::EventKind::WRITE, (uint64_t)addr, (uint64_t)val, 0});
        }

        __aicore__ inline int32_t Drain(ChannelHandle)
        {
            alltoall_urma_aiv_test::currentTrace->events.push_back({alltoall_urma_aiv_test::EventKind::DRAIN, 0, 0, 0});
            return 0;
        }
    } hcomm_;

    struct ChannelCfg {
        uint32_t channelsPerRank;
        uint32_t portsPerChannel;
    };
    static inline ChannelCfg channelCfg_{1, 1};

    uint64_t buffIn_ = 0;
    uint64_t input_ = 0;
    uint64_t output_ = 0;
    uint64_t notifyBase_ = 0;
    uint64_t outputAddrSlotPtr_ = 0;
    uint32_t rank_ = 0;
    uint32_t subCommRankSize_ = 0;
    uint32_t subCommRankLevel_ = 0;
    uint32_t numBlocks_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t len_ = 0;
};
#endif
