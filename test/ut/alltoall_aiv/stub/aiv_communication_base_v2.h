/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_AIV_TEST_COMMUNICATION_BASE_H
#define ALLTOALL_AIV_TEST_COMMUNICATION_BASE_H

#include <vector>
#include "aiv_defines.h"

namespace alltoall_aiv_test {
enum class EventKind { COPY, RECORD, WAIT };
struct Event {
    EventKind kind;
    uint64_t destination = 0;
    uint64_t source = 0;
    uint64_t bytes = 0;
    uint32_t target = 0;
    uint64_t slot = 0;
    int32_t tag = 0;
};
struct Trace {
    uint32_t rank;
    uint32_t block;
    uint32_t firstBarriers = 0;
    uint32_t endBarriers = 0;
    std::vector<Event> events;
};
inline Trace* currentTrace = nullptr;
constexpr uint64_t RANK_STRIDE = 0x10000000ULL;
constexpr uint64_t INPUT_BASE = 0x100000000ULL;
constexpr uint64_t CCL_BASE = 0x2000000000ULL;
constexpr uint64_t OUTPUT_BASE = 0x3000000000ULL;
inline uint64_t Address(uint64_t base, uint32_t rank) { return base + rank * RANK_STRIDE; }
} // namespace alltoall_aiv_test

// 在CPU上执行生产kernel，记录拷贝与flag依赖关系。
// 随后由测试调度器校验内存覆盖情况，并解析每一个WaitFlag。
class AivCommBase {
public:
    // 尾部开关参数(useDoubleBuffer/pingpong/isWrite等)不参与桩逻辑，用变参吸收，
    // 避免基类Init签名演进时UT编译失败。
    template <typename... ExtraArgs>
    void Init(
        GM_ADDR, uint64_t input, uint64_t output, uint32_t rank, uint32_t, uint32_t rankSize, uint64_t len, uint32_t,
        uint32_t, uint32_t, uint64_t inputStride, uint64_t outputStride, uint64_t, uint64_t, uint64_t, GM_ADDR, GM_ADDR,
        GM_ADDR, uint32_t, bool, uint32_t blocks, ExtraArgs...)
    {
        input_ = input;
        output_ = output;
        rank_ = rank;
        rankSize_ = rankSize;
        len_ = len;
        inputSliceStride_ = inputStride;
        outputSliceStride_ = outputStride;
        numBlocks_ = blocks;
        blockIdx_ = alltoall_aiv_test::currentTrace->block;
        myGmIn_ = GetGmIn(rank);
    }
    GM_ADDR GetGmIn(uint32_t rank)
    {
        return reinterpret_cast<GM_ADDR>(alltoall_aiv_test::Address(alltoall_aiv_test::CCL_BASE, rank));
    }
    template <typename T>
    void CpGM2GM(T* destination, T* source, uint64_t count)
    {
        using namespace alltoall_aiv_test;
        currentTrace->events.push_back(
            {EventKind::COPY, reinterpret_cast<uint64_t>(destination), reinterpret_cast<uint64_t>(source),
             count * sizeof(T)});
    }
    void Record(uint32_t target, uint64_t slot, int32_t tag)
    {
        alltoall_aiv_test::currentTrace->events.push_back(
            {alltoall_aiv_test::EventKind::RECORD, 0, 0, 0, target, slot, tag});
    }
    void WaitFlag(uint32_t target, uint64_t slot, int32_t tag)
    {
        alltoall_aiv_test::currentTrace->events.push_back(
            {alltoall_aiv_test::EventKind::WAIT, 0, 0, 0, target, slot, tag});
    }
    bool IsFirstOP(uint32_t) { return true; }
    bool IsIdleCore() const { return blockIdx_ >= numBlocks_; }
    void BarrierForFirstOP() { ++alltoall_aiv_test::currentTrace->firstBarriers; }
    void BarrierAll() { ++alltoall_aiv_test::currentTrace->endBarriers; }

    uint64_t len_ = 0;
    uint64_t input_ = 0;
    uint64_t output_ = 0;
    uint64_t inputSliceStride_ = 0;
    uint64_t outputSliceStride_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t numBlocks_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t tag_ = 1;
    uint32_t curTag_ = 0;
    GM_ADDR myGmIn_ = nullptr;
};
#endif
