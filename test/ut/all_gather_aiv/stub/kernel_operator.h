/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_AIV_TEST_KERNEL_OPERATOR_H
#define ALL_GATHER_AIV_TEST_KERNEL_OPERATOR_H

#include <cstdint>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;

namespace all_gather_aiv_test {
enum class EventKind { COPY, LOAD, STORE, ALLOC, FREE, RECORD, WAIT, SYNC };
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
    std::vector<Event> events;
};
inline Trace* currentTrace = nullptr;
constexpr uint64_t RANK_STRIDE = 0x10000000ULL;
constexpr uint64_t INPUT_BASE = 0x100000000ULL;
constexpr uint64_t CCL_BASE = 0x2000000000ULL;
constexpr uint64_t OUTPUT_BASE = 0x3000000000ULL;
constexpr uint64_t UB_BASE = 0x4000000000ULL;
inline uint64_t Address(uint64_t base, uint32_t rank) { return base + rank * RANK_STRIDE; }
} // namespace all_gather_aiv_test

namespace AscendC {
constexpr int PIPE_ALL = 0;
inline void pipe_barrier(int) {}
template <int pipe>
inline void PipeBarrier()
{}
enum class QuePosition { VECIN, VECOUT };
template <typename T>
struct LocalTensor {
    uint64_t address;
};
template <typename T>
struct GlobalTensor {
    uint64_t address = 0;
    void SetGlobalBuffer(T* pointer, uint64_t = 0) { address = reinterpret_cast<uint64_t>(pointer); }
};

// 单buffer队列：持有payload时再次AllocTensor或提前释放都由桩和事件调度器检出。
template <QuePosition source, QuePosition destination, int depth>
class TQueBind {
public:
    template <typename T>
    LocalTensor<T> AllocTensor()
    {
        using namespace all_gather_aiv_test;
        if (allocated_) {
            throw std::logic_error("single UB buffer allocated twice");
        }
        allocated_ = true;
        address_ = Address(UB_BASE, currentTrace->rank) + currentTrace->block * 0x100000ULL;
        currentTrace->events.push_back({EventKind::ALLOC, address_});
        return {address_};
    }
    template <typename T>
    void EnQue(LocalTensor<T> tensor)
    {
        Check(tensor.address);
        queued_ = true;
    }
    template <typename T>
    LocalTensor<T> DeQue()
    {
        if (!queued_) {
            throw std::logic_error("UB buffer dequeued before enqueue");
        }
        queued_ = false;
        return {address_};
    }
    template <typename T>
    void FreeTensor(LocalTensor<T> tensor)
    {
        Check(tensor.address);
        all_gather_aiv_test::currentTrace->events.push_back({all_gather_aiv_test::EventKind::FREE, address_});
        allocated_ = false;
    }

private:
    void Check(uint64_t address) const
    {
        if (!allocated_ || address != address_) {
            throw std::logic_error("invalid UB buffer lifetime");
        }
    }
    uint64_t address_ = 0;
    bool allocated_ = false;
    bool queued_ = false;
};
} // namespace AscendC
#endif
