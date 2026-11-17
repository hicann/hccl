/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_AIV_TEST_KERNEL_OPERATOR_H
#define BSR_AIV_TEST_KERNEL_OPERATOR_H

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

// 在 CPU 上执行生产 kernel 的最小 AscendC 底座。
//
// GM 用"按 4KB 页稀疏"的伪内存建模：地址是虚构值，绝不做真实解引用，被访问到的页才分配。
// 这样 CpGM2GM 能真正搬运字节（可断言端到端数据落点），同时能统计任意地址的 4B 字读写次数
// —— 后者是验证"BSR 不再读也不写 65MB 块里那个共享 tag 计数器"的唯一手段。
#define __aicore__
#define __gm__

using GM_ADDR = uint8_t*;

// 与 src/ops/op_common/algorithm/template/aiv/aiv_interface/sync_interface.h 保持一致。
// 生产 kernel 经 aiv_communication_base_v2.h 间接引入它；本桩替换了那个头，故在此补齐这两个常量。
constexpr uint64_t UB_FLAG_PAD_COUNT = 8;
constexpr uint64_t UB_ADDRESS_PAD_COUNT = 4;

namespace bsr_aiv_test {

// 每个 rank 的 65MB 通信信息块；80MB 步长留足 34M 起的 pingpong 区
constexpr uint64_t BLOCK_BASE = 0x1'0000'0000ULL;
constexpr uint64_t BLOCK_STRIDE = 0x0500'0000ULL;
// 用户数据区，每 rank 1MB
constexpr uint64_t USER_BASE = 0x4'0000'0000ULL;
constexpr uint64_t USER_STRIDE = 0x0010'0000ULL;
// UB，每核 1MB：InitBsr 注册的 buffer 合计约 800KB，各 buffer 偏移互不重叠
constexpr uint64_t UB_BASE = 0x8'0000'0000ULL;
constexpr uint64_t UB_STRIDE = 0x0010'0000ULL;

inline uint64_t BlockBase(uint32_t rank) { return BLOCK_BASE + static_cast<uint64_t>(rank) * BLOCK_STRIDE; }

inline uint64_t UserBuf(uint32_t rank, uint64_t offset = 0)
{
    return USER_BASE + static_cast<uint64_t>(rank) * USER_STRIDE + offset;
}

enum class EventKind {
    COPY,   // CpGM2GM：dst/src/bytes
    RECORD, // 写 flag：dst 为绝对地址，slot 为槽位号，tag 为写入值
    WAIT,   // 等 flag：dst 为绝对地址，tag 为期望值，actual 为读到的值
    SYNC,   // SyncAll
    ALLOC,  // UB AllocTensor（BSR 不该出现：不再走 GetTag / CpGM2GM 的实现细节）
    FREE
};

struct Event {
    EventKind kind;
    uint64_t dst = 0;
    uint64_t src = 0;
    uint64_t bytes = 0;
    uint32_t target = 0; // flag 的目标 rank
    uint64_t slot = 0;
    int32_t tag = 0;
    int32_t actual = 0;
    bool matched = false;
    uint32_t rank = 0;
    uint32_t core = 0;
};

constexpr uint64_t PAGE_SHIFT = 12;
constexpr uint64_t PAGE_SIZE = 1ULL << PAGE_SHIFT;
constexpr uint64_t PAGE_MASK = PAGE_SIZE - 1;

struct Store {
    std::map<uint64_t, std::array<uint8_t, PAGE_SIZE>> pages;
    std::map<uint64_t, uint32_t> writes; // 以 4B 字为粒度的写次数
    std::map<uint64_t, uint32_t> reads;
    std::vector<Event> events;
    uint32_t currentRank = 0;
    uint32_t currentBlock = 0;
    uint32_t waitMisses = 0;
    uint32_t ubAllocs = 0;
};

inline Store& store()
{
    static Store instance;
    return instance;
}

inline const std::vector<Event>& Events() { return store().events; }

inline void ResetTrace()
{
    store().events.clear();
    store().waitMisses = 0;
    store().ubAllocs = 0;
    store().writes.clear();
    store().reads.clear();
}

inline uint8_t* Page(uint64_t addr)
{
    const uint64_t index = addr >> PAGE_SHIFT;
    auto& pages = store().pages;
    auto it = pages.find(index);
    if (it == pages.end()) {
        it = pages.emplace(index, std::array<uint8_t, PAGE_SIZE>{}).first;
    }
    return it->second.data();
}

// 逐 4B 字计数：一次 n 字节写会命中 n/4 个字，读同理。
inline void CountAccess(std::map<uint64_t, uint32_t>& counter, uint64_t addr, uint64_t bytes)
{
    for (uint64_t cur = addr & ~0x3ULL; cur < addr + bytes; cur += 4) {
        counter[cur]++;
    }
}

inline void CopyBytes(uint64_t dst, uint64_t src, uint64_t bytes)
{
    if (bytes == 0) {
        return;
    }
    // 先取到临时区再写，避免自收发时源目的同页重叠。
    std::vector<uint8_t> buffer(static_cast<size_t>(bytes));
    for (uint64_t i = 0; i < bytes; i++) {
        buffer[static_cast<size_t>(i)] = Page(src + i)[static_cast<size_t>((src + i) & PAGE_MASK)];
    }
    for (uint64_t i = 0; i < bytes; i++) {
        Page(dst + i)[static_cast<size_t>((dst + i) & PAGE_MASK)] = buffer[static_cast<size_t>(i)];
    }
    CountAccess(store().reads, src, bytes);
    CountAccess(store().writes, dst, bytes);
}

inline uint32_t GmRead32(uint64_t addr)
{
    uint32_t value = 0;
    for (uint64_t i = 0; i < sizeof(value); i++) {
        reinterpret_cast<uint8_t*>(&value)[i] = Page(addr + i)[static_cast<size_t>((addr + i) & PAGE_MASK)];
    }
    CountAccess(store().reads, addr, sizeof(value));
    return value;
}

inline void GmWrite32(uint64_t addr, uint32_t value)
{
    for (uint64_t i = 0; i < sizeof(value); i++) {
        Page(addr + i)[static_cast<size_t>((addr + i) & PAGE_MASK)] = reinterpret_cast<const uint8_t*>(&value)[i];
    }
    CountAccess(store().writes, addr, sizeof(value));
}

inline uint64_t GmRead64(uint64_t addr)
{
    uint64_t value = 0;
    for (uint64_t i = 0; i < sizeof(value); i++) {
        reinterpret_cast<uint8_t*>(&value)[i] = Page(addr + i)[static_cast<size_t>((addr + i) & PAGE_MASK)];
    }
    CountAccess(store().reads, addr, sizeof(value));
    return value;
}

inline void GmWrite64(uint64_t addr, uint64_t value)
{
    for (uint64_t i = 0; i < sizeof(value); i++) {
        Page(addr + i)[static_cast<size_t>((addr + i) & PAGE_MASK)] = reinterpret_cast<const uint8_t*>(&value)[i];
    }
    CountAccess(store().writes, addr, sizeof(value));
}

// 测试铺数据用：不计入读写次数，避免污染"某地址被读写几次"的断言。
inline void SeatBytes(uint64_t addr, const std::vector<uint8_t>& bytes)
{
    for (size_t i = 0; i < bytes.size(); i++) {
        Page(addr + i)[(addr + i) & PAGE_MASK] = bytes[i];
    }
}

inline std::vector<uint8_t> ReadBytes(uint64_t addr, uint64_t bytes)
{
    std::vector<uint8_t> out(static_cast<size_t>(bytes));
    for (uint64_t i = 0; i < bytes; i++) {
        out[static_cast<size_t>(i)] = Page(addr + i)[static_cast<size_t>((addr + i) & PAGE_MASK)];
    }
    return out;
}

inline uint32_t WriteCountAt(uint64_t addr)
{
    auto it = store().writes.find(addr & ~0x3ULL);
    return it == store().writes.end() ? 0 : it->second;
}

inline uint32_t ReadCountAt(uint64_t addr)
{
    auto it = store().reads.find(addr & ~0x3ULL);
    return it == store().reads.end() ? 0 : it->second;
}

inline void Push(Event event)
{
    event.rank = store().currentRank;
    event.core = store().currentBlock;
    store().events.push_back(event);
}

inline uint64_t UbBase() { return UB_BASE + static_cast<uint64_t>(store().currentBlock) * UB_STRIDE; }

} // namespace bsr_aiv_test

namespace AscendC {

constexpr int PIPE_ALL = 0;
enum class QuePosition { VECIN, VECOUT };

inline void pipe_barrier(int) {}

template <int pipe>
inline void PipeBarrier()
{}

inline uint32_t GetBlockIdx() { return bsr_aiv_test::store().currentBlock; }

template <typename T>
struct LocalTensor {
    uint64_t address = 0;
};

template <typename T>
struct GlobalTensor {
    uint64_t address = 0;
    void SetGlobalBuffer(T* pointer, uint64_t = 0) { address = reinterpret_cast<uint64_t>(pointer); }
};

// UB 队列：只跟踪生命周期。BSR 不再走 GetTag/RestoreTagCounter，故 inOutQue 不应被分配，
// 分配次数（store().ubAllocs）因此可以作为"没有意外触碰 inOutQue"的断言点。
template <QuePosition source, QuePosition destination, int depth>
class TQueBind {
public:
    void Configure(uint32_t, uint64_t len, uint64_t base)
    {
        len_ = len;
        base_ = base;
    }

    template <typename T>
    LocalTensor<T> AllocTensor()
    {
        if (busy_) {
            throw std::logic_error("UB buffer allocated twice without free");
        }
        busy_ = true;
        bsr_aiv_test::store().ubAllocs++;
        bsr_aiv_test::Push({bsr_aiv_test::EventKind::ALLOC, base_});
        LocalTensor<T> tensor;
        tensor.address = base_;
        return tensor;
    }

    template <typename T>
    void FreeTensor(LocalTensor<T>&)
    {
        if (!busy_) {
            throw std::logic_error("UB buffer freed while not allocated");
        }
        busy_ = false;
        bsr_aiv_test::Push({bsr_aiv_test::EventKind::FREE, base_});
    }

private:
    uint64_t base_ = 0;
    uint64_t len_ = 0;
    bool busy_ = false;
};

template <typename T = void>
class TBuf {
public:
    template <typename U>
    LocalTensor<U> GetWithOffset(uint64_t, uint64_t offset) const
    {
        LocalTensor<U> tensor;
        tensor.address = base_ + offset;
        return tensor;
    }

private:
    friend class TPipe;
    uint64_t base_ = 0;
};

class TPipe {
public:
    // 与板上 InitBuffer 一致：按调用顺序在 UB 内顺次切分，所以不同 buffer 的地址必然不同。
    template <typename T>
    void InitBuffer(TBuf<T>& buf, uint64_t len)
    {
        buf.base_ = bsr_aiv_test::UbBase() + cursor_;
        cursor_ += Align(len);
    }

    template <QuePosition in, QuePosition out, int depth>
    void InitBuffer(TQueBind<in, out, depth>& que, uint32_t queDepth, uint64_t len)
    {
        que.Configure(queDepth, len, bsr_aiv_test::UbBase() + cursor_);
        cursor_ += Align(len * queDepth);
    }

private:
    static uint64_t Align(uint64_t len)
    {
        constexpr uint64_t align = 32;
        return (len + align - 1) / align * align;
    }

    uint64_t cursor_ = 0;
};

struct DataCopyExtParams {};
template <typename T>
struct DataCopyPadExtParams {};
inline void DataCopy(...) {}
inline void DataCopyPad(...) {}

} // namespace AscendC

#endif // BSR_AIV_TEST_KERNEL_OPERATOR_H
