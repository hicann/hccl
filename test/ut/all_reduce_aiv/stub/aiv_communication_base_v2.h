/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_AIV_TEST_COMMUNICATION_BASE_H
#define ALL_REDUCE_AIV_TEST_COMMUNICATION_BASE_H

#include <vector>
#include "aiv_defines.h"

namespace all_reduce_aiv_test {
enum class EventKind { COPY, RECORD, WAIT };
struct Event {
    EventKind kind;
    uint64_t destination = 0;
    uint64_t source = 0;
    uint64_t bytes = 0;
    uint32_t target = 0;
    uint64_t slot = 0;
    int32_t tag = 0;
    bool reduce = false;
};
struct Trace {
    uint32_t rank = 0;
    uint32_t block = 0;
    uint32_t firstBarriers = 0;
    uint32_t endBarriers = 0;
    uint32_t syncAlls = 0;
    std::vector<Event> events;
};
inline Trace* currentTrace = nullptr;
constexpr uint64_t RANK_STRIDE = 0x10000000ULL;
constexpr uint64_t INPUT_BASE = 0x100000000ULL;
constexpr uint64_t CCL_BASE = 0x2000000000ULL;
constexpr uint64_t OUTPUT_BASE = 0x3000000000ULL;
inline uint64_t Address(uint64_t base, uint32_t rank) { return base + rank * RANK_STRIDE; }
} // namespace all_reduce_aiv_test

// 设备侧 HcclReduceOp 来自 CANN 头文件，这里给出同名的非限定枚举，
// 既支持生产代码里的 `HcclReduceOp::HCCL_REDUCE_*` 写法，也保持与 uint32_t 可直接比较。
enum HcclReduceOp {
    HCCL_REDUCE_SUM = 0,
    HCCL_REDUCE_PROD = 1,
    HCCL_REDUCE_MAX = 2,
    HCCL_REDUCE_MIN = 3,
};

// 对称路径通过 inOutQue 申请 UB tensor；桩不模拟真实队列，只保证类型可编译。
class StubTensorQue {
public:
    template <typename T>
    AscendC::LocalTensor<T> AllocTensor()
    {
        return AscendC::LocalTensor<T>();
    }
    template <typename T>
    void FreeTensor(const AscendC::LocalTensor<T>&)
    {}
};

// 在CPU上执行生产kernel，记录拷贝与flag依赖关系，随后由测试调度器解析依赖并校验覆盖情况。
class AivCommBase {
public:
    // 尾部开关参数(presync/pingpong/isWrite等)不参与桩逻辑，用变参吸收，
    // 避免基类Init签名演进时UT编译失败。
    template <typename... ExtraArgs>
    void Init(
        GM_ADDR, uint64_t input, uint64_t output, uint32_t rank, uint32_t, uint32_t rankSize, uint64_t len,
        uint32_t dataType, uint32_t reduceOp, uint32_t root, uint64_t inputStride, uint64_t outputStride, uint64_t,
        uint64_t, uint64_t, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, bool, uint32_t blocks, ExtraArgs...)
    {
        input_ = input;
        output_ = output;
        rank_ = rank;
        rankSize_ = rankSize;
        len_ = len;
        dataType_ = dataType;
        reduceOp_ = reduceOp;
        root_ = root;
        inputSliceStride_ = inputStride;
        outputSliceStride_ = outputStride;
        numBlocks_ = blocks;
        blockIdx_ = all_reduce_aiv_test::currentTrace->block;
        myGmIn_ = GetGmIn(rank);
    }
    GM_ADDR GetGmIn(uint64_t rank)
    {
        return reinterpret_cast<GM_ADDR>(
            all_reduce_aiv_test::Address(all_reduce_aiv_test::CCL_BASE, static_cast<uint32_t>(rank)));
    }
    // 桩不区分对称/非对称内存：两侧均返回空地址，对称开关默认关闭，UT 继续覆盖两 shot 基线。
    GM_ADDR GetGmSymIn(uint32_t) { return nullptr; }
    GM_ADDR GetGmSymOut(uint32_t) { return nullptr; }
    template <typename T>
    void CpGM2GM(T* destination, T* source, uint64_t count)
    {
        PushCopy(destination, source, count * sizeof(T), false);
    }
    template <typename T>
    void CpGM2GM(T* destination, T* source, uint64_t count, uint32_t)
    {
        PushCopy(destination, source, count * sizeof(T), true);
    }
    void Record(uint32_t target, uint64_t slot, int32_t tag)
    {
        all_reduce_aiv_test::currentTrace->events.push_back(
            {all_reduce_aiv_test::EventKind::RECORD, 0, 0, 0, target, slot, tag});
    }
    void WaitFlag(uint32_t target, uint64_t slot, int32_t tag)
    {
        all_reduce_aiv_test::currentTrace->events.push_back(
            {all_reduce_aiv_test::EventKind::WAIT, 0, 0, 0, target, slot, tag});
    }
    template <bool lock = false>
    void SyncAll()
    {
        ++all_reduce_aiv_test::currentTrace->syncAlls;
    }
    static uint32_t GetLargestPowerOf2LessThan(uint32_t value)
    {
        if (value <= 1) {
            return 0;
        }
        uint32_t power = 1;
        while (power * 2 < value) {
            power *= 2;
        }
        return power;
    }
    bool IsFirstOP(uint32_t) { return true; }
    bool IsIdleCore() const { return blockIdx_ >= numBlocks_; }
    void BarrierForFirstOP() { ++all_reduce_aiv_test::currentTrace->firstBarriers; }
    void BarrierAll(uint32_t = 0) { ++all_reduce_aiv_test::currentTrace->endBarriers; }
    void HeadCounter() {}
    void TailCounter() {}
    void SyncBeforeTail() {}

    uint64_t len_ = 0;
    uint64_t input_ = 0;
    uint64_t output_ = 0;
    uint64_t inputSliceStride_ = 0;
    uint64_t outputSliceStride_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t dataType_ = 0;
    uint32_t reduceOp_ = 0;
    uint32_t root_ = 0;
    uint32_t numBlocks_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t tag_ = 1;
    uint32_t curTag_ = 0;
    GM_ADDR myGmIn_ = nullptr;
    // 对称路径开关：桩恒为 0，使 UseSymPath() 返回 false，生产代码落回两 shot 基线。
    uint32_t symInEnable_ = 0;
    uint32_t symOutEnable_ = 0;
    StubTensorQue inOutQue;

private:
    template <typename T>
    void PushCopy(T* destination, T* source, uint64_t bytes, bool reduce)
    {
        all_reduce_aiv_test::currentTrace->events.push_back(
            {all_reduce_aiv_test::EventKind::COPY, reinterpret_cast<uint64_t>(destination),
             reinterpret_cast<uint64_t>(source), bytes, 0, 0, 0, reduce});
    }
};

// 生产 kernel 的入口函数在类外调用 SyncAll<true>()，桩需提供同名自由函数并计入当核计数。
template <bool isAIVOnly = true>
inline void SyncAll()
{
    ++all_reduce_aiv_test::currentTrace->syncAlls;
}
#endif
