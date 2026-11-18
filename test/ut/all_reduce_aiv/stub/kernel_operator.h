/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_AIV_TEST_KERNEL_OPERATOR_H
#define ALL_REDUCE_AIV_TEST_KERNEL_OPERATOR_H

#include <cstdint>

#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
namespace AscendC {
constexpr int PIPE_ALL = 0;
constexpr uint32_t EVENT_ID0 = 0;
inline void pipe_barrier(int) {}
template <int pipe>
inline void PipeBarrier()
{}

// CPU 桩：只保证生产 kernel 头文件可编译，不模拟真实流水与数据搬运语义。
template <typename T>
class LocalTensor {
public:
    LocalTensor() = default;
    LocalTensor<T> operator[](uint64_t) const { return LocalTensor<T>(); }
};

template <typename T>
class GlobalTensor {
public:
    void SetGlobalBuffer(__gm__ T*, uint64_t) {}
};
struct DataCopyExtParams {};
template <typename T>
struct DataCopyPadExtParams {};
inline void DataCopy(...) {}
inline void DataCopyPad(...) {}

enum class HardEvent {
    MTE2_V,
    V_MTE3,
    MTE3_V,
    V_MTE2,
    MTE3_MTE2,
};

template <HardEvent event>
inline void SetFlag(int32_t)
{}
template <HardEvent event>
inline void WaitFlag(int32_t)
{}

template <typename T>
inline void Add(const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, uint32_t)
{}
template <typename T>
inline void Max(const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, uint32_t)
{}
template <typename T>
inline void Min(const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, uint32_t)
{}

template <typename T>
inline void DataCopyGM2UBByAddr(const LocalTensor<T>&, const GlobalTensor<T>&, uint32_t, uint64_t)
{}
template <typename T>
inline void DataCopyUB2GMByAddr(const GlobalTensor<T>&, const LocalTensor<T>&, uint32_t, uint64_t)
{}
} // namespace AscendC
#endif
