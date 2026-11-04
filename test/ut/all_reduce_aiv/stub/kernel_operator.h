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
inline void pipe_barrier(int) {}
template <int pipe>
inline void PipeBarrier()
{}
} // namespace AscendC
#endif
