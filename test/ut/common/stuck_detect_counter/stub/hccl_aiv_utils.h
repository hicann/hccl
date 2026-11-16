/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_STUCK_DETECT_HCCL_AIV_UTILS_H
#define UT_STUCK_DETECT_HCCL_AIV_UTILS_H

#include <cstdint>
#include <type_traits>

#include "hccl_types.h"

namespace ops_hccl {

// 与被测代码一致的计数信息载体（真实定义见 hccl_aiv_utils.h，此处仅保留 helper 用到的字段）
struct OpCounterInfo {
    u64 headCountMem{0};
    u64 tailCountMem{0};
    u64 addOneMem{0};
    u32 memSize{0};
    bool isEnableCounter{false};
};

struct AivKernelArgs {
    const void* headCountMem{nullptr};
    const void* tailCountMem{nullptr};
    const void* addOneMem{nullptr};
    u32 counterMemSize{0};
    alignas(4) bool isEnableCounter{false};
};

static_assert(std::is_same<decltype(AivKernelArgs::isEnableCounter), bool>::value, "AivKernelArgs must use bool");
static_assert(alignof(AivKernelArgs::isEnableCounter) == 4, "AivKernelArgs bool must be 4B aligned");

struct AivExtraKernelArgs : AivKernelArgs {};

// 与真实 AivSuperKernelArgsDef 保持一致：isEnableCounter 为 uint32_t，不继承 AivKernelArgs。
struct AivSuperKernelArgs {
    const void* headCountMem{nullptr};
    const void* tailCountMem{nullptr};
    const void* addOneMem{nullptr};
    u32 counterMemSize{0};
    uint32_t isEnableCounter{0};
};

static_assert(
    std::is_same<decltype(AivSuperKernelArgs::isEnableCounter), uint32_t>::value,
    "AivSuperKernelArgs must use uint32_t");

} // namespace ops_hccl

#endif // UT_STUCK_DETECT_HCCL_AIV_UTILS_H
