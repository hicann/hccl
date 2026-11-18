/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_AICPU_CPU_CTRL_STUB_H
#define UT_AICPU_CPU_CTRL_STUB_H

#include <cstdint>

// 仅链接进 kernel_launch_gate_test（带桩目标）。
// 不链接进 kernel_launch_gate_null_test —— 该目标的弱符号恒为 null，专测降级路径。

void GateStubSetCpuCnt(uint32_t idleCpuCnt, uint32_t reservedCpuCnt);

void GateStubResetCpuQueryCnt();

uint32_t GateStubGetIdleQueryCnt();

uint32_t GateStubGetReservedQueryCnt();

#endif // UT_AICPU_CPU_CTRL_STUB_H
