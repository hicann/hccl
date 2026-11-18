/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_cpu_ctrl_stub.h"

namespace {
uint32_t gStubIdleCpuCnt = 6;
uint32_t gStubReservedCpuCnt = 0;
uint32_t gStubIdleQueryCnt = 0;
uint32_t gStubReservedQueryCnt = 0;
} // namespace

// 强定义覆盖 kernel_launch.h 中的弱声明：桩返回可控值，并记录查询次数
extern "C" {
uint32_t AicpuGetIdleCpuCnt()
{
    gStubIdleQueryCnt++;
    return gStubIdleCpuCnt;
}

uint32_t AicpuGetReservedCpuCnt()
{
    gStubReservedQueryCnt++;
    return gStubReservedCpuCnt;
}
}

void GateStubSetCpuCnt(uint32_t idleCpuCnt, uint32_t reservedCpuCnt)
{
    gStubIdleCpuCnt = idleCpuCnt;
    gStubReservedCpuCnt = reservedCpuCnt;
}

void GateStubResetCpuQueryCnt()
{
    gStubIdleQueryCnt = 0;
    gStubReservedQueryCnt = 0;
}

uint32_t GateStubGetIdleQueryCnt() { return gStubIdleQueryCnt; }

uint32_t GateStubGetReservedQueryCnt() { return gStubReservedQueryCnt; }
