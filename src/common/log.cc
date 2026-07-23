/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log.h"
#include <atomic>

thread_local bool g_hcclErrToWarn = false;
constexpr int32_t HCCL_LOG_LEVEL_INVALID = -1;
static std::atomic<int32_t> g_logLevelCache{-1};
int32_t dlog_getlevel(int32_t moduleId, int32_t *enableEvent) __attribute((weak));

bool HcclCheckLogLevel(int logType, int moduleId)
{
    if ((moduleId & RUN_LOG_MASK) != 0) {
        return true;
    }
    if (UNLIKELY(g_logLevelCache.load(std::memory_order_relaxed) == HCCL_LOG_LEVEL_INVALID)) {
        int32_t enableEvent = -1;
        g_logLevelCache.store(dlog_getlevel(moduleId, &enableEvent), std::memory_order_relaxed);
    }
    return (logType >= g_logLevelCache.load(std::memory_order_relaxed));
}

void SetErrToWarnSwitch(bool flag)
{
    if (g_hcclErrToWarn != flag) {
        g_hcclErrToWarn = flag;
    }
}

bool IsErrorToWarn()
{
    return g_hcclErrToWarn;
}
