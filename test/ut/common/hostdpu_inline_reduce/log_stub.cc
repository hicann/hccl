/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"

#include <cstdarg>
#include <cstdio>
#include <string>

namespace {
bool g_logCaptureEnabled = false;
std::string g_capturedLogs;
} // namespace

bool HcclCheckLogLevel(int, int) { return g_logCaptureEnabled; }

bool IsErrorToWarn() { return false; }

extern "C" void DlogRecord(int32_t, int32_t, const char* format, ...)
{
    constexpr size_t LOG_BUFFER_SIZE = 1024;
    char buffer[LOG_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, format);
    (void)vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    g_capturedLogs.append(buffer);
    g_capturedLogs.push_back('\n');
}

void SetDpuInlineReduceLogCaptureEnabled(bool enabled) { g_logCaptureEnabled = enabled; }

void ClearDpuInlineReduceCapturedLogs() { g_capturedLogs.clear(); }

const std::string& GetDpuInlineReduceCapturedLogs() { return g_capturedLogs; }
