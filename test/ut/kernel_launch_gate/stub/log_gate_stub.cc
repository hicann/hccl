/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log_gate_stub.h"

#include <cstdint>
#include <mutex>
#include <vector>

#include "log.h"

namespace {
std::mutex gLogMutex;
std::vector<std::string> gLogFmts;
} // namespace

// 与 topo_match/stub/log_stub.cc 的差异：本桩让 HcclCheckLogLevel 返回 true 使日志体执行，
// 并捕获 DlogRecord 的格式串，用于断言"降级 WARNING 仅打一次"等日志行为
bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return true;
}

bool IsErrorToWarn() { return false; }

// dlog_def.h 中为弱声明，此处强定义捕获（格式串为字面量，生命周期安全）
void DlogRecord(int32_t moduleId, int32_t level, const char* fmt, ...)
{
    (void)moduleId;
    (void)level;
    std::lock_guard<std::mutex> lock(gLogMutex);
    if (fmt != nullptr) {
        gLogFmts.emplace_back(fmt);
    }
}

int32_t dlog_getlevel(int32_t moduleId, int32_t* enableEvent)
{
    (void)moduleId;
    if (enableEvent != nullptr) {
        *enableEvent = 0;
    }
    return HCCL_LOG_INFO;
}

int32_t acllogCheckDebugLevel(int32_t moduleId, int32_t logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return 0;
}

void GateStubClearLogs()
{
    std::lock_guard<std::mutex> lock(gLogMutex);
    gLogFmts.clear();
}

std::size_t GateStubCountLog(const std::string& keyword)
{
    std::lock_guard<std::mutex> lock(gLogMutex);
    std::size_t count = 0;
    for (const auto& fmt : gLogFmts) {
        if (fmt.find(keyword) != std::string::npos) {
            count++;
        }
    }
    return count;
}
