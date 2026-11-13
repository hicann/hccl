/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <sys/time.h>
#include <time.h>
#include <cstdio>
#include <unistd.h>
#include <securec.h>

// ST 日志契约（PR#3933 评审 21ec67 选路命中记录引入后确立；dev-r16 v2 桩侧双门禁分工定稿）：
//   ① dlog_getlevel 返回 min(logLevel, 0x01)——生产侧可见性门禁。真实 libhccl.so 的
//     HcclCheckLogLevel 仅在进程首笔 op 日志时经一次性缓存（src/common/log.cc g_logLevelCache）
//     探测本值并固化（中途改 logLevel 无法重开），min 钳制使缓存恒 ≤ INFO：生产侧 HCCL_INFO
//     （含 selector 选路命中行 "the selected algo type is ..."）恒生成并下发至桩；logLevel
//     置 0x00（DEBUG）时本值随之下探（首评前已设则缓存=0x00），保留显式调试能力。
//   ② logLevel 唯一控制控制台打印——DlogInner/DlogRecord 内部门禁（level < logLevel 不打印），
//     默认 0x03（upstream 原值，ERROR-only）；upstream 测试即依赖它：用例设 logLevel=0x01 +
//     dup2 重定向 stdout 后可在文件中检得 INFO 行（algTag 等），本桩口径与 upstream 一致。
//   ③ 选路命中记录（RecordSelectedAlgo）在 DlogInner/DlogRecord 内部打印门禁之前执行——
//     配合 ① 的 min 语义，生产侧 INFO 恒到达此处，选路命中记录无条件可用，无任何用例侧
//     logLevel 前置条件（见 DlogInner/DlogRecord）。
uint32_t logLevel = 0x03; // 唯一打印门禁：ERROR（upstream 默认原值，打印口径与 upstream 完全一致）
constexpr int TIME_FROM_1900 = 1900;
constexpr int LOG_STUB_BUFFER_SIZE = 1024;
std::map<int, std::string> LOG_LEVEL_STR_MAP
    = {{0x00, "[DEBUG]"}, {0x01, "[INFO]"}, {0x02, "[WARNING]"}, {0x03, "[ERROR]"}, {0x10, "[EVENT]"}};

// ---- 选路命中记录器（PR#3933 评审 21ec67：ST 用例"从未真正命中 RHD"类静默回退的自证数据源）----
// 生产 selector 选路完成后经 HCCL_INFO 打印 "the selected algo type is <algName>"
// （selector_engine.cc LogSelectedAlgo / execute_selector.cc 同款文案）。DlogInner/DlogRecord
// 是 ST 全部日志的门禁前汇聚点，在门禁前轻量提取 algName 存入进程级集合，供 testcase
// 断言选路命中。ST 多 rank 线程并发打印 → 互斥锁保护；同一测试进程内多 rank 各自打印，
// 记录出现过的 algName 集合（去重）；记录器为 log_stub 文件级全局，不随 SimWorld::Deinit
// 清空——断言在 Run 辅助函数（含 Deinit）返回后仍可查询。
static std::mutex g_selectedAlgoMutex;
static std::set<std::string> g_selectedAlgoNames;

static void RecordSelectedAlgo(const char* logBuffer)
{
    constexpr char ALGO_HIT_MARKER[] = "the selected algo type is";
    const char* pos = strstr(logBuffer, ALGO_HIT_MARKER);
    if (pos == nullptr) {
        return;
    }
    pos += sizeof(ALGO_HIT_MARKER) - 1;
    while (*pos == ' ') {
        ++pos;
    }
    const char* end = pos;
    while (*end != '\0' && *end != ' ' && *end != '\n' && *end != '\r') {
        ++end;
    }
    if (end == pos) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_selectedAlgoMutex);
    (void)g_selectedAlgoNames.emplace(pos, end);
}

#ifdef __cplusplus
extern "C" {
#endif

// ST 生产侧日志可见性固定 INFO（PR#3933 评审 21ec67 选路命中记录的最终契约）：
// 生产侧 HcclCheckLogLevel 经一次性缓存探测本值（src/common/log.cc g_logLevelCache，进程
// 首笔 op 日志后固化），返回 min(logLevel, INFO) 使 INFO 恒可见——selector 选路命中行
// （"the selected algo type is ..."，HCCL_INFO）无条件下发至桩的记录器，不受打印门禁
// 影响；logLevel 下探 0x00（DEBUG）时随之下探（首评前已设则缓存=0x00），保留显式调试
// 能力。控制台打印仍由 logLevel 门控（DlogInner/DlogRecord 内部，默认 0x03 ERROR-only）；
// 需要捕获完整 INFO 的用例设 logLevel=0x01（cf. upstream dpu_thread_arrangement_testcase）。
int32_t dlog_getlevel(int32_t moduleId, int32_t* enableEvent)
{
    return (logLevel < 0x01) ? static_cast<int32_t>(logLevel) : 0x01;
}

int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel) { return 1; }

void GetCurTimeStr(char* timeStr, int len)
{
    struct timeval tv;
    time_t tmpt;
    struct tm* now;

    if (timeStr == nullptr) {
        return;
    }

    if (0 > gettimeofday(&tv, nullptr)) {
        return;
    }

    tmpt = (time_t)tv.tv_sec;
    now = localtime(&tmpt);
    if (now == nullptr) {
        return;
    }

    int iLen = snprintf_s(
        timeStr, len, len, "%04d-%02d-%02d %02d:%02d:%02d.%06u", now->tm_year + TIME_FROM_1900, now->tm_mon + 1,
        now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec, (uint32_t)tv.tv_usec);
    if (iLen == -1) {
        printf("Print time failed\n");
    }
}

void DlogPrintStub(int level, char* logBuffer)
{
    std::string logLevelStr = "[INFO]";
    if (LOG_LEVEL_STR_MAP.find(level) != LOG_LEVEL_STR_MAP.end()) {
        logLevelStr = LOG_LEVEL_STR_MAP[level];
    }

    char timeBuffer[LOG_STUB_BUFFER_SIZE] = {0};
    GetCurTimeStr(timeBuffer, LOG_STUB_BUFFER_SIZE);
    printf("[%-26s][pid:%u]%s%s\n", timeBuffer, getpid(), logLevelStr.c_str(), logBuffer);
}

// 选路命中记录器查询接口（导出符号，testcase 链接桩库可见；与 log_stub 既有函数同款
// extern "C" 导出惯例）。Reset 由 fixture SetUp 调用防上一用例残留；Get 返回本测试进程
// 内出现过的 algName 集合（";" 连接，空集合返回空串），断言查集合含目标子串。
void ResetSelectedAlgoLog()
{
    std::lock_guard<std::mutex> lock(g_selectedAlgoMutex);
    g_selectedAlgoNames.clear();
}

const char* GetSelectedAlgoLog()
{
    static std::string joinedAlgoNames;
    std::lock_guard<std::mutex> lock(g_selectedAlgoMutex);
    joinedAlgoNames.clear();
    for (auto it = g_selectedAlgoNames.begin(); it != g_selectedAlgoNames.end(); ++it) {
        if (it != g_selectedAlgoNames.begin()) {
            joinedAlgoNames += ";";
        }
        joinedAlgoNames += *it;
    }
    return joinedAlgoNames.c_str();
}

void DlogInner(int moduleId, int level, const char* fmt, ...)
{
    // 定义一个缓冲区，用于存储格式化后的字符串
    char buffer[LOG_STUB_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, fmt);
    vsnprintf_s(buffer, sizeof(buffer), (sizeof(buffer) - 1), fmt, args);
    va_end(args);

    // 选路命中记录（PR#3933 评审 21ec67）：先于下方打印门禁执行；配合 dlog_getlevel 的
    // min(logLevel, 0x01) 语义，生产侧 HCCL_INFO（选路命中行）恒到达此处，选路命中记录
    // 无条件成立——无需用例侧预设 logLevel=0x01
    RecordSelectedAlgo(buffer);

    if (level < logLevel) {
        return;
    }

    DlogPrintStub(level, buffer);
}

void DlogRecord(int moduleId, int level, const char* fmt, ...)
{
    // 定义一个缓冲区，用于存储格式化后的字符串
    char buffer[LOG_STUB_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, fmt);
    vsnprintf_s(buffer, sizeof(buffer), (sizeof(buffer) - 1), fmt, args);
    va_end(args);

    // 选路命中记录（PR#3933 评审 21ec67）：先于下方打印门禁执行；配合 dlog_getlevel 的
    // min(logLevel, 0x01) 语义，生产侧 HCCL_INFO（选路命中行）恒到达此处，选路命中记录
    // 无条件成立——无需用例侧预设 logLevel=0x01
    RecordSelectedAlgo(buffer);

    if (level < logLevel) {
        return;
    }

    DlogPrintStub(level, buffer);
}

#ifdef __cplusplus
}
#endif
