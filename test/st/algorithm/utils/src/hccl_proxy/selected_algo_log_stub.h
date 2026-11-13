/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECTED_ALGO_LOG_STUB_H
#define SELECTED_ALGO_LOG_STUB_H

#include <cstring>

#include "gtest/gtest.h"

// log_stub.cc 选路命中记录器的查询接口（PR#3933 评审 21ec67：ST 用例选路命中自证断言）。
// 记录器无条件可用：dlog_getlevel 的 min(logLevel, 0x01) 语义保证生产侧 HCCL_INFO 恒
// 下发至 DlogInner/DlogRecord（打印门禁前汇聚点），选路命中行（"the selected algo type
// is <algName>"）恒被提取存入本测试进程级集合（多 rank 线程并发安全）——无任何用例侧
// logLevel 前置条件。不随 SimWorld::Deinit 清空，断言可在 Run 辅助函数返回后查询。
// 符号随 hccl-model-proxy 桩库导出（extern "C"）。
#ifdef __cplusplus
extern "C" {
#endif

// 清空已记录的选路命中集合（fixture SetUp 调用，防上一用例残留）
void ResetSelectedAlgoLog();

// 返回本测试进程内出现过的 algName 集合（";" 连接；空集合返回空串）
const char* GetSelectedAlgoLog();

#ifdef __cplusplus
}
#endif

// 选路命中断言（PR#3933 评审 21ec67）：期望子串（如 "RHD"）须出现在已记录的某个选中
// algName 中——强制选路用例静默回退（如 selected=AicpuAllGatherSoleNHR）时断言失败，
// 并在失败信息中打印实际命中算法名，使"从未真正命中 RHD"可见（dev-r13 pow2 实证）。
// 记录无条件可用（依赖 dlog_getlevel 的 min 语义保证 INFO 恒下发，无任何用例侧 logLevel
// 前置条件）：失败时打出 (none) 仅可能来自记录链路断裂，空集合同样断言失败。
static inline void ExpectAlgoHit(const char* expectedSubstr)
{
    const char* recordedAlgoNames = GetSelectedAlgoLog();
    EXPECT_TRUE(strstr(recordedAlgoNames, expectedSubstr) != nullptr)
        << "selected algo never hit [" << expectedSubstr << "], actual recorded algName(s): ["
        << ((recordedAlgoNames[0] == '\0') ? "(none)" : recordedAlgoNames) << "]";
}

#endif // SELECTED_ALGO_LOG_STUB_H
