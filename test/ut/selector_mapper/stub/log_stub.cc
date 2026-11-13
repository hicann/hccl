/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* AlgoNameMapper UT 桩——日志 + 维度表访问接口。
 * 日志桩：HcclCheckLogLevel 返回 false 使全部日志为空操作。
 * 维度表桩：与 alg_parse.cc 的 ENGINE_TYPES/EXECUTOR_TYPES 同数据，但不依赖 alg_parse.cc。 */

#include "log.h"
#include "alg_parse.h"

/* ===== 日志桩 ===== */
bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return false;
}

bool IsErrorToWarn() { return false; }

/* ===== 维度表桩（与 alg_parse.cc 同数据）===== */
namespace ops_hccl {

/* 与 alg_parse.cc ENGINE_TYPES 同数据 */
static const AlgoDimEntry g_engineEntries[] = {
    {"aicpu", "Aicpu"}, {"aiv", "Aiv"}, {"dpu", "Dpu"}, {"ccums", "CcuMS"}, {"ccusched", "CcuSched"},
};

/* 与 alg_parse.cc EXECUTOR_TYPES 同数据（常用子集，足够覆盖测试用例） */
static const AlgoDimEntry g_executorEntries[] = {
    {"sole", "Sole"},
    {"sequence", "Sequence"},
    {"parallel", "Parallel"},
    {"concur", "Concur"},
};

const AlgoDimEntry* GetAlgoEngines(int& count)
{
    count = static_cast<int>(sizeof(g_engineEntries) / sizeof(g_engineEntries[0]));
    return g_engineEntries;
}

const AlgoDimEntry* GetAlgoExecutors(int& count)
{
    count = static_cast<int>(sizeof(g_executorEntries) / sizeof(g_executorEntries[0]));
    return g_executorEntries;
}

} /* namespace ops_hccl */
