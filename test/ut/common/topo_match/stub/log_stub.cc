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

// UT 桩：log.h 中这两个函数为全局（非 namespace）声明，matcher 日志宏依赖之。
// HcclCheckLogLevel 返回 false 使日志体不执行。
bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return false;
}
bool IsErrorToWarn() { return false; }
