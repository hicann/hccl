/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log.h"

thread_local bool g_hcclErrToWarn = false;

bool HcclCheckLogLevel(int logType, int moduleId)
{
    return (CheckLogLevel(moduleId, logType) == 1);
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