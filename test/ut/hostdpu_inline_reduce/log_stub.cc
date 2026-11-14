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

/* HcclCheckLogLevel 返回 false 使 HCCL_INFO/HCCL_WARNING 日志体不执行；
 * dlsym 层（hcomm_primitives_dl.cc）的 HCCL_LOG_PRINT 直调 DlogRecord，此处提供空实现（覆盖弱符号）。 */
bool HcclCheckLogLevel(int logType, int moduleId) { return false; }
bool IsErrorToWarn() { return false; }

extern "C" void DlogRecord(int32_t moduleId, int32_t level, const char* fmt, ...) {}
