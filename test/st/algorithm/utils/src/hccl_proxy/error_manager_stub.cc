/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstdarg>
#include "base/err_mgr.h"

namespace error_message {
int32_t RegisterFormatErrorMessage(const char* error_msg, size_t error_msg_len) { return 0; }

int32_t ReportInnerErrMsg(
    const char* file_name, const char* func, uint32_t line, const char* error_code, const char* format, ...)
{
    return 0;
}

int32_t ReportPredefinedErrMsg(
    const char* error_code, const std::vector<const char*>& key, const std::vector<const char*>& value)
{
    return 0;
}

// libhccl.so中ccu_fallback协商worker线程透传ErrorMgr上下文时引用Get/SetErrMgrContext，
// ST不链接liberror_manager，此处打桩：Get返回零值上下文，Set为空实现
ErrorManagerContext GetErrMgrContext() { return {}; }

void SetErrMgrContext(ErrorManagerContext error_context) { (void)error_context; }
} // namespace error_message
