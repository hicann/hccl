/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <execinfo.h>
#include "hccl_exception.h"
#include "exception_util.h"
#include "null_ptr_exception.h"

namespace HcclSim {
const char *HcclException::what() const noexcept
{
    return errorMsg.c_str();
}

HcclResult HcclException::GetErrorCode() const
{
    return ExceptionInfo::GetErrorCode(exceptionType);
}

constexpr int BACKTRACE_DEPTH = 15;
void          HcclException::StoreBackTrace()
{
    void  *array[BACKTRACE_DEPTH];
    char **callBackStrings;
    int    size     = backtrace(array, BACKTRACE_DEPTH);
    callBackStrings = backtrace_symbols(array, size);
    if (callBackStrings == nullptr) {
        std::string msg = StringFormat("[%s] callBackStrings Get nullptr", __func__);
        THROW<NullPtrException>(msg);
    }
    for (auto i = 0; i < size; ++i) {
        backtraceStrings.emplace_back(callBackStrings[i]);
    }
    free(callBackStrings);
}

HcclException::HcclException(const ExceptionType &exceptionType, const std::string &userDefinedMsg)
    : exceptionType(exceptionType), userDefinedMsg(userDefinedMsg),
      errorMsg(ExceptionInfo::GetErrorMsg(exceptionType) + userDefinedMsg)
{
    StoreBackTrace();
}

std::vector<std::string> HcclException::GetBackTraceStrings() const
{
    return backtraceStrings;
}
} // namespace Hccl