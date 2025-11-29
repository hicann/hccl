/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXCEPTION_UTIL_H
#define EXCEPTION_UTIL_H

#include <string>
#include <exception>
#include "log.h"
#include "string_util.h"
#include "hccl_exception.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"
#include "not_support_exception.h"
#include "null_ptr_exception.h"
#include "timeout_exception.h"

#define CHK_RET_THROW(EXCEPTION, MSG, expr)                                                                            \
    do {                                                                                                               \
        auto ret = (expr);                                                                                             \
        if (ret != HcclResult::HCCL_SUCCESS) {                                                                         \
            THROW<EXCEPTION>(MSG);                                                                                     \
        }                                                                                                              \
    } while (0)

#define CHK_PRT_THROW(expr, exeLog, EXCEPTION, MSG)                                                                    \
    do {                                                                                                               \
        if (UNLIKELY(expr)) {                                                                                          \
            exeLog;                                                                                                    \
            throw EXCEPTION(MSG);                                                                                      \
        }                                                                                                              \
    } while (0)

template <typename EXCEPTION> inline void THROW(const std::string &msg)
{
    HCCL_ERROR("%s", msg.c_str());
    throw EXCEPTION(msg);
}

template <typename EXCEPTION, typename... Args> inline void THROW(const char *format, Args... args)
{
    auto msg = StringFormat(format, args...);
    HCCL_ERROR("%s", msg.c_str());
    throw EXCEPTION(msg);
}

#endif // EXCEPTION_UTIL_H