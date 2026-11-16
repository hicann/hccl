/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_LOG_H
#define BSR_HOST_TEST_LOG_H

// 日志与错误宏替身：保留 CHK_RET / CHK_PRT_RET / CHK_PTR_NULL 的语义（被测代码的错误分支靠它们返回
// 具体错误码，UT 断言的正是这些错误码），日志本身不输出。

#include "alg_param.h"

namespace ops_hccl {
namespace bsr_host_test {
    inline void Log(...) {}
} // namespace bsr_host_test
} // namespace ops_hccl

#define HCCL_ALG 0
#define HCCL_CONFIG_INFO(...) ops_hccl::bsr_host_test::Log(__VA_ARGS__)
#define HCCL_DEBUG(...) ops_hccl::bsr_host_test::Log(__VA_ARGS__)
#define HCCL_INFO(...) ops_hccl::bsr_host_test::Log(__VA_ARGS__)
#define HCCL_WARNING(...) ops_hccl::bsr_host_test::Log(__VA_ARGS__)
#define HCCL_ERROR(...) ops_hccl::bsr_host_test::Log(__VA_ARGS__)

#define CHK_RET(call)                             \
    do {                                          \
        const auto result = (call);               \
        if (result != HcclResult::HCCL_SUCCESS) { \
            return result;                        \
        }                                         \
    } while (false)

#define CHK_PRT_RET(condition, log, result) \
    do {                                    \
        if (condition) {                    \
            log;                            \
            return result;                  \
        }                                   \
    } while (false)

#define CHK_PTR_NULL(ptr)                  \
    do {                                   \
        if ((ptr) == nullptr) {            \
            return HcclResult::HCCL_E_PTR; \
        }                                  \
    } while (false)

#endif // BSR_HOST_TEST_LOG_H
