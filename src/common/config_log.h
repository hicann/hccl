/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONFIG_LOG_H
#define CONFIG_LOG_H

#include "log.h"

namespace ops_hccl {
constexpr u64 HCCL_ALG = 0x1ULL << 0;
constexpr u64 HCCL_TASK = 0x1ULL << 1;
constexpr u64 HCCL_RES = 0x1ULL << 2;
constexpr u64 HCCL_AIV_OPS_EXC = 0x1ULL << 3;

u64 GetDebugConfig();

HcclResult InitDebugConfigByEnv();

// config要求传入宏名字作为日志打印关键字，不可以传入其他变量或常量
#define HCCL_CONFIG_INFO(config, format,...) do {                                             \
    if (UNLIKELY(ops_hccl::GetDebugConfig() & config)) {                        \
        const char* configName = #config;                                                     \
        LOG_FUNC(HCCL | RUN_LOG_MASK, HCCL_LOG_INFO, "[%s:%d] [%u] [%s]: " format,            \
            __FILE__, __LINE__, syscall(SYS_gettid), configName, ##__VA_ARGS__);              \
    } else if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {                                  \
        HCCL_LOG_PRINT(HCCL, HCCL_LOG_INFO, format, ##__VA_ARGS__);                           \
    }                                                                                         \
} while(0)

#define HCCL_CONFIG_DEBUG(config, format,...) do {                                            \
    if (UNLIKELY(ops_hccl::GetDebugConfig() & config)) {                        \
        const char* configName = #config;                                                     \
        LOG_FUNC(HCCL | RUN_LOG_MASK, HCCL_LOG_INFO, "[%s:%d] [%u] [%s]: " format,            \
            __FILE__, __LINE__, syscall(SYS_gettid), configName, ##__VA_ARGS__);              \
    } else if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_DEBUG))) {                                 \
        HCCL_LOG_PRINT(HCCL, HCCL_LOG_DEBUG, format, ##__VA_ARGS__);                          \
    }                                                                                         \
} while(0)

}   // namespace ops_hccl
#endif