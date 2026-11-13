/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Tuner 插件自包含日志宏（不走宿主 logFunction 回调，printf 直出）。
 * 裁剪自 examples/05_custom_ops_allgather/inc/log.h：仅保留四级日志宏。
 * 编译期 LOG_LEVEL 控制级别；HOST_COMPILE 守卫（host 侧 printf，其余空宏）。 */

#ifndef HCCL_TUNER_PLUGIN_LOG_H
#define HCCL_TUNER_PLUGIN_LOG_H

#include <cstdio>

typedef enum {
    LOG_LEVEL_DEBUG = 0,   // DEBUG级别
    LOG_LEVEL_INFO = 1,    // INFO级别
    LOG_LEVEL_WARNING = 2, // WARNING级别
    LOG_LEVEL_ERROR = 3,   // ERROR级别
    LOG_LEVEL_NONE = 4     // 关闭所有日志
} LogLevel;

#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_LEVEL_ERROR // 默认日志级别为：ERROR
#endif

#ifdef HOST_COMPILE

#define HCCL_DEBUG(format, ...)                                                                    \
    do {                                                                                           \
        if (LOG_LEVEL <= LOG_LEVEL_DEBUG) {                                                        \
            printf("[DEBUG][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                                          \
    } while (0)

#define HCCL_INFO(format, ...)                                                                    \
    do {                                                                                          \
        if (LOG_LEVEL <= LOG_LEVEL_INFO) {                                                        \
            printf("[INFO][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                                         \
    } while (0)

#define HCCL_WARNING(format, ...)                                                                 \
    do {                                                                                          \
        if (LOG_LEVEL <= LOG_LEVEL_WARNING) {                                                     \
            printf("[WARN][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                                         \
    } while (0)

#define HCCL_ERROR(format, ...)                                                                    \
    do {                                                                                           \
        if (LOG_LEVEL <= LOG_LEVEL_ERROR) {                                                        \
            printf("[ERROR][%s][%s:%d]" format "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                                          \
    } while (0)

#else

#define HCCL_DEBUG(format, ...) \
    do {                        \
    } while (0)
#define HCCL_INFO(format, ...) \
    do {                       \
    } while (0)
#define HCCL_WARNING(format, ...) \
    do {                          \
    } while (0)
#define HCCL_ERROR(format, ...) \
    do {                        \
    } while (0)

#endif // HOST_COMPILE

#endif // HCCL_TUNER_PLUGIN_LOG_H
