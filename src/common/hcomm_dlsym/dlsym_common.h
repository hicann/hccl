/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DLSYM_COMMON_H
#define DLSYM_COMMON_H

#include <sys/syscall.h>
#include <unistd.h>
#include "dlog_pub.h"

#include "hccl/hccl_types.h"

/* 8.5.0 桩: HcclCommStatus (来自 hccl_types.h，9.0.0 新增) */
#if CANN_VERSION_NUM < 90000000
typedef enum {
    HCCL_COMM_STATUS_READY = 0,
    HCCL_COMM_STATUS_SUSPENDING = 1,
    HCCL_COMM_STATUS_INVALID = 254,
    HCCL_COMM_STATUS_RESERVED = 255
} HcclCommStatus;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define HCCL_LOG_DEBUG DLOG_DEBUG
#define HCCL_LOG_INFO  DLOG_INFO
#define HCCL_LOG_WARN  DLOG_WARN
#define HCCL_LOG_ERROR DLOG_ERROR

#define LOG_FUNC(module, level, fmt, ...) do { \
    DlogRecord(module, level, fmt, ##__VA_ARGS__); \
} while (0)

#define HCCL_LOG_PRINT(moduleId, logType, format, ...) do { \
    LOG_FUNC(moduleId, logType, "[%s:%d] [%u]" format, __FILE__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
} while(0)

#define HCCL_RUN_LOG_PRINT(format, ...) do { \
    LOG_FUNC(HCCL_LOG_MASK, HCCL_LOG_INFO, "[%s:%d] [%u]" format, \
             __FILE__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
} while(0)

/* 预定义日志宏, 便于使用 */
#define HCCL_COMPAT_DEBUG(format, ...) do { \
    HCCL_LOG_PRINT(HCCL, HCCL_LOG_DEBUG, format, ##__VA_ARGS__); \
} while(0)

#define HCCL_COMPAT_ERROR(format, ...) do { \
    HCCL_LOG_PRINT(HCCL, HCCL_LOG_ERROR, format, ##__VA_ARGS__); \
} while(0)

#define DECL_WEAK_FUNC(type, func_name, ...) \
    type func_name(__VA_ARGS__) __attribute__((weak));

#define DEFINE_WEAK_FUNC(type, func_name, ...) \
    static bool g_##func_name##Supported = false; \
    extern "C" bool HcommIsSupport##func_name(void) { \
        return g_##func_name##Supported; \
    } \
    type func_name(__VA_ARGS__) __attribute__((weak)); \
    type func_name(__VA_ARGS__) \
    { \
        HCCL_COMPAT_ERROR("[HcclWrapper] %s not supported", __func__); \
        return (type)(-1); \
    }

#define DECL_SUPPORT_FLAG(func_name) \
    extern "C" bool HcommIsSupport##func_name(void)

#define INIT_SUPPORT_FLAG(handle, func_name) \
    do { \
        void *ptr = (void *)dlsym(handle, #func_name); \
        if (ptr == nullptr) { \
            g_##func_name##Supported = false; \
            HCCL_COMPAT_DEBUG("[HcclWrapper] %s not supported", #func_name); \
        } else { \
            g_##func_name##Supported = true; \
        } \
    } while(0)


#ifdef __cplusplus
}
#endif

#endif // DLSYM_COMMON_H