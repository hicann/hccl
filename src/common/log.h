/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOG_H
#define LOG_H

#include <dlog_pub.h>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <securec.h>
#include <iostream>
#include <sstream>
#include <sys/syscall.h>
#include <unistd.h>
#include <string>
#include "dlog_pub.h"
#ifndef T_DESC
#define T_DESC(_msg, _y) ((_y) ? true : false)
#endif

#if T_DESC("日志处理适配", true)

enum class HcclSubModuleID {
    LOG_SUB_MODULE_ID_HCCL = 0,
    LOG_SUB_MODULE_ID_HCOM = 1,
    LOG_SUB_MODULE_ID_CLTM = 2,
    LOG_SUB_MODULE_ID_CUSTOM_OP = 3
};

/* 设置日志的commid和rankid */
#ifndef LIKELY
#define LIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 1)))
#define UNLIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 0)))
#endif

/* 每一条日志的长度,超过该长度会申请堆内存 */
constexpr s32 LOG_TMPBUF_SIZE = 512;

#define LOG_FUNC(module, level, fmt, ...) do { \
    DlogRecord(module, level, fmt, ##__VA_ARGS__); \
} while (0)

bool HcclCheckLogLevel(int logType, int moduleId = HCCL);

void SetErrToWarnSwitch(bool flag); // 设置True时修改日志级别：ERROR -> RUN_WARNING，设置False恢复日志级别

bool IsErrorToWarn();

#define HCCL_LOG_DEBUG DLOG_DEBUG
#define HCCL_LOG_INFO  DLOG_INFO
#define HCCL_LOG_WARN  DLOG_WARN
#define HCCL_LOG_ERROR DLOG_ERROR

#define HCCL_LOG_MASK (static_cast<int32_t>(HCCL) | static_cast<int32_t>(RUN_LOG_MASK))

#define HCCL_LOG_PRINT(moduleId, logType, format, ...) do { \
    LOG_FUNC(moduleId, logType, "[%s:%d] [%u]" format, __FILE__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
} while(0)

#define HCCL_ERROR_LOG_PRINT(format, ...) do { \
    if (IsErrorToWarn()) { \
        LOG_FUNC(HCCL_LOG_MASK, HCCL_LOG_WARN, "[%s:%d] [%u]ErrToWarn: " format, \
                 __FILE__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
    } else { \
        LOG_FUNC(HCCL, HCCL_LOG_ERROR, "[%s:%d] [%u]" format, \
                 __FILE__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
    } \
} while(0)

#define HCCL_RUN_LOG_PRINT(format, ...) do { \
    LOG_FUNC(HCCL_LOG_MASK, HCCL_LOG_INFO, "[%s:%d] [%u]" format, \
             __FILE__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
} while(0)

// 错误码
const u64 SYSTEM_RESERVE_ERROR = 0;
const u64 HCCL_MODULE_ID = 5;

/* 预定义日志宏, 便于使用 */
#define HCCL_DEBUG(format, ...) do { \
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_DEBUG))) { \
        HCCL_LOG_PRINT(HCCL, HCCL_LOG_DEBUG, format, ##__VA_ARGS__); \
    } \
} while(0)

#define HCCL_INFO(format, ...) do { \
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) { \
        HCCL_LOG_PRINT(HCCL, HCCL_LOG_INFO, format, ##__VA_ARGS__); \
    } \
} while(0)

#define HCCL_WARNING(format, ...) do { \
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_WARN))) { \
        HCCL_LOG_PRINT(HCCL, HCCL_LOG_WARN, format, ##__VA_ARGS__); \
    } \
} while(0)

#define HCCL_ERROR(format, ...) do { \
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_ERROR))) { \
        HCCL_ERROR_LOG_PRINT(format, ##__VA_ARGS__); \
    } \
} while(0)

/* 运行日志 */
#define HCCL_RUN_INFO(format, ...) do { \
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO, HCCL_LOG_MASK))) { \
        HCCL_RUN_LOG_PRINT(format, ##__VA_ARGS__); \
    } \
} while(0)

#define HCCL_RUN_WARNING(format, ...) do { \
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_WARN, HCCL_LOG_MASK))) { \
        HCCL_LOG_PRINT(HCCL_LOG_MASK, HCCL_LOG_WARN, format, ##__VA_ARGS__); \
    } \
} while(0)

#define HCCL_USER_CRITICAL_LOG(format, ...) do { \
    if (LIKELY(HcclCheckLogLevel(HCCL_LOG_INFO, HCCL_LOG_MASK))) { \
        HCCL_RUN_LOG_PRINT(format, ##__VA_ARGS__); \
    } \
} while(0)

#define HCCL_ERROR_CODE(error) ((SYSTEM_RESERVE_ERROR << 32) + (HCCL_MODULE_ID << 24) + \
    ((static_cast<u64>(HcclSubModuleID::LOG_SUB_MODULE_ID_HCCL)) << 16) + static_cast<u64>(error))
#define HCOM_ERROR_CODE(error) ((SYSTEM_RESERVE_ERROR << 32) + (HCCL_MODULE_ID << 24) + \
    ((static_cast<u64>(HcclSubModuleID::LOG_SUB_MODULE_ID_HCOM)) << 16) + static_cast<u64>(error))
#endif

#if T_DESC("公共代码宏", true)

// 检查C++11的智能指针, 若为空, 则记录日志, 并返回错误
#define CHK_SMART_PTR_NULL(smart_ptr)                                                            \
    do {                                                                                                    \
        if (UNLIKELY(!(smart_ptr))) {                                                   \
            HCCL_ERROR("[%s]errNo[0x%016llx] ptr [%s] is nullptr, return HCCL_E_PTR", \
                __func__, HCCL_ERROR_CODE(HCCL_E_PTR), \
                #smart_ptr);                                                                                \
            return HCCL_E_PTR;                                                                              \
        }                                                                                                   \
    } while (0)

// 检查C++11的智能指针, 若为空, 则记录日志, 并返回
#define CHK_SMART_PTR_RET_NULL(smart_ptr)                       \
    do {                                                        \
        if (UNLIKELY(!(smart_ptr))) {                           \
            HCCL_ERROR("[%s]errNo[0x%016llx]smart_ptr is nullptr.",   \
            __func__, HCCL_ERROR_CODE(HCCL_E_PTR));                       \
            return;                                             \
        }                                                       \
    } while (0)

/* 检查指针, 若指针为NULL, 则记录日志, 并返回错误 */
#define CHK_PTR_NULL(ptr)                                                                               \
    do {                                                                                                           \
        if (UNLIKELY((ptr) == nullptr)) {                  \
            HCCL_ERROR("[%s]errNo[0x%016llx]ptr [%s] is nullptr, return HCCL_E_PTR", \
            __func__, HCCL_ERROR_CODE(HCCL_E_PTR), #ptr); \
            return HCCL_E_PTR;                                                                                     \
        }                                                                                                          \
    } while (0)

/* 检查函数返回值, 记录指定日志, 并返回指定错误码 */
#define CHK_PRT_RET(result, exeLog, retCode) \
    do {                                      \
        if (UNLIKELY(result)) {                         \
            exeLog;                           \
            return retCode;                   \
        }                                     \
    } while (0)

/* 检查函数返回值, 记录指定日志, 函数不返回 */
#define CHK_PRT_CONT(result, exeLog) \
    do {                                      \
        if (UNLIKELY(result)) {               \
            exeLog;                           \
        }                                     \
    } while (0)

/* 检查函数返回值, 并返回指定错误码 */
#define CHK_RET(call)                                 \
    do {                                              \
        HcclResult hcclRet = call;                        \
        if (UNLIKELY(hcclRet != HCCL_SUCCESS)) {                    \
            if (hcclRet == HCCL_E_AGAIN) {                \
                HCCL_WARNING("[%s]call trace: hcclRet -> %d", __func__, hcclRet); \
            } else {                                  \
                HCCL_ERROR("[%s]call trace: hcclRet -> %d", __func__, hcclRet); \
            }                                         \
            return hcclRet;                               \
        }                                             \
    } while (0)

/* 检查函数返回值, 返错时打印函数名及通信域标识 */
#define CHK_RET_AND_PRINT_IDE(call, identifier)         \
    do {                                              \
        HcclResult hcclRet = call;                        \
        if (UNLIKELY(hcclRet != HCCL_SUCCESS)) {                    \
            HCCL_RUN_INFO("[HCCL_TRACE]%s identifier[%s]", __func__, identifier); \
            if (hcclRet == HCCL_E_AGAIN) {                \
                HCCL_WARNING("[%s]call trace: hcclRet -> %d", __func__, hcclRet); \
            } else {                                  \
                HCCL_ERROR("[%s]call trace: hcclRet -> %d", __func__, hcclRet); \
            }                                         \
            return hcclRet;                               \
        }                                             \
    } while (0)

/* 检查函数返回值, 并返回空 */
#define CHK_RET_NULL(call)                                 \
    do {                                              \
        HcclResult ret = call;                        \
        if (UNLIKELY(ret != HCCL_SUCCESS)) {                    \
            if (ret == HCCL_E_AGAIN) {                \
                HCCL_WARNING("[%s]call trace: ret -> %d", __func__, ret); \
            } else {                                  \
                HCCL_ERROR("[%s]call trace: ret -> %d", __func__, ret); \
            }                                         \
            return;                               \
        }                                             \
    } while (0)

/* 检查函数返回值, 打印错误码, 函数不返回 */
#define CHK_PRT(call)                                 \
    do {                                              \
        HcclResult ret = call;                        \
        if (UNLIKELY(ret != HCCL_SUCCESS)) {                    \
            if (ret == HCCL_E_AGAIN) {                \
                HCCL_WARNING("[%s]call trace: ret -> %d", __func__, ret); \
            } else {                                  \
                HCCL_ERROR("[%s]call trace: ret -> %d", __func__, ret); \
            }                                         \
        }                                             \
    } while (0)

/* 检查函数返回值, 并返回HCCL_E_INTERNAL错误码 */
#define CHK_SAFETY_FUNC_RET(call)                                 \
    do {                                              \
        s32 ret = call;                        \
        if (UNLIKELY(ret != EOK)) {                    \
            HCCL_ERROR("[%s]call trace: safety func err ret -> %d", __func__, ret); \
            return HCCL_E_INTERNAL;                               \
        }                                             \
    } while (0)

/* 检查result. 若错误, 则设置错误并break */
#define CHK_PRT_BREAK(result, exeLog, exeCmd) \
    if (UNLIKELY(result)) {                              \
        exeLog;                                \
        exeCmd;                                \
        break;                                 \
    }

#define EXECEPTION_CATCH(expression, retExp)                     \
    do {                                                         \
        try {                                                    \
            expression;                                          \
        } catch (std::exception& e) {                            \
            HCCL_ERROR("[%s]Failed, exception caught:%s", __func__, e.what()); \
            retExp;                                              \
        }                                                        \
    } while (0)

/* 若new失败, 则捕获异常返回 */
#ifndef NEW_NOTHROW
#define NEW_NOTHROW(pointer, constructor, retExp)        \
    do {                                                 \
        pointer = new(std::nothrow) constructor;         \
        if (pointer == nullptr) {                        \
            HCCL_ERROR("[%s]Memory application failed.", __func__);     \
            retExp;                                      \
        }                                                \
    } while (0)
#endif

template <typename T>
void ArrayToStringAndPrint(T arr[], int size, const char *name)
{
    if (arr == nullptr) {
        HCCL_ERROR("Array Data, %s: data is nullptr", name);
        return;
    }

    constexpr int NUM_IN_ROW = 50;
    std::stringstream ss;
    for (int i = 0; i < size; i++) {
        ss << arr[i] << " ";
        if ((i + 1) % NUM_IN_ROW == 0) {
            HCCL_DEBUG("Array Data, %s: %s", name, ss.str().c_str());
            ss.clear();
            ss.str(std::string());
        }
    }

    HCCL_DEBUG("Array Data, %s: %s", name, ss.str().c_str());
}

#ifdef DUMP_DATA
#define PRINT_ARRAY(arr, cnt, name) \
    ArrayToStringAndPrint(arr, cnt, name)

#else
#define PRINT_ARRAY(arr, cnt, name)
#endif

#endif

#endif // LOG_H