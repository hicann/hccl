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

/* CANN 版本号宏，与 CMake 注入的 CANN_VERSION_NUM 配套使用。
 * CANN_VERSION(M, m, p)    -> 正式版本号 = M*10000000 + m*100000 + p*1000
 * CANN_VERSION(M, m, p, b) -> beta 版本号 = 正式号 - 200 + b (b 为 beta 子版本号, 介于上一 patch 与本 patch 之间)
 */
#define CANN_VERSION_VAL(M, m, p) ((M) * 10000000 + (m) * 100000 + (p) * 1000)
#define CANN_VERSION_3(M, m, p)    (CANN_VERSION_VAL(M, m, p))
#define CANN_VERSION_4(M, m, p, b) (CANN_VERSION_VAL(M, m, p) - 200 + (b))
#define CANN_VERSION_PICK(_1, _2, _3, _4, NAME, ...) NAME
#define CANN_VERSION(...) CANN_VERSION_PICK(__VA_ARGS__, CANN_VERSION_4, CANN_VERSION_3)(__VA_ARGS__)

#include <sys/syscall.h>
#include <unistd.h>
#include "dlog_pub.h"

#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"

#include "hccl_res.h"

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "hcomm_res_defs.h"
#endif

/* beta.1 起 hccl_types.h 已提供 HcclCommStatus，仅 < 9.1.0_beta.1 (8.5.0/9.0.0) 需要桩 */
#if CANN_VERSION_NUM < CANN_VERSION(9, 1, 0, 1)
typedef enum {
    HCCL_COMM_STATUS_READY = 0,
    HCCL_COMM_STATUS_SUSPENDING = 1,
    HCCL_COMM_STATUS_INVALID = 254,
    HCCL_COMM_STATUS_RESERVED = 255
} HcclCommStatus;
#endif

/* 9.0.0 起 hccl_types.h 已提供 ThreadHandle，仅 < 9.0.0 (8.5.x) 需要桩 */
#if CANN_VERSION_NUM < CANN_VERSION(9, 0, 0)
typedef uint64_t ThreadHandle;
#endif

#if CANN_VERSION_NUM < CANN_VERSION(9, 1, 0)

const uint32_t P2P_MAX_ARG_SIZE = 8192U;
typedef struct {
    ThreadHandle sendRecvThread;
    uint8_t opParams[P2P_MAX_ARG_SIZE];
} HcclP2pKernelParam;

typedef struct {
    void *buffer;
    uint8_t reserved[8];
    HcclCMDType cmdType;
    HcclDataType dataType;
    uint64_t count;
    uint32_t remoteRank;
    void *unfoldStream;
} HcclOpP2pDesc;

const uint32_t HCCL_OP_DESC_OP_NAME_MAX_LEN = 256;

typedef struct {
    CommAbiHeader header;
    uint32_t opDescType;
    char opName[HCCL_OP_DESC_OP_NAME_MAX_LEN];
    union {
        uint8_t raws[76];
        HcclOpP2pDesc p2p;
    };
} HcclOpDesc;

const uint32_t HCCL_OPDESC_MAGIC_WORD = 0x0f0f0f0f;
const uint32_t HCCL_OPDESC_VERSION = 1;
const uint32_t HCCL_KERNEL_SO_NAME_MAX_LEN = 256;
const uint32_t HCCL_KERNEL_FUNC_NAME_MAX_LEN = 256;

typedef struct {
    char kernelSoName[HCCL_KERNEL_SO_NAME_MAX_LEN];
    char kernelFuncName[HCCL_KERNEL_FUNC_NAME_MAX_LEN];
    void *args;
    uint32_t argSize;
} HcclKernelFuncInfo;

const uint32_t HCCL_KERNEL_LAUNCH_CFG_MAGIC_WORD = 0x0f0f0f0f;
const uint32_t HCCL_KERNEL_LAUNCH_CFG_VERSION = 1;

typedef struct {
    CommAbiHeader header;
    uint64_t timeOut;
    uint8_t reserved[104];
} HcclKernelLaunchCfg;

typedef enum {
    HCCL_COMM_STATE_PHASE_INVALID = -1,
    HCCL_COMM_STATE_PHASE_DESTROY_PRE = 0,   /* 调用通信域销毁HcclCommDestroy前 */
    HCCL_COMM_STATE_PHASE_DESTROY_POST = 1,  /* 调用通信域销毁HcclCommDestroy后 */
    HCCL_COMM_STATE_PHASE_RESUME_PRE = 2,    /* 调用step快恢恢复通信域资源HcclCommResume前 */
    HCCL_COMM_STATE_PHASE_RESUME_POST = 3    /* 调用step快恢恢复通信域资源HcclCommResume后 */
} HcclCommStatePhase;

typedef HcclResult (*HcclCommStateCallback)(HcclComm comm, HcclCommStatePhase state, void *args);
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
    type func_name(__VA_ARGS__) __attribute__((weak))

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