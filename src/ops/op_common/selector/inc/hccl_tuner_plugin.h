/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TUNER_PLUGIN_H_
#define HCCL_TUNER_PLUGIN_H_

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===== 通信域信息（per-comm，init 时传入）===== */

typedef struct {
    uint32_t nRanks;
    uint32_t nNpusPerServer;
    uint32_t nServers;
    uint32_t nPods;
    uint32_t nSuperPods;
    const char* commName;
    uint64_t bufferSize;
    uint32_t structSize;
} hcclTunerCommInfo_t;

/* ===== 集合通信调用信息（per-op，getCollInfo 时传入）===== */

typedef struct {
    HcclCMDType collType;
    size_t nBytes;
    HcclDataType dataType;
    uint32_t structSize;
} hcclTunerCollInfo_t;

/* ===== 算法条目（命名 cost table）===== */

typedef struct {
    const char* algName;      /* "AicpuAllReduceSoleMeshOneShot" */
    const char* engineName;   /* "aicpu" — Enrich 填充，插件只读 */
    const char* executorName; /* "sole"  — Enrich 填充，插件只读 */
    const char* templateName; /* "meshoneshot" — Enrich 填充，插件只读 */
    float cost;               /* 可修改: <0=禁用, 0=偏好, >0=覆盖, 不改=用CostModel值 */
    uint32_t structSize;
} hcclTunerAlgoEntry_t;

/* ===== Host 函数表 ===== */

typedef void (*hcclTunerLogFn_t)(int level, const char* file, int line, const char* fmt, ...);

typedef struct {
    HcclResult (*ctxCreate)(HcclComm comm, const char* ctxTag, uint64_t size, void** ctx);
    HcclResult (*ctxGet)(HcclComm comm, const char* ctxTag, void** ctx, uint64_t* size);
    HcclResult (*ctxDestroy)(HcclComm comm, const char* ctxTag);
    hcclTunerLogFn_t logFunction;
    uint32_t structSize;
} hcclTunerHostFunctions_t;

/* ===== 回调函数签名 ===== */

typedef HcclResult (*hcclTunerInit_t)(
    HcclComm comm, const hcclTunerCommInfo_t* commInfo, const hcclTunerHostFunctions_t* hostFuncs);

typedef HcclResult (*hcclTunerGetCollInfo_t)(
    HcclComm comm, const hcclTunerCollInfo_t* collInfo, hcclTunerAlgoEntry_t* algoEntries, int algoCount,
    int* matched); /* 插件设 *matched=1=命中规则并修改 cost, 0=未命中（HCCL 核心初始化为 0） */

/* ===== 函数表（V1）===== */

typedef struct {
    hcclTunerInit_t init;
    hcclTunerGetCollInfo_t getCollInfo;
    uint32_t structSize; /* HCCL 设值，plugin 据此判断缓冲区大小（ABI 兼容） */
} hcclTunerFuncs_v1_t;

/* ===== 入口符号（插件 .so 导出，符号名带版本号）===== */

extern hcclTunerFuncs_v1_t hcclTunerPlugin_v1;

/* ===== 日志级别 ===== */

#define HCCL_TUNER_LOG_ERROR 0
#define HCCL_TUNER_LOG_WARN 1
#define HCCL_TUNER_LOG_INFO 2
#define HCCL_TUNER_LOG_DEBUG 3

#ifdef __cplusplus
}
#endif

#endif /* HCCL_TUNER_PLUGIN_H_ */
