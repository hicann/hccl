/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TUNER_PLUGIN_H_
#define HCCL_TUNER_PLUGIN_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hccl/hccl_types.h" /* HcclResult, HcclComm, HcclCMDType, HcclDataType, HcclReduceOp,
                                 HCCL_SUCCESS, HCCL_E_PTR
                                 注意：hccl_types.h 含 const 变量做数组尺寸（C 语言非常量表达式），
                                 纯 C 编译器无法通过——插件当前须以 C++ 编译（extern "C" 链接）。 */

#ifdef __cplusplus
extern "C" {
#endif

/* ===== ABI header（16 字节锚点：布局永不变，只可改名不可增删改类型）=====
 * version   : 结构体版本号，尾部追加字段时递增
 * magicWord : 结构体布局指纹，仅不兼容变更（改字段类型/顺序/删字段）时更换
 * size      : sizeof(本结构体)，运行时探查对方是否含新字段
 * 宿主校验策略（区间校验）：version > 期望 → 拒绝（旧宿主不认新插件尾部字段）；
 * version < 期望 → WARN 放行（新宿主兼容旧插件，只消费已知前缀字段）。 */
typedef struct {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
} HcclAbiHeader;

/* ===== 各结构体独立 MAGIC_WORD / VERSION（宏定义，C 插件可用于静态初始化）===== */
#define HCCL_TUNER_TOPO_INFO_MAGIC_WORD 0x0f0f0f31u
#define HCCL_TUNER_TOPO_INFO_VERSION 1u
#define HCCL_TUNER_COMM_INFO_MAGIC_WORD 0x0f0f0f32u
#define HCCL_TUNER_COMM_INFO_VERSION 1u
#define HCCL_TUNER_COLL_OP_INFO_MAGIC_WORD 0x0f0f0f33u
#define HCCL_TUNER_COLL_OP_INFO_VERSION 1u
#define HCCL_TUNER_ALGO_COST_ENTRY_MAGIC_WORD 0x0f0f0f34u
#define HCCL_TUNER_ALGO_COST_ENTRY_VERSION 1u
#define HCCL_TUNER_PLUGIN_OPS_MAGIC_WORD 0x0f0f0f35u
#define HCCL_TUNER_PLUGIN_OPS_VERSION 1u

/* 公共初始化：设 header 四字段（各具名 Init 负责 memset 后调用） */
static inline HcclResult HcclAbiHeaderInit(HcclAbiHeader* header, uint32_t magicWord, uint32_t version, uint32_t size)
{
    if (header == NULL) {
        return HCCL_E_PTR;
    }
    header->version = version;
    header->magicWord = magicWord;
    header->size = size;
    header->reserved = 0u;
    return HCCL_SUCCESS;
}

/* ===== 拓扑信息（per-comm，Create 时传入；仅纯拓扑）===== */
typedef struct {
    HcclAbiHeader header;
    uint32_t nRanks;
    uint32_t nNpusPerServer;
    uint32_t nServers;
    uint32_t nPods;
    uint32_t nSuperPods;
} HcclTunerTopoInfo;

static inline HcclResult HcclTunerTopoInfoInit(HcclTunerTopoInfo* topoInfo)
{
    if (topoInfo == NULL) {
        return HCCL_E_PTR;
    }
    (void)memset(topoInfo, 0, sizeof(HcclTunerTopoInfo));
    return HcclAbiHeaderInit(
        &topoInfo->header, HCCL_TUNER_TOPO_INFO_MAGIC_WORD, HCCL_TUNER_TOPO_INFO_VERSION, sizeof(HcclTunerTopoInfo));
}

/* ===== 通信域信息（per-comm，Create 时传入）===== */
typedef struct {
    HcclAbiHeader header;
    const char* commName; /* 宿主持有，插件如需持久化须自行拷贝 */
    uint64_t bufferSize;
} HcclTunerCommInfo;

static inline HcclResult HcclTunerCommInfoInit(HcclTunerCommInfo* commInfo)
{
    if (commInfo == NULL) {
        return HCCL_E_PTR;
    }
    (void)memset(commInfo, 0, sizeof(HcclTunerCommInfo));
    return HcclAbiHeaderInit(
        &commInfo->header, HCCL_TUNER_COMM_INFO_MAGIC_WORD, HCCL_TUNER_COMM_INFO_VERSION, sizeof(HcclTunerCommInfo));
}

/* ===== 集合通信调用信息（per-op，UpdateAlgoCostTable 时传入）===== */
typedef struct {
    HcclAbiHeader header;
    uint64_t nBytes;
    HcclCMDType collType;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
} HcclTunerCollOpInfo;

static inline HcclResult HcclTunerCollOpInfoInit(HcclTunerCollOpInfo* collOpInfo)
{
    if (collOpInfo == NULL) {
        return HCCL_E_PTR;
    }
    (void)memset(collOpInfo, 0, sizeof(HcclTunerCollOpInfo));
    return HcclAbiHeaderInit(
        &collOpInfo->header, HCCL_TUNER_COLL_OP_INFO_MAGIC_WORD, HCCL_TUNER_COLL_OP_INFO_VERSION,
        sizeof(HcclTunerCollOpInfo));
}

/* ===== 算法代价条目（cost table 元素）=====
 * 数组遍历约定：步长取 entries[0].header.size（运行时），不得用编译期 sizeof 下标。
 * header 由宿主在 cost table 物化时对每个条目填充。
 * 布局说明：cost 放尾部，其后 4 字节为对齐填充——未来追加 4 字节字段可免费复用。 */
typedef struct {
    HcclAbiHeader header;
    const char* algName;      /* "AicpuAllReduceSoleMeshOneShot" */
    const char* engineName;   /* "aicpu" -- 宿主物化填充，插件只读 */
    const char* executorName; /* "sole" -- 宿主物化填充，插件只读 */
    const char* templateName; /* "meshoneshot" -- 宿主物化填充，插件只读 */
    float cost;               /* 可修改: <0=禁用, 0=偏好, >0=覆盖, 不改=用 CostModel 值 */
} HcclTunerAlgoCostEntry;

/* 注意：本 Init 会 memset 整个条目，仅用于构造全新条目。宿主物化场景（条目已由
 * CostTableGen 填好 algName/cost）只应调 HcclAbiHeaderInit 填 header 四字段，勿用本函数。 */
static inline HcclResult HcclTunerAlgoCostEntryInit(HcclTunerAlgoCostEntry* algoCostEntry)
{
    if (algoCostEntry == NULL) {
        return HCCL_E_PTR;
    }
    (void)memset(algoCostEntry, 0, sizeof(HcclTunerAlgoCostEntry));
    return HcclAbiHeaderInit(
        &algoCostEntry->header, HCCL_TUNER_ALGO_COST_ENTRY_MAGIC_WORD, HCCL_TUNER_ALGO_COST_ENTRY_VERSION,
        sizeof(HcclTunerAlgoCostEntry));
}

/* ===== 插件操作集 ===== */
typedef struct {
    HcclAbiHeader header;
    /* 创建 tuner 上下文（每 comm 一次）。插件用标准 malloc 分配临时缓冲、填充后经出参返回；
     * 宿主拷贝进通信域 ctx 后以标准 free 释放——插件不得静态链接自定义分配器
     * （tcmalloc/jemalloc 等），跨分配器释放行为未定义。不干预时返回
     * *tunerCtx=NULL, *tunerCtxSize=0。
     * 并发契约：宿主须串行调用 Create（HCCL 宿主经 init 互斥保证，插件内部依赖此保证，
     * 如引用计数样例等含进程级可追加资源的实现）；Create 与 UpdateAlgoCostTable
     * 可并发（Update 只读 ctx 与进程级只增资源，发布/订阅语义见各实现）。 */
    HcclResult (*Create)(
        HcclComm comm, const HcclTunerCommInfo* commInfo, const HcclTunerTopoInfo* topoInfo, void** tunerCtx,
        uint32_t* tunerCtxSize);
    /* 修改算法代价表（每 op 一次）。命中并真正修改 cost 时置 *matched=true。
     * 条目遍历按 entries[0].header.size 步长。
     * tunerCtx 由宿主持有（通信域 ctx 拷贝件），插件只读、不得释放。 */
    HcclResult (*UpdateAlgoCostTable)(
        void* tunerCtx, const HcclTunerCollOpInfo* collOpInfo, HcclTunerAlgoCostEntry* algoCostEntries,
        uint32_t algoCount, bool* matched);
} HcclTunerPluginOps;

static inline HcclResult HcclTunerPluginOpsInit(HcclTunerPluginOps* pluginOps)
{
    if (pluginOps == NULL) {
        return HCCL_E_PTR;
    }
    (void)memset(pluginOps, 0, sizeof(HcclTunerPluginOps));
    return HcclAbiHeaderInit(
        &pluginOps->header, HCCL_TUNER_PLUGIN_OPS_MAGIC_WORD, HCCL_TUNER_PLUGIN_OPS_VERSION,
        sizeof(HcclTunerPluginOps));
}

/* ===== 入口符号（插件 .so 导出，带默认可见性）=====
 * 契约：幂等、无副作用、返回相同指针；实现须为纯静态初始化（禁止每次调用重写 static）。
 * 返回指针而非按值拷贝：宿主先校验 header 再取函数指针，sizeof 与版本协商解耦。 */
extern const HcclTunerPluginOps* HcclTunerPluginOpsGet(void);

#ifdef __cplusplus
}
#endif

#endif /* HCCL_TUNER_PLUGIN_H_ */
