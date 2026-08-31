/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 *
 * 本头文件被三方共同引用：
 *   1) HCCL 代码仓内的 HcclAlgoPluginMgr（src/algo_plugin/）
 *   2) PluginBroker 动态库（libhccl_algo_PluginBroker.so）
 *   3) 自定义算法 SDK（hccl_algo_plugin_sdk.h）
 */

#ifndef HCCL_ALGO_PLUGIN_COMMON_H
#define HCCL_ALGO_PLUGIN_COMMON_H

#include <cstdint>
#include <cstddef>

#if defined(__has_include)
#if __has_include(<hccl/hccl_types.h>)
#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include <acl/acl_rt.h>
#define HCCL_ALGO_PLUGIN_HAS_HCCL_TYPES 1
#endif
#endif

#ifndef HCCL_ALGO_PLUGIN_HAS_HCCL_TYPES
/*
 * 独立编译场景（例如仅编译 PluginBroker/SDK 而不link CANN环境）下的最小替身定义，
 * 保证本文件可以独立解析。真实环境下会优先使用 <hccl/hccl_types.h> 中的定义。
 */
typedef void* aclrtStream;
typedef void* HcclComm;
typedef enum { HCCL_SUCCESS = 0, HCCL_E_INTERNAL = 8, HCCL_E_NOT_SUPPORT = 12 } HcclResult;
typedef enum { HCCL_DATA_TYPE_RESERVED = -1 } HcclDataType;
typedef enum { HCCL_REDUCE_RESERVED = -1 } HcclReduceOp;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------------------------- */
/* 版本 / 魔数定义                                                                              */
/* ------------------------------------------------------------------------------------------- */

/* HCCL侧期望的PluginBroker版本号，用于校验PluginBroker合法性 */
#define HCCL_PLUGIN_API_VERSION 1U

/* HcclAlgoPluginParam / HcclAlgoPluginAlgEntry 结构体版本号 */
#define HCCL_ALGO_PLUGIN_STRUCT_VERSION 1U

/* 结构体魔数，用于跨.so传递结构体时做基本合法性校验，防止不同版本SDK头文件混用 */
#define HCCL_ALGO_PLUGIN_PARAM_MAGIC 0x48415050U /* "HAPP" HcclAlgoPluginParam */
#define HCCL_ALGO_PLUGIN_ENTRY_MAGIC 0x48415045U /* "HAPE" HcclAlgoPluginEntry */

/* 字符串字段长度限制 */
#define HCCL_ALGO_PLUGIN_ALG_NAME_LEN 128U
#define HCCL_ALGO_PLUGIN_SO_PATH_LEN 512U
#define HCCL_ALGO_PLUGIN_FN_SYMBOL_LEN 128U
#define HCCL_ALGO_PLUGIN_OP_NAME_LEN 32U
#define HCCL_ALGO_PLUGIN_TOPO_NAME_LEN 32U

/* ------------------------------------------------------------------------------------------- */
/* HcclAlgoPluginParam：自定义算法选择与执行所需的参数                                            */
/* 由 HCCL 从内部的 OpParam 和 TopoInfoWithNetLayerDetails 中提取填充。                          */
/* ------------------------------------------------------------------------------------------- */
typedef struct {
    uint32_t version;    /* 结构体版本号，取值 HCCL_ALGO_PLUGIN_STRUCT_VERSION */
    uint32_t magic;      /* 魔数，取值 HCCL_ALGO_PLUGIN_PARAM_MAGIC */
    uint32_t structSize; /* sizeof(HcclAlgoPluginParam)，用于ABI兼容性校验 */
    int opType;          /* 仅保留供日志/调试打印用 */
    char opName
        [HCCL_ALGO_PLUGIN_OP_NAME_LEN]; /* 算子名字符串，如"AllReduce"，由HCCL侧生成，PluginBroker/自定义算法应以此字段判断算子类型
                                         */
    uint64_t count;                     /* 元素个数 */
    uint32_t root;                      /* 根节点Rank（仅Broadcast/Reduce等算子有效） */
    int topoType;                       /* 仅保留供日志/调试打印用 */
    char topoName[HCCL_ALGO_PLUGIN_TOPO_NAME_LEN]; /* 拓扑类型字符串，如"CLOS"/"MESH_1D"/"MESH_1D_CLOS"，由HCCL侧生成 */
    uint32_t rankNum;                              /* 通信域总Rank数 */
    uint32_t serverNum;                            /* server数量 */
    void* sendBuf;                                 /* 发送缓冲区 */
    void* recvBuf;                                 /* 接收缓冲区 */
    aclrtStream stream;                            /* 执行流 */
    HcclDataType dataType;                         /* 数据类型 */
    HcclReduceOp reduceOp;         /* 规约类型（仅AllReduce/Reduce/ReduceScatter等算子有效） */
    uint32_t remoteRank;           /* Send/Recv对端Rank（仅Send/Recv算子有效） */
    uint32_t deviceNumPerModule;   /* 每个module（如超节点内节点）的卡数 */
    uint32_t moduleNum;            /* module数量 */
    uint32_t superPodNum;          /* 超节点数量 */
    uint32_t serverNumPerSuperPod; /* 每个超节点的服务器个数 */
    bool isAsymmetricTopo; /* 是否存在非对称拓扑（module间卡数不一致/超节点间server数不一致等） */
    /* ------ 以下字段用于向后兼容扩展，新增字段请追加在末尾，不要在中间插入 ------ */
    uint32_t reserved[7];
} HcclAlgoPluginParam;

/* ------------------------------------------------------------------------------------------- */
/* HcclAlgoPluginAlgEntry：算法条目，描述一个自定义算法的.so路径和执行函数符号名                 */
/* 由 libhccl_plugin_{op}_selector.so 通过                                                     */
/* HcclAlgoPluginQueryEntries() 返回，PluginBroker 在初始化阶段拷贝进全局算法注册表。            */
/* ------------------------------------------------------------------------------------------- */
typedef struct {
    uint32_t version;                              /* 结构体版本号 */
    uint32_t magic;                                /* 魔数，取值 HCCL_ALGO_PLUGIN_ENTRY_MAGIC */
    uint32_t structSize;                           /* sizeof(HcclAlgoPluginAlgEntry) */
    char algName[HCCL_ALGO_PLUGIN_ALG_NAME_LEN];   /* 算法名 */
    char soPath[HCCL_ALGO_PLUGIN_SO_PATH_LEN];     /* 集合通信算法实现动态库路径 */
    char fnSymbol[HCCL_ALGO_PLUGIN_FN_SYMBOL_LEN]; /* 该算法在soPath对应.so中导出的执行函数符号名 */
} HcclAlgoPluginAlgEntry;

/* 便捷初始化函数：填充version/magic/structSize等公共字段 */
static inline void HcclAlgoPluginParamInit(HcclAlgoPluginParam* param)
{
    if (param == nullptr) {
        return;
    }
    param->version = HCCL_ALGO_PLUGIN_STRUCT_VERSION;
    param->magic = HCCL_ALGO_PLUGIN_PARAM_MAGIC;
    param->structSize = sizeof(HcclAlgoPluginParam);
}

static inline void HcclAlgoPluginAlgEntryInit(HcclAlgoPluginAlgEntry* entry)
{
    if (entry == nullptr) {
        return;
    }
    entry->version = HCCL_ALGO_PLUGIN_STRUCT_VERSION;
    entry->magic = HCCL_ALGO_PLUGIN_ENTRY_MAGIC;
    entry->structSize = sizeof(HcclAlgoPluginAlgEntry);
}

#ifdef __cplusplus
}
#endif

#endif /* HCCL_ALGO_PLUGIN_COMMON_H */
