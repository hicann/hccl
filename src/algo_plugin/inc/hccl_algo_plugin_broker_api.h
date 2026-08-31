/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * PluginBroker动态库接口（HcclAlgoPlugin_t）。
 *
 * 本头文件被 libhccl_algo_PluginBroker.so（实现方）与 HcclAlgoPluginMgr（调用方，嵌入HCCL代码仓）
 * 共同引用，是HCCL与PluginBroker之间的唯一稳定ABI边界。
 */

#ifndef HCCL_ALGO_PLUGIN_BROKER_API_H
#define HCCL_ALGO_PLUGIN_BROKER_API_H

#include "hccl_algo_plugin_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct HcclAlgoPlugin_t {
    uint32_t version; /* PluginBroker版本号，须等于HCCL侧期望的HCCL_PLUGIN_API_VERSION */

    /* 查询自动初始化（构造函数触发的算子根目录扫描等操作）是否成功 */
    bool (*IsReady)(void);

    /* 获取自动构建的全局算法注册表指针，作为后续SelectAlg/ExecuteAlg/QueryAlgs的ctx入参 */
    void* (*FetchContext)(void);

    /*
     * 算法选择：调用对应算子的选择动态库Select()，返回true表示命中，algName填入选中算法名；
     * 返回false表示未命中，HCCL走原有逻辑。param为从HCCL侧提取的算法选择所需关键参数。
     */
    bool (*SelectAlg)(void* ctx, const HcclAlgoPluginParam* param, char* algName, size_t algNameLen);

    /*
     * 算法执行：根据algName定位集合通信算法实现动态库，调用对应算子的算法执行函数。
     * 返回非HCCL_SUCCESS表示执行失败，调用方（HCCL）不应回退至原有执行逻辑。
     */
    int (*ExecuteAlg)(void* ctx, const char* algName, const char* opName, const HcclAlgoPluginParam* param, void* comm);

    /* 算法查询：查询已注册的算法名列表，输出写入buf（以'\0'分隔的算法名序列） */
    int (*QueryAlgs)(void* ctx, const char* opName, char* buf, size_t bufLen);
};

/* libhccl_algo_PluginBroker.so 须导出此符号，供HcclAlgoPluginMgr通过dlsym获取函数表 */
typedef HcclAlgoPlugin_t* (*HcclAlgoPluginGetTableFn)(void);
#define HCCL_ALGO_PLUGIN_GET_TABLE_SYMBOL "GetHcclAlgoPlugin"

HcclAlgoPlugin_t* GetHcclAlgoPlugin(void);

#ifdef __cplusplus
}
#endif

#endif /* HCCL_ALGO_PLUGIN_BROKER_API_H */
