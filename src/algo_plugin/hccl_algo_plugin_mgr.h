/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * HcclAlgoPluginMgr（集成于HCCL内部）。本文件随HCCL主库libhccl.so一同编译发布。
 */

#ifndef OPS_HCCL_ALGO_PLUGIN_MGR_H
#define OPS_HCCL_ALGO_PLUGIN_MGR_H

#include <mutex>
#include <string>
#include <atomic>

#include "hccl_algo_plugin_common.h"
#include "hccl_algo_plugin_broker_api.h"

/* alg_param.h 定义了 OpParam / TopoInfoWithNetLayerDetails，供FillHcclAlgoPluginParam提取字段使用 */
#include "alg_param.h"

namespace ops_hccl {

class HcclAlgoPluginMgr {
public:
    static HcclAlgoPluginMgr& Instance();

    /** 初始化阶段调用，多次调用安全（幂等），dlopen PluginBroker动态库 */
    HcclResult Init();

    /** 获取HcclAlgoPlugin_t函数表指针，调用前须先IsLoaded()检查 */
    HcclAlgoPlugin_t* GetPlugin();

    /** 获取PluginBroker动态库的全局算法注册表（FetchContext()结果的缓存副本） */
    void* GetContext();

    /** 查询Plugin是否已成功加载，调用GetPlugin()前须先检查 */
    bool IsLoaded() const;

    ~HcclAlgoPluginMgr();

    HcclAlgoPluginMgr(const HcclAlgoPluginMgr&) = delete;
    HcclAlgoPluginMgr& operator=(const HcclAlgoPluginMgr&) = delete;

private:
    HcclAlgoPluginMgr() = default;
    void InitPlugin();
    HcclAlgoPlugin_t* LoadPluginTable(const char* pluginPath);
    bool ValidatePluginTable(const HcclAlgoPlugin_t* table) const;
    void Unload();

    std::once_flag initFlag_;
    std::mutex mutex_;
    void* soHandle_ = nullptr;
    HcclAlgoPlugin_t* pluginTable_ = nullptr;
    void* ctx_ = nullptr;
    std::atomic<bool> loaded_{false};
};

/*
 * 从HCCL内部的OpParam与TopoInfoWithNetLayerDetails中提取Plugin选择/执行所需的关键参数，
 * 填充为与HCCL内部结构解耦的HcclAlgoPluginParam。
 */
void FillHcclAlgoPluginParam(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, HcclAlgoPluginParam& pluginParam);

} // namespace ops_hccl

#endif // OPS_HCCL_ALGO_PLUGIN_MGR_H
