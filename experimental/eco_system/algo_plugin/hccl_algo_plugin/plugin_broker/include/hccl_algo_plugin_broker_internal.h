/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * PluginBroker（libhccl_algo_PluginBroker.so）内部实现，不对外暴露。
 */

#ifndef HCCL_ALGO_PLUGIN_BROKER_INTERNAL_H
#define HCCL_ALGO_PLUGIN_BROKER_INTERNAL_H

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>

#include "hccl_algo_plugin_common.h"
#include "hccl_algo_plugin_broker_api.h"

namespace hccl_algo_plugin {

/* HCCL_PLUGIN_ALG_DIR环境变量名，指向自定义算法实现动态库的根目录 */
constexpr const char* kPluginAlgDirEnv = "HCCL_PLUGIN_ALG_DIR";

/* 一个自定义集合通信算法实现动态库的懒加载状态 */
struct AlgLibEntry {
    HcclAlgoPluginAlgEntry entry{}; /* 算法名/so路径/符号名，来自Selector so的QueryEntries结果 */
    void* implHandle = nullptr;     /* lib{Name}Impl.so 的dlopen句柄，懒加载 */
    void* fnPtr = nullptr;          /* dlsym(fnSymbol)解析出的执行函数指针 */
    bool loadFailed = false; /* 曾经加载失败，避免后续重复尝试dlopen/dlsym（拦截重试风暴） */
    std::mutex loadMutex;    /* 保护懒加载过程的并发访问 */

    AlgLibEntry() = default;
    /* 显式声明拷贝/移动，因为std::mutex不可拷贝，registry内部使用指针/emplace管理，规避拷贝 */
    AlgLibEntry(const AlgLibEntry&) = delete;
    AlgLibEntry& operator=(const AlgLibEntry&) = delete;
};

/* 一个算子（如AllReduce）对应的算法选择动态库路径及其注册的全部算法条目 */
struct OpRegistry {
    std::string opDirName;                          /* 算子目录名，如"AllReduce" */
    std::string selectorSoPath;                     /* libhccl_plugin_{op}_selector.so 全路径 */
    std::vector<std::unique_ptr<AlgLibEntry>> algs; /* 该算子下所有已注册算法（顺序即注册顺序） */

    AlgLibEntry* FindByName(const char* algName);
};

/*
 * PluginBrokerContext：即HcclAlgoPlugin_t::FetchContext()返回的ctx实际类型。
 * 全局静态对象（详见plugin_broker.cc），构造函数触发算子根目录扫描与算法注册表构建。
 */
class PluginBrokerContext {
public:
    /* 由全局静态对象的构造函数调用，扫描HCCL_PLUGIN_ALG_DIR并构建注册表 */
    void AutoInit();

    bool IsReady() const { return ready_; }

    bool SelectAlg(const HcclAlgoPluginParam* param, char* algName, size_t algNameLen);

    int ExecuteAlg(const char* algName, const char* opName, const HcclAlgoPluginParam* param, void* comm);

    int QueryAlgs(const char* opName, char* buf, size_t bufLen);

private:
    bool CheckDirTrusted(const std::string& dir, std::string& resolvedDir) const;
    bool ScanOpDir(const std::string& opDirPath, const std::string& opDirName);
    bool LoadSelectorEntries(OpRegistry& reg);
    OpRegistry* FindOpRegistry(const char* opName);

    std::unordered_map<std::string, std::unique_ptr<OpRegistry>> opRegistries_; /* key: opDirName */
    bool ready_ = false;
    std::mutex
        mutex_; /* 保护opRegistries_的读写（ExecuteAlg懒加载时会修改AlgLibEntry，但registry结构本身初始化后只读） */
};

} // namespace hccl_algo_plugin

#endif /* HCCL_ALGO_PLUGIN_BROKER_INTERNAL_H */
