/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * libhccl_algo_PluginBroker.so 主实现文件。
 */

#include "hccl_algo_plugin_broker_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hccl_algo_plugin {

namespace {

/* 打印到stderr的简单日志 */
#define PLUGIN_LOG_INFO(fmt, ...) std::fprintf(stderr, "[HCCL-ALGO-PluginBroker][INFO] " fmt "\n", ##__VA_ARGS__)
#define PLUGIN_LOG_WARN(fmt, ...) std::fprintf(stderr, "[HCCL-ALGO-PluginBroker][WARN] " fmt "\n", ##__VA_ARGS__)
#define PLUGIN_LOG_ERROR(fmt, ...) std::fprintf(stderr, "[HCCL-ALGO-PluginBroker][ERROR] " fmt "\n", ##__VA_ARGS__)

    std::string ToLowerCopy(const std::string& s)
    {
        std::string out = s;
        for (auto& c : out) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return out;
    }

    bool IsRegularFile(const std::string& path)
    {
        struct stat st {};
        if (lstat(path.c_str(), &st) != 0) {
            return false;
        }
        return S_ISREG(st.st_mode);
    }

    bool IsDirectory(const std::string& path)
    {
        struct stat st {};
        if (stat(path.c_str(), &st) != 0) {
            return false;
        }
        return S_ISDIR(st.st_mode);
    }

} // namespace

/* ------------------------------------------------------------------------------------------- */
/* OpRegistry                                                                                    */
/* ------------------------------------------------------------------------------------------- */
AlgLibEntry* OpRegistry::FindByName(const char* algName)
{
    if (algName == nullptr) {
        return nullptr;
    }
    for (auto& e : algs) {
        if (std::strcmp(e->entry.algName, algName) == 0) {
            return e.get();
        }
    }
    return nullptr;
}

/* ------------------------------------------------------------------------------------------- */
/* PluginBrokerContext                                                                           */
/* ------------------------------------------------------------------------------------------- */

OpRegistry* PluginBrokerContext::FindOpRegistry(const char* opName)
{
    if (opName == nullptr || opName[0] == '\0')
        return nullptr;
    auto it = opRegistries_.find(ToLowerCopy(opName));
    return (it == opRegistries_.end()) ? nullptr : it->second.get();
}

bool PluginBrokerContext::CheckDirTrusted(const std::string& dir, std::string& resolvedDir) const
{
    /*
     * 安全校验：
     *   1) 拒绝根目录自身是符号链接；
     *   2) 使用realpath解析后的规范路径执行后续所有目录操作；
     *   3) 拒绝group-writable或world-writable目录，防止其他用户通过共享组或
     *      全局可写目录篡改/注入恶意.so。
     */
    struct stat pathStat {};
    if (lstat(dir.c_str(), &pathStat) != 0) {
        PLUGIN_LOG_WARN("stat HCCL_PLUGIN_ALG_DIR failed: %s", dir.c_str());
        return false;
    }

    if (S_ISLNK(pathStat.st_mode)) {
        PLUGIN_LOG_ERROR("HCCL_PLUGIN_ALG_DIR must not be a symlink: %s", dir.c_str());
        return false;
    }

    if (!S_ISDIR(pathStat.st_mode)) {
        PLUGIN_LOG_ERROR("HCCL_PLUGIN_ALG_DIR is not a directory: %s", dir.c_str());
        return false;
    }

    char resolved[PATH_MAX] = {0};
    if (realpath(dir.c_str(), resolved) == nullptr) {
        PLUGIN_LOG_WARN("realpath failed for HCCL_PLUGIN_ALG_DIR: %s", dir.c_str());
        return false;
    }

    struct stat resolvedStat {};
    if (stat(resolved, &resolvedStat) != 0) {
        PLUGIN_LOG_WARN("stat resolved HCCL_PLUGIN_ALG_DIR failed: %s", resolved);
        return false;
    }

    if (!S_ISDIR(resolvedStat.st_mode)) {
        PLUGIN_LOG_ERROR("resolved HCCL_PLUGIN_ALG_DIR is not a directory: %s", resolved);
        return false;
    }

    if ((resolvedStat.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
        PLUGIN_LOG_ERROR("HCCL_PLUGIN_ALG_DIR is group/world-writable, refuse to load: %s", resolved);
        return false;
    }

    resolvedDir = resolved;
    return true;
}

bool PluginBrokerContext::LoadSelectorEntries(OpRegistry& reg)
{
    /*
     * 初始化阶段：dlopen算法选择动态库（触发其构造函数完成算法自注册），
     * dlsym(HcclAlgoPluginQueryEntries)取出条目并拷贝，然后dlclose。
     */
    void* handle = dlopen(reg.selectorSoPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        PLUGIN_LOG_ERROR("dlopen selector so failed: %s, dlerror=%s", reg.selectorSoPath.c_str(), dlerror());
        return false;
    }

    using QueryEntriesFn = const HcclAlgoPluginAlgEntry* (*)(int*);
    auto queryFn = reinterpret_cast<QueryEntriesFn>(dlsym(handle, "HcclAlgoPluginQueryEntries"));
    if (queryFn == nullptr) {
        PLUGIN_LOG_ERROR("dlsym HcclAlgoPluginQueryEntries failed in %s: %s", reg.selectorSoPath.c_str(), dlerror());
        dlclose(handle);
        return false;
    }

    int count = 0;
    const HcclAlgoPluginAlgEntry* entries = queryFn(&count);
    if (entries == nullptr || count <= 0) {
        PLUGIN_LOG_WARN("no algorithm entries registered in %s", reg.selectorSoPath.c_str());
        dlclose(handle);
        return true; /* 空注册表不是错误，只是该算子暂无自定义算法 */
    }

    for (int i = 0; i < count; ++i) {
        if (entries[i].magic != HCCL_ALGO_PLUGIN_ENTRY_MAGIC) {
            PLUGIN_LOG_WARN("skip entry with bad magic in %s, index=%d", reg.selectorSoPath.c_str(), i);
            continue;
        }
        if (reg.FindByName(entries[i].algName) != nullptr) {
            PLUGIN_LOG_WARN(
                "duplicate algName [%s] in %s, keep the first one", entries[i].algName, reg.selectorSoPath.c_str());
            continue;
        }
        auto lib = std::make_unique<AlgLibEntry>();
        lib->entry = entries[i]; /* 拷贝条目内容，dlclose前完成拷贝 */
        reg.algs.push_back(std::move(lib));
    }

    dlclose(handle); /* 拷贝完成后立即dlclose，不常驻selector句柄 */
    return true;
}

bool PluginBrokerContext::ScanOpDir(const std::string& opDirPath, const std::string& opDirName)
{
    std::string selectorSoPath = opDirPath + "/libhccl_plugin_" + ToLowerCopy(opDirName) + "_selector.so";
    if (!IsRegularFile(selectorSoPath)) {
        /* 该算子目录下没有选择动态库，跳过（用户可能只为部分算子提供自定义算法） */
        return true;
    }

    auto reg = std::make_unique<OpRegistry>();
    reg->opDirName = opDirName;
    reg->selectorSoPath = selectorSoPath;
    if (!LoadSelectorEntries(*reg)) {
        PLUGIN_LOG_ERROR("failed to load selector entries for op [%s], skip this op", opDirName.c_str());
        return false;
    }
    PLUGIN_LOG_INFO(
        "op [%s] registered %zu custom algorithm(s) from %s", opDirName.c_str(), reg->algs.size(),
        selectorSoPath.c_str());
    opRegistries_[ToLowerCopy(opDirName)] = std::move(reg);
    return true;
}

void PluginBrokerContext::AutoInit()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_) {
        return; /* 幂等：全局静态对象只构造一次，这里额外保护避免被误重复调用 */
    }

    const char* rootDirEnv = std::getenv(kPluginAlgDirEnv);
    if (rootDirEnv == nullptr || rootDirEnv[0] == '\0') {
        PLUGIN_LOG_INFO("HCCL_PLUGIN_ALG_DIR not set, plugin algorithms disabled");
        ready_ = false;
        return;
    }

    std::string rootDir;
    if (!CheckDirTrusted(rootDirEnv, rootDir)) {
        PLUGIN_LOG_ERROR("HCCL_PLUGIN_ALG_DIR failed trust check, refuse to load any plugin algorithm: %s", rootDirEnv);
        ready_ = false;
        return;
    }

    DIR* dir = opendir(rootDir.c_str());
    if (dir == nullptr) {
        PLUGIN_LOG_ERROR("opendir failed: %s", rootDir.c_str());
        ready_ = false;
        return;
    }

    bool anyLoaded = false;
    struct dirent* ent = nullptr;
    while ((ent = readdir(dir)) != nullptr) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") {
            continue;
        }

        std::string subPath = rootDir + "/" + name;
        if (!IsDirectory(subPath)) {
            continue;
        }

        if (ScanOpDir(subPath, name)) {
            anyLoaded = anyLoaded || (opRegistries_.find(ToLowerCopy(name)) != opRegistries_.end());
        }
    }
    closedir(dir);

    ready_ = true; /* 只要扫描流程本身未出现致命错误即视为Ready，具体算子有没有算法不影响Ready状态 */
    PLUGIN_LOG_INFO(
        "PluginBroker auto init done, rootDir=%s, opCount=%zu, anyAlgLoaded=%d", rootDir.c_str(), opRegistries_.size(),
        static_cast<int>(anyLoaded));
}

bool PluginBrokerContext::SelectAlg(const HcclAlgoPluginParam* param, char* algName, size_t algNameLen)
{
    if (!ready_ || param == nullptr || algName == nullptr || algNameLen == 0) {
        return false;
    }
    if (param->magic != HCCL_ALGO_PLUGIN_PARAM_MAGIC) {
        PLUGIN_LOG_ERROR("SelectAlg: bad HcclAlgoPluginParam magic");
        return false;
    }

    OpRegistry* reg = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        reg = FindOpRegistry(param->opName);
    }
    if (reg == nullptr || reg->algs.empty()) {
        return false; /* 该算子无自定义算法注册，未命中 */
    }

    /*
     * 算法选择阶段：dlopen选择动态库 → Select() → dlclose。
     */
    void* handle = dlopen(reg->selectorSoPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        PLUGIN_LOG_ERROR("SelectAlg: dlopen failed: %s, dlerror=%s", reg->selectorSoPath.c_str(), dlerror());
        return false;
    }
    using SelectFn = bool (*)(const HcclAlgoPluginParam*, char*, size_t);
    auto selectFn = reinterpret_cast<SelectFn>(dlsym(handle, "Select"));
    bool hit = false;
    if (selectFn == nullptr) {
        PLUGIN_LOG_ERROR("SelectAlg: dlsym Select failed in %s: %s", reg->selectorSoPath.c_str(), dlerror());
    } else {
        hit = selectFn(param, algName, algNameLen);
    }
    dlclose(handle);
    return hit;
}

namespace {
    int EnsureAlgLoaded(AlgLibEntry* alg, const char* algName)
    {
        std::lock_guard<std::mutex> lock(alg->loadMutex);
        if (alg->loadFailed) {
            PLUGIN_LOG_ERROR("ExecuteAlg: algName [%s] previously failed to load, refuse to retry", algName);
            return HCCL_E_INTERNAL;
        }
        if (alg->implHandle != nullptr) {
            return HCCL_SUCCESS;
        }

        alg->implHandle = dlopen(alg->entry.soPath, RTLD_NOW | RTLD_LOCAL);
        if (alg->implHandle == nullptr) {
            PLUGIN_LOG_ERROR("ExecuteAlg: dlopen impl so failed: %s, dlerror=%s", alg->entry.soPath, dlerror());
            alg->loadFailed = true;
            return HCCL_E_INTERNAL;
        }

        alg->fnPtr = dlsym(alg->implHandle, alg->entry.fnSymbol);
        if (alg->fnPtr == nullptr) {
            PLUGIN_LOG_ERROR(
                "ExecuteAlg: dlsym [%s] failed in %s: %s", alg->entry.fnSymbol, alg->entry.soPath, dlerror());
            dlclose(alg->implHandle);
            alg->implHandle = nullptr;
            alg->loadFailed = true;
            return HCCL_E_INTERNAL;
        }

        return HCCL_SUCCESS;
    }

    bool
    TryExecuteBasicAlg(const char* opName, AlgLibEntry* alg, const HcclAlgoPluginParam* param, void* comm, int& result)
    {
        HcclComm hcclComm = static_cast<HcclComm>(comm);

        if (std::strcmp(opName, "Send") == 0) {
            using Fn = HcclResult (*)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
            result = reinterpret_cast<Fn>(alg->fnPtr)(
                param->sendBuf, param->count, param->dataType, param->remoteRank, hcclComm, param->stream);
            return true;
        }

        if (std::strcmp(opName, "Recv") == 0) {
            using Fn = HcclResult (*)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
            result = reinterpret_cast<Fn>(alg->fnPtr)(
                param->recvBuf, param->count, param->dataType, param->remoteRank, hcclComm, param->stream);
            return true;
        }

        if (std::strcmp(opName, "Broadcast") == 0) {
            using Fn = HcclResult (*)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
            result = reinterpret_cast<Fn>(alg->fnPtr)(
                param->sendBuf, param->count, param->dataType, param->root, hcclComm, param->stream);
            return true;
        }

        if (std::strcmp(opName, "Scatter") == 0) {
            using Fn = HcclResult (*)(void*, void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
            result = reinterpret_cast<Fn>(alg->fnPtr)(
                param->sendBuf, param->recvBuf, param->count, param->dataType, param->root, hcclComm, param->stream);
            return true;
        }

        return false;
    }

    bool TryExecuteCollectiveAlg(
        const char* opName, AlgLibEntry* alg, const HcclAlgoPluginParam* param, void* comm, int& result)
    {
        HcclComm hcclComm = static_cast<HcclComm>(comm);

        if (std::strcmp(opName, "AllReduce") == 0) {
            using Fn = HcclResult (*)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
            result = reinterpret_cast<Fn>(alg->fnPtr)(
                param->sendBuf, param->recvBuf, param->count, param->dataType, param->reduceOp, hcclComm,
                param->stream);
            return true;
        }

        if (std::strcmp(opName, "Reduce") == 0) {
            using Fn
                = HcclResult (*)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, uint32_t, HcclComm, aclrtStream);
            result = reinterpret_cast<Fn>(alg->fnPtr)(
                param->sendBuf, param->recvBuf, param->count, param->dataType, param->reduceOp, param->root, hcclComm,
                param->stream);
            return true;
        }

        if (std::strcmp(opName, "AllGather") == 0) {
            using Fn = HcclResult (*)(void*, void*, uint64_t, HcclDataType, HcclComm, aclrtStream);
            result = reinterpret_cast<Fn>(alg->fnPtr)(
                param->sendBuf, param->recvBuf, param->count, param->dataType, hcclComm, param->stream);
            return true;
        }

        if (std::strcmp(opName, "ReduceScatter") == 0) {
            using Fn = HcclResult (*)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
            result = reinterpret_cast<Fn>(alg->fnPtr)(
                param->sendBuf, param->recvBuf, param->count, param->dataType, param->reduceOp, hcclComm,
                param->stream);
            return true;
        }

        if (std::strcmp(opName, "AllToAll") == 0) {
            using Fn = HcclResult (*)(
                const void*, uint64_t, HcclDataType, const void*, uint64_t, HcclDataType, HcclComm, aclrtStream);
            result = reinterpret_cast<Fn>(alg->fnPtr)(
                param->sendBuf, param->count, param->dataType, param->recvBuf, param->count, param->dataType, hcclComm,
                param->stream);
            return true;
        }

        return false;
    }
} // namespace

int PluginBrokerContext::ExecuteAlg(
    const char* algName, const char* opName, const HcclAlgoPluginParam* param, void* comm)
{
    if (!ready_ || algName == nullptr || opName == nullptr || param == nullptr) {
        return HCCL_E_INTERNAL;
    }

    AlgLibEntry* alg = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        OpRegistry* reg = FindOpRegistry(opName);
        if (reg == nullptr) {
            PLUGIN_LOG_ERROR("ExecuteAlg: no registry for opName=%s", opName);
            return HCCL_E_INTERNAL;
        }

        alg = reg->FindByName(algName);
        if (alg == nullptr) {
            PLUGIN_LOG_ERROR("ExecuteAlg: algName [%s] not found for opName=%s", algName, opName);
            return HCCL_E_INTERNAL;
        }
    }

    int ret = EnsureAlgLoaded(alg, algName);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    int executeRet = HCCL_E_NOT_SUPPORT;
    /*
     * 按算子标准签名分发调用执行函数（对应sdk/hccl_algo_plugin_sdk.h中各算子的
     * HcclAlgoPluginXxxFn typedef）。目前已实现 Send/Recv/Broadcast/AllReduce/Reduce/
     * AllGather/ReduceScatter/AllToAll(非V，等长场景)/Scatter 共9个算子的分发。
     * AllToAllV/AllToAllVC/AllGatherV/ReduceScatterV/BatchSendRecv/Barrier等涉及
     * 变长参数或多item的算子，sdk尚未定义对应的标准执行函数签名，暂不支持，
     * 命中时统一返回HCCL_E_NOT_SUPPORT。
     * 后续如需支持，需先在sdk/hccl_algo_plugin_sdk.h中补充对应的标准签名，再在此处扩展分支。
     */
    if (TryExecuteBasicAlg(opName, alg, param, comm, executeRet)
        || TryExecuteCollectiveAlg(opName, alg, param, comm, executeRet)) {
        return executeRet;
    }

    PLUGIN_LOG_ERROR(
        "ExecuteAlg: opName=%s dispatch not implemented, please extend per "
        "sdk/hccl_algo_plugin_sdk.h standard signatures",
        opName);
    return HCCL_E_NOT_SUPPORT;
}

int PluginBrokerContext::QueryAlgs(const char* opName, char* buf, size_t bufLen)
{
    if (buf == nullptr || bufLen == 0) {
        return HCCL_E_INTERNAL;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    OpRegistry* reg = FindOpRegistry(opName);
    buf[0] = '\0';
    if (reg == nullptr) {
        return HCCL_SUCCESS; /* 无自定义算法也是正常查询结果，返回空列表 */
    }
    size_t offset = 0;
    for (const auto& alg : reg->algs) {
        size_t nameLen = std::strlen(alg->entry.algName);
        if (offset + nameLen + 1 > bufLen) {
            break; /* buf不足，截断 */
        }
        for (size_t i = 0; i < nameLen; ++i) {
            buf[offset + i] = alg->entry.algName[i];
        }
        offset += nameLen;
        buf[offset++] = '\0';
    }
    return HCCL_SUCCESS;
}

/* ------------------------------------------------------------------------------------------- */
/* 全局静态对象：dlopen本.so时构造函数自动触发扫描与注册表构建                                     */
/* ------------------------------------------------------------------------------------------- */
namespace {
    PluginBrokerContext g_pluginBrokerContext;

    struct GlobalAutoInitializer {
        GlobalAutoInitializer() { g_pluginBrokerContext.AutoInit(); }
    } g_globalAutoInitializer;

    bool IsReadyImpl() { return g_pluginBrokerContext.IsReady(); }
    void* FetchContextImpl() { return static_cast<void*>(&g_pluginBrokerContext); }

    bool SelectAlgImpl(void* ctx, const HcclAlgoPluginParam* param, char* algName, size_t algNameLen)
    {
        if (ctx == nullptr) {
            return false;
        }
        return static_cast<PluginBrokerContext*>(ctx)->SelectAlg(param, algName, algNameLen);
    }

    int ExecuteAlgImpl(void* ctx, const char* algName, const char* opName, const HcclAlgoPluginParam* param, void* comm)
    {
        if (ctx == nullptr) {
            return HCCL_E_INTERNAL;
        }
        return static_cast<PluginBrokerContext*>(ctx)->ExecuteAlg(algName, opName, param, comm);
    }

    int QueryAlgsImpl(void* ctx, const char* opName, char* buf, size_t bufLen)
    {
        if (ctx == nullptr) {
            return HCCL_E_INTERNAL;
        }
        return static_cast<PluginBrokerContext*>(ctx)->QueryAlgs(opName, buf, bufLen);
    }

    HcclAlgoPlugin_t g_functionTable = {
        HCCL_PLUGIN_API_VERSION, IsReadyImpl, FetchContextImpl, SelectAlgImpl, ExecuteAlgImpl, QueryAlgsImpl,
    };
} // namespace

} // namespace hccl_algo_plugin

/* 导出符号：HCCL侧HcclAlgoPluginMgr通过dlsym(GetHcclAlgoPlugin)获取函数表 */
extern "C" __attribute__((visibility("default"))) HcclAlgoPlugin_t* GetHcclAlgoPlugin(void)
{
    return &hccl_algo_plugin::g_functionTable;
}
