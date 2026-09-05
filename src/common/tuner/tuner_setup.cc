/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tuner_setup.h"

#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "alg_param.h"
#include "hccl_common.h"
#include "hccl_tuner_plugin.h"

using ops_hccl::TopoInfoWithNetLayerDetails;

namespace {
/* ===== 进程级单例状态（mutex 保护）===== */
constexpr int32_t LOAD_READY = 0;
constexpr int32_t LOAD_SUCCESS = 1;
constexpr int32_t LOAD_FAILED = -1;

std::mutex g_tunerMutex;
int32_t g_loadStatus = LOAD_READY;
uint32_t g_refCount = 0;
void* g_libHandle = nullptr;
hcclTunerFuncs_v1_t g_funcs = {};

constexpr uint32_t TUNER_COMM_NAME_MAX_LENGTH = 128;
constexpr const char* TUNER_CTX_PREFIX = "__tuner_";
constexpr uint64_t TUNER_SLOW_CALL_THRESHOLD_MS = 100;  /* getCollInfo 慢调用阈值 */
constexpr uint32_t TUNER_SLOW_CALL_LIMIT = 3;           /* 连续慢调用上限，超过则禁用插件 */
constexpr uint64_t TUNER_SLOW_INIT_THRESHOLD_MS = 5000; /* init 慢调用阈值（一次性，不禁用） */
std::atomic<uint32_t> g_slowCallCount{0};               /* 连续慢调用计数（原子，无锁） */
bool g_tunerModifiedCost = false;                       /* 上次 getCollInfo 是否命中 */

/* 读取环境变量，返回 string 便于比较。 */
std::string GetEnv(const char* name)
{
    const char* val = std::getenv(name);
    return (val != nullptr) ? std::string(val) : std::string();
}

/* 首次加载插件：dlopen + dlsym + 版本校验 + 获取函数表。调用前已持锁。 */
bool LoadPluginLocked()
{
    if (g_loadStatus == LOAD_SUCCESS) {
        g_refCount++;
        return true;
    }
    if (g_loadStatus == LOAD_FAILED) {
        return false;
    }

    /* LOAD_READY：首次加载 */
    std::string pluginPath = GetEnv("HCCL_TUNER_PLUGIN");
    if (pluginPath.empty() || pluginPath == "none") {
        g_loadStatus = LOAD_FAILED;
        return false;
    }

    /* 信任边界：插件 .so 在本进程内执行*/
    void* handle = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        HCCL_WARNING("[Tuner] dlopen failed, path[%s], err[%s].", pluginPath.c_str(), dlerror());
        g_loadStatus = LOAD_FAILED;
        return false;
    }

    auto* tuner = static_cast<hcclTunerFuncs_v1_t*>(dlsym(handle, "hcclTunerPlugin_v1"));
    if (tuner == nullptr || tuner->init == nullptr || tuner->getCollInfo == nullptr) {
        HCCL_WARNING("[Tuner] dlsym hcclTunerPlugin_v1 failed or null function pointers, err[%s].", dlerror());
        dlclose(handle);
        g_loadStatus = LOAD_FAILED;
        return false;
    }

    g_libHandle = handle;
    g_funcs = *tuner;
    g_funcs.structSize = sizeof(hcclTunerFuncs_v1_t);
    g_refCount = 1;
    g_loadStatus = LOAD_SUCCESS;
    HCCL_INFO("[Tuner] plugin loaded (v1).");
    return true;
}

/* 从 TopoInfoWithNetLayerDetails 填充 hcclTunerCommInfo_t。 */
HcclResult BuildCommInfo(
    HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, hcclTunerCommInfo_t& commInfo, char* commNameBuf,
    uint32_t bufLen)
{
    commInfo = {};
    if (topoInfo != nullptr) {
        commInfo.nRanks = topoInfo->userRankSize;
        commInfo.nServers = topoInfo->serverNum;
        const auto& layers = topoInfo->netLayerDetails;
        if (!layers.localNetInsSizeOfLayer.empty()) {
            commInfo.nNpusPerServer = layers.localNetInsSizeOfLayer[0];
        }
        if (layers.netInstNumOfLayer.size() > 1) {
            commInfo.nPods = layers.netInstNumOfLayer[1];
        }
        if (layers.netInstNumOfLayer.size() > 2) {
            commInfo.nSuperPods = layers.netInstNumOfLayer[2];
        }
    }

    if (commNameBuf != nullptr && bufLen > 0) {
        commNameBuf[0] = '\0';
        HcclResult ret = HcclGetCommName(comm, commNameBuf);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[Tuner] HcclGetCommName failed, ret[%d].", ret);
        }
        commInfo.commName = commNameBuf;
    }

    void* bufferAddr = nullptr;
    uint64_t bufferSize = 0;
    HcclResult ret = HcclGetHcclBuffer(comm, &bufferAddr, &bufferSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[Tuner] HcclGetHcclBuffer failed, ret[%d].", ret);
    }
    commInfo.bufferSize = bufferSize;

    commInfo.structSize = sizeof(hcclTunerCommInfo_t);
    return HCCL_SUCCESS;
}

/* 构造 hcclTunerHostFunctions_t。 */
void BuildHostFuncs(hcclTunerHostFunctions_t& hostFuncs)
{
    hostFuncs = {};
    hostFuncs.ctxCreate = TunerCtxCreate;
    hostFuncs.ctxGet = TunerCtxGet;
    hostFuncs.ctxDestroy = TunerCtxDestroy;
    hostFuncs.logFunction = TunerLogFunction;
    hostFuncs.structSize = sizeof(hcclTunerHostFunctions_t);
}
} /* namespace */

bool HcclTunerIsLoaded()
{
    std::lock_guard<std::mutex> lock(g_tunerMutex);
    return g_loadStatus == LOAD_SUCCESS;
}

HcclResult HcclTunerInit(HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo)
{
    /* 1. 加载插件（mutex 保护，首次 dlopen + dlsym + 版本校验） */
    hcclTunerFuncs_v1_t funcs = {};
    {
        std::lock_guard<std::mutex> lock(g_tunerMutex);
        if (!LoadPluginLocked()) {
            return HCCL_SUCCESS; /* 未配置或加载失败，no-op，回退 CostModel */
        }
        funcs = g_funcs; /* 锁内拷贝，避免 TOCTOU 竞态 */
    }

    /* 2. 填充 commInfo（不加锁，每个 comm 独立） */
    hcclTunerCommInfo_t commInfo = {};
    char commNameBuf[TUNER_COMM_NAME_MAX_LENGTH] = {};
    HcclResult ret = BuildCommInfo(comm, topoInfo, commInfo, commNameBuf, sizeof(commNameBuf));
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[Tuner] BuildCommInfo failed, ret[%d].", ret);
        return HCCL_SUCCESS;
    }

    /* 3. 构造 hostFuncs */
    hcclTunerHostFunctions_t hostFuncs = {};
    BuildHostFuncs(hostFuncs);

    /* 4. 调用插件 init（使用锁内拷贝的 funcs 副本） */
    HCCL_INFO(
        "[HcclTunerInit] comm[%p] nRanks[%u] nServers[%u] nNpusPerServer[%u] commName[%s] bufferSize[%llu].", comm,
        commInfo.nRanks, commInfo.nServers, commInfo.nNpusPerServer,
        (commInfo.commName != nullptr) ? commInfo.commName : "?", commInfo.bufferSize);
    auto initStart = std::chrono::steady_clock::now();
    ret = funcs.init(comm, &commInfo, &hostFuncs);
    auto initMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - initStart).count();
    if (static_cast<uint64_t>(initMs) > TUNER_SLOW_INIT_THRESHOLD_MS) {
        HCCL_WARNING("[Tuner] plugin init took %lldms (threshold %llums).", initMs, TUNER_SLOW_INIT_THRESHOLD_MS);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[HcclTunerInit] plugin init failed, ret[%d], fall back to CostModel.", ret);
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[HcclTunerInit] plugin init success, comm[%p].", comm);
    return HCCL_SUCCESS;
}

HcclResult HcclTunerCallGetCollInfo(
    HcclComm comm, HcclCMDType cmdType, size_t nBytes, HcclDataType dataType, hcclTunerAlgoEntry_t* algoEntries,
    int algoCount, bool* modified)
{
    if (modified != nullptr) {
        *modified = false;
    }
    if (algoEntries == nullptr || algoCount <= 0) {
        return HCCL_SUCCESS;
    }

    /* 插件未加载时 no-op；锁内拷贝 g_funcs 避免 TOCTOU 竞态 */
    hcclTunerFuncs_v1_t funcs = {};
    {
        std::lock_guard<std::mutex> lock(g_tunerMutex);
        if (g_loadStatus != LOAD_SUCCESS) {
            return HCCL_SUCCESS;
        }
        funcs = g_funcs;
    }

    /* 不支持的操作类型跳过 */
    if (cmdType == HCCL_CMD_INVALID) {
        return HCCL_SUCCESS;
    }

    hcclTunerCollInfo_t collInfo = {};
    collInfo.collType = cmdType;
    collInfo.nBytes = nBytes;
    collInfo.dataType = dataType;
    collInfo.structSize = sizeof(hcclTunerCollInfo_t);
    int matched = 0; /* 初始化：未命中 */

    /* 使用锁内拷贝的 funcs 副本，避免并发 HcclTunerDestroy 重置 g_funcs */
    auto callStart = std::chrono::steady_clock::now();
    HcclResult ret = funcs.getCollInfo(comm, &collInfo, algoEntries, algoCount, &matched);
    auto callMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - callStart).count();
    /* 慢调用检测：连续超过阈值则禁用 tuner，后续 op 回退 CostModel */
    if (static_cast<uint64_t>(callMs) > TUNER_SLOW_CALL_THRESHOLD_MS) {
        uint32_t count = ++g_slowCallCount;
        HCCL_WARNING(
            "[Tuner] getCollInfo took %lldms (threshold %llums), slowCallCount=%u/%u.", callMs,
            TUNER_SLOW_CALL_THRESHOLD_MS, count, TUNER_SLOW_CALL_LIMIT);
        if (count >= TUNER_SLOW_CALL_LIMIT) {
            std::lock_guard<std::mutex> lock(g_tunerMutex);
            g_loadStatus = LOAD_FAILED;
            HCCL_WARNING("[Tuner] plugin disabled after %u consecutive slow calls, fall back to CostModel.", count);
        }
    } else {
        g_slowCallCount.store(0, std::memory_order_relaxed);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[Tuner] getCollInfo failed, ret[%d], ignore plugin modification.", ret);
        g_tunerModifiedCost = false; /* ST 测试观测用 */
        return HCCL_SUCCESS;
    }
    bool didModify = (matched == 1);
    g_tunerModifiedCost = didModify; /* ST 测试观测用（生产路径用 modified 输出参数） */
    if (modified != nullptr) {
        *modified = didModify;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclTunerDestroy(HcclComm comm)
{
    (void)comm;
    std::lock_guard<std::mutex> lock(g_tunerMutex);
    if (g_loadStatus != LOAD_SUCCESS) {
        return HCCL_SUCCESS;
    }
    if (g_refCount > 0) {
        g_refCount--;
    }
    /* .so 故意不 dlclose：避免与在途 getCollInfo 竞争，且无额外内存代价。
     * .so 随进程退出由 OS 回收。refCount 仅记录存活 comm 数，不影响功能。 */
    return HCCL_SUCCESS;
}

bool HcclTunerDidModifyCost() { return g_tunerModifiedCost; }

void HcclTunerResetMatchStatus() { g_tunerModifiedCost = false; }
