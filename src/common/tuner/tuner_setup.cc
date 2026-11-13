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
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "alg_param.h"
#include "hccl_common.h"
#include "hccl_tuner_plugin.h"

using ops_hccl::TopoInfoWithNetLayerDetails;

/* g_loadStatus 外部链接（UT 重置用），定义在匿名 namespace 之后 */
extern int32_t g_loadStatus;

namespace {
/* ===== 进程级单例状态（mutex 保护）===== */
constexpr int32_t LOAD_READY = 0;
constexpr int32_t LOAD_SUCCESS = 1;
constexpr int32_t LOAD_FAILED = -1;

std::mutex g_tunerMutex;
/* init 专用互斥：序列化各 comm 的 Create/ctx 落盘段（闭合并发同 comm 双 Create 的 TOCTOU），
 * 与 g_tunerMutex 分离——插件 Create 容忍慢（5s 告警），不能阻塞 op 路径的状态检查。 */
std::mutex g_tunerInitMutex;
void* g_libHandle = nullptr;
/* 指向插件 .so 内的静态 const 对象；.so 不 dlclose，指针进程内恒有效。
 * 引用指针而非按宿主 sizeof 拷贝结构体，规避版本不一致时的越界读。 */
const HcclTunerPluginOps* g_ops = nullptr;

constexpr uint32_t TUNER_COMM_NAME_MAX_LENGTH = 128;
/* 插件 ctx tag/引擎约定（__tuner_ 前缀隔离 HCCL 内部 ctx tag 命名空间）。 */
constexpr const char* TUNER_CTX_TAG = "__tuner_main";
constexpr uint64_t TUNER_SLOW_CALL_THRESHOLD_MS = 100;  /* UpdateAlgoCostTable 慢调用阈值 */
constexpr uint32_t TUNER_SLOW_CALL_LIMIT = 3;           /* 连续慢调用上限，超过则禁用插件 */
constexpr uint64_t TUNER_SLOW_INIT_THRESHOLD_MS = 5000; /* Create 慢调用阈值（一次性，不禁用） */
std::atomic<uint32_t> g_slowCallCount{0};               /* 连续慢调用计数（原子，无锁） */

/* 读取环境变量，返回 string 便于比较。 */
std::string GetEnv(const char* name)
{
    const char* val = std::getenv(name);
    return (val != nullptr) ? std::string(val) : std::string();
}

/* 首次加载插件：dlopen + dlsym + header 三连校验（magicWord/version/size）+ 函数指针非空。
 * 调用前已持锁。仅探测当前入口符号，不做旧符号回退（单一 ABI 路径）。
 *
 * 版本校验策略（区间校验，新宿主兼容旧插件、旧宿主不兼容新插件）：
 *   version > 期望 → 拒绝（插件比宿主新，旧宿主不认新尾部字段/回调）
 *   version < 期望 → WARN + 放行（插件比宿主旧，宿主只消费已知前缀字段）
 *   version = 期望 → 正常
 * 当前所有结构体版本=1，高低分支不可达（预埋演进路径，行为与严格相等等价）。
 * 未来 bump 时配套义务：
 *   1. 宿主读新版本尾部字段/回调前，先判 ops->header.version 与 ops->header.size
 *      （旧插件静态对象较小，直读尾部字段越界）；
 *   2. entry 数组遍历按 entries[0].header.size 步长（已就绪，旧插件读新条目前缀安全）；
 *   3. 随 bump 补混跑测试：新宿主×旧插件（放行且不越界）、旧宿主×新插件（确认拒绝）。 */
bool LoadPluginLocked()
{
    if (g_loadStatus == LOAD_SUCCESS) {
        return true;
    }
    if (g_loadStatus == LOAD_FAILED) {
        return false;
    }

    /* LOAD_READY：首次加载 */
    std::string pluginPath = GetEnv("HCCL_ALG_TUNER_PLUGIN");
    if (pluginPath.empty() || pluginPath == "none") {
        // 未设置与显式禁用(none)同文案，避免设置none时被误读为"配置未生效"
        HCCL_RUN_INFO("[HCCL_ENV] HCCL_ALG_TUNER_PLUGIN is not set or set to none, tuner plugin disabled.");
        g_loadStatus = LOAD_FAILED;
        return false;
    }
    // 环境变量来源：按hcomm externalinput的[HCCL_ENV]官方句式打印（RUN通道，不随用户日志级别关闭）
    HCCL_RUN_INFO("[HCCL_ENV] HCCL_ALG_TUNER_PLUGIN set by environment to [%s]", pluginPath.c_str());

    /* 信任边界：插件 .so 在本进程内执行 */
    void* handle = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        HCCL_WARNING("[Tuner] dlopen failed, path[%s], err[%s].", pluginPath.c_str(), dlerror());
        g_loadStatus = LOAD_FAILED;
        return false;
    }

    auto getOps = reinterpret_cast<const HcclTunerPluginOps* (*)(void)>(dlsym(handle, "HcclTunerPluginOpsGet"));
    if (getOps == nullptr) {
        HCCL_WARNING("[Tuner] dlsym HcclTunerPluginOpsGet failed, err[%s].", dlerror());
        dlclose(handle);
        g_loadStatus = LOAD_FAILED;
        return false;
    }

    const HcclTunerPluginOps* ops = getOps();
    /* size 下限覆盖宿主实际访问的尾部字段（UpdateAlgoCostTable 指针），而非仅 sizeof(HcclAbiHeader)——
     * 声明过小的 ops 结构会让后续函数指针读取越界。offsetof 对标准布局结构体合法。
     * 尾部字段读取一律经 hasXxx 守卫：短路条件本身安全，但 WARN 参数无条件求值，
     * 旧版过小结构体仍会在日志路径越界读。 */
    bool hasCreate
        = (ops != nullptr && ops->header.size >= (offsetof(HcclTunerPluginOps, Create) + sizeof(ops->Create))
           && ops->Create != nullptr);
    bool hasUpdate
        = (ops != nullptr
           && ops->header.size >= (offsetof(HcclTunerPluginOps, UpdateAlgoCostTable) + sizeof(ops->UpdateAlgoCostTable))
           && ops->UpdateAlgoCostTable != nullptr);
    if (ops == nullptr || ops->header.magicWord != HCCL_TUNER_PLUGIN_OPS_MAGIC_WORD
        || ops->header.version > HCCL_TUNER_PLUGIN_OPS_VERSION || !hasCreate || !hasUpdate) {
        HCCL_WARNING(
            "[Tuner] invalid plugin ops (magic[0x%x] version[%u] size[%u] create[%d] update[%d]).",
            (ops != nullptr) ? ops->header.magicWord : 0u, (ops != nullptr) ? ops->header.version : 0u,
            (ops != nullptr) ? ops->header.size : 0u, hasCreate ? 1 : 0, hasUpdate ? 1 : 0);
        dlclose(handle);
        g_loadStatus = LOAD_FAILED;
        return false;
    }
    if (ops->header.version < HCCL_TUNER_PLUGIN_OPS_VERSION) {
        HCCL_WARNING(
            "[Tuner] plugin ops v%u older than host v%u, new callbacks unused.", ops->header.version,
            HCCL_TUNER_PLUGIN_OPS_VERSION);
    }

    g_libHandle = handle;
    g_ops = ops;
    g_loadStatus = LOAD_SUCCESS;
    HCCL_INFO("[Tuner] plugin loaded (ops version %u).", ops->header.version);
    return true;
}

/* 从 TopoInfoWithNetLayerDetails 填充 HcclTunerTopoInfo（仅纯拓扑）。 */
void BuildTopoInfo(const TopoInfoWithNetLayerDetails* topoInfo, HcclTunerTopoInfo& topo)
{
    (void)HcclTunerTopoInfoInit(&topo);
    if (topoInfo != nullptr) {
        topo.nRanks = topoInfo->userRankSize;
        topo.nServers = topoInfo->serverNum;
        const auto& layers = topoInfo->netLayerDetails;
        if (!layers.localNetInsSizeOfLayer.empty()) {
            topo.nNpusPerServer = layers.localNetInsSizeOfLayer[0];
        }
        if (layers.netInstNumOfLayer.size() > 1) {
            topo.nPods = layers.netInstNumOfLayer[1];
        }
        if (layers.netInstNumOfLayer.size() > 2) {
            topo.nSuperPods = layers.netInstNumOfLayer[2];
        }
    }
}

/* 填充 HcclTunerCommInfo（commName 指向调用方栈缓冲，仅 Create 调用期间有效，插件需自行持久化）。 */
void BuildCommInfo(HcclComm comm, HcclTunerCommInfo& commInfo, char* commNameBuf, uint32_t bufLen)
{
    (void)HcclTunerCommInfoInit(&commInfo);
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
}
} /* namespace */

/* g_loadStatus 在匿名 namespace 外：外部链接，UT 在 SetUp 中重置回 0(LOAD_READY)
 * 允许每个用例独立重新加载（生产行为不变——生产代码从不重置此值）。 */
int32_t g_loadStatus = 0;

bool HcclTunerIsLoaded()
{
    std::lock_guard<std::mutex> lock(g_tunerMutex);
    return g_loadStatus == LOAD_SUCCESS;
}

HcclResult HcclTunerInit(HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo)
{
    /* 1. 加载插件（mutex 保护，首次 dlopen + dlsym + header 校验） */
    const HcclTunerPluginOps* ops = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_tunerMutex);
        if (!LoadPluginLocked()) {
            return HCCL_SUCCESS; /* 未配置或加载失败，no-op，回退 CostModel */
        }
        ops = g_ops;
    }

    /* 2. 填充 topoInfo/commInfo（不加锁，每个 comm 独立） */
    HcclTunerTopoInfo topo = {};
    BuildTopoInfo(topoInfo, topo);
    HcclTunerCommInfo commInfo = {};
    char commNameBuf[TUNER_COMM_NAME_MAX_LENGTH] = {};
    BuildCommInfo(comm, commInfo, commNameBuf, sizeof(commNameBuf));

    /* 3. 幂等初始化（init 专用互斥：守卫与建 ctx 之间无 TOCTOU 窗口） */
    std::lock_guard<std::mutex> initLock(g_tunerInitMutex);
    /* 幂等守卫：__tuner_main 已存在（上次初始化已成功）则直接返回。
     * 使 tuner_init 标记创建失败后的重试不会重复 Create——hcomm 对已存在 tag 的
     * Create 行为因此不可达，无需依赖其语义。ctx 非空一并判定：不依赖
     * EngineCtxGet "未命中必返错" 的隐式契约，守卫本地自洽。 */
    {
        void* existCtx = nullptr;
        uint64_t existSize = 0;
        if (HcclEngineCtxGet(comm, TUNER_CTX_TAG, COMM_ENGINE_CPU_TS, &existCtx, &existSize) == HCCL_SUCCESS
            && existCtx != nullptr) {
            return HCCL_SUCCESS;
        }
    }

    /* 4. 调用插件 Create */
    HCCL_INFO(
        "[HcclTunerInit] comm[%p] nRanks[%u] nServers[%u] nNpusPerServer[%u] commName[%s] bufferSize[%llu].", comm,
        topo.nRanks, topo.nServers, topo.nNpusPerServer, (commInfo.commName != nullptr) ? commInfo.commName : "?",
        commInfo.bufferSize);
    void* tunerCtxTmp = nullptr;
    uint32_t tunerCtxSize = 0;
    auto createStart = std::chrono::steady_clock::now();
    HcclResult ret = ops->Create(comm, &commInfo, &topo, &tunerCtxTmp, &tunerCtxSize);
    auto createMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - createStart).count();
    if (static_cast<uint64_t>(createMs) > TUNER_SLOW_INIT_THRESHOLD_MS) {
        HCCL_WARNING("[Tuner] plugin Create took %lldms (threshold %llums).", createMs, TUNER_SLOW_INIT_THRESHOLD_MS);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[HcclTunerInit] plugin Create failed, ret[%d], fall back to CostModel.", ret);
        free(tunerCtxTmp); /* 插件可能已部分构造后失败（契约：标准 malloc，宿主释放；free(nullptr) 为 no-op） */
        return HCCL_SUCCESS;
    }
    /* 插件 inactive（无配置/JSON 解析失败/规则超限）：空 ctx 无需拷贝，no-op */
    if (tunerCtxTmp == nullptr || tunerCtxSize == 0) {
        HCCL_INFO("[HcclTunerInit] plugin inactive (empty tuner ctx), skip ctx acquire, comm[%p].", comm);
        free(tunerCtxTmp); /* 非 NULL 而 size=0 的异常形态同样释放，防泄漏 */
        return HCCL_SUCCESS;
    }

    /* ctx 申报尺寸上限（防插件 bug 写飞 size：未初始化值/尺寸公式错误）。典型 ctx 为 KB 级
     * （千条规则 ~2.7KB + 字符串区）；超限 WARN 回退 CostModel，不阻塞、不崩溃。 */
    constexpr uint32_t TUNER_CTX_MAX_SIZE = 4u * 1024u * 1024u;
    if (tunerCtxSize > TUNER_CTX_MAX_SIZE) {
        HCCL_WARNING(
            "[HcclTunerInit] tuner ctx size[%u] exceeds limit[%u], fall back to CostModel.", tunerCtxSize,
            TUNER_CTX_MAX_SIZE);
        free(tunerCtxTmp);
        return HCCL_SUCCESS;
    }

    /* 5. 拷贝进通信域 ctx 后释放插件临时缓冲（插件契约：标准 malloc 分配） */
    void* memCtx = nullptr;
    ret = HcclEngineCtxCreate(comm, TUNER_CTX_TAG, COMM_ENGINE_CPU_TS, tunerCtxSize, &memCtx);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[HcclTunerInit] HcclEngineCtxCreate failed, ret[%d], fall back to CostModel.", ret);
        free(tunerCtxTmp);
        return HCCL_SUCCESS;
    }
    /* Copy 前先清零：EngineCtxCreate 到 Copy 之间 ctx 已对 op 路径可见（同 comm 并发首 op），
     * 清零保证窗口内读者拿到确定性未就绪态（configValid=0 / ctxReady 哨兵未落盘，插件 no-op），
     * 而不是引擎池里的脏数据 */
    if (memCtx != nullptr) {
        (void)memset(memCtx, 0, tunerCtxSize);
    }
    ret = HcclEngineCtxCopy(comm, COMM_ENGINE_CPU_TS, TUNER_CTX_TAG, tunerCtxTmp, tunerCtxSize, 0);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[HcclTunerInit] HcclEngineCtxCopy failed, ret[%d], rollback and fall back to CostModel.", ret);
        /* 回滚半初始化的 ctx：防止后续 op 将未初始化内容当 StoredHeader 解析；
         * Destroy 失败不静默——残留 ctx 由插件侧头部 sanity 兜底拒绝，此处留响亮告警 */
        HcclResult destroyRet = HcclEngineCtxDestroy(comm, TUNER_CTX_TAG, COMM_ENGINE_CPU_TS);
        if (destroyRet != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[HcclTunerInit] rollback Destroy failed, ret[%d] — half-copied ctx may remain, "
                "plugin header sanity will reject it.",
                destroyRet);
        }
        free(tunerCtxTmp);
        return HCCL_SUCCESS;
    }
    free(tunerCtxTmp);
    HCCL_INFO("[HcclTunerInit] plugin Create success, comm[%p] ctxSize[%u].", comm, tunerCtxSize);
    return HCCL_SUCCESS;
}

HcclResult HcclTunerCallGetCollInfo(
    HcclComm comm, const HcclTunerCollOpInfo* collOpInfo, HcclTunerAlgoCostEntry* algoCostEntries, uint32_t algoCount,
    bool* modified)
{
    if (modified != nullptr) {
        *modified = false;
    }
    if (collOpInfo == nullptr || algoCostEntries == nullptr || algoCount == 0) {
        return HCCL_SUCCESS;
    }

    /* 插件未加载时 no-op；g_ops 指向插件静态对象，进程内恒有效 */
    const HcclTunerPluginOps* ops = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_tunerMutex);
        if (g_loadStatus != LOAD_SUCCESS) {
            return HCCL_SUCCESS;
        }
        ops = g_ops;
    }

    /* 不支持的操作类型跳过 */
    if (collOpInfo->collType == HCCL_CMD_INVALID) {
        return HCCL_SUCCESS;
    }

    /* 取通信域 ctx；不存在（Create 未成功/插件 inactive）时 no-op */
    void* memCtx = nullptr;
    uint64_t ctxSize = 0;
    HcclResult ret = HcclEngineCtxGet(comm, TUNER_CTX_TAG, COMM_ENGINE_CPU_TS, &memCtx, &ctxSize);
    if (ret != HCCL_SUCCESS || memCtx == nullptr) {
        return HCCL_SUCCESS;
    }

    bool matched = false;
    auto callStart = std::chrono::steady_clock::now();
    ret = ops->UpdateAlgoCostTable(memCtx, collOpInfo, algoCostEntries, algoCount, &matched);
    auto callMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - callStart).count();
    /* 慢调用检测：连续超过阈值则禁用 tuner，后续 op 回退 CostModel */
    if (static_cast<uint64_t>(callMs) > TUNER_SLOW_CALL_THRESHOLD_MS) {
        uint32_t count = ++g_slowCallCount;
        HCCL_WARNING(
            "[Tuner] UpdateAlgoCostTable took %lldms (threshold %llums), slowCallCount=%u/%u.", callMs,
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
        HCCL_WARNING("[Tuner] UpdateAlgoCostTable failed, ret[%d], ignore plugin modification.", ret);
        return HCCL_SUCCESS;
    }
    if (modified != nullptr) {
        *modified = matched;
    }
    return HCCL_SUCCESS;
}
