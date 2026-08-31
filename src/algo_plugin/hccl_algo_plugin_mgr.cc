/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_algo_plugin_mgr.h"

#include <cstdlib>
#include <dlfcn.h>
#include <cstring>

#include "log.h"

namespace ops_hccl {

namespace {
    constexpr const char* kPluginPathEnv = "HCCL_ALGO_PLUGIN_PATH";
}

HcclAlgoPluginMgr& HcclAlgoPluginMgr::Instance()
{
    static HcclAlgoPluginMgr instance;
    return instance;
}

HcclResult HcclAlgoPluginMgr::Init()
{
    std::call_once(initFlag_, [this]() {
        std::lock_guard<std::mutex> lock(mutex_);
        InitPlugin();
    });
    return HCCL_SUCCESS;
}

void HcclAlgoPluginMgr::InitPlugin()
{
    const char* pluginPath = std::getenv(kPluginPathEnv);
    if (pluginPath == nullptr || pluginPath[0] == '\0') {
        HCCL_INFO("[HcclAlgoPluginMgr] HCCL_ALGO_PLUGIN_PATH not set, plugin framework disabled, "
                  "HCCL behavior falls back to original logic entirely.");
        loaded_ = false;
        return;
    }

    HcclAlgoPlugin_t* table = LoadPluginTable(pluginPath);
    if (table == nullptr || !ValidatePluginTable(table)) {
        Unload();
        return;
    }

    pluginTable_ = table;
    ctx_ = table->FetchContext();
    loaded_ = true;
    HCCL_INFO("[HcclAlgoPluginMgr] PluginBroker loaded successfully from %s", pluginPath);
}

HcclAlgoPlugin_t* HcclAlgoPluginMgr::LoadPluginTable(const char* pluginPath)
{
    /*
     * dlopen加载动作本身即触发PluginBroker动态库内部全局静态对象的构造函数，
     * 自动完成算子根目录扫描与全局算法注册表构建，无需显式调用初始化接口。
     */
    soHandle_ = dlopen(pluginPath, RTLD_NOW | RTLD_LOCAL);
    if (soHandle_ == nullptr) {
        HCCL_WARNING(
            "[HcclAlgoPluginMgr] dlopen PluginBroker failed: %s, dlerror=%s. "
            "Fallback to original HCCL algorithm selection/execution logic.",
            pluginPath, dlerror());
        return nullptr;
    }

    using GetTableFn = HcclAlgoPlugin_t* (*)(void);
    auto getTableFn = reinterpret_cast<GetTableFn>(dlsym(soHandle_, HCCL_ALGO_PLUGIN_GET_TABLE_SYMBOL));
    if (getTableFn == nullptr) {
        HCCL_WARNING(
            "[HcclAlgoPluginMgr] dlsym GetHcclAlgoPlugin failed: %s. "
            "Reject this PluginBroker, fallback to original logic.",
            dlerror());
        return nullptr;
    }

    HcclAlgoPlugin_t* table = getTableFn();
    if (table == nullptr) {
        HCCL_WARNING("[HcclAlgoPluginMgr] GetHcclAlgoPlugin returned nullptr, reject PluginBroker.");
    }

    return table;
}

bool HcclAlgoPluginMgr::ValidatePluginTable(const HcclAlgoPlugin_t* table) const
{
    if (table->version != HCCL_PLUGIN_API_VERSION) {
        HCCL_WARNING(
            "[HcclAlgoPluginMgr] PluginBroker version mismatch: expect=%u, actual=%u. "
            "Reject this PluginBroker to avoid loading illegal/corrupted module.",
            HCCL_PLUGIN_API_VERSION, table->version);
        return false;
    }

    if (table->IsReady == nullptr || table->FetchContext == nullptr || table->SelectAlg == nullptr
        || table->ExecuteAlg == nullptr || table->QueryAlgs == nullptr) {
        HCCL_WARNING("[HcclAlgoPluginMgr] PluginBroker function table has null entries, reject.");
        return false;
    }

    if (!table->IsReady()) {
        HCCL_WARNING("[HcclAlgoPluginMgr] PluginBroker auto-init not ready "
                     "(HCCL_PLUGIN_ALG_DIR unset/invalid/untrusted or version mismatch inside PluginBroker). "
                     "Fallback to original logic.");
        return false;
    }

    return true;
}

void HcclAlgoPluginMgr::Unload()
{
    if (soHandle_ != nullptr) {
        dlclose(soHandle_);
        soHandle_ = nullptr;
    }
    pluginTable_ = nullptr;
    ctx_ = nullptr;
    loaded_ = false;
}

HcclAlgoPlugin_t* HcclAlgoPluginMgr::GetPlugin() { return loaded_ ? pluginTable_ : nullptr; }

void* HcclAlgoPluginMgr::GetContext() { return loaded_ ? ctx_ : nullptr; }

bool HcclAlgoPluginMgr::IsLoaded() const { return loaded_; }

HcclAlgoPluginMgr::~HcclAlgoPluginMgr()
{
    /*
     * PluginBroker及自定义算法.so的注册表/句柄为进程级资源，
     * 不与任一通信域生命周期绑定，此处不主动dlclose，交由进程退出时自然释放，
     * 避免其他仍在使用ctx_的通信域在进程退出前的收尾阶段访问已失效指针。
     */
}

namespace {
    // 独立实现，按OpParam自身的union语义提取本插件框架实际需要的count/dataType。
    // 注意：AllToAllV/AllToAllVC/AllGatherV/ReduceScatterV/BatchSendRecv这几个算子涉及
    // per-rank变长/多item参数，无法用一个标量count表达，且sdk/hccl_algo_plugin_sdk.h
    // 目前没有为它们定义标准执行函数签名（ExecuteAlg命中时会返回HCCL_E_NOT_SUPPORT），
    // 因此这里对它们保持count=0/dataType默认值，不做展开读取，避免维护一份没有消费方的解析逻辑。
    void ExtractCountAndDataType(const OpParam& param, uint64_t& count, HcclDataType& dataType)
    {
        count = 0;
        dataType = param.DataDes.dataType; // 默认按定长算子读取

        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
            // 非V的AllToAll在OpParam里约定用all2AllVDataDes表达（各Rank收发count相同，取首元素）
            if (param.all2AllVDataDes.sendCounts != nullptr) {
                count = *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts));
            }
            dataType = static_cast<HcclDataType>(param.all2AllVDataDes.sendType);
            return;
        }
        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC
            || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V
            || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V
            || param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            count = 0; // 暂不支持，见函数头注释
            return;
        }
        // 其余（Send/Recv/Broadcast/AllReduce/Reduce/AllGather/ReduceScatter/Scatter/Barrier等）
        // 定长算子，count就在DataDes.count里，函数开头已经取了DataDes.count/dataType默认值
        count = param.DataDes.count;
    }

    const char* OpTypeToName(HcclCMDType opType)
    {
        switch (opType) {
            case HcclCMDType::HCCL_CMD_BROADCAST:
                return "Broadcast";
            case HcclCMDType::HCCL_CMD_ALLREDUCE:
                return "AllReduce";
            case HcclCMDType::HCCL_CMD_REDUCE:
                return "Reduce";
            case HcclCMDType::HCCL_CMD_SEND:
                return "Send";
            case HcclCMDType::HCCL_CMD_RECEIVE:
                return "Recv";
            case HcclCMDType::HCCL_CMD_ALLGATHER:
                return "AllGather";
            case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
                return "ReduceScatter";
            case HcclCMDType::HCCL_CMD_ALLTOALL:
                return "AllToAll";
            case HcclCMDType::HCCL_CMD_SCATTER:
                return "Scatter";
            default:
                return "";
        }
    }

    const char* TopoShapeToName(Level0Shape shape)
    {
        switch (shape) {
            case Level0Shape::CLOS:
                return "CLOS";
            case Level0Shape::MESH_1D:
                return "MESH_1D";
            case Level0Shape::MESH_1D_CLOS:
                return "MESH_1D_CLOS";
            default:
                return "";
        }
    }

    void SafeCopyStr(char* dst, size_t dstLen, const char* src)
    {
        if (dst == nullptr || dstLen == 0) {
            return;
        }

        if (src == nullptr) {
            dst[0] = '\0';
            return;
        }

        size_t i = 0;
        while (i + 1 < dstLen && src[i] != '\0') {
            dst[i] = src[i];
            ++i;
        }

        dst[i] = '\0';
    }
} // namespace

void FillHcclAlgoPluginParam(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, HcclAlgoPluginParam& pluginParam)
{
    HcclAlgoPluginParamInit(&pluginParam);
    ExtractCountAndDataType(param, pluginParam.count, pluginParam.dataType);
    pluginParam.reduceOp = param.reduceType;
    pluginParam.opType = static_cast<int>(param.opType);
    SafeCopyStr(pluginParam.opName, sizeof(pluginParam.opName), OpTypeToName(param.opType));
    pluginParam.root = param.root;
    pluginParam.sendBuf = param.inputPtr;
    pluginParam.recvBuf = param.outputPtr;
    pluginParam.stream = param.stream;
    pluginParam.remoteRank = param.sendRecvRemoteRank;
    if (topoInfo != nullptr) {
        pluginParam.topoType = static_cast<int>(topoInfo->level0Topo);
        SafeCopyStr(pluginParam.topoName, sizeof(pluginParam.topoName), TopoShapeToName(topoInfo->level0Topo));
        pluginParam.rankNum = topoInfo->userRankSize;
        pluginParam.serverNum = topoInfo->serverNum;
        pluginParam.deviceNumPerModule = topoInfo->deviceNumPerModule;
        pluginParam.moduleNum = topoInfo->moduleNum;
        pluginParam.superPodNum = topoInfo->superPodNum;
        pluginParam.serverNumPerSuperPod = topoInfo->serverNumPerSuperPod;
        // 三个内部非对称拓扑标记聚合成一个对外语义，不直接暴露HCCL内部字段名，保持HcclAlgoPluginParam与HCCL内部结构解耦。
        pluginParam.isAsymmetricTopo = topoInfo->isDiffDeviceModule || topoInfo->multiModuleDiffDeviceNumMode
                                       || topoInfo->multiSuperPodDiffServerNumMode;
    } else {
        pluginParam.topoType = 0;
        pluginParam.topoName[0] = '\0';
        pluginParam.rankNum = 0;
        pluginParam.serverNum = 0;
        pluginParam.deviceNumPerModule = 0;
        pluginParam.moduleNum = 0;
        pluginParam.superPodNum = 0;
        pluginParam.serverNumPerSuperPod = 0;
        pluginParam.isAsymmetricTopo = false;
    }
}

} // namespace ops_hccl
