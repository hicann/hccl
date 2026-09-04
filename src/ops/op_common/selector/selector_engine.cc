/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selector_engine.h"

#include <cstring>
#include <algorithm>
#include <set>

#include "alg_env_config.h"
#include "alg_parse.h"
#include "algo_name_mapper.h"
#include "config_log.h"
#include "cost_model.h"
#include "cost_table.h"
#include "alg_attrs.h"
#include "alg_attrs_registry.h"
#include "hccl_algo_dims.h"
#include "hccl_common.h"
#include "tuner_setup.h"

namespace ops_hccl {

static constexpr const char* COST_MODEL_TAG = "costmodel";
static constexpr const char* TUNER_INIT_TAG = "tuner_init";

SelectorEngine* SelectorEngine::Global()
{
    static SelectorEngine* globalSelectorEngine = new SelectorEngine;
    return globalSelectorEngine;
}

bool SelectorEngine::IsOpSupported(HcclCMDType opType)
{
    // 本迭代新选择器支持 AllReduce/ReduceScatter/AllGather/Reduce/Scatter/Broadcast/AlltoAll, 其他算子走老流程
    static const std::set<HcclCMDType> supportedOps = {
        HcclCMDType::HCCL_CMD_ALLREDUCE, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclCMDType::HCCL_CMD_REDUCE,    HcclCMDType::HCCL_CMD_SCATTER,        HcclCMDType::HCCL_CMD_BROADCAST,
        HcclCMDType::HCCL_CMD_ALLTOALL,
    };
    return supportedOps.count(opType) > 0;
}

OpExecuteConfig SelectorEngine::GetEngineByAlgName(const std::string& algName)
{
    int count = 0;
    const EnginePrefixEntry* entries = GetEnginePrefixEntries(count);
    for (int i = 0; i < count; ++i) {
        size_t len = strlen(entries[i].pascal);
        if (algName.size() >= len && algName.substr(0, len) == entries[i].pascal) {
            return entries[i].engine;
        }
    }
    return OpExecuteConfig::AICPU_TS;
}

std::vector<std::string> SelectorEngine::CandidateEnginesToPrefixes(const std::vector<OpExecuteConfig>& engines)
{
    std::set<OpExecuteConfig> engineSet(engines.begin(), engines.end());
    std::vector<std::string> prefixes;
    int count = 0;
    const EnginePrefixEntry* entries = GetEnginePrefixEntries(count);
    for (int i = 0; i < count; ++i) {
        if (engineSet.count(entries[i].engine) != 0) {
            prefixes.emplace_back(entries[i].pascal);
        }
    }
    return prefixes;
}

std::vector<OpExecuteConfig> SelectorEngine::GetEnginePriority(OpExecuteConfig opExecuteConfig)
{
    switch (opExecuteConfig) {
        case OpExecuteConfig::CCU_MS:
            return {
                OpExecuteConfig::CCU_MS, OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS,
                OpExecuteConfig::HOSTCPU};
        case OpExecuteConfig::CCU_SCHED:
            return {OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
        case OpExecuteConfig::AIV:
            return {OpExecuteConfig::AIV, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
        case OpExecuteConfig::AIV_ONLY:
            return {OpExecuteConfig::AIV};
        case OpExecuteConfig::AICPU_TS:
            return {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
        case OpExecuteConfig::HOSTCPU:
            return {OpExecuteConfig::HOSTCPU};
        default:
            return {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
    }
}

HcclResult SelectorEngine::FilterCmByEngine(CostModel& cm, const std::vector<OpExecuteConfig>& candidateEngines)
{
    std::set<OpExecuteConfig> engineSet(candidateEngines.begin(), candidateEngines.end());
    for (int i = 0; i < cm.count; i++) {
        if (cm.costAlgoParams[i].algName == nullptr) {
            continue;
        }
        OpExecuteConfig engine = GetEngineByAlgName(cm.costAlgoParams[i].algName);
        if (engineSet.count(engine) == 0) {
            cm.costAlgoParams[i].count = 0;
        }
    }
    return HCCL_SUCCESS;
}

void SelectorEngine::LogAivOnlyNotMatch(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo)
{
    HCCL_ERROR(
        "Failed to select AIV algorithm while configured as AIV_ONLY. "
        "Current topology: topoLevelNums=%u, level0Topo=%u, level0PcieMix=%d, level2UbRtp=%d, "
        "Level1Nhr=%d, userRankSize=%u. opType=%d, dataType=%d, dataSize=%llu, reduceOp=%d.",
        topoInfo->topoLevelNums, static_cast<uint32_t>(topoInfo->level0Topo), static_cast<int>(topoInfo->level0PcieMix),
        static_cast<int>(topoInfo->level2UbRtp), static_cast<int>(topoInfo->Level1Nhr), topoInfo->userRankSize,
        static_cast<int>(param.opType), static_cast<int>(param.DataDes.dataType), param.inputSize,
        static_cast<int>(param.reduceType));

    // 回溯检查所有 AIV 算法被过滤的原因
    HCCL_ERROR("[SelectorEngine] AIV algorithm filter details:");
    const AllAlgos* allAlgos = GetAllAlgos();
    if (allAlgos == nullptr) {
        return;
    }
    for (int i = 0; i < allAlgos->count; ++i) {
        const AlgElement& alg = allAlgos->algElements[i];
        if (alg.opType != param.opType || alg.algName == nullptr) {
            continue;
        }
        const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(alg.algName);
        if (attrs == nullptr || attrs->engine != OpExecuteConfig::AIV) {
            continue;
        }
        std::string algName = alg.algName;

        // topo 层检查
        auto topoResult = CheckAlgoMatchTopoWithReason(algName, topoInfo);
        if (!topoResult.matched) {
            HCCL_ERROR("[SelectorEngine] algName=%s filtered by topo: %s.", algName.c_str(), topoResult.reason.c_str());
            continue;
        }

        // op 层检查
        auto opResult = CheckAlgoMatchOpWithReason(*attrs, param, topoInfo);
        if (!opResult.matched) {
            HCCL_ERROR("[SelectorEngine] algName=%s filtered by op: %s.", algName.c_str(), opResult.reason.c_str());
        } else {
            HCCL_ERROR("[SelectorEngine] algName=%s passed all filters.", algName.c_str());
        }
    }
}

HcclResult
SelectorEngine::InitCostModel(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, OpParam& param, CostModel*& cm)
{
    HCCL_INFO("[SelectorEngine] Initializing costModel for comm, engine=%d.", static_cast<int>(param.opExecuteConfig));

    // 全局一次性：构建算法名→3D 维度映射缓存
    static std::once_flag algoMapperFlag;
    std::call_once(algoMapperFlag, []() {
        AlgoNameMapper::Global()->Init(*GetAllAlgos());
    });

    // 调用 CostModelManager 初始化，costModel 作为出参返回（局部变量，无线程安全问题）
    CostModelManager* costModelMgr = CostModelManager::Global();
    CostModel srcCm{nullptr, 0};
    CHK_RET(costModelMgr->InitCostModel(comm, topoInfo, srcCm, param));

    // CostModel 含指针(costAlgoParams),分段拷贝: [CostModel header][CostAlgoParams array]
    void* ctxPtr = nullptr;
    uint64_t flatSize = sizeof(CostModel) + static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams);
    std::string costModelTag = std::string(COST_MODEL_TAG) + "_" + ENGINE_STR_MAP.at(param.opExecuteConfig);
    CHK_RET(HcclEngineCtxCreate(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, flatSize, &ctxPtr));

    CostModel* storedCm = static_cast<CostModel*>(ctxPtr);
    storedCm->count = srcCm.count;
    storedCm->costAlgoParams = reinterpret_cast<CostAlgoParams*>(storedCm + 1);
    if (srcCm.count > 0 && srcCm.costAlgoParams != nullptr) {
        CHK_SAFETY_FUNC_RET(memcpy_s(
            storedCm->costAlgoParams, static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams), srcCm.costAlgoParams,
            static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams)));
    }

    // 深拷贝 param 指针指向的内存，comm ctx 持有独立所有权
    for (int i = 0; i < storedCm->count; ++i) {
        int n = storedCm->costAlgoParams[i].count;
        const CostModelParam* srcParam = storedCm->costAlgoParams[i].param;
        if (n > 0 && srcParam != nullptr) {
            CostModelParam* owned = new (std::nothrow) CostModelParam[n];
            if (owned == nullptr) {
                HCCL_ERROR("[SelectorEngine] alloc param failed, i=%d count=%d.", i, n);
                return HcclResult::HCCL_E_PARA;
            }
            CHK_SAFETY_FUNC_RET(memcpy_s(
                owned, static_cast<uint64_t>(n) * sizeof(CostModelParam), srcParam,
                static_cast<uint64_t>(n) * sizeof(CostModelParam)));
            storedCm->costAlgoParams[i].param = owned;
        }
    }

    // 释放临时 srcCm（comm ctx 已有独立深拷贝副本）
    CostModelManager::FreeCostModel(srcCm);

    // 根据候选引擎过滤 costModel,再调用 algo 模块按 HCCL_ALGO 配置过滤
    std::vector<OpExecuteConfig> candidateEngines = GetEnginePriority(param.opExecuteConfig);
    CHK_RET(FilterCmByEngine(*storedCm, candidateEngines));
    std::vector<std::string> candidatePrefixes = CandidateEnginesToPrefixes(candidateEngines);
    CHK_RET(FilterCmByHcclAlgo(comm, *storedCm, candidatePrefixes));

    cm = storedCm;

    HCCL_INFO("[SelectorEngine] costModel initialized and stored in comm ctx, count=%d.", storedCm->count);
    return HCCL_SUCCESS;
}

HcclResult SelectorEngine::TunerEnrichCostTable(
    HcclComm comm, CostModel* cm, CostTable& ct, TopoInfoWithNetLayerDetails* topoInfo, OpParam& param)
{
    CHK_RET(CostTableManager::Global()->CostTableGen(*cm, ct, topoInfo, param));

    if (ct.count > 0 && HcclTunerIsLoaded()) {
        // tuner: Enrich 填 3D 名 + 调用插件改 cost
        AlgoNameMapper::Global()->Enrich(ct.costs, ct.count);
        bool tunerModified = false;
        // AllToAll(V/VC) 的 dataType 存在 all2AllVDataDes.sendType 中，而非 DataDes.dataType
        HcclDataType tunerDataType = param.DataDes.dataType;
        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
            || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
            tunerDataType = param.all2AllVDataDes.sendType;
        }
        CHK_RET(HcclTunerCallGetCollInfo(
            comm, param.opType, param.inputSize, tunerDataType, ct.costs, ct.count, &tunerModified));
        if (tunerModified) {
            HCCL_INFO("[SelectorEngine] tuner modified cost table.");
        } else {
            HCCL_INFO("[SelectorEngine] tuner did not modify cost table, using CostModel selection.");
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
SelectorEngine::Run(HcclComm comm, OpParam& param, TopoInfoWithNetLayerDetails* topoInfo, std::string& algName)
{
    HCCL_INFO(
        "[SelectorEngine] Run start, opType=%d, opExecuteConfig=%d.", static_cast<int>(param.opType),
        static_cast<int>(param.opExecuteConfig));

    // step 0: tuner 初始化（每通信域仅一次，独立于 costModel 引擎副本）
    void* tunerCtxPtr = nullptr;
    uint64_t tunerCtxSize = 0;
    if (HcclEngineCtxGet(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, &tunerCtxPtr, &tunerCtxSize)
        != HCCL_SUCCESS) {
        CHK_RET(HcclTunerInit(comm, topoInfo));
        CHK_RET(HcclEngineCtxCreate(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, 1, &tunerCtxPtr));
    }

    // step 1: 从通信域 ctx 获取或初始化 costModel（按引擎区分 tag，回退时引擎变更会命中不同副本）
    CostModel* cm = nullptr;
    void* ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    std::string costModelTag = std::string(COST_MODEL_TAG) + "_" + ENGINE_STR_MAP.at(param.opExecuteConfig);
    if (HcclEngineCtxGet(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, &ctxPtr, &ctxSize) == HCCL_SUCCESS) {
        cm = static_cast<CostModel*>(ctxPtr);
        HCCL_DEBUG("[SelectorEngine] costModel found in comm ctx, count=%d.", cm->count);
    } else {
        CHK_RET(InitCostModel(comm, topoInfo, param, cm));
    }

    // step 2: 生成 costTable 并调 tuner 改 cost
    CostTable ct{nullptr, 0};
    CHK_RET(TunerEnrichCostTable(comm, cm, ct, topoInfo, param));

    // step 3: min(ct)
    HcclResult ret = SelectMinCost(ct, param, algName);

    delete[] ct.costs;
    ct.costs = nullptr;
    ct.count = 0;

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[SelectorEngine] Run failed, no algorithm selected.");
        if (param.opExecuteConfig == OpExecuteConfig::AIV_ONLY) {
            LogAivOnlyNotMatch(param, topoInfo);
        }
        return ret;
    }

    LogSelectedAlgo(param, topoInfo, algName);

    return HCCL_SUCCESS;
}

void SelectorEngine::LogSelectedAlgo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const std::string& algName)
{
    HCCL_INFO(
        "[SelectorEngine] The opExecuteConfig is %s, the selected algo type is %s",
        ENGINE_STR_MAP.at(param.opExecuteConfig), algName.c_str());

    HCCL_CONFIG_INFO(
        HCCL_ALG,
        "op[%s] algName[%s] engine[%s] executor[%s] templates[%s] "
        "opMode[%d] deterministic[%u] isCapture[%d] "
        "dataSize[%llu] dataType[%s] reduceType[%s] root[%u] "
        "userRank[%u] rankSize[%u] serverNum[%u] superPodNum[%u] "
        "topoLevelNums[%u] level0Topo[%d] level0MeshType[%d] level0PcieMix[%d] "
        "deviceType[%d] commName[%s] enableDetour[%d] symMem[%d]",
        HcclCMDTypeToString(param.opType).c_str(), algName.c_str(), ENGINE_STR_MAP.at(param.opExecuteConfig),
        QueryExecutorName(algName).c_str(), QueryTemplateInfo(algName).c_str(), static_cast<int>(param.opMode),
        GetExternalInputHcclDeterministic(), static_cast<int>(param.isCapture), param.inputSize,
        GetDataTypeEnumStr(param.DataDes.dataType).c_str(), GetReduceOpEnumStr(param.reduceType).c_str(), param.root,
        topoInfo->userRank, topoInfo->userRankSize, topoInfo->serverNum, topoInfo->superPodNum, topoInfo->topoLevelNums,
        static_cast<int>(topoInfo->level0Topo), static_cast<int>(topoInfo->level0MeshType),
        static_cast<int>(topoInfo->level0PcieMix), static_cast<int>(param.deviceType), param.commName,
        static_cast<int>(param.enableDetour), static_cast<int>(param.supportSymmetricMemory));
}

std::string SelectorEngine::QueryTemplateInfo(const std::string& algName)
{
    std::string templateInfo;
    AllAlgos* allAlgos = GetAllAlgos();
    if (allAlgos == nullptr) {
        return templateInfo;
    }
    for (int i = 0; i < allAlgos->count; ++i) {
        if (allAlgos->algElements[i].algName != nullptr && algName == allAlgos->algElements[i].algName) {
            for (int t = 0; t < allAlgos->algElements[i].templateNum; ++t) {
                if (t > 0) {
                    templateInfo += ",";
                }
                templateInfo += allAlgos->algElements[i].templateName[t];
            }
            break;
        }
    }
    return templateInfo;
}

std::string SelectorEngine::QueryExecutorName(const std::string& algName)
{
    AllAlgos* allAlgos = GetAllAlgos();
    if (allAlgos == nullptr) {
        return "";
    }
    for (int i = 0; i < allAlgos->count; ++i) {
        if (allAlgos->algElements[i].algName != nullptr && algName == allAlgos->algElements[i].algName) {
            return allAlgos->algElements[i].executorName != nullptr ? allAlgos->algElements[i].executorName : "";
        }
    }
    return "";
}

HcclResult SelectorEngine::SelectMinCost(const CostTable& ct, OpParam& param, std::string& algName)
{
    if (ct.count <= 0) {
        HCCL_ERROR(
            "[SelectorEngine] SelectMinCost: costTable is empty, opType=%d, dataSize=%llu.",
            static_cast<int>(param.opType), param.inputSize);
        return HCCL_E_NOT_SUPPORT;
    }

    // 遍历找最小 cost 并打印 costTable 明细, 格式: | idx | algName | engine | cost | status |
    HCCL_INFO(
        "[SelectorEngine] SelectMinCost: costTable count=%d, opType=%d, dataSize=%llu.", ct.count,
        static_cast<int>(param.opType), param.inputSize);
    HCCL_INFO("[SelectorEngine] "
              "+-----+--------------------------------------------------+----------+--------------+----------+");
    HCCL_INFO("[SelectorEngine] | idx | algName                                          | engine   | cost         | "
              "status   |");
    HCCL_INFO("[SelectorEngine] "
              "+-----+--------------------------------------------------+----------+--------------+----------+");
    int minIdx = -1;
    float minCost = 0.0f;
    std::vector<std::string> tiedAlgos;
    for (int i = 0; i < ct.count; ++i) {
        const char* name = ct.costs[i].algName;
        float cost = ct.costs[i].cost;
        std::string status = (name == nullptr || cost < 0.0f) ? "filtered" : "valid";
        std::string engineStr = name == nullptr ? "-" : ENGINE_STR_MAP.at(GetEngineByAlgName(name));
        std::string nameStr = name != nullptr ? name : "-";
        char costBuf[32];
        const char* fmt = (cost >= 0.0f && cost < 1.0f) ? "%.6f" : "%.2f";
        int costRet = sprintf_s(costBuf, sizeof(costBuf), fmt, cost);
        if (costRet < 0) {
            HCCL_ERROR("[SelectorEngine] SelectMinCost: sprintf_s failed.");
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO(
            "[SelectorEngine] | %3d | %-48s | %-8s | %12s | %-8s |", i, nameStr.substr(0, 48).c_str(),
            engineStr.substr(0, 8).c_str(), costBuf, status.c_str());

        if (name == nullptr || cost < 0.0f) {
            continue;
        }
        if (minIdx == -1 || cost < minCost) {
            minIdx = i;
            minCost = cost;
            tiedAlgos.clear();
            tiedAlgos.emplace_back(name);
        } else if (cost == minCost) {
            tiedAlgos.emplace_back(name);
        }
    }
    HCCL_INFO("[SelectorEngine] "
              "+-----+--------------------------------------------------+----------+--------------+----------+");

    if (minIdx < 0) {
        HCCL_ERROR(
            "[SelectorEngine] SelectMinCost: no valid algorithm found, expansionMode=%d, costTable count=%d.",
            static_cast<int>(param.commOpExpansionMode), ct.count);
        return HCCL_E_NOT_SUPPORT;
    }

    if (tiedAlgos.size() > 1) {
        std::string algoNames;
        for (size_t i = 0; i < tiedAlgos.size(); ++i) {
            if (i > 0) {
                algoNames += ", ";
            }
            algoNames += tiedAlgos[i];
        }
        HCCL_WARNING(
            "[SelectorEngine] multiple algos with same cost=%f: [%s], selecting %s.", minCost, algoNames.c_str(),
            tiedAlgos[0].c_str());
    }

    algName = ct.costs[minIdx].algName;
    param.opExecuteConfig = GetEngineByAlgName(algName);
    HCCL_INFO(
        "[SelectorEngine] SelectMinCost: selected algName=%s, engine=%s, cost=%f.", algName.c_str(),
        ENGINE_STR_MAP.at(param.opExecuteConfig), minCost);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
