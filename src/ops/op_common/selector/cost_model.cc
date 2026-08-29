/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cost_model.h"

#include <new>
#include <memory>

#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs_registry.h"
#include "selector_engine.h"
#include "auto_selector_base.h"

namespace ops_hccl {

AllAlgos* GetAllAlgos()
{
#ifndef AICPU_COMPILE
    static AllAlgos globalAllAlgos{nullptr, 0, 0};
    return &globalAllAlgos;
#else
    return nullptr;
#endif
}

HcclResult AddAlgToAllAlgos(
    HcclCMDType opType, const char* algName, const char* executorName, const char** templateName, int templateNum)
{
#ifndef AICPU_COMPILE
    AllAlgos* allAlgos = GetAllAlgos();
    if (allAlgos->count >= allAlgos->capacity) {
        int newCapacity = (allAlgos->capacity == 0) ? 16 : allAlgos->capacity * 2;
        AlgElement* newElements = new (std::nothrow) AlgElement[newCapacity];
        if (newElements == nullptr) {
            HCCL_ERROR("[AllAlgos] alloc failed, newCapacity=%d.", newCapacity);
            return HcclResult::HCCL_E_PARA;
        }
        for (int i = 0; i < allAlgos->count; ++i) {
            newElements[i] = allAlgos->algElements[i];
        }
        delete[] allAlgos->algElements;
        allAlgos->algElements = newElements;
        allAlgos->capacity = newCapacity;
    }
    allAlgos->algElements[allAlgos->count] = {algName, executorName, templateName, templateNum, opType};
    ++allAlgos->count;
    HCCL_DEBUG(
        "[AllAlgos] add algName=%s executorName=%s templateNum=%d opType=%d, total=%d.", algName, executorName,
        templateNum, opType, allAlgos->count);
    return HcclResult::HCCL_SUCCESS;
#else
    (void)opType;
    (void)algName;
    (void)executorName;
    (void)templateName;
    (void)templateNum;
    return HcclResult::HCCL_SUCCESS;
#endif
}

CostModelManager::CostModelManager()
{
#ifndef AICPU_COMPILE
    InitBandwidth();
#endif
}

CostModelManager* CostModelManager::Global()
{
    static CostModelManager* globalCostModelManager = new CostModelManager;
    return globalCostModelManager;
}

void CostModelManager::FreeCostModel(CostModel& costModel)
{
#ifndef AICPU_COMPILE
    if (costModel.costAlgoParams != nullptr) {
        for (int i = 0; i < costModel.count; ++i) {
            delete[] costModel.costAlgoParams[i].param;
            costModel.costAlgoParams[i].param = nullptr;
        }
        delete[] costModel.costAlgoParams;
        costModel.costAlgoParams = nullptr;
    }
    costModel.count = 0;
#else
    (void)costModel;
#endif
}

void CostModelManager::InitBandwidth()
{
#ifndef AICPU_COMPILE
    HCCL_DEBUG("[CostModelManager] InitBandwidth.");
    localCopyBw_ = 750.0f * 1000 * 1000 * 1000;
    localReduceBw_ = 483.0f * 1000 * 1000 * 1000;
    crossChipBw_ = 56.0f * 1000 * 1000 * 1000;
    crossChipReduceBw_ = 56.0f * 1000 * 1000 * 1000;
    ccuLocalCopyBw_ = 200.0f * 1000 * 1000 * 1000;
    ccuLocalReduceBw_ = 35.0f * 1000 * 1000 * 1000;
    ccuCircleLocalCopyBw_ = 47.6f * 1024 * 1024 * 1024;
    ccuCircleLocalReduceBw_ = 47.6f * 1024 * 1024 * 1024;
    HCCL_DEBUG(
        "[CostModelManager] localCopyBw=%f localReduceBw=%f crossChipBw=%f crossChipReduceBw=%f "
        "ccuLocalCopyBw=%f ccuLocalReduceBw=%f ccuCircleLocalCopyBw=%f ccuCircleLocalReduceBw=%f.",
        localCopyBw_, localReduceBw_, crossChipBw_, crossChipReduceBw_, ccuLocalCopyBw_, ccuLocalReduceBw_,
        ccuCircleLocalCopyBw_, ccuCircleLocalReduceBw_);
#endif
}

CostModelManager::RankSizePerLevel
CostModelManager::CalcRankSizeByTopo(const TopoInfoWithNetLayerDetails* topoInfo) const
{
    RankSizePerLevel rs;
    if (topoInfo == nullptr) {
        return rs;
    }
    rs.level0 = topoInfo->userRankSize;
    const auto& nd = topoInfo->netLayerDetails;
    // level0: 每个实例（server内mesh组）的rank数
    if (!nd.instSizeListOfLayer.empty() && !nd.instSizeListOfLayer[0].empty()) {
        rs.level0 = nd.instSizeListOfLayer[0][0];
    }
    // level1: 同序号跨pod的rank数 = 总rank / level0（即pod/server数）
    if (rs.level0 > 0) {
        rs.level1 = topoInfo->userRankSize / rs.level0;
    }
    // level2: 同序号跨super-pod的rank数 = 总rank / (level0 * level1)（即super-pod数）
    if (rs.level0 > 0 && rs.level1 > 0) {
        rs.level2 = topoInfo->userRankSize / (rs.level0 * rs.level1);
    }
    return rs;
}

#ifndef AICPU_COMPILE
__attribute__((unused)) static bool
IsAlgoMatchTopo(const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo)
{
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(algName);
    if (attrs == nullptr) {
        return true;
    }

    const auto& t = attrs->topo;

    if (topoInfo->topoLevelNums < t.minTopoLevelNum) {
        HCCL_INFO(
            "[IsAlgoMatchTopo] algName=%s filtered: topoLevelNums=%u < minTopoLevelNum=%u.", algName.c_str(),
            topoInfo->topoLevelNums, t.minTopoLevelNum);
        return false;
    }
    if (topoInfo->topoLevelNums > t.maxTopoLevelNum) {
        HCCL_INFO(
            "[IsAlgoMatchTopo] algName=%s filtered: topoLevelNums=%u > maxTopoLevelNum=%u.", algName.c_str(),
            topoInfo->topoLevelNums, t.maxTopoLevelNum);
        return false;
    }

    if (!(t.supportLevel0Topos & (1 << static_cast<uint8_t>(topoInfo->level0Topo)))) {
        HCCL_INFO(
            "[IsAlgoMatchTopo] algName=%s filtered: level0Topo=%u not in supportLevel0Topos=0x%02x.", algName.c_str(),
            static_cast<uint8_t>(topoInfo->level0Topo), t.supportLevel0Topos);
        return false;
    }

    if (t.supportLevel0MeshTypes != MESH_TYPE_ANY) {
        if ((topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS)
            && (attrs->engine == OpExecuteConfig::CCU_MS || attrs->engine == OpExecuteConfig::CCU_SCHED)) {
            if (!(t.supportLevel0MeshTypes & (1 << static_cast<uint8_t>(topoInfo->level0MeshType)))) {
                HCCL_INFO(
                    "[IsAlgoMatchTopo] algName=%s filtered: level0MeshType=%u not in supportLevel0MeshTypes=0x%02x.",
                    algName.c_str(), static_cast<uint8_t>(topoInfo->level0MeshType), t.supportLevel0MeshTypes);
                return false;
            }
        }
    }

    if (topoInfo->is2DieFullMesh && !t.isSupport2DieFullMesh) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D
            && (attrs->engine == OpExecuteConfig::CCU_MS || attrs->engine == OpExecuteConfig::CCU_SCHED)) {
            HCCL_INFO(
                "[IsAlgoMatchTopo] algName=%s filtered: is2DieFullMesh=true, isSupport2DieFullMesh=false.",
                algName.c_str());
            return false;
        }
    }

    if (topoInfo->level0PcieMix && !t.isSupportLevel0PcieMix) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS || topoInfo->level0Topo == Level0Shape::CLOS) {
            HCCL_INFO(
                "[IsAlgoMatchTopo] algName=%s filtered: level0PcieMix=true, isSupportLevel0PcieMix=false.",
                algName.c_str());
            return false;
        }
    }

    if (topoInfo->Level1Nhr && !t.isSupportLevel1Nhr) {
        HCCL_INFO("[IsAlgoMatchTopo] algName=%s filtered: Level1Nhr=true, isSupportLevel1Nhr=false.", algName.c_str());
        return false;
    }

    if (t.requireAllMeshConnected && topoInfo->level0PcieMix
        && !AutoSelectorBase::IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
        HCCL_INFO(
            "[IsAlgoMatchTopo] algName=%s filtered: requireAllMeshConnected but not all mesh connected.",
            algName.c_str());
        return false;
    }

    if (!t.supportDevTypes.empty()) {
        bool devTypeMatched = false;
        for (auto devType : t.supportDevTypes) {
            if (devType == topoInfo->deviceType) {
                devTypeMatched = true;
                break;
            }
        }
        if (!devTypeMatched) {
            HCCL_INFO(
                "[IsAlgoMatchTopo] algName=%s filtered: deviceType=%d not in supportDevTypes.", algName.c_str(),
                static_cast<int>(topoInfo->deviceType));
            return false;
        }
    }

    // hostDpuOnly 算法仅在 hostDpuOnly 拓扑下可用，非 hostDpuOnly 算法在 hostDpuOnly
    // 拓扑下不可用，支持dpu算法的cost评估后可放开
    if (t.isHostDpuOnly && !topoInfo->hostDpuOnly) {
        HCCL_INFO(
            "[IsAlgoMatchTopo] algName=%s filtered: isHostDpuOnly=true but topo hostDpuOnly=false.", algName.c_str());
        return false;
    }
    if (!t.isHostDpuOnly && topoInfo->hostDpuOnly) {
        HCCL_INFO("[IsAlgoMatchTopo] algName=%s filtered: hostDpuOnly=true, isHostDpuOnly=false.", algName.c_str());
        return false;
    }

    if (t.topoCustomCheck) {
        if (!t.topoCustomCheck(topoInfo)) {
            HCCL_INFO("[IsAlgoMatchTopo] algName=%s filtered: topoCustomCheck returned false.", algName.c_str());
            return false;
        }
    }

    return true;
}
#endif

#ifndef AICPU_COMPILE
// 从已过滤的算法中筛选优先级算法。按 opType 分组，仅在有 priority 匹配的 opType 内过滤。
__attribute__((unused)) static void ApplyTopoPriority(CostModel& costModel, const TopoInfoWithNetLayerDetails* topoInfo)
{
    // 1. 收集每个 opType 的 priority 匹配索引
    std::map<HcclCMDType, std::vector<int>> priorityByOpType;
    for (int i = 0; i < costModel.count; ++i) {
        const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(costModel.costAlgoParams[i].algName);
        if (attrs != nullptr && attrs->topo.topoPriorityCheck && attrs->topo.topoPriorityCheck(topoInfo)) {
            priorityByOpType[attrs->opType].push_back(i);
            HCCL_INFO("[CostModelManager] topoPriority matched algName=%s.", costModel.costAlgoParams[i].algName);
        }
    }

    // 2. 对有 priority 匹配的 opType，只保留匹配的算法
    std::set<int> toRemove;
    for (auto& [opType, indices] : priorityByOpType) {
        std::set<int> keepSet(indices.begin(), indices.end());
        for (int i = 0; i < costModel.count; ++i) {
            const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(costModel.costAlgoParams[i].algName);
            if (attrs != nullptr && attrs->opType == opType && keepSet.count(i) == 0) {
                toRemove.insert(i);
            }
        }
    }

    if (toRemove.empty()) {
        return;
    }

    int newCount = costModel.count - static_cast<int>(toRemove.size());
    CostAlgoParams* newParams = new (std::nothrow) CostAlgoParams[newCount];
    if (newParams == nullptr) {
        HCCL_ERROR("[CostModelManager] alloc newParams for topoPriority failed.");
        return;
    }
    int j = 0;
    for (int i = 0; i < costModel.count; ++i) {
        if (toRemove.count(i) == 0) {
            newParams[j++] = costModel.costAlgoParams[i];
        }
    }
    delete[] costModel.costAlgoParams;
    costModel.costAlgoParams = newParams;
    costModel.count = newCount;
    HCCL_INFO("[CostModelManager] topoPriority applied, kept=%d.", costModel.count);
}
#endif

HcclResult CostModelManager::InitCostModel(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, CostModel& costModel, const OpParam& param)
{
#ifndef AICPU_COMPILE
    const AllAlgos& allAlgos = *GetAllAlgos();
    int algNum = allAlgos.count;
    if (algNum <= 0) {
        HCCL_WARNING("[CostModelManager] InitCostModel with empty AllAlgos.");
        return HcclResult::HCCL_SUCCESS;
    }

    costModel.costAlgoParams = new (std::nothrow) CostAlgoParams[algNum];
    if (costModel.costAlgoParams == nullptr) {
        HCCL_ERROR("[CostModelManager] alloc CostAlgoParams failed, algNum=%d.", algNum);
        return HcclResult::HCCL_E_PARA;
    }
    costModel.count = 0;

    for (int i = 0; i < algNum; ++i) {
        const AlgElement& alg = allAlgos.algElements[i];
        std::string algName = (alg.algName != nullptr) ? alg.algName : "";

        if (!IsAlgoMatchTopo(algName, topoInfo)) {
            HCCL_INFO("[CostModelManager] algName=%s skipped by topo filter.", algName.c_str());
            continue;
        }

        std::unique_ptr<InsCollAlgBase> exec = CollAlgExecRegistryV2::Instance().GetAlgExec(alg.opType, alg.algName);
        if (exec == nullptr) {
            HCCL_WARNING(
                "[CostModelManager] executor not registered, skip algName=%s opType=%d.", alg.algName, alg.opType);
            continue;
        }

        std::vector<CostModelParam> params = exec->CalcCostCoeff(comm, topoInfo, alg.algName, param);
        if (params.empty()) {
            HCCL_WARNING("[CostModelManager] CalcCostCoeff uncalibrated, skip algName=%s.", alg.algName);
            continue;
        }
        int paramCount = static_cast<int>(params.size());

        // 深拷贝 param 到堆，costModel 持有独立内存所有权
        CostModelParam* ownedParam = new (std::nothrow) CostModelParam[paramCount];
        if (ownedParam == nullptr) {
            HCCL_ERROR("[CostModelManager] alloc ownedParam failed, algName=%s count=%d.", alg.algName, paramCount);
            continue;
        }
        std::copy(params.begin(), params.end(), ownedParam);

        AlgNetMetaRegistry::Global()->Register(alg.algName, exec->GetAlgNetMeta(topoInfo));

        CostAlgoParams cap;
        cap.algName = alg.algName;
        cap.param = ownedParam;
        cap.count = paramCount;
        costModel.costAlgoParams[costModel.count] = cap;
        ++costModel.count;
    }

    if (costModel.count == 0) {
        delete[] costModel.costAlgoParams;
        costModel.costAlgoParams = nullptr;
    } else {
        ApplyTopoPriority(costModel, topoInfo);
    }

    HCCL_INFO("[CostModelManager] InitCostModel done, total=%d calibrated=%d.", algNum, costModel.count);
    return HcclResult::HCCL_SUCCESS;
#else
    (void)comm;
    (void)topoInfo;
    (void)costModel;
    return HcclResult::HCCL_SUCCESS;
#endif
}

void CostModelManager::CalcMeshParam(float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod)
{
    // n用来表示传输数据量和总数据量之间的关系
    A = 0.0f;
    if (isPod && netType == CommTopo::COMM_TOPO_CLOS) {
        portNum = portNum / 2;
    }
    if (netType == CommTopo::COMM_TOPO_1DMESH) {
        // cost = D/B(write)
        A = n / crossChipBw_;
    } else if (netType == CommTopo::COMM_TOPO_CLOS) {
        // cost = nD/B(write)
        A = (n * (rankSize - 1)) / (portNum * crossChipBw_);
    } else {
        HCCL_ERROR("[CostModelManager] CalcMeshParams unsupported netType=%d.", static_cast<int>(netType));
    }
    HCCL_DEBUG(
        "[CostModelManager] CalcMeshParams n=%f netType=%d portNum=%d A=%f.", n, static_cast<int>(netType), portNum, A);
    return;
}

void CostModelManager::CalcNHRParams(float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod)
{
    // n用来表示传输数据和总数据量之间的关系
    // rankSize是指总共通信的rankSize
    A = 0.0f;
    if (isPod && netType == CommTopo::COMM_TOPO_CLOS) {
        portNum = portNum / 2;
    }
    float data = n * (rankSize - 1);
    A = data / (portNum * crossChipBw_);
    HCCL_DEBUG(
        "[CostModelManager] CalcNHRParams n=%f netType=%d portNum=%d A=%f.", n, static_cast<int>(netType), portNum, A);
    return;
}

void CostModelManager::CalcLocalCopyParams(float n, EngineType scene, float& B)
{
    float bw = localCopyBw_;
    if (scene == EngineType::CCU) {
        bw = ccuLocalCopyBw_;
    } else if (scene == EngineType::CCU_CIR_MODE) {
        bw = ccuCircleLocalCopyBw_;
    }
    B = n / bw;
    HCCL_DEBUG("[CostModelManager] CalcLocalCopyParams n=%f scene=%d B=%f.", n, static_cast<int>(scene), B);
    return;
}

void CostModelManager::CalcLocalReduceParams(float n, EngineType scene, float& B)
{
    float bw = localReduceBw_;
    if (scene == EngineType::CCU) {
        bw = ccuLocalReduceBw_;
    } else if (scene == EngineType::CCU_CIR_MODE) {
        bw = ccuCircleLocalReduceBw_;
    }
    B = n / bw;
    HCCL_DEBUG("[CostModelManager] CalcLocalReduceParams n=%f scene=%d B=%f.", n, static_cast<int>(scene), B);
    return;
}

void CostModelManager::CalcLatencyParams(int taskNum, EngineType engine, float& C)
{
    C = 0.0f;
    if (engine == EngineType::AICPU) {
        C = 0.000002 * taskNum; // 和ccu一致，aicpu展开耗时在最后统一取max
    } else if (engine == EngineType::AIV) {
        C = 0.000001 * taskNum;
    } else if (engine == EngineType::CCU) {
        C = 0.000002 * taskNum; // 单位是s，10u
    }
    HCCL_DEBUG("[CostModelManager] CalcLatencyParams taskNum=%d engine=%d C=%f.", taskNum, static_cast<int>(engine), C);
    return;
}

void CostModelManager::CalcLaunchParams(int taskNum, EngineType engine, float& D)
{
    D = 0.0f;
    if (engine == EngineType::AICPU) {
        D = 0.0000005 * taskNum;
    } else if (engine == EngineType::AIV) {
        D = 0;
    } else if (engine == EngineType::CCU) {
        D = 0;
    }
    HCCL_DEBUG("[CostModelManager] CalcLaunchParams taskNum=%d engine=%d D=%f.", taskNum, static_cast<int>(engine), D);
    return;
}

int CostModelManager::CalcTransTaskNum(u32 rankSize) { return static_cast<int>(5 * (rankSize - 1)); }

int CostModelManager::CalcSyncTaskNum(u32 rankSize) { return static_cast<int>(2 * (rankSize - 1)); }

AlgNetMetaRegistry* AlgNetMetaRegistry::Global()
{
#ifndef AICPU_COMPILE
    static AlgNetMetaRegistry* globalRegistry = new AlgNetMetaRegistry;
    return globalRegistry;
#else
    return nullptr;
#endif
}

void AlgNetMetaRegistry::Register(const std::string& algName, AlgNetMeta meta)
{
#ifndef AICPU_COMPILE
    const std::lock_guard<std::mutex> lock(mu_);
    metas_[algName] = meta;
    HCCL_DEBUG(
        "[AlgNetMetaRegistry] register algName=%s netTypes=%zu intraGroupMode=%d groupSizes=%zu.", algName.c_str(),
        meta.netTypes.size(), static_cast<int>(meta.intraGroupMode), meta.groupSizes.size());
#else
    (void)algName;
    (void)meta;
#endif
}

bool AlgNetMetaRegistry::Query(const std::string& algName, AlgNetMeta& meta) const
{
#ifndef AICPU_COMPILE
    const std::lock_guard<std::mutex> lock(mu_);
    auto it = metas_.find(algName);
    if (it == metas_.end()) {
        return false;
    }
    meta = it->second;
    return true;
#else
    (void)algName;
    (void)meta;
    return false;
#endif
}

} // namespace ops_hccl
