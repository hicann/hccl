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

namespace ops_hccl {

AllAlgos* GetAllAlgos()
{
    static AllAlgos globalAllAlgos{nullptr, 0, 0};
    return &globalAllAlgos;
}

HcclResult AddAlgToAllAlgos(
    HcclCMDType opType, const char* algName, const char* executorName, const char** templateName, int templateNum)
{
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
}

CostModelManager::CostModelManager() { InitBandwidth(); }

CostModelManager* CostModelManager::Global()
{
    static CostModelManager* globalCostModelManager = new CostModelManager;
    return globalCostModelManager;
}

void CostModelManager::FreeCostModel(CostModel& costModel)
{
    if (costModel.costAlgoParams != nullptr) {
        for (int i = 0; i < costModel.count; ++i) {
            delete[] costModel.costAlgoParams[i].param;
            costModel.costAlgoParams[i].param = nullptr;
        }
        delete[] costModel.costAlgoParams;
        costModel.costAlgoParams = nullptr;
    }
    costModel.count = 0;
}

void CostModelManager::InitBandwidth()
{
    HCCL_DEBUG("[CostModelManager] InitBandwidth.");
    localCopyBw_ = 750.0f * 1000 * 1000 * 1000;
    localReduceBw_ = 483.0f * 1000 * 1000 * 1000;
    crossChipBw_ = 56.0f * 1000 * 1000 * 1000;
    crossChipReduceBw_ = 56.0f * 1000 * 1000 * 1000;
    ccuLocalCopyBw_ = 200.0f * 1000 * 1000 * 1000;
    ccuLocalReduceBw_ = 160.0f * 1000 * 1000 * 1000;
    ccuCircleLocalCopyBw_ = 47.6f * 1024 * 1024 * 1024;
    ccuCircleLocalReduceBw_ = 47.6f * 1024 * 1024 * 1024;
    HCCL_DEBUG(
        "[CostModelManager] localCopyBw=%f localReduceBw=%f crossChipBw=%f crossChipReduceBw=%f "
        "ccuLocalCopyBw=%f ccuLocalReduceBw=%f ccuCircleLocalCopyBw=%f ccuCircleLocalReduceBw=%f.",
        localCopyBw_, localReduceBw_, crossChipBw_, crossChipReduceBw_, ccuLocalCopyBw_, ccuLocalReduceBw_,
        ccuCircleLocalCopyBw_, ccuCircleLocalReduceBw_);
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

HcclResult CostModelManager::InitCostModel(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, CostModel& costModel)
{
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
        std::unique_ptr<InsCollAlgBase> exec = CollAlgExecRegistryV2::Instance().GetAlgExec(alg.opType, alg.algName);
        if (exec == nullptr) {
            HCCL_WARNING(
                "[CostModelManager] executor not registered, skip algName=%s opType=%d.", alg.algName, alg.opType);
            continue;
        }

        std::vector<CostModelParam> params = exec->CalcCostCoeff(comm, topoInfo, alg.algName);
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
    }

    HCCL_DEBUG("[CostModelManager] InitCostModel done, total=%d calibrated=%d.", algNum, costModel.count);
    return HcclResult::HCCL_SUCCESS;
}

void CostModelManager::CalcMeshParam(float n, AlgNetType netType, int portNum, u32 rankSize, float& A)
{
    // n用来表示传输数据量和总数据量之间的关系
    A = 0.0f;
    if (netType == AlgNetType::MESH) {
        // cost = D/B(write)
        A = n / crossChipBw_;
    } else if (netType == AlgNetType::CLOS) {
        // cost = nD/B(write)
        A = (n * (rankSize - 1)) / (portNum * crossChipBw_);
    } else {
        HCCL_ERROR("[CostModelManager] CalcMeshParams unsupported netType=%d.", static_cast<int>(netType));
    }
    HCCL_DEBUG(
        "[CostModelManager] CalcMeshParams n=%f netType=%d portNum=%d A=%f.", n, static_cast<int>(netType), portNum, A);
    return;
}

void CostModelManager::CalcNHRParams(float n, AlgNetType netType, int portNum, u32 rankSize, float& A)
{
    // n用来表示传输数据和总数据量之间的关系
    // rankSize是指总共通信的rankSize
    A = 0.0f;
    float data = n * (rankSize - 1);
    if (netType == AlgNetType::MESH) {
        A = data / crossChipBw_;
    } else if (netType == AlgNetType::CLOS) {
        A = data / (portNum * crossChipBw_);
    } else {
        HCCL_ERROR("[CostModelManager] CalcNHRParams unsupported netType=%d.", static_cast<int>(netType));
    }
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

AlgNetMetaRegistry* AlgNetMetaRegistry::Global()
{
    static AlgNetMetaRegistry* globalRegistry = new AlgNetMetaRegistry;
    return globalRegistry;
}

void AlgNetMetaRegistry::Register(const std::string& algName, AlgNetMeta meta)
{
    const std::lock_guard<std::mutex> lock(mu_);
    metas_[algName] = meta;
    HCCL_DEBUG(
        "[AlgNetMetaRegistry] register algName=%s netTypes=%zu intraGroupMode=%d groupSizes=%zu.", algName.c_str(),
        meta.netTypes.size(), static_cast<int>(meta.intraGroupMode), meta.groupSizes.size());
}

bool AlgNetMetaRegistry::Query(const std::string& algName, AlgNetMeta& meta) const
{
    const std::lock_guard<std::mutex> lock(mu_);
    auto it = metas_.find(algName);
    if (it == metas_.end()) {
        return false;
    }
    meta = it->second;
    return true;
}

} // namespace ops_hccl
