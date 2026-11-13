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

#include <algorithm>
#include <new>
#include <memory>
#include <set>

#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
#include "alg_parse.h"
// tuner 插件查询(HcclTunerIsLoaded)仅 host 侧选路管线(InitModel 等 phase)使用(已在 AICPU_COMPILE 保护内);
// device 核库(hccl_aicpu_kernel)的 include 路径不含 src/common/tuner, 无保护会 fatal error(2026-09-21 PR#3286 CI
// 实证)
#ifndef AICPU_COMPILE
#include "tuner_setup.h"
#include "algo_name_mapper.h"
#endif

namespace ops_hccl {

// Level1Nhr 拓扑检查跳过的算子集合
static const std::set<HcclCMDType> NO_L1NHR_CHECK_OPS
    = {HcclCMDType::HCCL_CMD_ALLTOALL, HcclCMDType::HCCL_CMD_ALLTOALLV,   HcclCMDType::HCCL_CMD_ALLTOALLVC,
       HcclCMDType::HCCL_CMD_SEND,     HcclCMDType::HCCL_CMD_RECEIVE,     HcclCMDType::HCCL_CMD_BATCH_SEND_RECV,
       HcclCMDType::HCCL_CMD_BARRIER,  HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V};

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
    HCCL_INFO(
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
    InitDpuSftCost();
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
    HCCL_INFO("[CostModelManager] InitBandwidth.");
    localCopyBw_ = 1400.0f * 1000 * 1000 * 1000;
    localReduceBw_ = 900.0f * 1000 * 1000 * 1000;
    crossChipBw_ = 56.0f * 1000 * 1000 * 1000;
    crossChipReduceBw_ = 56.0f * 1000 * 1000 * 1000;
    roceBw_ = 25.0f * 1000 * 1000 * 1000;
    ccuLocalCopyBw_ = 200.0f * 1000 * 1000 * 1000;
    ccuLocalReduceBw_ = 35.0f * 1000 * 1000 * 1000;
    ccuCircleLocalCopyBw_ = 47.6f * 1024 * 1024 * 1024;
    ccuCircleLocalReduceBw_ = 47.6f * 1024 * 1024 * 1024;
    HCCL_INFO(
        "[CostModelManager] localCopyBw=%f localReduceBw=%f crossChipBw=%f crossChipReduceBw=%f roceBw=%f"
        "ccuLocalCopyBw=%f ccuLocalReduceBw=%f ccuCircleLocalCopyBw=%f ccuCircleLocalReduceBw=%f.",
        localCopyBw_, localReduceBw_, crossChipBw_, crossChipReduceBw_, roceBw_, ccuLocalCopyBw_, ccuLocalReduceBw_,
        ccuCircleLocalCopyBw_, ccuCircleLocalReduceBw_);
#endif
}

void CostModelManager::InitDpuSftCost()
{
    preSync_ = 0.000016;
    channelFence_ = 0.000002;
    threadFence_ = 0.000006;
    sndRcvCall_ = 0.000006;
    hDLatency_ = 0.000014; // 单位是s，14u，包括D2H和H2D的处理
    HCCL_INFO(
        "[CostModelManager] InitDpuSftCost preSync=%f channelFence=%f threadFence=%f sndRcvCall=%f hDLatency=%f.",
        preSync_, channelFence_, threadFence_, sndRcvCall_, hDLatency_);
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
TopoMatchResult CheckAlgoMatchTopoWithReason(
    const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo, bool needSoftPolicyCheck)
{
    TopoMatchResult result;
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(algName);
    if (attrs == nullptr) {
        // 无 attrs 属注册异常, 视为不匹配(正常路径已在 step2 提前过滤)
        result.matched = false;
        result.reason = "no attrs";
        return result;
    }

    const auto& t = attrs->topo;

    if (topoInfo->topoLevelNums < t.minTopoLevelNum) {
        result.matched = false;
        result.reason = "topoLevelNums=" + std::to_string(topoInfo->topoLevelNums)
                        + " < minTopoLevelNum=" + std::to_string(t.minTopoLevelNum);
        return result;
    }
    if (topoInfo->topoLevelNums > t.maxTopoLevelNum) {
        result.matched = false;
        result.reason = "topoLevelNums=" + std::to_string(topoInfo->topoLevelNums)
                        + " > maxTopoLevelNum=" + std::to_string(t.maxTopoLevelNum);
        return result;
    }

    if (!(t.supportLevel0Topos & (1 << static_cast<uint8_t>(topoInfo->level0Topo)))) {
        result.matched = false;
        result.reason = "level0Topo=" + std::to_string(static_cast<uint8_t>(topoInfo->level0Topo))
                        + " not in supportLevel0Topos=0x" + std::to_string(t.supportLevel0Topos);
        return result;
    }

    if (t.supportLevel0MeshTypes != MESH_TYPE_ANY) {
        if ((topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS)
            && (attrs->engine == OpExecuteConfig::CCU_MS || attrs->engine == OpExecuteConfig::CCU_SCHED)) {
            if (!(t.supportLevel0MeshTypes & (1 << static_cast<uint8_t>(topoInfo->level0MeshType)))) {
                result.matched = false;
                result.reason = "level0MeshType=" + std::to_string(static_cast<uint8_t>(topoInfo->level0MeshType))
                                + " not in supportLevel0MeshTypes=0x" + std::to_string(t.supportLevel0MeshTypes);
                return result;
            }
        }
    }

    if (topoInfo->is2DieFullMesh && !t.isSupport2DieFullMesh) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D
            && (attrs->engine == OpExecuteConfig::CCU_MS || attrs->engine == OpExecuteConfig::CCU_SCHED)) {
            result.matched = false;
            result.reason = "is2DieFullMesh=true, isSupport2DieFullMesh=false";
            return result;
        }
    }

    if (topoInfo->level0PcieMix && !t.isSupportLevel0PcieMix) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS || topoInfo->level0Topo == Level0Shape::CLOS) {
            result.matched = false;
            result.reason = "level0PcieMix=true, isSupportLevel0PcieMix=false";
            return result;
        }
    }

    if (NO_L1NHR_CHECK_OPS.find(attrs->opType) == NO_L1NHR_CHECK_OPS.end() && topoInfo->Level1Nhr
        && !t.isSupportLevel1Nhr) {
        result.matched = false;
        result.reason = "Level1Nhr=true, isSupportLevel1Nhr=false";
        return result;
    }

    if (t.requireAllMeshConnected && topoInfo->level0PcieMix
        && !AutoSelectorBase::IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
        result.matched = false;
        result.reason = "requireAllMeshConnected but not all mesh connected";
        return result;
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
            result.matched = false;
            result.reason
                = "deviceType=" + std::to_string(static_cast<int>(topoInfo->deviceType)) + " not in supportDevTypes";
            return result;
        }
    }

    if (t.isHostDpuOnly && !topoInfo->hostDpuOnly) {
        result.matched = false;
        result.reason = "isHostDpuOnly=true but topo hostDpuOnly=false";
        return result;
    }
    if (!t.isHostDpuOnly && topoInfo->hostDpuOnly) {
        result.matched = false;
        result.reason = "hostDpuOnly=true, isHostDpuOnly=false";
        return result;
    }

    if (t.maxSupportRankSize != 0 && topoInfo->userRankSize > t.maxSupportRankSize) {
        result.matched = false;
        result.reason = "userRankSize=" + std::to_string(topoInfo->userRankSize)
                        + " > maxSupportRankSize=" + std::to_string(t.maxSupportRankSize);
        return result;
    }

    // 硬前提：无条件执行，preferred/tuner 均不可绕过
    if (t.topoPrecondCheck) {
        if (!t.topoPrecondCheck(topoInfo)) {
            result.matched = false;
            result.reason = "topoPrecondCheck returned false (hard precondition)";
            return result;
        }
    }

    if (needSoftPolicyCheck && t.topoCustomCheck) {
        if (!t.topoCustomCheck(topoInfo)) {
            result.matched = false;
            result.reason = "topoCustomCheck returned false";
            return result;
        }
    }

    return result;
}

bool IsAlgoMatchTopo(const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo, bool needSoftPolicyCheck)
{
    auto result = CheckAlgoMatchTopoWithReason(algName, topoInfo, needSoftPolicyCheck);
    if (!result.matched) {
        HCCL_INFO("[IsAlgoMatchTopo] algName=%s filtered: %s.", algName.c_str(), result.reason.c_str());
    }
    return result.matched;
}
#endif

#ifndef AICPU_COMPILE
HcclResult CostModelManager::GenerateCostModel(
    HcclComm comm, CostModel& cm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<OpExecuteConfig>& candidateEngines, const std::vector<std::string>& candidatePrefixes)
{
    // 通信域阶段选路管线: step1-5 按序执行
    CHK_RET(InitModel(cm));
    FilterByEngine(cm, candidateEngines);
    CHK_RET(ApplyHcclAlgoPriority(comm, cm, candidatePrefixes));
    CHK_RET(FilterByTopoAndCalibrate(comm, cm, param, topoInfo));
    ApplyTopoPriority(cm, topoInfo);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CostModelManager::BuildAllAlgoModel(CostModel& cm) { return InitModel(cm); }

HcclResult CostModelManager::InitModel(CostModel& cm)
{
    const AllAlgos& allAlgos = *GetAllAlgos();
    if (allAlgos.count <= 0) {
        HCCL_WARNING("[CostModelManager] InitModel with empty AllAlgos.");
        return HcclResult::HCCL_SUCCESS;
    }

    cm.costAlgoParams = new (std::nothrow) CostAlgoParams[allAlgos.count]();
    if (cm.costAlgoParams == nullptr) {
        HCCL_ERROR("[CostModelManager] alloc CostAlgoParams failed, algNum=%d.", allAlgos.count);
        return HcclResult::HCCL_E_PARA;
    }
    // 条目与 allAlgos 对齐, count=1 待处理
    // 3D 名 per-comm 一次物化(替代旧每 op 的 TunerEnrichCostTable 内 Enrich 查找); 仅 host——
    // device 核库(AICPU_COMPILE)不含 AlgoNameMapper(include 见文件头保护, PR#3286 同款教训),
    // device 条目三字段留 nullptr, 由 cost_table 生成侧兜底为 ""
#ifndef AICPU_COMPILE
    // Init 幂等(call_once), 此处先确保映射缓存就绪再逐条 Lookup
    AlgoNameMapper::Global()->Init(allAlgos);
#endif
    for (int i = 0; i < allAlgos.count; ++i) {
        cm.costAlgoParams[i].algName = allAlgos.algElements[i].algName;
        cm.costAlgoParams[i].count = 1;
#ifndef AICPU_COMPILE
        AlgoDims dims = {};
        if (!AlgoNameMapper::Global()->Lookup(allAlgos.algElements[i].algName, dims)) {
            HCCL_WARNING(
                "[CostModelManager] InitModel 3D-name lookup miss, algName=%s.",
                allAlgos.algElements[i].algName != nullptr ? allAlgos.algElements[i].algName : "(null)");
        }
        // miss 时 Lookup 输出空串(与旧 Enrich 未命中行为一致), 条目侧不再区分;
        // 三字段均为 mapper 借出指针(静态维度表/cache_ 节点), 进程生命周期内稳定
        cm.costAlgoParams[i].engineName = dims.engineUser;
        cm.costAlgoParams[i].executorName = dims.executorUser;
        cm.costAlgoParams[i].templateName = dims.templateUser;
#endif
    }
    cm.count = allAlgos.count;
    return HcclResult::HCCL_SUCCESS;
}

void CostModelManager::FilterByEngine(CostModel& cm, const std::vector<OpExecuteConfig>& candidateEngines)
{
    std::set<OpExecuteConfig> engineSet(candidateEngines.begin(), candidateEngines.end());
    int kept = 0;
    int filtered = 0;
    for (int i = 0; i < cm.count; ++i) {
        const char* algName = cm.costAlgoParams[i].algName;
        // 引擎取 attrs->engine(注册期解析缓存)
        const AlgAttrs* attrs = (algName != nullptr) ? AlgAttrsRegistry::Instance().Get(algName) : nullptr;
        if (attrs == nullptr) {
            // 无名或无 attrs 属注册异常, 提前过滤
            cm.costAlgoParams[i].count = 0;
            HCCL_INFO("[CostModelManager] algName=%s filtered: no attrs.", algName != nullptr ? algName : "(null)");
            ++filtered;
            continue;
        }
        if (engineSet.count(attrs->engine) == 0) {
            cm.costAlgoParams[i].count = 0;
            HCCL_INFO(
                "[CostModelManager] algName=%s filtered: engine=%d not in candidate engines.", algName,
                static_cast<int>(attrs->engine));
            ++filtered;
        } else {
            ++kept;
        }
    }
    HCCL_INFO("[CostModelManager] FilterByEngine done, kept=%d filtered=%d.", kept, filtered);
}

HcclResult
CostModelManager::ApplyHcclAlgoPriority(HcclComm comm, CostModel& cm, const std::vector<std::string>& candidatePrefixes)
{
    // 只打标, 过滤收敛在 costtable Phase 3
    CHK_RET(MarkHcclAlgoPriority(comm, cm, candidatePrefixes));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CostModelManager::FilterByTopoAndCalibrate(
    HcclComm comm, CostModel& cm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo)
{
    // 条目与 allAlgos 对齐(InitModel 保证), opType 取自 AlgElement
    const AllAlgos& allAlgos = *GetAllAlgos();
    const bool tunerLoaded = HcclTunerIsLoaded();
    int writeIdx = 0;
    for (int i = 0; i < cm.count; ++i) {
        // step2 已过滤的条目(非候选引擎/无名/无 attrs)直接跳过
        if (cm.costAlgoParams[i].count == 0) {
            continue;
        }
        const AlgElement& alg = allAlgos.algElements[i];
        std::string algName = (alg.algName != nullptr) ? alg.algName : "";

        // preferred 算法或 tuner 接管时跳过软策略 customCheck
        bool needSoftCheck = !(tunerLoaded || cm.costAlgoParams[i].hcclAlgoPriority > 0);
        if (!IsAlgoMatchTopo(algName, topoInfo, needSoftCheck)) {
            // preferred 被 topo 硬过滤即 HCCL_ALGO 未生效, WARNING 提示回退自动选路
            if (cm.costAlgoParams[i].hcclAlgoPriority > 0) {
                HCCL_WARNING(
                    "[CostModelManager] HCCL_ALGO configured algName=%s skipped by topo filter, fallback to "
                    "auto selection.",
                    algName.c_str());
            }
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

        // 原地压缩(writeIdx<=i), 保留 hcclAlgoPriority/3D 名, count 变为 param 组数
        cm.costAlgoParams[writeIdx].algName = alg.algName;
        cm.costAlgoParams[writeIdx].engineName = cm.costAlgoParams[i].engineName;
        cm.costAlgoParams[writeIdx].executorName = cm.costAlgoParams[i].executorName;
        cm.costAlgoParams[writeIdx].templateName = cm.costAlgoParams[i].templateName;
        cm.costAlgoParams[writeIdx].param = ownedParam;
        cm.costAlgoParams[writeIdx].count = paramCount;
        cm.costAlgoParams[writeIdx].hcclAlgoPriority = cm.costAlgoParams[i].hcclAlgoPriority;
        ++writeIdx;
    }
    HCCL_INFO("[CostModelManager] FilterByTopoAndCalibrate done, total=%d calibrated=%d.", cm.count, writeIdx);
    cm.count = writeIdx;
    if (cm.count == 0) {
        delete[] cm.costAlgoParams;
        cm.costAlgoParams = nullptr;
    }
    return HcclResult::HCCL_SUCCESS;
}

// 从已过滤的算法中筛选优先级算法。按 (opType, engine) 分组，仅在有 priority 匹配的组内过滤。
// 不同引擎（AICPU/CCU/AIV）的 priority 互不影响，避免 AIV 算法被 AICPU/CCU 的 priority 规则误删。
// tuner 接管的组不做排他；HCCL_ALGO 正向指定的算法不删（优先级在 costtable 阶段收敛）。
void CostModelManager::ApplyTopoPriority(CostModel& costModel, const TopoInfoWithNetLayerDetails* topoInfo)
{
    if (costModel.count <= 0) {
        return;
    }
    const bool tunerLoaded = HcclTunerIsLoaded();

    // 1. 收集每个 (opType, engine) 的 priority 匹配索引
    std::map<std::pair<HcclCMDType, OpExecuteConfig>, std::vector<int>> priorityByKey;
    for (int i = 0; i < costModel.count; ++i) {
        const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(costModel.costAlgoParams[i].algName);
        if (attrs != nullptr && attrs->topo.topoPriorityCheck && attrs->topo.topoPriorityCheck(topoInfo)) {
            priorityByKey[{attrs->opType, attrs->engine}].push_back(i);
            HCCL_INFO("[CostModelManager] algName=%s matched: topoPriority.", costModel.costAlgoParams[i].algName);
        }
    }

    // 2. 对有 priority 匹配的 (opType, engine) 组，只保留匹配的算法；tuner 接管的组跳过排他；
    //    HCCL_ALGO 正向指定(preferred)的算法不删，保证优先级在 costtable 阶段可收敛
    std::set<int> toRemove;
    for (auto& [key, indices] : priorityByKey) {
        auto [opType, engine] = key;
        if (tunerLoaded) {
            HCCL_INFO(
                "[CostModelManager] opType=%d engine=%d covered by tuner, skip topoPriority.", static_cast<int>(opType),
                static_cast<int>(engine));
            continue;
        }
        std::set<int> keepSet(indices.begin(), indices.end());
        for (int i = 0; i < costModel.count; ++i) {
            const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(costModel.costAlgoParams[i].algName);
            if (attrs != nullptr && attrs->opType == opType && attrs->engine == engine && keepSet.count(i) == 0
                && costModel.costAlgoParams[i].hcclAlgoPriority <= 0) {
                toRemove.insert(i);
            }
        }
    }

    if (toRemove.empty()) {
        return;
    }

    if (UNLIKELY(HcclCheckLogLevel(DLOG_INFO))) {
        for (int idx : toRemove) {
            HCCL_INFO("[CostModelManager] algName=%s filtered: topoPriority.", costModel.costAlgoParams[idx].algName);
        }
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
    // 被过滤条目的param数组同样由costModel持有, 需先释放, 与FreeCostModel的释放语义保持一致
    for (int i = 0; i < costModel.count; ++i) {
        if (toRemove.count(i) > 0) {
            delete[] costModel.costAlgoParams[i].param;
            costModel.costAlgoParams[i].param = nullptr;
        }
    }
    delete[] costModel.costAlgoParams;
    costModel.costAlgoParams = newParams;
    costModel.count = newCount;
    HCCL_INFO("[CostModelManager] topoPriority applied, kept=%d.", costModel.count);
}

#endif

void CostModelManager::CalcMeshParam(
    float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod, EngineType engine)
{
    // n用来表示传输数据量和总数据量之间的关系
    A = 0.0f;
    double portNumEff = static_cast<double>(portNum);
    if (isPod && netType == CommTopo::COMM_TOPO_CLOS) {
        portNumEff = portNumEff / 2.0;
    }
    float linkBw = crossChipBw_;
    if (engine == EngineType::CPU) {
        // host网卡是几个NPU共享的，这个和组网和使用的网卡都有关系。
        linkBw = roceBw_;
    }
    if (netType == CommTopo::COMM_TOPO_1DMESH) {
        // cost = D/B(write)
        A = n / linkBw;
    } else if (netType == CommTopo::COMM_TOPO_CLOS) {
        // cost = nD/B(write)
        A = static_cast<float>((n * (rankSize - 1)) / (portNumEff * linkBw));
    } else {
        HCCL_ERROR("[CostModelManager] CalcMeshParams unsupported netType=%d.", static_cast<int>(netType));
    }
    HCCL_INFO(
        "[CostModelManager] CalcMeshParams n=%f netType=%d portNum=%d portNumEff=%f A=%f linkBw=%f.", n,
        static_cast<int>(netType), portNum, portNumEff, A, linkBw);
    return;
}

void CostModelManager::CalcNHRParams(
    float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod, bool halvePodPort, EngineType engine)
{
    // n用来表示传输数据和总数据量之间的关系
    // rankSize是指总共通信的rankSize
    A = 0.0f;
    double portNumEff = static_cast<double>(portNum);
    if (isPod && netType == CommTopo::COMM_TOPO_CLOS && halvePodPort) {
        portNumEff = portNumEff / 2.0;
    }
    float linkBw = crossChipBw_;
    if (engine == EngineType::CPU) {
        // host网卡是几个NPU共享的，这个和组网和使用的网卡都有关系。
        linkBw = roceBw_;
    }
    float data = n * (rankSize - 1);
    A = static_cast<float>(data / (portNumEff * linkBw));
    HCCL_INFO(
        "[CostModelManager] CalcNHRParams n=%f netType=%d portNum=%d portNumEff=%f A=%f linkBw=%f "
        "rankSize=%u data=n*(rs-1)=%f.",
        n, static_cast<int>(netType), portNum, portNumEff, A, linkBw, rankSize, data);
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
    HCCL_INFO("[CostModelManager] CalcLocalCopyParams n=%f scene=%d B=%f.", n, static_cast<int>(scene), B);
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
    HCCL_INFO("[CostModelManager] CalcLocalReduceParams n=%f scene=%d B=%f.", n, static_cast<int>(scene), B);
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
    HCCL_INFO("[CostModelManager] CalcLatencyParams taskNum=%d engine=%d C=%f.", taskNum, static_cast<int>(engine), C);
    return;
}

void CostModelManager::CalcDpuLatencyParams(int stepNum, int syncNum, int channelNum, int sndRcvnum, float& C)
{
    float stepC = preSync_ * syncNum + channelFence_ * channelNum + threadFence_;
    float sndRcvC = sndRcvnum * sndRcvCall_;
    C = stepNum * stepC + sndRcvC + hDLatency_;
    HCCL_INFO(
        "[CostModelManager] CalcDpuLatencyParams stepNum=%d syncNum=%d channelNum=%d sndRcvnum=%d C=%f.", stepNum,
        syncNum, channelNum, sndRcvnum, C);
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
    HCCL_INFO("[CostModelManager] CalcLaunchParams taskNum=%d engine=%d D=%f.", taskNum, static_cast<int>(engine), D);
    return;
}

int CostModelManager::CalcTransTaskNum(u32 rankSize) { return static_cast<int>(5 * (rankSize - 1)); }

int CostModelManager::CalcSyncTaskNum(u32 rankSize) { return static_cast<int>(2 * (rankSize - 1)); }

} // namespace ops_hccl
