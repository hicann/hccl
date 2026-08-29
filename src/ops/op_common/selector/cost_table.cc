/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cost_table.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <functional>
#include <new>
#include <limits>

#include "auto_selector_base.h"
#include "hccl_aiv_utils.h"
#include "selector_engine.h"
#include "alg_attrs_registry.h"
#include "order_preserved_common.h"

namespace ops_hccl {

// ---------------------------------------------------------------------------
// 公共函数：判断输入输出是否 overlap（按算子类型区分）
// AllReduce/Reduce/ReduceScatter: input 和 output 是同一块内存
// AllGather/Broadcast/Scatter: output 可能包含 input 的部分
// AllToAll/AllToAllV: input 和 output 完全独立
// Send/Recv: 只有一个 buffer
// ---------------------------------------------------------------------------
bool IsInputOutputOverlap(const OpParam& opParam)
{
    if (opParam.inputPtr == nullptr || opParam.outputPtr == nullptr || opParam.inputSize == 0
        || opParam.outputSize == 0) {
        return false;
    }
    uintptr_t inStart = reinterpret_cast<uintptr_t>(opParam.inputPtr);
    uintptr_t outStart = reinterpret_cast<uintptr_t>(opParam.outputPtr);
    return inStart <= outStart + opParam.outputSize - 1 && outStart <= inStart + opParam.inputSize - 1;
}

// ---------------------------------------------------------------------------
// CostTableManager 方法实现
// ---------------------------------------------------------------------------

void CostTableManager::DumpCostTable(const CostTable& ct)
{
    HCCL_INFO("====== [DFX_CostTableDump] algoCount=%d ======", ct.count);
    for (int i = 0; i < ct.count; ++i) {
        const char* name = (ct.costs[i].algName != nullptr) ? ct.costs[i].algName : "";
        HCCL_INFO("  [DFX_CostTableDump] [%d/%d] algName=%s, cost=%.6f", i + 1, ct.count, name, ct.costs[i].cost);
    }
    HCCL_INFO("====== [DFX_CostTableDump] dump end ======");
}

HcclResult CostTableManager::InitAndFilterByAttrs(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    HCCL_INFO("[InitAndFilterByAttrs] filter, algCount=%d.", cm.count);
    ct.costs = nullptr;
    ct.count = 0;
    if (cm.count <= 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    ct.costs = new (std::nothrow) AlgoCost[cm.count]();
    if (ct.costs == nullptr) {
        HCCL_ERROR("[InitAndFilterByAttrs] alloc AlgoCost failed, count=%d.", cm.count);
        return HcclResult::HCCL_E_PARA;
    }

    u64 dataSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    HcclDataType dataType = opParam.DataDes.dataType;
    bool needOrderPreserved = IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize);
    bool isInplace = IsInputOutputOverlap(opParam);

    for (int i = 0; i < cm.count; ++i) {
        if (cm.costAlgoParams[i].count <= 0) {
            continue;
        }
        const char* algName = cm.costAlgoParams[i].algName;
        std::string name = (algName != nullptr) ? algName : "";
        const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(name);
        if (attrs == nullptr) {
            HCCL_DEBUG("[InitAndFilterByAttrs] algName=%s filtered: no attrs.", name.c_str());
            continue;
        }
        if (attrs->opType != opParam.opType) {
            HCCL_DEBUG("[InitAndFilterByAttrs] algName=%s filtered: opType mismatch.", name.c_str());
            continue;
        }

        // normal filter
        const auto& op = attrs->op;
        std::string filterReason;
        if (op.unsupportedDataTypes.count(dataType) > 0) {
            filterReason = "unsupportedDataTypes";
        }
        if (filterReason.empty() && !op.isSupportProd && opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            filterReason = "isSupportProd=false with PROD";
        }
        if (filterReason.empty() && !op.isSupportInplace && isInplace) {
            filterReason = "isSupportInplace=false with overlap";
        }
        if (filterReason.empty() && needOrderPreserved && !op.isSupportFloatOrderPreserved) {
            filterReason = "isSupportFloatOrderPreserved=false with order-preserved";
        }
        if (filterReason.empty() && op.opCustomCheck && !op.opCustomCheck(opParam, topoInfo)) {
            filterReason = "opCustomCheck returned false";
        }

        if (!filterReason.empty()) {
            HCCL_INFO("[InitAndFilterByAttrs] algName=%s filtered: %s.", name.c_str(), filterReason.c_str());
            continue;
        }

        float cost = CalcAlgCost(name, dataSize, cm.costAlgoParams[i], opParam.opType);
        ct.costs[ct.count].algName = algName;
        ct.costs[ct.count].cost = cost;
        ++ct.count;
        HCCL_INFO("[InitAndFilterByAttrs] algName=%s cost=%f.", name.c_str(), cost);
    }

    // Phase 2: priority — if any algo's opPriorityCheck returns true, keep only those.
    if (ct.count > 0) {
        std::vector<int> priorityIndices;
        for (int i = 0; i < ct.count; ++i) {
            const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(ct.costs[i].algName);
            if (attrs != nullptr && attrs->op.opPriorityCheck && attrs->op.opPriorityCheck(opParam, topoInfo)) {
                priorityIndices.push_back(i);
                HCCL_INFO("[InitAndFilterByAttrs] opPriority matched algName=%s.", ct.costs[i].algName);
            }
        }
        if (!priorityIndices.empty() && static_cast<int>(priorityIndices.size()) < ct.count) {
            AlgoCost* newCosts = new (std::nothrow) AlgoCost[ct.count]();
            if (newCosts == nullptr) {
                HCCL_ERROR("[InitAndFilterByAttrs] alloc newCosts for opPriority failed.");
                DumpCostTable(ct);
                return HcclResult::HCCL_SUCCESS;
            }
            for (size_t i = 0; i < priorityIndices.size(); ++i) {
                newCosts[i] = ct.costs[priorityIndices[i]];
            }
            delete[] ct.costs;
            ct.costs = newCosts;
            ct.count = static_cast<int>(priorityIndices.size());
            HCCL_INFO("[InitAndFilterByAttrs] opPriority applied, kept=%d.", ct.count);
        }
    }

    DumpCostTable(ct);
    return HcclResult::HCCL_SUCCESS;
}

float CostTableManager::CalcAlgCost(
    const std::string& algName, u64 dataSize, const CostAlgoParams& algoParams, HcclCMDType opType) const
{
    AlgNetMeta meta;
    AlgNetMetaRegistry::Global()->Query(algName, meta);

    OpExecuteConfig engine = SelectorEngine::GetEngineByAlgName(algName);

    const CostModelParam* params = algoParams.param;
    std::vector<u32> groups = meta.groupSizes;
    if (groups.empty()) {
        groups.assign(static_cast<size_t>(algoParams.count), 1);
    }

    std::vector<float> utils(static_cast<size_t>(algoParams.count), 1.0f);
    float cost = 0.0f;
    float totalD = 0.0f;
    u32 idx = 0;
    for (u32 g = 0; g < groups.size() && idx < static_cast<u32>(algoParams.count); ++g) {
        float groupCost = 0.0f;
        for (u32 k = 0; k < groups[g] && idx < static_cast<u32>(algoParams.count); ++k, ++idx) {
            CommTopo nt = (idx < meta.netTypes.size()) ? meta.netTypes[idx] : CommTopo::COMM_TOPO_1DMESH;
            float util = 1.0f;
            if (QueryUbUtil(nt, dataSize, engine, util, opType) != HcclResult::HCCL_SUCCESS) {
                util = 1.0f;
            }
            utils[idx] = util;
            float abCost = (params[idx].A / util + params[idx].B) * static_cast<float>(dataSize);
            float segCost = abCost + params[idx].C;
            groupCost = (meta.intraGroupMode == CostAggMode::MAX) ? std::max(groupCost, segCost) : groupCost + segCost;
            totalD += params[idx].D;
        }
        cost += groupCost;
    }

    cost = std::max(cost, totalD);
    HCCL_INFO(
        "[CalcAlgCost] algName=%s segCount=%d dataSize=%llu cost=%f totalD=%f "
        "groupCount=%zu intraMode=%d interMode=%d groups=[%s].",
        algName.c_str(), algoParams.count, dataSize, cost, totalD, groups.size(), static_cast<int>(meta.intraGroupMode),
        static_cast<int>(meta.interGroupMode),
        [&]() {
            std::string s;
            for (auto g : groups) {
                s += std::to_string(g) + ",";
            }
            if (!s.empty())
                s.pop_back();
            return s;
        }()
            .c_str());
    for (int j = 0; j < algoParams.count; ++j) {
        HCCL_INFO(
            "[CalcAlgCost]   seg[%d] A=%e B=%e A*ds=%e B*ds=%e C=%e D=%e util=%f segCost=%e.", j, params[j].A,
            params[j].B, params[j].A / utils[j] * static_cast<float>(dataSize),
            params[j].B * static_cast<float>(dataSize), params[j].C, params[j].D, utils[j],
            (params[j].A / utils[j] + params[j].B) * static_cast<float>(dataSize) + params[j].C);
    }
    return cost;
}

HcclResult CostTableManager::CostTableGen(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    HCCL_INFO("[CostTableGen] generate cost table, algCount=%d.", cm.count);
    HcclResult ret = InitAndFilterByAttrs(cm, ct, topoInfo, opParam);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CostTableGen] InitAndFilterByAttrs failed, ret=%d.", static_cast<int>(ret));
    }
    return ret;
}

const std::vector<UbUtilEntry> CostTableManager::closUbUtilTable_
    = {{0.125 * 1024 * 1024ULL, 0.02755f}, {0.25 * 1024 * 1024ULL, 0.05357f}, {0.5 * 1024 * 1024ULL, 0.10388f},
       {1 * 1024 * 1024ULL, 0.1855f},      {2 * 1024 * 1024ULL, 0.3f},        {4 * 1024 * 1024ULL, 0.4288f},
       {8 * 1024 * 1024ULL, 0.5302f},      {16 * 1024 * 1024ULL, 0.568f},     {32 * 1024 * 1024ULL, 0.6549f},
       {64 * 1024 * 1024ULL, 0.7184f},     {128 * 1024 * 1024ULL, 0.7408f},   {256 * 1024 * 1024ULL, 0.7644f}};

const std::vector<UbUtilEntry> CostTableManager::meshUbUtilTable_
    = {{1 * 1024 * 1024ULL, 0.7135f},  {2 * 1024 * 1024ULL, 0.7758f},   {4 * 1024 * 1024ULL, 0.8112f},
       {8 * 1024 * 1024ULL, 0.8301f},  {16 * 1024 * 1024ULL, 0.84f},    {32 * 1024 * 1024ULL, 0.8449f},
       {64 * 1024 * 1024ULL, 0.8475f}, {128 * 1024 * 1024ULL, 0.8487f}, {256 * 1024 * 1024ULL, 0.8494f}};

CostTableManager::~CostTableManager() {}

HcclResult CostTableManager::QueryUbUtil(
    CommTopo netType, u64 dataSize, OpExecuteConfig engine, float& utilization, HcclCMDType opType) const
{
    const std::vector<UbUtilEntry>& table = (netType == CommTopo::COMM_TOPO_CLOS) ? closUbUtilTable_ : meshUbUtilTable_;
    if (table.empty()) {
        HCCL_WARNING(
            "[CostTableManager] ub util table empty, netType=%d dataSize=%llu.", static_cast<int>(netType), dataSize);
        return HcclResult::HCCL_E_PARA;
    }
    // AllGather: CLOS 小数据量(< 1MB)统一用 1MB 的 util
    constexpr u64 AG_CLOS_MIN_UB_UTIL_DATA_SIZE = 1024ULL * 1024 * 1;
    if (opType == HcclCMDType::HCCL_CMD_ALLGATHER && netType == CommTopo::COMM_TOPO_CLOS
        && dataSize < AG_CLOS_MIN_UB_UTIL_DATA_SIZE) {
        dataSize = AG_CLOS_MIN_UB_UTIL_DATA_SIZE;
    }
    auto it = std::lower_bound(table.begin(), table.end(), dataSize, [](const UbUtilEntry& e, u64 ds) {
        return e.upperBound < ds;
    });
    if (it == table.end()) {
        utilization = table.back().utilization;
    } else {
        utilization = it->utilization;
    }
    if (engine == OpExecuteConfig::AIV) {
        utilization = utilization / 0.85f * 0.65f;
    }
    HCCL_DEBUG(
        "[CostTableManager] QueryUbUtil netType=%d dataSize=%llu engine=%d utilization=%f.", static_cast<int>(netType),
        dataSize, static_cast<int>(engine), utilization);
    return HcclResult::HCCL_SUCCESS;
}

CostTableManager* CostTableManager::Global()
{
    static CostTableManager* globalCostTableManager = new CostTableManager;
    return globalCostTableManager;
}

} // namespace ops_hccl
