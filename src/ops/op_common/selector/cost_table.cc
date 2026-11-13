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
#include <new>

#include "auto_selector_base.h"
#include "hccl_aiv_utils.h"
#include "alg_attrs_registry.h"
#include "alg_parse.h"
#include "coll_alg_v2_exec_registry.h"
#include "order_preserved_common.h"
#include "tuner_setup.h"

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

static inline u64 CalcCostTableDataSize(const OpParam& opParam, u32 userRankSize)
{
    switch (opParam.opType) {
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V: {
            if (opParam.vDataDes.counts == nullptr) {
                return 0;
            }
            const u64* counts = reinterpret_cast<const u64*>(opParam.vDataDes.counts);
            u64 maxC = 0;
            for (u32 i = 0; i < userRankSize; ++i) {
                maxC = std::max(maxC, counts[i]);
            }
            return maxC * DATATYPE_SIZE_TABLE[opParam.vDataDes.dataType];
        }
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC: {
            if (opParam.all2AllVDataDes.sendCounts == nullptr) {
                return 0;
            }
            u64 sendCount = *reinterpret_cast<const u64*>(opParam.all2AllVDataDes.sendCounts);
            return sendCount * DATATYPE_SIZE_TABLE[opParam.all2AllVDataDes.sendType];
        }
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV: {
            if (opParam.batchSendRecvDataDes.itemNum == 0 || opParam.batchSendRecvDataDes.sendRecvItemsPtr == nullptr) {
                return 0;
            }
            u64 count = static_cast<u64>(opParam.batchSendRecvDataDes.sendRecvItemsPtr[0].count);
            return count * DATATYPE_SIZE_TABLE[opParam.batchSendRecvDataDes.sendRecvItemsPtr[0].dataType];
        }
        case HcclCMDType::HCCL_CMD_BARRIER:
            return 0;
        default:
            return opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    }
}

bool IsHostNicToDeviceNicLink(const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo)
{
    const std::vector<u32>& netLayers = topoInfo->netLayerDetails.netLayers;
    if (netLayers.empty()) {
        return false;
    }
    for (auto it = netLayers.rbegin(); it != netLayers.rend(); ++it) {
        CommLink* linkList = nullptr;
        u32 listSize = 0;
        if (HcclRankGraphGetLinks(
                opParam.hcclComm, *it, topoInfo->userRank, opParam.sendRecvRemoteRank, &linkList, &listSize)
            != HCCL_SUCCESS) {
            continue;
        }
        if (listSize == 0 || linkList == nullptr) {
            continue;
        }
        return linkList[0].srcEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST
               && linkList[0].dstEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE;
    }
    return false;
}

OpMatchResult CheckAlgoMatchOpWithReason(
    const AlgAttrs& attrs, const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo,
    bool needSoftPolicyCheck)
{
    OpMatchResult result;
    const auto& op = attrs.op;
    HcclDataType dataType = GetOpDataType(opParam);
    bool isInplace = IsInputOutputOverlap(opParam);
    bool needOrderPreserved = IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize);

    if (!op.supportedDataTypes.empty()) {
        if (op.supportedDataTypes.count(dataType) == 0) {
            result.matched = false;
            result.reason = "not in supportedDataTypes";
            return result;
        }
    } else if (op.unsupportedDataTypes.count(dataType) > 0) {
        result.matched = false;
        result.reason = "unsupportedDataTypes";
        return result;
    }
    if (!op.isSupportProd && opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
        result.matched = false;
        result.reason = "isSupportProd=false with PROD";
        return result;
    }
    if (!op.isSupportInplace && isInplace) {
        result.matched = false;
        result.reason = "isSupportInplace=false with overlap";
        return result;
    }
    if (needOrderPreserved && !op.isSupportFloatOrderPreserved) {
        result.matched = false;
        result.reason = "isSupportFloatOrderPreserved=false with order-preserved";
        return result;
    }
    if (needSoftPolicyCheck && op.opCustomCheck && !op.opCustomCheck(opParam, topoInfo)) {
        result.matched = false;
        result.reason = "opCustomCheck returned false";
        return result;
    }
    return result;
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

HcclResult CostTableManager::GenerateCostTable(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    // 运行时阶段选路管线(通信域阶段见 cost_model.h):
    // Phase1 过滤+算cost → Phase2 opPriority 排他 → Phase3 HCCL_ALGO 收敛 → tuner 改cost → SelectMinCost
    HCCL_INFO("[CostTableManager] generate cost table, algCount=%d.", cm.count);
    ct.costs = nullptr;
    ct.count = 0;
    if (cm.count <= 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    ct.costs = new (std::nothrow) AlgoCost[cm.count]();
    if (ct.costs == nullptr) {
        HCCL_ERROR("[CostTableManager] alloc AlgoCost failed, count=%d.", cm.count);
        return HcclResult::HCCL_E_PARA;
    }

    u64 dataSize = CalcCostTableDataSize(opParam, topoInfo->userRankSize);
    // 与 ct.costs 对齐的 HCCL_ALGO 优先级(1=正向指定/-1=否定指定/0=未配置), Phase 2/3 收敛用
    std::vector<int> hcclPrio;
    hcclPrio.reserve(static_cast<size_t>(cm.count));

    OpFilterAndCalcCost(cm, ct, topoInfo, opParam, dataSize, hcclPrio); // Phase 1
    if (!HcclTunerIsLoaded()) {
        ApplyOpPriority(ct, opParam, topoInfo, hcclPrio); // Phase 2
    }
    ConvergeHcclAlgoTier(ct, hcclPrio); // Phase 3

    DumpCostTable(ct);
    return HcclResult::HCCL_SUCCESS;
}

void CostTableManager::FreeCostTable(CostTable& ct)
{
    delete[] ct.costs;
    ct.costs = nullptr;
    ct.count = 0;
}

void CostTableManager::OpFilterAndCalcCost(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, u64 dataSize,
    std::vector<int>& hcclPrio)
{
    const bool tunerLoaded = HcclTunerIsLoaded();
    int preferred = 0;
    int negated = 0;
    for (int i = 0; i < cm.count; ++i) {
        if (cm.costAlgoParams[i].count <= 0) {
            continue;
        }
        const char* algName = cm.costAlgoParams[i].algName;
        std::string name = (algName != nullptr) ? algName : "";
        const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(name);
        if (attrs == nullptr) {
            HCCL_INFO("[CostTableManager] algName=%s filtered: no attrs.", name.c_str());
            continue;
        }
        if (attrs->opType != opParam.opType) {
            HCCL_DEBUG("[CostTableManager] algName=%s filtered: opType mismatch.", name.c_str());
            continue;
        }

        // normal filter: preferred 算法或 tuner 接管时跳过软策略 opCustomCheck
        bool needSoftCheck = !(tunerLoaded || cm.costAlgoParams[i].hcclAlgoPriority > 0);
        auto opResult = CheckAlgoMatchOpWithReason(*attrs, opParam, topoInfo, needSoftCheck);
        if (!opResult.matched) {
            // preferred 被 op 硬过滤即 HCCL_ALGO 未生效, WARNING 提示回退自动选路
            if (cm.costAlgoParams[i].hcclAlgoPriority > 0) {
                HCCL_WARNING(
                    "[CostTableManager] HCCL_ALGO configured algName=%s filtered: %s, fallback to auto "
                    "selection.",
                    name.c_str(), opResult.reason.c_str());
            } else {
                HCCL_INFO("[CostTableManager] algName=%s filtered: %s.", name.c_str(), opResult.reason.c_str());
            }
            continue;
        }

        // 现查 meta：per-rank per-call，避免全局 Registry 多通信域覆盖
        std::unique_ptr<InsCollAlgBase> exec = CollAlgExecRegistryV2::Instance().GetAlgExec(opParam.opType, name);
        AlgNetMeta meta;
        if (exec != nullptr) {
            meta = exec->GetAlgNetMeta(topoInfo, opParam, name.c_str());
        }

        // 3D 名来自 costModel 条目(per-comm 一次物化), 表生成时拷贝并对 nullptr 兜底空串——
        // 替代旧每 op 的 TunerEnrichCostTable 内 AlgoNameMapper::Enrich 查找; ABI header 此处顺带
        // 初始化(插件契约四字段); 兜底对齐 cost_model.h 契约"条目生成侧兜底为空串"——未经 InitModel
        // 物化的旧式模型不得把 NULL 直达插件 ABI 条目(插件按 C 字符串 strcmp 使用)
        float cost = CalcAlgCost(
            name, dataSize, opParam, cm.costAlgoParams[i], opParam.opType, attrs->algoTypes, meta, attrs->engine);
        ct.costs[ct.count].algName = algName;
        ct.costs[ct.count].engineName
            = (cm.costAlgoParams[i].engineName != nullptr) ? cm.costAlgoParams[i].engineName : "";
        ct.costs[ct.count].executorName
            = (cm.costAlgoParams[i].executorName != nullptr) ? cm.costAlgoParams[i].executorName : "";
        ct.costs[ct.count].templateName
            = (cm.costAlgoParams[i].templateName != nullptr) ? cm.costAlgoParams[i].templateName : "";
        (void)HcclAbiHeaderInit(
            &ct.costs[ct.count].header, HCCL_TUNER_ALGO_COST_ENTRY_MAGIC_WORD, HCCL_TUNER_ALGO_COST_ENTRY_VERSION,
            sizeof(HcclTunerAlgoCostEntry));
        ct.costs[ct.count].cost = cost;
        ++ct.count;
        hcclPrio.push_back(cm.costAlgoParams[i].hcclAlgoPriority);
        if (cm.costAlgoParams[i].hcclAlgoPriority > 0) {
            ++preferred;
        } else if (cm.costAlgoParams[i].hcclAlgoPriority < 0) {
            ++negated;
        }
    }
    HCCL_INFO("[CostTableManager] op filter done, survivors=%d preferred=%d negated=%d.", ct.count, preferred, negated);
}

void CostTableManager::ApplyOpPriority(
    CostTable& ct, const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo, std::vector<int>& hcclPrio)
{
    // opPriority 排他: keep 集 = (opPriority 命中 ∪ preferred) ∩ 非否定, 其余删除;
    // preferred 的最终胜出在 Phase 3 层收敛, 否定一票否决(显式配置优先于软策略)
    if (ct.count <= 0) {
        return;
    }
    std::vector<int> priorityIndices;
    for (int i = 0; i < ct.count; ++i) {
        const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(ct.costs[i].algName);
        bool opPriorityMatched
            = attrs != nullptr && attrs->op.opPriorityCheck && attrs->op.opPriorityCheck(opParam, topoInfo);
        int prio = hcclPrio[static_cast<size_t>(i)];
        if (prio >= 0 && (prio > 0 || opPriorityMatched)) {
            priorityIndices.push_back(i);
            HCCL_INFO(
                "[CostTableManager] algName=%s kept: %s.", ct.costs[i].algName,
                opPriorityMatched ? "opPriority" : "hcclAlgo preferred");
        }
    }
    if (priorityIndices.empty() || static_cast<int>(priorityIndices.size()) == ct.count) {
        return;
    }
    if (UNLIKELY(HcclCheckLogLevel(DLOG_INFO))) {
        for (int i = 0; i < ct.count; ++i) {
            bool isPriority = std::find(priorityIndices.begin(), priorityIndices.end(), i) != priorityIndices.end();
            if (!isPriority) {
                HCCL_INFO("[CostTableManager] algName=%s filtered: opPriority.", ct.costs[i].algName);
            }
        }
    }
    CompactCostTable(ct, hcclPrio, priorityIndices);
    HCCL_INFO("[CostTableManager] opPriority applied, kept=%d.", ct.count);
}

void CostTableManager::ConvergeHcclAlgoTier(CostTable& ct, std::vector<int>& hcclPrio)
{
    // HCCL_ALGO 优先级收敛(正向=1/否定=-1/未配置=0), 只提供优先级不强制过滤:
    // 幸存者中仅保留最高优先级层(有正向指定则优先选, 否则有未配置算法时排除否定层);
    // 幸存者全为否定层时放开回退, 避免 HCCL_ALGO 导致无算法可选
    if (ct.count <= 0) {
        return;
    }
    int maxPrio = *std::max_element(hcclPrio.begin(), hcclPrio.end());
    if (maxPrio < 0) {
        HCCL_WARNING("[CostTableManager] all survivors are hcclAlgo-negated, fallback to full candidates.");
        return;
    }
    std::vector<int> keepIndices;
    for (int i = 0; i < ct.count; ++i) {
        if (hcclPrio[i] == maxPrio) {
            keepIndices.push_back(i);
        }
    }
    if (static_cast<int>(keepIndices.size()) == ct.count) {
        return;
    }
    if (UNLIKELY(HcclCheckLogLevel(DLOG_INFO))) {
        for (int i = 0; i < ct.count; ++i) {
            if (hcclPrio[i] != maxPrio) {
                HCCL_INFO(
                    "[CostTableManager] algName=%s filtered: hcclAlgoPriority=%d < %d.",
                    ct.costs[i].algName ? ct.costs[i].algName : "null", hcclPrio[i], maxPrio);
            }
        }
    }
    CompactCostTable(ct, hcclPrio, keepIndices);
    HCCL_INFO("[CostTableManager] hcclAlgoPriority applied, kept=%d.", ct.count);
}

void CostTableManager::CompactCostTable(CostTable& ct, std::vector<int>& hcclPrio, const std::vector<int>& keepIndices)
{
    // 原地前移压缩(keepIndices 升序, writeIdx<=idx 恒成立), 免整表分配拷贝
    int writeIdx = 0;
    for (int idx : keepIndices) {
        ct.costs[writeIdx] = ct.costs[idx];
        hcclPrio[static_cast<size_t>(writeIdx)] = hcclPrio[static_cast<size_t>(idx)];
        ++writeIdx;
    }
    ct.count = writeIdx;
    hcclPrio.resize(static_cast<size_t>(writeIdx));
}

float CostTableManager::CalcAlgCost(
    const std::string& algName, u64 dataSize, const OpParam& opParam, const CostAlgoParams& algoParams,
    HcclCMDType opType, const std::vector<AlgoType>& algoTypes, const AlgNetMeta& meta, OpExecuteConfig engine) const
{
    const CostModelParam* params = algoParams.param;
    std::vector<u32> groups = meta.groupSizes;
    if (groups.empty()) {
        groups.assign(static_cast<size_t>(algoParams.count), 1);
    }
    // 64bit/PROD 的 host 规约项 per-call 现算(当次调用的 dtype/reduceOp), 不随通信域级标定固化
    HcclDataType dataType = GetOpDataType(opParam);
    const u32 dataTypeSize = (dataType == HcclDataType::HCCL_DATA_TYPE_RESERVED) ? 0 : DATATYPE_SIZE_TABLE[dataType];

    std::vector<float> utils(static_cast<size_t>(algoParams.count), 1.0f);
    std::vector<float> bReduces(static_cast<size_t>(algoParams.count), 0.0f);
    float cost = 0.0f;
    float totalD = 0.0f;
    u32 idx = 0;
    for (u32 g = 0; g < groups.size() && idx < static_cast<u32>(algoParams.count); ++g) {
        float groupCost = 0.0f;
        for (u32 k = 0; k < groups[g] && idx < static_cast<u32>(algoParams.count); ++k, ++idx) {
            CommTopo nt = (idx < meta.netTypes.size()) ? meta.netTypes[idx] : CommTopo::COMM_TOPO_1DMESH;
            float dr = (idx < meta.dataRatios.size() && meta.dataRatios[idx] > 0.0f) ? meta.dataRatios[idx] : 1.0f;
            // 优先取 AlgNetMeta 的逐段类型(Parallel 算法4段), 空则回退 AlgAttrs 按名解析值(仅层级数个)
            AlgoType at = (idx < meta.algoTypes.size()) ?
                              meta.algoTypes[idx] :
                              ((idx < algoTypes.size()) ? algoTypes[idx] : AlgoType::UNKNOWN);
            if (at == AlgoType::NHR || at == AlgoType::NHR_MULTILINK || at == AlgoType::NHR_NDA) {
                u32 rs = (idx < meta.rankSizes.size()) ? meta.rankSizes[idx] : 1;
                dr *= static_cast<float>(rs) / 2.0f;
            }
            u64 perTransferSize = static_cast<u64>(static_cast<float>(dataSize) * dr);
            if (perTransferSize == 0) {
                perTransferSize = dataSize;
            }
            float util = 1.0f;
            if (QueryUbUtil(nt, perTransferSize, engine, util, opType, at) != HcclResult::HCCL_SUCCESS) {
                util = 1.0f;
            }
            utils[idx] = util;
            // host 规约项: 64bit/PROD 时按 1.2ns×count 折算进 B 的斜率(不除 util)
            bReduces[idx] = CalcHostReduceSlope(params[idx].reduceCopies, dataTypeSize, opParam.reduceType);
            float abCost = (params[idx].A / util + params[idx].B + bReduces[idx]) * static_cast<float>(dataSize);
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
            "[CalcAlgCost]   seg[%d] A=%e B=%e Bre=%e A*ds=%e B*ds=%e C=%e D=%e util=%f segCost=%e.", j, params[j].A,
            params[j].B, bReduces[j], params[j].A / utils[j] * static_cast<float>(dataSize),
            (params[j].B + bReduces[j]) * static_cast<float>(dataSize), params[j].C, params[j].D, utils[j],
            (params[j].A / utils[j] + params[j].B + bReduces[j]) * static_cast<float>(dataSize) + params[j].C);
    }
    return cost;
}

// 对于{m,n}来说，小于m的数据量取利用率n
const std::vector<UbUtilEntry> CostTableManager::closUbUtilTable_
    = {{0.125 * 1024 * 1024ULL, 0.10388f}, {0.25 * 1024 * 1024ULL, 0.10388f}, {0.5 * 1024 * 1024ULL, 0.10388f},
       {1 * 1024 * 1024ULL, 0.1855f},      {2 * 1024 * 1024ULL, 0.3f},        {4 * 1024 * 1024ULL, 0.4288f},
       {8 * 1024 * 1024ULL, 0.5302f},      {16 * 1024 * 1024ULL, 0.568f},     {32 * 1024 * 1024ULL, 0.6549f},
       {64 * 1024 * 1024ULL, 0.7184f},     {128 * 1024 * 1024ULL, 0.7408f},   {256 * 1024 * 1024ULL, 0.7644f}};

const std::vector<UbUtilEntry> CostTableManager::meshUbUtilTable_
    = {{1 * 1024 * 1024ULL, 0.7135f},  {2 * 1024 * 1024ULL, 0.7758f},   {4 * 1024 * 1024ULL, 0.8112f},
       {8 * 1024 * 1024ULL, 0.8301f},  {16 * 1024 * 1024ULL, 0.84f},    {32 * 1024 * 1024ULL, 0.8449f},
       {64 * 1024 * 1024ULL, 0.8475f}, {128 * 1024 * 1024ULL, 0.8487f}, {256 * 1024 * 1024ULL, 0.8494f}};

const std::vector<UbUtilEntry> CostTableManager::closOneJettyOnePortUbUtilTable_
    = {{1 * 1024 * 1024ULL, 0.217f},   {2 * 1024 * 1024ULL, 0.3453f},   {4 * 1024 * 1024ULL, 0.4904f},
       {8 * 1024 * 1024ULL, 0.6207f},  {16 * 1024 * 1024ULL, 0.7159f},  {32 * 1024 * 1024ULL, 0.7753f},
       {64 * 1024 * 1024ULL, 0.8089f}, {128 * 1024 * 1024ULL, 0.8268f}, {256 * 1024 * 1024ULL, 0.8361f}};

const std::vector<UbUtilEntry> CostTableManager::closAivUbUtilTable_
    = {{0.125 * 1024 * 1024ULL, 0.258f}, {0.25 * 1024 * 1024ULL, 0.375f}, {0.5 * 1024 * 1024ULL, 0.485f},
       {1 * 1024 * 1024ULL, 0.568f},     {2 * 1024 * 1024ULL, 0.622f},    {4 * 1024 * 1024ULL, 0.652f},
       {8 * 1024 * 1024ULL, 0.669f},     {16 * 1024 * 1024ULL, 0.678f},   {32 * 1024 * 1024ULL, 0.682f},
       {64 * 1024 * 1024ULL, 0.684f},    {128 * 1024 * 1024ULL, 0.685f},  {256 * 1024 * 1024ULL, 0.686f}};

CostTableManager::~CostTableManager() {}

HcclResult CostTableManager::QueryUbUtil(
    CommTopo netType, u64 dataSize, OpExecuteConfig engine, float& utilization, HcclCMDType opType,
    AlgoType algoType) const
{
    bool useOneJettyTable
        = (algoType == AlgoType::MESH || algoType == AlgoType::MESH_MULTILINK
           || algoType == AlgoType::MESH_SINGLE_CHANNEL || algoType == AlgoType::MESH_ONESHOT
           || algoType == AlgoType::MESH_TWOSHOT || algoType == AlgoType::MESH_NDA);
    bool useAivClosTable = (engine == OpExecuteConfig::AIV && netType == CommTopo::COMM_TOPO_CLOS);
    const std::vector<UbUtilEntry>& table = useAivClosTable ? closAivUbUtilTable_ :
                                            (netType == CommTopo::COMM_TOPO_CLOS && useOneJettyTable) ?
                                                              closOneJettyOnePortUbUtilTable_ :
                                            (netType == CommTopo::COMM_TOPO_CLOS) ? closUbUtilTable_ :
                                                                                    meshUbUtilTable_;
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
        return e.upperBound <= ds;
    });
    if (it == table.end()) {
        utilization = table.back().utilization;
    } else {
        utilization = it->utilization;
    }
    if (engine == OpExecuteConfig::AIV && !useAivClosTable) {
        utilization = utilization / 0.85f * 0.65f;
    }
    // Pairwise 多通道并发把带宽打满，util 不随数据量变化，不查表；真实利用率在模板折算，这里固定 1
    if (algoType == AlgoType::PAIRWISE && opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        utilization = 1.0f;
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
