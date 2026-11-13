/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_COST_TABLE
#define HCCLV2_COLL_ALG_SELECTOR_COST_TABLE

#include <functional>
#include <set>
#include <string>
#include <vector>

#include "alg_param.h"
#include "cost_model.h"
#include "hccl_tuner_plugin.h"
#include "log.h"
#include "alg_attrs.h"
#include "alg_parse.h"

namespace ops_hccl {

typedef HcclTunerAlgoCostEntry AlgoCost;

typedef struct {
    AlgoCost* costs;
    int count;
} CostTable;

// op 匹配检查结果（不打日志，供调用方决定日志级别）
struct OpMatchResult {
    bool matched = true;
    std::string reason; // matched=false 时的过滤原因
};
// OpParam中各数据描述子共享union内存, 按算子类型从对应成员取真实数据类型
inline HcclDataType GetOpDataType(const OpParam& opParam)
{
    switch (opParam.opType) {
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
            return opParam.vDataDes.dataType;
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            return opParam.all2AllVDataDes.sendType;
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
            if (opParam.batchSendRecvDataDes.itemNum == 0 || opParam.batchSendRecvDataDes.sendRecvItemsPtr == nullptr) {
                return HCCL_DATA_TYPE_RESERVED;
            }
            return opParam.batchSendRecvDataDes.sendRecvItemsPtr[0].dataType;
        default:
            return opParam.DataDes.dataType;
    }
}

OpMatchResult CheckAlgoMatchOpWithReason(
    const AlgAttrs& attrs, const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo,
    bool needSoftPolicyCheck = true);

struct UbUtilEntry {
    double upperBound;
    float utilization;
};

class CostTableManager {
public:
    static CostTableManager* Global();

    ~CostTableManager();

    // 生成 costTable: Phase1 op 过滤+算cost → Phase2 opPriority 排他 → Phase3 HCCL_ALGO 收敛
    HcclResult GenerateCostTable(
        CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam);
    // 释放 GenerateCostTable 内部申请的 costs(申请/释放同模块, 调用方 RAII 或收尾时调用)
    static void FreeCostTable(CostTable& ct);
    HcclResult QueryUbUtil(
        CommTopo netType, u64 dataSize, OpExecuteConfig engine, float& utilization,
        HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID, AlgoType algoType = AlgoType::UNKNOWN) const;

private:
    CostTableManager() = default;

    // Phase 1: op 过滤 + cost 计算(hcclPrio 与 ct.costs 对齐)
    void OpFilterAndCalcCost(
        CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, u64 dataSize,
        std::vector<int>& hcclPrio);
    // Phase 2: opPriority 排他(tuner 跳过): keep = (opPriority 命中 ∪ preferred) ∩ 非否定
    void ApplyOpPriority(
        CostTable& ct, const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo, std::vector<int>& hcclPrio);
    // Phase 3: HCCL_ALGO 优先级收敛(全否定层放开回退)
    void ConvergeHcclAlgoTier(CostTable& ct, std::vector<int>& hcclPrio);
    // 按 keepIndices 原地压缩 costs/hcclPrio
    static void CompactCostTable(CostTable& ct, std::vector<int>& hcclPrio, const std::vector<int>& keepIndices);
    float CalcAlgCost(
        const std::string& algName, u64 dataSize, const OpParam& opParam, const CostAlgoParams& algoParams,
        HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID, const std::vector<AlgoType>& algoTypes = {},
        const AlgNetMeta& meta = {}, OpExecuteConfig engine = OpExecuteConfig::AICPU_TS) const;

    static void DumpCostTable(const CostTable& ct);

    static const std::vector<UbUtilEntry> closUbUtilTable_;
    static const std::vector<UbUtilEntry> meshUbUtilTable_;
    static const std::vector<UbUtilEntry> closOneJettyOnePortUbUtilTable_;
    static const std::vector<UbUtilEntry> closAivUbUtilTable_;
};

} // namespace ops_hccl

#endif
