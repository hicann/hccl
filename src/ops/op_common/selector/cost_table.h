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
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "alg_param.h"
#include "cost_model.h"
#include "hccl_tuner_plugin.h"
#include "log.h"
#include "alg_attrs.h"

namespace ops_hccl {

typedef hcclTunerAlgoEntry_t AlgoCost;

typedef struct {
    AlgoCost* costs;
    int count;
} CostTable;

struct UbUtilEntry {
    double upperBound;
    float utilization;
};

class CostTableManager {
public:
    static CostTableManager* Global();

    ~CostTableManager();

    HcclResult
    CostTableGen(CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam);
    HcclResult QueryUbUtil(
        AlgNetType netType, u64 dataSize, OpExecuteConfig engine, float& utilization,
        HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID) const;

private:
    CostTableManager() = default;

    HcclResult InitAndFilterByAttrs(
        CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam);
    float CalcAlgCost(
        const std::string& algName, u64 dataSize, const CostAlgoParams& algoParams,
        HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID) const;

    static void DumpCostTable(const CostTable& ct);

    CostTable costTable_{nullptr, 0};
    static const std::vector<UbUtilEntry> closUbUtilTable_;
    static const std::vector<UbUtilEntry> meshUbUtilTable_;
    mutable std::mutex mu_;
};

} // namespace ops_hccl

#endif
