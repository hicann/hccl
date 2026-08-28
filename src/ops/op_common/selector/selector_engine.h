/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_ENGINE
#define HCCLV2_COLL_ALG_SELECTOR_ENGINE

#include <string>
#include <vector>

#include "alg_param.h"
#include "cost_model.h"
#include "cost_table.h"
#include "log.h"
#include "hccl_res.h"

namespace ops_hccl {

class SelectorEngine {
public:
    static SelectorEngine* Global();

    HcclResult Run(HcclComm comm, OpParam& param, TopoInfoWithNetLayerDetails* topoInfo, std::string& algName);

    // 新选择器算子白名单: 本迭代仅支持 AllReduce/ReduceScatter/AllGather
    static bool IsOpSupported(HcclCMDType opType);

    // 根据 algName 前缀推断所属引擎(OpExecuteConfig),使用 ENGINE_PREFIX_MAP
    static OpExecuteConfig GetEngineByAlgName(const std::string& algName);

    // 候选引擎列表转为前缀字符串列表
    static std::vector<std::string> CandidateEnginesToPrefixes(const std::vector<OpExecuteConfig>& engines);

    static std::vector<OpExecuteConfig> GetEnginePriority(OpExecuteConfig opExecuteConfig);

    // 根据候选引擎列表过滤 CostModel: 不属于候选引擎的算法 count 置 -1
    static HcclResult FilterCmByEngine(CostModel& cm, const std::vector<OpExecuteConfig>& candidateEngines);

private:
    SelectorEngine() = default;

    HcclResult InitCostModel(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, OpParam& param, CostModel*& cm);

    HcclResult SelectMinCost(const CostTable& ct, OpParam& param, std::string& algName);
};

} // namespace ops_hccl

#endif
