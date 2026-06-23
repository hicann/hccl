/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "all_gather_v_auto_selector.h"
#include "selector_registry.h"
 
namespace ops_hccl {
constexpr u64 AG_2D_SMALL_DATA_SIZE = 1024 * 1024;
constexpr u32 TOPO_LEVEL_NUM_3 = 3;
SelectorStatus AllGatherVAutoSelector::SelectCcuMsAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_WARNING("[Algo][AllGatherVAutoSelector] allgatherv is not supported yet for ccu_ms mode, reset to default.");
    return SelectorStatus::NOT_MATCH;
    HCCL_DEBUG("[AllGatherVAutoSelector][%s] end", __func__);
}

SelectorStatus AllGatherVAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos =
        std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType,
              algos[0], algos[1], algos[2], algos[3]);
 
    // ccu schedule 模式不支持 inplace 场景
    CHK_PRT_RET(IsInputOutputOverlap(opParam) == true,
        HCCL_WARNING("[Algo][AllGatherVAutoSelector] ccu schedule does not support inplace allgatherv."),
        SelectorStatus::NOT_MATCH);
    if (topoInfo->topoLevelNums == 1 && topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "CcuAllGatherVMesh1D";
    } else {
        HCCL_WARNING("[AllGatherVAutoSelector] ccu_schedule not supported for multi-level AllGatherV yet");
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}
 
SelectorStatus AllGatherVAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos =
        std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType,
              algos[0], algos[1], algos[2], algos[3]);
 
    if (topoInfo->topoLevelNums >= 1 && topoInfo->topoLevelNums <= TOPO_LEVEL_NUM_3) {
        selectAlgName = "InsAllGatherVMesh1D";
    } else {
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}
 
SelectorStatus AllGatherVAutoSelector::SelectAivAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[Algo][AllGatherVAutoSelector] allgatherv is not supported yet for aiv mode, reset to default.");
    return SelectorStatus::NOT_MATCH;
    HCCL_DEBUG("[AllGatherVAutoSelector][%s] end", __func__);
}
 
REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLGATHER_V, 18, AllGatherVAutoSelector);
 
}  // namespace ops_hccl