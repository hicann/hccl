/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
SelectorStatus BroadcastAutoSelector::SelectCcuMsAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectMeshAlgoCcuMs(TopoInfo* topoInfo, OpParam &opParam,
                                                    std::string &selectAlgName) const
{
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectCcuScheduleAlgo(TopoInfo* topoInfo,
                                                    OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectAicpuAlgo(TopoInfo* topoInfo,
                                                      OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }

    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos[0], algos[1], algos[2], algos[3]);

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->deviceNumPerModule <= 1) {
            selectAlgName = "InsBroadcastNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsBroadcastParallelMesh1DNHR";
        } else {
            HCCL_WARNING("[BroadcastAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }
    return SelectorStatus::MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectMeshAlgoAicpu(TopoInfo* topoInfo, OpParam &opParam,
                                                          std::string &selectAlgName) const
{

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "InsBroadcastMesh1DTwoShot";
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        selectAlgName = "InsBroadcastNHR";
    } else {
        HCCL_WARNING("[BroadcastAutoSelector] topo not match");
        return SelectorStatus::NOT_MATCH;
    }

    return SelectorStatus::MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectAivAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                                        const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                        std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
              opParam.opType, algos[0], algos[1], algos[2], algos[3]);

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "AivBroadcastMesh1D";
    } else {
        HCCL_WARNING("[BroadcastAutoSelector] topo not match for aiv algo");
        return  SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_BROADCAST, 18, BroadcastAutoSelector);
} // namespace Hccl
