/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
constexpr uint32_t INDEX_0 = 0;
constexpr uint32_t INDEX_1 = 1;
constexpr uint32_t INDEX_2 = 2;
constexpr uint32_t INDEX_3 = 3;
constexpr uint32_t CONST_4 = 4;

SelectorStatus AlltoAllVAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                    const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][AlltoAllVAutoSelector] levelNum > 1 is not supported yet for ccu_schedule mode.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            selectAlgName = "CcuAllToAllVMesh2Die";
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            HCCL_INFO("[Algo][%s] TWO_DIE_NOT_REGULAR not match", __func__);
            return SelectorStatus::NOT_MATCH;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            HCCL_INFO("Setlect CcuAlltoAllVMesh1D!");
            selectAlgName = "CcuAlltoAllVMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        bool isMeshNumEqualToClosNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
                    HCCL_ERROR("[Algo][AlltoAllAutoSelector] CheckMeshNumEqualToClosNum failed."),
                    SelectorStatus::NOT_MATCH);
        if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= CONST_4)) { // 同一组4P，走并发算法
            selectAlgName = "CcuAllToAllVMesh1DConcurrent";
        } else {
            HCCL_WARNING("[Algo][AlltoAllVAutoSelector] algo is not supported yet for ccu_schedule mode, reset to default.");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        HCCL_WARNING("[Algo][AlltoAllVAutoSelector] algo is not supported yet for ccu_schedule mode, reset to default.");
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllVAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                      const OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    (void) topoInfo;
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }

    HCCL_INFO("[AlltoAllV] hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos.at(INDEX_0), algos.at(INDEX_1), algos.at(INDEX_2), algos.at(INDEX_3));

    if (topoInfo->topoLevelNums > 1) {
        HCCL_ERROR("hccl algo no match");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "InsAlltoAllVMesh1D";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        bool isMeshNumEqualToClosNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
                    HCCL_ERROR("[Algo][AlltoAllAutoSelector] CheckMeshNumEqualToClosNum failed."),
                    SelectorStatus::NOT_MATCH);
        if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= 4)) { // 同一组4P，走并发算法
            selectAlgName = "InsAllToAllVMesh1DConcurrent";
        } else {
            selectAlgName = "InsAlltoAllVMesh1D";
        }
    }

    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllVAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                   std::string &selectAlgName) const
{
    (void) topoInfo;
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }

    HCCL_INFO("[AlltoAllV] hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
              opParam.opType, algos.at(INDEX_0), algos.at(INDEX_1), algos.at(INDEX_2), algos.at(INDEX_3));

    selectAlgName = "AivAlltoAllVMesh1D";

    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLTOALLV, 18, AlltoAllVAutoSelector);
} // namespace Hccl
