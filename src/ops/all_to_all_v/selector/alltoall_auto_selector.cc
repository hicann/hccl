/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoall_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
constexpr uint32_t INDEX_0 = 0;
constexpr uint32_t INDEX_1 = 1;
constexpr uint32_t INDEX_2 = 2;
constexpr uint32_t INDEX_3 = 3;
constexpr uint32_t CONCURRENT_RANK_LIMIT = 4;
constexpr uint64_t BIG_DATA_SIZE_LIMIT = 512;
SelectorStatus AlltoAllAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    HCCL_WARNING("[Algo][AlltoAllAutoSelector] is not supported yet for ccu_ms mode, reset to default.");
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][AlltoAllAutoSelector] levelNum > 1 is not supported yet for ccu_schedule mode.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            selectAlgName = "CcuAllToAllMesh2Die";
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            HCCL_INFO("[Algo][%s] TWO_DIE_NOT_REGULAR not match", __func__);
            return SelectorStatus::NOT_MATCH;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            HCCL_INFO("Setlect CcuAlltoAllMesh1D!");
            selectAlgName = "CcuAlltoAllMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        uint32_t dataTypeSize = DATATYPE_SIZE_TABLE[opParam.all2AllDataDes.sendType];
        uint64_t dataSize = opParam.all2AllDataDes.sendCount * dataTypeSize;
        bool isMeshNumEqualToClosNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
                    HCCL_ERROR("[Algo][AlltoAllAutoSelector] CheckMeshNumEqualToClosNum failed."),
                    SelectorStatus::NOT_MATCH);
        if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= CONCURRENT_RANK_LIMIT)
            && (dataSize > BIG_DATA_SIZE_LIMIT)) { // 同一组4P且大数据量，走并发算法
            selectAlgName = "CcuAllToAllMesh1DConcurrent";
        } else {
            selectAlgName = "CcuAlltoAllMesh1DMultiJetty";
        }
    } else {
        HCCL_WARNING("[Algo][AlltoAllAutoSelector] algo is not supported yet for ccu_schedule mode, reset to default.");
        return SelectorStatus::NOT_MATCH;
    }

    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                     const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                     std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }

    HCCL_INFO("[AlltoAll] hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos.at(INDEX_0), algos.at(INDEX_1), algos.at(INDEX_2), algos.at(INDEX_3));

    if (topoInfo->topoLevelNums > 1) {
        HCCL_ERROR("hccl algo no match");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "InsAlltoAllMesh1D";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        uint32_t dataTypeSize = DATATYPE_SIZE_TABLE[opParam.all2AllDataDes.sendType];
        uint64_t dataSize = opParam.all2AllDataDes.sendCount * dataTypeSize;
        bool isMeshNumEqualToClosNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
                    HCCL_ERROR("[Algo][AlltoAllAutoSelector] CheckMeshNumEqualToClosNum failed."),
                    SelectorStatus::NOT_MATCH);
        if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= CONCURRENT_RANK_LIMIT) &&
            (opParam.all2AllDataDes.sendCount > BIG_DATA_SIZE_LIMIT)) { // 同一组4P且大数据量，不走并发
            selectAlgName = "InsAllToAllMesh1DConcurrent";
        } else {
            selectAlgName = "InsAlltoAllMesh1D";
        }
    }

    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                   std::string &selectAlgName) const
{
    (void) topoInfo;
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }

    HCCL_INFO("[AlltoAll] hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos.at(INDEX_0), algos.at(INDEX_1), algos.at(INDEX_2), algos.at(INDEX_3));

    selectAlgName = "AivAlltoAllMesh1D";

    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLTOALL, 18, AlltoAllAutoSelector);
} // namespace Hccl
