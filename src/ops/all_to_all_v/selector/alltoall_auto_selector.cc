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
    HCCL_DEBUG("[AlltoAllAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "CcuAllToAllMesh1D2Die";
        } else {
            HCCL_WARNING("[AlltoAllAutoSelector] levelNum > 1 is not supported yet for 2d ccu_ms mode.");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
                selectAlgName = "CcuAllToAllMesh2Die";
            } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
                HCCL_DEBUG("[AlltoAllAutoSelector][%s] TWO_DIE_NOT_REGULAR not match", __func__);
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuAlltoAllMesh1D";
            }
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
            if (topoInfo->level0PcieMix) {
                if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                    selectAlgName = "CcuAlltoAllMesh1D";
                } else {
                    HCCL_WARNING("[AlltoAllAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
                    return SelectorStatus::NOT_MATCH;
                }
            } else {
                uint32_t dataTypeSize = DATATYPE_SIZE_TABLE[opParam.all2AllDataDes.sendType];
                uint64_t dataSize = opParam.all2AllDataDes.sendCount * dataTypeSize;
                bool isMeshNumEqualToClosNum = false;
                CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
                    HCCL_DEBUG("[AlltoAllAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
                if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= CONCURRENT_RANK_LIMIT)
                    && (dataSize > BIG_DATA_SIZE_LIMIT)) { // 同一组4P且大数据量，走并发算法
                    selectAlgName = "CcuAllToAllMesh1DConcurrent";
                } else {
                    selectAlgName = "CcuAlltoAllMesh1DMultiJetty";
                }
            }
        } else {
            HCCL_DEBUG("[Algo][AlltoAllAutoSelector] algo is not supported yet for ccu_schedule mode, reset to default.");
            return SelectorStatus::NOT_MATCH;
        }
    }

    HCCL_INFO("[AlltoAllAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                     const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                     std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::CLOS ||
            topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            selectAlgName = "InsAlltoAllMesh1D";
        } else {
            HCCL_ERROR("[AlltoAllAutoSelector][%s] hccl algo no match");
            return SelectorStatus::NOT_MATCH;
        }
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::CLOS) {
        selectAlgName = "InsAlltoAllMesh1D";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // PCIE-SW定制机型，使用mesh1d算法
        if (topoInfo->level0PcieMix) {
            selectAlgName = "InsAlltoAllMesh1D";
            HCCL_INFO("[AlltoAllAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
            return SelectorStatus::MATCH;
        }
        uint32_t dataTypeSize = DATATYPE_SIZE_TABLE[opParam.all2AllDataDes.sendType];
        uint64_t dataSize = opParam.all2AllDataDes.sendCount * dataTypeSize;
        bool isMeshNumEqualToClosNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
            HCCL_ERROR("[AlltoAllAutoSelector] CheckMeshNumEqualToClosNum failed."),
            SelectorStatus::NOT_MATCH);
        if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= CONCURRENT_RANK_LIMIT) &&
            (opParam.all2AllDataDes.sendCount > BIG_DATA_SIZE_LIMIT)) {
            // 同一组4P且大数据量，不走并发
            selectAlgName = "InsAlltoAllMesh1DUBX";
        } else {
            selectAlgName = "InsAlltoAllMesh1DUBX";
        }
    } else {
        HCCL_ERROR("[AlltoAllAutoSelector][%s] hccl algo no match");
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[AlltoAllAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());

    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                   std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;

    selectAlgName = "AivAlltoAllMesh1D";

    HCCL_INFO("[AlltoAllAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllAutoSelector::SelectDPUAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                   std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos =
        std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType,
              algos[0], algos[1], algos[2], algos[3]);
    if (topoInfo->topoLevelNums > 1) {
        if ((topoInfo->deviceNumPerModule == 1) || (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
            selectAlgName = "InsAlltoAllMesh1DDPU";
            return SelectorStatus::MATCH;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            selectAlgName = "InsAlltoAllClosMesh1DDPU";
            return SelectorStatus::MATCH;
        }
    }

    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLTOALL, 18, AlltoAllAutoSelector);
} // namespace Hccl
