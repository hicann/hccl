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
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;
    if (topoInfo->topoLevelNums > 1) {
        if (opParam.all2AllDataDes.sendType == HcclDataType::HCCL_DATA_TYPE_INT8) {
            HCCL_WARNING("[Algo][AlltoAllVAutoSelector] int8 is not supported yet for ccu_schedule mode.");
            return SelectorStatus::NOT_MATCH;
        }
        if (topoInfo->userRankSize > 128) {
            HCCL_WARNING("[Algo][AlltoAllVAutoSelector] rankSize > 128 is not supported yet for ccu_schedule mode.");
            return SelectorStatus::NOT_MATCH;
        }
        selectAlgName = "CcuAlltoAllVMesh1D";
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            selectAlgName = "CcuAllToAllVMesh2Die";
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            HCCL_DEBUG("[AlltoAllVAutoSelector][%s] TWO_DIE_NOT_REGULAR not match", __func__);
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuAlltoAllVMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
        if (topoInfo->level0PcieMix) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                selectAlgName = "CcuAlltoAllVMesh1D";
            } else {
                HCCL_WARNING("[AlltoAllVAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            }
        } else {
            bool isMeshNumEqualToClosNum = false;
            CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
                HCCL_DEBUG("[AlltoAllVAutoSelector] CheckMeshNumEqualToClosNum failed."),
                SelectorStatus::NOT_MATCH);
            if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= CONST_4)) { // 同一组4P，走并发算法
                selectAlgName = "CcuAllToAllVMesh1DConcurrent";
            } else {
                HCCL_DEBUG("[AlltoAllVAutoSelector] algo is not supported yet for ccu_schedule mode, "
                    "reset to default.");
                return SelectorStatus::NOT_MATCH;
            }
        }
    } else {
        HCCL_DEBUG("[AlltoAllVAutoSelector] algo is not supported yet for ccu_schedule mode, reset to default.");
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllVAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                      const OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::CLOS ||
            topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            selectAlgName = "InsAlltoAllVMesh1D";
        } else {
            HCCL_ERROR("[AlltoAllVAutoSelector][%s] hccl algo no match");
            return SelectorStatus::NOT_MATCH;
        }
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D || topoInfo->level0Topo == Level0Shape::CLOS) {
        selectAlgName = "InsAlltoAllVMesh1D";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // PCIE-SW定制机型，使用mesh1d算法
        if (topoInfo->level0PcieMix) {
            selectAlgName = "InsAlltoAllVMesh1D";
            HCCL_INFO("[AlltoAllVAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
            return SelectorStatus::MATCH;
        }
        bool isMeshNumEqualToClosNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Algo][AlltoAllAutoSelector] CheckMeshNumEqualToClosNum failed."),
            SelectorStatus::NOT_MATCH);
        if ((isMeshNumEqualToClosNum == true) && (topoInfo->userRankSize <= 4)) { // 同一组4P，走并发算法
            selectAlgName = "InsAlltoAllVMesh1DUBX";
        } else {
            selectAlgName = "InsAlltoAllVMesh1DUBX";
        }
    } else {
        HCCL_ERROR("[AlltoAllVAutoSelector][%s] hccl algo no match");
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllVAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                   const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                   std::string &selectAlgName) const
{
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)opParam;
    (void)configAlgMap;

    selectAlgName = "AivAlltoAllVMesh1D";
    HCCL_DEBUG("[AlltoAllVAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AlltoAllVAutoSelector::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
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
            selectAlgName = "InsAlltoAllVMesh1DDPU";
            return SelectorStatus::MATCH;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            selectAlgName = "InsAlltoAllVClosMesh1DDPU";
            return SelectorStatus::MATCH;
        }
    }

    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLTOALLV, 18, AlltoAllVAutoSelector);
} // namespace Hccl
