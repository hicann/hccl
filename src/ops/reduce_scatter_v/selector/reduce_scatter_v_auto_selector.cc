/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_v_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {

SelectorStatus ReduceScatterVAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceScatterVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    // MS 模式不支持 int8
    CHK_PRT_RET(opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_WARNING("[ReduceScatterVAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.",
            opParam.vDataDes.dataType),
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[ReduceScatterVAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.vDataDes.dataType)) {
        HCCL_WARNING("[ReduceScatterVAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[ReduceScatterVAutoSelector] layerNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }
    SelectorStatus ret = SelectMeshAlgoCcums(topoInfo, opParam, selectAlgName);
    if (ret == SelectorStatus::MATCH) {
        HCCL_INFO("[ReduceScatterVAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    }
    return ret;
}

SelectorStatus ReduceScatterVAutoSelector::SelectMeshAlgoCcums(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
    std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[ReduceScatterVAutoSelector] 2DieFullMesh is not supported yet for ccu_ms mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        } else {
            if (IsSmallData(dataSize)) {
                selectAlgName = "CcuReduceScatterVMesh1D";
            } else {
                selectAlgName = "CcuReduceScatterVMesh1DMultiMission";
            }
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
            selectAlgName = "CcuReduceScatterVMesh1D";
        } else { // MS 不支持
            HCCL_WARNING("[ReduceScatterVAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        HCCL_WARNING("[ReduceScatterVAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
                topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[ReduceScatterVAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
                topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceScatterVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    // ccu 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[ReduceScatterVAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.vDataDes.dataType)) {
        HCCL_WARNING("[ReduceScatterVAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    // 调度模式使用 MS 进行规约后不支持 int8
    CHK_PRT_RET(opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] dataType[%d] is not supported yet for ccu schedule mode.",
            opParam.vDataDes.dataType),
        SelectorStatus::NOT_MATCH);

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0) > 1) {
                selectAlgName = "CcuReduceScatterVParallelMesh1DNHR";
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuReduceScatterVNHR1DMem2Mem";
                return SelectorStatus::NOT_MATCH;
            }
        } else {
            HCCL_WARNING("[SelectCcuScheduleAlgo] layer0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoCcuSchedule(topoInfo, opParam, selectAlgName);
    }
    HCCL_INFO("[ReduceScatterVAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectMeshAlgoCcuSchedule(const TopoInfoWithNetLayerDetails* topoInfo,
    const OpParam &opParam, std::string &selectAlgName) const
{
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[ReduceScatterVAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuReduceScatterVMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
            selectAlgName = "CcuReduceScatterVMesh1D";
        } else {
            HCCL_WARNING("[ReduceScatterVAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        HCCL_WARNING("[ReduceScatterVAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[ReduceScatterVAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[ReduceScatterVAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    (void)configAlgMap;
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_ERROR("[Algo][ReduceScatterVAutoSelector] ReduceOp [PROD] is not supported yet for aicpu mode."),
        SelectorStatus::NOT_MATCH);
    if (Is64BitDataType(opParam.vDataDes.dataType)) {
        HCCL_ERROR("[SelectAicpuAlgo] INT64, UINT64, FP64 only support in-box fullmesh algo type now.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[ReduceScatterVAutoSelector] layerNum > 1 is not supported yet for aicpu_schedule mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }

    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectMeshAlgoAicpu(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                          std::string &selectAlgName) const
{
    (void) opParam;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D){
        selectAlgName = "InsReduceScatterVMesh1D";
    } else {
        HCCL_WARNING("[ReduceScatterVAutoSelector] topo not match");
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                       const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                       std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceScatterVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    HCCL_DEBUG("[Algo][ReduceScatterVAutoSelector] is not supported yet for AIV mode, reverting to AICPU mode.");
    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, 18, ReduceScatterVAutoSelector);
} // namespace ops_hccl
