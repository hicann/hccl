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

constexpr u32 TOPO_LEVEL_1 = 1;
constexpr u32 TOPO_LEVEL_3 = 3;
constexpr u64 RSV_CCU_8P_MIN_DATA_SIZE = 32 * 1024 * 1024;

SelectorStatus ReduceScatterVAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceScatterVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    // ccu ms 模式不支持 inplace 场景
    CHK_PRT_RET(IsInputOutputOverlap(opParam) == true,
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] ccu ms does not support inplace reduce_scatter_v."),
        SelectorStatus::NOT_MATCH);
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
    // ccu schedule 模式不支持 inplace 场景
    CHK_PRT_RET(IsInputOutputOverlap(opParam) == true,
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] ccu schedule does not support inplace reduce_scatter_v."),
        SelectorStatus::NOT_MATCH);
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
    const u64* varData = reinterpret_cast<const u64*>(opParam.varData);
    // 从0长数组中还原出任务信息
    std::vector<u64> sendCounts;
    sendCounts.assign(varData, varData + topoInfo->userRankSize);
    u64 inputCount = 0;
    for (u64 i = 0; i < topoInfo->userRankSize; i++) {
        inputCount += sendCounts[i];
    }
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.vDataDes.dataType];
    u64 inputDataSize = inputCount * perDataSize;
    
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[ReduceScatterVAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else if (inputDataSize < RSV_CCU_8P_MIN_DATA_SIZE) {
            selectAlgName = "CcuReduceScatterVMesh1D";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            if (inputDataSize < RSV_CCU_8P_MIN_DATA_SIZE) {
                selectAlgName = "CcuReduceScatterVMesh1D";
            } else {
                return SelectorStatus::NOT_MATCH;
            }
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
        HCCL_ERROR("[SelectAicpuAlgo] [ReduceScatterVAutoSelector] INT64, UINT64 or FP64 are not yet supported for all mode.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums >= TOPO_LEVEL_1 && topoInfo->topoLevelNums <= TOPO_LEVEL_3) {
        selectAlgName = "InsReduceScatterVMesh1D";
    } else {
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                       const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                       std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceScatterVAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_DEBUG, "[Algo][ReduceScatterVAutoSelector] is not supported yet for AIV mode.");
    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, 18, ReduceScatterVAutoSelector);
} // namespace ops_hccl
