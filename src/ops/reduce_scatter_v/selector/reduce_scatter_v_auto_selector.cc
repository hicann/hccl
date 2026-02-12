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
constexpr u64 RS_2D_SMALL_DATA_SIZE = 1024 * 1024;

SelectorStatus ReduceScatterVAutoSelector::SelectCcuMsAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] layerNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }

    // MS 模式不支持 int8
    CHK_PRT_RET(opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.",
            opParam.vDataDes.dataType),
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64 ||
        opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    HcclAlgoType levle0Algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 0)) {
        levle0Algo = it->second[0];
    }
    if (IsDefaultAlg(levle0Algo) || levle0Algo ==  HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) {
        return SelectMeshAlgo(topoInfo, opParam, selectAlgName);
    } else {
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] algo[%u] is not supported yet for ccu_ms mode, reset to default.", levle0Algo);
        return SelectorStatus::NOT_MATCH;
    }
}

SelectorStatus ReduceScatterVAutoSelector::SelectMeshAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                                    std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (IsSmallData(dataSize)) {
            selectAlgName = "CcuReduceScatterVMesh1D";
        } else {
            selectAlgName = "CcuReduceScatterVMesh1DMultiMission";
        }
    } else {
       return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectCcuScheduleAlgo(TopoInfo* topoInfo,
                                                    OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    // ccu 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->deviceNumPerModule > 1) {
                selectAlgName = "CcuReduceScatterVParallelMesh1DNHR";
                return SelectorStatus::MATCH;
            } else {
                selectAlgName = "CcuReduceScatterVNHR1DMem2Mem";
                return SelectorStatus::MATCH;
            }
        } else {
            HCCL_WARNING("[Algo][SelectCcuScheduleAlgo] layer0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        HcclAlgoType levle0Algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        auto it = configAlgMap.find(opParam.opType);
        if ((it != configAlgMap.end()) && (it->second.size() > 0)) {
            levle0Algo = it->second[0];
        }
        if ((IsDefaultAlg(levle0Algo) || levle0Algo ==  HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH)&&(topoInfo->level0Topo == Level0Shape::MESH_1D)) {
            selectAlgName = "CcuReduceScatterVMeshMem2Mem1D";
            return SelectorStatus::MATCH; 
        } else {
            HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] algo[%u] is not supported yet for ccu_schedule mode, reset to default.", levle0Algo);
            return SelectorStatus::NOT_MATCH;
        }
    }
}


SelectorStatus ReduceScatterVAutoSelector::SelectAicpuAlgo(TopoInfo* topoInfo,
                                                      OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        algos = it->second;
        if(algos[0] != HcclAlgoType::HCCL_ALGO_TYPE_NHR && algos[1] != HcclAlgoType::HCCL_ALGO_TYPE_NHR) {
            HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] algo[%u] is not supported yet, reset to default.", algos[0]);
        }
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos[0], algos[1], algos[2], algos[3]);

    if (opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_ERROR("[SelectAicpuAlgo] INT64, UINT64, FP64 only support in-box fullmesh algo type now.");
        return SelectorStatus::NOT_MATCH;
    }
    
    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] layerNum > 1 is not supported yet for aicpu_schedule mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }

    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectMeshAlgoAicpu(TopoInfo* topoInfo, OpParam &opParam,
                                                          std::string &selectAlgName) const
{
    if (topoInfo->level0Topo == Level0Shape::MESH_1D){
        selectAlgName = "InsReduceScatterVMesh1D";
    } else {
        HCCL_WARNING("[ReduceScatterVAutoSelector] topo not match");
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterVAutoSelector::SelectAivAlgo(TopoInfo* topoInfo, OpParam &opParam,
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

    //aiv 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.vDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_WARNING("[Algo][ReduceScatterVAutoSelector] aiv mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "AivReduceScatterVMesh1D";
    } else {
        HCCL_WARNING("[ReduceScatterVAutoSelector] topo not match for aiv algo");
        return  SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, 18, ReduceScatterVAutoSelector);
} // namespace ops_hccl
