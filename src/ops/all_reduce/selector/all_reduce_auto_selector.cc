/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_reduce_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
constexpr u64 AR_M2M_1D_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_MAX_DATA_SIZE = 32 * 1024 * 1024;

SelectorStatus AllReduceAutoSelector::SelectCcuMsAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] topoLevelNums > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }

    // MS 模式不支持 int8
    CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.",
            opParam.DataDes.dataType),
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
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
        HCCL_WARNING("[Algo][AllReduceAutoSelector] algo[%u] is not supported yet for ccu_ms mode, reset to default.", levle0Algo);
        return SelectorStatus::NOT_MATCH;
    }
}

SelectorStatus AllReduceAutoSelector::SelectMeshAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                                    std::string &selectAlgName) const
{
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectCcuScheduleAlgo(TopoInfo* topoInfo,
                                                    OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{   
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectAicpuAlgo(TopoInfo* topoInfo,
                                                      OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ReduceOp [PROD]] is not supported yet for aicpu mode."),
        SelectorStatus::NOT_MATCH);

    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        algos = it->second;
        if(algos[0] != HcclAlgoType::HCCL_ALGO_TYPE_NHR && algos[1] != HcclAlgoType::HCCL_ALGO_TYPE_NHR) {
            HCCL_WARNING("[Algo][AllReduceAutoSelector] algo[%u] is not supported yet, reset to default.", algos[0]);
        }
    }

    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos[0], algos[1], algos[2], algos[3]);
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->deviceNumPerModule > 1 && topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsAllReduceParallelMesh1DNHR";
        } else {
            selectAlgName = "InsAllReduceNHR";
        }
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_ERROR("[SelectAicpuAlgo] INT64, UINT64, FP64 only support in-box fullmesh algo type now.");
        return SelectorStatus::NOT_MATCH;
    }

    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectMeshAlgoAicpu(TopoInfo* topoInfo, OpParam &opParam,
                                                          std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
            opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
            opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
            HCCL_ERROR("[SelectAicpuAlgo] INT64, UINT64, FP64 only support in-box fullmesh algo type now.");
            return SelectorStatus::NOT_MATCH;
        } else if (dataSize <= AR_AICPU_1D_SMALL_DATA_SIZE) {
            selectAlgName = "InsAllReduceMesh1DOneShot";
        } else if (dataSize > AR_AICPU_1D_MAX_DATA_SIZE) { 
            selectAlgName = "InsAllReduceMesh1DTwoShotMeshChunk";
        } else {
            selectAlgName = "InsAllReduceMesh1DTwoShot";
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
            opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
            opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
            HCCL_WARNING("[AllReduceAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "InsAllReduceNHR";
        }
    } else {
        HCCL_WARNING("[AllReduceAutoSelector] topo not match");
        return SelectorStatus::NOT_MATCH;
    }

    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectAivAlgo(TopoInfo* topoInfo, OpParam &opParam,
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
        HCCL_WARNING("[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_WARNING("[Algo][AllReduceAutoSelector] aiv mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D && topoInfo->topoLevelNums <= 1) {
        u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
        u64 dataSize = opParam.DataDes.count * perDataSize;
        if (IsSmallData(dataSize)) {
            selectAlgName = "AivAllReduceMesh1DOneShot";
        } else {
            selectAlgName = "AivAllReduceMesh1DTwoShot";
        }
    } else {
        HCCL_WARNING("[AllReduceAutoSelector] topo not match for aiv algo");
        return  SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLREDUCE, 18, AllReduceAutoSelector);
} // namespace ops_hccl
