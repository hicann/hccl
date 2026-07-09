/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_auto_selector.h"
#include "selector_registry.h"
#include "hccl_aiv_utils.h"

namespace ops_hccl {
constexpr u64 REDUCE_AICPU_1D_MAX_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 REDUCE_NHR_CCU_MAX_DATA_SIZE = 256 * 1024;

SelectorStatus ReduceAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[Algo][ReduceAutoSelector] topoInfo is nullptr"),
        SelectorStatus::NOT_MATCH);
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[ReduceAutoSelector] layerNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }

    // MS 模式不支持 int8
    CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_WARNING("[ReduceAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.",
            opParam.DataDes.dataType), SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING(
            "[ReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.", opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[ReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[ReduceAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }
    
    SelectorStatus ret = SelectMeshAlgoCcums(topoInfo, opParam, selectAlgName);
    if (ret == SelectorStatus::NOT_MATCH) {
        return ret;
    }
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectMeshAlgoCcums(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, std::string &selectAlgName) const
{
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[Algo][ReduceAutoSelector] topoInfo is nullptr"),
        SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[ReduceAutoSelector] 2DieFullMesh is not supported yet for ccu_ms mode.");
            return SelectorStatus::NOT_MATCH;
        } else if(dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
            HCCL_INFO("[ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuReduceMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
            if(dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
                HCCL_INFO("[ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuReduceMesh1D";
            }
        } else { // MS 不支持
            HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[Algo][ReduceAutoSelector] topoInfo is nullptr"),
        SelectorStatus::NOT_MATCH);
    (void)configAlgMap;
    // ccu 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING( "[ReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.",
            opParam.reduceType), SelectorStatus::NOT_MATCH);
    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[ReduceAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }
    
    constexpr u64 CCU_SCHEDULE_2LEVEL_MAX_PER_RANK_DATA_SIZE = 1ULL * 1024 * 1024;
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->userRankSize == 0 ||
            dataSize / topoInfo->userRankSize > CCU_SCHEDULE_2LEVEL_MAX_PER_RANK_DATA_SIZE) {
            HCCL_INFO("[ReduceAutoSelector] 2 level topo perRankDataSize[%llu] exceeds limit, fallback to aicpu.",
                topoInfo->userRankSize == 0 ? dataSize : dataSize / topoInfo->userRankSize);
            return SelectorStatus::NOT_MATCH;
        }
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0) == 1) {
                // 每框出 1 卡
                selectAlgName = "CcuReduceNHR1DMem2Mem";
            } else if (topoInfo->is2DieFullMesh) {
                HCCL_WARNING("[ReduceAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
                HCCL_DEBUG("[ReduceAutoSelector] dataType[%d] is not supported yet"
                " for ccu schedule mode with ms reduce. levelNum[%u]", opParam.DataDes.dataType, topoInfo->topoLevelNums), SelectorStatus::NOT_MATCH);
                u64 perRankSize = (topoInfo->userRankSize > 0) ? (dataSize / topoInfo->userRankSize) : dataSize;
                if (perRankSize <= REDUCE_NHR_CCU_MAX_DATA_SIZE) {
                    selectAlgName = "CcuReduceNHR1DMem2Mem";
                } else {
                    selectAlgName = "CcuReduceParallelMesh1DNHR";
                }
            }
        } else {
            HCCL_WARNING("[SelectCcuScheduleAlgo] layer0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoCcuSchedule(topoInfo, opParam, selectAlgName);
    }
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectMeshAlgoCcuSchedule(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, std::string &selectAlgName) const
{
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[Algo][ReduceAutoSelector] topoInfo is nullptr"),
        SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[ReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else if (dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
            HCCL_INFO("[ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuReduceMesh1DMem2Mem";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
            if (dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
                HCCL_INFO("[ReduceAutoSelector] Mesh1D dataSize[%llu] >= 8MB, fallback to aicpu.", dataSize);
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuReduceMesh1DMem2Mem";
            }
        } else if (topoInfo->level0PcieMix) {
            HCCL_WARNING("[ReduceAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuReduceParallelMesh1DNHRUBX";
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING("[ReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[Algo][ReduceAutoSelector] topoInfo is nullptr"),
        SelectorStatus::NOT_MATCH);
    (void)configAlgMap;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->topoLevelNums == 3) {
            if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[1] == 1) {
                selectAlgName = "ReduceAicpuReduceNHR";
            } else {
                selectAlgName = "ReduceParallelNHRNHRUboe";
            }
        } else if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            selectAlgName = "ReduceAicpuReduceNHR";
        } else if (topoInfo->Level1Nhr) {
            selectAlgName = "ReduceNHR";
        } else if (topoInfo->deviceNumPerModule > 1 && topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "ReduceParallelMesh1DNHR";
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0) == 1 || topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "ReduceNHR";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectMeshAlgoAicpu(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    std::string &selectAlgName) const
{
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[Algo][ReduceAutoSelector] topoInfo is nullptr"),
        SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    HCCL_DEBUG("SelectMeshAlgoAicpu %u", topoInfo->level0Topo);
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
            selectAlgName = "ReduceMesh1DTwoShot";
        } else {
            selectAlgName = "ReduceMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (topoInfo->level0PcieMix) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                if (dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
                    selectAlgName = "ReduceMesh1DTwoShot";
                } else {
                    selectAlgName = "ReduceMesh1D";
                }
            } else if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
                selectAlgName = "ReduceAicpuReduceNHR";
            } else {
                selectAlgName = "ReduceParallelMesh1DNHRPcie";
            }
        } else if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
            if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
                selectAlgName = "ReduceMesh1D";
            } else if (dataSize >= REDUCE_AICPU_1D_MAX_DATA_SIZE) {
                selectAlgName = "ReduceMesh1DTwoShot";
            } else {
                selectAlgName = "ReduceMesh1D";
            }
        } else {
            if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
                selectAlgName = "ReduceAicpuReduceNHR";
            } else {
                selectAlgName = "ReduceParallelMesh1DNHRUBX";
            }
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            selectAlgName = "ReduceAicpuReduceNHR";
        } else {
            selectAlgName = "ReduceNHR";
        }
    } else {
        HCCL_WARNING("[ReduceAutoSelector] level0Shape[%d] is not supported yet.", topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[Algo][ReduceAutoSelector] topoInfo is nullptr"),
        SelectorStatus::NOT_MATCH);
    (void)configAlgMap;
    // aiv 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[ReduceAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.", opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[ReduceAutoSelector] aiv mode not support UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->userRankSize > MAX_RANK_SIZE) {
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_DEBUG, "[ReduceAutoSelector][%s] rankSize[%u] larger than [%u]", __func__, topoInfo->userRankSize, MAX_RANK_SIZE);
        return SelectorStatus::NOT_MATCH;
    }

    void *cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_PRT_RET(HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize) != HCCL_SUCCESS,
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[ReduceAutoSelector] HcclGetHcclBuffer failed."), SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY &&
        dataSize >= AIV_MAX_PER_RANK_DATA_SIZE * topoInfo->userRankSize) {
        HCCL_DEBUG("[ReduceAutoSelector][%s] dataSize[%llu] larger than AIV_MAX_PER_RANK_DATA_SIZE[%llu] * rankSize[%u]",
            __func__, dataSize, AIV_MAX_PER_RANK_DATA_SIZE, topoInfo->userRankSize);
        return SelectorStatus::NOT_MATCH;
    }
    if (dataSize > cclBufferSize * AIV_MAX_CCL_LOOP_NUM) {
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_DEBUG, "[ReduceAutoSelector][%s] dataSize[%llu] too large for cclBufferSize [%llu]", __func__, dataSize, cclBufferSize);
        return SelectorStatus::NOT_MATCH;
    }

    selectAlgName = "AivReduceMesh1D";

    HCCL_INFO("[ReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceAutoSelector::SelectDPUAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                const OpParam &opParam,
                                                const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                std::string &selectAlgName) const
{
    CHK_PRT_RET(topoInfo == nullptr, HCCL_ERROR("[Algo][ReduceAutoSelector] topoInfo is nullptr"),
        SelectorStatus::NOT_MATCH);
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos[0], algos[1], algos[2], algos[3]);
    if (topoInfo->topoLevelNums > 1) {
        if ((topoInfo->deviceNumPerModule == 1) || (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
            selectAlgName = "InsReduceSequenceMeshNhrDPU";
            HCCL_INFO("selectAlgName is InsReduceSequenceMeshNhrDPU");
            return SelectorStatus::MATCH;
        }
    }
    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_REDUCE, 18, ReduceAutoSelector);
}  // namespace ops_hccl
