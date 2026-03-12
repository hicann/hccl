/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
constexpr u64 RS_2D_SMALL_DATA_SIZE = 1024 * 1024;
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;

SelectorStatus ReduceScatterAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] layerNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }

    // MS 模式不支持 int8
    CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.",
            opParam.DataDes.dataType),
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
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
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] algo[%u] is not supported yet for ccu_ms mode, reset to default.", levle0Algo);
        return SelectorStatus::NOT_MATCH;
    }
}

SelectorStatus ReduceScatterAutoSelector::SelectMeshAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                    std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            selectAlgName = "CcuReduceScatterMesh2Die";
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            HCCL_INFO("[Algo][%s] TWO_DIE_NOT_REGULAR not match", __func__);
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuReduceScatterMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // UBX机型
        bool isMeshNumEqualToClosNum = false;
        bool isClosNumMultipleOfMeshNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Algo][ReduceScatterAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
        CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Algo][ReduceScatterAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
        if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
            // 4P mesh
            if (IsSmallData(dataSize)) {
                // 小数据量，用1d mesh算法
                selectAlgName = "CcuReduceScatterMesh1D";
            } else {
                // 大数据量，用mesh+clos并行算法
                selectAlgName = "CcuReduceScatterConcurrentMeshNHRMs";
            }
        } else if (isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
            HCCL_WARNING("[Algo][%s] MESH_1D_CLOS not match.", __func__);
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuReduceScatterMesh1D";
        }
    } else {
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                    const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{   
    // ccu 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->deviceNumPerModule > 1) {
                selectAlgName = "CcuReduceScatterParallelMesh1DNHR";
                return SelectorStatus::MATCH;
            } else {
                selectAlgName = "CcuReduceScatterNHR1DMem2Mem";
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
        if ((IsDefaultAlg(levle0Algo) || levle0Algo ==  HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH)) {
            if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
                if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
                    selectAlgName = "CcuReduceScatterMeshMem2Mem1D2Die";
                } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
                    HCCL_INFO("[Algo][%s] TWO_DIE_NOT_REGULAR not match", __func__);
                    return SelectorStatus::NOT_MATCH;
                } else {
                    selectAlgName = "CcuReduceScatterMesh1DMem2Mem";
                }
                return SelectorStatus::MATCH;
            } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
                // UBX机型
                bool isMeshNumEqualToClosNum = false;
                bool isClosNumMultipleOfMeshNum = false;
                CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
                    HCCL_ERROR("[Algo][ReduceScatterAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
                CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
                    HCCL_ERROR("[Algo][ReduceScatterAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
                if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
                    // 4P mesh
                    if (IsSmallData(dataSize)) {
                        // 小数据量，用1d mesh算法
                        selectAlgName = "CcuReduceScatterMesh1DMem2Mem";
                    } else {
                        // 大数据量，用mesh+clos并行算法
                        selectAlgName = "CcuReduceScatterConcurrentMeshNHRSche";
                    }
                } else if(isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
                    // 矩形场景大数据量，用2d并行算法
                    selectAlgName = "CcuReduceScatterParallelMesh1DNHRMultiJetty";
                } else {
                    // 其他场景，用1d NHR算法
                    selectAlgName = "CcuReduceScatterNhr1DMem2MemMultiJetty";
                }
            } else {
                HCCL_WARNING("[Algo][%s] MESH_1D_CLOS not match.", __func__);
                return SelectorStatus::NOT_MATCH;
            }
        } else {
            HCCL_WARNING("[Algo][ReduceScatterAutoSelector] algo[%u] is not supported yet for ccu_schedule mode, reset to default.", levle0Algo);
            return SelectorStatus::NOT_MATCH;
        }
    }
    return SelectorStatus::MATCH;
}


SelectorStatus ReduceScatterAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                      const OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] ReduceOp [PROD]] is not supported yet for aicpu mode."),
        SelectorStatus::NOT_MATCH);
    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_ERROR("[SelectAicpuAlgo] INT64, UINT64, FP64 only support in-box fullmesh algo type now.");
        return SelectorStatus::NOT_MATCH;
    }
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        algos = it->second;
        if(algos[0] != HcclAlgoType::HCCL_ALGO_TYPE_NHR && algos[1] != HcclAlgoType::HCCL_ALGO_TYPE_NHR) {
            HCCL_WARNING("[Algo][ReduceScatterAutoSelector] algo[%u] is not supported yet, reset to default.", algos[0]);
        }
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos[0], algos[1], algos[2], algos[3]);
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->deviceNumPerModule > 1 && topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsReduceScatterParallelMesh1DNHR";
        } else {
            selectAlgName = "InsReduceScatterNHR";
        }
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }

    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectMeshAlgoAicpu(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                          std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D){
        if (Is64BitDataType(opParam.DataDes.dataType) || dataSize < LARGE_COUNT_1024KB) {
            selectAlgName = "InsReduceScatterMesh1D";
        } else {
            selectAlgName = "InsReduceScatterMesh1DMeshChunk";
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            HCCL_WARNING("[ReduceScatterAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "InsReduceScatterNHR";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        return SelectMeshAlgoAicpuMeshClos(topoInfo, opParam, selectAlgName);
    } else {
        HCCL_WARNING("[ReduceScatterAutoSelector] topo not match");
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectMeshAlgoAicpuMeshClos(const TopoInfoWithNetLayerDetails* topoInfo,
    const OpParam &opParam, std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    bool isClosNumMultipleOfMeshNum = false;
    CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
        HCCL_ERROR("[Algo][ReduceScatterAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
    if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
        // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
        if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            selectAlgName = "InsReduceScatterAicpuReduce";
        } else if (!IsSmallData(dataSize)) {
            selectAlgName = "InsReduceScatterConcurrentMeshNHR";
        } else {
            double ratio; // 以8卡为基线确定ratio，用来表示不同卡数对下发的影响系数
            if (topoInfo->userRankSize == 0) {
                HCCL_WARNING("[ReduceScatterAutoSelector]the selector is not set RankSize_]");
                ratio = 1;
            } else {
                ratio = (DEFAULT_RANK_SIZE / topoInfo->userRankSize) * (DEFAULT_RANK_SIZE / topoInfo->userRankSize);
            }
            if (dataSize * ratio > LARGE_COUNT_1024KB) {
                selectAlgName = "InsReduceScatterMesh1DMeshChunk";
            } else {
                selectAlgName = "InsReduceScatterMesh1D";
            }
        }
    } else if (isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
        selectAlgName = "InsReduceScatterParallelMesh1DNHR";
    } else {
        if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            HCCL_ERROR("[SelectAicpuAlgo] INT64, UINT64, FP64, PROD only support in-box fullmesh algo type now.");
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "InsReduceScatterNHR";
        }
    }
    HCCL_DEBUG("[ReduceScatterAutoSelector] selectAlgName: %s", selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
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
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_WARNING("[Algo][ReduceScatterAutoSelector] aiv mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "AivReduceScatterMesh1D";
    } else {
        HCCL_WARNING("[ReduceScatterAutoSelector] topo not match for aiv algo");
        return  SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectDPUAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                        const OpParam &opParam,
                                                        const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                        std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u",
        opParam.opType, algos[0], algos[1], algos[2], algos[3]);
    HCCL_INFO("topoInfo->topoLevelNums is %u, topoInfo->level0Topo is %u", topoInfo->topoLevelNums, topoInfo->level0Topo);
    if (topoInfo->topoLevelNums > 1) {
        selectAlgName = "InsV2ReduceScatterSequenceMeshMesh";
        HCCL_INFO("Using algo InsV2ReduceScatterSequenceMeshMesh");
        return SelectorStatus::MATCH;
    }

    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, 18, ReduceScatterAutoSelector);
} // namespace ops_hccl
