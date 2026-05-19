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
constexpr u64 RS_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 AR_ONESHOT_1D_MAX_DATA_SIZE = 16 * 1024;
constexpr u64 AR_M2M_1D_MAX_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_MAX_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_64P_SMALL_DATA_SIZE = 32 * 1024 * 1024;
constexpr u64 AR_AICPU_1D_64DATATYPE_DATA_SIZE = 8 * 1024 * 1024;
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u64 AR_FLATTEN_MAX_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_CCU_CLOS_1D_SMALL_DATA_SIZE = 8 * 1024 * 1024;
constexpr u64 AR_AICPU_SEQUENCE_DATA_SIZE = 1 * 1024 * 1024 * 1024;

SelectorStatus AllReduceAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->topoLevelNums > 1) {
        HCCL_DEBUG("[AllReduceAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }

    // MS 模式不支持 int8
    CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_DEBUG("[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.",
            opParam.DataDes.dataType),
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_DEBUG("[AllReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_DEBUG("[AllReduceAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    return SelectMeshAlgo(topoInfo, opParam, selectAlgName);
}

SelectorStatus AllReduceAutoSelector::SelectMeshUBXAlgo(const TopoInfoWithNetLayerDetails* topoInfo, std::string &selectAlgName, u64 dataSize) const
{
    // UBX机型
    bool isMeshNumEqualToClosNum = false;
    bool isClosNumMultipleOfMeshNum = false;
    CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
        HCCL_DEBUG("[AllReduceAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
    CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
        HCCL_DEBUG("[AllReduceAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
    if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
        // 4P mesh
        if (IsSmallData(dataSize)) {
            // 小数据量，用1d mesh算法
            selectAlgName = "CcuAllReduceMesh1DOneShot";
        } else {
            // 大数据量，用mesh+clos并行算法
            selectAlgName = "CcuAllReduceConcurrentMs";
        }
    } else if (isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
        HCCL_DEBUG("[AllReduceAutoSelector][%s] MESH_1D_CLOS not match.", __func__);
        return SelectorStatus::NOT_MATCH;
    } else {
        // 跨4p回退
        selectAlgName = "CcuAllReduceMesh1D";
    }

    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}                                                

SelectorStatus AllReduceAutoSelector::SelectMeshAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                    std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (IsInputOutputOverlap(opParam) == true) {
            // 不支持 inplace 场景
            return SelectorStatus::NOT_MATCH;
        }
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            if(IsSmallData(dataSize)) {
                selectAlgName = "CcuAllReduceMesh2Die"; 
            } else {
                selectAlgName = "CcuAllreduceMesh2DieBigMs"; 
            }
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            HCCL_DEBUG("[AllReduceAutoSelector][%s] TWO_DIE_NOT_REGULAR not match", __func__);
            return SelectorStatus::NOT_MATCH;
        } else if (IsSmallData(dataSize)) {
            selectAlgName = "CcuAllReduceMesh1DOneShot";
        } else {
            selectAlgName = "CcuAllReduceMesh1D";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsInputOutputOverlap(opParam) == true) {
            // 不支持 inplace 场景
            return SelectorStatus::NOT_MATCH;
        }
        return SelectMeshUBXAlgo(topoInfo, selectAlgName, dataSize);
    } else {
        HCCL_DEBUG("[AllReduceAutoSelector] level0Topo[%u] is not supported yet.", topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                            const OpParam &opParam,
                                                            const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                            std::string &selectAlgName) const
{   
    (void)configAlgMap;
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    // ccu 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_DEBUG("[AllReduceAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.",
            opParam.reduceType), SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_DEBUG("[AllReduceAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            // Level1Nhr 已在 CalcTopoShape 中设置（GCD==1 时为 true）
            if (topoInfo->Level1Nhr) {
                selectAlgName = "CcuAllReduceNHR1D";
                HCCL_INFO("[AllReduceAutoSelector] Level1Nhr=true, select [%s]", selectAlgName.c_str());
                return SelectorStatus::MATCH;
            } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                selectAlgName = "CcuAllReduceNHR1D";
            } else if (topoInfo->is2DieFullMesh) {
                HCCL_DEBUG("[AllReduceAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else if (dataSize <= AR_FLATTEN_MAX_DATA_SIZE && topoInfo->userRankSize > 8) {
                selectAlgName = "CcuAllReduceMesh1DMem2Mem";
                return SelectorStatus::MATCH;
            } else if(IsSmallDataCCU(dataSize, topoInfo->userRankSize)){//64M以下跑ccu
                 // 性能优化改用MS做reduce后不支持int8
                CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
                    HCCL_DEBUG("[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode with ms "
                        "reduce. levelNum[%u]", opParam.DataDes.dataType, topoInfo->topoLevelNums), SelectorStatus::NOT_MATCH);
                selectAlgName = "CcuAllReduceParallelMesh1DNHR";
                return SelectorStatus::MATCH;
            } else {
                return SelectorStatus::NOT_MATCH;//64M以上切为aicpu
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "CcuAllReduceNHR1D";
        } else {
            HCCL_DEBUG("[AllReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectCcuScheduleLevel0Algo(topoInfo, opParam, selectAlgName, dataSize);
    }
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectCcuScheduleLevel0UBXAlgo(const TopoInfoWithNetLayerDetails* topoInfo, 
    std::string &selectAlgName, const u64 dataSize) const
{
    // UBX机型
    bool isMeshNumEqualToClosNum = false;
    bool isClosNumMultipleOfMeshNum = false;
    CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
        HCCL_DEBUG("[AllReduceAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
    CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
        HCCL_DEBUG("[AllReduceAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
    if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
        // 4P mesh
        if (IsSmallData(dataSize)) {
            // 小数据量，用1d mesh算法
            selectAlgName = "CcuAllReduceMesh1DMem2Mem";
        } else {
            // 大数据量，用mesh+clos并行算法
            selectAlgName = "CcuAllReduceConcurrentSche";
        }
    } else if(isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
        // 矩形场景大数据量，用Parallel并行算法
        selectAlgName = "CcuAllReduceParallelNHR1DMutiJetty";
    } else {
        // 其他场景，用1d NHR算法
        selectAlgName = "CcuAllReduceNHR1DMem2MemMultiJetty";
    }

    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectCcuScheduleLevel0AlgoMesh1D(const TopoInfoWithNetLayerDetails* topoInfo,
    const OpParam &opParam, std::string &selectAlgName, const u64 dataSize) const
{
    // 性能优化改用MS做reduce后不支持int8
    CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_DEBUG("[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode "
                   "with ms reduce.", opParam.DataDes.dataType), SelectorStatus::NOT_MATCH);
    double ratio;
    if (topoInfo->userRankSize == 0) {
        HCCL_DEBUG("[AllReduceAutoSelector] the selector userRankSize not set");
        ratio = 1;
    } else {
        ratio = DEFAULT_RANK_SIZE / topoInfo->userRankSize / topoInfo->userRankSize;
    }
    if (dataSize * ratio > AR_M2M_1D_MAX_DATA_SIZE) {
        return SelectorStatus::NOT_MATCH;
    }
    if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
        if (IsSmallData(dataSize)) {
            selectAlgName = "CcuAllReduceMesh1DMem2Mem2DieOneShot";
        } else {
            selectAlgName = "CcuAllreduceMesh2DieBigSche";
        }
    } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
        HCCL_DEBUG("[AllReduceAutoSelector][%s] TWO_DIE_NOT_REGULAR not match", __func__);
        return SelectorStatus::NOT_MATCH;
    } else {
        selectAlgName = "CcuAllReduceMesh1DMem2Mem";
    }
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectCcuScheduleLevel0Algo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                 std::string &selectAlgName, const u64 dataSize) const
{
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        return SelectCcuScheduleLevel0AlgoMesh1D(topoInfo, opParam, selectAlgName, dataSize);
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (topoInfo->level0PcieMix) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                return SelectCcuScheduleLevel0AlgoMesh1D(topoInfo, opParam, selectAlgName, dataSize);
            } else {
                // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
                HCCL_WARNING("[AllReduceAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            }
        } else {
            CHK_PRT_RET(opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
            HCCL_DEBUG("[AllReduceAutoSelector] dataType[%d] is not supported yet for ccu schedule mode "
                    "with ms reduce.", opParam.DataDes.dataType), SelectorStatus::NOT_MATCH);
        return SelectCcuScheduleLevel0UBXAlgo(topoInfo, selectAlgName, dataSize);
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (dataSize > AR_CCU_CLOS_1D_SMALL_DATA_SIZE) {
            selectAlgName = "CcuAllReduceNHR1D";
        } else {
            selectAlgName = "CcuAllReduceMesh1DMem2Mem";
        }
    } else {
        HCCL_DEBUG("[AllReduceAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo,
                                                      const OpParam &opParam,
                                                      const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                      std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;

    bool isDataTypeOrReduceTypeSpecial = 
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64 ||
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;

    if (topoInfo->topoLevelNums > 1) {
        if (isDataTypeOrReduceTypeSpecial) {
            selectAlgName = "InsAllReduceAicpuReduceNHR";
        } else if (topoInfo->Level1Nhr) {
            // Level1Nhr 已在 CalcTopoShape 中设置（GCD==1 时为 true）
            selectAlgName = "InsAllReduceNHR";
            HCCL_INFO("[AllReduceAutoSelector] Level1Nhr=true, select [%s]", selectAlgName.c_str());
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgName = "InsAllReduceNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (dataSize > AR_AICPU_1D_64P_SMALL_DATA_SIZE) {
                selectAlgName = (dataSize > AR_AICPU_SEQUENCE_DATA_SIZE) ?
                    "InsAllReduceSequenceMesh1DNhr" : "InsAllReduceParallelRSAG";
            } else {
                selectAlgName = "InsAllReduceNHR";
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "InsAllReduceNHR";
        } else {
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }

    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectMeshAlgoAicpuUBX(const TopoInfoWithNetLayerDetails* topoInfo, const u64 dataSize, 
                                                             std::string &selectAlgName, bool isDataTypeOrReduceTypeSpecial) const
{
    // UBX机型
    bool isMeshNumEqualToClosNum = false;
    bool isClosNumMultipleOfMeshNum = false;
    CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
        HCCL_ERROR("[Algo][AllReduceAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
    CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
        HCCL_ERROR("[Algo][AllReduceAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
    if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
        if (isDataTypeOrReduceTypeSpecial) {
            selectAlgName = dataSize <= AR_AICPU_1D_64DATATYPE_DATA_SIZE ?
                            "InsAllReduceMesh1DOneShot" :
                            "InsAllReduceMesh1DTwoShot";            
        } else if (dataSize <= AR_AICPU_1D_SMALL_DATA_SIZE) {
            selectAlgName = "InsAllReduceMesh1DOneShot";
        } else {
            // 大数据量，用mesh+clos并行算法
            selectAlgName = "InsAllReduceMesh1DTwoShot";
        }
    } else if (isDataTypeOrReduceTypeSpecial) {
        selectAlgName = "InsAllReduceAicpuReduceNHR";
    } else if(isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
        // 矩形场景大数据量，用Parallel并行算法
        selectAlgName = "InsAllReduceParallelRSAGUBX";
    } else {
        // 其他场景，用1d NHR算法
        selectAlgName = "InsAllReduceNHR";
    }

    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectMeshAlgoAicpu(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                          std::string &selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;

    bool isDataTypeOrReduceTypeSpecial = 
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64 ||
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;

    double ratio;
    if (topoInfo->userRankSize == 0) {
        HCCL_WARNING("[AllReduceAutoSelector] the selector userRankSize not set");
        ratio = 1;
    } else {
        ratio = DEFAULT_RANK_SIZE / topoInfo->userRankSize / topoInfo->userRankSize;
    }
    bool isTwoLevelFlag = IsTwoLevelNetLayer(topoInfo);
    bool overSequenceDataThreshold = dataSize > AR_AICPU_SEQUENCE_DATA_SIZE;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (isDataTypeOrReduceTypeSpecial) {
            selectAlgName = dataSize <= AR_AICPU_1D_64DATATYPE_DATA_SIZE ?
                            "InsAllReduceMesh1DOneShot" :
                            "InsAllReduceMesh1DTwoShot";
        } else if (dataSize <= AR_AICPU_1D_SMALL_DATA_SIZE) {
            selectAlgName = "InsAllReduceMesh1DOneShot";
        } else if (dataSize * ratio > AR_AICPU_1D_MAX_DATA_SIZE) {
            selectAlgName = (isTwoLevelFlag && overSequenceDataThreshold) ?
                "InsAllReduceMesh1DTwoShotZAxisDetour" : "InsAllReduceMesh1DTwoShotMeshChunk";
        } else {
            selectAlgName = "InsAllReduceMesh1DTwoShot";
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (isDataTypeOrReduceTypeSpecial) {
            selectAlgName = "InsAllReduceAicpuReduceNHR";
        } else {
            selectAlgName = "InsAllReduceNHR";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (topoInfo->level0PcieMix) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                if (isDataTypeOrReduceTypeSpecial) {
                    selectAlgName = dataSize <= AR_AICPU_1D_64DATATYPE_DATA_SIZE ?
                                    "InsAllReduceMesh1DOneShot" :
                                    "InsAllReduceMesh1DTwoShot";
                } else if (dataSize <= AR_AICPU_1D_SMALL_DATA_SIZE) {
                    selectAlgName = "InsAllReduceMesh1DOneShot";
                } else if (dataSize * ratio > AR_AICPU_1D_MAX_DATA_SIZE) { 
                    selectAlgName = "InsAllReduceMesh1DTwoShotMeshChunk";
                } else {
                    selectAlgName = "InsAllReduceMesh1DTwoShot";
                }
            } else {
                if (isDataTypeOrReduceTypeSpecial) {
                    selectAlgName = "InsAllReduceAicpuReduceNHR";
                } else {
                    selectAlgName = "InsAllReduceParallelMesh1DNHRPcie";
                }
            }
        } else { 
            return SelectMeshAlgoAicpuUBX(topoInfo, dataSize, selectAlgName, isDataTypeOrReduceTypeSpecial);
        }
    } else {
        HCCL_ERROR("[AllReduceAutoSelector] topo not match");
        return SelectorStatus::NOT_MATCH;
    }

    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[Algo][AllReduceAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    
    //aiv 模式不支持 PROD
    CHK_PRT_RET(opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_DEBUG("[Algo][AllReduceAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64 ||
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_DEBUG("[Algo][AllReduceAutoSelector] aiv mode not support UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

 
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (IsSmallData(dataSize)) {
        selectAlgName = "AivAllReduceMesh1DOneShot";
    } else {
        selectAlgName = "AivAllReduceMesh1DTwoShot";
    }
    HCCL_DEBUG("[AllReduceAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllReduceAutoSelector::SelectDPUAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    std::vector<HcclAlgoType> algos = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        algos = it->second;
    }
 
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType,
              algos[0], algos[1], algos[2], algos[3]);
    if (topoInfo->topoLevelNums > 1) {
        if ((topoInfo->deviceNumPerModule == 1) || (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
            selectAlgName = "InsAllReduceSequenceMeshNhrDPU";//对应executor最后register的第二个参数
            HCCL_INFO("Using algo InsAllReduceSequenceMeshNhrDPU");
            return SelectorStatus::MATCH;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            selectAlgName = "InsV2AllReduceOmniPipe";
            HCCL_INFO("Using algo InsV2AllReduceOmniPipe");
            return SelectorStatus::MATCH;
        }
    }
 
    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLREDUCE, 18, AllReduceAutoSelector);
} // namespace ops_hccl
