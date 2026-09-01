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
#include "hccl_aiv_utils.h"
#include "ins_v2_reduce_scatter_order_preserved_executor.h"
#include "order_preserved_common.h"

namespace ops_hccl {
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u32 MAX_RANK_NUM_FOR_REDUCE_MS_ALGO = 8;
constexpr u64 RS_AICPU_1D_MAX_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 RS_FLATTEN_MAX_DATA_SIZE = 512 * 1024;
constexpr u64 RS_AICPU_1D_MIN_DATA_SIZE = 4 * 1024 * 1024;
constexpr u64 RS_AICPU_1D_TWO_LEVEL_DATA_SIZE_THRESHOLD = 1536 * 1024 * 1024;

constexpr u64 RS_CCU_64P_MIN_DATA_SIZE = 128 * 1024 * 1024;
constexpr u64 RS_CCU_64P_SEQ_DATA_SIZE = 16 * 1024 * 1024;
constexpr u64 RS_CCU_8P_MIN_DATA_SIZE = 64 * 1024 * 1024;
constexpr u64 RS_AICPU_SEQUENCE_SIZE_THRESHOLD = 4ULL * 1024 * 1024 * 1024;

constexpr u32 RS_CCU_MAX_RANK_SIZE = 64;
constexpr u64 RS_2P_DETOUR_DATA_SIZE = 4 * 1024 * 1024;
constexpr u64 OMNI_PCIE_RS_DATA_SIZE = 4 * 1024 * 1024;
constexpr u64 OMNI_UBX_RS_SCHED_DATA_SIZE = 4 * 1024 * 1024;
constexpr u64 OMNI_UBX_RS_MS_DATA_SIZE = 2 * 1024 * 1024;
constexpr u32 DEVICE_NUM_PER_MODULE_8 = 8;

SelectorStatus ReduceScatterAutoSelector::SelectCcuMsAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;

    // 保序模式不支持CCU_MS，需要回退到AICPU
    CHK_PRT_RET(
        IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize),
        HCCL_DEBUG(
            "[ReduceScatterAutoSelector] DETERMINISTIC_STRICT mode not supported for CCU_MS, fallback to AICPU."),
        SelectorStatus::NOT_MATCH);

    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[ReduceScatterAutoSelector] layerNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    }

    // 2P场景且数据量大于阈值时回退到AICPU
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (IsTwoLevelNetLayer(topoInfo, opParam) && topoInfo->userRankSize == 2 && dataSize >= RS_2P_DETOUR_DATA_SIZE) {
        HCCL_DEBUG(
            "[ReduceScatterAutoSelector] 2P scenario with data size[%llu], "
            "fallback to AICPU for better performance.",
            dataSize);
        return SelectorStatus::NOT_MATCH;
    }

    // MS 模式不支持 int8
    CHK_PRT_RET(
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_WARNING(
            "[ReduceScatterAutoSelector] dataType[%d] is not supported yet for ccu_ms mode.", opParam.DataDes.dataType),
        SelectorStatus::NOT_MATCH);

    // MS 模式不支持 PROD
    CHK_PRT_RET(
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING(
            "[ReduceScatterAutoSelector] ReduceOp[%d] is not supported yet for ccu_ms mode.", opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[ReduceScatterAutoSelector] ccu_ms mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    SelectorStatus ret = SelectMeshAlgoCcums(topoInfo, opParam, selectAlgName);
    if (ret == SelectorStatus::MATCH) {
        HCCL_INFO("[ReduceScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    }
    return ret;
}

SelectorStatus ReduceScatterAutoSelector::SelectMeshAlgoCcums(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (IsInputOutputOverlap(opParam) == true) { // 不支持 inplace 场景
            return SelectorStatus::NOT_MATCH;
        }
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            selectAlgName = "CcuMSReduceScatterSoleMesh2Die";
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            HCCL_INFO("[%s] TWO_DIE_NOT_REGULAR not match", __func__);
            return SelectorStatus::NOT_MATCH;
        } else {
            if (IsDevType960()
                && (dataSize * topoInfo->userRankSize > SMALL_COUNT_16M && IsTwoLevelNetLayer(topoInfo, opParam))) {
                selectAlgName = "CcuMSReduceScatterSoleMeshConcur";
            } else {
                selectAlgName = "CcuMSReduceScatterSoleMesh";
            }
        }
    } else if (
        topoInfo->level0Topo
        == Level0Shape::MESH_1D_CLOS) { // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
        if (topoInfo->level0PcieMix && !IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            HCCL_WARNING("[ReduceScatterAutoSelector] pcie mixed topo is not supported yet for ccu ms mode.");
            return SelectorStatus::NOT_MATCH;
        }
        // UBX机型
        bool isMeshNumEqualToClosNum = false;
        bool isClosNumMultipleOfMeshNum = false;
        CHK_PRT_RET(
            CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceScatterAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
        CHK_PRT_RET(
            CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceScatterAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
        if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) { // 4P mesh
            if (IsSmallData(dataSize)) { // 小数据量，用1d mesh算法
                selectAlgName = "CcuMSReduceScatterSoleMesh";
            } else { // 大数据量，用mesh+clos并行算法
                selectAlgName = "CcuMSReduceScatterConcurMeshNHRMultiLink";
            }
        } else if (isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
            if (dataSize < OMNI_UBX_RS_MS_DATA_SIZE) {
                HCCL_WARNING("[%s] MESH_1D_CLOS not match.", __func__);
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuMSReduceScatterPipeLineMeshNHR";
            }
        } else if (topoInfo->userRankSize <= MAX_RANK_NUM_FOR_REDUCE_MS_ALGO) {
            selectAlgName = "CcuMSReduceScatterSoleMesh";
        } else {
            HCCL_DEBUG("[ReduceScatterAutoSelector] level0Topo[%u] is not supported mesh yet.", topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        HCCL_WARNING(
            "[ReduceScatterAutoSelector] level0Topo[%d] is not supported yet for ccu_ms mode.", topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[ReduceScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    if (topoInfo->level2UbRtp) {
        HCCL_INFO(
            "[ReduceScatterAutoSelector][%s] ccu schedule is not supported with level2UbRtp, reset to default.",
            __func__);
        return SelectorStatus::NOT_MATCH;
    }
    if (topoInfo->topoLevelNums >= TOPO_LEVEL_NUM_3) {
        HCCL_INFO(
            "[ReduceScatterAutoSelector][%s] ccu schedule is not supported when topoLevelNums >= 3(levelNum[%u]), "
            "reset to default.",
            __func__, topoInfo->topoLevelNums);
        return SelectorStatus::NOT_MATCH;
    }
    (void)configAlgMap;
    u32 ccuSize = RS_CCU_MAX_RANK_SIZE;

    // 保序模式不支持CCU_SCHED，需要回退到AICPU
    CHK_PRT_RET(
        IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize),
        HCCL_DEBUG(
            "[ReduceScatterAutoSelector] DETERMINISTIC_STRICT mode not supported for CCU_SCHED, fallback to AICPU."),
        SelectorStatus::NOT_MATCH);

    // ccu 模式不支持 PROD
    CHK_PRT_RET(
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_WARNING(
            "[ReduceScatterAutoSelector] ReduceOp[%d] is not supported yet for ccu schedule mode.", opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    // ccu 模式不支持 inplace 场景
    CHK_PRT_RET(
        IsInputOutputOverlap(opParam) == true,
        HCCL_WARNING("[ReduceScatterAutoSelector] ccu schedule mode not support inplace."), SelectorStatus::NOT_MATCH);

    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (Is64BitDataType(opParam.DataDes.dataType)) {
        HCCL_WARNING("[ReduceScatterAutoSelector] ccu_schedule mode not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    constexpr u64 CCU_SCHEDULE_2LEVEL_MAX_PER_RANK_DATA_SIZE = 32ULL * 1024 * 1024;

    u32 frameNum = AutoSelectorBase::CalcFrameNum(topoInfo);

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (dataSize >= CCU_SCHEDULE_2LEVEL_MAX_PER_RANK_DATA_SIZE) {
                HCCL_INFO(
                    "[ReduceScatterAutoSelector] 2 level topo perRankDataSize[%llu] exceeds limit, "
                    "fallback to aicpu.",
                    dataSize);
                return SelectorStatus::NOT_MATCH;
            }
            // Level1Nhr 已在 CalcTopoShape 中设置（GCD==1 时为 true）
            if (topoInfo->Level1Nhr) {
                selectAlgName = "CcuSchedReduceScatterSoleNHR";
                HCCL_INFO("[ReduceScatterAutoSelector] Level1Nhr=true, select [%s]", selectAlgName.c_str());
                return SelectorStatus::MATCH;
            } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0) > 1) {
                CHK_PRT_RET(
                    opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
                    HCCL_WARNING(
                        "[ReduceScatterAutoSelector] dataType[%d] is not supported yet for ccu schedule mode.",
                        opParam.DataDes.dataType),
                    SelectorStatus::NOT_MATCH);
                // 2*8p 组网且总数据量在 [4M, 16M] 时选择 2Die mem2mem 算法（仅支持 2 框 8 卡组网）
                if (topoInfo->userRankSize == RS_CCU_2DIE_RANK_SIZE && frameNum == RS_CCU_2DIE_FRAME_NUM
                    && dataSize * topoInfo->userRankSize >= RS_CCU_2DIE_MIN_DATA_SIZE
                    && dataSize * topoInfo->userRankSize <= RS_CCU_2DIE_MAX_DATA_SIZE) {
                    selectAlgName = "CcuSchedReduceScatterSoleMesh2Die";
                    return SelectorStatus::MATCH;
                } else if (
                    (dataSize * topoInfo->userRankSize) < RS_FLATTEN_MAX_DATA_SIZE
                    && topoInfo->userRankSize < ccuSize) {
                    selectAlgName = "CcuSchedReduceScatterSoleMesh";
                    return SelectorStatus::MATCH;
                } else if (
                    dataSize * topoInfo->userRankSize < RS_CCU_64P_SEQ_DATA_SIZE && topoInfo->userRankSize < ccuSize
                    && frameNum <= MAX_FRAME_NUM_FOR_CCU_ALGO) {
                    selectAlgName = "CcuSchedReduceScatterSequenceMeshMesh";
                    return SelectorStatus::MATCH;
                } else if (
                    dataSize * topoInfo->userRankSize <= RS_CCU_64P_SEQ_DATA_SIZE && topoInfo->userRankSize == ccuSize
                    && frameNum <= MAX_FRAME_NUM_FOR_CCU_ALGO) {
                    selectAlgName = "CcuSchedReduceScatterSequenceMeshMesh";
                    return SelectorStatus::MATCH;
                } else if (
                    dataSize * topoInfo->userRankSize <= RS_CCU_64P_MIN_DATA_SIZE && topoInfo->userRankSize == ccuSize
                    && frameNum <= MAX_FRAME_NUM_FOR_CCU_ALGO) {
                    selectAlgName = "CcuSchedReduceScatterParallelMeshNHR"; // 64M以下跑ccu
                    return SelectorStatus::MATCH;
                } else if (frameNum > MAX_FRAME_NUM_FOR_CCU_ALGO) {
                    // 框数超过 kernel repeatNum 上限，fallback 到 NHR1DMem2Mem
                    HCCL_INFO(
                        "[ReduceScatterAutoSelector] frameNum[%u] > %u, fallback to NHR1DMem2Mem.", frameNum,
                        MAX_FRAME_NUM_FOR_CCU_ALGO);
                    selectAlgName = "CcuSchedReduceScatterSoleNHR";
                    return SelectorStatus::MATCH;
                } else {
                    return SelectorStatus::NOT_MATCH; // 64M以上切为aicpu
                }
            } else {
                selectAlgName = "CcuSchedReduceScatterSoleNHR";
                return SelectorStatus::MATCH;
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "CcuSchedReduceScatterSoleNHR";
            return SelectorStatus::MATCH;
        } else {
            HCCL_WARNING(
                "[SelectCcuScheduleAlgo] layer0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoCcuSchedule(topoInfo, opParam, selectAlgName);
    }
    HCCL_INFO("[ReduceScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectMeshAlgoCcuScheduleMesh1D(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    double ratio; // 以8卡为基线确定ratio，用来表示不同卡数对下发的影响系数
    if (topoInfo->userRankSize == 0) {
        HCCL_WARNING("[ReduceScatterAutoSelector]the selector is not set topoInfo->userRankSize]");
        ratio = 1;
    } else {
        ratio = DEFAULT_RANK_SIZE / topoInfo->userRankSize;
    }
    if (dataSize * ratio >= RS_M2M_1D_MAX_DATA_SIZE) {
        HCCL_DEBUG(
            "[ReduceScatterAutoSelector] dataSize[%lu] * ratio[%f] >= MAX_DATA_SIZE[%lu].", dataSize, ratio,
            RS_M2M_1D_MAX_DATA_SIZE);
        return SelectorStatus::NOT_MATCH;
    }
    if (IsInputOutputOverlap(opParam) == true) { // 不支持 inplace 场景
        HCCL_WARNING("[ReduceScatterAutoSelector] ccu_ms mode not support inplace.");
        return SelectorStatus::NOT_MATCH;
    }
    if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
        selectAlgName = "CcuSchedReduceScatterSoleMesh2Die";
    } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
        HCCL_DEBUG("[ReduceScatterAutoSelector] TWO_DIE_NOT_REGULAR not match.");
        return SelectorStatus::NOT_MATCH;
    } else {
        selectAlgName = "CcuSchedReduceScatterSoleMesh";
    }
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectMeshAlgoCcuSchedule(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
{
    // ccu 模式不支持 inplace 场景
    CHK_PRT_RET(
        IsInputOutputOverlap(opParam) == true,
        HCCL_WARNING("[ReduceScatterAutoSelector] ccu schedule mode not support inplace."), SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    CHK_PRT_RET(
        opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT8,
        HCCL_WARNING(
            "[ReduceScatterAutoSelector] dataType[%d] is "
            "not supported yet for ccu_schedule mode with ms reduce.",
            opParam.DataDes.dataType),
        SelectorStatus::NOT_MATCH);
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        return SelectMeshAlgoCcuScheduleMesh1D(topoInfo, opParam, selectAlgName);
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
        if (topoInfo->level0PcieMix) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                return SelectMeshAlgoCcuScheduleMesh1D(topoInfo, opParam, selectAlgName);
            } else {
                HCCL_WARNING("[ReduceScatterAutoSelector] pcie mixed topo is not supported yet for ccu sched mode.");
                return SelectorStatus::NOT_MATCH;
            }
        }
        // UBX机型
        bool isMeshNumEqualToClosNum = false;
        bool isClosNumMultipleOfMeshNum = false;
        CHK_PRT_RET(
            CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
            HCCL_DEBUG("[ReduceScatterAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
        CHK_PRT_RET(
            CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_DEBUG("[ReduceScatterAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
        if (isMeshNumEqualToClosNum && topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) {
            // 4P mesh
            if (IsSmallData(dataSize)) {
                // 小数据量，用1d mesh算法
                selectAlgName = "CcuSchedReduceScatterSoleMesh";
            } else {
                // 大数据量，用mesh+clos并行算法
                selectAlgName = "CcuSchedReduceScatterConcurMeshNHRMultiLink";
            }
        } else if (isClosNumMultipleOfMeshNum && !IsSmallData(dataSize)) {
            // 矩形场景大数据量，用2d并行算法
            if (dataSize < OMNI_UBX_RS_SCHED_DATA_SIZE) {
                selectAlgName = "CcuSchedReduceScatterParallelMeshNHRMultiLink";
            } else {
                selectAlgName = "CcuSchedReduceScatterPipeLineMeshNHR";
            }
        } else {
            // 其他场景，用1d NHR算法
            selectAlgName = "CcuSchedReduceScatterSoleNHRMultiLink";
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (topoInfo->level0PcieMix) {
            // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
            HCCL_WARNING("[ReduceScatterAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        }
        selectAlgName = "CcuSchedReduceScatterSoleNHR";
    } else {
        HCCL_DEBUG("[ReduceScatterAutoSelector] MESH_1D_CLOS not match.");
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;

    if (IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize)) {
        if (topoInfo->userRankSize > MAX_RANK_NUM_FOR_ORDER_PRESERVED) {
            // 内部reducescatter中采用分组all2all
            selectAlgName = "AicpuReduceScatterStrictOrderedGroupMesh";
        } else {
            // 内部reducescatter中采用非分组all2all
            selectAlgName = "AicpuReduceScatterStrictOrderedMesh";
        }
        HCCL_INFO(
            "[ReduceScatterAutoSelector] DETERMINISTIC_STRICT mode, rankSize[%u], threshold[%u], "
            "select [%s]",
            topoInfo->userRankSize, MAX_RANK_NUM_FOR_ORDER_PRESERVED, selectAlgName.c_str());
        return SelectorStatus::MATCH;
    }

    if (topoInfo->topoLevelNums > 1) {
        bool level0AndLevel1Symetric = topoInfo->level0Symmetric && topoInfo->level1Symmetric;
        if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            selectAlgName = "AicpuReduceScatterSoleNHRAicpuReduce";
        } else if (
            level0AndLevel1Symetric && topoInfo->deviceNumPerModule == DEVICE_NUM_PER_MODULE_8
            && topoInfo->topLevelUboe) {
            selectAlgName = "AicpuReduceScatterPipeLine";
        } else if (level0AndLevel1Symetric && topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3 && topoInfo->topLevelUboe) {
            selectAlgName = "InsReduceScatterParallelNHRNHRUboe";
        } else if (topoInfo->Level1Nhr) {
            selectAlgName = "AicpuReduceScatterSoleNHR";
            HCCL_INFO("[ReduceScatterAutoSelector] Level1Nhr=true, select [%s]", selectAlgName.c_str());
        } else if (topoInfo->Level0Nhr) {
            selectAlgName = "AicpuReduceScatterSoleNHR"; // InsReduceScatterParallelNHRNHR备用
        } else if (
            topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0) > 1
            && topoInfo->level0Topo == Level0Shape::MESH_1D) {
            constexpr u64 AICPU_MAX_RANKSIZE = 256;
            constexpr u64 AICPU_2LEVEL_MAX_TOTAL_DATA_SIZE = 1ULL * 1024 * 1024 * 1024;
            if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3) {
                if (level0AndLevel1Symetric) {
                    selectAlgName = "AicpuReduceScatterSequenceMeshConcurNHRNHR";
                } else {
                    selectAlgName = "AicpuReduceScatterSoleNHR";
                }

            } else if (
                dataSize * topoInfo->userRankSize >= AICPU_2LEVEL_MAX_TOTAL_DATA_SIZE
                && topoInfo->userRankSize >= AICPU_MAX_RANKSIZE) {
                selectAlgName = "AicpuReduceScatterParallelMeshNHR";
            } else if (dataSize > RS_AICPU_1D_MIN_DATA_SIZE) {
                selectAlgName = (dataSize * topoInfo->userRankSize > RS_AICPU_SEQUENCE_SIZE_THRESHOLD) ?
                                    "AicpuReduceScatterSequenceMeshConcurNHR" :
                                    "AicpuReduceScatterParallelMeshNHR";
            } else {
                selectAlgName = "AicpuReduceScatterSoleNHR";
            }
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0) == 1) {
            selectAlgName = "AicpuReduceScatterSoleNHR"; // InsReduceScatterParallelNHRNHR备用
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "AicpuReduceScatterSoleNHRMultiLink";
        } else {
            HCCL_ERROR(
                "[ReduceScatterAutoSelector] topo not match, level0Topo [%d], deviceNumPerModule [%d]",
                topoInfo->level0Topo, topoInfo->netLayerDetails.localNetInsSizeOfLayer.at(0));
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }

    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectMeshAlgoAicpuForMesh1D(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName, u64 dataSize,
    double ratio) const
{
    if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
        selectAlgName = "AicpuReduceScatterSoleMesh";
    } else {
        if (IsTwoLevelNetLayer(topoInfo, opParam)) {
            if (topoInfo->userRankSize == 2 && dataSize >= RS_2P_DETOUR_DATA_SIZE) {
                selectAlgName = "AicpuReduceScatterSoleMeshConcur";
            } else if (dataSize * topoInfo->userRankSize > RS_AICPU_1D_TWO_LEVEL_DATA_SIZE_THRESHOLD) {
                selectAlgName = "AicpuReduceScatterSoleMeshChunk";
            } else if (dataSize * ratio > RS_AICPU_1D_MAX_DATA_SIZE) {
                selectAlgName = "AicpuReduceScatterSoleMeshChunk";
            } else {
                selectAlgName = "AicpuReduceScatterSoleMesh";
            }
        } else {
            if (dataSize * ratio > RS_AICPU_1D_MAX_DATA_SIZE) {
                selectAlgName = "AicpuReduceScatterSoleMeshChunk";
            } else {
                selectAlgName = "AicpuReduceScatterSoleMesh";
            }
        }
    }
    HCCL_DEBUG("[%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectMeshAlgoAicpu(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    double ratio;
    if (topoInfo->userRankSize == 0) {
        HCCL_WARNING("[ReduceScatterAutoSelector]the selector is not set userRankSize]");
        ratio = 1;
    } else {
        ratio = (DEFAULT_RANK_SIZE / topoInfo->userRankSize) * (DEFAULT_RANK_SIZE / topoInfo->userRankSize);
    }
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        return SelectMeshAlgoAicpuForMesh1D(topoInfo, opParam, selectAlgName, dataSize, ratio);
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            selectAlgName = "AicpuReduceScatterSoleNHRAicpuReduce";
        } else {
            selectAlgName = "AicpuReduceScatterSoleNHRMultiLink";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        bool isClosNumMultipleOfMeshNum = false;
        CHK_PRT_RET(
            CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceScatterAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
        return SelectMeshAlgoAicpuForMesh1DClos(
            topoInfo, opParam, dataSize, ratio, isClosNumMultipleOfMeshNum, selectAlgName);
    } else {
        HCCL_WARNING("[ReduceScatterAutoSelector] topo not match");
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectAivAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->level2UbRtp) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG,
            "[ReduceScatterAutoSelector][%s] aiv is not supported with level2UbRtp, reset to default.", __func__);
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums >= TOPO_LEVEL_NUM_3) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG,
            "[ReduceScatterAutoSelector][%s] aiv is not supported when topoLevelNums >= 3(levelNum[%u]), reset to "
            "default.",
            __func__, topoInfo->topoLevelNums);
        return SelectorStatus::NOT_MATCH;
    }

    // 保序模式不支持AIV，需要回退到AICPU
    CHK_PRT_RET(
        IsNeedStrictModeForOrderPreserved(opParam, topoInfo->userRankSize),
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG,
            "[ReduceScatterAutoSelector] DETERMINISTIC_STRICT mode is not supported yet for AIV mode."),
        SelectorStatus::NOT_MATCH);

    // aiv 模式不支持 PROD
    CHK_PRT_RET(
        opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD,
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_WARNING, "[ReduceScatterAutoSelector] ReduceOp[%d] is not supported yet for aiv mode.",
            opParam.reduceType),
        SelectorStatus::NOT_MATCH);

    if (opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64
        || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64) {
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[ReduceScatterAutoSelector] aiv mode not support UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->userRankSize > MAX_RANK_SIZE) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[ReduceScatterAutoSelector][%s] rankSize[%u] larger than [%u]", __func__,
            topoInfo->userRankSize, MAX_RANK_SIZE);
        return SelectorStatus::NOT_MATCH;
    }

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_PRT_RET(
        HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize) != HCCL_SUCCESS,
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[ReduceScatterAutoSelector] HcclGetHcclBuffer failed."),
        SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 totalSize = opParam.DataDes.count * perDataSize * topoInfo->userRankSize;
    if (opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY
        && totalSize >= AIV_MAX_PER_RANK_DATA_SIZE * topoInfo->userRankSize) {
        HCCL_DEBUG(
            "[ReduceScatterAutoSelector][%s] totalSize[%llu] larger than AIV_MAX_PER_RANK_DATA_SIZE[%llu] * "
            "rankSize[%u]",
            __func__, totalSize, AIV_MAX_PER_RANK_DATA_SIZE, topoInfo->userRankSize);
        return SelectorStatus::NOT_MATCH;
    }
    if (totalSize > cclBufferSize * AIV_MAX_CCL_LOOP_NUM) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[ReduceScatterAutoSelector][%s] totalSize[%llu] too large for cclBufferSize [%llu]",
            __func__, totalSize, cclBufferSize);
        return SelectorStatus::NOT_MATCH;
    }

    selectAlgName = "AivReduceScatterSoleMesh";
    HCCL_DEBUG("[ReduceScatterAutoSelector][%s] end, selectAlgName[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    HCCL_INFO(
        "topoInfo->topoLevelNums is %u, topoInfo->level0Topo is %u", topoInfo->topoLevelNums, topoInfo->level0Topo);
    (void)configAlgMap;
    bool isDataTypeOrReduceTypeSpecial = opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64
                                         || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_UINT64
                                         || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_FP64
                                         || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;
    if (isDataTypeOrReduceTypeSpecial) {
        HCCL_INFO("[ReduceScatterAutoSelector][SelectDPUAlgo] not support INT64, UINT64, FP64.");
        return SelectorStatus::NOT_MATCH;
    } else if (topoInfo->topoLevelNums > 1) {
        if ((topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1)
            || (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
            selectAlgName = "DpuReduceScatterSequenceMeshMesh";
            HCCL_DEBUG("[ReduceScatterAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
            return SelectorStatus::MATCH;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (!topoInfo->level0PcieMix) {
                selectAlgName = "DpuReduceScatterPipeLineUBX";
                HCCL_INFO("Using algo DpuReduceScatterPipeLineUBX");
                return SelectorStatus::MATCH;
            } else {
                selectAlgName = "DpuReduceScatterSequenceMeshMesh";
                HCCL_INFO("Using algo DpuReduceScatterSequenceMeshMesh");
                return SelectorStatus::MATCH;
            }
        } else {
            selectAlgName = "DpuReduceScatterSequenceMeshMesh";
            HCCL_INFO("Using algo DpuReduceScatterSequenceMeshMesh");
            return SelectorStatus::MATCH;
        }
    }

    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ReduceScatterAutoSelector::SelectMeshAlgoAicpuForMesh1DClos(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, u64 dataSize, double ratio,
    bool isClosNumMultipleOfMeshNum, std::string& selectAlgName) const
{
    if (topoInfo->level0PcieMix) {
        // PCIE机型算法选择
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            return SelectMeshAlgoAicpuForMesh1D(topoInfo, opParam, selectAlgName, dataSize, ratio);
        } else if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            selectAlgName = "InsReduceScatterSequenceMesh1DNHRAicpuReducePcie";
        } else if (dataSize < OMNI_PCIE_RS_DATA_SIZE) {
            selectAlgName = "InsReduceScatterParallelMesh1DNHRPcie";
        } else {
            selectAlgName = "AicpuReduceScatterPipeLinePcie";
        }
    } else if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
        // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
        if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
            selectAlgName = "AicpuReduceScatterSoleMesh";
        } else if (!IsSmallData(dataSize)) {
            selectAlgName = "AicpuReduceScatterConcurMeshNHR";
        } else if (dataSize * ratio > RS_AICPU_1D_MAX_DATA_SIZE) {
            selectAlgName = "AicpuReduceScatterSoleMeshChunk";
        } else {
            selectAlgName = "AicpuReduceScatterSoleMesh";
        }
    } else if (Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD) {
        selectAlgName = "AicpuReduceScatterSoleNHRAicpuReduce";
    } else if (isClosNumMultipleOfMeshNum && IsLargeData(dataSize)) {
        if (opParam.supportSymmetricMemory) {
            selectAlgName = "AicpuReduceScatterPipeLineUBX";
        } else {
            selectAlgName = "InsReduceScatterParallelMesh1DNHRUBX";
        }
    } else {
        selectAlgName = "AicpuReduceScatterSoleNHR";
    }
    HCCL_DEBUG("[%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, 18, ReduceScatterAutoSelector);
} // namespace ops_hccl
