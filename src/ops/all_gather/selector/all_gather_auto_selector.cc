/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_auto_selector.h"
#include "selector_registry.h"

namespace ops_hccl {
constexpr u64 AG_2D_SMALL_DATA_SIZE = 1024 * 1024;
constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;

SelectorStatus AllGatherAutoSelector::SelectCcuMsAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "CcuAllGatherParallelMeshNHR";
        } else {
            HCCL_WARNING("[AllGatherAutoSelector] levelNum > 1 is not supported yet for 2d ccu_ms mode.");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgo(topoInfo, opParam, selectAlgName);
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectMeshAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                     std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start", __func__);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "CcuAllGatherMesh1D";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        // UBX机型
        bool isMeshNumEqualToClosNum = false;
        bool isClosNumMultipleOfMeshNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
            HCCL_DEBUG("[AllGatherAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
        CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_DEBUG("[AllGatherAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
        if (dataSize > SMALL_COUNT_512KB) {
            // 大数据量场景，4P内并发executor，4P外回退ccu_sched模式
            if (isMeshNumEqualToClosNum && (topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO)) {
                selectAlgName = "CcuAllGatherConcurrentMesh1DNHR";
                return SelectorStatus::MATCH;
            } else {
                HCCL_DEBUG("[AllGatherAutoSelector] Level0Shape::MESH_1D_CLOS in large data scene is not supported for ccu_ms mode, reset to default.");
                return SelectorStatus::NOT_MATCH;
            }
        } else {
                selectAlgName = "CcuAllGatherMesh1DUBX";
                return SelectorStatus::MATCH;
            }
        } else {
        HCCL_DEBUG("[AllGatherAutoSelector] Level0Topo[%u] is not supported for ccu_ms mode, reset to default.", topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectCcuScheduleUBXAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, std::string &selectAlgName, const u64 dataSize) const
{
    // UBX机型
    bool isMeshNumEqualToClosNum = false;
    bool isClosNumMultipleOfMeshNum = false;
    CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
        HCCL_DEBUG("[AllGatherAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
    CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
        HCCL_DEBUG("[AllGatherAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
    if (dataSize > SMALL_COUNT_512KB) {
        if (isMeshNumEqualToClosNum && (topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO)) {
            selectAlgName = "CcuAllGatherConcurrentMesh1DNHRMem";
        } else if (isClosNumMultipleOfMeshNum) {
            selectAlgName = "CcuAllGatherParallelMesh1DNHRMemUBX";
        } else {
            selectAlgName = "CcuAllGatherNHR1DMem2MemUBX";
        }
    } else {
        selectAlgName = "CcuAllGatherMesh1DMem2MemUBX";
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectCcuScheduleLevel0Algo(
    const TopoInfoWithNetLayerDetails *topoInfo, std::string &selectAlgName, const u64 dataSize) const
{
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            selectAlgName = "CcuAllGatherMesh2Die";
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            HCCL_DEBUG("[AllGatherAutoSelector][%s] TWO_DIE_NOT_REGULAR not match", __func__);
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuAllGatherMesh1DMem2Mem";
        }
        return SelectorStatus::MATCH;
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        return SelectCcuScheduleUBXAlgo(topoInfo, selectAlgName, dataSize);
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->is2DieFullMesh) {
                HCCL_DEBUG("[AllGatherAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuAllGatherMesh1DMem2Mem";
            }
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                selectAlgName = "CcuAllGatherMesh1DMem2Mem";
            } else {
                selectAlgName = "CcuAllGatherParallelMesh1DNHR";
            }
        } else {
            HCCL_DEBUG("[AllGatherAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    }

    HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start", __func__);
    (void)configAlgMap;
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->is2DieFullMesh) {
                HCCL_DEBUG("[AllGatherAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                selectAlgName = "CcuAllGatherNHR1DMem2Mem";
            } else {
                selectAlgName = "CcuAllGatherParallelMesh1DNHR";
            }
        } else {
            HCCL_DEBUG("[AllGatherAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectCcuScheduleLevel0Algo(topoInfo, selectAlgName, dataSize);
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    HCCL_INFO("[AllGatherAutoSelector][SelectAicpuAlgo] topoLevelNums=[%d], deviceNumPerModule=[%d], level0Topo=[%d]",
              topoInfo->topoLevelNums, topoInfo->deviceNumPerModule, topoInfo->level0Topo);
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->Level1Nhr) {
            selectAlgName = "InsAllGatherNHR";
        } else if (topoInfo->Level0Nhr) {
            selectAlgName = "InsAllGatherNHR"; // 预留给NHRNHR
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgName = "InsAllGatherNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsAllGatherParallelMesh1DNHR";
        } else {
            HCCL_ERROR("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsAllGatherMesh1D";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            // UBX机型
            bool isMeshNumEqualToClosNum = false;
            bool isClosNumMultipleOfMeshNum = false;
            CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
            CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_ERROR("[AllGatherAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
            if (dataSize > SMALL_COUNT_512KB) {
                if (isMeshNumEqualToClosNum && (topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO)) {
                    selectAlgName = "InsAllGatherConcurrentMesh1DNHR";
                } else if (isClosNumMultipleOfMeshNum) {
                    selectAlgName = "InsAllGatherParallelMesh1DNHRUBX";
                } else {
                    selectAlgName = "InsAllGatherNHRUBX";
                }
            } else {
                selectAlgName = "InsAllGatherMesh1DUBX";
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "InsAllGatherNHR";
        } else {
            HCCL_ERROR("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAivAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    (void)configAlgMap;
    (void)opParam;

    selectAlgName = "AivAllGatherMesh1D";
    HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start, topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    if (topoInfo->topoLevelNums > 1) {
        if ((topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) || (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
            selectAlgName = "InsAllGatherMeshNhrDPU";
            HCCL_DEBUG("[AllGatherAutoSelector][%s] Algo match[%s]", __func__, selectAlgName.c_str());
            return SelectorStatus::MATCH;
        }
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] end", __func__);
    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLGATHER, 18, AllGatherAutoSelector);

}  // namespace ops_hccl
