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
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start", __func__);
    HCCL_DEBUG("[AllGatherAutoSelector][%s] topoInfo topoLevelNums[%u]", __func__, topoInfo->topoLevelNums);
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "CcuAllGatherParallelMeshNHR";
        } else {
            HCCL_WARNING("[Algo][AllGatherAutoSelector] levelNum > 1 is not supported yet for 2d ccu_ms mode.");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HcclAlgoType levle0Algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 0)) {
        levle0Algo = it->second[0];
    }
    if (IsDefaultAlg(levle0Algo) || levle0Algo == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH) {
        HCCL_DEBUG("[AllGatherAutoSelector][%s] SelectMeshAlgo", __func__);
        return SelectMeshAlgo(topoInfo, opParam, selectAlgName);
    } else {
        HCCL_WARNING("[Algo][AllGatherAutoSelector] algo[%u] is not supported yet for ccu_ms mode, reset to default.",
                     levle0Algo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] end", __func__);
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
            HCCL_ERROR("[Algo][AllGatherAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
        CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Algo][AllGatherAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
        if (dataSize > SMALL_COUNT_512KB) {
            // 大数据量场景，4P内并发executor，4P外回退ccu_sched模式
            if (isMeshNumEqualToClosNum && (topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO)) {
                selectAlgName = "CcuAllGatherConcurrentMesh1DNHR";
                return SelectorStatus::MATCH;
            } else {
                HCCL_WARNING("[Algo][AllGatherAutoSelector] Level0Shape::MESH_1D_CLOS in large data scene is not supported for ccu_ms mode, reset to default.");
                return SelectorStatus::NOT_MATCH;
            }
        } else {
                selectAlgName = "CcuAllGatherMesh1DUBX";
                return SelectorStatus::MATCH;
            }
        } else {
        HCCL_WARNING("[Algo][AllGatherAutoSelector] Level0Topo[%u] is not supported for ccu_ms mode, reset to default.", topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] end", __func__);
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start", __func__);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->deviceNumPerModule > 1) {
                selectAlgName = "CcuAllGatherParallelMesh1DNHR";
                return SelectorStatus::MATCH;
            } else {
                selectAlgName = "CcuAllGatherNHR1DMem2Mem";
                return SelectorStatus::MATCH;
            }
        } else {
            HCCL_WARNING("[Algo][AllGatherAutoSelector] level0Topo[%d] is not supported yet for ccu schedule mode.",
                         topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    }

    HcclAlgoType levle0Algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 0)) {
        levle0Algo = it->second[0];
    }
    const bool isDefaultOrFullMesh = IsDefaultAlg(levle0Algo) || (levle0Algo == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH);
    if (isDefaultOrFullMesh && (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
        if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_REGULAR) {
            selectAlgName = "CcuAllGatherMesh2Die";
        } else if (topoInfo->level0MeshType == Level0MeshType::TWO_DIE_NOT_REGULAR) {
            HCCL_INFO("[Algo][%s] TWO_DIE_NOT_REGULAR not match", __func__);
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuAllGatherMeshMem2Mem1D";
        }
        return SelectorStatus::MATCH;
    } else if (isDefaultOrFullMesh && (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS)) {
        // UBX机型
        bool isMeshNumEqualToClosNum = false;
        bool isClosNumMultipleOfMeshNum = false;
        CHK_PRT_RET(CheckMeshNumEqualToClosNum(topoInfo, isMeshNumEqualToClosNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Algo][AllGatherAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
        CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Algo][AllGatherAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
        if (dataSize > SMALL_COUNT_512KB) {
            if (isMeshNumEqualToClosNum && (topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO)) {
                selectAlgName = "CcuAllGatherConcurrentMesh1DNHRMem";
                return SelectorStatus::MATCH;
            } else if (isClosNumMultipleOfMeshNum) {
                selectAlgName = "CcuAllGatherParallelMesh1DNHRMemUBX";
                return SelectorStatus::MATCH;
            } else {
                selectAlgName = "CcuAllGatherNHR1DMem2MemUBX";
                return SelectorStatus::MATCH;
            }
        } else {
                selectAlgName = "CcuAllGatherMesh1DMem2MemUBX";
                return SelectorStatus::MATCH;
        }
    } else if ((IsDefaultAlg(levle0Algo) || (levle0Algo == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH))) {
        selectAlgName = "CcuAllGatherMesh2DMem2Mem";
        return SelectorStatus::MATCH;
    } else {
        HCCL_WARNING(
            "[Algo][AllGatherAutoSelector] algo[%u] is not supported yet for ccu_schedule mode, reset to default.",
            levle0Algo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] end", __func__);
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start", __func__);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    std::vector<HcclAlgoType> algos =
        std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType,
              algos[0], algos[1], algos[2], algos[3]);
    HCCL_INFO("[AllGatherAutoSelector][SelectAicpuAlgo] topoLevelNums=[%d], deviceNumPerModule=[%d], level0Topo=[%d]",
              topoInfo->topoLevelNums, topoInfo->deviceNumPerModule, topoInfo->level0Topo);
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->deviceNumPerModule == 1) {
            selectAlgName = "InsAllGatherNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsAllGatherParallelMesh1DNHR";
        } else {
            HCCL_WARNING("[AllGatherAutoSelector] topo not match");
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
            HCCL_ERROR("[Algo][AllGatherAutoSelector] CheckMeshNumEqualToClosNum failed."), SelectorStatus::NOT_MATCH);
            CHK_PRT_RET(CheckClosNumMultipleOfMeshNum(topoInfo, isClosNumMultipleOfMeshNum) != HCCL_SUCCESS,
            HCCL_ERROR("[Algo][AllGatherAutoSelector] CheckClosNumMultipleOfMeshNum failed."), SelectorStatus::NOT_MATCH);
            if (dataSize > SMALL_COUNT_512KB) {
                if (isMeshNumEqualToClosNum && (topoInfo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO)) {
                    selectAlgName = "InsAllGatherConcurrentMesh1DNHR";
                    return SelectorStatus::MATCH;
                } else if (isClosNumMultipleOfMeshNum) { 
                    selectAlgName = "InsAllGatherParallelMesh1DNHRUBX";
                    return SelectorStatus::MATCH;
                } else {
                    selectAlgName = "InsAllGatherNHRUBX";
                    return SelectorStatus::MATCH;
                } 
            } else {
                    selectAlgName = "InsAllGatherMesh1DUBX";
                    return SelectorStatus::MATCH;
            }
        } else {
            HCCL_WARNING("[AllGatherAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] end", __func__);
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectAivAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start", __func__);
    std::vector<HcclAlgoType> algos =
        std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if (it != configAlgMap.end()) {
        algos = it->second;
    }
    HCCL_INFO("hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType,
              algos[0], algos[1], algos[2], algos[3]);

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "AivAllGatherMesh1D";
    } else {
        HCCL_WARNING("[AllGatherAutoSelector] topo not match for aiv algo");
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] end", __func__);
    return SelectorStatus::MATCH;
}

SelectorStatus AllGatherAutoSelector::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam, const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
    std::string &selectAlgName) const
{
    HCCL_DEBUG("[AllGatherAutoSelector][%s] start", __func__);
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
            selectAlgName = "InsAllGatherMeshNhr";
            return SelectorStatus::MATCH;
        }
    }
    HCCL_DEBUG("[AllGatherAutoSelector][%s] end", __func__);
    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_ALLGATHER, 18, AllGatherAutoSelector);

}  // namespace ops_hccl
