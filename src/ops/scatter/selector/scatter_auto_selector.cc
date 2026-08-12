/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_auto_selector.h"
#include "selector_registry.h"
#include "hccl_aiv_utils.h"

namespace ops_hccl {
constexpr u64 OMNI2D_UBX_SC_DATA_SIZE = 16 * 1024 * 1024;
constexpr uint32_t TOPO_LEVEL_3 = 3;

SelectorStatus ScatterAutoSelector::SelectCcuMsAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    HCCL_WARNING("[Algo][ScatterAutoSelector] not supported yet for ccu_ms mode, reset to default.");
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ScatterAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->level2Ubg) {
        HCCL_INFO(
            "[ScatterAutoSelector][%s] ccu schedule is not supported with level2Ubg, reset to default.", __func__);
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums >= TOPO_LEVEL_NUM_3) {
        HCCL_INFO(
            "[ScatterAutoSelector][%s] ccu schedule is not supported when topoLevelNums >= 3(levelNum[%u]), reset to "
            "default.",
            __func__, topoInfo->topoLevelNums);
        return SelectorStatus::NOT_MATCH;
    }

    constexpr u64 CCU_SCHEDULE_2LEVEL_MAX_PER_RANK_DATA_SIZE = 1ULL * 1024 * 1024;
    constexpr u64 CCU_SCHEDULE_SCATTER_MAX_RANK_SIZE = 64;
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (dataSize > CCU_SCHEDULE_2LEVEL_MAX_PER_RANK_DATA_SIZE
                || topoInfo->userRankSize > CCU_SCHEDULE_SCATTER_MAX_RANK_SIZE) {
                HCCL_INFO(
                    "[ScatterAutoSelector] 2 level topo perRankDataSize[%llu] or rankSize[%u] exceeds limit, "
                    "fallback to aicpu.",
                    dataSize, topoInfo->userRankSize);
                return SelectorStatus::NOT_MATCH;
            }
            if (topoInfo->Level1Nhr) {
                selectAlgName = "CcuSchedScatterSoleNHR";
                HCCL_INFO("[ScatterAutoSelector] Level1Nhr=true, select [%s]", selectAlgName.c_str());
            } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                selectAlgName = "CcuSchedScatterSoleNHR";
            } else if (topoInfo->is2DieFullMesh) {
                HCCL_WARNING("[ScatterAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuSchedScatterParallelMeshNHR";
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            if (topoInfo->level0PcieMix) { // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
                HCCL_WARNING("pcie mixed topo is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            }
            selectAlgName = "CcuSchedScatterSoleNHR";
        } else {
            HCCL_WARNING(
                "[Algo][SelectCcuScheduleAlgo] layer0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        SelectorStatus ret = SelectMeshAlgoCcuSchedule(topoInfo, opParam, selectAlgName);
        if (ret != SelectorStatus::MATCH) {
            return ret;
        }
    }
    HCCL_INFO("[ScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectMeshAlgoCcuSchedule(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 userRankSize = topoInfo->userRankSize;
    u64 dataSize = opParam.DataDes.count * perDataSize * userRankSize;
    HCCL_INFO("[SelectMeshAlgoCcuSchedule] dataSize[%llu]", dataSize);
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        CHK_PRT_RET(
            IsInputOutputOverlap(opParam) == true,
            HCCL_WARNING("[Algo][ScatterAutoSelector] ccu schedule does not support inplace scatter."),
            SelectorStatus::NOT_MATCH);
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[ScatterAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuSchedScatterSoleMesh";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            selectAlgName = "CcuSchedScatterSoleMesh";
        } else if (topoInfo->level0PcieMix) {
            HCCL_WARNING("[ScatterAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else {
            if (dataSize < OMNI2D_UBX_SC_DATA_SIZE) {
                selectAlgName = "CcuScatterParallelMesh1DNHRUBX";
            } else {
                selectAlgName = "CcuSchedScatterPipeLineMeshNHR";
            }
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (topoInfo->level0PcieMix) { // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
            HCCL_WARNING("pcie mixed topo is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        }
        selectAlgName = "CcuSchedScatterSoleNHR";
    } else {
        HCCL_WARNING(
            "[Algo][ScatterAutoSelector] level0Topo[%d] is not supported yet for ccu_schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)opParam;
    (void)configAlgMap;
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    SelectorStatus ret = (topoInfo->topoLevelNums > 1) ? SelectMultiLevelAicpuAlgo(topoInfo, selectAlgName) :
                                                         SelectSingleLevelAicpuAlgo(topoInfo, selectAlgName);
    if (ret == SelectorStatus::MATCH) {
        HCCL_INFO("[ScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    }
    return ret;
}

SelectorStatus ScatterAutoSelector::SelectMultiLevelAicpuAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, std::string& selectAlgName) const
{
    if (topoInfo->topoLevelNums == TOPO_LEVEL_3) {
        bool level0AndLevel1Sym = topoInfo->level0Symmetric && topoInfo->level1Symmetric;
        if (!level0AndLevel1Sym || topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgName = "AicpuScatterSoleNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D && !topoInfo->level2Uboe) {
            selectAlgName = "AicpuScatterSequenceMeshConcurNHRNHR";
        } else {
            selectAlgName = "AicpuScatterSoleNHR";
        }
    } else if (topoInfo->Level1Nhr) {
        selectAlgName = "AicpuScatterSoleNHR";
    } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
        selectAlgName = "AicpuScatterSoleNHR";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "AicpuScatterParallelMeshNHR";
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        selectAlgName = "AicpuScatterSoleNHR";
    } else {
        HCCL_WARNING("[ScatterAutoSelector] topo not match for aicpu algo");
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectSingleLevelAicpuAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, std::string& selectAlgName) const
{
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "AicpuScatterSoleMesh";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            selectAlgName = "AicpuScatterSoleMesh";
        } else if (topoInfo->level0PcieMix) {
            selectAlgName = "InsScatterParallelMesh1DNHRPcie";
        } else {
            selectAlgName = "InsScatterParallelMesh1DNHRUBX";
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (topoInfo->level0PcieMix) {
            selectAlgName = "AicpuScatterSoleNHR";
        } else {
            selectAlgName = "AicpuScatterSoleMesh";
        }
    } else {
        HCCL_WARNING("[ScatterAutoSelector] topo not match for aicpu algo");
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectAivAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->userRankSize > MAX_RANK_SIZE) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[ScatterAutoSelector][%s] rankSize[%u] larger than [%u]", __func__,
            topoInfo->userRankSize, MAX_RANK_SIZE);
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level2Ubg) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[ScatterAutoSelector][%s] aiv is not supported with level2Ubg, reset to default.",
            __func__);
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums >= TOPO_LEVEL_NUM_3) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG,
            "[ScatterAutoSelector][%s] aiv is not supported when topoLevelNums >= 3(levelNum[%u]), reset to default.",
            __func__, topoInfo->topoLevelNums);
        return SelectorStatus::NOT_MATCH;
    }

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_PRT_RET(
        HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize) != HCCL_SUCCESS,
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[ScatterAutoSelector] HcclGetHcclBuffer failed."),
        SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 totalSize = opParam.DataDes.count * perDataSize * topoInfo->userRankSize;
    if (opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY
        && totalSize >= AIV_MAX_PER_RANK_DATA_SIZE * topoInfo->userRankSize) {
        HCCL_DEBUG(
            "[ScatterAutoSelector][%s] totalSize[%llu] larger than AIV_MAX_PER_RANK_DATA_SIZE[%llu] * rankSize[%u]",
            __func__, totalSize, AIV_MAX_PER_RANK_DATA_SIZE, topoInfo->userRankSize);
        return SelectorStatus::NOT_MATCH;
    }
    if (totalSize > cclBufferSize * AIV_MAX_CCL_LOOP_NUM) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[ScatterAutoSelector][%s] totalSize[%llu] too large for cclBufferSize [%llu]",
            __func__, totalSize, cclBufferSize);
        return SelectorStatus::NOT_MATCH;
    }

    selectAlgName = "AivScatterSoleMesh";

    HCCL_INFO("[ScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    HCCL_INFO(
        "topoInfo->topoLevelNums is %u, topoInfo->level0Topo is %u", topoInfo->topoLevelNums, topoInfo->level0Topo);
    (void)configAlgMap;
    (void)opParam;
    if (topoInfo->topoLevelNums > 1) {
        selectAlgName = "DpuScatterSequenceMeshNHR";
        HCCL_INFO("Using algo DpuScatterSequenceMeshNHR");
        return SelectorStatus::MATCH;
    }

    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_SCATTER, 18, ScatterAutoSelector);
} // namespace ops_hccl
