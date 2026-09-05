/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_auto_selector.h"
#include "selector_registry.h"
#include "hccl_aiv_utils.h"

namespace ops_hccl {

constexpr u64 BROADCAST_MESH_CCU_MAX_DATA_SIZE = 16 * 1024;
constexpr u64 BROADCAST_NHR_LESS_64P_CCU_MAX_DATA_SIZE = 4 * 1024 * 1024;
constexpr u64 BROADCAST_NHR_CCU_MAX_DATA_SIZE = 1 * 1024 * 1024;
constexpr u64 OMNI2D_UBX_BR_DATA_SIZE = 16 * 1024 * 1024;
constexpr u32 BROADCAST_CCU_MAX_RANK_SIZE = 64;
constexpr u32 BROADCAST_UBX_AIV_MAX_RANK = 8;
constexpr u32 UBX_BC_CONCURR_DATA_SIZE = 2 * 1024 * 1024;

SelectorStatus BroadcastAutoSelector::SelectCcuMsAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[BroadcastAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);
    if (topoInfo->topoLevelNums > 1) {
        HCCL_WARNING("[Algo][BroadcastAutoSelector] levelNum > 1 is not supported yet for ccu_ms mode.");
        return SelectorStatus::NOT_MATCH;
    } else {
        return SelectMeshAlgoCcuMs(topoInfo, opParam, selectAlgName);
    }
}

SelectorStatus BroadcastAutoSelector::SelectMeshAlgoCcuMs(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
{
    (void)opParam;
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[BroadcastAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuMSBroadcastSoleMesh";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            if (topoInfo->level0PcieMix || dataSize < UBX_BC_CONCURR_DATA_SIZE) {
                selectAlgName = "CcuMSBroadcastSoleMesh";
            } else {
                selectAlgName = "CcuMsBroadcastConcurMeshNHR";
            }
        } else { // MS 不支持
            HCCL_WARNING(
                "[Algo][BroadcastAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        HCCL_WARNING(
            "[Algo][BroadcastAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    } else {
        HCCL_WARNING(
            "[Algo][BroadcastAutoSelector] level0Shape[%d] is not supported yet for ccu_ms mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    HCCL_INFO("[BroadcastAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectCcuScheduleAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[BroadcastAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->level2UbRtp) {
        HCCL_INFO(
            "[BroadcastAutoSelector][%s] ccu schedule is not supported with level2UbRtp, reset to default.", __func__);
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums >= TOPO_LEVEL_NUM_3) {
        HCCL_INFO(
            "[BroadcastAutoSelector][%s] ccu schedule is not supported when topoLevelNums >= 3(levelNum[%u]), reset to "
            "default.",
            __func__, topoInfo->topoLevelNums);
        return SelectorStatus::NOT_MATCH;
    }
    u32 ccuSize = BROADCAST_CCU_MAX_RANK_SIZE;
    constexpr u64 CCU_SCHEDULE_2LEVEL_MAX_PER_RANK_DATA_SIZE = 1ULL * 1024 * 1024;
    constexpr u64 CCU_SCHEDULE_2LEVEL_LESS_64P_MAX_SIZE = 64ULL * 1024 * 1024;
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->userRankSize < ccuSize && dataSize > CCU_SCHEDULE_2LEVEL_LESS_64P_MAX_SIZE) {
                HCCL_INFO("[BroadcastAutoSelector] 2 level topo less than 64P, which dataSize exceeds limit, fallback "
                          "to aicpu.");
                return SelectorStatus::NOT_MATCH;
            }
            if (topoInfo->userRankSize == 0
                || (dataSize / topoInfo->userRankSize > CCU_SCHEDULE_2LEVEL_MAX_PER_RANK_DATA_SIZE
                    && topoInfo->userRankSize >= ccuSize)) {
                HCCL_INFO(
                    "[BroadcastAutoSelector] 2 level topo perRankDataSize[%llu] exceeds limit, "
                    "fallback to aicpu.",
                    topoInfo->userRankSize == 0 ? dataSize : dataSize / topoInfo->userRankSize);
                return SelectorStatus::NOT_MATCH;
            }
            if (topoInfo->Level1Nhr) {
                selectAlgName = "CcuSchedBroadcastSoleNHR";
                HCCL_INFO("[BroadcastAutoSelector] Level1Nhr=true, select [%s]", selectAlgName.c_str());
            } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) { // 每框出1卡
                selectAlgName = "CcuSchedBroadcastSoleNHR";
            } else if (topoInfo->is2DieFullMesh) {
                HCCL_WARNING("[BroadcastAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
                u64 dataSize = opParam.DataDes.count * perDataSize;
                u64 perRankSize = (topoInfo->userRankSize > 0) ? (dataSize / topoInfo->userRankSize) : dataSize;
                if (perRankSize <= BROADCAST_MESH_CCU_MAX_DATA_SIZE
                    && topoInfo->userRankSize <= BROADCAST_CCU_MAX_RANK_SIZE) {
                    selectAlgName = "CcuSchedBroadcastSoleMesh";
                } else if (
                    (perRankSize <= BROADCAST_NHR_LESS_64P_CCU_MAX_DATA_SIZE && topoInfo->userRankSize < ccuSize)
                    || (perRankSize <= BROADCAST_NHR_CCU_MAX_DATA_SIZE && topoInfo->userRankSize >= ccuSize)) {
                    selectAlgName = "CcuSchedBroadcastSoleNHR";
                } else {
                    selectAlgName = "CcuSchedBroadcastParallelMeshNHR";
                }
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS && !topoInfo->level0PcieMix) {
            selectAlgName = "CcuSchedBroadcastSoleNHR";
        } else {
            HCCL_WARNING(
                "[Algo][BroadcastAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        SelectorStatus ret = SelectMeshAlgoCcuSchedule(topoInfo, opParam, selectAlgName);
        if (ret != SelectorStatus::MATCH) {
            return ret;
        }
    }
    HCCL_INFO("[BroadcastAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectMeshAlgoCcuSchedule(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    HCCL_DEBUG("[SelectMeshAlgoCcuSchedule] dataSize[%llu]", dataSize);
    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        if (topoInfo->is2DieFullMesh) {
            HCCL_WARNING("[BroadcastAutoSelector] 2DieFullMesh is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else {
            selectAlgName = "CcuSchedBroadcastSoleMesh";
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            if (topoInfo->level0PcieMix || dataSize < UBX_BC_CONCURR_DATA_SIZE) {
                selectAlgName = "CcuSchedBroadcastSoleMesh";
            } else {
                selectAlgName = "CcuSchedBroadcastConcurMeshNHR";
            }
        } else if (topoInfo->level0PcieMix) {
            HCCL_WARNING("[BroadcastAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        } else {
            if (dataSize < OMNI2D_UBX_BR_DATA_SIZE) {
                selectAlgName = "CcuSchedBroadcastParallelMeshNHRMultiJetty";
            } else {
                selectAlgName = "CcuSchedBroadcastPipeLineMeshNHR";
            }
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        if (topoInfo->level0PcieMix) { // PCIE-SW定制机型，Mesh无法链接全卡时，需要跨pcie链路，不支持ccu模式
            HCCL_WARNING("[BroadcastAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
            return SelectorStatus::NOT_MATCH;
        }
        selectAlgName = "CcuSchedBroadcastSoleNHR";
    } else {
        HCCL_WARNING(
            "[Algo][BroadcastAutoSelector] level0Shape[%d] is not supported yet for ccu schedule mode.",
            topoInfo->level0Topo);
        return SelectorStatus::NOT_MATCH;
    }
    return SelectorStatus::MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[BroadcastAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3) {
            bool level0AndLevel1Symetric = topoInfo->level0Symmetric && topoInfo->level1Symmetric;
            if (!level0AndLevel1Symetric || topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                selectAlgName = "AicpuBroadcastSoleNHR";
            } else if (topoInfo->level0Topo == Level0Shape::MESH_1D && !topoInfo->topLevelUboe) {
                selectAlgName = "AicpuBroadcastSequenceMeshConcurNHRNHR";
            } else {
                selectAlgName = "AicpuBroadcastParallelNHRNHR";
            }
        } else if (topoInfo->Level1Nhr) {
            selectAlgName = "AicpuBroadcastSoleNHR";
        } else if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgName = "AicpuBroadcastSoleNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "AicpuBroadcastParallelMeshNHR";
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            selectAlgName = "AicpuBroadcastSoleNHRMultiLink";
        } else {
            HCCL_WARNING("[BroadcastAutoSelector] topo not match");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        return SelectMeshAlgoAicpu(topoInfo, opParam, selectAlgName);
    }

    HCCL_INFO("[BroadcastAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectMeshAlgoAicpu(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam, std::string& selectAlgName) const
{
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    HCCL_DEBUG("[SelectMeshAlgoCcuSchedule] dataSize[%llu]", dataSize);

    if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        selectAlgName = "AicpuBroadcastSoleMeshTwoShot";
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
            if (topoInfo->level0PcieMix || dataSize < UBX_BC_CONCURR_DATA_SIZE) {
                selectAlgName = "AicpuBroadcastSoleMeshTwoShot";
            } else {
                selectAlgName = "AicpuBroadcastConcurMeshNHR";
            }
        } else if (topoInfo->level0PcieMix) {
            selectAlgName = "AicpuBroadcastParallelMeshNHR";
        } else {
            if (IsLargeData(dataSize)) {
                selectAlgName = "AicpuBroadcastParallelMeshNHRMultiJetty";
            } else {
                selectAlgName = "AicpuBroadcastSoleNHR";
            }
        }
    } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
        selectAlgName = "AicpuBroadcastSoleNHRMultiLink";
    } else {
        HCCL_WARNING("[BroadcastAutoSelector] topo not match");
        return SelectorStatus::NOT_MATCH;
    }

    HCCL_INFO("[BroadcastAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectAivAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)configAlgMap;

    if (topoInfo->userRankSize > MAX_RANK_SIZE) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[BroadcastAutoSelector][%s] rankSize[%u] larger than [%u]", __func__,
            topoInfo->userRankSize, MAX_RANK_SIZE);
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->level2UbRtp) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG, "[BroadcastAutoSelector][%s] aiv is not supported with level2UbRtp, reset to default.",
            __func__);
        return SelectorStatus::NOT_MATCH;
    }

    if (topoInfo->topoLevelNums >= TOPO_LEVEL_NUM_3) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG,
            "[BroadcastAutoSelector][%s] aiv is not supported when topoLevelNums >= 3(levelNum[%u]), reset to default.",
            __func__, topoInfo->topoLevelNums);
        return SelectorStatus::NOT_MATCH;
    }

    void* cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_PRT_RET(
        HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize) != HCCL_SUCCESS,
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[BroadcastAutoSelector] HcclGetHcclBuffer failed."),
        SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 dataSize = opParam.DataDes.count * perDataSize;
    bool isAivBigdata = opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY
                        && dataSize >= AIV_MAX_PER_RANK_DATA_SIZE * topoInfo->userRankSize;
    bool isUBX = topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix;
    if ((isAivBigdata && !isUBX) || (isUBX && isAivBigdata && topoInfo->userRankSize > BROADCAST_UBX_AIV_MAX_RANK)) {
        HCCL_DEBUG(
            "[BroadcastAutoSelector][%s] dataSize[%llu] larger than AIV_MAX_PER_RANK_DATA_SIZE[%llu] * rankSize[%u]",
            __func__, dataSize, AIV_MAX_PER_RANK_DATA_SIZE, topoInfo->userRankSize);
        return SelectorStatus::NOT_MATCH;
    }
    if (dataSize > cclBufferSize * AIV_MAX_CCL_LOOP_NUM) {
        HCCL_AIV_NOT_MATCH_LOG(
            opParam, HCCL_DEBUG,
            "[BroadcastAutoSelector][%s] dataSize[%llu] too large for cclBufferSize[%llu], maxSupportSize[%llu]",
            __func__, dataSize, cclBufferSize, cclBufferSize * AIV_MAX_CCL_LOOP_NUM);
        return SelectorStatus::NOT_MATCH;
    }
    selectAlgName = "AivBroadcastSoleMesh";
    HCCL_INFO("[BroadcastAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus BroadcastAutoSelector::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    std::vector<HcclAlgoType> algos
        = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    auto it = configAlgMap.find(opParam.opType);
    if ((it != configAlgMap.end()) && (it->second.size() > 1)) {
        algos = it->second;
    }

    HCCL_INFO(
        "hccl algo op config: config opType:%d, level0:%u, level1:%u, level2:%u, level3:%u", opParam.opType, algos[0],
        algos[1], algos[2], algos[3]);
    if (topoInfo->topoLevelNums > 1) {
        if ((topoInfo->deviceNumPerModule == 1) || (topoInfo->level0Topo == Level0Shape::MESH_1D)) {
            selectAlgName = "DpuBroadcastSequenceMeshNHR";
            return SelectorStatus::MATCH;
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (!topoInfo->level0PcieMix) {
                if (!(IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)
                      || topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1)) {
                    selectAlgName = "DpuBroadcastOmniPipeMeshNHR";
                    return SelectorStatus::MATCH;
                }
            }
            selectAlgName = "DpuBroadcastSequenceMeshNHR";
            return SelectorStatus::MATCH;
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            // seq算法兼容level0为clos的场景
            selectAlgName = "DpuBroadcastSequenceMeshNHR";
            HCCL_DEBUG(
                "[BroadcastAutoSelector][%s] Level0Shape is CLOS, use algo [%s]", __func__, selectAlgName.c_str());
            return SelectorStatus::MATCH;
        }
    }

    return SelectorStatus::NOT_MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_BROADCAST, 18, BroadcastAutoSelector);
} // namespace ops_hccl
