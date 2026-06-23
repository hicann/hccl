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

SelectorStatus ScatterAutoSelector::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    HCCL_WARNING("[Algo][ScatterAutoSelector] not supported yet for ccu_ms mode, reset to default.");
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus ScatterAutoSelector::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)opParam;
    (void)configAlgMap; 
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
                selectAlgName = "CcuScatterNHRMem2Mem1D";
            } else if (topoInfo->is2DieFullMesh) {
                HCCL_WARNING("[ScatterAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuScatterParallelMesh1DNHR";
            }
        } else {
            HCCL_WARNING("[Algo][SelectCcuScheduleAlgo] layer0Shape[%d] is not supported yet for ccu schedule mode.",
                topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
        CHK_PRT_RET(IsInputOutputOverlap(opParam) == true,
            HCCL_WARNING("[Algo][ScatterAutoSelector] ccu schedule does not support inplace allreduce."),
            SelectorStatus::NOT_MATCH);
            if (topoInfo->is2DieFullMesh) {
                HCCL_WARNING("[ScatterAutoSelector] 2DieFullMesh is not supported yet for schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuScatterMesh1D";
            }
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                selectAlgName = "CcuScatterMesh1D";
            } else if (topoInfo->level0PcieMix) {
                HCCL_WARNING("[ScatterAutoSelector] pcie mixed topo is not supported yet for ccu schedule mode.");
                return SelectorStatus::NOT_MATCH;
            } else {
                selectAlgName = "CcuScatterParallelMesh1DNHRUBX";
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            HCCL_WARNING("[Algo][ScatterAutoSelector] level0Topo[%d] is not supported yet for ccu_schedule mode.", topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING("[Algo][ScatterAutoSelector] level0Topo[%d] is not supported yet for ccu_schedule mode.", topoInfo->level0Topo);
            return SelectorStatus::NOT_MATCH;
        }
    }
    HCCL_INFO("[ScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)opParam;
    (void)configAlgMap; 
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->topoLevelNums > 1) {
        if (topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] == 1) {
            selectAlgName = "InsScatterNHR";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsScatterParallelMesh1DNHR";
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            HCCL_WARNING("[ScatterAutoSelector] level0Shape[%d] is not supported yet for levelNum > 1.");
            return SelectorStatus::NOT_MATCH;
        } else {
            HCCL_WARNING("[ScatterAutoSelector] topo not match for aicpu algo");
            return SelectorStatus::NOT_MATCH;
        }
    } else {
        if (topoInfo->level0Topo == Level0Shape::MESH_1D) {
            selectAlgName = "InsScatterMesh1D";
        } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            if (IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)) {
                // MESH_1D 即可链接所有卡， 使用 MESH_1D 算法
                selectAlgName = "InsScatterMesh1D";
            } else if (topoInfo->level0PcieMix) {
                selectAlgName = "InsScatterParallelMesh1DNHRPcie";
            } else {
                selectAlgName = "InsScatterParallelMesh1DNHRUBX";
            }
        } else if (topoInfo->level0Topo == Level0Shape::CLOS) {
            if (topoInfo->level0PcieMix) {
                selectAlgName = "InsScatterNHR";
            } else {
                selectAlgName = "InsScatterMesh1D";
            }
        } 
        else {
            HCCL_WARNING("[ScatterAutoSelector] topo not match for aicpu algo");
            return SelectorStatus::NOT_MATCH;
        }
    }

    HCCL_INFO("[ScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

SelectorStatus ScatterAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
                                                  const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                  std::string &selectAlgName) const
{
    (void)configAlgMap;
    HCCL_DEBUG("[ScatterAutoSelector][%s] start, topoInfo levelNum[%u]", __func__, topoInfo->topoLevelNums);

    if (topoInfo->userRankSize > MAX_RANK_SIZE) {
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_DEBUG, "[ScatterAutoSelector][%s] rankSize[%u] larger than [%u]", __func__, topoInfo->userRankSize, MAX_RANK_SIZE);
        return SelectorStatus::NOT_MATCH;
    }

    void *cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_PRT_RET(HcclGetHcclBuffer(opParam.hcclComm, &cclBufferAddr, &cclBufferSize) != HCCL_SUCCESS,
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_WARNING, "[ScatterAutoSelector] HcclGetHcclBuffer failed."), SelectorStatus::NOT_MATCH);
    u64 perDataSize = DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
    u64 totalSize = opParam.DataDes.count * perDataSize * topoInfo->userRankSize;
    if (totalSize > cclBufferSize * AIV_MAX_CCL_LOOP_NUM) {
        HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_DEBUG, "[ScatterAutoSelector][%s] totalSize[%llu] too large for cclBufferSize [%llu]", __func__, totalSize, cclBufferSize);
        return SelectorStatus::NOT_MATCH;
    }

    selectAlgName = "AivScatterMesh1D";

    HCCL_INFO("[ScatterAutoSelector][%s] Algo match [%s]", __func__, selectAlgName.c_str());
    return SelectorStatus::MATCH;
}

REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_SCATTER, 18, ScatterAutoSelector);
} // namespace ops_hccl
