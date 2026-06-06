/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_3_level.h"
#include "op_common.h"

namespace ops_hccl {
TopoMatch3Level::TopoMatch3Level()
    : TopoMatchBase()
{
}

TopoMatch3Level::~TopoMatch3Level()
{
}

HcclResult TopoMatch3Level::TopoForLayer0(
    const HcclComm comm, uint32_t& layer0Size, const uint32_t myRank,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo) const
{
#ifndef AICPU_COMPILE
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &topoInstNum));

    if (topoInstNum == NET_INST_NUM_1) {
        // mesh1d
        HCCL_INFO("[CollAlgFactory] [TopoMatch3Level] layer0 topoInstNum [%d], Mesh 1D (Symmetric).", topoInstNum);
        uint32_t* ranks;
        uint32_t rankNum = 0;
        CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 0, topoInsts[0], &ranks, &rankNum));
        HCCL_DEBUG("[CollAlgFactory] [TopoMatch3Level] Rank [%d], all [%u] ranks in this server: [%s]",
            myRank, rankNum, PrintCArray<uint32_t>(ranks, rankNum).c_str());
        // 仅支持对称逻辑
        std::vector<uint32_t> rankVecLayer0(ranks, ranks + rankNum);
        algHierarchyInfo.infos[0].push_back({rankVecLayer0});
        layer0Size = rankVecLayer0.size();
    } else {
        // 单卡情况
        algHierarchyInfo.infos[0].push_back({{myRank}});
        layer0Size = 1;
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatch3Level::TopoForLayerGeneric(
    const HcclComm comm, uint32_t netLayer, uint32_t baseModSize, const uint32_t myRank,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, uint32_t targetLayerIdx) const
{
    HCCL_DEBUG("[TopoMatch3Level::TopoForLayerGeneric] netLayer[%d], baseModSize[%d], targetLayerIdx[%d]",
               netLayer, baseModSize, targetLayerIdx);
#ifndef AICPU_COMPILE
    // 获取当前网络层的所有Rank
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInstNum));
    CHK_PRT_RET(
        (topoInstNum != NET_INST_NUM_1),
        HCCL_ERROR("[TopoMatch3Level::TopoForLayerGeneric] layer[%d] topoInstNum [%d], Invalid topo (expect 1D).",
                   targetLayerIdx, topoInstNum),
        HcclResult::HCCL_E_PARA);

    uint32_t* ranks;
    uint32_t rankNum;
    CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInsts[0], &ranks, &rankNum));
    HCCL_DEBUG("[TopoMatch3Level::TopoForLayerGeneric] Rank [%d], all [%u] ranks in layer[%d]",
               myRank, rankNum, targetLayerIdx);

    std::vector<uint32_t> rankVecLayerWithSameIdx;
    for (uint32_t i = 0; i < rankNum; i++) {
        uint32_t rankId = ranks[i];
        if (myRank == rankId) {
            rankVecLayerWithSameIdx.push_back(rankId);
            continue;
        }
        if (rankId % baseModSize != myRank % baseModSize) {
            continue;
        }
        CommLink *links;
        uint32_t linkNum = 0;
        HcclRankGraphGetLinks(comm, netLayer, myRank, rankId, &links, &linkNum);
        if (linkNum == 0) {
            continue;
        }
        rankVecLayerWithSameIdx.push_back(rankId);
    }
    algHierarchyInfo.infos[targetLayerIdx].push_back({rankVecLayerWithSameIdx});
    HCCL_DEBUG("[TopoMatch3Level::TopoForLayerGeneric] Rank [%d], layer[%d] group: [%s]",
               myRank, targetLayerIdx, PrintCArray<uint32_t>(rankVecLayerWithSameIdx.data(),
               static_cast<u32>(rankVecLayerWithSameIdx.size())).c_str());
#endif
    return HcclResult::HCCL_SUCCESS;
}

bool TopoMatch3Level::CheckVecElementAllSame(const uint32_t* instSizeList, uint32_t listSize) const
{
#ifndef AICPU_COMPILE
    uint32_t firstSize = instSizeList[0];
    for (uint32_t i = 1; i < listSize; i++) {
        if (firstSize != instSizeList[i]) {
            return false;
        }
    }
#endif
    return true;
}

HcclResult TopoMatch3Level::MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
#ifndef AICPU_COMPILE
    // 支持3层
    constexpr uint32_t EXPECTED_TOPO_LEVEL_NUM_3 = 3;
    CHK_PRT_RET(topoInfo->topoLevelNums == 0 || topoInfo->topoLevelNums > EXPECTED_TOPO_LEVEL_NUM_3,
        HCCL_ERROR("[CalcTopoLevelNums] topoLevelNum[%u] is invalid (expect 1 to 3).",
            topoInfo->topoLevelNums),
        HCCL_E_INTERNAL);

    uint32_t myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));
    #ifdef MACRO_DEV_TYPE_NEW
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_950,
    #else
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_910_95,
    #endif
        HCCL_ERROR("[CollAlgFactory] [TopoMatch3Level] Rank [%d], deviceType not supported yet.",
            myRank),
        HcclResult::HCCL_E_PARA);

    // 获取物理网络层数
    uint32_t *netLayers;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &layerNum));
    HCCL_DEBUG("[CollAlgFactory] [TopoMatch3Level] Rank [%d], netLayers[%u][%s]",
                myRank, layerNum, PrintCArray<uint32_t>(netLayers, layerNum).c_str());

    // 校验是否对称
    uint32_t *instSizeList;
    uint32_t listSize = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &listSize));
    HCCL_INFO("[CollAlgFactory] [TopoMatch3Level] Rank [%d], [%u] servers ,ranksize on each server :[%s]",
        myRank, listSize, PrintCArray<uint32_t>(instSizeList, listSize).c_str());
    
    bool isSymmetric = CheckVecElementAllSame(instSizeList, listSize);
    CHK_PRT_RET(!isSymmetric,
        HCCL_ERROR("[TopoMatch3Level][MatchTopo] This version only supports Symmetric topology."),
        HcclResult::HCCL_E_NOT_SUPPORT);

    algHierarchyInfo.infos.resize(EXPECTED_TOPO_LEVEL_NUM_3);
    uint32_t layer0Size = 0;
    CHK_RET(TopoForLayer0(comm, layer0Size, myRank, algHierarchyInfo));
    uint32_t layer1Size = listSize;
    uint32_t baseModSizeL1 = layer0Size;
    
    if (layerNum >= 2) {
        uint32_t netLayerL1 = 1;
        CHK_RET(TopoForLayerGeneric(comm, netLayerL1, baseModSizeL1, myRank, algHierarchyInfo, netLayerL1));
    }
    if (layerNum >= 3) {
        uint32_t netLayerL2 = 2;
        // 应该除以超节点数量
        uint32_t superPodNum = topoInfo->netLayerDetails.localNetInsSizeOfLayer[2] / topoInfo->netLayerDetails.localNetInsSizeOfLayer[1];
        uint32_t baseModSizeL2 = layer0Size * layer1Size / superPodNum;
        CHK_RET(TopoForLayerGeneric(comm, netLayerL2, baseModSizeL2, myRank, algHierarchyInfo, netLayerL2));
    }

#endif
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl