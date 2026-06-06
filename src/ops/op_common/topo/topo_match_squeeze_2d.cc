/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_squeeze_2d.h"
#include "op_common.h"

namespace ops_hccl {
TopoMatchSqueeze2D::TopoMatchSqueeze2D()
    : TopoMatchBase()
{
}

TopoMatchSqueeze2D::~TopoMatchSqueeze2D()
{
}

HcclResult TopoMatchSqueeze2D::TopoForLayer0(
    const HcclComm comm, uint32_t& combinedSize, const uint32_t myRank,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo) const
{
#ifndef AICPU_COMPILE
    // only for mesh1D
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &topoInstNum));
    CHK_PRT_RET(topoInstNum != NET_INST_NUM_1,
        HCCL_ERROR("[TopoMatchSqueeze2D][TopoForLayer0] layer0 topoInstNum [%u] != 1, only Mesh1D supported.",
            topoInstNum),
        HcclResult::HCCL_E_NOT_SUPPORT);

    uint32_t *topoInstsLayer1;
    uint32_t topoInstNumLayer1 = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 1, &topoInstsLayer1, &topoInstNumLayer1));
    CHK_PRT_RET(topoInstNumLayer1 != NET_INST_NUM_1,
        HCCL_ERROR("[TopoMatchSqueeze2D][TopoForLayer0] layer1 topoInstNum [%u] != 1.",
            topoInstNumLayer1),
        HcclResult::HCCL_E_NOT_SUPPORT);

    uint32_t* ranksLayer1;
    uint32_t rankNumLayer1 = 0;
    CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 1, topoInstsLayer1[0], &ranksLayer1, &rankNumLayer1));
    HCCL_DEBUG("[TopoMatchSqueeze2D] Rank [%u], all [%u] ranks in layer1: [%s]",
        myRank, rankNumLayer1, PrintCArray<uint32_t>(ranksLayer1, rankNumLayer1).c_str());

    std::vector<uint32_t> ranksInfos0;
    for (uint32_t i = 0; i < rankNumLayer1; i++) {
        ranksInfos0.push_back(ranksLayer1[i]);
    }
    combinedSize = ranksInfos0.size();
    algHierarchyInfo.infos[0].push_back({ranksInfos0});
    HCCL_DEBUG("[TopoMatchSqueeze2D][TopoForLayer0] Rank [%u], layer0 group: [%s]",
        myRank, PrintCArray<uint32_t>(ranksInfos0.data(),
        static_cast<u32>(ranksInfos0.size())).c_str());
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchSqueeze2D::TopoForLayer1(
    const HcclComm comm, uint32_t baseModSize, const uint32_t myRank,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo) const
{
#ifndef AICPU_COMPILE
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 2, &topoInsts, &topoInstNum));

    CHK_PRT_RET(topoInstNum != NET_INST_NUM_1,
        HCCL_ERROR("[TopoMatchSqueeze2D][TopoForLayer1] layer2 topoInstNum [%u] != 1.", topoInstNum),
        HcclResult::HCCL_E_NOT_SUPPORT);

    uint32_t* ranks;
    uint32_t rankNum;
    CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 2, topoInsts[0], &ranks, &rankNum));
    HCCL_DEBUG("[TopoMatchSqueeze2D][TopoForLayer1] Rank [%u], all [%u] ranks in layer2: [%s]",
        myRank, rankNum, PrintCArray<uint32_t>(ranks, rankNum).c_str());

    std::vector<uint32_t> rankVecLayer1WithSameIdx;
    for (uint32_t i = 0; i < rankNum; i++) {
        uint32_t rankId = ranks[i];
        if (myRank == rankId) {
            rankVecLayer1WithSameIdx.push_back(rankId);
            continue;
        }
        if (rankId % baseModSize != myRank % baseModSize) {
            continue;
        }
        CommLink *links;
        uint32_t linkNum = 0;
        HcclRankGraphGetLinks(comm, 2, myRank, rankId, &links, &linkNum);
        if (linkNum == 0) {
            continue;
        }
        rankVecLayer1WithSameIdx.push_back(rankId);
    }

    algHierarchyInfo.infos[1].push_back({rankVecLayer1WithSameIdx});
    HCCL_DEBUG("[TopoMatchSqueeze2D][TopoForLayer1] Rank [%u], layer1 group: [%s]",
        myRank, PrintCArray<uint32_t>(rankVecLayer1WithSameIdx.data(),
        static_cast<u32>(rankVecLayer1WithSameIdx.size())).c_str());
#endif
    return HcclResult::HCCL_SUCCESS;
}

bool TopoMatchSqueeze2D::CheckVecElementAllSame(const uint32_t* instSizeList, uint32_t listSize) const
{
#ifndef AICPU_COMPILE
    if (listSize == 0) return true;
    uint32_t firstSize = instSizeList[0];
    for (uint32_t i = 1; i < listSize; i++) {
        if (firstSize != instSizeList[i]) {
            return false;
        }
    }
#endif
    return true;
}

HcclResult TopoMatchSqueeze2D::MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
#ifndef AICPU_COMPILE
    constexpr uint32_t EXPECTED_TOPO_LEVEL_NUM_3 = 3;
    CHK_PRT_RET(topoInfo->topoLevelNums != EXPECTED_TOPO_LEVEL_NUM_3,
        HCCL_ERROR("[TopoMatchSqueeze2D][MatchTopo] topoLevelNums [%u] != 3, only 3-level topo supported.",
            topoInfo->topoLevelNums),
        HcclResult::HCCL_E_INTERNAL);

    uint32_t myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));

    #ifdef MACRO_DEV_TYPE_NEW
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_950,
    #else
    CHK_PRT_RET(topoInfo->deviceType != DevType::DEV_TYPE_910_95,
    #endif
        HCCL_ERROR("[TopoMatchSqueeze2D][MatchTopo] Rank [%u], deviceType not supported yet.", myRank),
        HcclResult::HCCL_E_PARA);

    uint32_t *netLayers;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &layerNum));
    HCCL_DEBUG("[TopoMatchSqueeze2D] Rank [%u], netLayers [%u][%s]",
        myRank, layerNum, PrintCArray<uint32_t>(netLayers, layerNum).c_str());

    CHK_PRT_RET(layerNum < EXPECTED_TOPO_LEVEL_NUM_3,
        HCCL_ERROR("[TopoMatchSqueeze2D][MatchTopo] layerNum [%u] < 3, insufficient physical layers.", layerNum),
        HcclResult::HCCL_E_INTERNAL);

    uint32_t *instSizeList;
    uint32_t listSize = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &listSize));
    HCCL_INFO("[TopoMatchSqueeze2D] Rank [%u], [%u] pods, ranksize on each pod: [%s]",
        myRank, listSize, PrintCArray<uint32_t>(instSizeList, listSize).c_str());

    bool isSymmetric = CheckVecElementAllSame(instSizeList, listSize);
    CHK_PRT_RET(!isSymmetric,
        HCCL_ERROR("[TopoMatchSqueeze2D][MatchTopo] Only symmetric topology supported."),
        HcclResult::HCCL_E_NOT_SUPPORT);

    algHierarchyInfo.infos.resize(COMM_LAYER_SIZE_2);

    uint32_t combinedSize = 0;
    CHK_RET(TopoForLayer0(comm, combinedSize, myRank, algHierarchyInfo));
    CHK_RET(TopoForLayer1(comm, combinedSize, myRank, algHierarchyInfo));
#endif
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl