/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_nlevel.h"

namespace ops_hccl {

TopoMatchNLevel::TopoMatchNLevel() : TopoMatchBase() {}

TopoMatchNLevel::~TopoMatchNLevel() {}

HcclResult TopoMatchNLevel::MatchTopo(
    const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
#ifndef AICPU_COMPILE
    CHK_PRT_RET(
        topoInfo->topoLevelNums == 0 || topoInfo->topoLevelNums > MAX_TOPO_LEVEL_NUM,
        HCCL_ERROR("[TopoMatchNLevel] topoLevelNums[%u] is invalid.", topoInfo->topoLevelNums), HCCL_E_INTERNAL);

    uint32_t myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));

    uint32_t* netLayers;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &layerNum));
    HCCL_DEBUG(
        "[TopoMatchNLevel] Rank [%d], layerNum[%u], topoLevelNums[%u]", myRank, layerNum, topoInfo->topoLevelNums);

    CHK_PRT_RET(
        layerNum < topoInfo->topoLevelNums,
        HCCL_ERROR(
            "[TopoMatchNLevel] layerNum[%u] < topoLevelNums[%u], invalid topo.", layerNum, topoInfo->topoLevelNums),
        HCCL_E_INTERNAL);

    uint32_t* instSizeList;
    uint32_t listSize = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &listSize));
    bool isSymmetric = CheckVecElementAllSame(instSizeList, listSize);
    CHK_PRT_RET(
        !isSymmetric, HCCL_ERROR("[TopoMatchNLevel] Asymmetric topology is not supported yet."), HCCL_E_NOT_SUPPORT);

    algHierarchyInfo.infos.resize(topoInfo->topoLevelNums);

    uint32_t layer0Size = 0;
    CHK_RET(TopoForLayer0(comm, layer0Size, myRank, algHierarchyInfo));
    HCCL_INFO("[TopoMatchNLevel] Rank [%d], layer0Size[%u]", myRank, layer0Size);

    uint32_t baseModSize = layer0Size;
    for (uint32_t level = 1; level < topoInfo->topoLevelNums && level < layerNum; level++) {
        uint32_t netLayer = netLayers[level];
        CHK_RET(TopoForLayerGeneric(comm, netLayer, baseModSize, myRank, algHierarchyInfo, level));

        uint32_t curLevelRankSize = static_cast<uint32_t>(algHierarchyInfo.infos[level][0].size());
        HCCL_INFO(
            "[TopoMatchNLevel] Rank [%d], level[%u], rankSize[%u], baseModSize[%u] -> [%u]", myRank, level,
            curLevelRankSize, baseModSize, baseModSize * curLevelRankSize);
        baseModSize *= curLevelRankSize;
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchNLevel::TopoForLayer0(
    const HcclComm comm, uint32_t& layer0Size, const uint32_t myRank,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo) const
{
#ifndef AICPU_COMPILE
    uint32_t* topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &topoInstNum));

    CHK_PRT_RET(
        topoInstNum != NET_INST_NUM_1,
        HCCL_ERROR("[TopoMatchNLevel] layer0 topoInstNum[%u], only support single instance.", topoInstNum),
        HCCL_E_INTERNAL);

    uint32_t* ranks;
    uint32_t rankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 0, topoInsts[0], &ranks, &rankNum));
    HCCL_DEBUG(
        "[TopoMatchNLevel] Rank [%d], layer0 has [%u] ranks: [%s]", myRank, rankNum,
        PrintCArray<uint32_t>(ranks, rankNum).c_str());

    std::vector<uint32_t> rankVecLayer0(ranks, ranks + rankNum);
    algHierarchyInfo.infos[0].push_back(rankVecLayer0);
    layer0Size = rankVecLayer0.size();
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchNLevel::TopoForLayerGeneric(
    const HcclComm comm, uint32_t netLayer, uint32_t baseModSize, const uint32_t myRank,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo, uint32_t targetLayerIdx) const
{
#ifndef AICPU_COMPILE
    uint32_t* topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInstNum));

    CHK_PRT_RET(
        topoInstNum != NET_INST_NUM_1,
        HCCL_ERROR(
            "[TopoMatchNLevel] layer[%u] netLayer[%u] topoInstNum[%u], only support single instance.", targetLayerIdx,
            netLayer, topoInstNum),
        HCCL_E_INTERNAL);

    uint32_t* ranks;
    uint32_t rankNum;
    CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInsts[0], &ranks, &rankNum));
    HCCL_DEBUG(
        "[TopoMatchNLevel] Rank [%d], layer[%u] netLayer[%u], all [%u] ranks, baseModSize[%u]", myRank, targetLayerIdx,
        netLayer, rankNum, baseModSize);

    std::vector<uint32_t> rankVecSameIdx;
    for (uint32_t i = 0; i < rankNum; i++) {
        uint32_t rankId = ranks[i];
        if (myRank == rankId) {
            rankVecSameIdx.push_back(rankId);
            continue;
        }
        if (rankId % baseModSize != myRank % baseModSize) {
            continue;
        }
        CommLink* links;
        uint32_t linkNum = 0;
        HcclRankGraphGetLinks(comm, netLayer, myRank, rankId, &links, &linkNum);
        if (linkNum == 0) {
            continue;
        }
        rankVecSameIdx.push_back(rankId);
    }
    HCCL_INFO(
        "[TopoMatchNLevel] Rank [%d], layer[%u] same-index group [%u] ranks: [%s]", myRank, targetLayerIdx,
        static_cast<uint32_t>(rankVecSameIdx.size()),
        PrintCArray<uint32_t>(rankVecSameIdx.data(), static_cast<u32>(rankVecSameIdx.size())).c_str());

    algHierarchyInfo.infos[targetLayerIdx].push_back(rankVecSameIdx);
#endif
    return HcclResult::HCCL_SUCCESS;
}

bool TopoMatchNLevel::CheckVecElementAllSame(const uint32_t* instSizeList, uint32_t listSize) const
{
#ifndef AICPU_COMPILE
    if (listSize == 0) {
        return false;
    }
    uint32_t firstSize = instSizeList[0];
    for (uint32_t i = 1; i < listSize; i++) {
        if (firstSize != instSizeList[i]) {
            return false;
        }
    }
#endif
    return true;
}

} // namespace ops_hccl
