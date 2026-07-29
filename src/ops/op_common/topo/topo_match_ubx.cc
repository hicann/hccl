/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_ubx.h"

namespace ops_hccl {
constexpr u64 NATLAYER_ROCE = 3;
TopoMatchUBX::TopoMatchUBX()
    : TopoMatchBase()
{
}

TopoMatchUBX::~TopoMatchUBX()
{
}

HcclResult TopoMatchUBX::TopoForLayer0(const HcclComm comm, uint32_t &layer0Size, const uint32_t myRank,
                                                  AlgHierarchyInfoForAllLevel& algHierarchyInfo) const
{
#ifndef AICPU_COMPILE
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &topoInstNum));
    HCCL_INFO("[CollAlgFactory] [TopoMatchUBX] layer0 topoInstNum [%d].", topoInstNum);
    if (topoInstNum == NET_INST_NUM_1) { // mesh1d
        HCCL_INFO("[CollAlgFactory] [TopoMatchUBX] layer0 topoInstNum [%d], Mesh 1D.", topoInstNum);
        uint32_t* ranks;
        uint32_t rankNum;
        CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 0, topoInsts[0], &ranks, &rankNum));
        HCCL_DEBUG("[CollAlgFactory] [TopoMatchUBX] Rank [%d], all [%u] ranks in this pod: [%s]",
            myRank, rankNum, PrintCArray<uint32_t>(ranks, rankNum).c_str());
        std::vector<uint32_t> rankVecLayer0(ranks, ranks + rankNum);
        algHierarchyInfo.infos[0].push_back({rankVecLayer0});
        layer0Size = rankVecLayer0.size();
    } else if (topoInstNum == 0) {
        algHierarchyInfo.infos[0].push_back({{myRank}});
        layer0Size = 1;
    } else if (topoInstNum >= NET_INST_NUM_2) { // MESH1DCLOS
        std::vector<u32> mesh1DRanks, closRanks;
        for (uint32_t idx = 0; idx < topoInstNum; idx++) {
            CommTopo topoType;
            CHK_RET(HcclRankGraphGetTopoType(comm, 0, topoInsts[idx], &topoType));
            uint32_t* ranks;
            uint32_t rankNum;
            CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 0, topoInsts[idx], &ranks, &rankNum));
            if (topoType == CommTopo::COMM_TOPO_1DMESH) {
                mesh1DRanks.insert(mesh1DRanks.end(), ranks, ranks + rankNum);
            } else if (topoType == CommTopo::COMM_TOPO_CLOS) {
                closRanks.insert(closRanks.end(), ranks, ranks + rankNum);
            }
            layer0Size = rankNum;
        }
        if (!mesh1DRanks.empty()) {
            algHierarchyInfo.infos[0].push_back(mesh1DRanks);
            layer0Size = mesh1DRanks.size();
        }
        if (!closRanks.empty()) {
            algHierarchyInfo.infos[0].push_back(closRanks);
            if (closRanks.size() > layer0Size) {
                layer0Size = closRanks.size();
            }
        }
        HCCL_INFO("[TopoMatchUBX] layer0Size %u topoInstNum [%d], infos[0].size %u, mesh1DRanks[%u], closRanks[%u]", 
                layer0Size, topoInstNum, algHierarchyInfo.infos[0].size(), mesh1DRanks.size(), closRanks.size());
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchUBX::TopoForLayer3(const HcclComm comm, uint32_t layer0Size, const uint32_t myRank,
                                                  AlgHierarchyInfoForAllLevel& algHierarchyInfo) const
{
    HCCL_DEBUG("[TopoMatchUBX::MeshNHRTopoForLayer3] layer0Size [%d]", layer0Size);
#ifndef AICPU_COMPILE
    // 1. 查出layer 3的所有ranks
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, NATLAYER_ROCE, &topoInsts, &topoInstNum));
    CHK_PRT_RET(
        (topoInstNum != NET_INST_NUM_1),
        HCCL_ERROR("[TopoMatchUBX::MeshNHRTopoForLayer3] layer3 topoInstNum [%d], Invalid topo.", topoInstNum),
        HcclResult::HCCL_E_PARA);
    uint32_t* ranks;
    uint32_t rankNum;
    CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, NATLAYER_ROCE, topoInsts[0], &ranks, &rankNum));
    HCCL_DEBUG("[TopoMatchUBX::MeshNHRTopoForLayer3] Rank [%d], all [%u] ranks in layer3", myRank, rankNum);
    // 2. 取出同序号卡，作为layer3的ranks
    std::vector<uint32_t> rankVecLayer3WithSameIdx;
    for (uint32_t i = 0; i < rankNum; i++) {
        uint32_t rankId = ranks[i];
        if (myRank == rankId) {
            rankVecLayer3WithSameIdx.push_back(rankId);
            continue;
        }
        if (rankId % layer0Size != myRank % layer0Size) {
            continue;
        }
        CommLink *links;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, NATLAYER_ROCE, myRank, rankId, &links, &linkNum));
        if (linkNum == 0) {
            continue;
        }
        rankVecLayer3WithSameIdx.push_back(rankId);
    }
    algHierarchyInfo.infos[1].push_back({rankVecLayer3WithSameIdx});
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchUBX::CheckVecElementAllSame(const uint32_t *instSizeList, uint32_t listSize) const
{
#ifndef AICPU_COMPILE
    uint32_t firstSize = instSizeList[0];
    for (uint32_t i = 1; i < listSize; i++) {
        if (firstSize != instSizeList[i]) {
            HCCL_ERROR("[TopoMatchUBX::CheckVecElementAllSame] instSizeList [%u] [%u] not equal, Invalid topo.",
                      firstSize, instSizeList[i]);
            return HcclResult::HCCL_E_PARA;
        }
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchUBX::MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
#ifndef AICPU_COMPILE
    constexpr uint32_t EXPECTED_TOPO_LEVEL_NUM_2 = 2;
    CHK_PRT_RET(topoInfo->topoLevelNums == 0 || topoInfo->topoLevelNums > EXPECTED_TOPO_LEVEL_NUM_2,
        HCCL_ERROR("[CalcTopoLevelNums] topoLevelNum[%u] is invalid.",
            topoInfo->topoLevelNums),
        HCCL_E_INTERNAL);
    uint32_t myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));
    CHK_PRT_RET(!shouldGoOutPlace(topoInfo->deviceType),
        HCCL_ERROR("[CollAlgFactory] [TopoMatchUBX] Rank [%d], deviceType not supported yet.",
            myRank),
        HcclResult::HCCL_E_PARA);
    // 1.获取并校验通信层数
    uint32_t layerNum = 0;
    uint32_t *netLayers;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &layerNum));

    HCCL_DEBUG("[CollAlgFactory] [TopoMatchUBX] Rank [%d], netLayers[%u][%s]",
                myRank, layerNum, PrintCArray<uint32_t>(netLayers, layerNum).c_str());

    // 2. 获取每个pod上rank数量以及pod数量
    uint32_t listSize = 0;
    uint32_t *instSizeList;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &instSizeList, &listSize));
    HCCL_INFO("[CollAlgFactory] [TopoMatchUBX] Rank [%d], [%u] pods ,ranksize on each pod :[%s]",
        myRank,
        listSize,
        PrintCArray<uint32_t>(instSizeList, listSize).c_str());
    CHK_RET(CheckVecElementAllSame(instSizeList, listSize));
    // 3. 计算layer0的topo
    algHierarchyInfo.infos.resize(COMM_LAYER_SIZE_2);
    uint32_t layer0Size = 0;
    HCCL_INFO("[CollAlgFactory] [TopoMatchUBX] TopoForLayer0.");
    CHK_RET(TopoForLayer0(comm, layer0Size, myRank, algHierarchyInfo));
    // 4. 计算layer3的topo
    if (layerNum >= COMM_LAYER_SIZE_2) {
        CHK_RET(TopoForLayer3(comm, layer0Size, myRank, algHierarchyInfo));
    }
#endif
    return HcclResult::HCCL_SUCCESS;
}
}  // namespace ops_hccl