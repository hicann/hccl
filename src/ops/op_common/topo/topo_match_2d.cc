/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_2d.h"

namespace ops_hccl {
TopoMatch2D::TopoMatch2D()
    : TopoMatchBase()
{
}

TopoMatch2D::~TopoMatch2D()
{
}

HcclResult TopoMatch2D::MatchTopo(const HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
#ifndef AICPU_COMPILE
    CHK_PRT_RET(topoInfo->topoLevelNums == 0 || topoInfo->topoLevelNums > 2,
        HCCL_ERROR("[CalcTopoLevelNums] topoLevelNum[%u] is invalid.",
            topoInfo->topoLevelNums),
        HCCL_E_INTERNAL);
    
    uint32_t myRank;
    CHK_RET(HcclGetRankId(comm, &myRank));
    // 校验DevType
    CHK_PRT_RET((topoInfo->deviceType != DevType::DEV_TYPE_910_95),
                HCCL_ERROR("[CollAlgFactory] [TopoMatch2D] Rank [%d], Invalid DeviceType.", myRank),
                HcclResult::HCCL_E_PARA);
    
    uint32_t *netLayers;
    uint32_t layerNum;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &layerNum));
    CHK_PRT_RET((layerNum != 1),
        HCCL_ERROR("[CollAlgFactory] [TopoMatch2D] Rank [%d], Invalid virtual topo.", myRank),
        HcclResult::HCCL_E_PARA);
    
    uint32_t *topoInsts;
    uint32_t topoInstNum = 0;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, 0, &topoInsts, &topoInstNum));

    std::vector<uint32_t> ranks_x;
    std::vector<uint32_t> ranks_y;

    for (uint32_t idx = 0; idx < topoInstNum; idx++) {
        CommTopo topoType;
        CHK_RET(HcclRankGraphGetTopoType(comm, 0, topoInsts[idx], &topoType));
        
        if (topoType != CommTopo::COMM_TOPO_1DMESH) continue;
        
        uint32_t* ranks;
        uint32_t rankNum;
        CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, 0, topoInsts[idx], &ranks, &rankNum));

        // todo: 接口输出是否按顺序排列，还需要再排序吗？
        std::sort(ranks, ranks + rankNum);
        // 区分x轴y轴
        if (rankNum == 1) {
            HCCL_ERROR("[CollAlgFactory] [TopoMatch2D] topoInsts [%d], ranksNum [%d], Invalid topo.", topoInsts[idx], rankNum);
        }
        if (ranks[1] - ranks[0] == 1) {
            ranks_x.assign(ranks, ranks + rankNum);
        } else {
            ranks_y.assign(ranks, ranks + rankNum);
        }
    }

    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].push_back(ranks_x);
    algHierarchyInfo.infos[0].push_back(ranks_y);
#endif
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl