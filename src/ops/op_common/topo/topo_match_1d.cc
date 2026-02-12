/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_1d.h"

namespace ops_hccl {
TopoMatch1D::TopoMatch1D()
{
}

TopoMatch1D::~TopoMatch1D()
{
}

HcclResult TopoMatch1D::MatchTopo(HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
#ifndef AICPU_COMPILE
    CHK_PRT_RET(topoInfo->topoLevelNums == 0 || topoInfo->topoLevelNums > 2,
        HCCL_ERROR("[CalcTopoLevelNums] topoLevelNum[%u] is invalid.",
            topoInfo->topoLevelNums),
        HCCL_E_INTERNAL);

    if (topoInfo->deviceType != DevType::DEV_TYPE_910_95) {
        HCCL_ERROR("[CollAlgFactory] [TopoMatchMesh] Rank [%d], deviceType not supported yet.", myRank_);
    }

    CHK_PRT_RET((topoInfo->userRankSize == 0),
                HCCL_ERROR("[CollAlgFactory] [TopoMatchMesh1D] Rank [%d], rankSize is 0.", myRank_),
                HcclResult::HCCL_E_PARA);

    for (uint32_t netLayerIdx = 0; netLayerIdx < topoInfo->topoLevelNums; netLayerIdx++) {
        CommTopo topoType;
        HcclRankGraphGetTopoTypeByLayer(comm, netLayerIdx, &topoType);
        CHK_PRT_RET((topoType != CommTopo::COMM_TOPO_CUSTOM && topoType != CommTopo::COMM_TOPO_CLOS),
                HCCL_ERROR("[CollAlgFactory] [TopoMatchMesh1D] netLayer [%d], topoType not COMM_TOPO_CUSTOM or COMM_TOPO_CLOS.", netLayerIdx),
                HcclResult::HCCL_E_PARA);
    }

    for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
        rankIds_.push_back(rankId);
    }
    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].resize(1);
    algHierarchyInfo.infos[0][0] = rankIds_;
#endif
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl