/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo.h"
#include "topo_experimental.h"
#include "channel.h"

namespace ops_hccl_experimental {
using ops_hccl::COMM_LEVEL0;

HcclResult
CalcGeneralTopoInfoInterServer(const HcclComm comm, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo)
{
    (void)comm;
    algHierarchyInfo.levels = 1;
    algHierarchyInfo.infos[COMM_LEVEL0].localRank = topoInfo->userRank;
    algHierarchyInfo.infos[COMM_LEVEL0].localRankSize = topoInfo->userRankSize;
    HCCL_INFO(
        "[CalcGeneralTopoInfoInterServer] userRank[%u] serverIdx[%u] superPodIdx[%u] l0Rank[%u]"
        "deviceNumPerModule[%u] serverNumPerSuperPod[%u] superPodNum[%u]"
        "l0RankSize[%u]",
        topoInfo->userRank, topoInfo->serverIdx, topoInfo->superPodIdx, algHierarchyInfo.infos[COMM_LEVEL0].localRank,
        topoInfo->deviceNumPerModule, topoInfo->serverNumPerSuperPod, topoInfo->superPodNum,
        algHierarchyInfo.infos[COMM_LEVEL0].localRankSize);

    return HCCL_SUCCESS;
}

} // namespace ops_hccl_experimental
