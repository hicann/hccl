/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_concurrent_v2.h"
#include "log.h"
#include "hccl_common.h"

namespace ops_hccl {

TopoMatchConcurrentV2::TopoMatchConcurrentV2() {}

TopoMatchConcurrentV2::~TopoMatchConcurrentV2() {}

HcclResult TopoMatchConcurrentV2::MatchTopo(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& profile)
{
    u32 myRank = topoInfo->userRank;
    const auto& physicalLevels = topoInfo->physicalLevels;
    if (physicalLevels.empty()) {
        HCCL_ERROR("[TopoMatchConcurrentV2] Rank [%u], physicalLevels is empty.", myRank);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 引擎过滤后收集有效层
    std::vector<u32> effIdx = CollectEffectiveIndices(physicalLevels, profile.engine);
    u32 effNum = effIdx.size();
    CHK_PRT_RET(
        effNum == 0 || effNum > ALGO_LEVEL_NUM_TWO,
        HCCL_INFO("[TopoMatchConcurrentV2] Rank [%u], level num[%u] not support.", myRank, effNum),
        HcclResult::HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(
        (topoInfo->userRankSize == 0), HCCL_ERROR("[TopoMatchConcurrentV2] Rank [%d], rankSize is 0.", myRank),
        HcclResult::HCCL_E_INTERNAL);

    // infos 沿用原 Concurrent：两组同 rank（mesh 组 + clos 组并发），不依赖 physicalLevels 内容
    std::vector<u32> rankIds;
    rankIds.reserve(topoInfo->userRankSize);
    for (u32 rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
        rankIds.push_back(rankId);
    }
    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].resize(CONCURRENT_SUBGROUP_NUM);
    algHierarchyInfo.infos[0][0] = rankIds;
    algHierarchyInfo.infos[0][1] = rankIds;

    // physicalIdx 指向最高有效层
    u32 highestIdx = effIdx.back();
    algHierarchyInfo.physicalIdxForAlgoLevels = {{static_cast<PhysicalLevelIndex>(highestIdx)}};
    HCCL_INFO(
        "[TopoMatchConcurrentV2] Rank [%u], rankSize[%u], physicalIdxForAlgoLevels: [%s].", myRank,
        topoInfo->userRankSize, FormatPhysicalIdxForAlgoLevels(algHierarchyInfo.physicalIdxForAlgoLevels).c_str());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
