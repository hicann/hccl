/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_one_level.h"
#include <algorithm>
#include "log.h"

namespace ops_hccl {

TopoMatchOneLevel::TopoMatchOneLevel() {}

TopoMatchOneLevel::~TopoMatchOneLevel() {}

namespace {
    // 从 effIdx 中找 localRanks==userRankSize 的最低有效层；hostdpu 额外要求 locType==HOST
    u32 PickFullLocalRanksLayer(
        const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx, u32 userRankSize,
        bool requireHost)
    {
        for (u32 idx : effIdx) {
            if (physicalLevels[idx].localRanks.size() != userRankSize) {
                continue;
            }
            if (requireHost && physicalLevels[idx].locType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
                continue;
            }
            return idx;
        }
        return INVALID_UINT;
    }
} // namespace

HcclResult TopoMatchOneLevel::MatchTopo(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& profile)
{
    const auto& physicalLevels = topoInfo->physicalLevels;
    if (physicalLevels.empty() || topoInfo->userRankSize == 0) {
        HCCL_ERROR("[TopoMatchOneLevel] Rank [%u], physicalLevels empty or userRankSize 0.", topoInfo->userRank);
        return HcclResult::HCCL_E_INTERNAL;
    }

    std::vector<u32> effIdx = CollectEffectiveIndices(physicalLevels, profile.engine);
    if (effIdx.empty()) {
        HCCL_INFO("[TopoMatchOneLevel] Rank [%u], no valid layer after engine filter.", topoInfo->userRank);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    bool requireHost = (profile.engine == OpExecuteConfig::HOSTCPU);
    u32 picked = PickFullLocalRanksLayer(physicalLevels, effIdx, topoInfo->userRankSize, requireHost);
    if (picked == INVALID_UINT) {
        HCCL_INFO(
            "[TopoMatchOneLevel] Rank [%u], no layer with localRanks == userRankSize[%u] (requireHost[%d]).",
            topoInfo->userRank, topoInfo->userRankSize, static_cast<int32_t>(requireHost));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].resize(1);
    algHierarchyInfo.infos[0][0] = physicalLevels[picked].localRanks;
    algHierarchyInfo.physicalIdxForAlgoLevels = {{static_cast<PhysicalLevelIndex>(picked)}};
    HCCL_INFO(
        "[TopoMatchOneLevel] Rank [%u], userRankSize [%u], physicalIdxForAlgoLevels: [%s].", topoInfo->userRank,
        topoInfo->userRankSize, FormatPhysicalIdxForAlgoLevels(algHierarchyInfo.physicalIdxForAlgoLevels).c_str());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
