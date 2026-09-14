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

// ConcurrentV2 专用：Mesh 类算法取最低覆盖全 rank 的 Mesh 物理层 + 其上层超集层；非 Mesh 或无上层则 not support
static HcclResult ResolveConcurrentPhysicalIdx(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx, const AlgAttrs& algAttrs,
    u32 userRankSize, u32 myRank, std::vector<std::vector<PhysicalLevelIndex>>& physicalIdxForAlgoLevels)
{
    if (algAttrs.algoTypes.empty()) {
        HCCL_INFO("[TopoMatchConcurrentV2] Rank [%u], algAttrs.algoTypes is empty.", myRank);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    if (!IsMeshAlgo(algAttrs.algoTypes[0])) {
        HCCL_INFO(
            "[TopoMatchConcurrentV2] Rank [%u], algAttrs.algoTypes[0] is [%u], not mesh.", myRank,
            static_cast<u32>(algAttrs.algoTypes[0]));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    int32_t meshPos = INVALID_PHYSICAL_LEVEL_IDX;
    for (u32 k = 0; k < effIdx.size(); k++) {
        if (physicalLevels[effIdx[k]].topoType == COMM_TOPO_1DMESH
            && physicalLevels[effIdx[k]].localRanks.size() == userRankSize) {
            meshPos = static_cast<int32_t>(k);
            break;
        }
    }
    CHK_PRT_RET(
        meshPos == INVALID_PHYSICAL_LEVEL_IDX,
        HCCL_INFO("[TopoMatchConcurrentV2] Rank [%u], Mesh algo but no mesh topo layer, not support.", myRank),
        HcclResult::HCCL_E_NOT_SUPPORT);
    int32_t upperPos = FindUpperEncompassingLevel(physicalLevels, effIdx, static_cast<u32>(meshPos));
    CHK_PRT_RET(
        upperPos == INVALID_PHYSICAL_LEVEL_IDX,
        HCCL_INFO("[TopoMatchConcurrentV2] Rank [%u], mesh layer no upper encompassing layer, not support.", myRank),
        HcclResult::HCCL_E_NOT_SUPPORT);
    physicalIdxForAlgoLevels
        = {{static_cast<PhysicalLevelIndex>(effIdx[meshPos]), static_cast<PhysicalLevelIndex>(effIdx[upperPos])}};
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TopoMatchConcurrentV2::MatchTopo(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    u32 myRank = topoInfo->userRank;
    const auto& physicalLevels = topoInfo->physicalLevels;
    if (physicalLevels.empty()) {
        HCCL_ERROR(
            "[TopoMatchConcurrentV2] Rank [%u], physicalLevels is empty. "
            "physicalLevels.size[%zu], userRankSize[%u].",
            myRank, physicalLevels.size(), topoInfo->userRankSize);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 引擎过滤后收集有效层
    std::vector<u32> effIdx = CollectEffectiveIndices(physicalLevels, algAttrs.engine);
    u32 effNum = effIdx.size();
    CHK_PRT_RET(
        effNum == 0 || effNum > ALGO_LEVEL_NUM_TWO,
        HCCL_INFO("[TopoMatchConcurrentV2] Rank [%u], level num[%u] not support.", myRank, effNum),
        HcclResult::HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(
        (topoInfo->userRankSize == 0), HCCL_ERROR("[TopoMatchConcurrentV2] Rank [%u], rankSize is 0.", myRank),
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

    // Mesh 类算法：取最低覆盖全 rank 的 Mesh 物理层 + 其上层超集层；否则 not support
    if (ResolveConcurrentPhysicalIdx(
            physicalLevels, effIdx, algAttrs, topoInfo->userRankSize, myRank, algHierarchyInfo.physicalIdxForAlgoLevels)
        != HcclResult::HCCL_SUCCESS) {
        HCCL_INFO("[TopoMatchConcurrentV2] Rank [%u], get physical index failed.", myRank);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    HCCL_INFO(
        "[TopoMatchConcurrentV2] Rank [%u], rankSize[%u], physicalIdxForAlgoLevels: [%s].", myRank,
        topoInfo->userRankSize, FormatPhysicalIdxForAlgoLevels(algHierarchyInfo.physicalIdxForAlgoLevels).c_str());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
