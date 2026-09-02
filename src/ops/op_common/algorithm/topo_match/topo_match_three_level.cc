/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_three_level.h"
#include <algorithm>
#include "log.h"

namespace ops_hccl {

namespace {
    // 校验 level 对称并取维度：GLOBAL 看 instList 是否全等；LOCAL 视为对称
    HcclResult ValidateLevelAndCalcDim(
        u32 levelIdx, const std::vector<PhysicalLevelInfo>& physicalLevels, bool& symmetricOut, u32& dim)
    {
        const PhysicalLevelInfo& level = physicalLevels[levelIdx];
        if (level.view == PhysicalLevelView::LOCAL) {
            // LOCAL 无全局 instList，对称性由其上级 netLayer 层判定
            dim = static_cast<u32>(level.localRanks.size());
            symmetricOut = true;
            return HcclResult::HCCL_SUCCESS;
        }
        if (!IsInstListSymmetric(level.instSizeListByLayer)) {
            symmetricOut = false;
            return HcclResult::HCCL_SUCCESS;
        }
        symmetricOut = true;
        dim = static_cast<u32>(level.localRanks.size());
        return HcclResult::HCCL_SUCCESS;
    }

    // ThreeLevel 不支持非对称：p_0/p_1 任一非对称即 not support；维度 d0/d1/d2
    HcclResult CalcDimsAndCheckSymmetry(
        const std::vector<PhysicalLevelInfo>& physicalLevels, u32 phys0, u32 phys1, u32 userRankSize, u32 myRank,
        u32& d0, u32& d1, u32& d2)
    {
        u32 level1TotalSize = 0;
        bool sym0 = false;
        bool sym1 = false;
        CHK_RET(ValidateLevelAndCalcDim(phys0, physicalLevels, sym0, d0));
        CHK_RET(ValidateLevelAndCalcDim(phys1, physicalLevels, sym1, level1TotalSize));
        if (!sym0 || !sym1) {
            HCCL_INFO(
                "[TopoMatchThreeLevel] Rank [%u], asymmetric detected (sym0[%d] sym1[%d]), not support.", myRank,
                static_cast<int32_t>(sym0), static_cast<int32_t>(sym1));
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        if (d0 == 0 || level1TotalSize == 0 || level1TotalSize % d0 != 0) {
            HCCL_INFO(
                "[TopoMatchThreeLevel] Rank [%u], level1TotalSize[%u] not divisible by d0[%u].", myRank,
                level1TotalSize, d0);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        d1 = level1TotalSize / d0;
        if (userRankSize % d0 != 0 || (userRankSize / d0) % d1 != 0) {
            HCCL_INFO(
                "[TopoMatchThreeLevel] Rank [%u], userRankSize[%u] not divisible by d0[%u]*d1[%u].", myRank,
                userRankSize, d0, d1);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        d2 = userRankSize / d0 / d1;
        return HcclResult::HCCL_SUCCESS;
    }
} // namespace

TopoMatchThreeLevel::TopoMatchThreeLevel() {}
TopoMatchThreeLevel::~TopoMatchThreeLevel() {}

HcclResult TopoMatchThreeLevel::MatchTopo(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& profile)
{
    const auto& physicalLevels = topoInfo->physicalLevels;
    u32 myRank = topoInfo->userRank;
    u32 userRankSize = topoInfo->userRankSize;
    if (physicalLevels.empty() || userRankSize == 0 || profile.algoTypes.size() != ALGO_LEVEL_NUM_THREE) {
        HCCL_ERROR("[TopoMatchThreeLevel] Rank [%u], invalid input.", myRank);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 引擎过滤 + 锚点匹配 + 分段 + 最高层校验
    std::vector<u32> effIdx;
    std::vector<u32> pIndices;
    CHK_RET(ResolveMapping(physicalLevels, profile, userRankSize, effIdx, pIndices));
    u32 phys0 = effIdx[pIndices[0]];
    u32 phys1 = effIdx[pIndices[1]];

    // 非对称判定 + 维度计算（ThreeLevel 不支持非对称）
    u32 d0 = 0;
    u32 d1 = 0;
    u32 d2 = 0;
    CHK_RET(CalcDimsAndCheckSymmetry(physicalLevels, phys0, phys1, userRankSize, myRank, d0, d1, d2));

    // 构造 infos；level1 代表环须落在 myRank 所在 level1 instance 内，故 offset 取 instance 基址 + 层内偏移
    std::vector<u32> group0 = physicalLevels[phys0].localRanks;
    u32 level1Base = (myRank / (d0 * d1)) * (d0 * d1);
    std::vector<u32> group1 = BuildRepresentativeGroup(d0, d1, level1Base + myRank % d0);
    std::vector<u32> group2 = BuildRepresentativeGroup(d0 * d1, d2, myRank % (d0 * d1));
    CHK_RET(ValidateGroup(group0, d0, myRank, "level0"));
    CHK_RET(ValidateGroup(group1, d1, myRank, "level1"));
    CHK_RET(ValidateGroup(group2, d2, myRank, "level2"));
    algHierarchyInfo.infos.resize(ALGO_LEVEL_NUM_THREE);
    for (u32 i = 0; i < ALGO_LEVEL_NUM_THREE; i++) {
        algHierarchyInfo.infos[i].resize(1);
    }
    algHierarchyInfo.infos[0][0] = std::move(group0);
    algHierarchyInfo.infos[1][0] = std::move(group1);
    algHierarchyInfo.infos[ALGO_LEVEL_NUM_TWO][0] = std::move(group2);

    // 填充 physicalIdxForAlgoLevels（二级：MeshConcur 双层，普通单层）
    CHK_RET(FillPhysicalIdxForAlgoLevels(
        physicalLevels, effIdx, pIndices, profile.algoTypes, algHierarchyInfo.physicalIdxForAlgoLevels));
    HCCL_INFO(
        "[TopoMatchThreeLevel] Rank [%u], d0[%u] d1[%u] d2[%u], physicalIdxForAlgoLevels: [%s].", myRank, d0, d1, d2,
        FormatPhysicalIdxForAlgoLevels(algHierarchyInfo.physicalIdxForAlgoLevels).c_str());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
