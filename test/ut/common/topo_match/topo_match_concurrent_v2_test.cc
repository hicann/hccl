/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "topo_match_concurrent_v2.h"
#include "topo_match_test_common.h"

using namespace ops_hccl;
using ops_hccl::ut_helper::MakeLevel;
using ops_hccl::ut_helper::MakeProfile;
using ops_hccl::ut_helper::MakeTopoInfo;
using ops_hccl::ut_helper::Range;

class TopoMatchConcurrentV2Test : public ::testing::Test {
protected:
    TopoMatchConcurrentV2 matcher_;
};

// C1: 1 层 effIdx，两组同 rank
TEST_F(TopoMatchConcurrentV2Test, SingleLayer)
{
    auto topo = MakeTopoInfo(3, 8, {MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8})});
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCURRENT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos.size(), 1u);
    ASSERT_EQ(info.infos[0].size(), 2u);
    ASSERT_EQ(info.infos[0][0], Range(8));
    ASSERT_EQ(info.infos[0][1], Range(8));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
}

// C2: 2 层 effIdx，physIdx 指向最高有效层
TEST_F(TopoMatchConcurrentV2Test, TwoLayers)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCURRENT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos[0][0], Range(16));
    ASSERT_EQ(info.infos[0][1], Range(16));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// C3: 有效层 > 2 → NOT_SUPPORT
TEST_F(TopoMatchConcurrentV2Test, TooManyLayers)
{
    auto topo = MakeTopoInfo(
        3, 32,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCURRENT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// C4: 有效层 = 0（全 !hasTopoInst）→ NOT_SUPPORT
TEST_F(TopoMatchConcurrentV2Test, NoEffectiveLayers)
{
    auto topo = MakeTopoInfo(3, 8, {MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {}, false)});
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCURRENT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// C5: physicalLevels 空 → INTERNAL
TEST_F(TopoMatchConcurrentV2Test, EmptyPhysicalLevels)
{
    auto topo = MakeTopoInfo(0, 8, {});
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCURRENT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_INTERNAL);
}

// C6: userRankSize=0 → INTERNAL
TEST_F(TopoMatchConcurrentV2Test, ZeroUserRankSize)
{
    auto topo = MakeTopoInfo(0, 0, {MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8})});
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCURRENT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_INTERNAL);
}

// C7: 引擎过滤排除 HOST 层（非 hostdpu）
TEST_F(TopoMatchConcurrentV2Test, EngineExcludesHostLayer)
{
    auto topo = MakeTopoInfo(
        3, 8,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCURRENT}, OpExecuteConfig::AICPU_TS);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // HOST 层被排除，effIdx={1}，physIdx={1}
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// C8: hostdpu 保留 HOST 层
TEST_F(TopoMatchConcurrentV2Test, HostdpuKeepsHostLayer)
{
    auto topo = MakeTopoInfo(
        3, 8,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCURRENT}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // HOST 层保留，effIdx={0,1}，physIdx=effIdx.back()={1}
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}
