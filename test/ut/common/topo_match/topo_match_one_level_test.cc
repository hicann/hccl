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
#include "topo_match_one_level.h"
#include "topo_match_test_common.h"

using namespace ops_hccl;
using ops_hccl::ut_helper::MakeLevel;
using ops_hccl::ut_helper::MakeProfile;
using ops_hccl::ut_helper::MakeTopoInfo;
using ops_hccl::ut_helper::Range;

class TopoMatchOneLevelTest : public ::testing::Test {
protected:
    TopoMatchOneLevel matcher_;
};

// O1: 非 hostdpu 单层满 localRanks
TEST_F(TopoMatchOneLevelTest, NonHostdpuSingleLevelFullLocalRanks)
{
    auto topo = MakeTopoInfo(3, 8, {MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8})});
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos.size(), 1u);
    ASSERT_EQ(info.infos[0].size(), 1u);
    ASSERT_EQ(info.infos[0][0], Range(8));
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 1u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0].size(), 1u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
}

// O2: 非 hostdpu 多层中取 localRanks==userRankSize 的层
TEST_F(TopoMatchOneLevelTest, NonHostdpuPicksFullLocalRanksLevel)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos[0][0], Range(16));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// O3: hostdpu 选 HOST+满 localRanks 层
TEST_F(TopoMatchOneLevelTest, HostdpuPicksHostFullLocalRanks)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos[0][0], Range(16));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// O4: hostdpu 无 HOST+满 localRanks 层 → NOT_SUPPORT
TEST_F(TopoMatchOneLevelTest, HostdpuNoHostFullLocalRanks)
{
    auto topo = MakeTopoInfo(
        3, 8,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// O5: 非 hostdpu HOST 层被排除，选 DEVICE 层
TEST_F(TopoMatchOneLevelTest, NonHostdpuExcludesHostLayer)
{
    auto topo = MakeTopoInfo(
        3, 8,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// O6: AIV 排除含 UBG 链路层
TEST_F(TopoMatchOneLevelTest, AivExcludesUbgLayer)
{
    auto topo = MakeTopoInfo(
        3, 8,
        {
            MakeLevel(
                Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE,
                {COMM_PROTOCOL_UBG}),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE, {}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT}, OpExecuteConfig::AIV);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// O7: physicalLevels 空 → INTERNAL
TEST_F(TopoMatchOneLevelTest, EmptyPhysicalLevels)
{
    auto topo = MakeTopoInfo(0, 8, {});
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_INTERNAL);
}

// O8: userRankSize=0 → INTERNAL
TEST_F(TopoMatchOneLevelTest, ZeroUserRankSize)
{
    auto topo = MakeTopoInfo(0, 0, {MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8})});
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_INTERNAL);
}

// O9: 无满 localRanks 层 → NOT_SUPPORT
TEST_F(TopoMatchOneLevelTest, NoFullLocalRanksLevel)
{
    auto topo = MakeTopoInfo(3, 8, {MakeLevel(Range(4), PhysicalLevelView::GLOBAL, {4})});
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// O10: hostdpu 选 HOST+满 localRanks 层（层序倒置，排除顺序影响，按字段选）
TEST_F(TopoMatchOneLevelTest, HostdpuPicksHostFullLocalRanksReversed)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos[0][0], Range(16));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
}

// O11: AIV 排除含 UBG 链路层（层序倒置，排除顺序影响，按字段排除）
TEST_F(TopoMatchOneLevelTest, AivExcludesUbgLayerReversed)
{
    auto topo = MakeTopoInfo(
        3, 8,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE, {}),
            MakeLevel(
                Range(8), PhysicalLevelView::GLOBAL, {8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE,
                {COMM_PROTOCOL_UBG}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_ONESHOT}, OpExecuteConfig::AIV);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
}
