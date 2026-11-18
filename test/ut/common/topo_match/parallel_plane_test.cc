/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <utility>
#include <vector>
#include "topo_match_one_level.h"
#include "topo_match_test_common.h"

using namespace ops_hccl;
using ops_hccl::ut_helper::MakeLevel;
using ops_hccl::ut_helper::MakeProfile;
using ops_hccl::ut_helper::MakeTopoInfo;
using ops_hccl::ut_helper::Range;

namespace {
PhysicalLevelInfo MakePlane(
    u32 netLayer, u32 topoInstId, std::vector<u32> ranks, std::vector<u32> portNums, CommTopo topoType = COMM_TOPO_CLOS,
    EndpointLocType locType = ENDPOINT_LOC_TYPE_DEVICE, std::vector<CommProtocol> protocols = {COMM_PROTOCOL_UB_CTP})
{
    PhysicalLevelInfo level
        = MakeLevel(std::move(ranks), PhysicalLevelView::GLOBAL, {8}, true, topoType, locType, std::move(protocols));
    level.ref.netLayer = netLayer;
    level.ref.topoInstId = topoInstId;
    level.portNums = std::move(portNums);
    return level;
}
} // namespace

TEST(NhrMultiLinkTopoSourceTest, ExpandsSameLayerParallelPlanes)
{
    auto topo = MakeTopoInfo(
        0, 8,
        {
            MakePlane(0, 0, Range(8), {16}),
            MakePlane(0, 1, Range(8), {16}),
            MakePlane(1, 0, Range(8), {16}),
        });
    AlgHierarchyInfoForAllLevel info;

    ASSERT_EQ(
        TopoMatchOneLevel().MatchTopo(&topo, info, MakeProfile({AlgoType::NHR_MULTILINK})), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 1u);
    EXPECT_EQ(
        info.physicalIdxForAlgoLevels[0],
        std::vector<PhysicalLevelIndex>(
            {PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0, PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1}));
}

TEST(NhrMultiLinkTopoSourceTest, DoesNotExpandAcrossNetLayer)
{
    auto topo = MakeTopoInfo(
        0, 8,
        {
            MakePlane(0, 0, Range(8), {16}),
            MakePlane(1, 0, Range(8), {16}),
        });
    AlgHierarchyInfoForAllLevel info;

    ASSERT_EQ(
        TopoMatchOneLevel().MatchTopo(&topo, info, MakeProfile({AlgoType::NHR_MULTILINK})), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(
        info.physicalIdxForAlgoLevels[0], std::vector<PhysicalLevelIndex>({PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0}));
}

TEST(NhrMultiLinkTopoSourceTest, KeepsSingleSourceForNormalNhr)
{
    auto topo = MakeTopoInfo(
        0, 8,
        {
            MakePlane(0, 0, Range(8), {16}),
            MakePlane(0, 1, Range(8), {16}),
        });
    AlgHierarchyInfoForAllLevel info;

    ASSERT_EQ(TopoMatchOneLevel().MatchTopo(&topo, info, MakeProfile({AlgoType::NHR})), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(
        info.physicalIdxForAlgoLevels[0], std::vector<PhysicalLevelIndex>({PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0}));
}

TEST(NhrMultiLinkTopoSourceTest, KeepsAllSameLayerPlanes)
{
    auto topo = MakeTopoInfo(
        0, 8,
        {
            MakePlane(0, 0, Range(8), {16}),
            MakePlane(0, 1, Range(8), {16}),
            MakePlane(0, 2, Range(8), {16}),
        });
    AlgHierarchyInfoForAllLevel info;

    ASSERT_EQ(
        TopoMatchOneLevel().MatchTopo(&topo, info, MakeProfile({AlgoType::NHR_MULTILINK})), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(
        info.physicalIdxForAlgoLevels[0],
        std::vector<PhysicalLevelIndex>(
            {PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0, PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1,
             PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2}));
}

TEST(NhrMultiLinkTopoSourceTest, KeepsStructuralSourceWithEmptyPorts)
{
    auto topo = MakeTopoInfo(
        0, 8,
        {
            MakePlane(0, 0, Range(8), {16}),
            MakePlane(0, 1, Range(8), {}),
        });
    AlgHierarchyInfoForAllLevel info;

    ASSERT_EQ(
        TopoMatchOneLevel().MatchTopo(&topo, info, MakeProfile({AlgoType::NHR_MULTILINK})), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(info.physicalIdxForAlgoLevels[0].size(), 2u);
}

TEST(NhrMultiLinkTopoSourceTest, KeepsMeshConcurCrossLayerSources)
{
    auto topo = MakeTopoInfo(
        0, 8,
        {
            MakePlane(0, 0, Range(8), {8}, COMM_TOPO_1DMESH),
            MakePlane(1, 0, Range(16), {16}, COMM_TOPO_CLOS),
        });
    AlgHierarchyInfoForAllLevel info;

    ASSERT_EQ(
        TopoMatchOneLevel().MatchTopo(&topo, info, MakeProfile({AlgoType::MESH_CONCUR})), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(
        info.physicalIdxForAlgoLevels[0],
        std::vector<PhysicalLevelIndex>(
            {PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0, PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1}));
}
