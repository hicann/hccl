/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "topo_match_two_level.h"
#include "topo_match_test_common.h"

using namespace ops_hccl;
using ops_hccl::ut_helper::MakeLevel;
using ops_hccl::ut_helper::MakeProfile;
using ops_hccl::ut_helper::MakeTopoInfo;
using ops_hccl::ut_helper::Range;

class TopoMatchTwoLevelA2aTest : public ::testing::Test {
protected:
    TopoMatchTwoLevel matcher_;
};

TEST_F(TopoMatchTwoLevelA2aTest, EmptyPhysicalLevelsRejected)
{
    auto topo = MakeTopoInfo(3, 8, {});
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    EXPECT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(TopoMatchTwoLevelA2aTest, ZeroUserRankSizeRejected)
{
    auto topo = MakeTopoInfo(
        3, 0,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    EXPECT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(TopoMatchTwoLevelA2aTest, WrongAlgoTypesSizeRejected)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH});
    AlgHierarchyInfoForAllLevel info;
    EXPECT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_INTERNAL);
}

// HOSTCPU 豁免 GCD=1 非对称拓扑 → 放行
TEST_F(TopoMatchTwoLevelA2aTest, HostcpuEngineAdmitsGcdOneAsymmetricTopo)
{
    auto topo = MakeTopoInfo(
        3, 7,
        {
            MakeLevel(Range(7), PhysicalLevelView::GLOBAL, {3, 4}),
            MakeLevel(Range(7), PhysicalLevelView::GLOBAL, {7}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    EXPECT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos.size(), 2u);
    EXPECT_EQ(info.infos[0][0], (std::vector<u32>{3})); // level0 子组退化为单元素
    EXPECT_EQ(info.infos[1][0].size(), 7u);
}
