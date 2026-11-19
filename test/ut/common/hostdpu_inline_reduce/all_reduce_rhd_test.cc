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
#include "alg_attrs_registry.h"
#include "cost_model.h"
#include "ins_v2_all_reduce_sequence_rhd_tail.h"

namespace {
using namespace ops_hccl;

const char* RhdAlgorithm(u32 levels)
{
    return levels == 2 ? "AicpuAllReduceSequenceMeshConcurRHD" : "AicpuAllReduceSequenceMeshConcurRHDRHD";
}

TopoInfoWithNetLayerDetails MakeRhdTopo(u32 levels)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.topoLevelNums = levels;
    topo.level0Topo = Level0Shape::MESH_1D;
    topo.userRankSize = levels == 2 ? 3 : 9;
    topo.Level1Nhr = true;
    topo.physicalLevels.resize(levels);
    for (auto& level : topo.physicalLevels) {
        level.topoType = CommTopo::COMM_TOPO_CLOS;
        level.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    }
    topo.physicalLevels[0].topoType = CommTopo::COMM_TOPO_1DMESH;
    return topo;
}

TEST(AllReduceRhdSelection, ExplicitTwoLevelConfigAllowsLevel1Nhr)
{
    auto topo = MakeRhdTopo(2);
    ASSERT_NE(AlgAttrsRegistry::Instance().Get(RhdAlgorithm(2)), nullptr);
    const auto result = CheckAlgoMatchTopoWithReason(RhdAlgorithm(2), &topo, false);
    EXPECT_TRUE(result.matched) << result.reason;
}

TEST(AllReduceRhdSelection, ExplicitThreeLevelConfigAllowsLevel1Nhr)
{
    auto topo = MakeRhdTopo(3);
    ASSERT_NE(AlgAttrsRegistry::Instance().Get(RhdAlgorithm(3)), nullptr);
    const auto result = CheckAlgoMatchTopoWithReason(RhdAlgorithm(3), &topo, false);
    EXPECT_TRUE(result.matched) << result.reason;
}

TEST(AllReduceRhdSelection, DefaultPolicyStillExcludesRhd)
{
    for (u32 levels : {2, 3}) {
        auto topo = MakeRhdTopo(levels);
        const auto result = CheckAlgoMatchTopoWithReason(RhdAlgorithm(levels), &topo, true);
        EXPECT_FALSE(result.matched);
        EXPECT_EQ(result.reason, "topoCustomCheck returned false");
    }
}

TEST(AllReduceRhdSelection, AicpuStillRejectsHostDpuTopology)
{
    for (u32 levels : {2, 3}) {
        auto topo = MakeRhdTopo(levels);
        topo.hostDpuOnly = true;
        const auto result = CheckAlgoMatchTopoWithReason(RhdAlgorithm(levels), &topo, false);
        EXPECT_FALSE(result.matched);
        EXPECT_EQ(result.reason, "hostDpuOnly=true, isHostDpuOnly=false");
    }
}

TEST(AllReduceRhdSelection, ThreeLevelStillRequiresDeviceLinks)
{
    for (u32 hostLevel : {1, 2}) {
        auto topo = MakeRhdTopo(3);
        topo.physicalLevels[hostLevel].locType = EndpointLocType::ENDPOINT_LOC_TYPE_HOST;
        const auto result = CheckAlgoMatchTopoWithReason(RhdAlgorithm(3), &topo, false);
        EXPECT_FALSE(result.matched);
        EXPECT_EQ(result.reason, "topoPrecondCheck returned false (hard precondition)");
    }
}

TEST(AllReduceRhdLoopPlan, AlignedInputDoesNotUseTailStorage)
{
    const auto plan = ArSeqRhdPlanLoop(144, 72, 12);
    EXPECT_EQ(plan.validCount, 72u);
    EXPECT_EQ(plan.currDataCount, 72u);
    EXPECT_FALSE(plan.paddedTail);
}

TEST(AllReduceRhdLoopPlan, EachElementIsProcessedOnceAndOnlyFinalLoopIsPadded)
{
    for (u64 ranks : {1, 2, 3, 6, 12, 16, 24, 72}) {
        const u64 capacity = 3 * ranks;
        for (u64 total = 1; total <= 5 * ranks + 1; ++total) {
            u64 processed = 0;
            u64 paddedLoops = 0;
            while (processed < total) {
                const auto plan = ArSeqRhdPlanLoop(total - processed, capacity, ranks);
                ASSERT_GT(plan.validCount, 0u);
                ASSERT_LE(plan.validCount, total - processed);
                ASSERT_LE(plan.currDataCount, capacity);
                EXPECT_EQ(plan.currDataCount % ranks, 0u);
                if (plan.paddedTail) {
                    ++paddedLoops;
                    EXPECT_LT(plan.validCount, ranks);
                    EXPECT_EQ(plan.currDataCount, ranks);
                    EXPECT_EQ(processed + plan.validCount, total);
                } else {
                    EXPECT_EQ(plan.currDataCount, plan.validCount);
                }
                processed += plan.validCount;
            }
            EXPECT_EQ(processed, total);
            EXPECT_EQ(paddedLoops, static_cast<u64>(total % ranks != 0));
        }
    }
}

TEST(AllReduceRhdLoopPlan, OneMiBOn72RanksHas64ValidTailElements)
{
    const u64 total = 1024 * 1024 / sizeof(int32_t);
    u64 processed = 0;
    u64 tailCount = 0;
    while (processed < total) {
        const auto plan = ArSeqRhdPlanLoop(total - processed, 4032, 72);
        ASSERT_GT(plan.validCount, 0u);
        if (plan.paddedTail) {
            tailCount = plan.validCount;
            EXPECT_EQ(plan.currDataCount, 72u);
        }
        processed += plan.validCount;
    }
    EXPECT_EQ(processed, total);
    EXPECT_EQ(tailCount, 64u);
}
} // namespace
