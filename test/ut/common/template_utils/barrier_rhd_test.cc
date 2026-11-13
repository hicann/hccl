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
#include "barrier_rhd_sync_stub.h"
#include <numeric>
#include <set>
#include "barrier_rhd_utils.h"

using namespace ops_hccl;

namespace {
std::vector<BarrierRhdStep> ReferencePlan(u32 rankSize, u32 rank)
{
    u32 power = 1;
    while (power <= rankSize / 2) {
        power *= 2;
    }
    const u32 pairs = rankSize - power;
    const bool folded = rank >= pairs && rank < 2 * pairs;
    const bool paired = rank < 2 * pairs;
    const u32 partner = rank < pairs ? rank + pairs : rank - pairs;
    std::vector<BarrierRhdStep> result;
    if (paired) {
        result.push_back({RhdPhase::PRE, 0, partner});
    }
    if (!folded) {
        const u32 virtualRank = rank < pairs ? rank : rank - pairs;
        u32 step = 0;
        for (u32 distance = 1; distance < power; distance *= 2) {
            const u32 peer = virtualRank ^ distance;
            result.push_back({RhdPhase::MAIN_AG, step++, peer < pairs ? peer : peer + pairs});
        }
    }
    if (paired) {
        result.push_back({RhdPhase::POST, 0, partner});
    }
    return result;
}
} // namespace

TEST(BarrierRhdPlan, MatchesIndependentPartnerTableForAllRoles)
{
    for (u32 size : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 16U, 17U, 31U, 32U, 33U}) {
        std::vector<u32> ranks(size);
        std::iota(ranks.begin(), ranks.end(), 0);
        for (u32 rank : ranks) {
            SCOPED_TRACE(::testing::Message() << "size=" << size << " rank=" << rank);
            std::vector<BarrierRhdStep> actual;
            ASSERT_EQ(BuildBarrierRhdSteps(rank, {ranks}, actual), HCCL_SUCCESS);
            const auto expected = ReferencePlan(size, rank);
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t step = 0; step < actual.size(); ++step) {
                EXPECT_EQ(actual[step].phase, expected[step].phase);
                EXPECT_EQ(actual[step].step, expected[step].step);
                EXPECT_EQ(actual[step].peerRank, expected[step].peerRank);
            }
        }
    }
}

TEST(BarrierRhdPlan, MapsMainAndPart1PeersToDifferentUserRanks)
{
    const std::vector<std::vector<u32>> ranks{{10, 2, 21, 7, 15, 4}};
    std::vector<BarrierRhdStep> steps;
    ASSERT_EQ(BuildBarrierRhdSteps(10, ranks, steps), HCCL_SUCCESS);
    ASSERT_EQ(steps.size(), 4U);
    EXPECT_EQ(steps[0].peerRank, 21U);
    EXPECT_EQ(steps[1].peerRank, 2U);
    EXPECT_EQ(steps[2].peerRank, 15U);
    EXPECT_EQ(steps[2].virtualPeer, 2U);
    EXPECT_EQ(steps[2].algPeer, 4U);
    EXPECT_EQ(steps[3].peerRank, 21U);
    ASSERT_EQ(BuildBarrierRhdSteps(21, ranks, steps), HCCL_SUCCESS);
    ASSERT_EQ(steps.size(), 2U);
    EXPECT_EQ(steps[0].phase, RhdPhase::PRE);
    EXPECT_EQ(steps[1].phase, RhdPhase::POST);
    EXPECT_EQ(steps[0].peerRank, 10U);
    EXPECT_EQ(steps[1].peerRank, 10U);
}

TEST(BarrierRhdPlan, RejectsEmptyDuplicateMissingAndMultidimensionalRanks)
{
    std::vector<BarrierRhdStep> steps;
    EXPECT_EQ(BuildBarrierRhdSteps(0, {}, steps), HCCL_E_PARA);
    EXPECT_EQ(BuildBarrierRhdSteps(0, {{}}, steps), HCCL_E_PARA);
    EXPECT_EQ(BuildBarrierRhdSteps(0, {{0, 0}}, steps), HCCL_E_PARA);
    EXPECT_EQ(BuildBarrierRhdSteps(2, {{0, 1}}, steps), HCCL_E_PARA);
    EXPECT_EQ(BuildBarrierRhdSteps(0, {{0}, {0}}, steps), HCCL_E_PARA);
}

TEST(BarrierRhdPlan, ResourcesContainExactlyThePlanPeers)
{
    const std::vector<std::vector<u32>> ranks{{10, 2, 21, 7, 15, 4}};
    for (bool hostDpu : {false, true}) {
        for (u32 rank : ranks[0]) {
            TopoInfoWithNetLayerDetails topo{};
            topo.userRank = rank;
            OpParam param{};
            AlgResourceRequest request{};
            ASSERT_EQ(CalcBarrierRhdResources(nullptr, param, &topo, ranks, hostDpu, request), HCCL_SUCCESS);
            ASSERT_EQ(request.channels.size(), 1U);
            std::vector<BarrierRhdStep> steps;
            ASSERT_EQ(BuildBarrierRhdSteps(rank, ranks, steps), HCCL_SUCCESS);
            std::set<u32> expected;
            for (const auto& step : steps) {
                expected.insert(step.peerRank);
            }
            std::set<u32> actual;
            for (const auto& channel : request.channels[0]) {
                actual.insert(channel.remoteRank);
            }
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(request.channels[0].size(), expected.size());
            EXPECT_EQ(request.slaveThreadNum, 0U);
            EXPECT_EQ(request.notifyNumOnMainThread, 0U);
            EXPECT_TRUE(request.notifyNumPerThread.empty());
        }
    }
}

TEST(BarrierRhdPlan, RejectsHostEndpointDuringAicpuResourceCalculation)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = 0;
    OpParam param{};
    AlgResourceRequest request{};
    barrier_rhd_sync_ut::Reset();
    barrier_rhd_sync_ut::useHostRemoteEndpoint = true;
    EXPECT_EQ(CalcBarrierRhdResources(nullptr, param, &topo, {{0, 1}}, false, request), HCCL_E_NOT_SUPPORT);
    barrier_rhd_sync_ut::Reset();
}

TEST(BarrierRhdPlan, ValidatesAllChannelsBeforeExecutingAnyStep)
{
    std::vector<BarrierRhdStep> steps;
    ASSERT_EQ(BuildBarrierRhdSteps(0, {{0, 1, 2, 3}}, steps), HCCL_SUCCESS);
    std::map<u32, std::vector<ChannelInfo>> channels;
    EXPECT_EQ(CheckBarrierRhdChannels(steps, channels, ENDPOINT_LOC_TYPE_DEVICE), HCCL_E_INTERNAL);
    channels[1].resize(1);
    channels[1][0].locationType = ENDPOINT_LOC_TYPE_DEVICE;
    EXPECT_EQ(CheckBarrierRhdChannels(steps, channels, ENDPOINT_LOC_TYPE_DEVICE), HCCL_E_INTERNAL);
    channels[2].resize(1);
    for (auto& entry : channels) {
        entry.second[0].locationType = ENDPOINT_LOC_TYPE_DEVICE;
    }
    EXPECT_EQ(CheckBarrierRhdChannels(steps, channels, ENDPOINT_LOC_TYPE_DEVICE), HCCL_SUCCESS);
}

TEST(BarrierRhdPlan, RejectsHostHandlesInAicpuAndDeviceHandlesInDpu)
{
    std::vector<BarrierRhdStep> steps;
    ASSERT_EQ(BuildBarrierRhdSteps(0, {{0, 1, 2, 3}}, steps), HCCL_SUCCESS);
    std::map<u32, std::vector<ChannelInfo>> channels;
    for (const auto& step : steps) {
        channels[step.peerRank].resize(1);
        channels[step.peerRank][0].locationType = ENDPOINT_LOC_TYPE_DEVICE;
    }
    EXPECT_EQ(CheckBarrierRhdChannels(steps, channels, ENDPOINT_LOC_TYPE_HOST), HCCL_E_NOT_SUPPORT);
    channels.at(steps.back().peerRank)[0].locationType = ENDPOINT_LOC_TYPE_HOST;
    EXPECT_EQ(CheckBarrierRhdChannels(steps, channels, ENDPOINT_LOC_TYPE_DEVICE), HCCL_E_NOT_SUPPORT);
    for (auto& entry : channels) {
        entry.second[0].locationType = ENDPOINT_LOC_TYPE_HOST;
    }
    EXPECT_EQ(CheckBarrierRhdChannels(steps, channels, ENDPOINT_LOC_TYPE_HOST), HCCL_SUCCESS);
}
