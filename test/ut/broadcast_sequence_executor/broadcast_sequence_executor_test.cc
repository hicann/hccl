/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <vector>
#include "broadcast_sequence_executor_utils.h"

using namespace ops_hccl;
using namespace ops_hccl::broadcast_sequence_executor;

namespace {
HcclChannelDesc MakeChannel(
    u32 remoteRank, CommProtocol protocol = CommProtocol::COMM_PROTOCOL_HCCS,
    EndpointLocType localLocType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE,
    EndpointLocType remoteLocType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE)
{
    HcclChannelDesc channel{};
    channel.remoteRank = remoteRank;
    channel.channelProtocol = protocol;
    channel.localEndpoint.loc.locType = localLocType;
    channel.remoteEndpoint.loc.locType = remoteLocType;
    return channel;
}

std::map<u32, std::vector<HcclChannelDesc>> GroupByRank(const std::vector<HcclChannelDesc>& channels)
{
    std::map<u32, std::vector<HcclChannelDesc>> grouped;
    for (const auto& channel : channels) {
        grouped[channel.remoteRank].push_back(channel);
    }
    return grouped;
}

class ChannelsPerRankRecorder {
public:
    HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
    {
        ++callCount;
        lastChannelCount = channels.size();
        return HCCL_SUCCESS;
    }

    u32 callCount{0};
    size_t lastChannelCount{0};
};
} // namespace

TEST(BroadcastSequenceExecutorTest, MergeNotifyNumPerThreadUsesPerThreadMaximum)
{
    std::vector<u32> merged = {2, 5};
    MergeNotifyNumPerThread(merged, {4, 1, 3});
    EXPECT_EQ(merged, (std::vector<u32>{4, 5, 3}));

    MergeNotifyNumPerThread(merged, {1});
    EXPECT_EQ(merged, (std::vector<u32>{4, 5, 3}));

    MergeNotifyNumPerThread(merged, {});
    EXPECT_EQ(merged, (std::vector<u32>{4, 5, 3}));
}

TEST(BroadcastSequenceExecutorTest, SetChannelsPerRankIfNotEmptySkipsEmptyAndDelegatesNonEmpty)
{
    ChannelsPerRankRecorder recorder;
    using ChannelMap = std::map<u32, std::vector<ChannelInfo>>;
    ChannelMap channels;

    EXPECT_EQ(SetChannelsPerRankIfNotEmpty(&recorder, channels), HCCL_SUCCESS);
    EXPECT_EQ(recorder.callCount, 0U);

    channels[1];
    EXPECT_EQ(SetChannelsPerRankIfNotEmpty(&recorder, channels), HCCL_SUCCESS);
    EXPECT_EQ(recorder.callCount, 1U);
    EXPECT_EQ(recorder.lastChannelCount, 1U);
}

TEST(BroadcastSequenceExecutorTest, ValidateChannelRequestCompatibleAllowsDifferentPeersWithSameSignature)
{
    const std::vector<HcclChannelDesc> scatterChannels = {MakeChannel(1), MakeChannel(1), MakeChannel(2)};
    const std::vector<HcclChannelDesc> allGatherChannels = {MakeChannel(2), MakeChannel(3)};

    EXPECT_EQ(ValidateChannelRequestCompatible(scatterChannels, allGatherChannels), HCCL_SUCCESS);
}

TEST(BroadcastSequenceExecutorTest, ValidateChannelRequestCompatibleRejectsProtocolConflict)
{
    const std::vector<HcclChannelDesc> scatterChannels = {MakeChannel(
        1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST,
        EndpointLocType::ENDPOINT_LOC_TYPE_HOST)};
    const std::vector<HcclChannelDesc> allGatherChannels = {MakeChannel(2)};

    EXPECT_EQ(ValidateChannelRequestCompatible(scatterChannels, allGatherChannels), HCCL_E_PARA);
}

TEST(BroadcastSequenceExecutorTest, ValidateChannelRequestCompatibleRejectsEndpointTypeConflict)
{
    const std::vector<HcclChannelDesc> scatterChannels = {MakeChannel(1)};
    const std::vector<HcclChannelDesc> allGatherChannels = {MakeChannel(
        1, CommProtocol::COMM_PROTOCOL_HCCS, EndpointLocType::ENDPOINT_LOC_TYPE_HOST,
        EndpointLocType::ENDPOINT_LOC_TYPE_HOST)};

    EXPECT_EQ(ValidateChannelRequestCompatible(scatterChannels, allGatherChannels), HCCL_E_PARA);
}

TEST(BroadcastSequenceExecutorTest, ValidateChannelRequestCompatibleRejectsPerPeerSignatureMismatch)
{
    const std::vector<HcclChannelDesc> scatterChannels
        = {MakeChannel(1), MakeChannel(
                               2, CommProtocol::COMM_PROTOCOL_HCCS, EndpointLocType::ENDPOINT_LOC_TYPE_HOST,
                               EndpointLocType::ENDPOINT_LOC_TYPE_HOST)};
    const std::vector<HcclChannelDesc> allGatherChannels
        = {MakeChannel(
               1, CommProtocol::COMM_PROTOCOL_HCCS, EndpointLocType::ENDPOINT_LOC_TYPE_HOST,
               EndpointLocType::ENDPOINT_LOC_TYPE_HOST),
           MakeChannel(2)};

    EXPECT_EQ(ValidateChannelRequestCompatible(scatterChannels, allGatherChannels), HCCL_E_PARA);
}

TEST(BroadcastSequenceExecutorTest, ValidateChannelRequestCompatibleRejectsOneSidedRequest)
{
    const std::vector<HcclChannelDesc> scatterChannels = {MakeChannel(1)};

    EXPECT_EQ(ValidateChannelRequestCompatible(scatterChannels, {}), HCCL_E_PARA);
    EXPECT_EQ(ValidateChannelRequestCompatible({}, scatterChannels), HCCL_E_PARA);
}

TEST(BroadcastSequenceExecutorTest, MergeChannelRequestUnionsPeersAndKeepsMaxChannels)
{
    const std::vector<HcclChannelDesc> scatterChannels = {MakeChannel(1), MakeChannel(1), MakeChannel(2)};
    const std::vector<HcclChannelDesc> allGatherChannels = {MakeChannel(2), MakeChannel(3), MakeChannel(3)};
    std::vector<HcclChannelDesc> merged;

    ASSERT_EQ(MergeChannelRequest(scatterChannels, allGatherChannels, merged), HCCL_SUCCESS);
    const auto grouped = GroupByRank(merged);
    ASSERT_EQ(grouped.size(), 3U);
    EXPECT_EQ(grouped.at(1).size(), 2U);
    EXPECT_EQ(grouped.at(2).size(), 1U);
    EXPECT_EQ(grouped.at(3).size(), 2U);
}

TEST(BroadcastSequenceExecutorTest, MergeChannelRequestUsesPeerUnionWhenScatterVectorIsLonger)
{
    const std::vector<HcclChannelDesc> scatterChannels = {MakeChannel(1), MakeChannel(1), MakeChannel(2)};
    const std::vector<HcclChannelDesc> allGatherChannels = {MakeChannel(2), MakeChannel(4)};
    std::vector<HcclChannelDesc> merged;

    ASSERT_EQ(MergeChannelRequest(scatterChannels, allGatherChannels, merged), HCCL_SUCCESS);
    const auto grouped = GroupByRank(merged);
    EXPECT_EQ(grouped.size(), 3U);
    EXPECT_EQ(grouped.at(1).size(), 2U);
    EXPECT_EQ(grouped.at(2).size(), 1U);
    EXPECT_EQ(grouped.at(4).size(), 1U);
}

TEST(BroadcastSequenceExecutorTest, MergeChannelRequestHandlesEmptyAndSingleSidedRequests)
{
    std::vector<HcclChannelDesc> merged;
    ASSERT_EQ(MergeChannelRequest({}, {}, merged), HCCL_SUCCESS);
    EXPECT_TRUE(merged.empty());

    const std::vector<HcclChannelDesc> allGatherChannels = {MakeChannel(3), MakeChannel(4)};
    ASSERT_EQ(MergeChannelRequest({}, allGatherChannels, merged), HCCL_SUCCESS);
    EXPECT_EQ(merged.size(), allGatherChannels.size());
}

TEST(BroadcastSequenceExecutorTest, MergeChannelRequestRejectsProtocolConflict)
{
    const std::vector<HcclChannelDesc> scatterChannels = {MakeChannel(5, CommProtocol::COMM_PROTOCOL_HCCS)};
    const std::vector<HcclChannelDesc> allGatherChannels = {MakeChannel(5, CommProtocol::COMM_PROTOCOL_ROCE)};
    std::vector<HcclChannelDesc> merged;

    EXPECT_EQ(MergeChannelRequest(scatterChannels, allGatherChannels, merged), HCCL_E_PARA);
}

TEST(BroadcastSequenceExecutorTest, SplitFloorTailAssignsRemainderToLastRank)
{
    u64 sliceCount = 0;
    u64 tailCount = 0;

    SplitFloorTail(10, 3, sliceCount, tailCount);
    EXPECT_EQ(sliceCount, 3U);
    EXPECT_EQ(tailCount, 4U);

    SplitFloorTail(9, 3, sliceCount, tailCount);
    EXPECT_EQ(sliceCount, 3U);
    EXPECT_EQ(tailCount, 3U);

    SplitFloorTail(2, 4, sliceCount, tailCount);
    EXPECT_EQ(sliceCount, 0U);
    EXPECT_EQ(tailCount, 2U);
}

TEST(BroadcastSequenceExecutorTest, SplitFloorTailHandlesZeroRankSize)
{
    u64 sliceCount = 7;
    u64 tailCount = 9;
    SplitFloorTail(10, 0, sliceCount, tailCount);
    EXPECT_EQ(sliceCount, 0U);
    EXPECT_EQ(tailCount, 0U);
}

TEST(BroadcastSequenceExecutorTest, FillFloorTailSlicesCoversAllDataWithLastTail)
{
    constexpr u64 dataCount = 10;
    constexpr u64 dataTypeSize = 4;
    constexpr u32 rankSize = 3;
    TemplateDataParams params;

    FillFloorTailSlices(dataCount, dataTypeSize, rankSize, params);

    EXPECT_EQ(params.allRankProcessedDataCount, (std::vector<u64>{3, 3, 4}));
    EXPECT_EQ(params.allRankSliceSize, (std::vector<u64>{12, 12, 16}));
    EXPECT_EQ(params.allRankDispls, (std::vector<u64>{0, 12, 24}));
    EXPECT_EQ(params.sliceSize, 12U);
    EXPECT_EQ(params.tailSize, 16U);
    EXPECT_EQ(
        std::accumulate(params.allRankSliceSize.begin(), params.allRankSliceSize.end(), u64{0}),
        dataCount * dataTypeSize);
}

TEST(BroadcastSequenceExecutorTest, FillFloorTailSlicesSupportsDivisibleAndDegenerateCases)
{
    TemplateDataParams params;
    FillFloorTailSlices(12, 4, 3, params);
    EXPECT_EQ(params.allRankProcessedDataCount, (std::vector<u64>{4, 4, 4}));
    EXPECT_EQ(params.allRankSliceSize, (std::vector<u64>{16, 16, 16}));
    EXPECT_EQ(params.allRankDispls, (std::vector<u64>{0, 16, 32}));
    EXPECT_EQ(params.sliceSize, 16U);
    EXPECT_EQ(params.tailSize, 16U);

    FillFloorTailSlices(2, 4, 4, params);
    EXPECT_EQ(params.allRankProcessedDataCount, (std::vector<u64>{0, 0, 0, 2}));
    EXPECT_EQ(params.allRankSliceSize, (std::vector<u64>{0, 0, 0, 8}));
    EXPECT_EQ(params.allRankDispls, (std::vector<u64>{0, 0, 0, 0}));
    EXPECT_EQ(params.sliceSize, 0U);
    EXPECT_EQ(params.tailSize, 8U);

    FillFloorTailSlices(2, 4, 0, params);
    EXPECT_TRUE(params.allRankProcessedDataCount.empty());
    EXPECT_TRUE(params.allRankSliceSize.empty());
    EXPECT_TRUE(params.allRankDispls.empty());
    EXPECT_EQ(params.sliceSize, 0U);
    EXPECT_EQ(params.tailSize, 0U);
}
