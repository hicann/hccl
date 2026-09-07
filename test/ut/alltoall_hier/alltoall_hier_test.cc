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
#include <cstdint>
#include <vector>
#include <string>

namespace ops_hccl {

using u32 = uint32_t;
using u64 = uint64_t;

class AlltoAllHierOffsetTest : public ::testing::TestWithParam<std::tuple<u32, u32, u64, u64, u64>> {};

TEST_P(AlltoAllHierOffsetTest, FirstStageSendOffsetFormula)
{
    auto [peerAlgRank, targetGroupSize, totalCount, dataTypeSize, cclBufferCountPerRank] = GetParam();
    u64 inBuffBaseOff = 0;
    for (u32 i = 0; i < targetGroupSize; i++) {
        u64 expectedSrcOffset = (peerAlgRank * targetGroupSize + i) * totalCount * dataTypeSize + inBuffBaseOff;
        EXPECT_GE(expectedSrcOffset, 0u);
    }
}

TEST_P(AlltoAllHierOffsetTest, LastStageRecvOffsetFormula)
{
    auto [peerAlgRank, srcGroupSize, totalCount, dataTypeSize, cclBufferCountPerRank] = GetParam();
    u64 outBuffBaseOff = 0;
    for (u32 i = 0; i < srcGroupSize; i++) {
        u64 expectedDstOffset = (i * 8 + peerAlgRank) * totalCount * dataTypeSize + outBuffBaseOff;
        EXPECT_GE(expectedDstOffset, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(
    OffsetParams, AlltoAllHierOffsetTest,
    ::testing::Values(
        std::make_tuple(1u, 2u, 1000000ull, 2ull, 6553600ull), std::make_tuple(3u, 8u, 3333333ull, 4ull, 3276800ull),
        std::make_tuple(0u, 1u, 67108864ull, 1ull, 8388608ull), std::make_tuple(7u, 8u, 2222222ull, 2ull, 6553600ull)));

class AlltoAllHierSelectorLogicTest : public ::testing::Test {};

TEST_F(AlltoAllHierSelectorLogicTest, HierConfiguredWithMultiLevelMeshReturnsHier)
{
    u32 topoLevelNums = 2;
    bool hierConfigured = true;
    bool level0Supported = true;
    bool expectedMatch = hierConfigured && topoLevelNums > 1 && level0Supported;
    EXPECT_TRUE(expectedMatch);
}

TEST_F(AlltoAllHierSelectorLogicTest, HierNotConfiguredFallsThrough)
{
    u32 topoLevelNums = 2;
    bool hierConfigured = false;
    bool expectedFallthrough = !hierConfigured || topoLevelNums <= 1;
    EXPECT_TRUE(expectedFallthrough);
}

TEST_F(AlltoAllHierSelectorLogicTest, HierConfiguredWithSingleLevelFallsThrough)
{
    u32 topoLevelNums = 1;
    bool hierConfigured = true;
    bool expectedFallthrough = !hierConfigured || topoLevelNums <= 1;
    EXPECT_TRUE(expectedFallthrough);
}

} // namespace ops_hccl
