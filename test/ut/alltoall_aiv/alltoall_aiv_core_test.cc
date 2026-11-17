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
#include <iterator>
#include <map>
#include <utility>
#include <vector>
#include "aiv_all_to_all_mesh_1D_write.h"

namespace alltoall_aiv_test {
namespace {
    struct Region {
        uint64_t end;
        uint32_t value;
    };
    using Flag = std::pair<uint32_t, uint64_t>;

    template <typename T>
    void
    RunScenario(uint32_t ranks, uint32_t blocks, uint32_t expectedActive, uint64_t count, uint32_t physicalBlocks = 0)
    {
        SCOPED_TRACE(
            ::testing::Message() << "ranks=" << ranks << " blocks=" << blocks << " count=" << count
                                 << " elementBytes=" << sizeof(T));
        const uint64_t bytes = count * sizeof(T);
        // 各peer分片之间留有空隙，用于验证kernel使用用户传入的输入/输出stride。
        const uint64_t stride = bytes + 64;
        const uint32_t launchedBlocks = physicalBlocks == 0 ? blocks : physicalBlocks;
        std::vector<Trace> traces;
        traces.reserve(ranks * launchedBlocks);
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            uint32_t active = 0;
            for (uint32_t block = 0; block < launchedBlocks; ++block) {
                traces.push_back(Trace{rank, block, 0, 0, {}});
                currentTrace = &traces.back();
                AivAlltoAllV2Mesh1DWrite<T>(
                    nullptr, Address(INPUT_BASE, rank), Address(OUTPUT_BASE, rank), rank, 0, ranks, count, 0, 0, 0, 1,
                    stride, stride, 1, 0, 0, blocks, true, nullptr, nullptr, nullptr, 0, false, true);
                active += !currentTrace->events.empty();
                if (block >= expectedActive) {
                    EXPECT_TRUE(currentTrace->events.empty()) << "non-copying core copies data or accesses flags";
                } else if (blocks >= ranks && bytes > DATA_LIMIT) {
                    // 多核大数据量走ProcessMultiCore：无presync，首个动作即直写对端。
                    ASSERT_FALSE(currentTrace->events.empty());
                    EXPECT_EQ(currentTrace->events.front().kind, EventKind::COPY);
                } else if (blocks >= ranks && bytes <= DATA_LIMIT) {
                    // 多核小数据量走ProcessSmallData：每peer一个核，直接直写对端并发布数据flag。
                    ASSERT_FALSE(currentTrace->events.empty());
                    const auto& putRemote = currentTrace->events.front();
                    EXPECT_EQ(putRemote.kind, EventKind::COPY);
                }
                EXPECT_EQ(currentTrace->firstBarriers, 1u);
                if (bytes > DATA_LIMIT || ranks * bytes > PINGPONG_TOTAL_DATA_LIMIT) {
                    EXPECT_EQ(currentTrace->endBarriers, 1u);
                }
            }
            EXPECT_EQ(active, expectedActive);
        }
        currentTrace = nullptr;

        std::map<uint64_t, Region> memory;
        if (bytes != 0) {
            for (uint32_t source = 0; source < ranks; ++source) {
                for (uint32_t destination = 0; destination < ranks; ++destination) {
                    const uint64_t begin = Address(INPUT_BASE, source) + destination * stride;
                    memory.emplace(begin, Region{begin + bytes, source * 64 + destination});
                }
            }
        }
        std::map<Flag, int32_t> flags;
        std::map<Flag, uint32_t> writers;
        std::map<Flag, uint32_t> readers;
        std::vector<size_t> positions(traces.size(), 0);
        size_t remaining = 0;
        for (const auto& trace : traces) {
            remaining += trace.events.size();
        }
        while (remaining != 0) {
            bool progressed = false;
            for (size_t index = 0; index < traces.size(); ++index) {
                const auto& trace = traces[index];
                if (positions[index] == trace.events.size()) {
                    continue;
                }
                const auto& event = trace.events[positions[index]];
                const Flag flag{event.target, event.slot};
                if (event.kind == EventKind::WAIT) {
                    auto it = flags.find(flag);
                    if (it == flags.end() || it->second != event.tag) {
                        continue;
                    }
                    ++readers[flag];
                } else if (event.kind == EventKind::RECORD) {
                    EXPECT_LT(event.target, ranks);
                    // 多核大数据量数据flag在[0, blocks)，按(源rank, 分片核)编号；小数据量与少核路径在[0, ranks)。
                    const uint32_t flagSlots = (blocks >= ranks && bytes > DATA_LIMIT) ? blocks : ranks;
                    EXPECT_LT(event.slot, flagSlots);
                    EXPECT_EQ(++writers[flag], 1u) << "multiple cores publish the same flag";
                    flags[flag] = event.tag;
                } else if (event.bytes != 0) {
                    auto source = memory.upper_bound(event.source);
                    ASSERT_NE(source, memory.begin()) << "copy reads unwritten memory";
                    --source;
                    ASSERT_LE(event.source + event.bytes, source->second.end) << "copy reads across an unwritten slice";
                    auto next = memory.lower_bound(event.destination);
                    if (next != memory.end()) {
                        ASSERT_LE(event.destination + event.bytes, next->first) << "overlapping destination writes";
                    }
                    if (next != memory.begin()) {
                        ASSERT_LE(std::prev(next)->second.end, event.destination) << "overlapping destination writes";
                    }
                    memory.emplace(event.destination, Region{event.destination + event.bytes, source->second.value});
                }
                ++positions[index];
                --remaining;
                progressed = true;
            }
            ASSERT_TRUE(progressed) << "unmatched flag dependencies would deadlock the kernel";
        }
        EXPECT_EQ(readers, writers);
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            for (uint32_t peer = 0; peer < ranks; ++peer) {
                uint64_t position = Address(OUTPUT_BASE, rank) + peer * stride;
                const uint64_t end = position + bytes;
                while (position < end) {
                    auto region = memory.find(position);
                    ASSERT_NE(region, memory.end()) << "output slice has a gap";
                    EXPECT_EQ(region->second.value, peer * 64 + rank) << "wrong source/destination data";
                    ASSERT_LE(region->second.end, end) << "output writes into the peer stride gap";
                    position = region->second.end;
                }
            }
        }
    }

    TEST(AlltoAllAivCoreTest, KernelBudgetLimitsTransfersAndCoversEveryPeer)
    {
        const uint32_t cases[][3]
            = {{1, 48, 8},   {2, 48, 8},   {3, 48, 6},   {4, 48, 8},   {5, 45, 5},   {6, 48, 6},   {7, 42, 7},
               {8, 48, 8},   {9, 45, 9},   {10, 40, 10}, {11, 44, 11}, {12, 48, 12}, {13, 39, 13}, {14, 42, 14},
               {15, 45, 15}, {16, 48, 16}, {17, 34, 17}, {32, 32, 32}, {48, 48, 48}};
        for (const auto& scenario : cases) {
            RunScenario<uint8_t>(scenario[0], scenario[1], scenario[2], 512 * 1024 + 3);
            RunScenario<uint32_t>(scenario[0], scenario[1], scenario[2], 128 * 1024 + 3);
        }
    }

    TEST(AlltoAllAivCoreTest, PaddedPhysicalCoresDoNotAccessInvalidRanks)
    {
        RunScenario<uint8_t>(2, 4, 4, 512 * 1024 + 1);
        RunScenario<uint8_t>(4, 8, 8, 512 * 1024 + 1);
        RunScenario<uint8_t>(8, 8, 8, 512 * 1024 + 1);
        RunScenario<uint8_t>(8, 8, 8, 512 * 1024 + 1, 24);
        for (uint32_t blocks : {16u, 20u, 24u, 30u, 31u}) {
            RunScenario<uint8_t>(16, 16, 16, 512 * 1024 + 3, blocks);
        }
    }

    TEST(AlltoAllAivCoreTest, UserLimitedPeerGroupsKeepTheirSliceCoverage)
    {
        for (uint32_t blocks : {2u, 4u, 6u, 8u}) {
            RunScenario<uint8_t>(2, blocks, blocks, 512 * 1024 + 1);
        }
        RunScenario<uint8_t>(4, 4, 4, 512 * 1024 + 1);
        RunScenario<uint8_t>(4, 8, 8, 512 * 1024 + 1);
        RunScenario<uint8_t>(3, 3, 3, 512 * 1024 + 1);
        RunScenario<uint8_t>(3, 6, 6, 512 * 1024 + 1);
        RunScenario<uint8_t>(6, 6, 6, 512 * 1024 + 1);
        RunScenario<uint8_t>(2, 10, 8, 512 * 1024 + 1);
        RunScenario<uint8_t>(3, 15, 6, 512 * 1024 + 1);
        RunScenario<uint8_t>(4, 16, 8, 512 * 1024 + 1);
        RunScenario<uint8_t>(8, 16, 8, 512 * 1024 + 1);
        RunScenario<uint8_t>(16, 32, 16, 512 * 1024 + 1);
        RunScenario<uint8_t>(8, 32, 8, 2 * 1024 * 1024 + 3);
    }

    TEST(AlltoAllAivCoreTest, SmallDataUsesPerPeerWritePath)
    {
        for (uint64_t count : {0ULL, 1ULL, 2ULL, 3ULL, 5ULL, 31ULL, 32ULL, 33ULL, 64 * 1024ULL, 512 * 1024ULL}) {
            RunScenario<uint8_t>(2, 2, 2, count);
            RunScenario<uint8_t>(4, 4, 4, count);
            RunScenario<uint8_t>(8, 8, 8, count);
            RunScenario<uint8_t>(16, 16, 16, count);
            RunScenario<uint32_t>(4, 4, 4, count / sizeof(uint32_t));
            // 核数不足时仍回落到少核拉取路径。
            RunScenario<uint8_t>(8, 4, 4, count);
        }
    }

    TEST(AlltoAllAivCoreTest, SmallDataBoundaryIsMeasuredInBytes)
    {
        RunScenario<uint8_t>(8, 8, 8, DATA_LIMIT);
        RunScenario<uint8_t>(8, 48, 8, DATA_LIMIT + 1);
        RunScenario<uint32_t>(8, 8, 8, DATA_LIMIT / sizeof(uint32_t));
        RunScenario<uint32_t>(8, 48, 8, DATA_LIMIT / sizeof(uint32_t) + 1);
        RunScenario<uint8_t>(4, 4, 4, DATA_LIMIT);
        RunScenario<uint8_t>(4, 48, 8, DATA_LIMIT + 1);
    }

    TEST(AlltoAllAivCoreTest, FewCorePathKeepsItsPeerCoverage)
    {
        RunScenario<uint8_t>(8, 4, 4, 1);
        RunScenario<uint8_t>(8, 4, 4, 512 * 1024 + 1);
        RunScenario<uint8_t>(16, 6, 6, 512 * 1024 + 1);
        RunScenario<uint8_t>(2, 1, 1, 512 * 1024 + 1);
    }

    TEST(AlltoAllAivCoreTest, IdleCoresSkipTransfersAndReachOuterBarriers)
    {
        RunScenario<uint8_t>(8, 4, 4, 512 * 1024 + 1, 32);
        RunScenario<uint8_t>(8, 8, 8, 512 * 1024 + 1, 32);
        RunScenario<uint8_t>(8, 8, 8, 1, 32);
        RunScenario<uint8_t>(4, 48, 8, 512 * 1024 + 1, 64);
        RunScenario<uint8_t>(4, 48, 4, 1, 64);
    }
} // namespace
} // namespace alltoall_aiv_test
