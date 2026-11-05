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
#include <algorithm>
#include <iterator>
#include <map>
#include <numeric>
#include <utility>
#include <vector>
#include "aiv_all_gather_mesh_1d_write.h"

namespace all_gather_aiv_test {
namespace {
    enum class Layout { SEPARATE, OUTPUT_BASE, OWN_OUTPUT };
    using Flag = std::pair<uint32_t, uint64_t>;
    using Flags = std::map<Flag, int32_t>;
    struct Stats {
        uint64_t cachedBytes = 0;
        uint64_t directInputBytes = 0;
        uint64_t readDoneRecords = 0;
    };
    struct Region {
        uint64_t end;
        int source;
        uint64_t offset;
    };

    // 区间记录数据原始rank和字节偏移，避免为大数据测试实际分配R倍结果和CCL内存。
    // 覆盖input时同步修改来源，错误的再次读取会在最终output校验中暴露。
    class Memory {
    public:
        void Write(uint64_t begin, uint64_t bytes, int source, uint64_t offset)
        {
            if (bytes == 0) {
                return;
            }
            Split(begin);
            Split(begin + bytes);
            regions_.erase(regions_.lower_bound(begin), regions_.lower_bound(begin + bytes));
            regions_.emplace(begin, Region{begin + bytes, source, offset});
        }
        void Copy(uint64_t destination, uint64_t source, uint64_t bytes)
        {
            std::vector<std::pair<uint64_t, Region>> snapshot;
            for (uint64_t cursor = source; cursor < source + bytes;) {
                auto it = regions_.upper_bound(cursor);
                ASSERT_NE(it, regions_.begin()) << "read before initialization";
                --it;
                ASSERT_GT(it->second.end, cursor) << "read across an unwritten gap";
                ASSERT_GE(it->second.source, 0) << "read uninitialized CCL or padding";
                uint64_t end = std::min(source + bytes, it->second.end);
                snapshot.push_back(
                    {destination + cursor - source,
                     Region{destination + end - source, it->second.source, it->second.offset + cursor - it->first}});
                cursor = end;
            }
            for (const auto& part : snapshot) {
                Write(part.first, part.second.end - part.first, part.second.source, part.second.offset);
            }
        }
        void Check(uint64_t begin, uint64_t bytes, int source, uint64_t offset) const
        {
            for (uint64_t cursor = begin; cursor < begin + bytes;) {
                auto it = regions_.upper_bound(cursor);
                ASSERT_NE(it, regions_.begin());
                --it;
                ASSERT_GT(it->second.end, cursor);
                ASSERT_EQ(it->second.source, source) << "wrong source data or damaged guard";
                if (source >= 0) {
                    ASSERT_EQ(it->second.offset + cursor - it->first, offset + cursor - begin)
                        << "wrong input byte offset";
                }
                cursor = std::min(begin + bytes, it->second.end);
            }
        }

    private:
        void Split(uint64_t position)
        {
            auto it = regions_.upper_bound(position);
            if (it == regions_.begin()) {
                return;
            }
            --it;
            if (it->first < position && position < it->second.end) {
                Region right{it->second.end, it->second.source, it->second.offset + position - it->first};
                it->second.end = position;
                regions_.emplace(position, right);
            }
        }
        std::map<uint64_t, Region> regions_;
    };

    template <typename T>
    void RunScenario(
        uint32_t ranks, uint32_t blocks, uint64_t count, Layout layout, Stats& stats, Flags& flags,
        uint32_t sliceId = 1)
    {
        SCOPED_TRACE(
            ::testing::Message() << "ranks=" << ranks << " blocks=" << blocks << " count=" << count
                                 << " typeBytes=" << sizeof(T) << " layout=" << static_cast<int>(layout));
        const uint64_t bytes = count * sizeof(T);
        const uint64_t stride = bytes + 64;
        ASSERT_GE(bytes, DATA_LIMIT);
        ASSERT_GE(blocks, 2 * ranks);
        std::vector<Trace> traces;
        traces.reserve(ranks * blocks);
        std::vector<uint64_t> inputs(ranks);
        uint64_t producerCount = 0;
        uint64_t pieces = 0;
        Memory memory;
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            const uint64_t output = Address(OUTPUT_BASE, rank);
            inputs[rank] = layout == Layout::OUTPUT_BASE ? output :
                           layout == Layout::OWN_OUTPUT  ? output + rank * stride :
                                                           Address(INPUT_BASE, rank);
            memory.Write(output - 64, ranks * stride + 128, -1, 0);
            memory.Write(inputs[rank], bytes, static_cast<int>(rank), 0);
            memory.Write(Address(CCL_BASE, rank), ranks * bytes, -1, 0);
            for (uint32_t block = 0; block < blocks; ++block) {
                traces.push_back({rank, block, {}});
                currentTrace = &traces.back();
                AivAllGatherMesh1DWrite<T> op;
                op.input_ = inputs[rank];
                op.output_ = output;
                op.rank_ = rank;
                op.rankSize_ = ranks;
                op.numBlocks_ = blocks;
                op.blockIdx_ = block;
                op.myGmIn_ = op.GetGmIn(rank);
                op.Process(count, sliceId, stride);
                producerCount = op.coreNumStage1;
                pieces = op.coreNumStage2;
            }
        }
        currentTrace = nullptr;
        ASSERT_LE((ranks + 1) * pieces * FLAG_SIZE, FLAG2_OFFSET - FLAG1_OFFSET);

        // 优先推进rank0 producer，其次consumer，最后其他producer。
        // 刻意让rank0数据先到，检查远端consumer不能覆盖尚未保存的本地input。
        std::vector<size_t> order(traces.size());
        std::iota(order.begin(), order.end(), 0);
        auto priority = [&](size_t index) {
            const auto& trace = traces[index];
            return trace.rank == 0 && trace.block < producerCount ? 0 : trace.block >= producerCount ? 1 : 2;
        };
        std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
            return priority(left) < priority(right);
        });
        std::vector<size_t> positions(traces.size(), 0);
        std::map<Flag, uint32_t> writers;
        std::map<uint64_t, uint32_t> ubStores;
        size_t remaining = 0;
        for (const auto& trace : traces) {
            remaining += trace.events.size();
        }
        while (remaining != 0) {
            bool progressed = false;
            for (size_t index : order) {
                const auto& trace = traces[index];
                if (positions[index] == trace.events.size()) {
                    continue;
                }
                const auto& event = trace.events[positions[index]];
                const Flag flag{event.target, event.slot};
                if (event.kind == EventKind::WAIT && flags[flag] != event.tag) {
                    continue;
                }
                ASSERT_NE(event.kind, EventKind::SYNC) << "large path adds a global core synchronization";
                if (event.kind == EventKind::RECORD) {
                    ASSERT_LT(event.target, ranks);
                    ASSERT_LT(event.slot, (ranks + 1) * pieces);
                    ASSERT_EQ(++writers[flag], 1u) << "flag reused by different pieces";
                    flags[flag] = event.tag;
                    if (event.slot >= ranks * pieces) {
                        ASSERT_EQ(layout, Layout::OUTPUT_BASE);
                        ASSERT_NE(trace.rank, 0u);
                        ASSERT_EQ(event.target, trace.rank);
                        ++stats.readDoneRecords;
                    }
                } else if (event.kind == EventKind::ALLOC) {
                    ASSERT_EQ(ubStores.count(event.destination), 0u);
                    ubStores[event.destination] = 0;
                } else if (event.kind == EventKind::FREE) {
                    ASSERT_EQ(ubStores.count(event.destination), 1u);
                    ASSERT_EQ(ubStores[event.destination], ranks) << "payload released before all targets use it";
                    ubStores.erase(event.destination);
                } else if (event.kind != EventKind::WAIT) {
                    if (event.kind == EventKind::LOAD) {
                        ASSERT_EQ(ubStores.count(event.destination), 1u);
                        stats.cachedBytes += event.bytes;
                    } else if (event.kind == EventKind::STORE) {
                        ASSERT_EQ(ubStores.count(event.source), 1u) << "payload read after FreeTensor";
                        ++ubStores[event.source];
                    } else if (event.source >= inputs[trace.rank] && event.source < inputs[trace.rank] + bytes) {
                        stats.directInputBytes += event.bytes;
                    }
                    if (layout == Layout::OWN_OUTPUT || (layout == Layout::OUTPUT_BASE && trace.rank == 0)) {
                        uint64_t ownOutput = Address(OUTPUT_BASE, trace.rank) + trace.rank * stride;
                        ASSERT_FALSE(event.destination >= ownOutput && event.destination < ownOutput + bytes)
                            << "standard in-place rewrites the input slot";
                    }
                    ASSERT_NO_FATAL_FAILURE(memory.Copy(event.destination, event.source, event.bytes));
                }
                ++positions[index];
                --remaining;
                progressed = true;
                break;
            }
            ASSERT_TRUE(progressed) << "unmatched data/READ_DONE dependencies deadlock";
        }
        EXPECT_TRUE(ubStores.empty());
        EXPECT_EQ(stats.cachedBytes + stats.directInputBytes / ranks, ranks * bytes);
        EXPECT_EQ(stats.directInputBytes % ranks, 0u);
        EXPECT_EQ(stats.readDoneRecords, layout == Layout::OUTPUT_BASE ? (ranks - 1) * pieces : 0u);
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            uint64_t output = Address(OUTPUT_BASE, rank);
            ASSERT_NO_FATAL_FAILURE(memory.Check(output - 64, 64, -1, 0));
            for (uint32_t source = 0; source < ranks; ++source) {
                ASSERT_NO_FATAL_FAILURE(memory.Check(output + source * stride, bytes, static_cast<int>(source), 0));
                ASSERT_NO_FATAL_FAILURE(memory.Check(output + source * stride + bytes, 64, -1, 0));
            }
            ASSERT_NO_FATAL_FAILURE(memory.Check(output + ranks * stride, 64, -1, 0));
        }
    }

    TEST(AllGatherAivCoreTest, CachedPiecesReadInputOnceAndKeepPayloadUntilEveryTargetCompletes)
    {
        const uint32_t scenarios[][2] = {{2, 12}, {3, 12}, {4, 16}, {7, 14}, {7, 42}};
        for (const auto& scenario : scenarios) {
            Stats stats;
            Flags flags;
            ASSERT_NO_FATAL_FAILURE(
                RunScenario<uint8_t>(scenario[0], scenario[1], DATA_LIMIT + 3, Layout::SEPARATE, stats, flags));
            EXPECT_EQ(stats.cachedBytes, scenario[0] * (DATA_LIMIT + 3));
            EXPECT_EQ(stats.directInputBytes, 0u);
        }
    }

    template <typename T>
    void CheckBoundaryAndAliases()
    {
        // 2rank、12核使用8个逻辑片，精确构造95KiB边界及只多一个元素的混合快/慢路径。
        const uint64_t boundary = 8 * UB_DB_DATA_BATCH_SIZE / sizeof(T);
        for (uint64_t count : {boundary - 1, boundary, boundary + 1, 2 * boundary + 1}) {
            for (Layout layout : {Layout::SEPARATE, Layout::OUTPUT_BASE, Layout::OWN_OUTPUT}) {
                Stats stats;
                Flags flags;
                ASSERT_NO_FATAL_FAILURE(RunScenario<T>(2, 12, count, layout, stats, flags));
                if (count <= boundary) {
                    EXPECT_EQ(stats.directInputBytes, 0u);
                } else if (count == boundary + 1) {
                    EXPECT_GT(stats.cachedBytes, 0u);
                    EXPECT_GT(stats.directInputBytes, 0u);
                } else {
                    EXPECT_EQ(stats.cachedBytes, 0u);
                    EXPECT_EQ(stats.directInputBytes, 4 * count * sizeof(T));
                }
            }
        }
    }

    TEST(AllGatherAivCoreTest, BoundaryAndMultiChunkFallbackProtectBothAliasLayouts)
    {
        ASSERT_NO_FATAL_FAILURE(CheckBoundaryAndAliases<uint8_t>());
        ASSERT_NO_FATAL_FAILURE(CheckBoundaryAndAliases<uint32_t>());
        ASSERT_NO_FATAL_FAILURE(CheckBoundaryAndAliases<uint64_t>());
    }

    TEST(AllGatherAivCoreTest, MultipleProducerPiecesUseTheSameFlagIndexAsConsumers)
    {
        for (Layout layout : {Layout::OUTPUT_BASE, Layout::OWN_OUTPUT}) {
            Stats stats;
            Flags flags;
            ASSERT_NO_FATAL_FAILURE(
                RunScenario<uint32_t>(7, 42, DATA_LIMIT / sizeof(uint32_t) + 3, layout, stats, flags));
        }
    }

    template <typename T = uint32_t>
    void CheckCtrlCore(uint32_t ranks, uint32_t blocks, Layout layout, uint64_t overlapOffset = 0, uint64_t count = 17)
    {
        SCOPED_TRACE(
            ::testing::Message() << "ranks=" << ranks << " blocks=" << blocks << " count=" << count
                                 << " typeBytes=" << sizeof(T) << " layout=" << static_cast<int>(layout));
        const uint64_t bytes = count * sizeof(T);
        const uint64_t stride = bytes + 64;
        const bool splitTxRx = ranks == 8 && bytes >= DATA_LIMIT && blocks > 2 * ranks;
        const uint64_t consumedFlagOffset = splitTxRx ? 2 * ranks : ranks;
        const uint64_t readyFlagOffset = consumedFlagOffset + ranks;
        ASSERT_LE((readyFlagOffset + ranks) * FLAG_SIZE, FLAG2_OFFSET - FLAG1_OFFSET);
        std::vector<Trace> traces;
        Memory memory;
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            const uint64_t output = Address(OUTPUT_BASE, rank);
            const uint64_t ownOutput = output + rank * stride;
            const uint64_t input = layout == Layout::OWN_OUTPUT  ? ownOutput :
                                   layout == Layout::OUTPUT_BASE ? output + overlapOffset :
                                                                   Address(INPUT_BASE, rank);
            const bool expectBypass = layout == Layout::SEPARATE || input == ownOutput;
            const uint64_t ownCcl = Address(CCL_BASE, rank) + rank * bytes;
            memory.Write(output - 64, ranks * stride + 128, -1, 0);
            memory.Write(Address(CCL_BASE, rank), ranks * bytes, -1, 0);
            memory.Write(input, bytes, static_cast<int>(rank), 0);
            uint64_t selfStoreBytes = 0;
            uint64_t selfCopyBytes = 0;
            for (uint32_t block = 0; block < blocks; ++block) {
                Trace trace{rank, block, {}};
                AivAllGatherMesh1DWrite<T> op;
                op.rank_ = rank;
                op.rankSize_ = ranks;
                op.numBlocks_ = blocks;
                op.blockIdx_ = block;
                op.input_ = input;
                op.output_ = output;
                op.myGmIn_ = op.GetGmIn(rank);
                currentTrace = &trace;
                op.Process(count, 1, stride);
                currentTrace = nullptr;
                uint32_t syncs = 0;
                for (const auto& event : trace.events) {
                    syncs += event.kind == EventKind::SYNC;
                    const bool sendCore = block < std::min(blocks, ranks);
                    const bool copyCore = splitTxRx ? block >= ranks && block < 2 * ranks : sendCore;
                    if (event.kind == EventKind::COPY) {
                        if (event.destination >= ownCcl && event.destination < ownCcl + bytes) {
                            selfStoreBytes += event.bytes;
                        }
                        if (event.destination >= ownOutput && event.destination < ownOutput + bytes) {
                            selfCopyBytes += event.bytes;
                        }
                        if (event.destination >= Address(CCL_BASE, 0) && event.destination < Address(CCL_BASE, ranks)) {
                            EXPECT_TRUE(sendCore) << "CCL push assigned to a receive or idle core";
                        } else {
                            EXPECT_TRUE(copyCore) << "output copy assigned to a send or idle core";
                            EXPECT_GE(event.destination, output);
                            EXPECT_LE(event.destination + event.bytes, output + (ranks - 1) * stride + bytes);
                        }
                    }
                    if (event.kind == EventKind::WAIT || event.kind == EventKind::RECORD) {
                        EXPECT_LT(event.target, ranks);
                        EXPECT_LT(event.slot, readyFlagOffset + ranks);
                        const bool dataFlag = event.slot < consumedFlagOffset;
                        const bool sendEvent = event.kind == EventKind::RECORD ? dataFlag : !dataFlag;
                        EXPECT_TRUE(sendEvent ? sendCore : copyCore);
                    }
                    if (expectBypass && (event.kind == EventKind::WAIT || event.kind == EventKind::RECORD)) {
                        EXPECT_FALSE(event.target == rank && event.slot % ranks == rank);
                    }
                }
                EXPECT_EQ(syncs, expectBypass ? 1u : 2u);
                traces.push_back(std::move(trace));
            }
            EXPECT_EQ(selfStoreBytes, expectBypass ? 0u : bytes);
            EXPECT_EQ(selfCopyBytes, input == ownOutput ? 0u : bytes);
        }

        Flags flags;
        std::map<Flag, uint32_t> writers;
        std::vector<size_t> positions(traces.size());
        size_t finished = 0;
        while (finished != traces.size()) {
            bool progressed = false;
            finished = 0;
            for (size_t i = 0; i < traces.size(); ++i) {
                const auto& trace = traces[i];
                if (positions[i] == trace.events.size()) {
                    ++finished;
                    continue;
                }
                const auto& event = trace.events[positions[i]];
                const Flag flag{event.target, event.slot};
                if (event.kind == EventKind::WAIT && flags[flag] != event.tag) {
                    continue;
                }
                if (event.kind == EventKind::SYNC) {
                    const size_t first = trace.rank * blocks;
                    size_t peer = first;
                    for (; peer < first + blocks; ++peer) {
                        if (positions[peer] == traces[peer].events.size()
                            || traces[peer].events[positions[peer]].kind != EventKind::SYNC) {
                            break;
                        }
                    }
                    if (peer != first + blocks) {
                        continue;
                    }
                    for (peer = first; peer < first + blocks; ++peer) {
                        ++positions[peer];
                    }
                } else {
                    if (event.kind == EventKind::RECORD) {
                        EXPECT_EQ(++writers[flag], 1u) << "control-core flag has multiple writers";
                        flags[flag] = event.tag;
                    } else if (event.kind == EventKind::COPY) {
                        ASSERT_NO_FATAL_FAILURE(memory.Copy(event.destination, event.source, event.bytes));
                    }
                    ++positions[i];
                }
                progressed = true;
                break;
            }
            ASSERT_TRUE(progressed || finished == traces.size()) << "control-core handshake stalled";
        }
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            const uint64_t output = Address(OUTPUT_BASE, rank);
            ASSERT_NO_FATAL_FAILURE(memory.Check(output - 64, 64, -1, 0));
            for (uint32_t source = 0; source < ranks; ++source) {
                ASSERT_NO_FATAL_FAILURE(
                    memory.Check(Address(OUTPUT_BASE, rank) + source * stride, bytes, static_cast<int>(source), 0));
                if (layout != Layout::OUTPUT_BASE || overlapOffset == 0) {
                    ASSERT_NO_FATAL_FAILURE(memory.Check(output + source * stride + bytes, 64, -1, 0));
                }
            }
            ASSERT_NO_FATAL_FAILURE(memory.Check(output + ranks * stride, 64, -1, 0));
        }
    }

    TEST(AllGatherAivCoreTest, CtrlCoreBypassesSelfAndPipelinesOnlySafeLayouts)
    {
        for (uint32_t blocks : {1u, 3u, 12u}) {
            for (Layout layout : {Layout::SEPARATE, Layout::OWN_OUTPUT, Layout::OUTPUT_BASE}) {
                ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(8, blocks, layout));
            }
            ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(8, blocks, Layout::OUTPUT_BASE, sizeof(uint32_t)));
        }
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(1, 1, Layout::SEPARATE));
    }

    TEST(AllGatherAivCoreTest, EightOrMoreRanksUseCtrlCoreForLargeInput)
    {
        constexpr uint64_t COUNT = DATA_LIMIT / sizeof(uint32_t) + 1;
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(8, 16, Layout::SEPARATE, 0, COUNT));
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(8, 48, Layout::OWN_OUTPUT, 0, COUNT));
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(9, 18, Layout::OUTPUT_BASE, 0, COUNT));
    }

    TEST(AllGatherAivCoreTest, CtrlCoreSplitsSendAndCopyOnlyAboveTwiceTheRankCount)
    {
        constexpr uint64_t COUNT = DATA_LIMIT / sizeof(uint32_t) + 1;
        for (uint32_t blocks : {12u, 15u, 16u, 17u, 24u}) {
            for (Layout layout : {Layout::SEPARATE, Layout::OWN_OUTPUT, Layout::OUTPUT_BASE}) {
                ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(8, blocks, layout, 0, COUNT));
            }
        }
        for (uint32_t blocks : {24u, 32u, 33u, 48u}) {
            ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(16, blocks, Layout::SEPARATE));
            ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(16, blocks, Layout::OUTPUT_BASE, sizeof(uint32_t)));
        }
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(1, 3, Layout::SEPARATE));
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(2, 24, Layout::SEPARATE));
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(4, 12, Layout::OWN_OUTPUT));
    }

    TEST(AllGatherAivCoreTest, CtrlCoreSplitRequiresEightRanksAndLargeInput)
    {
        constexpr uint64_t BOUNDARY = DATA_LIMIT / sizeof(uint32_t);
        for (uint32_t ranks : {8u, 9u, 16u}) {
            for (uint64_t count : {BOUNDARY - 1, BOUNDARY, BOUNDARY + 1}) {
                for (Layout layout : {Layout::SEPARATE, Layout::OWN_OUTPUT, Layout::OUTPUT_BASE}) {
                    ASSERT_NO_FATAL_FAILURE(CheckCtrlCore(ranks, 48, layout, 0, count));
                }
            }
        }
    }

    template <typename T>
    void CheckCtrlCoreSplitSizes()
    {
        constexpr uint64_t BOUNDARY = DATA_LIMIT / sizeof(T);
        const uint64_t counts[] = {0,
                                   1,
                                   2,
                                   3,
                                   17,
                                   18,
                                   UB_MAX_DATA_SIZE / sizeof(T) + 1,
                                   BOUNDARY - 1,
                                   BOUNDARY,
                                   BOUNDARY + 1,
                                   2 * BOUNDARY + 1};
        for (uint64_t count : counts) {
            for (Layout layout : {Layout::SEPARATE, Layout::OWN_OUTPUT, Layout::OUTPUT_BASE}) {
                ASSERT_NO_FATAL_FAILURE(CheckCtrlCore<T>(8, 24, layout, 0, count));
            }
            ASSERT_NO_FATAL_FAILURE(CheckCtrlCore<T>(8, 24, Layout::OUTPUT_BASE, sizeof(T), count));
        }
    }

    TEST(AllGatherAivCoreTest, CtrlCoreSplitCopiesKeepOddTailsAndProtectOverlappingInput)
    {
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCoreSplitSizes<uint8_t>());
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCoreSplitSizes<uint32_t>());
        ASSERT_NO_FATAL_FAILURE(CheckCtrlCoreSplitSizes<uint64_t>());
    }

    TEST(AllGatherAivCoreTest, PreviousSliceFlagsCannotReleaseCurrentSliceConsumers)
    {
        Flags flags;
        for (uint32_t slice : {1u, 2u, 3u, 65535u}) {
            Stats stats;
            ASSERT_NO_FATAL_FAILURE(
                RunScenario<uint8_t>(4, 24, DATA_LIMIT + 7, Layout::OUTPUT_BASE, stats, flags, slice));
        }
    }
} // namespace
} // namespace all_gather_aiv_test
