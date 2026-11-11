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
#include <string>
#include <utility>
#include <vector>
#include "aiv_all_reduce_mesh_1d_twoshot_write.h"

namespace allreduce_aiv_test {
namespace {
    constexpr uint64_t REDUCER_BYTES = 4 * 1024;
    using Flag = std::pair<uint32_t, uint64_t>;
    struct Region {
        uint64_t end;
        std::string expression;
    };

    // Symbolic intervals preserve the exact binary reduction tree without allocating every data element.
    class Memory {
    public:
        std::vector<std::pair<uint64_t, Region>> Read(uint64_t begin, uint64_t bytes) const
        {
            std::vector<std::pair<uint64_t, Region>> result;
            const uint64_t end = begin + bytes;
            while (begin < end) {
                auto it = regions_.upper_bound(begin);
                if (it == regions_.begin() || std::prev(it)->second.end <= begin) {
                    ADD_FAILURE() << "read from uninitialized memory at " << begin;
                    return {};
                }
                --it;
                const uint64_t next = std::min(end, it->second.end);
                result.push_back({begin, {next, it->second.expression}});
                begin = next;
            }
            return result;
        }

        void Insert(uint64_t begin, uint64_t bytes, const std::string& expression)
        {
            if (bytes == 0) {
                return;
            }
            auto next = regions_.lower_bound(begin);
            if (next != regions_.end()) {
                ASSERT_LE(begin + bytes, next->first) << "overlapping copy destination";
            }
            if (next != regions_.begin()) {
                ASSERT_LE(std::prev(next)->second.end, begin) << "overlapping copy destination";
            }
            regions_.emplace(begin, Region{begin + bytes, expression});
        }

        void Transfer(const Event& event)
        {
            const auto sources = Read(event.source, event.bytes);
            for (const auto& source : sources) {
                const uint64_t destination = event.destination + source.first - event.source;
                const uint64_t bytes = source.second.end - source.first;
                if (event.kind == EventKind::COPY) {
                    Insert(destination, bytes, source.second.expression);
                    continue;
                }
                const auto destinations = Read(destination, bytes);
                for (const auto& region : destinations) {
                    Split(region.first);
                    Split(region.second.end);
                    regions_.erase(regions_.lower_bound(region.first), regions_.lower_bound(region.second.end));
                    regions_.emplace(
                        region.first,
                        Region{
                            region.second.end, "(" + region.second.expression + "+" + source.second.expression + ")"});
                }
            }
        }

    private:
        void Split(uint64_t address)
        {
            auto it = regions_.upper_bound(address);
            if (it == regions_.begin()) {
                return;
            }
            --it;
            if (it->first == address || it->second.end <= address) {
                return;
            }
            const Region tail = it->second;
            it->second.end = address;
            regions_.emplace(address, tail);
        }

        std::map<uint64_t, Region> regions_;
    };

    std::string ExpectedTree(uint32_t ranks)
    {
        std::vector<std::string> slots;
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            slots.push_back(std::to_string(rank));
        }
        for (uint32_t active = ranks; active > 1;) {
            uint32_t split = 1;
            while (split * 2 < active) {
                split *= 2;
            }
            for (uint32_t source = split; source < active; ++source) {
                slots[source % split] = "(" + slots[source % split] + "+" + slots[source] + ")";
            }
            active = split;
        }
        return slots.front();
    }

    template <typename T>
    AivAllReduceMesh1DTwoShotWrite<T> Prepare(uint32_t ranks, uint32_t blocks, uint64_t count)
    {
        AivAllReduceMesh1DTwoShotWrite<T> op;
        op.rankSize_ = ranks;
        op.numBlocks_ = blocks;
        op.len_ = count;
        op.myGmIn_ = op.GetGmIn(0);
        op.Prepare(1);
        return op;
    }

    template <typename T>
    void RunScenario(uint32_t ranks, uint32_t blocks, uint64_t count, uint32_t physicalBlocks = 0)
    {
        SCOPED_TRACE(
            ::testing::Message() << "ranks=" << ranks << " blocks=" << blocks << " count=" << count
                                 << " elementBytes=" << sizeof(T));
        const uint64_t bytes = count * sizeof(T);
        const uint32_t launched = physicalBlocks == 0 ? blocks : physicalBlocks;
        const bool smallCore = blocks < 2 * ranks;
        uint64_t flagLimit = 3 * ranks;
        if (!smallCore) {
            const auto allocation = Prepare<T>(ranks, blocks, count);
            flagLimit = allocation.allGatherFlagOffset_ + allocation.sliceNum;
        }
        std::vector<Trace> traces;
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            for (uint32_t block = 0; block < launched; ++block) {
                traces.push_back(Trace{rank, block});
                currentTrace = &traces.back();
                AivAllReduceV2Mesh1DTwoShotWrite<T>(
                    nullptr, Address(INPUT_BASE, rank), Address(OUTPUT_BASE, rank), rank, 0, ranks, count, 0, 0, 0, 1,
                    bytes, bytes, 1, 0, 0, blocks, true, nullptr, nullptr, nullptr, 0, false, true);
                EXPECT_EQ(currentTrace->firstBarriers, 1u);
                EXPECT_EQ(currentTrace->endBarriers, 1u);
                const auto& events = currentTrace->events;
                const auto syncCount = std::count_if(events.begin(), events.end(), [](const Event& event) {
                    return event.kind == EventKind::SYNC_ALL;
                });
                EXPECT_EQ(syncCount, smallCore ? 1 : 0);
                if (block >= blocks) {
                    EXPECT_EQ(events.size(), smallCore ? 1u : 0u);
                }
            }
        }
        currentTrace = nullptr;

        Memory memory;
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            memory.Insert(Address(INPUT_BASE, rank), bytes, std::to_string(rank));
        }
        std::map<Flag, int32_t> flags;
        std::map<Flag, uint32_t> writers;
        std::vector<size_t> positions(traces.size(), 0);
        size_t remaining = 0;
        for (const auto& trace : traces) {
            remaining += trace.events.size();
        }
        while (remaining > 0) {
            bool progressed = false;
            for (size_t index = 0; index < traces.size(); ++index) {
                const auto& trace = traces[index];
                if (positions[index] == trace.events.size()) {
                    continue;
                }
                const auto& event = trace.events[positions[index]];
                if (event.kind == EventKind::SYNC_ALL) {
                    // Release the barrier only when every launched core on this rank has arrived.
                    const size_t first = static_cast<size_t>(trace.rank) * launched;
                    const size_t end = first + launched;
                    size_t peer = first;
                    for (; peer < end; ++peer) {
                        const auto& events = traces[peer].events;
                        if (positions[peer] == events.size() || events[positions[peer]].kind != EventKind::SYNC_ALL) {
                            break;
                        }
                    }
                    if (peer == end) {
                        for (size_t core = first; core < end; ++core) {
                            ++positions[core];
                        }
                        remaining -= launched;
                        progressed = true;
                    }
                    continue;
                }
                const Flag flag{event.target, event.slot};
                if (event.kind == EventKind::WAIT || event.kind == EventKind::RECORD) {
                    EXPECT_LT(event.target, ranks);
                    EXPECT_LT(event.slot, flagLimit);
                }
                if (event.kind == EventKind::WAIT) {
                    auto it = flags.find(flag);
                    if (it == flags.end() || it->second != event.tag) {
                        continue;
                    }
                } else if (event.kind == EventKind::RECORD) {
                    EXPECT_EQ(++writers[flag], 1u) << "multiple writers publish one flag";
                    flags[flag] = event.tag;
                } else {
                    if (event.destination >= OUTPUT_BASE) {
                        ASSERT_GE(event.destination, Address(OUTPUT_BASE, trace.rank));
                        ASSERT_LE(event.destination + event.bytes, Address(OUTPUT_BASE, trace.rank) + bytes)
                            << "copy writes past the output guard";
                    } else if (smallCore) {
                        const uint64_t localCcl = Address(CCL_BASE, trace.rank);
                        ASSERT_GE(event.destination, localCcl) << "few-core fallback writes data to a peer";
                        ASSERT_LE(event.destination + event.bytes, localCcl + RANK_STRIDE)
                            << "few-core fallback writes data to a peer";
                    }
                    memory.Transfer(event);
                }
                ++positions[index];
                --remaining;
                progressed = true;
            }
            ASSERT_TRUE(progressed) << "unmatched flags would deadlock the kernel";
        }
        const std::string expected = ExpectedTree(ranks);
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            for (const auto& region : memory.Read(Address(OUTPUT_BASE, rank), bytes)) {
                EXPECT_EQ(region.second.expression, expected) << "per-element reduction tree changed";
            }
        }
    }

    TEST(AllReduceAivCoreTest, DualProducersRequireSendBytesAndReducerWork)
    {
        for (uint32_t ranks : {2u, 3u, 5u, 8u}) {
            for (uint32_t groups : {4u, 5u, 6u}) {
                const uint64_t pieces = groups - 2;
                const uint64_t pivot = UB_DB_DATA_BATCH_SIZE / sizeof(float) * ranks * pieces;
                for (uint64_t count = pivot - ranks * pieces; count <= pivot + ranks * pieces; ++count) {
                    uint64_t minimum = count;
                    const uint64_t sliceNum = ranks * pieces;
                    // Enumerate the actual producer tasks, independently of the kernel's closed-form bound.
                    for (uint32_t destination = 0; destination < ranks; ++destination) {
                        for (uint64_t producer = 0; producer < 2; ++producer) {
                            uint64_t work = 0;
                            for (uint64_t piece = producer; piece < pieces; piece += 2) {
                                work += count / sliceNum + (destination * pieces + piece < count % sliceNum);
                            }
                            minimum = std::min(minimum, work);
                        }
                    }
                    const bool dual = minimum * sizeof(float) >= UB_DB_DATA_BATCH_SIZE
                                      && count / sliceNum / ranks * sizeof(float) >= REDUCER_BYTES;
                    const auto op = Prepare<float>(ranks, groups * ranks, count);
                    EXPECT_EQ(op.producerNum, (dual ? 2 : 1) * ranks);
                }
            }
        }
    }

    TEST(AllReduceAivCoreTest, ProducerThresholdUsesBytesAndPreservesConsumerBudget)
    {
        const uint64_t threshold = 2 * 8 * UB_DB_DATA_BATCH_SIZE;
        EXPECT_EQ(Prepare<uint8_t>(8, 48, threshold - 1).producerNum, 8u);
        EXPECT_EQ(Prepare<uint8_t>(8, 48, threshold).producerNum, 16u);
        EXPECT_EQ(Prepare<uint16_t>(8, 48, threshold / sizeof(uint16_t) - 1).producerNum, 8u);
        EXPECT_EQ(Prepare<uint16_t>(8, 48, threshold / sizeof(uint16_t)).producerNum, 16u);
        EXPECT_EQ(Prepare<uint64_t>(8, 48, threshold / sizeof(uint64_t) - 1).producerNum, 8u);
        EXPECT_EQ(Prepare<uint64_t>(8, 48, threshold / sizeof(uint64_t)).producerNum, 16u);
        EXPECT_EQ(Prepare<float>(8, 16, threshold).producerNum, 8u);
        EXPECT_EQ(Prepare<float>(8, 24, threshold).producerNum, 8u);
        // With 56 cores, two ranks can satisfy the send threshold while starving the remaining reducers.
        const uint64_t sendOnlyCount = UB_DB_DATA_BATCH_SIZE / (13 * sizeof(float)) + 1;
        EXPECT_EQ(Prepare<float>(2, 56, 2 * 26 * sendOnlyCount).producerNum, 2u);
        EXPECT_EQ(Prepare<float>(2, 56, 2 * 26 * 2 * REDUCER_BYTES / sizeof(float)).producerNum, 4u);
    }

    TEST(AllReduceAivCoreTest, OneReducerPublishesEachOriginalSliceWithoutFanIn)
    {
        for (uint64_t count : {0ULL, 24ULL, 262147ULL, 1048577ULL}) {
            auto op = Prepare<float>(8, 48, count);
            for (uint32_t rank : {0u, 7u}) {
                std::vector<uint64_t> publishers(op.groupSize, 0);
                std::vector<uint64_t> reducedBytes(op.groupSize, 0);
                std::vector<uint64_t> inputWaits(op.groupSize, 0);
                op.rank_ = rank;
                op.myGmIn_ = op.GetGmIn(rank);
                for (uint32_t block = op.producerNum; block < op.producerNum + op.consumerNum; ++block) {
                    Trace trace{rank, block};
                    currentTrace = &trace;
                    op.blockIdx_ = block;
                    op.Prepare(1);
                    op.ReduceScatter();
                    if (op.groupIdx != rank) {
                        EXPECT_TRUE(trace.events.empty());
                    }
                    for (const auto& event : trace.events) {
                        if (event.kind == EventKind::REDUCE) {
                            EXPECT_EQ(op.groupIdx, rank);
                            reducedBytes[op.idxInGroup] += event.bytes;
                        } else if (event.kind == EventKind::RECORD) {
                            const bool localFlag = event.slot >= op.localReduceFlagOffset
                                                   && event.slot < op.localReduceFlagOffset + op.groupSize;
                            EXPECT_TRUE(localFlag) << "unexpected intermediate completion flag";
                            if (localFlag) {
                                const uint64_t piece = event.slot - op.localReduceFlagOffset;
                                ++publishers[piece];
                                EXPECT_EQ(block, op.producerNum + rank * op.groupSize + piece);
                            }
                        } else if (event.kind == EventKind::WAIT) {
                            // Reduction waits only for RS input, never for another reducer's completion.
                            EXPECT_GE(event.slot, op.getRemoteFlagOffset);
                            EXPECT_LT(event.slot, op.localReduceFlagOffset);
                            ++inputWaits[op.idxInGroup];
                        }
                    }
                }
                currentTrace = nullptr;
                for (uint64_t piece = 0; piece < op.groupSize; ++piece) {
                    const uint64_t slice = rank * op.groupSize + piece;
                    const uint64_t elements = count / op.sliceNum + (slice < count % op.sliceNum);
                    EXPECT_EQ(publishers[piece], 1u);
                    EXPECT_EQ(reducedBytes[piece], elements * sizeof(float) * (op.rankSize_ - 1));
                    EXPECT_EQ(inputWaits[piece], elements > 0 ? op.rankSize_ : 0u);
                }
            }
        }
    }

    TEST(AllReduceAivCoreTest, OriginalSlicesCoverEveryElementInTheOriginalTree)
    {
        for (uint64_t count : {0ULL, 1ULL, 17ULL, 24ULL, 4099ULL}) {
            RunScenario<float>(8, 48, count);
            RunScenario<uint8_t>(3, 17, count);
        }
        // Retain uneven slice lengths around the former reduction-participation boundary.
        const uint64_t reductionBoundary = 40 * 2 * REDUCER_BYTES / sizeof(float);
        for (uint64_t count : {reductionBoundary - 1, reductionBoundary, reductionBoundary + 1}) {
            RunScenario<float>(8, 48, count);
        }
        const uint64_t producerBoundary = 2 * 8 * UB_DB_DATA_BATCH_SIZE / sizeof(float);
        for (uint64_t count : {producerBoundary - 1, producerBoundary, producerBoundary + 1}) {
            RunScenario<float>(8, 48, count);
        }
        RunScenario<uint64_t>(5, 24, 1048577);
        RunScenario<float>(16, 48, 1048577);
        RunScenario<float>(1, 8, 4099);
        RunScenario<float>(8, 16, 4099, 48);
        RunScenario<float>(8, 7, 4099);
    }

    TEST(AllReduceAivCoreTest, RankBudgetsCoverTwoByteElementsAndFewCoreFallback)
    {
        // Exercise the host's default write budgets, including the 18/19-rank fallback boundary.
        const uint32_t cases[][2] = {{1, 12}, {2, 24},  {3, 12},  {4, 12},  {5, 24},  {6, 24},  {7, 24}, {8, 24},
                                     {9, 36}, {16, 36}, {18, 36}, {19, 36}, {24, 36}, {32, 36}, {48, 36}};
        for (const auto& scenario : cases) {
            for (uint64_t count : {0ULL, 1ULL, 17ULL, 4099ULL, 262147ULL}) {
                RunScenario<uint16_t>(scenario[0], scenario[1], count);
            }
        }
    }

    TEST(AllReduceAivCoreTest, SmallCoreFallbackUsesPullWithExtraPhysicalCores)
    {
        // The 16-rank, 256-byte FP32 case previously exposed incorrect cross-rank writes.
        RunScenario<float>(16, 16, 64, 48);
        // Cover one active core, rankSize cores, and the limit just below the normal-path boundary.
        const uint32_t cases[][2] = {{1, 1}, {8, 1}, {8, 8}, {8, 15}, {19, 36}, {48, 36}};
        for (const auto& scenario : cases) {
            for (uint64_t count : {0ULL, 1ULL, 17ULL, 4099ULL}) {
                RunScenario<float>(scenario[0], scenario[1], count, 48);
            }
        }
    }

} // namespace
} // namespace allreduce_aiv_test
