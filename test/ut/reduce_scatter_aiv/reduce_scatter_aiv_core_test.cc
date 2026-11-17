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
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include "aiv_reduce_scatter_local_tree_write.h"
#include "aiv_reduce_scatter_mesh_1d_bigdata_write.h"

namespace reduce_scatter_aiv_test {
namespace {

    // {持有flag的rank, flag槽位, tag}。同一槽位会在不同轮次以不同tag重复使用，故tag必须参与配对。
    using FlagKey = std::tuple<uint32_t, uint64_t, int32_t>;

    enum class KernelKind { LOCAL_TREE, BIG_DATA };

    struct Config {
        uint32_t ranks = 1;
        uint32_t blocks = 1;
        uint32_t launched = 0; // 实际启动核数, 0表示等于blocks
        uint64_t count = 1;    // 每rank分片元素个数
    };

    struct Scenario {
        Config cfg;
        uint64_t lenBytes = 0;
        uint64_t stride = 0;
        uint32_t launched = 0;
        bool pingpong = false;
    };

    Scenario MakeScenario(const Config& cfg, uint64_t elementBytes)
    {
        Scenario s;
        s.cfg = cfg;
        s.lenBytes = cfg.count * elementBytes;
        // 各peer分片之间留有空隙, 用于验证kernel使用用户传入的输入stride。
        s.stride = s.lenBytes + 64;
        s.launched = cfg.launched == 0 ? cfg.blocks : cfg.launched;
        s.pingpong = s.lenBytes <= DATA_LIMIT && cfg.ranks * s.lenBytes <= PINGPONG_TOTAL_DATA_LIMIT;
        return s;
    }

    // 在CPU上逐rank、逐核执行生产kernel入口，收集每个核的事件序列。
    template <typename T>
    std::vector<Trace> RunScenario(KernelKind kind, const Scenario& s)
    {
        std::vector<Trace> traces;
        traces.reserve(s.cfg.ranks * s.launched);
        for (uint32_t rank = 0; rank < s.cfg.ranks; ++rank) {
            for (uint32_t block = 0; block < s.launched; ++block) {
                traces.push_back(Trace{rank, block});
                currentTrace = &traces.back();
                if (kind == KernelKind::LOCAL_TREE) {
                    AivReduceScatterV2LocalTreeWrite<T>(
                        nullptr, Address(INPUT_BASE, rank), Address(OUTPUT_BASE, rank), rank, 0, s.cfg.ranks,
                        s.cfg.count, 0, 0, 0, 1, s.stride, s.stride, 1, 0, 0, s.cfg.blocks, true, nullptr, nullptr,
                        nullptr, 0, false, true);
                } else {
                    AivReduceScatterV2Mesh1DBigDataWrite<T>(
                        nullptr, Address(INPUT_BASE, rank), Address(OUTPUT_BASE, rank), rank, 0, s.cfg.ranks,
                        s.cfg.count, 0, 0, 0, 1, s.stride, s.stride, 1, 0, 0, s.cfg.blocks, true, nullptr, nullptr,
                        nullptr, 0, false, true);
                }
            }
        }
        currentTrace = nullptr;
        return traces;
    }

    // 按依赖顺序推进所有核的事件流: 只有{目标rank, 槽位, tag}已被记录过的WaitFlag才放行, 借此识别死锁与配对缺失。
    void ResolveFlagDependencies(
        const std::vector<Trace>& traces, uint32_t ranks, std::map<FlagKey, uint32_t>& writers,
        std::map<FlagKey, uint32_t>& readers)
    {
        std::set<FlagKey> recorded;
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
                const FlagKey key{event.target, event.slot, event.tag};
                if (event.kind == EventKind::WAIT) {
                    if (recorded.count(key) == 0) {
                        continue;
                    }
                    ++readers[key];
                } else if (event.kind == EventKind::RECORD) {
                    EXPECT_LT(event.target, ranks) << "record targets a rank outside the group";
                    EXPECT_EQ(++writers[key], 1u) << "the same {rank, slot, tag} flag is published twice";
                    recorded.insert(key);
                }
                ++positions[index];
                --remaining;
                progressed = true;
            }
            ASSERT_TRUE(progressed) << "unmatched flag dependencies would deadlock the kernel";
        }
    }

    // 每个rank的output必须被本次kernel完整且不重不漏地覆盖。
    void CheckOutputTiling(const std::vector<Trace>& traces, uint32_t ranks, uint64_t lenBytes)
    {
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            const uint64_t base = Address(OUTPUT_BASE, rank);
            std::vector<std::pair<uint64_t, uint64_t>> ranges;
            for (const auto& trace : traces) {
                if (trace.rank != rank) {
                    continue;
                }
                for (const auto& event : trace.events) {
                    if (event.kind != EventKind::COPY || event.bytes == 0) {
                        continue;
                    }
                    if (event.destination < base || event.destination >= base + lenBytes) {
                        continue;
                    }
                    EXPECT_LE(event.destination + event.bytes, base + lenBytes) << "output write exceeds its shard";
                    ranges.emplace_back(event.destination - base, event.destination - base + event.bytes);
                }
            }
            std::sort(ranges.begin(), ranges.end());
            ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());
            uint64_t expect = 0;
            for (const auto& range : ranges) {
                EXPECT_EQ(range.first, expect) << "rank[" << rank << "] output has a gap or overlap";
                expect = range.second;
            }
            EXPECT_EQ(expect, lenBytes) << "rank[" << rank << "] output is not fully written";
        }
    }

    // 每次拷贝的源必须落在本rank的输入分片或某个rank的ccl缓冲内, 目的必须落在某个rank的ccl缓冲或本rank输出内。
    void CheckCopyRegions(const std::vector<Trace>& traces, const Scenario& s, uint64_t cclSpanBytes)
    {
        const auto& cfg = s.cfg;
        const uint64_t cclBegin = Address(CCL_BASE, 0);
        const uint64_t cclEnd = Address(CCL_BASE, cfg.ranks - 1) + cclSpanBytes;
        for (const auto& trace : traces) {
            const uint64_t myInput = Address(INPUT_BASE, trace.rank);
            const uint64_t myOutput = Address(OUTPUT_BASE, trace.rank);
            for (const auto& event : trace.events) {
                if (event.kind != EventKind::COPY) {
                    continue;
                }
                const bool fromInput = event.source >= myInput && event.source < myInput + cfg.ranks * s.stride;
                const bool fromCcl = event.source >= cclBegin && event.source < cclEnd;
                EXPECT_TRUE(fromInput || fromCcl)
                    << "copy from an address outside the rank input/ccl buffers: " << event.source;
                const bool toCcl = event.destination >= cclBegin && event.destination < cclEnd;
                const bool toOutput = event.destination >= myOutput && event.destination < myOutput + s.lenBytes;
                EXPECT_TRUE(toCcl || toOutput)
                    << "copy to an address outside the ccl buffers/output: " << event.destination;
            }
        }
    }

    // LocalTreeWrite的写语义入口: 逐peer直写前必须先做一次逐对端presync握手, 数据落地后再发布数据flag。
    void CheckLocalTreePush(const Trace& trace, const Scenario& s)
    {
        const uint32_t ranks = s.cfg.ranks;
        const uint32_t rank = trace.rank;
        const uint32_t peer = trace.block;
        ASSERT_GE(trace.events.size(), 5u);
        EXPECT_EQ(trace.events[0].kind, EventKind::RECORD);
        EXPECT_EQ(trace.events[0].target, peer);
        EXPECT_EQ(trace.events[0].slot, ranks + rank); // PreSyncFlagOffset(rank_)
        EXPECT_EQ(trace.events[1].kind, EventKind::WAIT);
        EXPECT_EQ(trace.events[1].target, rank);
        EXPECT_EQ(trace.events[1].slot, ranks + peer); // PreSyncFlagOffset(peerRank)
        EXPECT_EQ(trace.events[2].kind, EventKind::COPY);
        EXPECT_EQ(trace.events[2].source, Address(INPUT_BASE, rank) + static_cast<uint64_t>(peer) * s.stride);
        EXPECT_EQ(trace.events[2].destination, Address(CCL_BASE, peer) + (ranks + rank) * s.lenBytes);
        EXPECT_EQ(trace.events[2].bytes, s.lenBytes);
        EXPECT_EQ(trace.events[3].kind, EventKind::RECORD);
        EXPECT_EQ(trace.events[3].target, peer);
        EXPECT_EQ(trace.events[3].slot, rank); // 数据flag按源rank编号
        EXPECT_EQ(trace.events[4].kind, EventKind::WAIT);
        EXPECT_EQ(trace.events[4].target, rank);
        EXPECT_EQ(trace.events[4].slot, peer);
    }

    // 分片布局: cclBuf前R*len为原fetch区, [R*len, 2R*len)为各源rank的归约暂存区。
    void CheckLocalTree(const std::vector<Trace>& traces, const Scenario& s)
    {
        const uint32_t ranks = s.cfg.ranks;
        const bool enoughCore = s.cfg.blocks >= ranks;
        for (const auto& trace : traces) {
            EXPECT_EQ(trace.firstBarriers, 1u);
            EXPECT_EQ(trace.syncAlls, enoughCore ? 1u : 0u);
            EXPECT_EQ(trace.endBarriers, s.pingpong ? 0u : 1u);
            if (!enoughCore) {
                EXPECT_TRUE(trace.events.empty()) << "core copies without the minimum core count";
                continue;
            }
            if (trace.block >= ranks) {
                EXPECT_TRUE(trace.events.empty()) << "idle core copies data or touches flags";
                continue;
            }
            CheckLocalTreePush(trace, s);
        }
        CheckCopyRegions(traces, s, DOUBLE * ranks * s.lenBytes);
        if (enoughCore) {
            CheckOutputTiling(traces, ranks, s.lenBytes);
        }
    }

    // 核数预算与生产侧一致: stage1负责直写对端, stage2在本地做树形归约, 超出预算的核保持静默。
    struct BigDataBudget {
        uint64_t stageCoreNum = 0;
        uint64_t stage1CoreNum = 0;
        uint64_t stage2CoreNum = 0;
        uint64_t cutNum = 0;
    };

    BigDataBudget CalcBigDataBudget(uint32_t blocks, uint32_t ranks, uint64_t len)
    {
        BigDataBudget budget;
        budget.stageCoreNum = static_cast<uint64_t>(blocks) / ranks * ranks;
        if (budget.stageCoreNum < 2ULL * ranks) {
            return budget;
        }
        budget.stage1CoreNum = ranks;
        const uint64_t candidateProducerNum = 2ULL * ranks;
        if (budget.stageCoreNum >= candidateProducerNum + ranks) {
            const uint64_t candidateCutNum = (budget.stageCoreNum - candidateProducerNum) / ranks;
            const uint64_t nonEmptyPieceNum = std::min<uint64_t>(len, candidateCutNum);
            // 只有非空分片足以铺满两个producer时, 才把stage1核数翻倍。
            if (nonEmptyPieceNum >= 2ULL) {
                budget.stage1CoreNum = candidateProducerNum;
            }
        }
        budget.stage2CoreNum = budget.stageCoreNum - budget.stage1CoreNum;
        budget.cutNum = budget.stage2CoreNum / ranks;
        return budget;
    }

    void CheckBigData(const std::vector<Trace>& traces, const Scenario& s)
    {
        const uint32_t ranks = s.cfg.ranks;
        const BigDataBudget budget = CalcBigDataBudget(s.cfg.blocks, ranks, s.cfg.count);
        const bool enoughCore = budget.stageCoreNum >= 2ULL * ranks;
        const uint64_t cclStride = s.lenBytes * ranks;
        for (const auto& trace : traces) {
            EXPECT_EQ(trace.firstBarriers, 1u);
            EXPECT_EQ(trace.endBarriers, 1u);
            if (!enoughCore || trace.block >= budget.stageCoreNum) {
                EXPECT_TRUE(trace.events.empty()) << "core outside the budget copies data or touches flags";
                continue;
            }
            const bool producer = trace.block < budget.stage1CoreNum;
            const uint64_t peer = trace.block % ranks;
            for (const auto& event : trace.events) {
                if (event.kind == EventKind::COPY) {
                    if (producer) {
                        // stage1: 从本rank输入分片直写到对端cclOut区 [R*len, 2R*len)。
                        EXPECT_GE(event.destination, Address(CCL_BASE, peer) + cclStride);
                        EXPECT_LE(event.destination + event.bytes, Address(CCL_BASE, peer) + 2 * cclStride);
                        EXPECT_LE(
                            event.source + event.bytes, Address(INPUT_BASE, trace.rank) + peer * s.stride + s.lenBytes);
                        EXPECT_GT(event.bytes, 0u);
                    } else {
                        // stage2: 只读写本地cclOut区。
                        EXPECT_GE(event.source, Address(CCL_BASE, trace.rank) + cclStride);
                        EXPECT_LE(event.source + event.bytes, Address(CCL_BASE, trace.rank) + 2 * cclStride);
                    }
                } else {
                    EXPECT_LT(event.slot, budget.stage2CoreNum) << "flag slot outside the data-ready area";
                }
            }
        }
        CheckCopyRegions(traces, s, 2 * ranks * s.lenBytes);
        if (enoughCore) {
            CheckOutputTiling(traces, ranks, s.lenBytes);
        }
    }

    template <typename T>
    void Validate(KernelKind kind, const Config& config)
    {
        const Scenario s = MakeScenario(config, sizeof(T));
        SCOPED_TRACE(
            ::testing::Message() << "ranks=" << config.ranks << " blocks=" << config.blocks << " launched="
                                 << s.launched << " count=" << config.count << " elementBytes=" << sizeof(T));
        const auto traces = RunScenario<T>(kind, s);
        if (kind == KernelKind::LOCAL_TREE) {
            CheckLocalTree(traces, s);
        } else {
            CheckBigData(traces, s);
        }
        std::map<FlagKey, uint32_t> writers;
        std::map<FlagKey, uint32_t> readers;
        ResolveFlagDependencies(traces, config.ranks, writers, readers);
        if (kind == KernelKind::BIG_DATA) {
            // 大数据量的每个数据就绪flag都必须被consumer消费, 否则说明有分片漏归约。
            for (const auto& writer : writers) {
                EXPECT_NE(readers.count(writer.first), 0u) << "recorded flag is never consumed";
            }
        }
    }

    TEST(ReduceScatterAivCoreTest, LocalTreeWriteCoversOutputForEveryRankSize)
    {
        for (uint32_t ranks : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 16u}) {
            Validate<uint32_t>(KernelKind::LOCAL_TREE, Config{ranks, ranks, 0, 64});
            Validate<uint8_t>(KernelKind::LOCAL_TREE, Config{ranks, ranks, 0, 4097});
        }
    }

    TEST(ReduceScatterAivCoreTest, LocalTreeWriteIdleCoresStaySilent)
    {
        for (uint32_t extra : {1u, 3u, 8u}) {
            Validate<uint32_t>(KernelKind::LOCAL_TREE, Config{4, 4 + extra, 0, 64});
            Validate<uint32_t>(KernelKind::LOCAL_TREE, Config{8, 8 + extra, 0, 64});
        }
        // 物理核数大于算子核数(SK满核启动)时, 超出算子核数的核不参与flag收发。
        Validate<uint32_t>(KernelKind::LOCAL_TREE, Config{8, 8, 32, 64});
    }

    TEST(ReduceScatterAivCoreTest, LocalTreeWriteNeedsRankSizeCores)
    {
        for (uint32_t blocks : {1u, 2u, 3u}) {
            Validate<uint32_t>(KernelKind::LOCAL_TREE, Config{8, blocks, 0, 64});
        }
        Validate<uint32_t>(KernelKind::LOCAL_TREE, Config{4, 3, 0, 64});
    }

    TEST(ReduceScatterAivCoreTest, LocalTreeWriteHandlesPingpongBoundary)
    {
        const uint64_t limitCount = DATA_LIMIT / sizeof(uint32_t);
        // 单分片刚好达到/超过DATA_LIMIT: 前者仍走pingpong, 后者必须BarrierAll兜底。
        Validate<uint32_t>(KernelKind::LOCAL_TREE, Config{8, 8, 0, limitCount});
        Validate<uint32_t>(KernelKind::LOCAL_TREE, Config{8, 8, 0, limitCount + 1});
        // 单分片满足DATA_LIMIT, 但全局数据量超过PINGPONG_TOTAL_DATA_LIMIT时同样退出pingpong。
        Validate<uint32_t>(KernelKind::LOCAL_TREE, Config{32, 32, 0, limitCount});
    }

    TEST(ReduceScatterAivCoreTest, BigDataWriteSplitsProducerAndConsumerStages)
    {
        for (uint32_t ranks : {2u, 3u, 4u, 5u, 8u}) {
            for (uint32_t multiple = 2u; multiple <= 4u; ++multiple) {
                Validate<uint32_t>(KernelKind::BIG_DATA, Config{ranks, ranks * multiple, 0, 257});
            }
        }
        // 单字节类型下len较大, 保证双producer分支被覆盖。
        Validate<uint8_t>(KernelKind::BIG_DATA, Config{4, 16, 0, 1024});
    }

    TEST(ReduceScatterAivCoreTest, BigDataWriteDoubleProducerNeedsNonEmptyPieces)
    {
        // 分片数不足2时无法铺满双producer, stage1必须退回单producer。
        Validate<uint32_t>(KernelKind::BIG_DATA, Config{8, 32, 0, 1});
        Validate<uint32_t>(KernelKind::BIG_DATA, Config{8, 32, 0, 2});
        Validate<uint32_t>(KernelKind::BIG_DATA, Config{8, 32, 0, 257});
    }

    TEST(ReduceScatterAivCoreTest, BigDataWriteIdleCoresStaySilent)
    {
        // 核数向下取整到rankSize的整数倍, 尾部核不参与搬运与flag收发。
        for (uint32_t blocks : {9u, 11u, 13u, 17u, 20u, 24u, 30u, 31u, 32u}) {
            Validate<uint32_t>(KernelKind::BIG_DATA, Config{8, blocks, 0, 257});
        }
        Validate<uint32_t>(KernelKind::BIG_DATA, Config{4, 16, 48, 257});
    }

    TEST(ReduceScatterAivCoreTest, BigDataWriteNeedsAtLeastTwoCoresPerRank)
    {
        for (uint32_t blocks : {1u, 2u, 3u, 7u}) {
            Validate<uint32_t>(KernelKind::BIG_DATA, Config{8, blocks, 0, 257});
        }
        Validate<uint32_t>(KernelKind::BIG_DATA, Config{4, 6, 0, 257});
    }
} // namespace
} // namespace reduce_scatter_aiv_test
