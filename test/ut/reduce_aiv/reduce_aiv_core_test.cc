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
#include "aiv_reduce_mesh_1d.h"

namespace reduce_aiv_test {
namespace {

    // {持有flag的rank, flag槽位, tag}。同一槽位会在不同轮次以不同tag重复使用，故tag必须参与配对。
    using FlagKey = std::tuple<uint32_t, uint64_t, int32_t>;
    using Range = std::pair<uint64_t, uint64_t>;

    struct Config {
        uint32_t ranks = 1;
        uint32_t blocks = 1;
        uint32_t launched = 0; // 实际启动核数, 0表示等于blocks
        uint64_t count = 1;    // 每rank的元素个数
        uint32_t root = 0;
    };

    struct Scenario {
        Config cfg;
        uint64_t lenBytes = 0;
        uint32_t launched = 0;
    };

    Scenario MakeScenario(const Config& cfg, uint64_t elementBytes)
    {
        Scenario s;
        s.cfg = cfg;
        s.lenBytes = cfg.count * elementBytes;
        s.launched = cfg.launched == 0 ? cfg.blocks : cfg.launched;
        return s;
    }

    // 在CPU上逐rank、逐核执行生产kernel入口，收集每个核的事件序列。
    template <typename T>
    std::vector<Trace> RunScenario(const Scenario& s)
    {
        std::vector<Trace> traces;
        traces.reserve(s.cfg.ranks * s.launched);
        for (uint32_t rank = 0; rank < s.cfg.ranks; ++rank) {
            for (uint32_t block = 0; block < s.launched; ++block) {
                traces.push_back(Trace{rank, block});
                currentTrace = &traces.back();
                AivReduceV2Mesh1D<T>(
                    nullptr, Address(INPUT_BASE, rank), Address(OUTPUT_BASE, rank), rank, 0, s.cfg.ranks, s.cfg.count,
                    0, 0, s.cfg.root, 1, 0, 0, 1, 0, 0, s.cfg.blocks, true, nullptr, nullptr, nullptr, 0, false, false);
            }
        }
        currentTrace = nullptr;
        return traces;
    }

    // 按依赖顺序推进所有核的事件流: 只有{目标rank, 槽位, tag}已被记录过的WaitFlag才放行,
    // 借此识别死锁与配对缺失; 同一flag被发布两次视为错误(会导致消费方读到未就绪数据)。
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

    bool Overlaps(const Range& a, const Range& b) { return a.first < b.second && b.first < a.second; }

    // 归并区间便于做交集/覆盖判定
    std::vector<Range> Normalize(std::vector<Range> ranges)
    {
        std::sort(ranges.begin(), ranges.end());
        std::vector<Range> out;
        for (const auto& r : ranges) {
            if (r.first == r.second) {
                continue;
            }
            if (!out.empty() && out.back().second >= r.first) {
                out.back().second = std::max(out.back().second, r.second);
            } else {
                out.push_back(r);
            }
        }
        return out;
    }

    bool Intersects(const std::vector<Range>& a, const std::vector<Range>& b)
    {
        for (const auto& x : a) {
            for (const auto& y : b) {
                if (Overlaps(x, y)) {
                    return true;
                }
            }
        }
        return false;
    }

    // 本 PR 的核心不变式之一: 归约前的数据面不再有跨卡写。
    // 所有 COPY 的目的地必须落在本rank的ccl或本rank的output内。
    void CheckNoCrossCardDataWrite(const std::vector<Trace>& traces)
    {
        for (const auto& trace : traces) {
            const uint64_t myCcl = Address(CCL_BASE, trace.rank);
            const uint64_t myOutput = Address(OUTPUT_BASE, trace.rank);
            for (const auto& event : trace.events) {
                if (event.kind != EventKind::COPY) {
                    continue;
                }
                const bool toMyCcl = event.destination >= myCcl && event.destination < myCcl + RANK_STRIDE;
                const bool toMyOutput = event.destination >= myOutput && event.destination < myOutput + RANK_STRIDE;
                EXPECT_TRUE(toMyCcl || toMyOutput)
                    << "rank[" << trace.rank << "] block[" << trace.block << "] writes to a remote buffer at "
                    << event.destination << "; the reduce data plane must be pull-only";
            }
        }
    }

    // 本 PR 的核心不变式之二: 发布区不得被本卡的归约写覆盖。
    // 发布区 = 属主把自己 input 拷进本卡ccl的那段(供对端跨卡读); 折叠是就地写 slot(dst)，
    // 若两者重合，本卡折叠会在对端仍在跨卡读发布区时改掉数据。
    // 该校验完全从观测事件推导(源落在本rank input内的COPY即发布写)，不依赖 kernel 的分区公式。
    // 注意不能用"对端读过的一切"当发布区: gather 阶段本来就要跨卡读对端的折叠结果。
    void CheckPublishRegionNotReduced(const std::vector<Trace>& traces, uint32_t ranks)
    {
        for (uint32_t owner = 0; owner < ranks; ++owner) {
            const uint64_t ccl = Address(CCL_BASE, owner);
            const uint64_t input = Address(INPUT_BASE, owner);
            std::vector<Range> published; // 属主从自己input拷入本卡ccl的区间
            std::vector<Range> reduced;   // 属主用原子加写入本卡ccl的区间
            for (const auto& trace : traces) {
                if (trace.rank != owner) {
                    continue;
                }
                for (const auto& event : trace.events) {
                    if (event.kind != EventKind::COPY || event.bytes == 0) {
                        continue;
                    }
                    const bool dstInMyCcl = event.destination >= ccl && event.destination < ccl + RANK_STRIDE;
                    if (!dstInMyCcl) {
                        continue;
                    }
                    if (event.source >= input && event.source < input + RANK_STRIDE) {
                        published.emplace_back(event.destination - ccl, event.destination - ccl + event.bytes);
                    }
                    if (event.reduce) {
                        reduced.emplace_back(event.destination - ccl, event.destination - ccl + event.bytes);
                    }
                }
            }
            EXPECT_FALSE(Intersects(Normalize(published), Normalize(reduced)))
                << "rank[" << owner << "] reduces in-place into the region it published for peers to read";
        }
    }

    // 对端跨卡读到的每一段，都必须是属主发布过或本地填充过的区间，否则会读到未初始化数据。
    void CheckRemoteReadsArePublished(const std::vector<Trace>& traces, uint32_t ranks)
    {
        for (uint32_t owner = 0; owner < ranks; ++owner) {
            const uint64_t base = Address(CCL_BASE, owner);
            std::vector<Range> written;
            std::vector<Range> readByPeer;
            for (const auto& trace : traces) {
                for (const auto& event : trace.events) {
                    if (event.kind != EventKind::COPY || event.bytes == 0) {
                        continue;
                    }
                    if (event.destination >= base && event.destination < base + RANK_STRIDE && trace.rank == owner) {
                        written.emplace_back(event.destination - base, event.destination - base + event.bytes);
                    }
                    if (event.source >= base && event.source < base + RANK_STRIDE && trace.rank != owner) {
                        readByPeer.emplace_back(event.source - base, event.source - base + event.bytes);
                    }
                }
            }
            const auto w = Normalize(written);
            for (const auto& r : Normalize(readByPeer)) {
                bool covered = false;
                for (const auto& x : w) {
                    if (x.first <= r.first && r.second <= x.second) {
                        covered = true;
                        break;
                    }
                }
                EXPECT_TRUE(covered) << "peer reads rank[" << owner << "] ccl range [" << r.first << "," << r.second
                                     << ") which the owner never wrote";
            }
        }
    }

    // root 的 output 必须被完整且不重不漏地覆盖; 非 root 的 output 不得被写。
    void CheckOutputTiling(const std::vector<Trace>& traces, const Scenario& s)
    {
        for (uint32_t rank = 0; rank < s.cfg.ranks; ++rank) {
            const uint64_t base = Address(OUTPUT_BASE, rank);
            std::vector<Range> ranges;
            for (const auto& trace : traces) {
                if (trace.rank != rank) {
                    continue;
                }
                for (const auto& event : trace.events) {
                    if (event.kind != EventKind::COPY || event.bytes == 0) {
                        continue;
                    }
                    if (event.destination < base || event.destination >= base + RANK_STRIDE) {
                        continue;
                    }
                    EXPECT_LE(event.destination + event.bytes, base + s.lenBytes) << "output write exceeds dataSize";
                    ranges.emplace_back(event.destination - base, event.destination - base + event.bytes);
                }
            }
            if (rank != s.cfg.root) {
                EXPECT_TRUE(ranges.empty()) << "non-root rank[" << rank << "] writes its output";
                continue;
            }
            uint64_t expect = 0;
            for (const auto& range : Normalize(ranges)) {
                EXPECT_EQ(range.first, expect) << "root output has a gap or overlap at " << expect;
                expect = range.second;
            }
            EXPECT_EQ(expect, s.lenBytes) << "root output is not fully written";
        }
    }

    // SyncAll 是卡内栅栏，同一次 launch 内每个核(含缩核早退核与SK空闲核)的调用次数必须相等，否则栅栏不匹配。
    void CheckSyncAllBalanced(const std::vector<Trace>& traces)
    {
        if (traces.empty()) {
            return;
        }
        const uint32_t expect = traces.front().syncAlls;
        for (const auto& trace : traces) {
            EXPECT_EQ(trace.syncAlls, expect) << "rank[" << trace.rank << "] block[" << trace.block
                                              << "] calls SyncAll " << trace.syncAlls << " times, expected " << expect;
        }
    }

    template <typename T>
    void Validate(const Config& config)
    {
        const Scenario s = MakeScenario(config, sizeof(T));
        SCOPED_TRACE(
            ::testing::Message() << "ranks=" << config.ranks << " blocks=" << config.blocks
                                 << " launched=" << s.launched << " count=" << config.count << " root=" << config.root
                                 << " elementBytes=" << sizeof(T));
        const auto traces = RunScenario<T>(s);
        CheckNoCrossCardDataWrite(traces);
        CheckPublishRegionNotReduced(traces, config.ranks);
        CheckRemoteReadsArePublished(traces, config.ranks);
        CheckOutputTiling(traces, s);
        CheckSyncAllBalanced(traces);
        std::map<FlagKey, uint32_t> writers;
        std::map<FlagKey, uint32_t> readers;
        ResolveFlagDependencies(traces, config.ranks, writers, readers);
    }

    // ProcessMultiCore(blocks >= ranks+1) 与 ProcessMultiRank(blocks < ranks+1) 两条分支都要覆盖。
    TEST(ReduceAivCoreTest, CoversBothBranchesAcrossRankSizes)
    {
        for (uint32_t ranks : {1u, 2u, 3u, 4u, 8u, 16u, 32u}) {
            for (uint32_t blocks : {1u, 2u, 4u, 8u, 17u, 48u, 64u}) {
                Validate<uint32_t>(Config{ranks, blocks, 0, 64, 0});
                Validate<uint8_t>(Config{ranks, blocks, 0, 4097, 0});
            }
        }
    }

    // 16p/256B(fp32) 是本 PR 修复的复现场景; 64核与48核(仿真stub的核数)都要覆盖。
    TEST(ReduceAivCoreTest, RegressionCase16p256B)
    {
        Validate<float>(Config{16, 64, 0, 64, 0});
        Validate<float>(Config{16, 48, 0, 64, 0});
        for (uint32_t root : {0u, 1u, 7u, 15u}) {
            Validate<float>(Config{16, 64, 0, 64, root});
        }
    }

    // 128p/2MB 走 ProcessMultiRank 的单核折叠分支。
    TEST(ReduceAivCoreTest, RegressionCase128pSmallCoreBranch)
    {
        Validate<float>(Config{128, 64, 0, 2 * 1024 * 1024 / sizeof(float), 0});
        Validate<float>(Config{128, 64, 0, 2 * 1024 * 1024 / sizeof(float), 127});
    }

    // dataCount < rankSize: sliceCount 退化为0、全部数据挤入末片，各片长度与槽步长严重不均。
    TEST(ReduceAivCoreTest, DataCountSmallerThanRankSize)
    {
        Validate<uint32_t>(Config{8, 8, 0, 1, 0});
        Validate<uint32_t>(Config{16, 4, 0, 3, 0});
        Validate<uint32_t>(Config{32, 8, 0, 7, 31});
        Validate<uint8_t>(Config{64, 16, 0, 5, 0});
    }

    // count 不是 rankSize 整数倍: 末片长度与前面不同，槽步长/覆盖都受影响。
    TEST(ReduceAivCoreTest, NonDivisibleCount)
    {
        for (uint64_t count : {1u, 3u, 17u, 65u, 129u, 1023u, 4093u}) {
            Validate<uint32_t>(Config{16, 64, 0, count, 0});
            Validate<uint32_t>(Config{7, 5, 0, count, 3});
        }
    }

    // 缩核早退分支(len < blocks && rankSize < blocks)与 SK 满核启动的空闲核。
    TEST(ReduceAivCoreTest, ShrinkAndIdleCores)
    {
        // len=8 < blocks=64 且 ranks=4 < 64 -> 触发缩核, 多余核走 Process() 的早退分支
        Validate<uint32_t>(Config{4, 64, 0, 8, 0});
        Validate<uint32_t>(Config{8, 64, 0, 4, 0});
        // 物理核数大于算子核数(SK满核启动), 超出部分为空闲核
        Validate<uint32_t>(Config{16, 16, 64, 64, 0});
        Validate<uint32_t>(Config{8, 8, 32, 4097, 0});
    }

    TEST(ReduceAivCoreTest, SingleRankDegenerate)
    {
        for (uint32_t blocks : {1u, 2u, 8u, 64u}) {
            Validate<uint32_t>(Config{1, blocks, 0, 64, 0});
            Validate<float>(Config{1, blocks, 0, 1, 0});
        }
    }
} // namespace
} // namespace reduce_aiv_test
