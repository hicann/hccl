/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_URMA_AIV_CORE_TEST_H
#define ALLTOALL_URMA_AIV_CORE_TEST_H

#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

// 必须先于一切模板头: 版本门控宏与真基类头guard预占
#define ASC_DEVKIT_VERSION_NUM 90500000
#define HCOMM_VERSION_NUM 90500000
#define AIV_URMA_COMM_BASE_H // 短路真aiv_urma_comm_base.h(与核同目录, 引号include无法用-I屏蔽)
#include "aiv_urma_comm_base_stub.h"
#include "aiv_urma_all_to_all_v_mesh_1d.h"

namespace alltoall_urma_aiv_test {
namespace {
    constexpr uint32_t DT_BYTES = 4; // int32_t

    struct Scenario {
        uint32_t ranks;
        uint32_t blocks;
        uint32_t channelsPerRank = 1;
        uint32_t portsPerChannel = 1;
        bool equalCounts = true;       // false: 每对sendCounts互异
        bool ownCountMismatch = false; // sendCounts[me] != recvCounts[me]
    };

    struct CountPlan {
        std::vector<uint64_t> sendCounts;
        std::vector<uint64_t> sendDispls;
        std::vector<uint64_t> recvCounts;
        std::vector<uint64_t> recvDispls;
    };

    CountPlan MakeCounts(const Scenario& sc)
    {
        CountPlan plan;
        plan.sendCounts.resize(sc.ranks);
        plan.sendDispls.resize(sc.ranks);
        plan.recvCounts.resize(sc.ranks);
        plan.recvDispls.resize(sc.ranks);
        uint64_t sendAcc = 0;
        uint64_t recvAcc = 0;
        for (uint32_t peer = 0; peer < sc.ranks; peer++) {
            uint64_t count = sc.equalCounts ? 8 : 1 + (peer * 7 + 3) % 13;
            if (peer == 0 && sc.ownCountMismatch) {
                // 本rank发自己的量与收自己的量刻意不等, 验证min防御
                plan.sendCounts[peer] = count + 5;
                plan.recvCounts[peer] = count;
            } else {
                plan.sendCounts[peer] = count;
                plan.recvCounts[peer] = count;
            }
            plan.sendDispls[peer] = sendAcc;
            plan.recvDispls[peer] = recvAcc;
            sendAcc += plan.sendCounts[peer];
            recvAcc += plan.recvCounts[peer];
        }
        return plan;
    }

    // pass: 对每个(rank, block)在CPU上执行kernel入口, 记录事件; 收集地址slot值
    void
    RunAllCores(uint32_t ranks, uint32_t blocks, const CountPlan& plan, std::vector<Trace>& traces, bool collectSlots)
    {
        traces.clear();
        traces.reserve(ranks * blocks);
        std::map<uint64_t, uint64_t> slots;
        for (uint32_t rank = 0; rank < ranks; ++rank) {
            ExtraArgs args;
            for (uint32_t peer = 0; peer < ranks; peer++) {
                args.sendCounts[peer] = plan.sendCounts[peer];
                args.sendDispls[peer] = plan.sendDispls[peer];
                args.recvCounts[peer] = plan.recvCounts[peer];
                args.recvDispls[peer] = plan.recvDispls[peer];
            }
            for (uint32_t block = 0; block < blocks; ++block) {
                traces.push_back(Trace{rank, block, {}});
                currentTrace = &traces.back();
                AivUrmaAllToAllVV2Mesh1D<int32_t>(
                    reinterpret_cast<GM_ADDR>(Address(BUFFIN_BASE, rank)), Address(INPUT_BASE, rank),
                    Address(OUTPUT_BASE, rank), rank, 0, ranks, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, blocks, false, nullptr,
                    nullptr, nullptr, 0, false, false, 0, 0, args);
                if (collectSlots) {
                    for (const auto& ev : currentTrace->events) {
                        if (ev.kind == EventKind::WRITE) {
                            slots[ev.a] = ev.b;
                        }
                    }
                }
            }
        }
        currentTrace = nullptr;
        if (collectSlots) {
            static std::map<uint64_t, uint64_t> slotTable;
            slotTable = slots;
            outputSlotValues = &slotTable;
        }
    }

    // 调度器: 离线重演全部核事件; WRITE更新模拟GM, WAIT需值匹配才推进, 收敛失败即死锁
    void ReplayAndCheckConvergence(const std::vector<Trace>& traces, uint32_t ranks)
    {
        std::map<uint64_t, uint64_t> gm;
        std::vector<size_t> positions(traces.size(), 0);
        size_t remaining = 0;
        for (const auto& t : traces) {
            remaining += t.events.size();
        }
        while (remaining != 0) {
            bool progressed = false;
            for (size_t i = 0; i < traces.size(); ++i) {
                const Trace& trace = traces[i];
                if (positions[i] == trace.events.size()) {
                    continue;
                }
                const Event& ev = trace.events[positions[i]];
                if (ev.kind == EventKind::WAIT) {
                    auto it = gm.find(ev.a);
                    if (it == gm.end() || it->second != ev.b) {
                        continue; // 未就绪, 等下一轮
                    }
                }
                switch (ev.kind) {
                    case EventKind::WRITE:
                        gm[ev.a] = ev.b;
                        break;
                    case EventKind::CLEAR:
                        gm[ev.a] = 0;
                        break;
                    default:
                        break;
                }
                ++positions[i];
                --remaining;
                progressed = true;
            }
            ASSERT_TRUE(progressed) << "deadlock: " << remaining << " events pending, ranks=" << ranks;
        }
    }

    // 断言: 地址交换pair全集性(活跃pair恰1次, 非活跃0次), 数据COPY writer核唯一,
    // 通道切分守恒, notify每slot恰1次(含0字节补发), own拷贝min防御与内存覆盖
    void CheckProtocol(const std::vector<Trace>& traces, const Scenario& sc, const CountPlan& plan)
    {
        const uint32_t ranks = sc.ranks;
        std::map<uint64_t, int> addrSlotWrites;
        std::map<uint64_t, int> notify2Writes;
        std::map<std::pair<uint32_t, uint32_t>, std::set<std::pair<uint32_t, uint32_t>>> peerWriters;
        std::map<std::pair<uint32_t, uint32_t>, uint64_t> peerBytes;
        std::vector<std::pair<uint32_t, Event>> ownCopies;

        for (const auto& t : traces) {
            for (const auto& ev : t.events) {
                if (ev.kind == EventKind::WRITE) {
                    if (ev.b != 1ULL && ev.b != 2ULL) {
                        ++addrSlotWrites[ev.a]; // 地址slot写入值是output地址, 区别于notify值1/2
                    }
                    if (ev.b == 2ULL) {
                        ++notify2Writes[ev.a];
                    }
                    continue;
                }
                if (ev.kind != EventKind::COPY || ev.c == 0) {
                    continue;
                }
                // own拷贝: 0号核LocalCopy, dst落在本rank的OUTPUT段(数据面COPY的dst只落在对端段)
                const uint64_t myOutBase = Address(OUTPUT_BASE, t.rank);
                if (t.block == 0 && ev.a >= myOutBase && ev.a < myOutBase + RANK_STRIDE) {
                    ownCopies.push_back({t.rank, ev});
                } else {
                    // 数据面COPY: dst反解目的rank, 聚合writer核与bytes
                    for (uint32_t peer = 0; peer < ranks; peer++) {
                        const uint64_t base = Address(OUTPUT_BASE, peer);
                        if (ev.a >= base && ev.a < base + RANK_STRIDE) {
                            peerWriters[{peer, t.rank}].insert({t.rank, t.block});
                            peerBytes[{peer, t.rank}] += ev.c;
                            break;
                        }
                    }
                }
            }
        }

        // 1. 地址交换: 每(me,peer)活跃pair恰好写1次slot
        for (uint32_t me = 0; me < ranks; me++) {
            for (uint32_t peer = 0; peer < ranks; peer++) {
                if (me == peer) {
                    continue;
                }
                const bool active = (plan.sendCounts[peer] != 0) || (plan.recvCounts[peer] != 0);
                const uint64_t slot
                    = Address(BUFFIN_BASE, peer) + AIV_URMA_OUTPUT_ADDR_SLOT_OFFSET + me * sizeof(uint64_t);
                const int writes = addrSlotWrites.count(slot) ? addrSlotWrites[slot] : 0;
                EXPECT_EQ(writes, active ? 1 : 0)
                    << "addr exchange writes for pair(" << me << "," << peer << ") expect " << (active ? 1 : 0);
            }
        }

        // 2. 数据到齐notify: 每(receiver,peer,ch) slot恰1个WRITE(2) —— receiver在所有trace的
        // WAIT(2)对象上校验: 数据notify slot = buffin(me)+FLAG1+MAX_RANK_SIZE*8+(peer*8+ch)*8
        for (uint32_t me = 0; me < ranks; me++) {
            for (uint32_t peer = 0; peer < ranks; peer++) {
                if (me == peer || plan.recvCounts[peer] == 0) {
                    continue;
                }
                for (uint32_t ch = 0; ch < sc.channelsPerRank; ch++) {
                    const uint64_t slot = Address(BUFFIN_BASE, me) + FLAG1_OFFSET + MAX_RANK_SIZE * sizeof(uint64_t)
                                          + (peer * MAX_CHANNELS_PER_RANK + ch) * sizeof(uint64_t);
                    EXPECT_EQ(notify2Writes.count(slot) ? notify2Writes[slot] : 0, 1)
                        << "data notify(2) count for recv pair(" << me << "," << peer << ") ch=" << ch;
                }
            }
        }

        // 3. 每(dst,src)数据COPY的writer核唯一 + bytes守恒(通道切分总和不丢不重)
        for (const auto& [pair, writers] : peerWriters) {
            EXPECT_EQ(writers.size(), 1u)
                << "pair(dst=" << pair.first << ",src=" << pair.second << ") written by multiple cores";
        }
        for (uint32_t src = 0; src < ranks; src++) {
            for (uint32_t dst = 0; dst < ranks; dst++) {
                if (src == dst || plan.sendCounts[dst] == 0) {
                    continue;
                }
                const uint64_t expected = plan.sendCounts[dst] * DT_BYTES;
                const uint64_t actual = peerBytes.count({dst, src}) ? peerBytes[{dst, src}] : 0;
                EXPECT_EQ(actual, expected) << "bytes sent by src=" << src << " to dst=" << dst;
            }
        }

        // 4. own拷贝: 每rank恰1份, bytes==min(该rank sendCounts, recvCounts)*dt防越界
        EXPECT_EQ(ownCopies.size(), static_cast<size_t>(ranks)) << "own copy expected once per rank";
        for (const auto& [rank, ev] : ownCopies) {
            const uint64_t sendBytes = plan.sendCounts[rank] * DT_BYTES;
            const uint64_t recvBytes = plan.recvCounts[rank] * DT_BYTES;
            EXPECT_EQ(ev.c, std::min(sendBytes, recvBytes)) << "own copy bytes must be min(send, recv), rank=" << rank;
        }
    }

    void RunScenario(const Scenario& sc)
    {
        SCOPED_TRACE(
            ::testing::Message() << "ranks=" << sc.ranks << " blocks=" << sc.blocks << " ch=" << sc.channelsPerRank
                                 << " ports=" << sc.portsPerChannel << " equal=" << sc.equalCounts
                                 << " ownMismatch=" << sc.ownCountMismatch);
        AivUrmaCommBase::channelCfg_ = {sc.channelsPerRank, sc.portsPerChannel};
        const CountPlan plan = MakeCounts(sc);

        std::vector<Trace> traces;
        RunAllCores(sc.ranks, sc.blocks, plan, traces, true);  // pass1: 收集地址slot值
        RunAllCores(sc.ranks, sc.blocks, plan, traces, false); // pass2: 用真实值重跑
        ReplayAndCheckConvergence(traces, sc.ranks);
        CheckProtocol(traces, sc, plan);
    }
} // namespace

TEST(AlltoallUrmaAivCore, PairSetFullCoverageAcrossRounds)
{
    // 奇/偶rankSize与跨GROUP_SIZE分轮: 2/3/5/17/33(多轮+尾轮)
    for (uint32_t ranks : {2u, 3u, 5u, 9u, 17u, 33u}) {
        RunScenario({ranks, 4});
    }
}

TEST(AlltoallUrmaAivCore, CoreAssignmentRemainderAndIdle)
{
    // 核数<对端数(remainder前若干核多1个), 核数>对端数(空闲核只出SYNC)
    RunScenario({9, 1});
    RunScenario({5, 3});
    RunScenario({9, 20});
    RunScenario({33, 4});
}

TEST(AlltoallUrmaAivCore, MultiChannelSplitConservation)
{
    // 多通道按count切分: 通道数2/3, ports 1/2(不均权切分)
    RunScenario({5, 4, 2, 1});
    RunScenario({9, 4, 3, 2});
}

TEST(AlltoallUrmaAivCore, NotifyProtocolMultiRoundProgress)
{
    // 33卡=3轮, 每轮WAIT(1)/WAIT(2)清零后复用同一slot, 由调度器收敛性保证
    RunScenario({33, 8});
    RunScenario({17, 8, 2, 1});
}

TEST(AlltoallUrmaAivCore, ZeroCountPairsSkipped)
{
    // 双向0对端两端一致跳过, 不产生WRITE/WAIT(死锁检测兜底)
    RunScenario({5, 4});
    RunScenario({9, 4, 2, 1});
}

TEST(AlltoallUrmaAivCore, OwnCopyMinGuard)
{
    // sendCounts[me] != recvCounts[me]时own拷贝量取min防越界
    RunScenario({5, 4, 1, 1, true, true});
    RunScenario({17, 4, 2, 1, false, true});
}
} // namespace alltoall_urma_aiv_test

#endif
