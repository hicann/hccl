/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <vector>
#include <gtest/gtest.h>
#include "template_utils.h"

using namespace ops_hccl;

// GetRhdStepNum = ⌊log₂rankSize⌋（物理规模 p 与有效规模 P = p−r 结果相同）
TEST(GetRhdStepNumTest, FloorLog2Values)
{
    const std::vector<std::pair<u32, u32>> cases
        = {{1, 0}, {2, 1}, {3, 1}, {4, 2}, {5, 2}, {7, 2}, {8, 3}, {9, 3}, {16, 4}, {17, 4}, {32, 5}, {33, 5}};
    for (const auto& c : cases) {
        EXPECT_EQ(GetRhdStepNum(c.first), c.second) << "rankSize=" << c.first;
    }
}

// RhdStepInfo 默认构造：区段 [0,0)、nSlices 为 0、phase 为枚举首值 MAIN_RS
TEST(RhdStructDefaultsTest, StepInfoDefaults)
{
    RhdStepInfo s;
    EXPECT_EQ(s.step, 0u);
    EXPECT_EQ(s.phase, RhdPhase::MAIN_RS);
    EXPECT_EQ(s.peer, 0u);
    EXPECT_EQ(s.nSlices, 0u);
    EXPECT_EQ(s.txSegStart, 0u);
    EXPECT_EQ(s.txSegEnd, 0u);
    EXPECT_EQ(s.rxSegStart, 0u);
    EXPECT_EQ(s.rxSegEnd, 0u);
}

// RhdPart1Info 默认构造：2^N 空划分形态（无前后处理、无配对）
TEST(RhdStructDefaultsTest, Part1InfoDefaults)
{
    RhdPart1Info p;
    EXPECT_EQ(p.r, 0u);
    EXPECT_TRUE(p.isPowerOfTwo);
    EXPECT_FALSE(p.inPart1);
    EXPECT_FALSE(p.isRepresentative);
    EXPECT_EQ(p.pairedAlgRank, INVALID_VALUE_RANKID);
    EXPECT_EQ(p.virtualRank, 0u);
    EXPECT_EQ(p.effectiveRankSize, 0u);
    EXPECT_EQ(p.unitStart, 0u);
    EXPECT_EQ(p.unitCount, 1u);
}

// ==================== CalcRhdPart1Info + σ 映射辅助 ====================

namespace {
// 测试基准：2^⌊log₂p⌋（与实现独立推导）
u32 LargestPow2Le(u32 p)
{
    u32 hp = 1;
    while (hp <= p / 2) {
        hp *= 2;
    }
    return hp;
}

// 虚拟序 v 的默认承载 algRank：配对取低位侧（默认代表），block 取 v+r
u32 CarrierOfVirtual(u32 r, u32 v) { return (v < r) ? v : v + r; }
} // namespace

// v 的 part1 上下文（配对取低位侧承载）
RhdPart1Info MakeInfoForVirtual(u32 p, u32 v)
{
    const u32 r = p - LargestPow2Le(p);
    RhdPart1Info info;
    EXPECT_EQ(CalcRhdPart1Info(p, CarrierOfVirtual(r, v), INVALID_VALUE_RANKID, info), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(info.virtualRank, v);
    return info;
}

// p ∈ {2,4,8,16}（2^N）与 p ∈ {5,6,7,9,13}（r=1..3,5）：r/P/part1 划分/有效序/代表恒低位（root=INVALID）
TEST(CalcRhdPart1InfoTest, Part1Partition)
{
    for (u32 p : {2u, 4u, 8u, 16u}) {
        for (u32 j = 0; j < p; j++) {
            RhdPart1Info info;
            ASSERT_EQ(CalcRhdPart1Info(p, j, INVALID_VALUE_RANKID, info), HcclResult::HCCL_SUCCESS)
                << "p=" << p << " j=" << j;
            EXPECT_TRUE(info.isPowerOfTwo);
            EXPECT_EQ(info.r, 0u);
            EXPECT_EQ(info.effectiveRankSize, p);
            EXPECT_FALSE(info.inPart1);
            EXPECT_EQ(info.pairedAlgRank, INVALID_VALUE_RANKID); // 2^N 无前后处理
            EXPECT_EQ(info.virtualRank, j);                      // 2^N：有效序恒等
            EXPECT_EQ(info.unitStart, j);
            EXPECT_EQ(info.unitCount, 1u);
        }
    }
    for (u32 p : {5u, 6u, 7u, 9u, 13u}) {
        const u32 r = p - LargestPow2Le(p);
        const u32 eff = p - r;
        for (u32 j = 0; j < p; j++) {
            RhdPart1Info info;
            ASSERT_EQ(CalcRhdPart1Info(p, j, INVALID_VALUE_RANKID, info), HcclResult::HCCL_SUCCESS)
                << "p=" << p << " j=" << j;
            EXPECT_EQ(info.r, r) << "p=" << p;
            EXPECT_FALSE(info.isPowerOfTwo);
            EXPECT_EQ(info.effectiveRankSize, eff);
            EXPECT_EQ(info.inPart1, j < 2 * r) << "part1 = [0,2r)";
            EXPECT_EQ(info.virtualRank, (j < 2 * r) ? (j % r) : (j - r)) << "p=" << p << " j=" << j;
            EXPECT_EQ(info.isRepresentative, j < r) << "root=INVALID：代表恒低位";
        }
    }
}

// 非 2^N × root ∈ [0,p) × myAlgRank ∈ [0,p) 全扫描：代表跟随 root、配对互指、持有区段
TEST(CalcRhdPart1InfoTest, RoleAllRootSweep)
{
    for (u32 p : {5u, 7u, 9u, 13u}) {
        const u32 r = p - LargestPow2Le(p);
        for (u32 root = 0; root < p; root++) {
            const bool rootInPart1 = root < 2 * r;
            const bool repIsLow = rootInPart1 ? (root < r) : true; // root 不在 part1 → 默认低位代表
            for (u32 j = 0; j < p; j++) {
                RhdPart1Info info;
                ASSERT_EQ(CalcRhdPart1Info(p, j, root, info), HcclResult::HCCL_SUCCESS)
                    << "p=" << p << " root=" << root << " j=" << j;
                if (j >= 2 * r) { // block 成员
                    EXPECT_FALSE(info.inPart1);
                    EXPECT_FALSE(info.isRepresentative);
                    EXPECT_EQ(info.pairedAlgRank, INVALID_VALUE_RANKID);
                    EXPECT_EQ(info.unitStart, j); // block 恒等布局
                    EXPECT_EQ(info.unitCount, 1u);
                    continue;
                }
                const bool lowSide = j < r;
                const u32 t = j % r;
                EXPECT_TRUE(info.inPart1);
                EXPECT_EQ(info.pairedAlgRank, lowSide ? (t + r) : t) << "配对 (t ↔ t+r) 互指";
                EXPECT_EQ(info.isRepresentative, repIsLow ? lowSide : !lowSide);
                if (rootInPart1 && j == root) {
                    EXPECT_TRUE(info.isRepresentative) << "root 落 pair 内：root 侧当代表";
                }
                // 持有区段（σ 布局与代表选取解耦）：代表占配对整块 [2t,2t+2)，被合并方占自身份额 1 unit
                if (info.isRepresentative) {
                    EXPECT_EQ(info.unitStart, 2 * t);
                    EXPECT_EQ(info.unitCount, 2u);
                } else {
                    EXPECT_EQ(info.unitStart, lowSide ? 2 * t : 2 * t + 1) << "自身份额槽位";
                    EXPECT_EQ(info.unitCount, 1u);
                }
            }
        }
    }
}

// σ 映射：函数公式/单射（[0,p) 置换）/配对份额相邻/虚拟序单调/对齐区段映射单连通（三种边界）
TEST(SigmaMappingTest, UnitStartCountAndContinuity)
{
    for (u32 p : {5u, 6u, 7u, 13u}) {
        const u32 r = p - LargestPow2Le(p);
        const u32 eff = p - r;
        std::vector<u32> starts(p);
        for (u32 j = 0; j < p; j++) {
            starts[j] = RhdUnitStartOfRank(p, j);
            EXPECT_EQ(RhdUnitCountOfRank(p, j), (j < r) ? 2u : 1u) << "份数 (j<r)?2:1";
        }
        // σ 单射：start 是 [0,p) 的置换
        std::vector<u32> sorted = starts;
        std::sort(sorted.begin(), sorted.end());
        for (u32 u = 0; u < p; u++) {
            EXPECT_EQ(sorted[u], u) << "p=" << p << " unit " << u << " 缺失/重复";
        }
        // 配对两成员份额相邻：t 在 2t（占 2 unit），t+r 在 2t+1
        for (u32 t = 0; t < r; t++) {
            EXPECT_EQ(RhdUnitStartOfRank(p, t), 2 * t);
            EXPECT_EQ(RhdUnitStartOfRank(p, t + r), 2 * t + 1);
            EXPECT_EQ(RhdUnitCountOfRank(p, t), 2u);
        }
        // 虚拟序单调：startV(v) = (v<r)?2v:v+r 严格递增
        u32 prevStart = 0;
        for (u32 v = 0; v < eff; v++) {
            const u32 s = (v < r) ? 2 * v : v + r;
            if (v > 0) {
                EXPECT_GT(s, prevStart) << "p=" << p << " v=" << v;
            }
            prevStart = s;
        }
        // 任意虚拟对齐区段 [a, a+dist) 映射 unit 区间单连通（连续性不变量，三种边界形态）
        auto uStart = [r](u32 x) {
            return (x < r) ? 2 * x : x + r;
        };
        auto uEnd = [r](u32 y) {
            return (y <= r) ? 2 * y : y + r;
        };
        for (u32 dist = 1; dist <= eff / 2; dist++) {
            for (u32 a = 0; a + dist <= eff; a += dist) {
                const u32 b = a + dist;
                const u32 us = uStart(a);
                const u32 ue = uEnd(b);
                EXPECT_LT(us, ue) << "p=" << p << " seg=[" << a << "," << b << ")";
                u32 expectLen = 0; // 区间长度 = 成员 unit 数之和（v<r 计 2，v≥r 计 1）
                for (u32 v = a; v < b; v++) {
                    expectLen += (v < r) ? 2 : 1;
                }
                EXPECT_EQ(ue - us, expectLen) << "p=" << p << " seg=[" << a << "," << b << ")";
            }
        }
    }
}

// 防御校验：rankSize==0 / myAlgRank 越界 / root 越界 → HCCL_E_PARA；p=1 退化合法
TEST(CalcRhdPart1InfoTest, ParamValidation)
{
    RhdPart1Info info;
    EXPECT_EQ(CalcRhdPart1Info(0, 0, INVALID_VALUE_RANKID, info), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(CalcRhdPart1Info(4, 4, 0, info), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(CalcRhdPart1Info(4, 0, 4, info), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(CalcRhdPart1Info(4, 0, 5, info), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(CalcRhdPart1Info(1, 0, INVALID_VALUE_RANKID, info), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(info.isPowerOfTwo);
    EXPECT_EQ(info.effectiveRankSize, 1u);
    EXPECT_EQ(info.unitCount, 1u);
}

// 虚拟序 → 承载 algRank（MAIN 相位对端的物理解析）：
// 2^N 恒等；非 2^N 配对取代表侧（跟随 root）、block 恒 v+r；越界入参返回 INVALID_VALUE_RANKID
TEST(RhdVirtualMappingTest, AlgRankOfVirtual)
{
    // 2^N：恒等
    for (u32 p : {2u, 4u, 8u}) {
        for (u32 v = 0; v < p; v++) {
            EXPECT_EQ(RhdAlgRankOfVirtual(p, INVALID_VALUE_RANKID, v), v);
        }
    }
    // 非 2^N × root 全扫描：承载者的 virtualRank 一致，配对承载者必为代表侧，block 恒 v+r
    for (u32 p : {5u, 6u, 7u, 9u, 13u}) {
        const u32 r = p - LargestPow2Le(p);
        const u32 P = p - r;
        for (u32 root = 0; root < p; root++) {
            for (u32 v = 0; v < P; v++) {
                const u32 carrier = RhdAlgRankOfVirtual(p, root, v);
                ASSERT_NE(carrier, INVALID_VALUE_RANKID) << "p=" << p << " root=" << root << " v=" << v;
                RhdPart1Info info;
                ASSERT_EQ(CalcRhdPart1Info(p, carrier, root, info), HcclResult::HCCL_SUCCESS);
                EXPECT_EQ(info.virtualRank, v) << "承载者虚拟序一致 p=" << p << " root=" << root << " v=" << v;
                if (v < r) {
                    EXPECT_TRUE(info.isRepresentative) << "配对承载者必为代表侧";
                } else {
                    EXPECT_EQ(carrier, v + r) << "block 恒 v+r";
                }
            }
        }
        // root = INVALID_VALUE_RANKID → 恒低位承载
        for (u32 v = 0; v < P; v++) {
            EXPECT_EQ(RhdAlgRankOfVirtual(p, INVALID_VALUE_RANKID, v), CarrierOfVirtual(r, v));
        }
    }
    // 非法入参 → INVALID_VALUE_RANKID
    EXPECT_EQ(RhdAlgRankOfVirtual(0, INVALID_VALUE_RANKID, 0), INVALID_VALUE_RANKID);
    EXPECT_EQ(RhdAlgRankOfVirtual(5, INVALID_VALUE_RANKID, 4), INVALID_VALUE_RANKID); // P=4，v=4 越界
}

// ==================== GenRhdReduceScatterSteps ====================

// RS 真语义模拟：RS 输入为全量 buffer——每物理 rank 对全部 p 个 unit 各贡献 +1；
// pre 用 FULL 段（RS 族：代表须含被合并方全量贡献）把被合并方贡献并入代表后其退出，
// 主体按步骤表搬运 + SUM，终态每 unit 在 owner 处计数 == p（每份贡献恰好到达一次）。
// pre 若退化为份额段（SHARE），除被合并方槽位外的 unit 恒缺一份贡献（计数 p−1），缺陷直接暴露。
TEST(GenRhdReduceScatterStepsTest, RsFinalState)
{
    for (u32 p : {2u, 3u, 4u, 5u, 6u, 7u, 9u, 17u, 32u}) {
        const u32 r = p - LargestPow2Le(p);
        const u32 P = p - r;
        std::vector<std::vector<RhdStepInfo>> steps(P);
        for (u32 v = 0; v < P; v++) {
            const RhdPart1Info info = MakeInfoForVirtual(p, v);
            ASSERT_EQ(GenRhdReduceScatterSteps(info, steps[v]), HcclResult::HCCL_SUCCESS) << "p=" << p << " v=" << v;
            ASSERT_EQ(steps[v].size(), static_cast<size_t>(GetRhdStepNum(P)));
        }
        // 全量输入：每物理 rank 持全部 p unit 各 +1
        std::vector<std::vector<u32>> hold(p, std::vector<u32>(p, 1));
        // pre(FULL)：被合并方全量并入代表（默认低位代表承载），被合并方退出
        for (u32 t = 0; t < r; t++) {
            RhdPart1Info mergedInfo;
            ASSERT_EQ(CalcRhdPart1Info(p, t + r, INVALID_VALUE_RANKID, mergedInfo), HcclResult::HCCL_SUCCESS);
            RhdStepInfo pre, post;
            ASSERT_EQ(
                GenRhdPrePostSteps(mergedInfo, pre, post, RhdSegMode::FULL, RhdSegMode::SHARE),
                HcclResult::HCCL_SUCCESS);
            ASSERT_EQ(pre.txSegStart, 0u) << "FULL 段 = [0, p)";
            ASSERT_EQ(pre.txSegEnd, p);
            for (u32 u = pre.txSegStart; u < pre.txSegEnd; u++) {
                hold[t][u] += hold[t + r][u];
                hold[t + r][u] = 0;
            }
        }
        // 主体：每步按 tx 段搬出发送（清零），按 rx 段累加对端送入（承载：配对=代表，block=v+r）
        const u32 nSteps = GetRhdStepNum(P);
        for (u32 k = 0; k < nSteps; k++) {
            std::vector<std::vector<u32>> sent(p, std::vector<u32>(p, 0));
            for (u32 v = 0; v < P; v++) {
                const RhdStepInfo& s = steps[v][k];
                const u32 phys = CarrierOfVirtual(r, v);
                for (u32 u = s.txSegStart; u < s.txSegEnd; u++) {
                    sent[phys][u] = hold[phys][u];
                    hold[phys][u] = 0;
                }
            }
            for (u32 v = 0; v < P; v++) {
                const RhdStepInfo& s = steps[v][k];
                const u32 phys = CarrierOfVirtual(r, v);
                const u32 peerPhys = CarrierOfVirtual(r, s.peer);
                for (u32 u = s.rxSegStart; u < s.rxSegEnd; u++) {
                    hold[phys][u] += sent[peerPhys][u];
                }
            }
        }
        // 终态：每 unit 在 owner（虚拟 vOf(u) 的承载）处计数 == p
        for (u32 u = 0; u < p; u++) {
            const u32 v = (u < 2 * r) ? u / 2 : u - r;
            const u32 owner = CarrierOfVirtual(r, v);
            EXPECT_EQ(hold[owner][u], p) << "p=" << p << " unit=" << u;
        }
    }
}

// RS 镜像一致性：每步每对收发互为镜像（tx(v)==rx(peer)）、peer 互指、无孤儿步骤
TEST(GenRhdReduceScatterStepsTest, RsMirrorConsistency)
{
    for (u32 p : {2u, 3u, 5u, 7u}) {
        const u32 r = p - LargestPow2Le(p);
        const u32 P = p - r;
        std::vector<std::vector<RhdStepInfo>> steps(P);
        for (u32 v = 0; v < P; v++) {
            const RhdPart1Info info = MakeInfoForVirtual(p, v);
            ASSERT_EQ(GenRhdReduceScatterSteps(info, steps[v]), HcclResult::HCCL_SUCCESS);
        }
        for (u32 v = 0; v < P; v++) {
            for (const RhdStepInfo& s : steps[v]) {
                ASSERT_LT(s.peer, P) << "孤儿步骤：对端越界";
                const RhdStepInfo& ps = steps[s.peer][s.step];
                EXPECT_EQ(ps.peer, v) << "peer 互指";
                EXPECT_EQ(ps.rxSegStart, s.txSegStart) << "tx(v) == rx(peer)";
                EXPECT_EQ(ps.rxSegEnd, s.txSegEnd);
                EXPECT_EQ(ps.txSegStart, s.rxSegStart) << "tx(peer) == rx(v)";
                EXPECT_EQ(ps.txSegEnd, s.rxSegEnd);
                EXPECT_EQ(ps.nSlices, s.nSlices);
            }
        }
    }
}

// RS 区段公式：p=8（r=0，unit=恒等）与手推表逐步一致（含步数/步序/相位/nSlices）
TEST(GenRhdReduceScatterStepsTest, RsSegmentFormula)
{
    struct Row {
        u32 v, k, peer, txS, txE, rxS, rxE;
    };
    // 手推表：k=0 dist=4（v<4: tx=[4,8)；v≥4: tx=[0,4)）；k=1 dist=2；k=2 dist=1
    const std::vector<Row> rows = {
        {0, 0, 4, 4, 8, 0, 4}, {4, 0, 0, 0, 4, 4, 8}, {1, 0, 5, 4, 8, 0, 4}, {5, 0, 1, 0, 4, 4, 8},
        {0, 1, 2, 2, 4, 0, 2}, {2, 1, 0, 0, 2, 2, 4}, {1, 1, 3, 2, 4, 0, 2}, {5, 1, 7, 6, 8, 4, 6},
        {0, 2, 1, 1, 2, 0, 1}, {1, 2, 0, 0, 1, 1, 2}, {6, 2, 7, 7, 8, 6, 7}, {7, 2, 6, 6, 7, 7, 8},
    };
    ASSERT_EQ(GetRhdStepNum(8u), 3u);
    for (const Row& row : rows) {
        const RhdPart1Info info = MakeInfoForVirtual(8, row.v);
        std::vector<RhdStepInfo> steps;
        ASSERT_EQ(GenRhdReduceScatterSteps(info, steps), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(steps.size(), 3u);
        ASSERT_LT(row.k, steps.size());
        const RhdStepInfo& s = steps[row.k];
        EXPECT_EQ(s.step, row.k);
        EXPECT_EQ(s.phase, RhdPhase::MAIN_RS);
        EXPECT_EQ(s.peer, row.peer) << "v=" << row.v << " k=" << row.k;
        EXPECT_EQ(s.nSlices, 8u >> (row.k + 1));
        EXPECT_EQ(s.txSegStart, row.txS) << "v=" << row.v << " k=" << row.k;
        EXPECT_EQ(s.txSegEnd, row.txE);
        EXPECT_EQ(s.rxSegStart, row.rxS);
        EXPECT_EQ(s.rxSegEnd, row.rxE);
    }
}

// RS 参数校验：info.effectiveRankSize == 0 / info.virtualRank >= P → HCCL_E_PARA（手造 POD）
TEST(GenRhdReduceScatterStepsTest, RsParamValidation)
{
    RhdPart1Info info{};
    std::vector<RhdStepInfo> steps;
    EXPECT_EQ(GenRhdReduceScatterSteps(info, steps), HcclResult::HCCL_E_PARA); // P == 0
    info.effectiveRankSize = 4;
    info.r = 0;
    info.virtualRank = 4; // >= P
    EXPECT_EQ(GenRhdReduceScatterSteps(info, steps), HcclResult::HCCL_E_PARA);
    EXPECT_TRUE(steps.empty());
}

// ==================== GenRhdAllGatherSteps ====================

// AG 真语义模拟：AG 输入为各自份额——各物理 rank 仅持自身份额（+1 于 own unit）；
// pre 用 SHARE 段（AG 族：仅需被合并方份额）并入代表；主体副本发送 + SUM，
// 终态：每 virtual 承载者对每个 unit 计数 == 1（每份额恰好聚合一次，重复/缺失即 2/0）。
TEST(GenRhdAllGatherStepsTest, AgFinalState)
{
    for (u32 p : {2u, 3u, 4u, 5u, 6u, 7u, 9u, 17u, 32u}) {
        const u32 r = p - LargestPow2Le(p);
        const u32 P = p - r;
        std::vector<std::vector<RhdStepInfo>> steps(P);
        for (u32 v = 0; v < P; v++) {
            const RhdPart1Info info = MakeInfoForVirtual(p, v);
            ASSERT_EQ(GenRhdAllGatherSteps(info, steps[v]), HcclResult::HCCL_SUCCESS) << "p=" << p << " v=" << v;
            ASSERT_EQ(steps[v].size(), static_cast<size_t>(GetRhdStepNum(P)));
        }
        // 各自份额：物理 rank j 持自身槽位 +1
        std::vector<std::vector<u32>> hold(p, std::vector<u32>(p, 0));
        for (u32 j = 0; j < p; j++) {
            hold[j][RhdUnitStartOfRank(p, j)] = 1;
        }
        // pre(SHARE)：被合并方份额并入代表，被合并方退出
        for (u32 t = 0; t < r; t++) {
            RhdPart1Info mergedInfo;
            ASSERT_EQ(CalcRhdPart1Info(p, t + r, INVALID_VALUE_RANKID, mergedInfo), HcclResult::HCCL_SUCCESS);
            RhdStepInfo pre, post;
            ASSERT_EQ(
                GenRhdPrePostSteps(mergedInfo, pre, post, RhdSegMode::SHARE, RhdSegMode::FULL),
                HcclResult::HCCL_SUCCESS);
            for (u32 u = pre.txSegStart; u < pre.txSegEnd; u++) {
                hold[t][u] += hold[t + r][u];
                hold[t + r][u] = 0;
            }
        }
        // 主体：AG 副本发送（不清零），rx 段累加对端送入（承载：配对=代表，block=v+r）
        const u32 nSteps = GetRhdStepNum(P);
        for (u32 k = 0; k < nSteps; k++) {
            std::vector<std::vector<u32>> sent(p, std::vector<u32>(p, 0));
            for (u32 v = 0; v < P; v++) {
                const RhdStepInfo& s = steps[v][k];
                const u32 phys = CarrierOfVirtual(r, v);
                for (u32 u = s.txSegStart; u < s.txSegEnd; u++) {
                    sent[phys][u] = hold[phys][u]; // 副本
                }
            }
            for (u32 v = 0; v < P; v++) {
                const RhdStepInfo& s = steps[v][k];
                const u32 phys = CarrierOfVirtual(r, v);
                const u32 peerPhys = CarrierOfVirtual(r, s.peer);
                for (u32 u = s.rxSegStart; u < s.rxSegEnd; u++) {
                    hold[phys][u] += sent[peerPhys][u];
                }
            }
        }
        // 终态：每 virtual 承载者持有全量份额且每 unit 计数恰为 1
        for (u32 v = 0; v < P; v++) {
            const u32 phys = CarrierOfVirtual(r, v);
            for (u32 u = 0; u < p; u++) {
                EXPECT_EQ(hold[phys][u], 1u) << "p=" << p << " v=" << v << " unit=" << u;
            }
        }
    }
}

// AG 区段公式：p=8（r=0）手推表（k=0 dist=1 tx=[heldStart,+1)；k=1 dist=2；k=2 dist=4）
TEST(GenRhdAllGatherStepsTest, AgSegmentFormula)
{
    struct Row {
        u32 v, k, peer, txS, txE, rxS, rxE;
    };
    const std::vector<Row> rows = {
        {0, 0, 1, 0, 1, 1, 2}, {1, 0, 0, 1, 2, 0, 1}, {5, 0, 4, 5, 6, 4, 5}, {7, 0, 6, 7, 8, 6, 7},
        {0, 1, 2, 0, 2, 2, 4}, {5, 1, 7, 4, 6, 6, 8}, {0, 2, 4, 0, 4, 4, 8}, {7, 2, 3, 4, 8, 0, 4},
    };
    ASSERT_EQ(GetRhdStepNum(8u), 3u);
    for (const Row& row : rows) {
        const RhdPart1Info info = MakeInfoForVirtual(8, row.v);
        std::vector<RhdStepInfo> steps;
        ASSERT_EQ(GenRhdAllGatherSteps(info, steps), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(steps.size(), 3u);
        ASSERT_LT(row.k, steps.size());
        const RhdStepInfo& s = steps[row.k];
        EXPECT_EQ(s.step, row.k);
        EXPECT_EQ(s.phase, RhdPhase::MAIN_AG);
        EXPECT_EQ(s.peer, row.peer) << "v=" << row.v << " k=" << row.k;
        EXPECT_EQ(s.nSlices, 1u << row.k);
        EXPECT_EQ(s.txSegStart, row.txS) << "v=" << row.v << " k=" << row.k;
        EXPECT_EQ(s.txSegEnd, row.txE);
        EXPECT_EQ(s.rxSegStart, row.rxS);
        EXPECT_EQ(s.rxSegEnd, row.rxE);
    }
}

// AG 参数校验（手造 POD）
TEST(GenRhdAllGatherStepsTest, AgParamValidation)
{
    RhdPart1Info info{};
    std::vector<RhdStepInfo> steps;
    EXPECT_EQ(GenRhdAllGatherSteps(info, steps), HcclResult::HCCL_E_PARA); // P == 0
    info.effectiveRankSize = 4;
    info.r = 0;
    info.virtualRank = 4; // >= P
    EXPECT_EQ(GenRhdAllGatherSteps(info, steps), HcclResult::HCCL_E_PARA);
    EXPECT_TRUE(steps.empty());
}

// AG 与 RS 相位镜像：AG 第 k 步 == RS 第 (⌊log₂P⌋−1−k) 步的时间反演
// （AG tx == RS rx、AG rx == RS tx、对端一致）
TEST(GenRhdAllGatherStepsTest, AgRsMirror)
{
    for (u32 p : {2u, 3u, 5u, 7u}) {
        const u32 r = p - LargestPow2Le(p);
        const u32 P = p - r;
        const u32 n = GetRhdStepNum(P);
        for (u32 v = 0; v < P; v++) {
            const RhdPart1Info info = MakeInfoForVirtual(p, v);
            std::vector<RhdStepInfo> rs;
            std::vector<RhdStepInfo> ag;
            ASSERT_EQ(GenRhdReduceScatterSteps(info, rs), HcclResult::HCCL_SUCCESS);
            ASSERT_EQ(GenRhdAllGatherSteps(info, ag), HcclResult::HCCL_SUCCESS);
            ASSERT_EQ(rs.size(), static_cast<size_t>(n));
            ASSERT_EQ(ag.size(), static_cast<size_t>(n));
            for (u32 k = 0; k < n; k++) {
                const RhdStepInfo& a = ag[k];
                const RhdStepInfo& s = rs[n - 1 - k];
                EXPECT_EQ(a.peer, s.peer) << "镜像步对端一致 p=" << p << " v=" << v << " k=" << k;
                EXPECT_EQ(a.txSegStart, s.rxSegStart) << "AG tx == RS rx";
                EXPECT_EQ(a.txSegEnd, s.rxSegEnd);
                EXPECT_EQ(a.rxSegStart, s.txSegStart) << "AG rx == RS tx";
                EXPECT_EQ(a.rxSegEnd, s.txSegEnd);
            }
        }
    }
}

// ==================== GenRhdPrePostSteps + 字节换算辅助 ====================

// 2^N：pre/post 均为空步骤（nSlices==0、区段 [0,0)、无对端），与段模式无关
TEST(GenRhdPrePostStepsTest, PrePostPowerOfTwo)
{
    for (u32 p : {2u, 4u, 8u, 16u}) {
        for (u32 j = 0; j < p; j++) {
            RhdPart1Info info;
            ASSERT_EQ(CalcRhdPart1Info(p, j, INVALID_VALUE_RANKID, info), HcclResult::HCCL_SUCCESS);
            RhdStepInfo pre, post;
            ASSERT_EQ(
                GenRhdPrePostSteps(info, pre, post, RhdSegMode::FULL, RhdSegMode::FULL), HcclResult::HCCL_SUCCESS);
            EXPECT_EQ(pre.phase, RhdPhase::PRE);
            EXPECT_EQ(post.phase, RhdPhase::POST);
            EXPECT_EQ(pre.step, 0u);
            EXPECT_EQ(post.step, 0u);
            EXPECT_EQ(pre.nSlices, 0u);
            EXPECT_EQ(pre.txSegStart, 0u);
            EXPECT_EQ(pre.txSegEnd, 0u);
            EXPECT_EQ(pre.rxSegStart, 0u);
            EXPECT_EQ(pre.rxSegEnd, 0u);
            EXPECT_EQ(post.nSlices, 0u);
            EXPECT_EQ(post.txSegStart, 0u);
            EXPECT_EQ(post.txSegEnd, 0u);
            EXPECT_EQ(post.rxSegStart, 0u);
            EXPECT_EQ(post.rxSegEnd, 0u);
        }
    }
}

// 非 2^N × root 全扫描 × 三角色（代表/被合并方/block）× SHARE/FULL 双模式：
// pre 被合并方发送（SHARE=自身份额段 / FULL=[0,p) 全量段）、代表为接收镜像；post 反向；block 双向均空
TEST(GenRhdPrePostStepsTest, PrePostDirection)
{
    for (u32 p : {5u, 7u, 13u}) {
        for (u32 root = 0; root < p; root++) {
            for (u32 j = 0; j < p; j++) {
                RhdPart1Info info;
                ASSERT_EQ(CalcRhdPart1Info(p, j, root, info), HcclResult::HCCL_SUCCESS);
                for (const RhdSegMode mode : {RhdSegMode::SHARE, RhdSegMode::FULL}) {
                    RhdStepInfo pre, post;
                    ASSERT_EQ(GenRhdPrePostSteps(info, pre, post, mode, mode), HcclResult::HCCL_SUCCESS);
                    EXPECT_EQ(pre.phase, RhdPhase::PRE);
                    EXPECT_EQ(post.phase, RhdPhase::POST);
                    if (!info.inPart1) { // block 成员：双向均空
                        EXPECT_EQ(pre.nSlices, 0u);
                        EXPECT_EQ(post.nSlices, 0u);
                        EXPECT_EQ(pre.peer, INVALID_VALUE_RANKID);
                        EXPECT_EQ(post.peer, INVALID_VALUE_RANKID);
                        EXPECT_EQ(pre.txSegEnd - pre.txSegStart + pre.rxSegEnd - pre.rxSegStart, 0u);
                        continue;
                    }
                    // 双方引用同一段：SHARE=被合并方自身份额槽位（恒 1 unit）；FULL=[0, p) 全量段
                    const u32 mergedSlot
                        = info.isRepresentative ? RhdUnitStartOfRank(p, info.pairedAlgRank) : info.unitStart;
                    const u32 segStart = (mode == RhdSegMode::FULL) ? 0 : mergedSlot;
                    const u32 segEnd = (mode == RhdSegMode::FULL) ? p : mergedSlot + 1;
                    const u32 segSlices = (mode == RhdSegMode::FULL) ? p : 1;
                    EXPECT_EQ(pre.peer, info.pairedAlgRank);
                    EXPECT_EQ(post.peer, info.pairedAlgRank);
                    if (info.isRepresentative) { // 代表：pre 接收镜像 / post 发送
                        EXPECT_EQ(pre.nSlices, segSlices);
                        EXPECT_EQ(pre.txSegStart, 0u);
                        EXPECT_EQ(pre.txSegEnd, 0u);
                        EXPECT_EQ(pre.rxSegStart, segStart);
                        EXPECT_EQ(pre.rxSegEnd, segEnd);
                        EXPECT_EQ(post.nSlices, segSlices);
                        EXPECT_EQ(post.txSegStart, segStart);
                        EXPECT_EQ(post.txSegEnd, segEnd);
                        EXPECT_EQ(post.rxSegStart, 0u);
                        EXPECT_EQ(post.rxSegEnd, 0u);
                    } else { // 被合并方：pre 发送 / post 接收
                        EXPECT_EQ(pre.nSlices, segSlices);
                        EXPECT_EQ(pre.txSegStart, segStart);
                        EXPECT_EQ(pre.txSegEnd, segEnd);
                        EXPECT_EQ(pre.rxSegStart, 0u);
                        EXPECT_EQ(pre.rxSegEnd, 0u);
                        EXPECT_EQ(post.nSlices, segSlices);
                        EXPECT_EQ(post.txSegStart, 0u);
                        EXPECT_EQ(post.txSegEnd, 0u);
                        EXPECT_EQ(post.rxSegStart, segStart);
                        EXPECT_EQ(post.rxSegEnd, segEnd);
                    }
                }
            }
        }
    }
}

// pre/post 段模式：SHARE=被合并方份额段（1 unit）；FULL=[0,p) 全量段（RS 族 pre / AG 族 post 按语义取）
TEST(GenRhdPrePostStepsTest, SegMode)
{
    const u32 p = 5; // r=1：pair (0,1)，被合并方（默认低位代表）槽位 = 1
    for (u32 j = 0; j < 2; j++) {
        RhdPart1Info info;
        ASSERT_EQ(CalcRhdPart1Info(p, j, INVALID_VALUE_RANKID, info), HcclResult::HCCL_SUCCESS);
        ASSERT_TRUE(info.inPart1);
        RhdStepInfo pre, post;
        // SHARE × SHARE：区段 = 被合并方份额段，nSlices = 1
        ASSERT_EQ(GenRhdPrePostSteps(info, pre, post, RhdSegMode::SHARE, RhdSegMode::SHARE), HcclResult::HCCL_SUCCESS);
        if (info.isRepresentative) {
            EXPECT_EQ(pre.rxSegStart, 1u);
            EXPECT_EQ(pre.rxSegEnd, 2u);
            EXPECT_EQ(post.txSegStart, 1u);
            EXPECT_EQ(post.txSegEnd, 2u);
        } else {
            EXPECT_EQ(pre.txSegStart, 1u);
            EXPECT_EQ(pre.txSegEnd, 2u);
            EXPECT_EQ(post.rxSegStart, 1u);
            EXPECT_EQ(post.rxSegEnd, 2u);
        }
        EXPECT_EQ(pre.nSlices, 1u);
        EXPECT_EQ(post.nSlices, 1u);
        // FULL × FULL：区段 = [0, p) 全量段，nSlices = p
        ASSERT_EQ(GenRhdPrePostSteps(info, pre, post, RhdSegMode::FULL, RhdSegMode::FULL), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(pre.nSlices, p);
        EXPECT_EQ(post.nSlices, p);
        if (info.isRepresentative) {
            EXPECT_EQ(pre.rxSegStart, 0u);
            EXPECT_EQ(pre.rxSegEnd, p);
            EXPECT_EQ(post.txSegStart, 0u);
            EXPECT_EQ(post.txSegEnd, p);
        } else {
            EXPECT_EQ(pre.txSegStart, 0u);
            EXPECT_EQ(pre.txSegEnd, p);
            EXPECT_EQ(post.rxSegStart, 0u);
            EXPECT_EQ(post.rxSegEnd, p);
        }
        // RS 族组合 pre=FULL × post=SHARE：pre 全量、post 份额
        ASSERT_EQ(GenRhdPrePostSteps(info, pre, post, RhdSegMode::FULL, RhdSegMode::SHARE), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(pre.nSlices, p);
        EXPECT_EQ(post.nSlices, 1u);
        if (info.isRepresentative) {
            EXPECT_EQ(pre.rxSegStart, 0u);
            EXPECT_EQ(pre.rxSegEnd, p);
            EXPECT_EQ(post.txSegStart, 1u);
            EXPECT_EQ(post.txSegEnd, 2u);
        } else {
            EXPECT_EQ(pre.txSegStart, 0u);
            EXPECT_EQ(pre.txSegEnd, p);
            EXPECT_EQ(post.rxSegStart, 1u);
            EXPECT_EQ(post.rxSegEnd, 2u);
        }
    }
}

// 均匀布局区段 → 字节：含/不含末 unit、跨末 unit、tailSize==sliceSize 退化恒等
TEST(RhdSegBytesTest, BytesUniform)
{
    const u64 sliceSize = 100;
    const u64 tailSize = 60;
    const u32 tailUnit = 4;                                                    // p=5
    RhdSegBytes b = RhdSegToBytesUniform(4, 5, sliceSize, tailSize, tailUnit); // 含末 unit
    EXPECT_EQ(b.offset, 400u);
    EXPECT_EQ(b.len, 60u);
    b = RhdSegToBytesUniform(2, 5, sliceSize, tailSize, tailUnit); // 跨末 unit
    EXPECT_EQ(b.offset, 200u);
    EXPECT_EQ(b.len, 260u);
    b = RhdSegToBytesUniform(1, 3, sliceSize, tailSize, tailUnit); // 不含末 unit：槽数×sliceSize
    EXPECT_EQ(b.offset, 100u);
    EXPECT_EQ(b.len, 200u);
    // 退化 tailSize == sliceSize：含/不含末 unit 两路径字节观恒等
    b = RhdSegToBytesUniform(4, 5, sliceSize, sliceSize, tailUnit);
    EXPECT_EQ(b.offset, 400u);
    EXPECT_EQ(b.len, 100u);
    b = RhdSegToBytesUniform(2, 5, sliceSize, sliceSize, tailUnit);
    EXPECT_EQ(b.offset, 200u);
    EXPECT_EQ(b.len, 300u);
}

// σ 反查构建 unit 尺寸表 + 变长契约与均匀契约字节观一致
TEST(RhdSegBytesTest, BuildUnitSizesAndBytesAllRank)
{
    // p=6（r=2）非 2^N：unitSizes.size()==p 且 unitSizes[σ(j)]==allRankSliceSize[j]
    const u32 p6 = 6;
    const std::vector<u64> allRank6 = {10, 20, 30, 40, 50, 60};
    std::vector<u64> unitSizes6;
    ASSERT_EQ(RhdBuildUnitSizes(p6, allRank6, unitSizes6), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(unitSizes6.size(), static_cast<size_t>(p6));
    for (u32 j = 0; j < p6; j++) {
        EXPECT_EQ(unitSizes6[RhdUnitStartOfRank(p6, j)], allRank6[j]) << "σ 反查 j=" << j;
    }
    // p=5（r=1）尾块变长契约与均匀契约在相同布局下输出一致
    const std::vector<u64> allRank5 = {100, 100, 100, 100, 60};
    std::vector<u64> unitSizes5;
    ASSERT_EQ(RhdBuildUnitSizes(5, allRank5, unitSizes5), HcclResult::HCCL_SUCCESS);
    const std::vector<std::pair<u32, u32>> segs = {{0, 1}, {1, 3}, {2, 5}, {4, 5}, {0, 5}};
    for (const auto& seg : segs) {
        const RhdSegBytes a = RhdSegToBytesAllRank(seg.first, seg.second, unitSizes5);
        const RhdSegBytes u = RhdSegToBytesUniform(seg.first, seg.second, 100, 60, 4);
        EXPECT_EQ(a.offset, u.offset) << "seg=[" << seg.first << "," << seg.second << ")";
        EXPECT_EQ(a.len, u.len);
    }
}

// 字节换算参数校验：unitStart>=unitEnd / unitSizes 越界 → {0,0}；BuildUnitSizes 尺寸不符 → E_PARA
TEST(RhdSegBytesTest, BytesParamValidation)
{
    RhdSegBytes b = RhdSegToBytesUniform(3, 3, 100, 60, 4);
    EXPECT_EQ(b.offset, 0u);
    EXPECT_EQ(b.len, 0u);
    b = RhdSegToBytesUniform(4, 2, 100, 60, 4);
    EXPECT_EQ(b.offset, 0u);
    EXPECT_EQ(b.len, 0u);
    const std::vector<u64> unitSizes = {100, 100, 100};
    b = RhdSegToBytesAllRank(1, 4, unitSizes); // unitEnd=4 > size=3，不越界
    EXPECT_EQ(b.offset, 0u);
    EXPECT_EQ(b.len, 0u);
    b = RhdSegToBytesAllRank(2, 2, unitSizes);
    EXPECT_EQ(b.offset, 0u);
    EXPECT_EQ(b.len, 0u);
    std::vector<u64> out;
    EXPECT_EQ(RhdBuildUnitSizes(5, {100, 100}, out), HcclResult::HCCL_E_PARA); // 防越界读
    EXPECT_TRUE(out.empty());
}
