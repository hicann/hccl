/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Scatter RHD 主体步骤生成 UT：root 持有者链单向折半 + 非 2 次幂后处理 + skipPost 融合直通。
// part1 划分/σ 映射/字节换算的公共断言归 template_utils 公共 UT，此处只断言 Scatter 专属
// 行为；虚拟序/unit 映射的期望值在测试侧独立推导。
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "rhd_scatter_step.h"
#include "template_utils.h"

using namespace ops_hccl;

namespace {
// 2^⌊log₂p⌋（与实现独立推导）
u32 LargestPow2Le(u32 p)
{
    u32 hp = 1;
    while (hp <= p / 2) {
        hp *= 2;
    }
    return hp;
}

u32 PairNumOf(u32 p) { return p - LargestPow2Le(p); }

// 虚拟序 v → unit 映射（测试侧独立推导）
u32 VUnitStart(u32 r, u32 v) { return (v < r) ? 2 * v : v + r; }

u32 VUnitEnd(u32 r, u32 b) { return (b <= r) ? 2 * b : b + r; }

using StepTables = std::vector<std::vector<RhdScatterStep>>;

void GenAllTables(u32 p, u32 root, StepTables& tables)
{
    tables.assign(p, {});
    for (u32 j = 0; j < p; j++) {
        ASSERT_EQ(GenRhdScatterSteps(p, root, j, tables[j]), HcclResult::HCCL_SUCCESS)
            << "p=" << p << " root=" << root << " j=" << j;
    }
}

// 本 rank 的主体有效序（经 CalcRhdPart1Info，其行为由公共 UT 独立保证）
u32 VirtualOf(u32 p, u32 root, u32 j)
{
    RhdPart1Info info;
    EXPECT_EQ(CalcRhdPart1Info(p, j, root, info), HcclResult::HCCL_SUCCESS);
    return info.virtualRank;
}

// unit → 虚拟槽位映射（反解单连通断言用）
std::vector<u32> UnitToVirtual(u32 p)
{
    const u32 r = PairNumOf(p);
    const u32 eff = p - r;
    std::vector<u32> u2v(p, INVALID_U32);
    for (u32 v = 0; v < eff; v++) {
        const u32 cnt = (v < r) ? 2u : 1u;
        for (u32 c = 0; c < cnt; c++) {
            u2v[VUnitStart(r, v) + c] = v;
        }
    }
    return u2v;
}

// unit 的份额归属 rank（σ 单射反查）
u32 OwnerOfUnit(u32 p, u32 u)
{
    for (u32 j = 0; j < p; j++) {
        if (RhdUnitStartOfRank(p, j) == u) {
            return j;
        }
    }
    return INVALID_U32;
}

// 全局 (phase, step) 序收集接收步：跨步依赖要求发送方数据已就绪（步内窗口互不相交，
// 同步内顺序无关）；按 rank 序逐表模拟会在"接收方 j < 其上游 j"时读到未就绪数据
struct RecvStep {
    int phase;
    u32 step;
    u32 rank;
    RhdScatterStep s;
};

std::vector<RecvStep> CollectRecvSteps(const StepTables& tables)
{
    std::vector<RecvStep> recvs;
    for (u32 j = 0; j < tables.size(); j++) {
        for (const auto& s : tables[j]) {
            if (!s.isSend) {
                recvs.push_back({static_cast<int>(s.phase), s.step, j, s});
            }
        }
    }
    std::sort(recvs.begin(), recvs.end(), [](const RecvStep& a, const RecvStep& b) {
        return std::make_pair(a.phase, a.step) < std::make_pair(b.phase, b.step);
    });
    return recvs;
}

// 步骤逐字段等价（skipPost 方向用例：主体步不受影响、无重排）
bool SameStep(const RhdScatterStep& a, const RhdScatterStep& b)
{
    return a.phase == b.phase && a.step == b.step && a.isSend == b.isSend && a.peerAlgRank == b.peerAlgRank
           && a.segStartUnit == b.segStartUnit && a.segEndUnit == b.segEndUnit;
}
} // namespace

// p=8、root=3（2^N，r=0，unit 空间 = 物理槽恒等）：逐步与 3.2 验证示例完全一致
TEST(GenRhdScatterStepsTest, MainBodyPowerOfTwo)
{
    // rank3 每步均为 holder（无空转步）：k=0 发 [4,8) 给 7；k=1 发 [0,2) 给 1；k=2 发 [2,3) 给 2
    std::vector<RhdScatterStep> steps;
    ASSERT_EQ(GenRhdScatterSteps(8, 3, 3, steps), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(steps.size(), 3u) << "rank3 为主链 holder，3 步全有任务";
    struct ExpectStep {
        u32 peer;
        u32 segStart;
        u32 segEnd;
    };
    const std::vector<ExpectStep> expect = {{7u, 4u, 8u}, {1u, 0u, 2u}, {2u, 2u, 3u}};
    for (u32 k = 0; k < expect.size(); k++) {
        EXPECT_EQ(steps[k].phase, RhdScatterStep::Phase::MAIN) << "k=" << k;
        EXPECT_EQ(steps[k].step, k) << "k=" << k;
        EXPECT_TRUE(steps[k].isSend) << "k=" << k;
        EXPECT_EQ(steps[k].peerAlgRank, expect[k].peer) << "k=" << k;
        EXPECT_EQ(steps[k].segStartUnit, expect[k].segStart) << "k=" << k;
        EXPECT_EQ(steps[k].segEndUnit, expect[k].segEnd) << "k=" << k;
    }

    // rank7：k=0 收 [4,8)、k=1 发 [4,6) 给 5、k=2 发 [6,7) 给 6
    std::vector<RhdScatterStep> steps7;
    ASSERT_EQ(GenRhdScatterSteps(8, 3, 7, steps7), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(steps7.size(), 3u);
    EXPECT_EQ(steps7[0].phase, RhdScatterStep::Phase::MAIN);
    EXPECT_EQ(steps7[0].step, 0u);
    EXPECT_FALSE(steps7[0].isSend);
    EXPECT_EQ(steps7[0].peerAlgRank, 3u);
    EXPECT_EQ(steps7[0].segStartUnit, 4u);
    EXPECT_EQ(steps7[0].segEndUnit, 8u);
    EXPECT_TRUE(steps7[1].isSend);
    EXPECT_EQ(steps7[1].peerAlgRank, 5u);
    EXPECT_EQ(steps7[1].segStartUnit, 4u);
    EXPECT_EQ(steps7[1].segEndUnit, 6u);
    EXPECT_TRUE(steps7[2].isSend);
    EXPECT_EQ(steps7[2].peerAlgRank, 6u);
    EXPECT_EQ(steps7[2].segStartUnit, 6u);
    EXPECT_EQ(steps7[2].segEndUnit, 7u);

    // rank0：仅 k=2 一步（k=0/k=1 空转过滤）：收 [0,1) 自 1
    std::vector<RhdScatterStep> steps0;
    ASSERT_EQ(GenRhdScatterSteps(8, 3, 0, steps0), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(steps0.size(), 1u) << "空转步不入表";
    EXPECT_EQ(steps0[0].phase, RhdScatterStep::Phase::MAIN);
    EXPECT_EQ(steps0[0].step, 2u);
    EXPECT_FALSE(steps0[0].isSend);
    EXPECT_EQ(steps0[0].peerAlgRank, 1u);
    EXPECT_EQ(steps0[0].segStartUnit, 0u);
    EXPECT_EQ(steps0[0].segEndUnit, 1u);
}

// p ∈ {2..13,16,32} × 全部 root：模拟执行步骤表（只记区段归属），每 rank 终态恰好持有自己份额；
// 每步发送/接收互为 XOR 伙伴（虚拟序空间，距离 = P>>(k+1)）
TEST(GenRhdScatterStepsTest, AllRanksEndWithOwnSlice)
{
    const std::vector<u32> sizes = {2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 16u, 32u};
    for (u32 p : sizes) {
        for (u32 root = 0; root < p; root++) {
            StepTables tables;
            GenAllTables(p, root, tables);

            // 区段归属模拟：holds[rank][unit] = 是否持有 root 侧正确数据（push 模型，recv 步搬运）。
            // 按全局 (phase, step) 序执行，保证上游 rank 的接收先于下游消费
            std::vector<std::vector<bool>> holds(p, std::vector<bool>(p, false));
            for (u32 u = 0; u < p; u++) {
                holds[root][u] = true; // root 持有全部
            }
            for (const auto& r : CollectRecvSteps(tables)) {
                for (u32 u = r.s.segStartUnit; u < r.s.segEndUnit; u++) {
                    holds[r.rank][u] = holds[r.s.peerAlgRank][u];
                }
            }
            // 终态：每 rank 自己的 unit（σ(j)）持有 root 侧正确数据。root 是唯一数据源且持有者链
            // 保证其每步均为发送方（永不接收），holds 为发送保留式语义，"恰好一份"由
            // DataPlaneSimulation 的 +1 计数模式（每 unit 恰投递 owner 一次）与镜像断言覆盖
            for (u32 j = 0; j < p; j++) {
                EXPECT_TRUE(holds[j][RhdUnitStartOfRank(p, j)])
                    << "p=" << p << " root=" << root << " j=" << j << " 终态未持有自己份额的正确数据";
            }

            // 每步：发送/接收互为 XOR 伙伴，窗口内恰一发送一接收
            const u32 r = PairNumOf(p);
            const u32 eff = p - r;
            const u32 nSteps = GetRhdStepNum(eff);
            for (u32 k = 0; k < nSteps; k++) {
                const u32 dist = eff >> (k + 1);
                std::map<u32, u32> senders;   // v(发送方) -> v(接收方)
                std::map<u32, u32> receivers; // v(接收方) -> v(发送方)
                for (u32 j = 0; j < p; j++) {
                    for (const auto& s : tables[j]) {
                        if (s.phase != RhdScatterStep::Phase::MAIN || s.step != k) {
                            continue;
                        }
                        const u32 myV = VirtualOf(p, root, j);
                        const u32 peerV = VirtualOf(p, root, s.peerAlgRank);
                        if (s.isSend) {
                            senders[myV] = peerV;
                        } else {
                            receivers[myV] = peerV;
                        }
                    }
                }
                ASSERT_EQ(senders.size(), receivers.size()) << "p=" << p << " root=" << root << " k=" << k;
                EXPECT_EQ(senders.size(), eff / (2 * dist))
                    << "p=" << p << " root=" << root << " k=" << k << " 每窗口恰一对收发";
                for (const auto& kv : senders) {
                    EXPECT_EQ(kv.first ^ dist, kv.second) << "p=" << p << " root=" << root << " k=" << k;
                    ASSERT_TRUE(receivers.count(kv.second) > 0) << "p=" << p << " root=" << root << " k=" << k;
                    EXPECT_EQ(receivers[kv.second], kv.first) << "p=" << p << " root=" << root << " k=" << k;
                }
            }
        }
    }
}

// 防对端算错位/死锁：每个发送步必须在对端表中有镜像接收步（peerAlgRank 互指、区段完全相同），无孤儿
TEST(GenRhdScatterStepsTest, MirrorConsistency)
{
    const std::vector<u32> sizes = {2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 16u, 32u};
    for (u32 p : sizes) {
        for (u32 root = 0; root < p; root++) {
            StepTables tables;
            GenAllTables(p, root, tables);

            // 每 rank 每 (phase, step) 至多一步
            for (u32 j = 0; j < p; j++) {
                std::set<std::pair<int, u32>> seen;
                for (const auto& s : tables[j]) {
                    const auto key = std::make_pair(static_cast<int>(s.phase), s.step);
                    EXPECT_TRUE(seen.insert(key).second)
                        << "p=" << p << " root=" << root << " j=" << j << " 重复 (phase,step)";
                }
            }
            // 每步在对端表中有镜像（收发互补、互指、区段相同）
            for (u32 j = 0; j < p; j++) {
                for (const auto& s : tables[j]) {
                    bool found = false;
                    for (const auto& m : tables[s.peerAlgRank]) {
                        if (m.phase != s.phase || m.step != s.step) {
                            continue;
                        }
                        EXPECT_FALSE(found) << "p=" << p << " root=" << root << " j=" << j << " 镜像不唯一";
                        found = true;
                        EXPECT_NE(m.isSend, s.isSend) << "p=" << p << " root=" << root << " j=" << j;
                        EXPECT_EQ(m.peerAlgRank, j) << "p=" << p << " root=" << root << " j=" << j;
                        EXPECT_EQ(m.segStartUnit, s.segStartUnit) << "p=" << p << " root=" << root << " j=" << j;
                        EXPECT_EQ(m.segEndUnit, s.segEndUnit) << "p=" << p << " root=" << root << " j=" << j;
                    }
                    EXPECT_TRUE(found) << "p=" << p << " root=" << root << " j=" << j << " 存在无镜像的孤儿步骤";
                }
            }
        }
    }
}

// 数据内容验证：p ∈ {4,5,6,8,12} × 全部 root。溯源标记模式（slice j 每元素填 j+1）按步骤表做
// 字节级搬运（偏移/长度经 RhdSegToBytesUniform 换算），终态 rank i 持有值 i+1；
// 再用 +1 计数模式跑一遍（unit 投递到 owner 计 1，root 自有 unit 起始为 1）：多走=2、少走=0
TEST(GenRhdScatterStepsTest, DataPlaneSimulation)
{
    const u64 sliceSize = 16; // 每 unit 字节数（4 个 u32）
    const u64 wordsPerUnit = sliceSize / sizeof(u32);
    for (u32 p : {4u, 5u, 6u, 8u, 12u}) {
        for (u32 root = 0; root < p; root++) {
            StepTables tables;
            GenAllTables(p, root, tables);

            // 溯源标记模式：root 输入 slice j 填 j+1（slice j 位于 unit σ(j)）
            std::vector<std::vector<u32>> buf(p, std::vector<u32>(p * wordsPerUnit, 0));
            for (u32 j = 0; j < p; j++) {
                const u32 unit = RhdUnitStartOfRank(p, j);
                for (u64 w = 0; w < wordsPerUnit; w++) {
                    buf[root][unit * wordsPerUnit + w] = j + 1;
                }
            }
            // 字节级搬运模拟：偏移/长度经 RhdSegToBytesUniform 换算（顺带端到端覆盖换算）。
            // 按全局 (phase, step) 序执行，保证上游 rank 的接收先于下游消费
            for (const auto& r : CollectRecvSteps(tables)) {
                const RhdSegBytes sb
                    = RhdSegToBytesUniform(r.s.segStartUnit, r.s.segEndUnit, sliceSize, sliceSize, p - 1);
                ASSERT_GT(sb.len, 0u) << "p=" << p << " root=" << root << " j=" << r.rank;
                ASSERT_EQ(sb.len % sizeof(u32), 0u);
                for (u64 w = 0; w < sb.len / sizeof(u32); w++) {
                    buf[r.rank][sb.offset / sizeof(u32) + w] = buf[r.s.peerAlgRank][sb.offset / sizeof(u32) + w];
                }
            }
            // 终态：rank i 的输出区（自己 unit）值全为 i+1
            for (u32 i = 0; i < p; i++) {
                const u32 own = RhdUnitStartOfRank(p, i);
                const RhdSegBytes ob = RhdSegToBytesUniform(own, own + 1, sliceSize, sliceSize, p - 1);
                for (u64 w = 0; w < ob.len / sizeof(u32); w++) {
                    EXPECT_EQ(buf[i][ob.offset / sizeof(u32) + w], i + 1)
                        << "p=" << p << " root=" << root << " i=" << i;
                }
            }

            // +1 计数模式：unit 恰好投递到 owner 一次（root 自有 unit 从 1 起计；计数与顺序无关）
            std::vector<u32> count(p, 0);
            count[RhdUnitStartOfRank(p, root)] = 1;
            for (const auto& r : CollectRecvSteps(tables)) {
                for (u32 u = r.s.segStartUnit; u < r.s.segEndUnit; u++) {
                    if (OwnerOfUnit(p, u) == r.rank) {
                        count[u]++;
                    }
                }
            }
            for (u32 u = 0; u < p; u++) {
                EXPECT_EQ(count[u], 1u) << "p=" << p << " root=" << root << " unit=" << u;
            }
        }
    }
}

// 每步区段在 unit 空间单连通：segStartUnit < segEndUnit，且反解为单连通虚拟区段
// （unit→v 序列不回退、不跳跃——σ 细节由公共函数保证，此处验证消费正确性）
TEST(GenRhdScatterStepsTest, SegmentContiguity)
{
    const std::vector<u32> sizes = {2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 16u, 32u};
    for (u32 p : sizes) {
        const auto u2v = UnitToVirtual(p);
        for (u32 root = 0; root < p; root++) {
            StepTables tables;
            GenAllTables(p, root, tables);
            for (u32 j = 0; j < p; j++) {
                for (const auto& s : tables[j]) {
                    EXPECT_LT(s.segStartUnit, s.segEndUnit) << "p=" << p << " root=" << root << " j=" << j;
                    EXPECT_LE(s.segEndUnit, p) << "p=" << p << " root=" << root << " j=" << j;
                    for (u32 u = s.segStartUnit + 1; u < s.segEndUnit; u++) {
                        EXPECT_GE(u2v[u], u2v[u - 1])
                            << "p=" << p << " root=" << root << " j=" << j << " u=" << u << " unit 段回退";
                        EXPECT_LE(u2v[u], u2v[u - 1] + 1)
                            << "p=" << p << " root=" << root << " j=" << j << " u=" << u << " unit 段跳跃";
                    }
                }
            }
        }
    }
}

// root 落 part1 被合并方侧：root 以代表身份、按 CalcRhdPart1Info 的 virtualRank 参与主体
// （holder 链从 root 出发）；被顶掉的低位仅剩 POST 接收步——防误按物理 algRank 直算 vRoot。
// p=6/10：r=2，被合并方侧 = {2,3}；p=12：r=4，被合并方侧 = [4,8) 取 {4,5}
TEST(GenRhdScatterStepsTest, VirtualMapping)
{
    struct ExpectFirst {
        u32 p;
        u32 root;
        u32 peer;
        u32 segStart;
        u32 segEnd;
    };
    const std::vector<ExpectFirst> firsts = {
        {6u, 2u, 4u, 4u, 6u},   {6u, 3u, 5u, 4u, 6u},   {10u, 2u, 6u, 6u, 10u},
        {10u, 3u, 7u, 6u, 10u}, {12u, 4u, 8u, 8u, 12u}, {12u, 5u, 9u, 8u, 12u},
    };
    for (const auto& e : firsts) {
        const u32 r = PairNumOf(e.p);
        ASSERT_GE(r, 2u) << "p=" << e.p << " 需 r>=2 的非 2 次幂形态";
        ASSERT_LT(e.root, 2 * r) << "p=" << e.p << " root 须落 part1 内";
        ASSERT_GE(e.root, r) << "p=" << e.p << " root 须落被合并方侧（t+r 侧）";
        // root 是代表，且第 0 步即发送（vRoot 是自己 → holder 链从 root 出发）
        RhdPart1Info rootInfo;
        ASSERT_EQ(CalcRhdPart1Info(e.p, e.root, e.root, rootInfo), HcclResult::HCCL_SUCCESS);
        EXPECT_TRUE(rootInfo.inPart1) << "p=" << e.p << " root=" << e.root;
        EXPECT_TRUE(rootInfo.isRepresentative) << "p=" << e.p << " root=" << e.root;
        std::vector<RhdScatterStep> rootSteps;
        ASSERT_EQ(GenRhdScatterSteps(e.p, e.root, e.root, rootSteps), HcclResult::HCCL_SUCCESS);
        ASSERT_FALSE(rootSteps.empty()) << "p=" << e.p << " root=" << e.root;
        EXPECT_EQ(rootSteps.front().phase, RhdScatterStep::Phase::MAIN);
        EXPECT_EQ(rootSteps.front().step, 0u);
        EXPECT_TRUE(rootSteps.front().isSend) << "holder 链从 root 出发";
        EXPECT_EQ(rootSteps.front().peerAlgRank, e.peer) << "p=" << e.p << " root=" << e.root;
        EXPECT_EQ(rootSteps.front().segStartUnit, e.segStart) << "p=" << e.p << " root=" << e.root;
        EXPECT_EQ(rootSteps.front().segEndUnit, e.segEnd) << "p=" << e.p << " root=" << e.root;

        // 被顶掉的低位伙伴不参与主体：表仅一条 POST 接收步，对端为 root
        const u32 lowPeer = e.root - r;
        std::vector<RhdScatterStep> lowSteps;
        ASSERT_EQ(GenRhdScatterSteps(e.p, e.root, lowPeer, lowSteps), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(lowSteps.size(), 1u) << "p=" << e.p << " root=" << e.root;
        EXPECT_EQ(lowSteps[0].phase, RhdScatterStep::Phase::POST);
        EXPECT_FALSE(lowSteps[0].isSend);
        EXPECT_EQ(lowSteps[0].peerAlgRank, e.root);
    }
}

// 非 2^N：主体步数 = log2(P)；表尾恰一条 POST 步（方向/对端/区段与 GenRhdPrePostSteps 一致）；
// 2^N：无 POST 步
TEST(GenRhdScatterStepsTest, NonPowerOfTwoPostPhase)
{
    for (u32 p : {5u, 6u, 7u, 10u, 11u, 12u, 13u}) {
        const u32 r = PairNumOf(p);
        const u32 eff = p - r;
        const u32 nSteps = GetRhdStepNum(eff);
        for (u32 root = 0; root < p; root++) {
            StepTables tables;
            GenAllTables(p, root, tables);

            // 主体步序并集恰为 {0..log2(P)-1}
            std::set<u32> mainStepIdx;
            for (const auto& tb : tables) {
                for (const auto& s : tb) {
                    if (s.phase == RhdScatterStep::Phase::MAIN) {
                        mainStepIdx.insert(s.step);
                    }
                }
            }
            EXPECT_EQ(mainStepIdx.size(), nSteps) << "p=" << p << " root=" << root;
            for (u32 k = 0; k < nSteps; k++) {
                EXPECT_TRUE(mainStepIdx.count(k) > 0) << "p=" << p << " root=" << root << " k=" << k;
            }

            for (u32 j = 0; j < p; j++) {
                RhdPart1Info info;
                ASSERT_EQ(CalcRhdPart1Info(p, j, root, info), HcclResult::HCCL_SUCCESS);
                const auto& tb = tables[j];
                // MAIN 步序严格递增
                for (u32 a = 1; a < tb.size(); a++) {
                    if (tb[a].phase == RhdScatterStep::Phase::MAIN && tb[a - 1].phase == RhdScatterStep::Phase::MAIN) {
                        EXPECT_GT(tb[a].step, tb[a - 1].step) << "p=" << p << " root=" << root << " j=" << j;
                    }
                }
                const u32 postCnt = static_cast<u32>(std::count_if(tb.begin(), tb.end(), [](const RhdScatterStep& s) {
                    return s.phase == RhdScatterStep::Phase::POST;
                }));
                if (info.inPart1 && !info.isRepresentative) {
                    // 被合并方：不参与主体，表仅一条 POST 接收步
                    ASSERT_EQ(tb.size(), 1u) << "p=" << p << " root=" << root << " j=" << j;
                    EXPECT_EQ(tb[0].phase, RhdScatterStep::Phase::POST);
                    EXPECT_FALSE(tb[0].isSend);
                } else if (info.inPart1) {
                    // 代表：表尾恰一条 POST 发送步
                    ASSERT_GE(tb.size(), 2u) << "p=" << p << " root=" << root << " j=" << j;
                    EXPECT_EQ(postCnt, 1u) << "p=" << p << " root=" << root << " j=" << j;
                    EXPECT_EQ(tb.back().phase, RhdScatterStep::Phase::POST);
                    EXPECT_TRUE(tb.back().isSend);
                } else {
                    // block 成员：无 POST
                    EXPECT_EQ(postCnt, 0u) << "p=" << p << " root=" << root << " j=" << j;
                }
                // POST 步与 GenRhdPrePostSteps 的 post 一致（SHARE/SHARE：份额段模式）
                if (info.inPart1) {
                    RhdStepInfo pre;
                    RhdStepInfo post;
                    ASSERT_EQ(
                        GenRhdPrePostSteps(info, pre, post, RhdSegMode::SHARE, RhdSegMode::SHARE),
                        HcclResult::HCCL_SUCCESS)
                        << "p=" << p << " root=" << root << " j=" << j;
                    ASSERT_EQ(postCnt, 1u) << "p=" << p << " root=" << root << " j=" << j;
                    const auto& s = tb.back();
                    ASSERT_EQ(s.phase, RhdScatterStep::Phase::POST);
                    EXPECT_EQ(s.step, 0u);
                    EXPECT_EQ(s.peerAlgRank, post.peer) << "p=" << p << " root=" << root << " j=" << j;
                    if (info.isRepresentative) {
                        EXPECT_TRUE(s.isSend);
                        EXPECT_EQ(s.segStartUnit, post.txSegStart) << "p=" << p << " root=" << root << " j=" << j;
                        EXPECT_EQ(s.segEndUnit, post.txSegEnd) << "p=" << p << " root=" << root << " j=" << j;
                    } else {
                        EXPECT_FALSE(s.isSend);
                        EXPECT_EQ(s.segStartUnit, post.rxSegStart) << "p=" << p << " root=" << root << " j=" << j;
                        EXPECT_EQ(s.segEndUnit, post.rxSegEnd) << "p=" << p << " root=" << root << " j=" << j;
                    }
                }
            }
        }
    }
    // 2^N：无 POST 步
    for (u32 p : {2u, 4u, 8u, 16u}) {
        for (u32 root = 0; root < p; root++) {
            StepTables tables;
            GenAllTables(p, root, tables);
            for (u32 j = 0; j < p; j++) {
                for (const auto& s : tables[j]) {
                    EXPECT_EQ(s.phase, RhdScatterStep::Phase::MAIN) << "p=" << p << " root=" << root << " j=" << j;
                }
            }
        }
    }
}

// 防御性校验：RhdScatterParamValid 边界；p=1 合法退化（空表）；越界入参 HCCL_E_PARA
TEST(GenRhdScatterStepsTest, ParamValidation)
{
    EXPECT_FALSE(RhdScatterParamValid(0, 0));
    EXPECT_FALSE(RhdScatterParamValid(4, 4));
    EXPECT_TRUE(RhdScatterParamValid(1, 0));

    std::vector<RhdScatterStep> steps;
    steps.push_back(RhdScatterStep{}); // 预填脏数据验证 clear
    ASSERT_EQ(GenRhdScatterSteps(1, 0, 0, steps), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(steps.empty()) << "p=1 合法退化：无主体步（log2(1)=0）、2^N 无后处理";

    EXPECT_EQ(GenRhdScatterSteps(0, 0, 0, steps), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(GenRhdScatterSteps(4, 4, 0, steps), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(GenRhdScatterSteps(4, 0, 4, steps), HcclResult::HCCL_E_PARA);
}

// —— skipPost（fusedWithNextAgPhase：Scatter→AG 融合直通）方向用例 ——

// 非 2 次幂：skipPost=true 仅删 POST 步——主体步逐字段不变、无重排；被合并方（整表仅一条 POST
// 接收步）变空表，由模板 RunRHD 的退化早退兜住；代表表尾不再有 POST 发送步
TEST(GenRhdScatterStepsTest, SkipPostRemovesOnlyPostSteps)
{
    for (u32 p : {5u, 6u, 7u, 10u, 11u, 12u, 13u}) {
        for (u32 root = 0; root < p; root++) {
            for (u32 j = 0; j < p; j++) {
                std::vector<RhdScatterStep> withPost;
                std::vector<RhdScatterStep> skipped;
                ASSERT_EQ(GenRhdScatterSteps(p, root, j, withPost), HcclResult::HCCL_SUCCESS)
                    << "p=" << p << " root=" << root << " j=" << j;
                ASSERT_EQ(GenRhdScatterSteps(p, root, j, skipped, true), HcclResult::HCCL_SUCCESS)
                    << "p=" << p << " root=" << root << " j=" << j;

                // skipped 表 == withPost 表过滤 POST 后的剩余（主体步不改字段、不重排）
                std::vector<RhdScatterStep> expect;
                for (const auto& s : withPost) {
                    if (s.phase != RhdScatterStep::Phase::POST) {
                        expect.push_back(s);
                    }
                }
                ASSERT_EQ(skipped.size(), expect.size()) << "p=" << p << " root=" << root << " j=" << j;
                for (size_t k = 0; k < expect.size(); k++) {
                    EXPECT_TRUE(SameStep(skipped[k], expect[k]))
                        << "p=" << p << " root=" << root << " j=" << j << " k=" << k;
                }
                for (const auto& s : skipped) {
                    EXPECT_NE(s.phase, RhdScatterStep::Phase::POST)
                        << "p=" << p << " root=" << root << " j=" << j << " POST 步须被整体删除";
                }

                // 被合并方整表变空（σ 双份布局直通后其份额由相邻 AG 侧消费）
                RhdPart1Info info;
                ASSERT_EQ(CalcRhdPart1Info(p, j, root, info), HcclResult::HCCL_SUCCESS);
                if (info.inPart1 && !info.isRepresentative) {
                    EXPECT_TRUE(skipped.empty()) << "p=" << p << " root=" << root << " j=" << j;
                }
            }
        }
    }
}

// 2^N：无 POST 可删——skipPost=true 与默认路径逐 rank 逐字段完全一致（零影响）
TEST(GenRhdScatterStepsTest, SkipPostNoopForPowerOfTwo)
{
    for (u32 p : {2u, 4u, 8u, 16u, 32u}) {
        for (u32 root = 0; root < p; root++) {
            for (u32 j = 0; j < p; j++) {
                std::vector<RhdScatterStep> def;
                std::vector<RhdScatterStep> skipped;
                ASSERT_EQ(GenRhdScatterSteps(p, root, j, def), HcclResult::HCCL_SUCCESS);
                ASSERT_EQ(GenRhdScatterSteps(p, root, j, skipped, true), HcclResult::HCCL_SUCCESS);
                ASSERT_EQ(def.size(), skipped.size()) << "p=" << p << " root=" << root << " j=" << j;
                for (size_t k = 0; k < def.size(); k++) {
                    EXPECT_TRUE(SameStep(skipped[k], def[k]))
                        << "p=" << p << " root=" << root << " j=" << j << " k=" << k;
                }
            }
        }
    }
}

// 默认参数兼容：不传 skipPost（存量 3 调用方路径）与显式 false 逐字段等价
TEST(GenRhdScatterStepsTest, SkipPostDefaultFalseCompat)
{
    for (u32 p : {5u, 6u, 8u, 12u}) {
        for (u32 root = 0; root < p; root++) {
            for (u32 j = 0; j < p; j++) {
                std::vector<RhdScatterStep> def;
                std::vector<RhdScatterStep> expl;
                ASSERT_EQ(GenRhdScatterSteps(p, root, j, def), HcclResult::HCCL_SUCCESS);
                ASSERT_EQ(GenRhdScatterSteps(p, root, j, expl, false), HcclResult::HCCL_SUCCESS);
                ASSERT_EQ(def.size(), expl.size()) << "p=" << p << " root=" << root << " j=" << j;
                for (size_t k = 0; k < def.size(); k++) {
                    EXPECT_TRUE(SameStep(def[k], expl[k])) << "p=" << p << " root=" << root << " j=" << j << " k=" << k;
                }
            }
        }
    }
}

// 载体读取口 RhdFusedWithNextAg：bit1 置位才为 true；omnipipe 整体值 1（bit0）不误触发；位可组合
TEST(GenRhdScatterStepsTest, FusedFlagCarrierBit)
{
    TemplateDataParams tempAlgParams;
    tempAlgParams.localCopyFlag = 0;
    EXPECT_FALSE(RhdFusedWithNextAg(tempAlgParams));
    tempAlgParams.localCopyFlag = 1; // omnipipe localCopy 模式整体值
    EXPECT_FALSE(RhdFusedWithNextAg(tempAlgParams));
    tempAlgParams.localCopyFlag = RHD_FLAG_FUSED_WITH_NEXT_AG;
    EXPECT_TRUE(RhdFusedWithNextAg(tempAlgParams));
    tempAlgParams.localCopyFlag = RHD_FLAG_FUSED_WITH_NEXT_AG | 1; // 位组合互不干扰
    EXPECT_TRUE(RhdFusedWithNextAg(tempAlgParams));
}

// —— RhdCalcConnectRanks（建链连接集）——

// 核心不变量：对全部 p × root × algRank，步骤表的每个对端都落在连接集内
// （连接集 = 全部 root 步骤表对端之并——root 影响代表侧映射与步活跃性两个因子，全并集按构造覆盖）
TEST(RhdConnectRanksTest, PeerSetCoversAllRoots)
{
    for (u32 p = 2; p <= 32; ++p) {
        for (u32 j = 0; j < p; ++j) {
            std::set<u32> connectRanks;
            ASSERT_EQ(RhdCalcConnectRanks(p, j, connectRanks), HcclResult::HCCL_SUCCESS) << "p=" << p << " j=" << j;
            for (u32 root = 0; root < p; root++) {
                std::vector<RhdScatterStep> steps;
                ASSERT_EQ(GenRhdScatterSteps(p, root, j, steps), HcclResult::HCCL_SUCCESS)
                    << "p=" << p << " root=" << root << " j=" << j;
                for (const auto& s : steps) {
                    EXPECT_TRUE(connectRanks.count(s.peerAlgRank) > 0)
                        << "p=" << p << " root=" << root << " j=" << j << " peer=" << s.peerAlgRank;
                }
            }
        }
    }
}

// 踩洞场景精确断言：p=9（r=1）rank0 需 ±3 距离的 algRank3（±2^k 集外）——全 root 并集下仍为
// {1,2,3,5}（伙伴全在 block 侧、与代表侧无关）；rank3 含 vRoot∈{2,6} 时 dist=4 步的对端 7；
// p=8 退化为 XOR 伙伴；p=1 空集；越界 HCCL_E_PARA
TEST(RhdConnectRanksTest, KnownSetsForBugScenario)
{
    std::set<u32> ranks0;
    ASSERT_EQ(RhdCalcConnectRanks(9, 0, ranks0), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ranks0, (std::set<u32>{1, 2, 3, 5})); // k0:5 k1:3 k2:2 POST:1——3 即 ±2^k 缺失者

    std::set<u32> ranks3;
    ASSERT_EQ(RhdCalcConnectRanks(9, 3, ranks3), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ranks3, (std::set<u32>{0, 1, 4, 7})); // 两种代表侧 holder（0/1）+ k2 对端 4 + vRoot∈{2,6} 的 k0 对端 7

    std::set<u32> ranks8;
    ASSERT_EQ(RhdCalcConnectRanks(8, 0, ranks8), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ranks8, (std::set<u32>{1, 2, 4})); // 2^N：恰为逐步 XOR 伙伴

    std::set<u32> ranks1;
    ASSERT_EQ(RhdCalcConnectRanks(1, 0, ranks1), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(ranks1.empty()); // p=1 合法退化：无对端

    std::set<u32> bad;
    EXPECT_EQ(RhdCalcConnectRanks(0, 0, bad), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(RhdCalcConnectRanks(4, 4, bad), HcclResult::HCCL_E_PARA);
}

// 尾块契约：tailSize==0 无尾恒合法；否则须为类型整数倍（可大于 sliceSize）
TEST(RhdTailSizeValidTest, TailContractBoundaries)
{
    // 无尾约定：0 恒合法（不依赖类型尺寸）
    EXPECT_TRUE(RhdTailSizeValid(0, 4));

    // 均匀退化（当前编排层恒传 tailSize == sliceSize）与常规合法尾块（以 sliceSize=64 为参照）
    EXPECT_TRUE(RhdTailSizeValid(64, 4)); // tail == 64：均匀退化
    EXPECT_TRUE(RhdTailSizeValid(32, 4)); // 短尾：类型整数倍
    EXPECT_TRUE(RhdTailSizeValid(60, 4)); // 非槽宽对齐但类型对齐：合法

    // 大尾块合法（末位吸收余数口径）：tail > 参照 sliceSize=64 且类型对齐
    EXPECT_TRUE(RhdTailSizeValid(128, 4)); // tail = 2×64：AllReduce 式吸收余数
    EXPECT_TRUE(RhdTailSizeValid(80, 4));  // tail = 64 + 16B 余数
    EXPECT_TRUE(RhdTailSizeValid(72, 8));  // tail = 64 + 8B 余数，对齐 8B 类型

    // 非数据类型整数倍：元素计数 flooring 丢字节（与尺寸无关——大尾块同样要求对齐）
    EXPECT_FALSE(RhdTailSizeValid(30, 4));
    EXPECT_FALSE(RhdTailSizeValid(65, 4)); // 大于 64 且未对齐：拒绝原因是对齐而非尺寸
    EXPECT_FALSE(RhdTailSizeValid(66, 4)); // 同上：66 % 4 = 2
    EXPECT_FALSE(RhdTailSizeValid(33, 8));

    // dataTypeSize == 0：除法无意义，与模板前置校验口径一致拒绝
    EXPECT_FALSE(RhdTailSizeValid(32, 0));
}
