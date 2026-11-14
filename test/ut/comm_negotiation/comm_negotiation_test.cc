/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * comm_negotiation 核心功能 UT：
 * 1. IsNegotiationComm 防嵌套判定（null / 名称后缀 / 子通信域标记）
 * 2. NegotiateAllReduceMin 全域 AND 协商（单 rank 短路 / 全员支持 / 任一不支持 / 资源复用）
 * 3. NegotiateAllGather 全域收集（参数校验 / 单 rank 本地拷贝 / 多 rank 收集 / 容量校验）
 * 4. 生命周期闭环（DESTROY_PRE 清理后可重建 / RESUME_POST 快恢续用与失败降级重建）
 * 5. 兼容与边界（长名截断保后缀 / 大载荷缓冲扩容重建）
 * 6. C 接口弱壳直测（NegotiateCommC：入口参数契约 / ALL_REDUCE_MIN 载荷
 *    尺寸协议 / 未登记 op 拒绝 / 两原语分发透传与 C++ 本体等价）
 * 注：HcclCreateSubCommConfig 弱符号缺失降级用例在独立 target（comm_negotiation_symbol_missing_test）
 */
#include <gtest/gtest.h>
#include <cstring>
#include "comm_negotiation.h"
#include "comm_negotiation_c.h"
#include "ut_neg_stub.h"

using namespace ops_hccl;
using ut::g_negStub;

namespace {
constexpr int32_t SUPPORT = ut::REMOTE_SUPPORT_VAL;
constexpr int32_t NOT_SUPPORT = ut::REMOTE_NOT_SUPPORT_VAL;

HcclComm FakeComm(uintptr_t id) { return reinterpret_cast<HcclComm>(id); }
} // namespace

class CommNegotiationTest : public ::testing::Test {
protected:
    void SetUp() override { ut::NegStubReset(); }
};

/* ---------------- IsNegotiationComm：防嵌套判定 ---------------- */

TEST_F(CommNegotiationTest, IsNegotiationCommNullPtr) { EXPECT_FALSE(IsNegotiationComm(nullptr)); }

TEST_F(CommNegotiationTest, IsNegotiationCommNameSuffixMatch)
{
    // 无标记、名称以 _negotiation 结尾 → 快速路径判真
    g_negStub.commName = "user_negotiation";
    EXPECT_TRUE(IsNegotiationComm(FakeComm(0x1001)));
}

TEST_F(CommNegotiationTest, IsNegotiationCommNameSuffixMiss)
{
    g_negStub.commName = "ut_world";
    EXPECT_FALSE(IsNegotiationComm(FakeComm(0x1002)));
}

TEST_F(CommNegotiationTest, IsNegotiationCommGetCommNameFailSafe)
{
    // 名称查询失败按 true 处理（fail-safe）：防递归门控消费方误判 false 会放行递归协商（挂死），
    // 误判 true 仅损失一次协商机会
    g_negStub.getCommNameRet = HCCL_E_INTERNAL;
    EXPECT_TRUE(IsNegotiationComm(FakeComm(0x1005)));
}

TEST_F(CommNegotiationTest, IsNegotiationCommSuffixMissNoEngineCtxProbe)
{
    // 性能特性锁定：普通通信域判定不触发任何EngineCtx探测（IsNegotiationComm被hostDPU门控逐算子
    // 调用，探测未命中在HCOMM侧打WARNING，逐算子探测会算子粒度刷屏，且多一次哈希+锁开销）
    g_negStub.commName = "ut_world";
    EXPECT_FALSE(IsNegotiationComm(FakeComm(0x1004)));
    EXPECT_EQ(g_negStub.ctxGetCnt, 0);
}

TEST_F(CommNegotiationTest, IsNegotiationCommMarkedSubComm)
{
    // 协商建链后：协商子通信域（名称带_negotiation后缀）判真，用户通信域判假；
    // 判定由名称后缀承担
    HcclComm comm = FakeComm(0x1003);
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    ASSERT_NE(g_negStub.createdSubComm, nullptr);
    EXPECT_TRUE(IsNegotiationComm(g_negStub.createdSubComm));
    EXPECT_FALSE(IsNegotiationComm(comm));
}

/* ---------------- NegotiateAllReduceMin：全域 AND 协商 ---------------- */

TEST_F(CommNegotiationTest, AllReduceMinNullComm)
{
    bool allRanksCapable = true;
    // SUT 用 CHK_PTR_NULL 检查空指针，返回 HCCL_E_PTR 而非 HCCL_E_PARA；
    // 出参安全值由调用方负责初始化（SUT错误出口不承诺覆写出参），故错误后保持调用方初值
    EXPECT_EQ(NegotiateAllReduceMin(nullptr, "ut", true, allRanksCapable), HCCL_E_PTR);
    EXPECT_TRUE(allRanksCapable);
}

TEST_F(CommNegotiationTest, AllReduceMinSingleRankShortCircuit)
{
    // rankSize <= 1 短路：直接返回本端结果，不建链不协商
    g_negStub.rankSize = 1;
    HcclComm comm = FakeComm(0x2001);
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", false, allRanksCapable), HCCL_SUCCESS);
    EXPECT_FALSE(allRanksCapable);
    EXPECT_EQ(g_negStub.createSubCommCnt, 0);
    EXPECT_EQ(g_negStub.allReduceCnt, 0);
}

TEST_F(CommNegotiationTest, AllReduceMinAllRanksSupport)
{
    // 全部 rank 支持（本端 true + 远端全 true）→ allRanksCapable = true
    g_negStub.rankSize = 8;
    g_negStub.remoteAllReduceVals = {SUPPORT, SUPPORT, SUPPORT, SUPPORT, SUPPORT, SUPPORT, SUPPORT};
    HcclComm comm = FakeComm(0x2002);
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    EXPECT_EQ(g_negStub.allReduceCnt, 1);
}

TEST_F(CommNegotiationTest, AllReduceMinAnyRankNotSupport)
{
    // MIN 聚合 = 全域 AND：本端不支持 或 任一远端不支持，均输出 false
    HcclComm comm = FakeComm(0x2003);
    bool allRanksCapable = true;
    // 本端不支持，远端全支持
    g_negStub.remoteAllReduceVals = {SUPPORT, SUPPORT};
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", false, allRanksCapable), HCCL_SUCCESS);
    EXPECT_FALSE(allRanksCapable);
    // 本端支持，任一远端不支持
    g_negStub.remoteAllReduceVals = {SUPPORT, NOT_SUPPORT};
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_FALSE(allRanksCapable);
}

TEST_F(CommNegotiationTest, AllReduceMinResourceReuse)
{
    // 同 (comm, typeTag) 二次协商：懒创建的资源（子通信域/缓冲）复用，不重建
    HcclComm comm = FakeComm(0x2004);
    g_negStub.remoteAllReduceVals = {SUPPORT};
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    EXPECT_EQ(g_negStub.createSubCommCnt, 1);
    EXPECT_EQ(g_negStub.regCbCnt, 1);
}

/* ---------------- NegotiateAllGather：全域收集 ---------------- */

TEST_F(CommNegotiationTest, AllGatherInvalidParams)
{
    HcclComm comm = FakeComm(0x3001);
    uint8_t payload[4] = {1, 2, 3, 4};
    uint8_t recv[32] = {0};
    EXPECT_EQ(NegotiateAllGather(comm, "ut", nullptr, 4, recv, sizeof(recv)), HCCL_E_PARA);
    EXPECT_EQ(NegotiateAllGather(comm, "ut", payload, 4, nullptr, sizeof(recv)), HCCL_E_PARA);
    EXPECT_EQ(NegotiateAllGather(comm, "ut", payload, 0, recv, sizeof(recv)), HCCL_E_PARA);
}

TEST_F(CommNegotiationTest, AllGatherSingleRankLocalCopy)
{
    // rankSize <= 1：仅本地拷贝，不建链不收集
    g_negStub.rankSize = 1;
    HcclComm comm = FakeComm(0x3002);
    uint8_t payload[4] = {0xA0, 0xA1, 0xA2, 0xA3};
    uint8_t recv[4] = {0};
    ASSERT_EQ(NegotiateAllGather(comm, "ut", payload, sizeof(payload), recv, sizeof(recv)), HCCL_SUCCESS);
    EXPECT_EQ(memcmp(recv, payload, sizeof(payload)), 0);
    EXPECT_EQ(g_negStub.allGatherCnt, 0);
}

TEST_F(CommNegotiationTest, AllGatherRecvBufTooSmall)
{
    g_negStub.rankSize = 8;
    HcclComm comm = FakeComm(0x3003);
    uint8_t payload[4] = {1, 2, 3, 4};
    uint8_t recv[16] = {0}; // 容量 16 < rankSize * payloadSize = 32
    EXPECT_EQ(NegotiateAllGather(comm, "ut", payload, sizeof(payload), recv, sizeof(recv)), HCCL_E_PARA);
    // 单rank分支同样前置容量校验：错误码与多rank统一为入参错误HCCL_E_PARA
    g_negStub.rankSize = 1;
    uint8_t recvSmall[2] = {0}; // 容量 2 < payloadSize = 4
    EXPECT_EQ(NegotiateAllGather(comm, "ut", payload, sizeof(payload), recvSmall, sizeof(recvSmall)), HCCL_E_PARA);
}

TEST_F(CommNegotiationTest, AllGatherMultiRankCollect)
{
    // 按 rank 序收集：rank0 = 本端载荷，rank1..3 = 注入的远端载荷
    g_negStub.rankSize = 4;
    HcclComm comm = FakeComm(0x3004);
    uint8_t payload[8];
    for (size_t i = 0; i < sizeof(payload); i++) {
        payload[i] = static_cast<uint8_t>(i);
    }
    uint8_t remote1[8];
    uint8_t remote2[8];
    uint8_t remote3[8];
    memset(remote1, 0x11, sizeof(remote1));
    memset(remote2, 0x22, sizeof(remote2));
    memset(remote3, 0x33, sizeof(remote3));
    g_negStub.remoteGatherPayloads = {
        std::vector<uint8_t>(remote1, remote1 + sizeof(remote1)),
        std::vector<uint8_t>(remote2, remote2 + sizeof(remote2)),
        std::vector<uint8_t>(remote3, remote3 + sizeof(remote3)),
    };
    uint8_t recv[32] = {0};
    ASSERT_EQ(NegotiateAllGather(comm, "ut", payload, sizeof(payload), recv, sizeof(recv)), HCCL_SUCCESS);
    EXPECT_EQ(memcmp(recv, payload, sizeof(payload)), 0);
    EXPECT_EQ(memcmp(recv + 8, remote1, sizeof(remote1)), 0);
    EXPECT_EQ(memcmp(recv + 16, remote2, sizeof(remote2)), 0);
    EXPECT_EQ(memcmp(recv + 24, remote3, sizeof(remote3)), 0);
}

/* ---------------- 生命周期闭环 ---------------- */

TEST_F(CommNegotiationTest, DestroyPreCleanupAndRebuild)
{
    // DESTROY_PRE 回调清理协商资源（子通信域销毁 + tag 摘除），之后可完整懒重建
    HcclComm comm = FakeComm(0x4001);
    g_negStub.remoteAllReduceVals = {SUPPORT};
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    ASSERT_EQ(g_negStub.createSubCommCnt, 1);

    // 触发销毁前回调：协商子通信域被销毁，EngineCtx tag 被摘除
    ASSERT_EQ(ut::UtInvokeCommStateCb(comm, HCCL_COMM_STATE_PHASE_DESTROY_PRE), HCCL_SUCCESS);
    EXPECT_GE(g_negStub.commDestroyCnt, 1);
    EXPECT_FALSE(ut::NegStubHasCtx(comm, "ut_world_ut_negotiation"));

    // 清理后再次协商：完整重建且结果正确
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    EXPECT_EQ(g_negStub.createSubCommCnt, 2);
}

TEST_F(CommNegotiationTest, DestroyPreConcurrentEvictedTagNotBlockOthers)
{
    // 并发清理竞争：Probe查tagX成功入列后，其EngineCtx条目被并发路径销毁（Release再查得NOT_FOUND）；
    // 清理循环不得因单tag缺失中断：tagY资源仍须完整回收，且回调整体返回SUCCESS而非HCCL_E_INTERNAL
    HcclComm comm = FakeComm(0x4002);
    g_negStub.remoteAllReduceVals = {SUPPORT};
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "tagX", true, allRanksCapable), HCCL_SUCCESS);
    ASSERT_EQ(NegotiateAllReduceMin(comm, "tagY", true, allRanksCapable), HCCL_SUCCESS);
    ASSERT_EQ(g_negStub.createSubCommCnt, 2);
    ASSERT_TRUE(ut::NegStubHasCtx(comm, "ut_world_tagX_negotiation"));
    ASSERT_TRUE(ut::NegStubHasCtx(comm, "ut_world_tagY_negotiation"));

    int destroyBefore = g_negStub.commDestroyCnt;
    // 注入：tagX条目命中查询即被"并发销毁"（Probe阶段成功，Release阶段NOT_FOUND）
    g_negStub.ctxEvictTagOnGet = "ut_world_tagX_negotiation";
    ASSERT_EQ(ut::UtInvokeCommStateCb(comm, HCCL_COMM_STATE_PHASE_DESTROY_PRE), HCCL_SUCCESS);
    // tagY完整回收：协商子通信域销毁 + ctx tag摘除（旧代码此处中断：回调整体报错且tagY泄漏）
    EXPECT_EQ(g_negStub.commDestroyCnt, destroyBefore + 1);
    EXPECT_FALSE(ut::NegStubHasCtx(comm, "ut_world_tagY_negotiation"));
    // tagX条目已被并发路径销毁（模拟注入），同样不存在
    EXPECT_FALSE(ut::NegStubHasCtx(comm, "ut_world_tagX_negotiation"));
}

/* ---------------- RESUME_POST 快恢：续用与失败降级 ---------------- */

TEST_F(CommNegotiationTest, ResumePostFastPathReuse)
{
    // 快恢成功：HcclCommResume续用协商资源（tag保留、子通信域不销毁），下次协商零重建
    HcclComm comm = FakeComm(0x5001);
    g_negStub.remoteAllReduceVals = {SUPPORT};
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    ASSERT_EQ(g_negStub.createSubCommCnt, 1);

    ASSERT_EQ(ut::UtInvokeCommStateCb(comm, HCCL_COMM_STATE_PHASE_RESUME_POST), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.resumeCnt, 1);
    EXPECT_EQ(g_negStub.commDestroyCnt, 0);                          // 续用：协商子通信域未销毁
    EXPECT_TRUE(ut::NegStubHasCtx(comm, "ut_world_ut_negotiation")); // 资源ctx保留

    // 快恢后再次协商：资源复用，不重建子通信域
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    EXPECT_EQ(g_negStub.createSubCommCnt, 1);
    EXPECT_EQ(g_negStub.allReduceCnt, 2);
}

TEST_F(CommNegotiationTest, ResumePostFallbackRebuild)
{
    // 快恢失败（如未Suspend即resume）：降级走清理（子通信域销毁+tag摘除），下次协商完整重建
    HcclComm comm = FakeComm(0x5002);
    g_negStub.remoteAllReduceVals = {SUPPORT};
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    ASSERT_EQ(g_negStub.createSubCommCnt, 1);

    g_negStub.resumeRet = HCCL_E_INTERNAL; // 注入resume失败
    ASSERT_EQ(ut::UtInvokeCommStateCb(comm, HCCL_COMM_STATE_PHASE_RESUME_POST), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.resumeCnt, 1);
    EXPECT_GE(g_negStub.commDestroyCnt, 1);                           // 降级：销毁协商子通信域
    EXPECT_FALSE(ut::NegStubHasCtx(comm, "ut_world_ut_negotiation")); // tag摘除

    // 降级清理后再次协商：完整重建且结果正确
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    EXPECT_EQ(g_negStub.createSubCommCnt, 2);
}

/* ---------------- 长名截断：后缀保留 ---------------- */

TEST_F(CommNegotiationTest, LongCommNameTruncation)
{
    // negTag超hcclCommName上限：尾部截断至127字节，"_negotiation"后缀必须保留（名称快速路径判定前提）
    g_negStub.commName = std::string(127, 'a'); // negTag = 127 + "_ut" + "_negotiation" = 142字节
    HcclComm comm = FakeComm(0x5003);
    g_negStub.remoteAllReduceVals = {SUPPORT};
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.createdSubCommName.size(), 127u); // 截断至上限
    // 去头保尾：截断名 = 112个'a' + "_ut" + 后缀（与生成方同源常量），后缀完整保留
    EXPECT_EQ(g_negStub.createdSubCommName, std::string(112, 'a') + "_ut" + NEGOTIATION_COMM_SUFFIX);
    EXPECT_TRUE(IsNegotiationComm(g_negStub.createdSubComm)); // 截断名仍被名称快速路径判定

    // 截断不影响 (comm, typeTag) 懒创建键一致性：二次协商复用资源
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.createSubCommCnt, 1);
}

/* ---------------- 缓冲扩容：小载荷协商→大载荷收集 ---------------- */

TEST_F(CommNegotiationTest, BufferGrowOnLargerPayload)
{
    // CCU真实时序「资源协商(4B MIN)→参数检视(大载荷Gather)」：容量不足释放重建，同载荷复用
    HcclComm comm = FakeComm(0x5004);
    g_negStub.remoteAllReduceVals = {SUPPORT, SUPPORT, SUPPORT, SUPPORT, SUPPORT, SUPPORT, SUPPORT};
    bool allRanksCapable = false;
    ASSERT_EQ(NegotiateAllReduceMin(comm, "ut", true, allRanksCapable), HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    EXPECT_EQ(g_negStub.mallocHostCnt, 2); // 首次懒建：send+recv两块host缓冲

    // 大载荷AllGather（64B > 已分配4B）触发释放重建
    uint8_t payload[64];
    for (size_t i = 0; i < sizeof(payload); i++) {
        payload[i] = static_cast<uint8_t>(i);
    }
    uint8_t recv[64 * 8] = {0};
    ASSERT_EQ(NegotiateAllGather(comm, "ut", payload, sizeof(payload), recv, sizeof(recv)), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.freeHostCnt, 2);                  // 旧4B缓冲释放
    EXPECT_EQ(g_negStub.mallocHostCnt, 4);                // 按64B/512B重建
    EXPECT_EQ(memcmp(recv, payload, sizeof(payload)), 0); // 扩容后数据搬运正确

    // 同载荷再次收集：容量足够复用，无重建
    uint8_t recv2[64 * 8] = {0};
    ASSERT_EQ(NegotiateAllGather(comm, "ut", payload, sizeof(payload), recv2, sizeof(recv2)), HCCL_SUCCESS);
    EXPECT_EQ(g_negStub.freeHostCnt, 2);
    EXPECT_EQ(g_negStub.mallocHostCnt, 4);
    EXPECT_EQ(memcmp(recv2, payload, sizeof(payload)), 0);
}

/* ---------------- C 接口弱壳直测：NegotiateCommC ---------------- */

TEST_F(CommNegotiationTest, CEntryInvalidParams)
{
    // C 壳入口契约：null 指针一律 HCCL_E_PARA，不触碰 C++ 本体（协商未发生）
    HcclComm comm = FakeComm(0x6001);
    bool local = true;
    EXPECT_EQ(
        NegotiateCommC(comm, nullptr, NEGOTIATE_OP_ALL_REDUCE_MIN, &local, sizeof(bool), &local, sizeof(bool)),
        HCCL_E_PARA);
    EXPECT_EQ(
        NegotiateCommC(comm, "ut", NEGOTIATE_OP_ALL_REDUCE_MIN, nullptr, sizeof(bool), &local, sizeof(bool)),
        HCCL_E_PARA);
    EXPECT_EQ(
        NegotiateCommC(comm, "ut", NEGOTIATE_OP_ALL_REDUCE_MIN, &local, sizeof(bool), nullptr, sizeof(bool)),
        HCCL_E_PARA);
    EXPECT_EQ(g_negStub.allReduceCnt, 0);
}

TEST_F(CommNegotiationTest, CEntryAllReduceMinSizeMismatch)
{
    // ALL_REDUCE_MIN 载荷协议锁定：bool 进 bool 出，尺寸不符拒绝（防跨算子误用载荷布局）
    HcclComm comm = FakeComm(0x6002);
    uint32_t wide = 1;
    bool narrow = true;
    EXPECT_EQ(
        NegotiateCommC(comm, "ut", NEGOTIATE_OP_ALL_REDUCE_MIN, &wide, sizeof(wide), &narrow, sizeof(bool)),
        HCCL_E_PARA);
    EXPECT_EQ(
        NegotiateCommC(comm, "ut", NEGOTIATE_OP_ALL_REDUCE_MIN, &narrow, sizeof(bool), &wide, sizeof(wide)),
        HCCL_E_PARA);
    EXPECT_EQ(g_negStub.allReduceCnt, 0);
}

TEST_F(CommNegotiationTest, CEntryUnsupportedOp)
{
    // 未登记枚举值拒绝：后续新算子接入须先扩 NegotiateOpC 枚举再走壳内分发（单入口扩展契约）
    HcclComm comm = FakeComm(0x6003);
    bool local = true;
    auto badOp = static_cast<NegotiateOpC>(99);
    EXPECT_EQ(NegotiateCommC(comm, "ut", badOp, &local, sizeof(bool), &local, sizeof(bool)), HCCL_E_PARA);
    EXPECT_EQ(g_negStub.allReduceCnt, 0);
    EXPECT_EQ(g_negStub.allGatherCnt, 0);
}

TEST_F(CommNegotiationTest, CEntryAllReduceMinDispatch)
{
    // ALL_REDUCE_MIN 分发透传：typeTag 字符串化后与直接调 NegotiateAllReduceMin 等价
    g_negStub.rankSize = 8;
    g_negStub.remoteAllReduceVals = {SUPPORT, SUPPORT, SUPPORT, SUPPORT, SUPPORT, SUPPORT, SUPPORT};
    HcclComm comm = FakeComm(0x6004);
    bool local = true;
    bool allRanksCapable = false;
    ASSERT_EQ(
        NegotiateCommC(comm, "ut", NEGOTIATE_OP_ALL_REDUCE_MIN, &local, sizeof(bool), &allRanksCapable, sizeof(bool)),
        HCCL_SUCCESS);
    EXPECT_TRUE(allRanksCapable);
    EXPECT_EQ(g_negStub.allReduceCnt, 1);
    EXPECT_EQ(g_negStub.createdSubCommName, "ut_world_ut_negotiation"); // typeTag 经 std::string 正确透传
}

TEST_F(CommNegotiationTest, CEntryAllGatherDispatch)
{
    // ALL_GATHER 分发透传：与直接调 NegotiateAllGather 等价（本端槽 + 远端注入槽按 rank 序收集）
    g_negStub.rankSize = 4;
    HcclComm comm = FakeComm(0x6005);
    uint8_t payload[8];
    for (size_t i = 0; i < sizeof(payload); i++) {
        payload[i] = static_cast<uint8_t>(i);
    }
    uint8_t remote[3][8];
    memset(remote[0], 0x11, sizeof(remote[0]));
    memset(remote[1], 0x22, sizeof(remote[1]));
    memset(remote[2], 0x33, sizeof(remote[2]));
    g_negStub.remoteGatherPayloads = {
        std::vector<uint8_t>(remote[0], remote[0] + sizeof(remote[0])),
        std::vector<uint8_t>(remote[1], remote[1] + sizeof(remote[1])),
        std::vector<uint8_t>(remote[2], remote[2] + sizeof(remote[2])),
    };
    uint8_t recv[32] = {0};
    ASSERT_EQ(
        NegotiateCommC(comm, "ut", NEGOTIATE_OP_ALL_GATHER, payload, sizeof(payload), recv, sizeof(recv)),
        HCCL_SUCCESS);
    EXPECT_EQ(memcmp(recv, payload, sizeof(payload)), 0);
    EXPECT_EQ(memcmp(recv + 8, remote[0], sizeof(remote[0])), 0);
    EXPECT_EQ(memcmp(recv + 16, remote[1], sizeof(remote[1])), 0);
    EXPECT_EQ(memcmp(recv + 24, remote[2], sizeof(remote[2])), 0);
    EXPECT_EQ(g_negStub.allGatherCnt, 1);
}
