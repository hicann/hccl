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
#include <string>

#include "stuck_detect_counter_helper.h"
#include "stuck_stub.h"

namespace {

using ops_hccl::GetStuckDetectCounterInfo;
using ops_hccl::OpCounterInfo;
using ops_hccl::StuckDetectCounterCache;
using ops_hccl::StuckDetectCounterEntry;

HcclComm MakeComm(uintptr_t value) { return reinterpret_cast<HcclComm>(value); }

#if !defined(UT_SCENARIO_UNSUPPORTED) && !defined(UT_SCENARIO_REGFAIL)
class StuckDetectCounterHelperTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ut_stub::g_supportCounterAddr = true;
        ut_stub::g_queryRet = HCCL_SUCCESS;
        ut_stub::g_headAddr = 0x1000;
        ut_stub::g_tailAddr = 0x2000;
        ut_stub::g_addOneAddr = 0x3000;
        ut_stub::g_token = 0xABCD;
        ut_stub::g_queryCalls = 0;
        ut_stub::g_eraseDuringQueryPhase = HCCL_COMM_STATE_PHASE_INVALID;
        ut_stub::ResetCommIdentity();
    }

    // 直接写缓存条目，用于构造“新鲜/过期/容量”等边界场景
    bool PutEntry(HcclComm comm, uint64_t headAddr, uint64_t lastQueryMs, const std::string& name = "")
    {
        StuckDetectCounterEntry entry;
        uint64_t generation = 0;
        (void)StuckDetectCounterCache::GetInstance().Get(comm, entry, generation); // 只为取当前代次
        entry = {};
        entry.isQuerySucceeded = true;
        entry.counter.headCountMem = headAddr;
        entry.counter.tailCountMem = headAddr + 0x1000;
        entry.counter.addOneMem = headAddr + 0x2000;
        entry.counter.memSize = static_cast<uint32_t>(sizeof(int32_t));
        entry.counter.isEnableCounter = true;
        entry.token = 0x1234;
        entry.commName = name.empty() ? ut_stub::CommNameOf(comm) : name;
        entry.lastQueryMs = lastQueryMs;
        bool isFirstInsert = false;
        return StuckDetectCounterCache::GetInstance().Put(comm, entry, generation, isFirstInsert);
    }
};

TEST_F(StuckDetectCounterHelperTest, CommNullReturnsDisabled)
{
    OpCounterInfo counter;
    counter.isEnableCounter = true;
    uint64_t token = 123;
    EXPECT_EQ(GetStuckDetectCounterInfo(nullptr, counter, &token), HCCL_SUCCESS);
    EXPECT_FALSE(counter.isEnableCounter);
    EXPECT_EQ(token, 0u);
    EXPECT_EQ(ut_stub::g_queryCalls, 0);
}

TEST_F(StuckDetectCounterHelperTest, HcommUnsupportedCounterAddrReturnsDisabled)
{
    ut_stub::g_supportCounterAddr = false;
    OpCounterInfo counter;
    counter.isEnableCounter = true;
    EXPECT_EQ(GetStuckDetectCounterInfo(MakeComm(0x10), counter), HCCL_SUCCESS);
    EXPECT_FALSE(counter.isEnableCounter);
    EXPECT_EQ(ut_stub::g_queryCalls, 0);
}

TEST_F(StuckDetectCounterHelperTest, QueryFailRetriesEveryTime)
{
    ut_stub::g_queryRet = HCCL_E_INTERNAL;
    OpCounterInfo counter;
    HcclComm comm = MakeComm(0x11);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_FALSE(counter.isEnableCounter);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_FALSE(counter.isEnableCounter);
    EXPECT_EQ(ut_stub::g_queryCalls, 2); // 查询失败不写缓存，每次重试
}

TEST_F(StuckDetectCounterHelperTest, ZeroAddrVariantsDisabled)
{
    struct ZeroCase {
        uint64_t head;
        uint64_t tail;
        uint64_t addOne;
    };
    const ZeroCase cases[] = {{0, 0x2000, 0x3000}, {0x1000, 0, 0x3000}, {0x1000, 0x2000, 0}, {0, 0, 0}};

    uintptr_t commValue = 0x30;
    for (const auto& zeroCase : cases) {
        ut_stub::g_headAddr = zeroCase.head;
        ut_stub::g_tailAddr = zeroCase.tail;
        ut_stub::g_addOneAddr = zeroCase.addOne;
        ut_stub::g_queryCalls = 0;
        OpCounterInfo counter;
        counter.isEnableCounter = true;
        uint64_t token = 999;
        HcclComm comm = MakeComm(commValue++);
        EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter, &token), HCCL_SUCCESS);
        EXPECT_FALSE(counter.isEnableCounter);
        EXPECT_EQ(token, 0u);
        EXPECT_EQ(ut_stub::g_queryCalls, 1); // 三地址不齐：不使能，但仍会直查一次
    }
}

TEST_F(StuckDetectCounterHelperTest, CacheHitAvoidsRequery)
{
    OpCounterInfo counter;
    HcclComm comm = MakeComm(0x13);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_queryCalls, 1); // 身份一致且在时效窗口内：命中
}

TEST_F(StuckDetectCounterHelperTest, FreshEntryReusedWithoutQuery)
{
    HcclComm comm = MakeComm(0x14);
    ASSERT_TRUE(PutEntry(comm, 0x7777, StuckDetectCounterCache::NowMs()));
    OpCounterInfo counter;
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_queryCalls, 0); // 新鲜条目直接复用
    EXPECT_EQ(counter.headCountMem, 0x7777u);
}

TEST_F(StuckDetectCounterHelperTest, ExpiredEntryTriggersRequery)
{
    HcclComm comm = MakeComm(0x15);
    ut_stub::g_headAddr = 0x5000;
    ASSERT_TRUE(PutEntry(comm, 0x7777, StuckDetectCounterCache::NowMs() - ops_hccl::STUCK_DETECT_CACHE_REQUERY_MS));
    OpCounterInfo counter;
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_queryCalls, 1);                  // 超时：按 miss 重查
    EXPECT_EQ(counter.headCountMem, ut_stub::g_headAddr); // 返回新地址，不再返回陈旧地址
}

TEST_F(StuckDetectCounterHelperTest, IdleEntrySweptOnPut)
{
    HcclComm idleComm = MakeComm(0x16);
    HcclComm activeComm = MakeComm(0x17);
    ASSERT_TRUE(PutEntry(idleComm, 0x7777, 0)); // 时间戳为 0：闲置过期
    ASSERT_TRUE(PutEntry(activeComm, 0x8888, StuckDetectCounterCache::NowMs()));
    StuckDetectCounterEntry entry;
    uint64_t generation = 0;
    EXPECT_FALSE(StuckDetectCounterCache::GetInstance().Get(idleComm, entry, generation)); // 已被清理
    EXPECT_TRUE(StuckDetectCounterCache::GetInstance().Get(activeComm, entry, generation));
}

TEST_F(StuckDetectCounterHelperTest, IdentityMismatchByNameTriggersRequery)
{
    HcclComm comm = MakeComm(0x18);
    OpCounterInfo counter;
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    ASSERT_EQ(ut_stub::g_queryCalls, 1);

    // 模拟通信域销毁后同地址被新通信域复用
    ut_stub::g_commNames[comm] = "reused_comm";
    ut_stub::g_headAddr = 0x6000;
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_queryCalls, 2);                  // 身份不一致：按 miss 重查
    EXPECT_EQ(counter.headCountMem, ut_stub::g_headAddr); // 返回新通信域的地址
}

TEST_F(StuckDetectCounterHelperTest, IdentityMismatchByRankTriggersRequery)
{
    HcclComm comm = MakeComm(0x19);
    OpCounterInfo counter;
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    ASSERT_EQ(ut_stub::g_queryCalls, 1);

    ut_stub::g_commRanks[comm] = 3; // 同地址新通信域的 rank 不同
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_queryCalls, 2);
}

TEST_F(StuckDetectCounterHelperTest, IdentityUnavailableSkipsCache)
{
    ut_stub::g_commNameRet = HCCL_E_INTERNAL;
    OpCounterInfo counter;
    HcclComm comm = MakeComm(0x1A);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_TRUE(counter.isEnableCounter);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_queryCalls, 2); // 取不到身份：不入缓存，每次直查
}

TEST_F(StuckDetectCounterHelperTest, CacheCapacityBounded)
{
    constexpr uintptr_t base = 0xF000;
    const uint64_t nowMs = StuckDetectCounterCache::NowMs();
    for (size_t i = 0; i < ops_hccl::STUCK_DETECT_CACHE_MAX_ENTRIES + 44; ++i) {
        ASSERT_TRUE(PutEntry(MakeComm(base + i), 0x9000 + i, nowMs - 500 + i));
    }
    EXPECT_LE(StuckDetectCounterCache::GetInstance().Size(), ops_hccl::STUCK_DETECT_CACHE_MAX_ENTRIES);
    StuckDetectCounterEntry entry;
    uint64_t generation = 0;
    EXPECT_FALSE(StuckDetectCounterCache::GetInstance().Get(MakeComm(base), entry, generation)); // 最旧被淘汰
    EXPECT_TRUE(StuckDetectCounterCache::GetInstance().Get(
        MakeComm(base + ops_hccl::STUCK_DETECT_CACHE_MAX_ENTRIES + 43), entry, generation)); // 最新保留
}

TEST_F(StuckDetectCounterHelperTest, CallbackRegisteredOnce)
{
    OpCounterInfo counter;
    EXPECT_EQ(GetStuckDetectCounterInfo(MakeComm(0x1B), counter), HCCL_SUCCESS);
    EXPECT_EQ(GetStuckDetectCounterInfo(MakeComm(0x1C), counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_regCalls, 1); // once_flag 保证仅注册一次
    EXPECT_NE(ut_stub::g_registeredCb, nullptr);
}

TEST_F(StuckDetectCounterHelperTest, CacheErasedOnLifecyclePhase)
{
    const HcclCommStatePhase phases[] = {
        HCCL_COMM_STATE_PHASE_DESTROY_PRE,
        HCCL_COMM_STATE_PHASE_DESTROY_POST,
        HCCL_COMM_STATE_PHASE_RESUME_POST,
    };
    uintptr_t commVal = 0x50;
    for (HcclCommStatePhase phase : phases) {
        OpCounterInfo counter;
        HcclComm comm = MakeComm(commVal++);
        EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
        ASSERT_NE(ut_stub::g_registeredCb, nullptr);
        EXPECT_EQ(ut_stub::g_registeredCb(comm, phase, nullptr), HCCL_SUCCESS);
        EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    }
    EXPECT_EQ(ut_stub::g_queryCalls, 2 * static_cast<int>(sizeof(phases) / sizeof(phases[0])));
}

TEST_F(StuckDetectCounterHelperTest, QueryInflightEraseDropsPut)
{
    OpCounterInfo counter;
    // 先确保状态回调已注册
    EXPECT_EQ(GetStuckDetectCounterInfo(MakeComm(0x3E), counter), HCCL_SUCCESS);
    ASSERT_NE(ut_stub::g_registeredCb, nullptr);
    ut_stub::g_queryCalls = 0;

    const HcclCommStatePhase phases[] = {
        HCCL_COMM_STATE_PHASE_DESTROY_PRE,
        HCCL_COMM_STATE_PHASE_DESTROY_POST,
        HCCL_COMM_STATE_PHASE_RESUME_POST,
    };
    uintptr_t commVal = 0x40;
    for (HcclCommStatePhase phase : phases) {
        HcclComm comm = MakeComm(commVal++);
        ut_stub::g_eraseDuringQueryPhase = phase;
        EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
        EXPECT_TRUE(counter.isEnableCounter); // 本次仍返回查询结果
        EXPECT_EQ(ut_stub::g_queryCalls, 1);

        // 查询期间发生 Erase 导致 Put 失效：下次仍未命中，需要重新查询
        ut_stub::g_eraseDuringQueryPhase = HCCL_COMM_STATE_PHASE_INVALID;
        EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
        EXPECT_EQ(ut_stub::g_queryCalls, 2);
        ut_stub::g_queryCalls = 0;
    }
}

TEST_F(StuckDetectCounterHelperTest, OtherStatePhaseKeepsCache)
{
    OpCounterInfo counter;
    HcclComm comm = MakeComm(0x60);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    ASSERT_NE(ut_stub::g_registeredCb, nullptr);
    EXPECT_EQ(ut_stub::g_registeredCb(comm, HCCL_COMM_STATE_PHASE_RESUME_PRE, nullptr), HCCL_SUCCESS);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_queryCalls, 1); // 非销毁/快恢阶段不清缓存
}

TEST_F(StuckDetectCounterHelperTest, CallbackNullCommNoop)
{
    OpCounterInfo counter;
    EXPECT_EQ(GetStuckDetectCounterInfo(MakeComm(0x1F), counter), HCCL_SUCCESS);
    ASSERT_NE(ut_stub::g_registeredCb, nullptr);
    EXPECT_EQ(ut_stub::g_registeredCb(nullptr, HCCL_COMM_STATE_PHASE_DESTROY_PRE, nullptr), HCCL_SUCCESS);
}

#endif // !UT_SCENARIO_UNSUPPORTED && !UT_SCENARIO_REGFAIL

#ifdef UT_SCENARIO_UNSUPPORTED
TEST(StuckDetectCounterHelperDegradedTest, CacheDisabledWhenCallbackUnsupported)
{
    ut_stub::g_supportCounterAddr = true;
    ut_stub::g_queryRet = HCCL_SUCCESS;
    ut_stub::g_headAddr = 0x1000;
    ut_stub::g_tailAddr = 0x2000;
    ut_stub::g_addOneAddr = 0x3000;
    ut_stub::g_queryCalls = 0;
    ut_stub::ResetCommIdentity();

    OpCounterInfo counter;
    HcclComm comm = MakeComm(0x21);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_queryCalls, 2); // 缓存关闭，每次都查询
    EXPECT_TRUE(counter.isEnableCounter);
}
#endif // UT_SCENARIO_UNSUPPORTED

#ifdef UT_SCENARIO_REGFAIL
TEST(StuckDetectCounterHelperDegradedTest, CacheDisabledWhenRegisterFail)
{
    ut_stub::g_supportCounterAddr = true;
    ut_stub::g_queryRet = HCCL_SUCCESS;
    ut_stub::g_headAddr = 0x1000;
    ut_stub::g_tailAddr = 0x2000;
    ut_stub::g_addOneAddr = 0x3000;
    ut_stub::g_queryCalls = 0;
    ut_stub::ResetCommIdentity();

    OpCounterInfo counter;
    HcclComm comm = MakeComm(0x22);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(GetStuckDetectCounterInfo(comm, counter), HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::g_queryCalls, 2); // 注册失败关闭缓存，每次都查询
    EXPECT_EQ(ut_stub::g_regCalls, 1);
}
#endif // UT_SCENARIO_REGFAIL

} // namespace
