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

#include "alg_param.h"
#include "ccu_primitives_dl.hpp"
#include "ccu_stuck_counter.h"
#include "ccu_stuck_counter_trampoline.h"

namespace {

using namespace ops_hccl;
using namespace hcomm::ccu;

CcuResult g_origRet = CCU_SUCCESS;
uint32_t g_origCalls = 0;
bool g_origSawHeadMark = false;

CcuResult FakeOrigKernel(CcuKernelArg)
{
    ++g_origCalls;
    g_origSawHeadMark = (g_reduceRecords.size() == 1);
    return g_origRet;
}

CcuKernelInfo MakeKernelInfo(CcuKernelArgBase& arg, void* func)
{
    CcuKernelInfo info;
    info.kernelFunc = func;
    info.kernelArg = &arg;
    return info;
}

class CcuStuckCounterTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        g_reduceRecords.clear();
        g_waitRecords.clear();
        g_reduceRet = 0;
        g_waitRet = 0;
        g_eventAllocCount = 0;
        g_allocThrow = false;
        g_allocThrowNonStd = false;
        g_origRet = CCU_SUCCESS;
        g_origCalls = 0;
        g_origSawHeadMark = false;
    }
};

CcuKernelArgBase MakeEnabledArg()
{
    CcuKernelArgBase arg;
    arg.stuckCounterEnabled = true;
    arg.stuckHeadAddr = 0x10;
    arg.stuckTailAddr = 0x20;
    arg.stuckAddOneAddr = 0x30;
    arg.stuckCounterToken = 0x99;
    return arg;
}

TEST_F(CcuStuckCounterTest, MarkDisabledNoEmit)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    arg.stuckCounterEnabled = false;
    CcuStuckCounter counter(&arg);
    EXPECT_EQ(counter.Head(), CCU_SUCCESS);
    EXPECT_EQ(counter.Tail(), CCU_SUCCESS);
    EXPECT_TRUE(g_reduceRecords.empty());
    EXPECT_TRUE(g_waitRecords.empty());
    EXPECT_EQ(g_eventAllocCount, 0); // 未开启时不申请事件
}

TEST_F(CcuStuckCounterTest, MarkZeroAddrNoEmit)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    arg.stuckHeadAddr = 0;
    arg.stuckTailAddr = 0;
    CcuStuckCounter counter(&arg);
    EXPECT_EQ(counter.Head(), CCU_SUCCESS);
    EXPECT_EQ(counter.Tail(), CCU_SUCCESS);
    EXPECT_TRUE(g_reduceRecords.empty());
    EXPECT_EQ(g_eventAllocCount, 0);
}

TEST_F(CcuStuckCounterTest, MarkNullArgNoEmit)
{
    CcuStuckCounter counter(nullptr);
    EXPECT_EQ(counter.Head(), CCU_SUCCESS);
    EXPECT_EQ(counter.Tail(), CCU_SUCCESS);
    EXPECT_TRUE(g_reduceRecords.empty());
    EXPECT_EQ(g_eventAllocCount, 0);
}

TEST_F(CcuStuckCounterTest, MarkEmitsLocalReduceAndEventWait)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    CcuStuckCounter counter(&arg);
    EXPECT_EQ(counter.Head(), CCU_SUCCESS);
    EXPECT_EQ(counter.Tail(), CCU_SUCCESS);
    ASSERT_EQ(g_reduceRecords.size(), 2u);
    const auto& head = g_reduceRecords[0];
    EXPECT_EQ(head.dstAddr, arg.stuckHeadAddr);
    EXPECT_EQ(head.dstToken, arg.stuckCounterToken);
    EXPECT_EQ(head.srcAddr, arg.stuckAddOneAddr);
    EXPECT_EQ(head.srcToken, arg.stuckCounterToken);
    EXPECT_EQ(head.len, sizeof(int32_t));
    EXPECT_EQ(head.dataType, static_cast<int>(HcclDataType::HCCL_DATA_TYPE_INT32));
    EXPECT_EQ(head.reduceOp, static_cast<int>(HcclReduceOp::HCCL_REDUCE_SUM));
    EXPECT_EQ(head.mask, CCU_STUCK_COUNTER_EVT_MASK);
    const auto& tail = g_reduceRecords[1];
    EXPECT_EQ(tail.dstAddr, arg.stuckTailAddr);
    EXPECT_EQ(tail.srcAddr, arg.stuckAddOneAddr);
    ASSERT_EQ(g_waitRecords.size(), 2u);
    EXPECT_EQ(g_waitRecords[0].mask, CCU_STUCK_COUNTER_EVT_MASK);
    EXPECT_EQ(g_waitRecords[1].mask, CCU_STUCK_COUNTER_EVT_MASK);
    EXPECT_EQ(g_eventAllocCount, 1); // head/tail 复用同一个 Event
    EXPECT_EQ(g_waitRecords[0].eventId, g_waitRecords[1].eventId);
}

TEST_F(CcuStuckCounterTest, EmitReturnsLocalReduceError)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    g_reduceRet = -1;
    CcuStuckCounter counter(&arg);
    EXPECT_EQ(counter.Head(), -1);
    EXPECT_EQ(g_reduceRecords.size(), 1u);
    EXPECT_TRUE(g_waitRecords.empty()); // LocalReduce 失败不等待事件
}

TEST_F(CcuStuckCounterTest, EmitReturnsEventWaitError)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    g_waitRet = -2;
    CcuStuckCounter counter(&arg);
    EXPECT_EQ(counter.Head(), -2);
    EXPECT_EQ(g_reduceRecords.size(), 1u);
    EXPECT_EQ(g_waitRecords.size(), 1u);
}

TEST_F(CcuStuckCounterTest, AddOneZeroSkipsEmit)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    arg.stuckAddOneAddr = 0;
    CcuStuckCounter counter(&arg);
    EXPECT_EQ(counter.Head(), CCU_SUCCESS);
    EXPECT_EQ(counter.Tail(), CCU_SUCCESS);
    EXPECT_TRUE(g_reduceRecords.empty());
    EXPECT_TRUE(g_waitRecords.empty());
    EXPECT_EQ(g_eventAllocCount, 0); // addOne 缺失时不申请事件
}

TEST_F(CcuStuckCounterTest, FillFields)
{
    CcuKernelArgBase arg;
    FillCcuStuckCounter(arg, true, 0x10, 0x20, 0x30, 0x99);
    EXPECT_TRUE(arg.stuckCounterEnabled);
    EXPECT_EQ(arg.stuckHeadAddr, 0x10u);
    EXPECT_EQ(arg.stuckTailAddr, 0x20u);
    EXPECT_EQ(arg.stuckAddOneAddr, 0x30u);
    EXPECT_EQ(arg.stuckCounterToken, 0x99u);
}

TEST_F(CcuStuckCounterTest, InstallThenGetTrampolineWhenEnabled)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    CcuKernelInfo info = MakeKernelInfo(arg, reinterpret_cast<void*>(&FakeOrigKernel));
    InstallCcuStuckTrampoline(info);
    EXPECT_EQ(arg.stuckOrigKernelFunc, reinterpret_cast<void*>(&FakeOrigKernel)); // Install 显式回填
    EXPECT_EQ(GetCcuRegisteredKernelFunc(info), reinterpret_cast<void*>(&CcuStuckCounterTrampoline));
}

TEST_F(CcuStuckCounterTest, GetWithoutInstallFallsBackToOriginal)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    CcuKernelInfo info = MakeKernelInfo(arg, reinterpret_cast<void*>(&FakeOrigKernel));
    EXPECT_EQ(arg.stuckOrigKernelFunc, nullptr); // 未 Install
    EXPECT_EQ(GetCcuRegisteredKernelFunc(info), reinterpret_cast<void*>(&FakeOrigKernel));
}

TEST_F(CcuStuckCounterTest, SelectOriginalWhenDisabled)
{
    CcuKernelArgBase disabled = MakeEnabledArg();
    disabled.stuckCounterEnabled = false;
    CcuKernelInfo disabledInfo = MakeKernelInfo(disabled, reinterpret_cast<void*>(&FakeOrigKernel));
    InstallCcuStuckTrampoline(disabledInfo);
    EXPECT_EQ(disabled.stuckOrigKernelFunc, nullptr); // 关闭时不写
    EXPECT_EQ(GetCcuRegisteredKernelFunc(disabledInfo), reinterpret_cast<void*>(&FakeOrigKernel));
}

TEST_F(CcuStuckCounterTest, SelectIsStableForQueryAndRegister)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    CcuKernelInfo info = MakeKernelInfo(arg, reinterpret_cast<void*>(&FakeOrigKernel));
    InstallCcuStuckTrampoline(info); // 查询前安装
    EXPECT_EQ(GetCcuRegisteredKernelFunc(info), reinterpret_cast<void*>(&CcuStuckCounterTrampoline));
    InstallCcuStuckTrampoline(info); // 注册前再次安装，同值幂等
    EXPECT_EQ(GetCcuRegisteredKernelFunc(info), reinterpret_cast<void*>(&CcuStuckCounterTrampoline));
    EXPECT_EQ(arg.stuckOrigKernelFunc, reinterpret_cast<void*>(&FakeOrigKernel));
}

TEST_F(CcuStuckCounterTest, TrampolineWithoutOrigReturnsError)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    arg.stuckOrigKernelFunc = nullptr;
    EXPECT_EQ(CcuStuckCounterTrampoline(&arg), CCU_E_INTERNAL); // 兜底返回错误，不空指针调用
    EXPECT_EQ(g_origCalls, 0u);
    EXPECT_TRUE(g_reduceRecords.empty());
}

TEST_F(CcuStuckCounterTest, TrampolineHeadThenOrigThenTail)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    arg.stuckOrigKernelFunc = reinterpret_cast<void*>(&FakeOrigKernel);
    EXPECT_EQ(CcuStuckCounterTrampoline(&arg), CCU_SUCCESS);
    EXPECT_EQ(g_origCalls, 1u);
    EXPECT_TRUE(g_origSawHeadMark);
    ASSERT_EQ(g_reduceRecords.size(), 2u);
    EXPECT_EQ(g_reduceRecords[0].dstAddr, arg.stuckHeadAddr);
    EXPECT_EQ(g_reduceRecords[1].dstAddr, arg.stuckTailAddr);
    ASSERT_EQ(g_waitRecords.size(), 2u);
    EXPECT_EQ(g_eventAllocCount, 1); // head/tail 复用同一个 Event
    EXPECT_EQ(g_waitRecords[0].eventId, g_waitRecords[1].eventId);
}

TEST_F(CcuStuckCounterTest, TrampolineEmitsTailOnOrigFail)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    arg.stuckOrigKernelFunc = reinterpret_cast<void*>(&FakeOrigKernel);
    g_origRet = -1;
    EXPECT_EQ(CcuStuckCounterTrampoline(&arg), -1);
    EXPECT_EQ(g_origCalls, 1u);
    ASSERT_EQ(g_reduceRecords.size(), 2u); // head + tail，失败也配对
    EXPECT_EQ(g_reduceRecords[0].dstAddr, arg.stuckHeadAddr);
    EXPECT_EQ(g_reduceRecords[1].dstAddr, arg.stuckTailAddr);
    ASSERT_EQ(g_waitRecords.size(), 2u);
    EXPECT_EQ(g_eventAllocCount, 1);
    EXPECT_EQ(g_waitRecords[0].eventId, g_waitRecords[1].eventId);
}

TEST_F(CcuStuckCounterTest, RepeatedMarksReuseSingleEvent)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    CcuStuckCounter counter(&arg);
    constexpr uint32_t MARK_ROUNDS = 4;
    for (uint32_t i = 0; i < MARK_ROUNDS; ++i) {
        EXPECT_EQ(counter.Head(), CCU_SUCCESS);
        EXPECT_EQ(counter.Tail(), CCU_SUCCESS);
    }
    EXPECT_EQ(g_reduceRecords.size(), MARK_ROUNDS * 2);
    EXPECT_EQ(g_waitRecords.size(), MARK_ROUNDS * 2);
    EXPECT_EQ(g_eventAllocCount, 1); // 多次打点只申请 1 个事件
    for (const auto& wait : g_waitRecords) {
        EXPECT_EQ(wait.eventId, g_waitRecords[0].eventId);
    }
}

TEST_F(CcuStuckCounterTest, EventAllocatedOnlyWhenFirstEmit)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    CcuStuckCounter counter(&arg);
    EXPECT_EQ(g_eventAllocCount, 0); // 构造不申请 Event
    EXPECT_EQ(counter.Head(), CCU_SUCCESS);
    EXPECT_EQ(g_eventAllocCount, 1); // 首次 Emit 才申请
    EXPECT_EQ(counter.Tail(), CCU_SUCCESS);
    EXPECT_EQ(g_eventAllocCount, 1); // head/tail 复用
}

TEST_F(CcuStuckCounterTest, EventAllocThrowsDegrades)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    g_allocThrow = true;
    CcuStuckCounter counter(&arg);
    EXPECT_EQ(counter.Head(), CCU_SUCCESS); // 申请失败不抛给调用方
    EXPECT_EQ(counter.Tail(), CCU_SUCCESS);
    EXPECT_TRUE(g_reduceRecords.empty()); // 不计数
    EXPECT_TRUE(g_waitRecords.empty());
    EXPECT_EQ(g_eventAllocCount, 1); // 本次 build 只尝试一次，不反复抛异常
}

TEST_F(CcuStuckCounterTest, EventAllocThrowsNonStdDegrades)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    g_allocThrowNonStd = true; // 抛非 std::exception 派生类型
    CcuStuckCounter counter(&arg);
    EXPECT_EQ(counter.Head(), CCU_SUCCESS); // catch(...) 兜底，不抛给调用方
    EXPECT_EQ(counter.Tail(), CCU_SUCCESS);
    EXPECT_TRUE(g_reduceRecords.empty()); // 不计数
    EXPECT_TRUE(g_waitRecords.empty());
    EXPECT_EQ(g_eventAllocCount, 1); // 本次 build 只尝试一次
}

TEST_F(CcuStuckCounterTest, TrampolineSurvivesEventAllocFail)
{
    CcuKernelArgBase arg = MakeEnabledArg();
    arg.stuckOrigKernelFunc = reinterpret_cast<void*>(&FakeOrigKernel);
    g_allocThrow = true;
    g_origRet = -3;
    EXPECT_EQ(CcuStuckCounterTrampoline(&arg), -3); // 业务返回值原样透传
    EXPECT_EQ(g_origCalls, 1u);                     // 原 kernel 正常执行
    EXPECT_TRUE(g_reduceRecords.empty());
    EXPECT_TRUE(g_waitRecords.empty());
}

} // namespace
