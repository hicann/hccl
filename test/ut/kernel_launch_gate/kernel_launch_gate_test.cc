/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AICPU 控核 UT：覆盖 HcclAicpuCpuLimitReached 判定（边界/短路/降级）与
// HcclOrderLaunchDelayGuard 守卫（Commit/析构兜底/恰好一次/幂等）。
//
// 源包含缝隙：被测的判定函数为 static、守卫类位于匿名命名空间，无法从外部 TU 直测，
// 本文件直接 #include "kernel_launch.cc" 将其纳入本 TU（kernel_launch.cc 不得同时作为独立源编译）。
//
// 双目标（见 CMakeLists.txt）：
//   kernel_launch_gate_test       链接 aicpu_cpu_ctrl_stub（取值可控，覆盖判定与守卫全部分支）
//   kernel_launch_gate_null_test  不链接该桩（弱符号恒 null，专测降级路径 + WARNING 仅一次）

#include <gtest/gtest.h>

#include "kernel_launch.cc"

#include "aicpu_cpu_ctrl_stub.h"
#include "hcomm_gate_stub.h"
#include "log_gate_stub.h"

namespace {
constexpr uint64_t VALID_HOST_ORDER_THREAD = 0x1200000e6000;
constexpr uint64_t VALID_DEVICE_ORDER_THREAD = 0x12000013e000;

OpParam MakeParam(uint64_t hostOrderThread, uint64_t devOrderThread)
{
    OpParam param{};
    param.exportHostOrderThread = static_cast<ThreadHandle>(hostOrderThread);
    param.deviceOrderThread = static_cast<ThreadHandle>(devOrderThread);
    return param;
}
} // namespace

class KernelLaunchGateTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        GateStubResetNotifyRecordCnt();
        GateStubClearLogs();
#ifdef UT_WITH_CPU_CTRL_STUB
        GateStubResetCpuQueryCnt();
        GateStubSetCpuCnt(6, 0); // 默认不超限
#endif
    }
};

#ifdef UT_WITH_CPU_CTRL_STUB

// 未超限（idle > threshold = reserved+1）：不延迟，且两个框架查询各被调用一次
TEST_F(KernelLaunchGateTest, judgeNotOverLimitWhenIdleAboveThreshold)
{
    GateStubSetCpuCnt(6, 4); // threshold=5，idle=6 > 5
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    EXPECT_FALSE(HcclAicpuCpuLimitReached(&param));
    EXPECT_EQ(GateStubGetIdleQueryCnt(), 1u);
    EXPECT_EQ(GateStubGetReservedQueryCnt(), 1u);
}

// 边界：idle == threshold（reserved+1）判定超限（专家方案 +1 补偿的门限边界）
TEST_F(KernelLaunchGateTest, judgeOverLimitAtBoundaryIdleEqualsThreshold)
{
    GateStubSetCpuCnt(5, 4); // threshold=5，idle=5 == 5
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    EXPECT_TRUE(HcclAicpuCpuLimitReached(&param));
}

// idle < threshold 同样判定超限
TEST_F(KernelLaunchGateTest, judgeOverLimitWhenIdleBelowThreshold)
{
    GateStubSetCpuCnt(2, 4);
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    EXPECT_TRUE(HcclAicpuCpuLimitReached(&param));
}

// 接口合法取值域为[0, 64]（框架约定）：单侧超界（如错误哨兵 UINT32_MAX）判信息不可用，
// fail-open 不限流，且 WARNING 仅打一次
TEST_F(KernelLaunchGateTest, judgeFailOpenWhenReservedInvalid)
{
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    // 三组取值均构造"无校验时会误判超限"的场景（数值上满足 idle<=reserved+1），
    // 使断言真正依赖 [0,64] 域校验：删掉任一侧校验，对应断言即红
    GateStubSetCpuCnt(0, 65); // reserved 侧越界：无校验时 0<=66 为 TRUE
    for (int i = 0; i < 3; i++) {
        EXPECT_FALSE(HcclAicpuCpuLimitReached(&param));
    }
    GateStubSetCpuCnt(65, 64); // idle 侧越界：无校验时 65<=65 为 TRUE
    EXPECT_FALSE(HcclAicpuCpuLimitReached(&param));
    GateStubSetCpuCnt(100, 100); // 双侧越界：无校验时 100<=101 为 TRUE
    EXPECT_FALSE(HcclAicpuCpuLimitReached(&param));
    EXPECT_EQ(GateStubCountLog("invalid cpu cnt"), std::size_t(1)); // 三次异常仅告警一次
    GateStubSetCpuCnt(6, 0);
    EXPECT_FALSE(HcclAicpuCpuLimitReached(&param)); // 恢复合法值后门控正常工作
}

// 合法域边界：idle=0/reserved=0（核满+无预留）为合法状态，不触发 fail-open，正常判定超限
TEST_F(KernelLaunchGateTest, judgeZeroZeroIsLegitimateOverLimit)
{
    GateStubSetCpuCnt(0, 0);
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    EXPECT_TRUE(HcclAicpuCpuLimitReached(&param)); // 0 <= 0+1，设计内超限场景
    GateStubSetCpuCnt(0, 64);                      // 域上界值合法
    EXPECT_TRUE(HcclAicpuCpuLimitReached(&param));
    GateStubSetCpuCnt(64, 64);
    EXPECT_TRUE(HcclAicpuCpuLimitReached(&param));
}

// 无保序线程（任一句柄为 0）：不延迟，且短路于框架查询之前（查询计数为 0）
TEST_F(KernelLaunchGateTest, judgeSkipWhenNoOrderThreadAndQueryNotConsulted)
{
    GateStubSetCpuCnt(1, 5); // 取值上"超限"，但无线程时应恒不延迟
    OpParam noHost = MakeParam(0, VALID_DEVICE_ORDER_THREAD);
    EXPECT_FALSE(HcclAicpuCpuLimitReached(&noHost));
    OpParam noDevice = MakeParam(VALID_HOST_ORDER_THREAD, 0);
    EXPECT_FALSE(HcclAicpuCpuLimitReached(&noDevice));
    EXPECT_EQ(GateStubGetIdleQueryCnt(), 0u);
    EXPECT_EQ(GateStubGetReservedQueryCnt(), 0u);
}

#endif // UT_WITH_CPU_CTRL_STUB

#ifndef UT_WITH_CPU_CTRL_STUB

// 符号缺失（本目标不定义桩，弱符号恒 null）：恒不延迟，且降级 WARNING 仅打一次
TEST_F(KernelLaunchGateTest, judgeDisabledWhenSymbolsNullAndWarnOnlyOnce)
{
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    for (int i = 0; i < 5; i++) {
        EXPECT_FALSE(HcclAicpuCpuLimitReached(&param));
    }
    EXPECT_EQ(GateStubCountLog("core control symbols are nullptr"), std::size_t(1));
}

#endif // !UT_WITH_CPU_CTRL_STUB

// 有效保序线程：HcclOrderLaunchNotifyRecord 调用底层原语一次，句柄与 notifyIdx 透传正确
TEST_F(KernelLaunchGateTest, recordInvokesPrimitiveOnceWithValidThreads)
{
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    EXPECT_EQ(HcclOrderLaunchNotifyRecord(&param), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(GateStubGetNotifyRecordCnt(), 1u);
    EXPECT_EQ(GateStubGetLastNotifySrcThread(), VALID_DEVICE_ORDER_THREAD);
    EXPECT_EQ(GateStubGetLastNotifyDstThread(), VALID_HOST_ORDER_THREAD);
}

// 无保序线程：record 走 skip 分支，不触碰底层原语且返回成功
TEST_F(KernelLaunchGateTest, recordSkipsPrimitiveWhenNoOrderThread)
{
    OpParam param = MakeParam(0, 0);
    EXPECT_EQ(HcclOrderLaunchNotifyRecord(&param), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(GateStubGetNotifyRecordCnt(), 0u);
}

// 非延迟模式：守卫全程（Commit + 析构）不产生任何 record
TEST_F(KernelLaunchGateTest, guardNoDelayNeverRecords)
{
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    {
        HcclOrderLaunchDelayGuard guard(&param, false);
        EXPECT_EQ(guard.Commit(), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(GateStubGetNotifyRecordCnt(), 0u);
    }
    EXPECT_EQ(GateStubGetNotifyRecordCnt(), 0u);
}

// 延迟模式 + Commit：构造不发、Commit 发一次、Commit 后析构不重复 —— 恰好一次
TEST_F(KernelLaunchGateTest, guardDelayCommitRecordsExactlyOnce)
{
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    {
        HcclOrderLaunchDelayGuard guard(&param, true);
        EXPECT_EQ(GateStubGetNotifyRecordCnt(), 0u);
        EXPECT_EQ(guard.Commit(), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(GateStubGetNotifyRecordCnt(), 1u);
    }
    EXPECT_EQ(GateStubGetNotifyRecordCnt(), 1u);
}

// 延迟模式 + 未 Commit：析构兜底恰好补发一次
TEST_F(KernelLaunchGateTest, guardDelayDtorFallbackRecordsExactlyOnce)
{
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    {
        HcclOrderLaunchDelayGuard guard(&param, true);
        EXPECT_EQ(GateStubGetNotifyRecordCnt(), 0u);
    }
    EXPECT_EQ(GateStubGetNotifyRecordCnt(), 1u);
}

// Commit 幂等：重复 Commit 返回成功且不再发
TEST_F(KernelLaunchGateTest, guardCommitIsIdempotent)
{
    OpParam param = MakeParam(VALID_HOST_ORDER_THREAD, VALID_DEVICE_ORDER_THREAD);
    {
        HcclOrderLaunchDelayGuard guard(&param, true);
        EXPECT_EQ(guard.Commit(), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(guard.Commit(), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(GateStubGetNotifyRecordCnt(), 1u);
    }
    EXPECT_EQ(GateStubGetNotifyRecordCnt(), 1u);
}
