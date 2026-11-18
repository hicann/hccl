/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "acl/acl_rt.h"
#include "alg_param.h"
#include "dlhcomm_function.h"
#include "hcomm_stub.h"
#include "order_launch.h"

using namespace ops_hccl;

namespace {
constexpr ThreadHandle UT_UNFOLD_THREAD = 0x3333;

HcclComm MakeComm() { return reinterpret_cast<HcclComm>(0x2222); }

OpParam MakeParam(bool isCapture, OpMode opMode)
{
    OpParam param{};
    param.isCapture = isCapture;
    param.opMode = opMode;
    param.stream = reinterpret_cast<aclrtStream>(0x1111);
    param.opConfig.execTimeout = 1000;
    return param;
}
} // namespace

class OrderLaunchUtTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        OrderLaunchUtStub::Reset();
        // 覆写 dlsym 成员：反查结果与返回码由桩旋钮控制
        auto& dlFuncs = ops_hccl::DlHcommFunction::GetInstance();
        dlFuncs.dlHcommThreadResGetInfo = [](ThreadHandle thread, void* key, uint32_t size, void** out) -> HcclResult {
            (void)thread;
            (void)key;
            (void)size;
            if (out == nullptr) {
                return HCCL_E_PTR;
            }
            *out = OrderLaunchUtStub::g_threadResStream;
            return static_cast<HcclResult>(OrderLaunchUtStub::g_threadResRet);
        };
    }
};

// ACLGRAPH 主链路：入图恰一次且参数正确，入图先于 notify record
TEST_F(OrderLaunchUtTest, aclgraph_add_host_order_stream_to_model)
{
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    HcclComm comm = MakeComm();
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(comm, param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(comm, param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    ASSERT_EQ(OrderLaunchUtStub::g_addCalls.size(), 1u);
    EXPECT_EQ(OrderLaunchUtStub::g_addCalls[0].stream, OrderLaunchUtStub::g_threadResStream);
    EXPECT_EQ(OrderLaunchUtStub::g_addCalls[0].model, OrderLaunchUtStub::g_rtModel);
    EXPECT_LT(OrderLaunchUtStub::g_addSeq, OrderLaunchUtStub::g_firstNotifyRecordSeq);
}

// capture NONE：不入图降级，握手仍执行
TEST_F(OrderLaunchUtTest, aclgraph_capture_none_skip_add)
{
    OrderLaunchUtStub::g_captureStatus = static_cast<int32_t>(aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE);
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
    // 降级契约：仅跳过入图，握手照常
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 1);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 1);
}

// 接口不支持：降级返回成功，零入图，握手仍执行
TEST_F(OrderLaunchUtTest, aclgraph_capture_feature_not_support_degrade)
{
    OrderLaunchUtStub::g_captureRet = ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
    // 降级契约：仅跳过入图，握手照常
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 1);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 1);
}

// GetInfo 失败：HCCL_E_RUNTIME，notify 零调用
TEST_F(OrderLaunchUtTest, aclgraph_capture_getinfo_error)
{
    OrderLaunchUtStub::g_captureRet = ACL_ERROR_INVALID_PARAM;
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_E_RUNTIME);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 0);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
}

// 反查为空：HCCL_E_RUNTIME，notify 零调用
TEST_F(OrderLaunchUtTest, aclgraph_host_order_stream_null)
{
    OrderLaunchUtStub::g_threadResStream = nullptr;
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_E_RUNTIME);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 0);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
}

// 入图失败：HCCL_E_RUNTIME，notify 零调用
TEST_F(OrderLaunchUtTest, aclgraph_add_to_model_fail)
{
    OrderLaunchUtStub::g_addRet = 1; // 非 RT_ERROR_NONE
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_E_RUNTIME);
    EXPECT_EQ(OrderLaunchUtStub::g_addCalls.size(), 1u); // 入图被尝试恰一次后失败
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 0);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
}

// export 失败：原样传播，零入图、零 notify
TEST_F(OrderLaunchUtTest, aclgraph_export_fail_returns_error)
{
    OrderLaunchUtStub::g_exportRet = HCCL_E_INTERNAL;
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_E_INTERNAL);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 0);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
}

// notify record 失败：入图恰一次，零 notify wait
TEST_F(OrderLaunchUtTest, aclgraph_notify_record_fail_returns_error)
{
    OrderLaunchUtStub::g_notifyRecordRet = HCCL_E_INTERNAL;
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_E_INTERNAL);
    EXPECT_EQ(OrderLaunchUtStub::g_addCalls.size(), 1u); // 入图先于 record
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 1);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
}

// resGetInfo 非 SUCCESS：反查降级为空指针 → HCCL_E_RUNTIME，notify 零调用
TEST_F(OrderLaunchUtTest, aclgraph_thread_res_getinfo_not_success)
{
    OrderLaunchUtStub::g_threadResRet = HCCL_E_NOT_SUPPORT;
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_E_RUNTIME);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 0);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
}

// dlsym 符号缺失：反查降级为空指针 → HCCL_E_RUNTIME，notify 零调用
TEST_F(OrderLaunchUtTest, aclgraph_dlsym_missing_returns_error)
{
    auto& dlFuncs = ops_hccl::DlHcommFunction::GetInstance();
    auto savedFunc = dlFuncs.dlHcommThreadResGetInfo;
    dlFuncs.dlHcommThreadResGetInfo = nullptr; // 触发 dlsym 符号缺失分支
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_E_RUNTIME);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 0);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
    dlFuncs.dlHcommThreadResGetInfo = savedFunc; // 保守恢复
}

// OPBASE 模式：零入图
TEST_F(OrderLaunchUtTest, opbase_mode_no_add)
{
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(false, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
}

// GE 模式：零入图
TEST_F(OrderLaunchUtTest, ge_mode_no_add)
{
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(false, OpMode::OFFLOAD);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
}

// unfoldThread 为 0：Phase1 跳过，零入图
TEST_F(OrderLaunchUtTest, skip_when_unfold_thread_null)
{
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = 0;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 0);
}

// 保序跳过：HcclDedicatedThreadAcquire 不支持
TEST_F(OrderLaunchUtTest, skip_when_dedicated_thread_unsupported)
{
    OrderLaunchUtStub::g_supportDedicated = false;
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    EXPECT_TRUE(OrderLaunchUtStub::g_addCalls.empty());
}

// deviceOrderThread 为 0：入图恰一次，握手零调用，出参清零，Phase2 跳过
TEST_F(OrderLaunchUtTest, skip_when_device_order_thread_null)
{
    OrderLaunchUtStub::g_deviceAcqThread = 0;
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    HcclComm comm = MakeComm();
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(comm, param, unfoldThread), HCCL_SUCCESS);
    EXPECT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(comm, param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    EXPECT_EQ(OrderLaunchUtStub::g_addCalls.size(), 1u);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyRecordCalls, 0);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
    EXPECT_EQ(param.exportHostOrderThread, 0);
    EXPECT_EQ(param.deviceOrderThread, 0);
    EXPECT_EQ(orderLaunch.HcclOrderLaunchToKernelStream(comm, HOST_ORDER_THREAD_NOTIFY_IDX), HCCL_SUCCESS);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
}

// Phase2：恰一次 notify wait，且晚于入图
TEST_F(OrderLaunchUtTest, phase2_wait_notify_once)
{
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = UT_UNFOLD_THREAD;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    ASSERT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    int32_t waitCallsBeforePhase2 = OrderLaunchUtStub::g_notifyWaitCalls;
    EXPECT_EQ(orderLaunch.HcclOrderLaunchToKernelStream(MakeComm(), HOST_ORDER_THREAD_NOTIFY_IDX), HCCL_SUCCESS);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls - waitCallsBeforePhase2, 1);
    EXPECT_EQ(OrderLaunchUtStub::g_addCalls.size(), 1u);
    EXPECT_GT(OrderLaunchUtStub::g_lastNotifyWaitSeq, OrderLaunchUtStub::g_addSeq);
}

// Phase2 跳过：零 notify wait
TEST_F(OrderLaunchUtTest, phase2_skip_when_not_required)
{
    OrderLaunch orderLaunch;
    ThreadHandle unfoldThread = 0;
    OpParam param = MakeParam(true, OpMode::OPBASE);
    ASSERT_EQ(orderLaunch.HcclOrderLaunchPrepare(MakeComm(), param, unfoldThread), HCCL_SUCCESS);
    ASSERT_EQ(
        orderLaunch.HcclOrderLaunchToOrderStream(MakeComm(), param, unfoldThread, ORDER_UNFOLD_THREAD_NOTIFY_IDX),
        HCCL_SUCCESS);
    EXPECT_EQ(orderLaunch.HcclOrderLaunchToKernelStream(MakeComm(), HOST_ORDER_THREAD_NOTIFY_IDX), HCCL_SUCCESS);
    EXPECT_EQ(OrderLaunchUtStub::g_notifyWaitCalls, 0);
}
