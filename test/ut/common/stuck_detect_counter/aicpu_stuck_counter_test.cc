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

#include "aicpu_stuck_counter.h"
#include "stuck_stub.h"

namespace {

using namespace ops_hccl;

class AicpuStuckCounterTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ut_stub::g_reduceCalls.clear();
        ut_stub::g_reduceRet = 0;
    }
};

TEST_F(AicpuStuckCounterTest, EmitSkipWhenDisabled)
{
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 0;
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());
}

TEST_F(AicpuStuckCounterTest, EmitSkipWhenThreadZero)
{
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;
    EXPECT_FALSE(AicpuStuckCounterHead(0, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());
}

TEST_F(AicpuStuckCounterTest, EmitSkipsBothWhenAnyAddrZero)
{
    OpParam param;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;

    // head=0、tail 非 0：两侧都不发，避免 head<tail 永久失衡
    param.counterHeadAddr = 0;
    param.counterTailAddr = 0x20;
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());

    // head 非 0、tail=0：两侧都不发，避免 head>tail 永久失衡
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0;
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());

    // addOne=0：两侧都不发
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0;
    EXPECT_FALSE(AicpuStuckCounterHead(0x1, param));
    EXPECT_FALSE(AicpuStuckCounterTail(0x1, param));
    EXPECT_TRUE(ut_stub::g_reduceCalls.empty());
}

TEST_F(AicpuStuckCounterTest, EmitHeadUsesHeadAddrTailUsesTailAddr)
{
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;
    EXPECT_TRUE(AicpuStuckCounterHead(0x7, param));
    EXPECT_TRUE(AicpuStuckCounterTail(0x7, param));
    ASSERT_EQ(ut_stub::g_reduceCalls.size(), 2u);

    const auto& head = ut_stub::g_reduceCalls[0];
    EXPECT_EQ(head.thread, 0x7u);
    EXPECT_EQ(head.dst, 0x10u);
    EXPECT_EQ(head.src, 0x30u);
    EXPECT_EQ(head.count, 1u);
    EXPECT_EQ(head.dataType, static_cast<int>(HCOMM_DATA_TYPE_INT32));
    EXPECT_EQ(head.reduceOp, static_cast<int>(HCOMM_REDUCE_SUM));

    const auto& tail = ut_stub::g_reduceCalls[1];
    EXPECT_EQ(tail.dst, 0x20u);
    EXPECT_EQ(tail.src, 0x30u);
}

TEST_F(AicpuStuckCounterTest, EmitIgnoresReduceFail)
{
    ut_stub::g_reduceRet = HCCL_E_INTERNAL;
    OpParam param;
    param.counterHeadAddr = 0x10;
    param.counterTailAddr = 0x20;
    param.counterAddOneAddr = 0x30;
    param.isEnableCounter = 1;
    EXPECT_TRUE(AicpuStuckCounterHead(0x1, param)); // 失败不传播，仅忽略，仍视为已插入
    EXPECT_EQ(ut_stub::g_reduceCalls.size(), 1u);
}

} // namespace
