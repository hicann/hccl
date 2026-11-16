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

#include "aiv_stuck_counter_util.h"
#include "hccl_aiv_utils.h"

namespace {

using namespace ops_hccl;

OpCounterInfo MakeEnabledCounter()
{
    OpCounterInfo counter;
    counter.headCountMem = 0x1000;
    counter.tailCountMem = 0x2000;
    counter.addOneMem = 0x3000;
    counter.memSize = sizeof(int32_t);
    counter.isEnableCounter = true;
    return counter;
}

TEST(AivStuckCounterTest, FillKernelArgs)
{
    AivKernelArgs args;
    FillAivStuckCounter(args, MakeEnabledCounter());
    EXPECT_EQ(args.headCountMem, reinterpret_cast<const void*>(0x1000));
    EXPECT_EQ(args.tailCountMem, reinterpret_cast<const void*>(0x2000));
    EXPECT_EQ(args.addOneMem, reinterpret_cast<const void*>(0x3000));
    EXPECT_EQ(args.counterMemSize, sizeof(int32_t));
    EXPECT_TRUE(args.isEnableCounter);
}

TEST(AivStuckCounterTest, FillExtraKernelArgs)
{
    AivExtraKernelArgs args;
    FillAivStuckCounter(args, MakeEnabledCounter());
    EXPECT_EQ(args.headCountMem, reinterpret_cast<const void*>(0x1000));
    EXPECT_EQ(args.tailCountMem, reinterpret_cast<const void*>(0x2000));
    EXPECT_EQ(args.addOneMem, reinterpret_cast<const void*>(0x3000));
    EXPECT_EQ(args.counterMemSize, sizeof(int32_t));
    EXPECT_TRUE(args.isEnableCounter);
}

TEST(AivStuckCounterTest, FillSuperKernelArgs)
{
    AivSuperKernelArgs args;
    FillAivStuckCounter(args, MakeEnabledCounter());
    EXPECT_EQ(args.headCountMem, reinterpret_cast<const void*>(0x1000));
    EXPECT_EQ(args.tailCountMem, reinterpret_cast<const void*>(0x2000));
    EXPECT_EQ(args.addOneMem, reinterpret_cast<const void*>(0x3000));
    EXPECT_EQ(args.counterMemSize, sizeof(int32_t));
    EXPECT_EQ(args.isEnableCounter, 1u); // SuperKernel 走 bool -> uint32_t 转换
}

TEST(AivStuckCounterTest, FillSuperKernelDisabled)
{
    AivSuperKernelArgs args;
    args.isEnableCounter = 1;
    FillAivStuckCounter(args, OpCounterInfo{});
    EXPECT_EQ(args.headCountMem, nullptr);
    EXPECT_EQ(args.tailCountMem, nullptr);
    EXPECT_EQ(args.addOneMem, nullptr);
    EXPECT_EQ(args.counterMemSize, 0u);
    EXPECT_EQ(args.isEnableCounter, 0u);
}

TEST(AivStuckCounterTest, FillDisabledCounter)
{
    AivKernelArgs args;
    args.headCountMem = reinterpret_cast<const void*>(0x1);
    args.isEnableCounter = true;
    FillAivStuckCounter(args, OpCounterInfo{});
    EXPECT_EQ(args.headCountMem, nullptr);
    EXPECT_EQ(args.tailCountMem, nullptr);
    EXPECT_EQ(args.addOneMem, nullptr);
    EXPECT_EQ(args.counterMemSize, 0u);
    EXPECT_FALSE(args.isEnableCounter);
}

} // namespace
