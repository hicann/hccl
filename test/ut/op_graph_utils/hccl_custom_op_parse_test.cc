/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ParseAivCoreLimit/ParseVectorCoreCntFromHardwareInfo 纯字符串解析 UT (aivCoreLimit 回退链 ①②)
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>

#include "hccl_op_utils.h"

using hccl::ParseAivCoreLimit;
using hccl::ParseVectorCoreCntFromHardwareInfo;

TEST(ParseAivCoreLimitTest, ValidValues)
{
    EXPECT_EQ(56U, ParseAivCoreLimit("56"));
    EXPECT_EQ(1U, ParseAivCoreLimit("1"));
    EXPECT_EQ(UINT32_MAX, ParseAivCoreLimit("4294967295"));
}

TEST(ParseAivCoreLimitTest, NullAndEmpty)
{
    EXPECT_EQ(0U, ParseAivCoreLimit(nullptr));
    EXPECT_EQ(0U, ParseAivCoreLimit(""));
}

TEST(ParseAivCoreLimitTest, InvalidValues)
{
    EXPECT_EQ(0U, ParseAivCoreLimit("0"));          // 零值
    EXPECT_EQ(0U, ParseAivCoreLimit("-1"));         // 负数
    EXPECT_EQ(0U, ParseAivCoreLimit("abc"));        // 非数字
    EXPECT_EQ(0U, ParseAivCoreLimit("12x"));        // 带尾缀
    EXPECT_EQ(0U, ParseAivCoreLimit("4294967296")); // > UINT32_MAX
    // 注: strtol 跳过前导空格, " 42" 解析为 42 (正确行为, 非错误)
}

TEST(ParseVectorCoreCntFromHardwareInfoTest, ValidField)
{
    // InfoStore ge.hardwareInfo 格式: key:value;key:value
    EXPECT_EQ(56U, ParseVectorCoreCntFromHardwareInfo("vector_core_cnt:56;other:1"));
    EXPECT_EQ(72U, ParseVectorCoreCntFromHardwareInfo("prefix:x;vector_core_cnt:72"));
    EXPECT_EQ(1U, ParseVectorCoreCntFromHardwareInfo("vector_core_cnt:1"));
}

TEST(ParseVectorCoreCntFromHardwareInfoTest, MissingField)
{
    EXPECT_EQ(0U, ParseVectorCoreCntFromHardwareInfo("other:1;another:2"));
    EXPECT_EQ(0U, ParseVectorCoreCntFromHardwareInfo(""));
    EXPECT_EQ(0U, ParseVectorCoreCntFromHardwareInfo(nullptr));
}

TEST(ParseVectorCoreCntFromHardwareInfoTest, GarbledValues)
{
    EXPECT_EQ(0U, ParseVectorCoreCntFromHardwareInfo("vector_core_cnt:abc"));
    EXPECT_EQ(0U, ParseVectorCoreCntFromHardwareInfo("vector_core_cnt:0"));
    EXPECT_EQ(0U, ParseVectorCoreCntFromHardwareInfo("vector_core_cnt:-5"));
    EXPECT_EQ(0U, ParseVectorCoreCntFromHardwareInfo("vector_core_cnt:99999999999"));
    // 重复 key: find() 取首个匹配, 56
    EXPECT_EQ(56U, ParseVectorCoreCntFromHardwareInfo("vector_core_cnt:56;vector_core_cnt:72"));
}
