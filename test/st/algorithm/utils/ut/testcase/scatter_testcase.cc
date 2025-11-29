/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "checker.h"

class UT_SCATTER_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

TEST_F(UT_SCATTER_TEST, ut_test_demo_1)
{
    int superPodNum = 1;
    int serverNum = 2;
    int rankNum = 8;
    TopoMeta topoMeta;
    HcclResult ret = HcclSim::GenTopoMeta(topoMeta, superPodNum, serverNum, rankNum);
    EXPECT_TRUE(ret == HCCL_SUCCESS);
}