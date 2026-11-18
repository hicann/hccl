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
#include "cost_model.h"
#include "ins_temp_all_reduce_nhr.h"

using namespace ops_hccl;

namespace {
CalcCostCoeffParam MakeNhrParam(const char* algName, bool isPod = false)
{
    CalcCostCoeffParam param;
    param.rankSize = 8;
    param.dataRatio = 1.0f / 8;
    param.netType = CommTopo::COMM_TOPO_CLOS;
    param.inputBuffer = BufferType::INPUT;
    param.outputBuffer = BufferType::OUTPUT;
    param.scratchBuffer = BufferType::HCCL_BUFFER;
    param.portNum = {16};
    param.isPod = isPod;
    param.algName = algName;
    return param;
}
} // namespace

TEST(AllReduceNhrCostTest, MultiLinkAggregatesParallelPlanePorts)
{
    CalcCostCoeffParam nhr = MakeNhrParam("AicpuAllReduceSoleNHR");
    nhr.phyLevelPortNums = {{16}, {16}};
    CalcCostCoeffParam multiLink = MakeNhrParam("AicpuAllReduceSoleNHRMultiLink");
    multiLink.phyLevelPortNums = {{16}, {16}};

    auto nhrCost = InsTempAllReduceNHR::CalcCostCoeff(nhr);
    auto multiLinkCost = InsTempAllReduceNHR::CalcCostCoeff(multiLink);

    ASSERT_EQ(nhrCost.size(), 1u);
    ASSERT_EQ(multiLinkCost.size(), 1u);
    EXPECT_NEAR(nhrCost[0].A, 1.75f / (16.0f * 56.0e9f), 1e-14);
    EXPECT_NEAR(multiLinkCost[0].A, 1.75f / (32.0f * 56.0e9f), 1e-14);
    EXPECT_FLOAT_EQ(multiLinkCost[0].A, nhrCost[0].A / 2.0f);
    EXPECT_FLOAT_EQ(multiLinkCost[0].B, nhrCost[0].B);
    EXPECT_FLOAT_EQ(multiLinkCost[0].C, nhrCost[0].C);
    EXPECT_FLOAT_EQ(multiLinkCost[0].D, nhrCost[0].D);
}

TEST(AllReduceNhrCostTest, PodKeepsNhrAndMultiLinkEquivalent)
{
    CalcCostCoeffParam nhr = MakeNhrParam("AicpuAllReduceSoleNHR", true);
    nhr.phyLevelPortNums = {{16}, {16}};
    CalcCostCoeffParam multiLink = MakeNhrParam("AicpuAllReduceSoleNHRMultiLink", true);
    multiLink.phyLevelPortNums = {{16}, {16}};

    auto nhrCost = InsTempAllReduceNHR::CalcCostCoeff(nhr);
    auto multiLinkCost = InsTempAllReduceNHR::CalcCostCoeff(multiLink);

    ASSERT_EQ(nhrCost.size(), 1u);
    ASSERT_EQ(multiLinkCost.size(), 1u);
    EXPECT_FLOAT_EQ(nhrCost[0].A, multiLinkCost[0].A);
    EXPECT_FLOAT_EQ(nhrCost[0].B, multiLinkCost[0].B);
    EXPECT_FLOAT_EQ(nhrCost[0].C, multiLinkCost[0].C);
    EXPECT_FLOAT_EQ(nhrCost[0].D, multiLinkCost[0].D);
}

TEST(AllReduceNhrCostTest, LegacyPortVectorStillWorksWithoutPlaneInfo)
{
    CalcCostCoeffParam multiLink = MakeNhrParam("AicpuAllReduceSoleNHRMultiLink");
    multiLink.portNum = {16, 16};

    auto cost = InsTempAllReduceNHR::CalcCostCoeff(multiLink);

    ASSERT_EQ(cost.size(), 1u);
    EXPECT_NEAR(cost[0].A, 1.75f / (32.0f * 56.0e9f), 1e-14);
}
