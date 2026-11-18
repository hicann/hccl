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
#include "ins_temp_all_gather_nhr.h"

using namespace ops_hccl;

namespace {
CalcCostCoeffParam MakeAllGatherParam(const char* algName)
{
    CalcCostCoeffParam param;
    param.rankSize = 8;
    param.dataRatio = 1.0f;
    param.netType = CommTopo::COMM_TOPO_CLOS;
    param.inputBuffer = BufferType::INPUT;
    param.outputBuffer = BufferType::OUTPUT;
    param.scratchBuffer = BufferType::HCCL_BUFFER;
    param.portNum = {16};
    param.algName = algName;
    return param;
}
} // namespace

TEST(AllGatherNhrCostTest, MultiLinkAggregatesPhysicalSources)
{
    CalcCostCoeffParam single = MakeAllGatherParam("AicpuAllGatherSoleNHR");
    CalcCostCoeffParam multiLink = MakeAllGatherParam("AicpuAllGatherSoleNHRMultiLink");
    multiLink.phyLevelPortNums = {{16}, {16}};

    auto singleCost = InsTempAllGatherNHR::CalcCostCoeff(single);
    auto multiLinkCost = InsTempAllGatherNHR::CalcCostCoeff(multiLink);

    ASSERT_EQ(singleCost.size(), 1u);
    ASSERT_EQ(multiLinkCost.size(), 1u);
    EXPECT_NEAR(singleCost[0].A, 7.0f / (16.0f * 56.0e9f), 1e-14);
    EXPECT_NEAR(multiLinkCost[0].A, 7.0f / (32.0f * 56.0e9f), 1e-14);
    EXPECT_FLOAT_EQ(multiLinkCost[0].A, singleCost[0].A / 2.0f);
}

TEST(AllGatherNhrCostTest, IncompleteSourcesFallBackToAnchor)
{
    CalcCostCoeffParam multiLink = MakeAllGatherParam("AicpuAllGatherSoleNHRMultiLink");
    multiLink.phyLevelPortNums = {{16}, {}};

    auto cost = InsTempAllGatherNHR::CalcCostCoeff(multiLink);

    ASSERT_EQ(cost.size(), 1u);
    EXPECT_NEAR(cost[0].A, 7.0f / (16.0f * 56.0e9f), 1e-14);
}
