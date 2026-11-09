/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CalcCostCoeff 参数传递：端口合并规则、host DPU RoCE 的 CPU 引擎带宽、B/C 系数入参。

#include "nhr_test_access.h"

namespace nhr_ut {
namespace {

    TEST(NhrCostCoeffTest, AggregatesPortsOnlyForPodClosAndUsesCpuEngine)
    {
        ResetStubState();
        CalcCostCoeffParam param{};
        param.rankSize = 8;
        param.dataRatio = 0.5f;
        param.netType = CommTopo::COMM_TOPO_CLOS;
        param.portNum = {2, 3, 7};
        param.isPod = true;

        const std::vector<CostModelParam> costs = InsTempReduceScatterNhrDpu::CalcCostCoeff(param);

        ASSERT_EQ(costs.size(), 1u);
        const CostCapture& capture = CostCalls();
        // pod+CLOS 时模板自行合并前两个端口，桩侧必须收到合并值且 isPod 为 false（否则会再次折半）
        EXPECT_EQ(capture.meshPortNum, 5);
        EXPECT_FALSE(capture.meshIsPod);
        EXPECT_EQ(capture.meshNetType, CommTopo::COMM_TOPO_CLOS);
        EXPECT_FLOAT_EQ(capture.meshN, param.dataRatio);
        EXPECT_EQ(capture.meshRankSize, param.rankSize);
        // host DPU 走 host 网卡 RoCE，带宽须按 CPU 引擎取 25GB/s 而非跨芯片 56GB/s
        EXPECT_EQ(capture.meshEngine, EngineType::CPU);

        EXPECT_FLOAT_EQ(costs[0].A, capture.meshA);
        EXPECT_FLOAT_EQ(costs[0].B, capture.localCopyB);
        EXPECT_FLOAT_EQ(costs[0].C, capture.dpuLatencyC);
        EXPECT_FLOAT_EQ(costs[0].D, 0.0f);
        EXPECT_FLOAT_EQ(capture.localCopyN, 0.5f * (1.0f + 1.0f / 8.0f));
        EXPECT_EQ(capture.localCopyEngine, EngineType::AICPU);
        EXPECT_EQ(capture.dpuStepNum, static_cast<int>(GetNHRStepNum(param.rankSize)));
        EXPECT_EQ(capture.dpuSyncNum, 1);
        EXPECT_EQ(capture.dpuChannelNum, 2);
        EXPECT_EQ(capture.dpuSndRcvNum, 7);
    }

    TEST(NhrCostCoeffTest, KeepsFirstPortWhenNotPodClos)
    {
        ResetStubState();
        CalcCostCoeffParam param{};
        param.rankSize = 4;
        param.dataRatio = 0.25f;
        param.netType = CommTopo::COMM_TOPO_1DMESH;
        param.portNum = {3, 4};
        param.isPod = true;

        ASSERT_EQ(InsTempReduceScatterNhrDpu::CalcCostCoeff(param).size(), 1u);

        EXPECT_EQ(CostCalls().meshPortNum, 3);
        EXPECT_EQ(CostCalls().meshEngine, EngineType::CPU);
    }

} // namespace
} // namespace nhr_ut
