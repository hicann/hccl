/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "alltoall_auto_selector.h"
#include "alltoallv_auto_selector.h"
#include "alltoallvc_auto_selector.h"

namespace ops_hccl {
namespace {

    uint8_t g_input[0x100];
    uint8_t g_output[0x100];

    OpParam MakeOpParam(HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLTOALL)
    {
        OpParam op{};
        op.opType = opType;
        op.inputPtr = g_input;
        op.inputSize = sizeof(g_input);
        op.outputPtr = g_output;
        op.outputSize = sizeof(g_output);
        return op;
    }

    TopoInfoWithNetLayerDetails MakeTopo(u32 levelNums, Level0Shape level0Topo)
    {
        TopoInfoWithNetLayerDetails topo{};
        topo.topoLevelNums = levelNums;
        topo.level0Topo = level0Topo;
        return topo;
    }

    template <typename Selector>
    SelectorStatus
    SelectNda(Selector& selector, const TopoInfoWithNetLayerDetails* topo, const OpParam& op, std::string& algName)
    {
        const std::map<HcclCMDType, std::vector<HcclAlgoType>> configAlgMap;
        return selector.SelectNdaAlgo(topo, op, configAlgMap, algName);
    }

    class AlltoAllNdaSelectorTest : public ::testing::Test {};

    TEST_F(AlltoAllNdaSelectorTest, SelectNdaAlgoRejectsNullTopoInfo)
    {
        AlltoAllAutoSelector selector;
        OpParam op = MakeOpParam();
        std::string algName;

        EXPECT_EQ(SelectNda(selector, nullptr, op, algName), SelectorStatus::NOT_MATCH);
        EXPECT_TRUE(algName.empty());
    }

    TEST_F(AlltoAllNdaSelectorTest, SelectNdaAlgoRejectsSingleLevelTopo)
    {
        AlltoAllAutoSelector selector;
        OpParam op = MakeOpParam();
        TopoInfoWithNetLayerDetails topo = MakeTopo(1, Level0Shape::MESH_1D);
        std::string algName;

        EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::NOT_MATCH);
    }

    TEST_F(AlltoAllNdaSelectorTest, SelectNdaAlgoMatchesFourLevelTopo)
    {
        AlltoAllAutoSelector selector;
        OpParam op = MakeOpParam();
        // NDA 编排只依赖单层全 rank，与 CheckSupportNda 一致不设层级上限，4 层同样匹配
        TopoInfoWithNetLayerDetails topo = MakeTopo(4, Level0Shape::MESH_1D);
        std::string algName;

        EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::MATCH);
        EXPECT_EQ(algName, "DpuAllToAllSoleMeshNDA");
    }

    TEST_F(AlltoAllNdaSelectorTest, SelectNdaAlgoRejectsInputOutputOverlap)
    {
        AlltoAllAutoSelector selector;
        OpParam op = MakeOpParam();
        op.outputPtr = op.inputPtr; // in-place
        TopoInfoWithNetLayerDetails topo = MakeTopo(2, Level0Shape::MESH_1D);
        std::string algName;

        EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::NOT_MATCH);
    }

    TEST_F(AlltoAllNdaSelectorTest, SelectNdaAlgoRejectsUnsupportedLevel0Topo)
    {
        AlltoAllAutoSelector selector;
        OpParam op = MakeOpParam();
        // Level0Shape 仅有 CLOS/MESH_1D/MESH_1D_CLOS 三态，用越界值覆盖"非支持拓扑"守卫分支
        TopoInfoWithNetLayerDetails topo = MakeTopo(2, static_cast<Level0Shape>(99));
        std::string algName;

        EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::NOT_MATCH);
    }

    TEST_F(AlltoAllNdaSelectorTest, SelectNdaAlgoMatchesMesh1DTwoLevels)
    {
        AlltoAllAutoSelector selector;
        OpParam op = MakeOpParam();
        TopoInfoWithNetLayerDetails topo = MakeTopo(2, Level0Shape::MESH_1D);
        std::string algName;

        EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::MATCH);
        EXPECT_EQ(algName, "DpuAllToAllSoleMeshNDA");
    }

    TEST_F(AlltoAllNdaSelectorTest, SelectNdaAlgoMatchesMesh1DClosThreeLevels)
    {
        AlltoAllAutoSelector selector;
        OpParam op = MakeOpParam();
        TopoInfoWithNetLayerDetails topo = MakeTopo(3, Level0Shape::MESH_1D_CLOS);
        std::string algName;

        EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::MATCH);
        EXPECT_EQ(algName, "DpuAllToAllSoleMeshNDA");
    }

    TEST_F(AlltoAllNdaSelectorTest, SelectNdaAlgoMatchesClosTwoLevels)
    {
        AlltoAllAutoSelector selector;
        OpParam op = MakeOpParam();
        TopoInfoWithNetLayerDetails topo = MakeTopo(2, Level0Shape::CLOS);
        std::string algName;

        EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::MATCH);
        EXPECT_EQ(algName, "DpuAllToAllSoleMeshNDA");
    }

    // alltoallv / alltoallvc 的 SelectNdaAlgo 复用同一套守卫，这里参数化覆盖其层级边界、重叠拦截与算法名断言
    template <typename Selector>
    void ExpectNdaSelectorGuards(HcclCMDType opType, const char* expectedAlgName)
    {
        {
            Selector selector;
            OpParam op = MakeOpParam(opType);
            std::string algName;
            EXPECT_EQ(SelectNda(selector, nullptr, op, algName), SelectorStatus::NOT_MATCH);
            EXPECT_TRUE(algName.empty());
        }
        {
            Selector selector;
            OpParam op = MakeOpParam(opType);
            TopoInfoWithNetLayerDetails topo = MakeTopo(1, Level0Shape::MESH_1D);
            std::string algName;
            EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::NOT_MATCH);
        }
        {
            Selector selector;
            OpParam op = MakeOpParam(opType);
            // 层级只限制下界，与 CheckSupportNda 保持一致，4 层同样匹配
            TopoInfoWithNetLayerDetails topo = MakeTopo(4, Level0Shape::MESH_1D);
            std::string algName;
            EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::MATCH);
            EXPECT_EQ(algName, expectedAlgName);
        }
        {
            Selector selector;
            OpParam op = MakeOpParam(opType);
            op.outputPtr = op.inputPtr; // in-place
            TopoInfoWithNetLayerDetails topo = MakeTopo(2, Level0Shape::MESH_1D);
            std::string algName;
            EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::NOT_MATCH);
        }
        {
            Selector selector;
            OpParam op = MakeOpParam(opType);
            TopoInfoWithNetLayerDetails topo = MakeTopo(2, Level0Shape::MESH_1D);
            std::string algName;
            EXPECT_EQ(SelectNda(selector, &topo, op, algName), SelectorStatus::MATCH);
            EXPECT_EQ(algName, expectedAlgName);
        }
    }

    TEST_F(AlltoAllNdaSelectorTest, AlltoAllVNdaSelectorGuards)
    {
        ExpectNdaSelectorGuards<AlltoAllVAutoSelector>(HcclCMDType::HCCL_CMD_ALLTOALLV, "DpuAllToAllVSoleMeshNDA");
    }

    TEST_F(AlltoAllNdaSelectorTest, AlltoAllVCNdaSelectorGuards)
    {
        ExpectNdaSelectorGuards<AlltoAllVCAutoSelector>(HcclCMDType::HCCL_CMD_ALLTOALLVC, "DpuAllToAllVCSoleMeshNDA");
    }

} // namespace
} // namespace ops_hccl
