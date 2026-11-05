/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_temp_reduce_scatter_mesh_1D.h"
#include "selector_test_common.h"
#include <numeric>

namespace ops_hccl {
namespace aiv_selector_test {
    namespace {

        TEST_F(AivTemplateSelectorTest, ReduceScatterRetainsOriginalAlgorithmAndCommonWriteGates)
        {
            EXPECT_TRUE(IsReduceScatterAivWriteEnabled(param_, &topo_));
            param_.opMode = OpMode::OFFLOAD;
            EXPECT_FALSE(IsReduceScatterAivWriteEnabled(param_, &topo_));
            param_.opMode = OpMode::OPBASE;
            topo_.level0PcieMix = false;
            EXPECT_FALSE(IsReduceScatterAivWriteEnabled(param_, &topo_));
            topo_.level0PcieMix = true;
            EXPECT_FALSE(IsReduceScatterAivWriteEnabled(param_, nullptr));
            EXPECT_EQ(AivTempReduceScatterMesh1D::props.algoType, AlgoType::MESH);
        }

        TEST_F(AivTemplateSelectorTest, ReduceScatterFullMeshSelectsReadAtLaunch)
        {
            AivTempReduceScatterMesh1D algorithm(param_, 0, ranks_);
            for (bool allMesh : {false, true, false}) {
                topo_.level0AllMesh = allMesh;
                TemplateDataParams data;
                data.isWrite = IsReduceScatterAivWriteEnabled(param_, &topo_);
                data.sliceSize = param_.DataDes.count * DATATYPE_SIZE_TABLE[param_.DataDes.dataType];
                data.inputSliceStride = data.sliceSize;
                data.outputSliceStride = data.sliceSize;
                kernelLaunches = 0;
                ASSERT_EQ(algorithm.KernelRun(param_, data, TemplateResource{}), HCCL_SUCCESS);
                ASSERT_EQ(kernelLaunches, 1u);
                EXPECT_EQ(launchedArgs.isWrite, !allMesh);
            }
        }

        TEST_F(AivTemplateSelectorTest, ReduceScatterHif8DoesNotSelectWrite)
        {
            param_.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;
            EXPECT_FALSE(IsReduceScatterAivWriteEnabled(param_, &topo_));
            EXPECT_EQ(aiv_selector_test::strictOrderQueries, 0u);
        }

        TEST_F(AivTemplateSelectorTest, ReduceScatterStrictOrderStillSelectsWrite)
        {
            strictOrderRequired = true;
            EXPECT_TRUE(IsReduceScatterAivWriteEnabled(param_, &topo_));
            EXPECT_EQ(strictOrderQueries, 0u);
        }

        class ReduceScatterCoreBudgetTest : public AivTemplateSelectorTest {
        protected:
            void CheckKernelBlocks(u32 rankSize, u64 dataSize, u32 limit, bool isWrite, u32 expected)
            {
                SCOPED_TRACE(
                    ::testing::Message() << "ranks=" << rankSize << " bytes=" << dataSize << " limit=" << limit
                                         << " isWrite=" << isWrite);
                std::vector<std::vector<u32>> ranks(1, std::vector<u32>(rankSize));
                std::iota(ranks.front().begin(), ranks.front().end(), 0);
                param_.numBlocksLimit = limit;
                TemplateDataParams tempAlgParams;
                tempAlgParams.sliceSize = dataSize;
                tempAlgParams.inputSliceStride = dataSize;
                tempAlgParams.outputSliceStride = dataSize;
                tempAlgParams.isWrite = isWrite;
                kernelLaunches = 0;
                AivTempReduceScatterMesh1D algorithm(param_, 0, ranks);
                ASSERT_EQ(algorithm.KernelRun(param_, tempAlgParams, TemplateResource{}), HCCL_SUCCESS);
                ASSERT_EQ(kernelLaunches, 1u);
                EXPECT_EQ(launchedArgs.cmdType, HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
                EXPECT_EQ(launchedArgs.numBlocks, expected);
                EXPECT_LE(launchedArgs.numBlocks, limit);
                EXPECT_EQ(launchedArgs.isWrite, isWrite);
            }
        };

        TEST_F(ReduceScatterCoreBudgetTest, WriteLargeDataUsesRankSpecificCoreBudget)
        {
            const u32 cases[][2]
                = {{1, 12},  {2, 48},  {3, 12},  {4, 12},  {5, 24},  {6, 24},  {7, 24},  {8, 24},  {9, 48},
                   {10, 48}, {11, 48}, {12, 48}, {13, 48}, {14, 48}, {15, 48}, {16, 48}, {17, 48}, {48, 48}};
            const u64 dataSizes[] = {512 * 1024ULL, 512 * 1024ULL + 4, 2 * 1024 * 1024ULL};
            for (u64 dataSize : dataSizes) {
                for (const auto& scenario : cases) {
                    CheckKernelBlocks(scenario[0], dataSize, 48, true, scenario[1]);
                }
            }
        }

        TEST_F(ReduceScatterCoreBudgetTest, WriteSmallDataUsesLocalTreeBudget)
        {
            const u32 cases[][2] = {{1, 2},  {2, 4},  {3, 6},   {4, 8},   {5, 10},  {6, 12}, {7, 14},
                                    {8, 16}, {9, 18}, {16, 32}, {17, 34}, {18, 36}, {48, 48}};
            for (u64 dataSize : {0ULL, 4ULL, 512 * 1024ULL - 4}) {
                for (const auto& scenario : cases) {
                    CheckKernelBlocks(scenario[0], dataSize, 48, true, scenario[1]);
                    EXPECT_LE(launchedArgs.numBlocks, 2 * scenario[0]);
                }
            }
        }

        TEST_F(ReduceScatterCoreBudgetTest, ReadSmallDataPreservesTwoStageBudget)
        {
            const u32 cases[][2]
                = {{1, 2}, {2, 4}, {3, 6}, {4, 8}, {5, 10}, {8, 16}, {9, 18}, {16, 32}, {17, 34}, {18, 36}, {48, 48}};
            for (u64 dataSize : {0ULL, 4ULL, 512 * 1024ULL - 4}) {
                for (const auto& scenario : cases) {
                    CheckKernelBlocks(scenario[0], dataSize, 48, false, scenario[1]);
                }
            }
        }

        TEST_F(ReduceScatterCoreBudgetTest, ReadLargeDataPreservesUserLimit)
        {
            for (u32 rankSize : {2u, 4u, 8u, 16u, 48u}) {
                for (u64 dataSize : {512 * 1024ULL, 512 * 1024ULL + 4, 2 * 1024 * 1024ULL}) {
                    CheckKernelBlocks(rankSize, dataSize, 48, false, 48);
                    CheckKernelBlocks(rankSize, dataSize, 24, false, 24);
                }
            }
        }

        TEST_F(ReduceScatterCoreBudgetTest, WriteLaunchBudgetRespectsLowerUserLimit)
        {
            const u32 cases[][3]
                = {{2, 0, 0},   {2, 24, 24}, {2, 31, 31}, {2, 32, 32}, {2, 33, 33}, {2, 44, 44},  {2, 48, 48},
                   {2, 64, 64}, {4, 0, 0},   {4, 1, 1},   {4, 6, 6},   {8, 12, 12}, {16, 24, 24}, {48, 6, 6}};
            for (u64 dataSize : {512 * 1024ULL, 512 * 1024ULL + 4}) {
                for (const auto& scenario : cases) {
                    CheckKernelBlocks(scenario[0], dataSize, scenario[1], true, scenario[2]);
                }
                CheckKernelBlocks(4, dataSize, 13, true, 12);
                CheckKernelBlocks(8, dataSize, 25, true, 24);
                CheckKernelBlocks(16, dataSize, 37, true, 37);
                CheckKernelBlocks(4, dataSize, 10, true, 10);
            }
        }

        TEST_F(ReduceScatterCoreBudgetTest, WriteSmallDataRespectsLowerUserLimit)
        {
            const u32 cases[][3]
                = {{2, 0, 0},   {2, 1, 1},   {2, 3, 3},    {2, 4, 4},    {2, 24, 4}, {4, 6, 6},   {4, 13, 8},
                   {8, 12, 12}, {8, 25, 16}, {16, 24, 24}, {16, 37, 32}, {48, 6, 6}, {48, 48, 48}};
            for (u64 dataSize : {4ULL, 512 * 1024ULL - 4}) {
                for (const auto& scenario : cases) {
                    CheckKernelBlocks(scenario[0], dataSize, scenario[1], true, scenario[2]);
                }
            }
        }

        TEST_F(ReduceScatterCoreBudgetTest, BaseInterfacePreservesReadBlockAllocation)
        {
            AivTempReduceScatterMesh1D algorithm(param_, 0, ranks_);
            AivAlgTemplateBase& base = algorithm;
            u32 numBlocks = 0;
            EXPECT_EQ(base.CalNumBlocks(numBlocks, 512 * 1024, 48), HCCL_SUCCESS);
            EXPECT_EQ(numBlocks, 48u);
            EXPECT_EQ(base.CalNumBlocks(numBlocks, 512 * 1024 - 4, 48), HCCL_SUCCESS);
            EXPECT_EQ(numBlocks, 16u);
            EXPECT_EQ(base.CalNumBlocks(numBlocks, 512 * 1024, 6), HCCL_SUCCESS);
            EXPECT_EQ(numBlocks, 6u);
            CheckKernelBlocks(4, 4, 10, false, 8);
        }

    } // namespace
} // namespace aiv_selector_test
} // namespace ops_hccl
