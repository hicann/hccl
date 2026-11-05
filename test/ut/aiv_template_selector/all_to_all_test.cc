/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_temp_all_to_all_mesh_1D.h"
#include "aiv_alg_template_base.h"
#include "hccl_aiv_utils.h"
#include "selector_test_common.h"
#include <numeric>

namespace ops_hccl {
namespace aiv_selector_test {
    namespace {

        class AllToAllCoreBudgetTest : public AivTemplateSelectorTest {
        protected:
            void CheckKernelBlocks(u32 rankSize, u64 dataSize, u32 limit, bool isWrite, u32 expected)
            {
                SCOPED_TRACE(
                    ::testing::Message() << "ranks=" << rankSize << " bytes=" << dataSize << " limit=" << limit
                                         << " isWrite=" << isWrite);
                std::vector<std::vector<u32>> ranks(1, std::vector<u32>(rankSize));
                std::iota(ranks.front().begin(), ranks.front().end(), 0);
                sendCount_ = dataSize / DATATYPE_SIZE_TABLE[param_.all2AllVDataDes.sendType];
                param_.all2AllVDataDes.sendCounts = &sendCount_;
                param_.numBlocksLimit = limit;
                topo_.level0PcieMix = isWrite;
                topo_.userRankSize = rankSize;
                TemplateDataParams tempAlgParams;
                tempAlgParams.sendCounts = {sendCount_};
                tempAlgParams.sliceSize = dataSize;
                tempAlgParams.isWrite = IsAivWriteEnabled(param_, &topo_);
                kernelLaunches = 0;
                AivTempAlltoAllMesh1D algorithm(param_, 0, ranks);
                ASSERT_EQ(algorithm.KernelRun(param_, tempAlgParams, TemplateResource{}), HCCL_SUCCESS);
                ASSERT_EQ(kernelLaunches, 1u);
                EXPECT_EQ(launchedArgs.numBlocks, expected);
                EXPECT_LE(launchedArgs.numBlocks, limit);
                EXPECT_EQ(launchedArgs.isWrite, isWrite);
            }
        };

        TEST_F(AllToAllCoreBudgetTest, CalcResPreservesConstructorRanksForBothChannelPaths)
        {
            const std::vector<std::vector<u32>> expectedRanks = {{7, 3, 11}};
            for (Level0Shape shape : {Level0Shape::MESH_1D, Level0Shape::MESH_1D_CLOS}) {
                for (bool pcie : {false, true}) {
                    topo_.level0Topo = shape;
                    topo_.level0PcieMix = pcie;
                    topo_.userRank = 7;
                    topo_.userRankSize = 12;
                    auto constructorRanks = expectedRanks;
                    AivTempAlltoAllMesh1D algorithm(param_, topo_.userRank, constructorRanks);
                    // 模板须保存自己的rank列表，不能依赖调用方容器的后续内容或生命周期。
                    constructorRanks.clear();
                    channelRequestRanks.clear();
                    meshChannelRequests = 0;
                    meshClosChannelRequests = 0;
                    AlgResourceRequest request;

                    ASSERT_EQ(algorithm.CalcRes(nullptr, param_, &topo_, request), HCCL_SUCCESS);
                    EXPECT_EQ(channelRequestRanks, expectedRanks);
                    const bool meshClos = shape == Level0Shape::MESH_1D_CLOS && !pcie;
                    EXPECT_EQ(meshClosChannelRequests, meshClos ? 1u : 0u);
                    EXPECT_EQ(meshChannelRequests, meshClos ? 0u : 1u);
                    EXPECT_EQ(request.channels.size(), 1u);
                }
            }
        }

        TEST_F(AllToAllCoreBudgetTest, LargeDataLaunchesSameCoreCountForReadAndWrite)
        {
            const u32 cases[][2]
                = {{1, 48},  {2, 48},  {3, 48},  {4, 48},  {5, 45},  {6, 48},  {7, 42},  {8, 48},  {9, 45}, {10, 40},
                   {11, 44}, {12, 48}, {13, 39}, {14, 42}, {15, 45}, {16, 48}, {17, 34}, {32, 32}, {48, 48}};
            for (bool isWrite : {false, true}) {
                for (const auto& scenario : cases) {
                    CheckKernelBlocks(scenario[0], 512 * 1024 + 4, 48, isWrite, scenario[1]);
                }
            }
        }

        TEST_F(AllToAllCoreBudgetTest, LaunchCoreCountRespectsCoreLimitAndPeerAlignment)
        {
            const u32 cases[][3]
                = {{1, 3, 3},  {2, 7, 6},   {3, 3, 3},   {3, 5, 3},    {3, 14, 12},  {3, 15, 15},  {3, 16, 15},
                   {4, 7, 4},  {4, 8, 8},   {4, 15, 12}, {4, 16, 16},  {5, 14, 10},  {6, 11, 6},   {7, 13, 7},
                   {8, 15, 8}, {8, 16, 16}, {9, 26, 18}, {10, 29, 20}, {15, 29, 15}, {16, 31, 16}, {16, 32, 32}};
            for (bool isWrite : {false, true}) {
                for (const auto& scenario : cases) {
                    CheckKernelBlocks(scenario[0], 512 * 1024 + 4, scenario[1], isWrite, scenario[2]);
                    EXPECT_EQ(launchedArgs.numBlocks % scenario[0], 0u);
                }
            }
        }

        TEST_F(AllToAllCoreBudgetTest, SmallDataUsesOneCorePerPeerForReadAndWrite)
        {
            for (bool isWrite : {false, true}) {
                for (u32 rankSize : {1u, 2u, 3u, 4u, 8u, 17u, 48u}) {
                    for (u64 dataSize : {0ULL, 4ULL, 512 * 1024ULL}) {
                        CheckKernelBlocks(rankSize, dataSize, 48, isWrite, rankSize);
                    }
                }
            }
        }

        TEST_F(AllToAllCoreBudgetTest, LaunchBoundaryUsesSliceSizeForReadAndWrite)
        {
            const HcclDataType types[] = {HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_HIF8};
            for (bool isWrite : {false, true}) {
                for (HcclDataType type : types) {
                    param_.all2AllVDataDes.sendType = type;
                    CheckKernelBlocks(4, 512 * 1024, 48, isWrite, 4);
                    CheckKernelBlocks(4, 512 * 1024 + 1, 48, isWrite, 48);
                    CheckKernelBlocks(4, 512 * 1024 + 4, 48, isWrite, 48);
                }
            }
        }

        TEST_F(AllToAllCoreBudgetTest, FewCoreAndZeroLimitKeepTheirPeerAssignment)
        {
            const u32 cases[][3] = {{8, 0, 0}, {8, 1, 1}, {8, 6, 4}, {16, 6, 6}, {48, 31, 24}, {48, 48, 48}};
            for (bool isWrite : {false, true}) {
                for (u64 dataSize : {4ULL, 512 * 1024ULL + 4}) {
                    for (const auto& scenario : cases) {
                        CheckKernelBlocks(scenario[0], dataSize, scenario[1], isWrite, scenario[2]);
                    }
                }
            }
        }

        TEST_F(AllToAllCoreBudgetTest, LargeDataKeepsRankAndSizeCapForReadAndWrite)
        {
            for (bool isWrite : {false, true}) {
                CheckKernelBlocks(8, 2 * 1024 * 1024 - 4, 48, isWrite, 48);
                CheckKernelBlocks(8, 2 * 1024 * 1024, 48, isWrite, 32);
                CheckKernelBlocks(8, 2 * 1024 * 1024 + 4, 48, isWrite, 32);
                CheckKernelBlocks(8, 2 * 1024 * 1024, 24, isWrite, 24);
                CheckKernelBlocks(4, 2 * 1024 * 1024, 48, isWrite, 48);
                CheckKernelBlocks(9, 2 * 1024 * 1024, 48, isWrite, 45);
            }
        }

        TEST_F(AllToAllCoreBudgetTest, BaseInterfaceUsesSharedBlockAllocation)
        {
            AivTempAlltoAllMesh1D algorithm(param_, 0, ranks_);
            AivAlgTemplateBase& base = algorithm;
            u32 numBlocks = 0;
            EXPECT_EQ(base.CalNumBlocks(numBlocks, 512 * 1024, 48), HCCL_SUCCESS);
            EXPECT_EQ(numBlocks, 8u);
            EXPECT_EQ(base.CalNumBlocks(numBlocks, 512 * 1024 + 4, 6), HCCL_SUCCESS);
            EXPECT_EQ(numBlocks, 4u);
            EXPECT_EQ(base.CalNumBlocks(numBlocks, 2 * 1024 * 1024, 48), HCCL_SUCCESS);
            EXPECT_EQ(numBlocks, 32u);
        }

        TEST_F(AllToAllCoreBudgetTest, SemanticSelectionKeepsSameHostCoreAllocation)
        {
            struct Scenario {
                OpMode mode;
                bool pcie;
                bool hasTopo;
                bool allMesh;
                bool expectedWrite;
            };
            const Scenario cases[]
                = {{OpMode::OPBASE, true, true, false, true},
                   {OpMode::OPBASE, true, true, true, false},
                   {OpMode::OPBASE, false, true, false, false},
                   {OpMode::OFFLOAD, true, true, false, false},
                   {OpMode::OPBASE, true, false, false, false}};
            for (const auto& scenario : cases) {
                SCOPED_TRACE(
                    ::testing::Message() << "mode=" << static_cast<int>(scenario.mode) << " pcie=" << scenario.pcie
                                         << " hasTopo=" << scenario.hasTopo << " allMesh=" << scenario.allMesh);
                param_.opMode = scenario.mode;
                param_.numBlocksLimit = 48;
                topo_.level0PcieMix = scenario.pcie;
                topo_.level0AllMesh = scenario.allMesh;
                sendCount_ = 512 * 1024 / DATATYPE_SIZE_TABLE[param_.all2AllVDataDes.sendType];
                param_.all2AllVDataDes.sendCounts = &sendCount_;
                TemplateDataParams tempAlgParams;
                tempAlgParams.sendCounts = {sendCount_};
                tempAlgParams.sliceSize = 512 * 1024;
                tempAlgParams.isWrite = IsAivWriteEnabled(param_, scenario.hasTopo ? &topo_ : nullptr);
                kernelLaunches = 0;
                AivTempAlltoAllMesh1D algorithm(param_, 0, ranks_);
                ASSERT_EQ(algorithm.KernelRun(param_, tempAlgParams, TemplateResource{}), HCCL_SUCCESS);
                EXPECT_EQ(launchedArgs.numBlocks, 8u);
                EXPECT_EQ(launchedArgs.isWrite, scenario.expectedWrite);
            }
        }

    } // namespace
} // namespace aiv_selector_test
} // namespace ops_hccl
