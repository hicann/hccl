/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "aiv_temp_all_gather_mesh_1D.h"
#include "selector_test_common.h"

namespace ops_hccl {
namespace aiv_selector_test {
    namespace {

        TEST_F(AivTemplateSelectorTest, AllGatherForwardsRuntimeWriteFlagWithoutTemplateState)
        {
            AivTempAllGatherMesh1D algorithm(param_, 3, ranks_);
            for (OpMode mode : {OpMode::OPBASE, OpMode::OFFLOAD}) {
                for (bool pcie : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << "mode=" << static_cast<int>(mode) << " pcie=" << pcie);
                    param_.opMode = mode;
                    topo_.level0PcieMix = pcie;
                    const bool write = mode == OpMode::OPBASE && pcie;
                    TemplateDataParams data;
                    data.isWrite = IsAivWriteEnabled(param_, &topo_);
                    data.sliceSize = param_.DataDes.count * DATATYPE_SIZE_TABLE[param_.DataDes.dataType];
                    data.buffInfo.hcclBuff.size = data.sliceSize * ranks_.front().size();
                    data.outputSliceStride = data.sliceSize;
                    kernelLaunches = 0;
                    ASSERT_EQ(algorithm.KernelRun(param_, data, TemplateResource{}), HCCL_SUCCESS);
                    ASSERT_EQ(kernelLaunches, 1u);
                    EXPECT_EQ(launchedArgs.cmdType, HcclCMDType::HCCL_CMD_ALLGATHER);
                    EXPECT_EQ(launchedArgs.argsType, KernelArgsType::ARGS_TYPE_SERVER);
                    EXPECT_EQ(launchedArgs.isWrite, write);
                    EXPECT_EQ(launchedArgs.rank, 3u);
                    EXPECT_EQ(launchedArgs.rankSize, 8u);
                    EXPECT_EQ(launchedArgs.count, param_.DataDes.count);
                }
            }
        }

        TEST_F(AivTemplateSelectorTest, AllGatherFullMeshSelectsReadAtLaunch)
        {
            AivTempAllGatherMesh1D algorithm(param_, 0, ranks_);
            for (bool allMesh : {false, true, false}) {
                topo_.level0AllMesh = allMesh;
                TemplateDataParams data;
                data.isWrite = IsAivWriteEnabled(param_, &topo_);
                data.sliceSize = param_.DataDes.count * DATATYPE_SIZE_TABLE[param_.DataDes.dataType];
                data.buffInfo.hcclBuff.size = data.sliceSize * ranks_.front().size();
                data.outputSliceStride = data.sliceSize;
                kernelLaunches = 0;
                ASSERT_EQ(algorithm.KernelRun(param_, data, TemplateResource{}), HCCL_SUCCESS);
                ASSERT_EQ(kernelLaunches, 1u);
                EXPECT_EQ(launchedArgs.isWrite, !allMesh);
            }
        }

        TEST_F(AivTemplateSelectorTest, AllGatherWriteLaunchChecksEntireGatherCapacity)
        {
            AivTempAllGatherMesh1D algorithm(param_, 0, ranks_);
            const u64 capacity = 4096;
            const u64 limit = capacity / ranks_.front().size();
            const u64 typeSize = DATATYPE_SIZE_TABLE[param_.DataDes.dataType];
            for (u64 slice : {limit, limit + typeSize, capacity, std::numeric_limits<u64>::max()}) {
                SCOPED_TRACE(::testing::Message() << "slice=" << slice);
                TemplateDataParams data;
                data.isWrite = true;
                data.sliceSize = slice;
                data.buffInfo.hcclBuff.size = capacity;
                kernelLaunches = 0;
                const bool fits = slice <= limit;
                EXPECT_EQ(algorithm.KernelRun(param_, data, TemplateResource{}), fits ? HCCL_SUCCESS : HCCL_E_PARA);
                EXPECT_EQ(kernelLaunches, fits ? 1u : 0u);
            }
        }

        TEST_F(AivTemplateSelectorTest, AllGatherWriteUsesRankBandsWithinAvailableCoreLimit)
        {
            struct Scenario {
                u32 ranks;
                u32 targetBlocks;
            };
            const Scenario scenarios[]
                = {{1, 24},  {2, 24},  {3, 12},  {4, 12},  {5, 24},  {6, 24},  {7, 24},  {8, 24},  {9, 24},
                   {10, 24}, {11, 24}, {12, 24}, {13, 24}, {14, 24}, {15, 24}, {16, 24}, {17, 24}, {24, 24}};
            for (const auto& scenario : scenarios) {
                std::vector<std::vector<u32>> ranks(1);
                for (u32 rank = 0; rank < scenario.ranks; ++rank) {
                    ranks.front().push_back(rank);
                }
                AivTempAllGatherMesh1D algorithm(param_, 0, ranks);
                for (u32 limit : {1u, 6u, 8u, 12u, 16u, 24u, 32u, 48u}) {
                    param_.numBlocksLimit = limit;
                    for (u64 slice : {512ULL, 512ULL * 1024 + 4}) {
                        for (bool write : {true, false}) {
                            SCOPED_TRACE(
                                ::testing::Message() << "ranks=" << scenario.ranks << " limit=" << limit
                                                     << " slice=" << slice << " write=" << write);
                            TemplateDataParams data;
                            data.isWrite = write;
                            data.sliceSize = slice;
                            data.outputSliceStride = slice;
                            data.buffInfo.hcclBuff.size = slice * scenario.ranks;
                            kernelLaunches = 0;
                            ASSERT_EQ(algorithm.KernelRun(param_, data, TemplateResource{}), HCCL_SUCCESS);
                            ASSERT_EQ(kernelLaunches, 1u);
                            // 读语义沿用完整核预算；写语义才按rank区间控核。
                            const u32 expectedBlocks = write ? std::min(scenario.targetBlocks, limit) : limit;
                            EXPECT_EQ(launchedArgs.numBlocks, expectedBlocks);
                            EXPECT_EQ(launchedArgs.isWrite, write);
                            EXPECT_EQ(launchedArgs.rankSize, scenario.ranks);
                        }
                    }
                }
            }
        }

        TEST_F(AivTemplateSelectorTest, AllGather256MiBTotalRequiresTwoWriteLaunches)
        {
            const u64 totalBytes = 256 * 1024 * 1024;
            const u64 capacity = 200 * 1024 * 1024;
            const u64 typeSize = DATATYPE_SIZE_TABLE[param_.DataDes.dataType];
            for (u32 rankSize : {2u, 4u, 8u}) {
                SCOPED_TRACE(::testing::Message() << "ranks=" << rankSize);
                std::vector<std::vector<u32>> ranks(1);
                for (u32 rank = 0; rank < rankSize; rank++) {
                    ranks.front().push_back(rank);
                }
                const u64 perRankBytes = totalBytes / rankSize;
                param_.DataDes.count = perRankBytes / typeSize;
                AivTempAllGatherMesh1D algorithm(param_, 0, ranks);
                TemplateDataParams data;
                data.isWrite = true;
                data.buffInfo.hcclBuff.size = capacity;
                data.outputSliceStride = perRankBytes;
                data.sliceSize = perRankBytes;
                kernelLaunches = 0;
                EXPECT_EQ(algorithm.KernelRun(param_, data, TemplateResource{}), HCCL_E_PARA);
                EXPECT_EQ(kernelLaunches, 0u);

                u64 processedBytes = 0;
                for (u64 slice : {capacity / rankSize, (totalBytes - capacity) / rankSize}) {
                    data.sliceSize = slice;
                    data.buffInfo.inBuffBaseOff = processedBytes;
                    data.buffInfo.outBuffBaseOff = processedBytes;
                    ASSERT_EQ(algorithm.KernelRun(param_, data, TemplateResource{}), HCCL_SUCCESS);
                    EXPECT_EQ(launchedArgs.count, slice / typeSize);
                    EXPECT_EQ(launchedArgs.input, processedBytes);
                    EXPECT_EQ(launchedArgs.output, processedBytes);
                    EXPECT_EQ(launchedArgs.outputSliceStride, perRankBytes);
                    processedBytes += slice;
                }
                EXPECT_EQ(kernelLaunches, 2u);
                EXPECT_EQ(processedBytes, perRankBytes);
            }
        }

        TEST_F(AivTemplateSelectorTest, AllGatherWriteRejectsEmptyBufferBeforeLaunch)
        {
            AivTempAllGatherMesh1D algorithm(param_, 0, ranks_);
            TemplateDataParams data;
            data.isWrite = true;
            data.sliceSize = DATATYPE_SIZE_TABLE[param_.DataDes.dataType];
            EXPECT_EQ(algorithm.KernelRun(param_, data, TemplateResource{}), HCCL_E_PARA);
            EXPECT_EQ(kernelLaunches, 0u);
        }

        TEST_F(AivTemplateSelectorTest, AllGatherWriteRejectsZeroRanksBeforeLaunch)
        {
            AivTempAllGatherMesh1D algorithm(param_, 0, {{}});
            TemplateDataParams data;
            data.isWrite = true;
            data.sliceSize = DATATYPE_SIZE_TABLE[param_.DataDes.dataType];
            data.buffInfo.hcclBuff.size = 4096;
            EXPECT_EQ(algorithm.KernelRun(param_, data, TemplateResource{}), HCCL_E_PARA);
            EXPECT_EQ(kernelLaunches, 0u);
        }

        TEST_F(AivTemplateSelectorTest, AllGatherMissingTopologyDisablesWrite)
        {
            EXPECT_FALSE(IsAivWriteEnabled(param_, nullptr));
        }

        TEST_F(AivTemplateSelectorTest, AllGatherRetainsOriginalAlgorithmAndCommonWriteGates)
        {
            EXPECT_TRUE(IsAivWriteEnabled(param_, &topo_));
            param_.opMode = OpMode::OFFLOAD;
            EXPECT_FALSE(IsAivWriteEnabled(param_, &topo_));
            param_.opMode = OpMode::OPBASE;
            topo_.level0PcieMix = false;
            EXPECT_FALSE(IsAivWriteEnabled(param_, &topo_));
            EXPECT_EQ(AivTempAllGatherMesh1D::props.algoType, AlgoType::MESH);
        }

    } // namespace
} // namespace aiv_selector_test
} // namespace ops_hccl
