/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_temp_all_reduce_mesh_1D_twoshot.h"
#include "selector_test_common.h"
#include <numeric>

namespace ops_hccl {
namespace aiv_selector_test {
    namespace {

        class AllReduceCoreBudgetTest : public AivTemplateSelectorTest {
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
                AivTempAllReduceMesh1DTwoShot algorithm(param_, 0, ranks);
                ASSERT_EQ(algorithm.KernelRun(param_, tempAlgParams, TemplateResource{}), HCCL_SUCCESS);
                ASSERT_EQ(kernelLaunches, 1u);
                EXPECT_EQ(launchedArgs.cmdType, HcclCMDType::HCCL_CMD_ALLREDUCE);
                EXPECT_EQ(launchedArgs.argsType, KernelArgsType::ARGS_TYPE_TWO_SHOT);
                EXPECT_EQ(launchedArgs.numBlocks, expected);
                EXPECT_LE(launchedArgs.numBlocks, limit);
                EXPECT_EQ(launchedArgs.isWrite, isWrite);
            }
        };

        TEST_F(AllReduceCoreBudgetTest, WriteUsesRankSpecificCoreBudget)
        {
            const u32 cases[][2]
                = {{1, 12},  {2, 24},  {3, 12},  {4, 12},  {5, 24},  {6, 24},  {7, 24},  {8, 24}, {9, 36},
                   {12, 36}, {16, 36}, {17, 36}, {18, 36}, {19, 36}, {32, 36}, {47, 36}, {48, 36}};
            for (u64 dataSize : {4ULL, 512 * 1024ULL, 512 * 1024ULL + 4, 2 * 1024 * 1024ULL}) {
                for (const auto& scenario : cases) {
                    CheckKernelBlocks(scenario[0], dataSize, 48, true, scenario[1]);
                }
            }
        }

        TEST_F(AllReduceCoreBudgetTest, WriteRespectsLowerUserLimit)
        {
            const u32 cases[][3] = {{2, 0, 0},   {2, 1, 1},   {2, 24, 24},  {2, 25, 24},  {2, 31, 24}, {2, 32, 24},
                                    {3, 13, 12}, {4, 10, 10}, {4, 13, 12},  {5, 35, 24},  {8, 37, 24}, {8, 48, 24},
                                    {8, 6, 6},   {9, 24, 24}, {16, 24, 24}, {17, 36, 36}, {48, 48, 36}};
            for (u64 dataSize : {512 * 1024ULL, 512 * 1024ULL + 4}) {
                for (const auto& scenario : cases) {
                    CheckKernelBlocks(scenario[0], dataSize, scenario[1], true, scenario[2]);
                }
            }
        }

        TEST_F(AllReduceCoreBudgetTest, ReadPreservesOriginalBlockAllocation)
        {
            // 读语义沿用原 CalNumBlocks：(limit / (rankSize + 1)) * (rankSize + 1)。
            CheckKernelBlocks(2, 512 * 1024ULL, 48, false, 48);
            CheckKernelBlocks(4, 512 * 1024ULL, 48, false, 45);
            CheckKernelBlocks(8, 512 * 1024ULL, 48, false, 45);
            CheckKernelBlocks(16, 512 * 1024ULL, 48, false, 34);
            CheckKernelBlocks(8, 512 * 1024ULL, 6, false, 6);
            CheckKernelBlocks(8, 4ULL, 48, false, 45);
        }

        TEST_F(AllReduceCoreBudgetTest, KernelRunPreservesSliceParametersWhenSwitchingSemantics)
        {
            TemplateDataParams tempAlgParams;
            tempAlgParams.sliceSize = 4096;
            tempAlgParams.inputSliceStride = 8192;
            tempAlgParams.outputSliceStride = 12288;
            tempAlgParams.repeatNum = 3;
            tempAlgParams.inputRepeatStride = 16384;
            tempAlgParams.outputRepeatStride = 20480;
            AivTempAllReduceMesh1DTwoShot algorithm(param_, 0, ranks_);
            u32 expectedSliceId = 0;
            for (bool isWrite : {true, false, true}) {
                SCOPED_TRACE(::testing::Message() << "isWrite=" << isWrite << " sliceId=" << expectedSliceId + 1);
                tempAlgParams.isWrite = isWrite;
                ASSERT_EQ(algorithm.KernelRun(param_, tempAlgParams, TemplateResource{}), HCCL_SUCCESS);
                ++expectedSliceId;
                EXPECT_EQ(kernelLaunches, expectedSliceId);
                EXPECT_EQ(launchedArgs.sliceId, expectedSliceId);
                EXPECT_EQ(launchedArgs.cmdType, HcclCMDType::HCCL_CMD_ALLREDUCE);
                EXPECT_EQ(launchedArgs.argsType, KernelArgsType::ARGS_TYPE_TWO_SHOT);
                EXPECT_EQ(launchedArgs.isWrite, isWrite);
                EXPECT_EQ(launchedArgs.numBlocks, isWrite ? 24u : 45u);
                EXPECT_EQ(launchedArgs.inputSliceStride, tempAlgParams.inputSliceStride);
                EXPECT_EQ(launchedArgs.outputSliceStride, tempAlgParams.outputSliceStride);
                EXPECT_EQ(launchedArgs.repeatNum, tempAlgParams.repeatNum);
                EXPECT_EQ(launchedArgs.inputRepeatStride, tempAlgParams.inputRepeatStride);
                EXPECT_EQ(launchedArgs.outputRepeatStride, tempAlgParams.outputRepeatStride);
            }
        }

    } // namespace
} // namespace aiv_selector_test
} // namespace ops_hccl
