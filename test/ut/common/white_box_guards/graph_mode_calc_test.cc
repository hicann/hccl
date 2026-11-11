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

#include <cstring>
#include <string>
#include <utility>

#include "calc_resource_graph_mode.h"
#include "hcom.h"

namespace ops_hccl {
namespace {

    constexpr uint64_t MIB = 1024ULL * 1024ULL;
    constexpr uint64_t UB_MAX_DATA_SIZE_UT = 256ULL * MIB;
    constexpr uint32_t GE_PARALLEL_UT = 36;

    std::pair<HcclResult, uint32_t> RunCalcTaskNum(
        const std::string& opType, uint64_t dataCount, uint32_t rankSize, uint64_t hcclBufferSize,
        HcclDataType dataType = HCCL_DATA_TYPE_INT8)
    {
        OpParamGraphMode param{};
        std::strncpy(param.opType, opType.c_str(), sizeof(param.opType) - 1);
        param.dataCount = dataCount;
        param.rankSize = rankSize;
        param.hcclBufferSize = hcclBufferSize;
        param.dataType = dataType;
        uint32_t taskNum = 0;
        HcclResult ret = CalcTaskNum(&param, taskNum);
        return {ret, taskNum};
    }

    TEST(GraphModeCalcTaskNumTest, ScratchBoundBranchesRejectZeroMaxDataCountPerLoop)
    {
        for (const auto& opType :
             {HCCL_KERNEL_OP_TYPE_REDUCE, HCCL_KERNEL_OP_TYPE_REDUCESCATTER, HCCL_KERNEL_OP_TYPE_SCATTER,
              HCCL_KERNEL_OP_TYPE_REDUCESCATTERV}) {
            auto result = RunCalcTaskNum(opType, 1, 1, 1, HCCL_DATA_TYPE_INT64);
            EXPECT_EQ(result.first, HCCL_E_PARA) << opType;
            EXPECT_EQ(result.second, 0u) << opType;
        }

        // ALLREDUCE: scratchBoundDataSize = 1 / 1 / 128 * 128 = 0。
        auto result = RunCalcTaskNum(HCCL_KERNEL_OP_TYPE_ALLREDUCE, 1, 1, 1, HCCL_DATA_TYPE_INT64);
        EXPECT_EQ(result.first, HCCL_E_PARA);
        EXPECT_EQ(result.second, 0u);
    }

    TEST(GraphModeCalcTaskNumTest, AllToAllRejectsZeroMaxDataCountPerLoop)
    {
        // 256MB / 8B / 33554433 = 0，覆盖 rankSize 过大时的除零路径。
        auto result = RunCalcTaskNum(HCCL_KERNEL_OP_TYPE_ALLTOALL, 1, 33554433, 1, HCCL_DATA_TYPE_INT64);
        EXPECT_EQ(result.first, HCCL_E_PARA);
        EXPECT_EQ(result.second, 0u);
    }

    TEST(GraphModeCalcTaskNumTest, AllToAllVBranchesReturnOneTask)
    {
        for (const auto& opType : {HCCL_KERNEL_OP_TYPE_ALLTOALLV, HCCL_KERNEL_OP_TYPE_ALLTOALLVC}) {
            auto result = RunCalcTaskNum(opType, 1, 1, MIB);
            EXPECT_EQ(result.first, HCCL_SUCCESS) << opType;
            EXPECT_EQ(result.second, 1u) << opType;
        }
    }

    TEST(GraphModeCalcTaskNumTest, ZeroBufferOrRankReturnsDefaultTaskNum)
    {
        auto bufferZero = RunCalcTaskNum(HCCL_KERNEL_OP_TYPE_ALLREDUCE, 1, 8, 0);
        EXPECT_EQ(bufferZero.first, HCCL_SUCCESS);
        EXPECT_EQ(bufferZero.second, GE_PARALLEL_UT);

        auto rankZero = RunCalcTaskNum(HCCL_KERNEL_OP_TYPE_ALLREDUCE, 1, 0, MIB);
        EXPECT_EQ(rankZero.first, HCCL_SUCCESS);
        EXPECT_EQ(rankZero.second, GE_PARALLEL_UT);
    }

    TEST(GraphModeCalcTaskNumTest, NormalPathsKeepExpectedTaskNum)
    {
        // scratch 受限分支：1MB buffer、int8、1MB + 1 个元素 -> 2 轮。
        for (const auto& opType :
             {HCCL_KERNEL_OP_TYPE_REDUCE, HCCL_KERNEL_OP_TYPE_REDUCESCATTER, HCCL_KERNEL_OP_TYPE_SCATTER,
              HCCL_KERNEL_OP_TYPE_REDUCESCATTERV}) {
            auto result = RunCalcTaskNum(opType, MIB + 1, 8, MIB);
            EXPECT_EQ(result.first, HCCL_SUCCESS) << opType;
            EXPECT_EQ(result.second, 2 * GE_PARALLEL_UT) << opType;
        }

        // ALLREDUCE: scratch 上限 = 1MB / 8 / 128 * 128 = 128KB。
        auto allReduce = RunCalcTaskNum(HCCL_KERNEL_OP_TYPE_ALLREDUCE, 131072 + 1, 8, MIB);
        EXPECT_EQ(allReduce.first, HCCL_SUCCESS);
        EXPECT_EQ(allReduce.second, 2 * GE_PARALLEL_UT);

        // transport 受限分支：上限 256MB。
        for (const auto& opType :
             {HCCL_KERNEL_OP_TYPE_BROADCAST, HCCL_KERNEL_OP_TYPE_ALLGATHER, HCCL_KERNEL_OP_TYPE_ALLGATHERV}) {
            auto result = RunCalcTaskNum(opType, UB_MAX_DATA_SIZE_UT + 1, 8, MIB);
            EXPECT_EQ(result.first, HCCL_SUCCESS) << opType;
            EXPECT_EQ(result.second, 2 * GE_PARALLEL_UT) << opType;
        }

        // ALLTOALL: 256MB / 8 ranks = 32MB 每轮，ccuTaskNum 不乘 GE_PARALLEL。
        auto allToAll = RunCalcTaskNum(HCCL_KERNEL_OP_TYPE_ALLTOALL, 33554432 + 1, 8, MIB);
        EXPECT_EQ(allToAll.first, HCCL_SUCCESS);
        EXPECT_EQ(allToAll.second, 2u);
    }

    TEST(GraphModeCalcTaskNumTest, InvalidDataTypePropagates)
    {
        auto result = RunCalcTaskNum(HCCL_KERNEL_OP_TYPE_ALLREDUCE, 1, 8, MIB, HCCL_DATA_TYPE_RESERVED);
        EXPECT_EQ(result.first, HCCL_E_NOT_SUPPORT);
        EXPECT_EQ(result.second, 0u);
    }

} // namespace
} // namespace ops_hccl
