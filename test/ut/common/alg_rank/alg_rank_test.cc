/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "ins_temp_all_reduce_aicpu_reduce_nhr.h"
#include "reduce_aicpu_reduce_nhr.h"
#include "reduce_mesh_1D.h"
#include "reduce_nhr.h"

using ops_hccl::InsTempAllReduceAicpuReduceNHR;
using ops_hccl::ReduceAicpuReduceNHR;
using ops_hccl::ReduceMesh1D;
using ops_hccl::ReduceNHR;

namespace {
constexpr u32 INVALID_ALG_RANK = UINT32_MAX;
} // namespace

TEST(InsTempAllReduceAicpuReduceNHRGetAlgRankTest, FindRankSuccess)
{
    InsTempAllReduceAicpuReduceNHR op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {{0, 1, 2}};

    ASSERT_EQ(op.GetAlgRank(1, algRank), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(algRank, 1u);
}

TEST(InsTempAllReduceAicpuReduceNHRGetAlgRankTest, MissingRankReturnsInternal)
{
    InsTempAllReduceAicpuReduceNHR op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {{0, 1, 2}};

    ASSERT_EQ(op.GetAlgRank(3, algRank), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(algRank, INVALID_ALG_RANK);
}

TEST(InsTempAllReduceAicpuReduceNHRGetAlgRankTest, EmptySubCommRanksReturnsInternal)
{
    InsTempAllReduceAicpuReduceNHR op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {};

    ASSERT_EQ(op.GetAlgRank(0, algRank), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(algRank, INVALID_ALG_RANK);
}

TEST(InsTempAllReduceAicpuReduceNHRGetAlgRankTest, EmptyFirstSubCommReturnsInternal)
{
    InsTempAllReduceAicpuReduceNHR op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {{}};

    ASSERT_EQ(op.GetAlgRank(0, algRank), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(algRank, INVALID_ALG_RANK);
}

TEST(ReduceAicpuReduceNHRGetAlgRankTest, FindRankSuccess)
{
    ReduceAicpuReduceNHR op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {{3, 5, 7}};

    ASSERT_EQ(op.GetAlgRank(5, algRank), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(algRank, 1u);
}

TEST(ReduceAicpuReduceNHRGetAlgRankTest, MissingRankReturnsInternal)
{
    ReduceAicpuReduceNHR op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {{3, 5, 7}};

    ASSERT_EQ(op.GetAlgRank(4, algRank), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(algRank, INVALID_ALG_RANK);
}

TEST(ReduceAicpuReduceNHRGetAlgRankTest, EmptySubCommRanksReturnsInternal)
{
    ReduceAicpuReduceNHR op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {};

    ASSERT_EQ(op.GetAlgRank(3, algRank), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(algRank, INVALID_ALG_RANK);
}

TEST(ReduceAicpuReduceNHRGetAlgRankTest, EmptyFirstSubCommReturnsInternal)
{
    ReduceAicpuReduceNHR op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {{}};

    ASSERT_EQ(op.GetAlgRank(3, algRank), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(algRank, INVALID_ALG_RANK);
}

TEST(ReduceMesh1DGetAlgRankTest, FindRankSuccess)
{
    ReduceMesh1D op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {{8, 9}};

    ASSERT_EQ(op.GetAlgRank(9, algRank), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(algRank, 1u);
}

TEST(ReduceMesh1DGetAlgRankTest, MissingRankReturnsInternal)
{
    ReduceMesh1D op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {{8, 9}};

    ASSERT_EQ(op.GetAlgRank(10, algRank), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(algRank, INVALID_ALG_RANK);
}

TEST(ReduceMesh1DGetAlgRankTest, EmptySubCommRanksReturnsInternal)
{
    ReduceMesh1D op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {};

    ASSERT_EQ(op.GetAlgRank(8, algRank), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(algRank, INVALID_ALG_RANK);
}

TEST(ReduceMesh1DGetAlgRankTest, EmptyFirstSubCommReturnsInternal)
{
    ReduceMesh1D op;
    u32 algRank = INVALID_ALG_RANK;
    op.subCommRanks_ = {{}};

    ASSERT_EQ(op.GetAlgRank(8, algRank), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(algRank, INVALID_ALG_RANK);
}

TEST(ReduceNHRGetMyAlgRankTest, FindRankSuccess)
{
    ReduceNHR op;
    op.subCommRanks_ = {{20, 21, 22}};
    op.myRank_ = 21;
    op.templateRankSize_ = 3;

    ASSERT_EQ(op.getMyAlgRank(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(op.myIdx_, 1u);
}

TEST(ReduceNHRGetMyAlgRankTest, MissingRankReturnsInternal)
{
    ReduceNHR op;
    op.subCommRanks_ = {{20, 21, 22}};
    op.myRank_ = 99;

    ASSERT_EQ(op.getMyAlgRank(), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(op.myIdx_, INVALID_ALG_RANK);
}

TEST(ReduceNHRGetMyAlgRankTest, EmptySubCommRanksReturnsInternal)
{
    ReduceNHR op;
    op.myRank_ = 20;
    op.subCommRanks_ = {};

    ASSERT_EQ(op.getMyAlgRank(), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(op.myIdx_, INVALID_ALG_RANK);
}

TEST(ReduceNHRGetMyAlgRankTest, EmptyFirstSubCommReturnsInternal)
{
    ReduceNHR op;
    op.myRank_ = 20;
    op.subCommRanks_ = {{}};

    ASSERT_EQ(op.getMyAlgRank(), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(op.myIdx_, INVALID_ALG_RANK);
}
