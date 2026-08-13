/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_testcase_common.h"

class ST_REDUCE_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }

    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_BUFFSIZE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
    }

    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(ST_REDUCE_DPU_TEST, host_dpu_opbase_reduce_1_fp32_sum)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    u64 dataCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    u32 root = 1;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp, root);
}

TEST_F(ST_REDUCE_DPU_TEST, host_dpu_opbase_reduce_301M_fp32_sum)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    u64 dataCount = 301 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    u32 root = 0;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp, root);
}

TEST_F(ST_REDUCE_DPU_TEST, host_dpu_opbase_reduce_1_fp32_sum_4x4)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};
    u64 dataCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    u32 root = 0;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp, root);
}

TEST_F(ST_REDUCE_DPU_TEST, host_dpu_opbase_reduce_301M_fp32_sum_4x4)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};
    u64 dataCount = 301 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    u32 root = 0;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp, root);
}

TEST_F(ST_REDUCE_DPU_TEST, host_dpu_opbase_reduce_1_fp32_sum_1x8)
{
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};
    u64 dataCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    u32 root = 0;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp, root);
}

TEST_F(ST_REDUCE_DPU_TEST, host_dpu_opbase_reduce_301M_fp32_sum_1x8)
{
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};
    u64 dataCount = 301 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    u32 root = 0;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp, root);
}

// asymmetric topology
TEST_F(ST_REDUCE_DPU_TEST, host_dpu_opbase_reduce_asymmetric_100_int16_max)
{
    TopoMeta topoMeta{
        {{0, 1},
         {0, 1, 2},
         {0, 1, 2, 3},
         {0, 1, 2, 3, 4},
         {0, 1, 2, 3, 4, 5},
         {0, 1, 2, 3, 4, 5, 6},
         {0, 1, 2, 3, 4, 5, 6, 7},
         {0}}};
    u64 dataCount = 100;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    u32 dataTypeSize = 2;
    u32 root = 0;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp, root);
}
