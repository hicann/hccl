/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_aicpu_testcase_common.h"

class ST_REDUCE_SCATTER_AICPU_2_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(ST_REDUCE_SCATTER_AICPU_2_TEST, st_reduce_scatter_a5_aicpu_meshchunk_8rank_int16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};      // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200 * 1024 * 1024;                 // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_2_TEST, st_reduce_scatter_a5_aicpu_meshnhr_3x2rank_fp32_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};         // 三维数组指定超节点-Server-Device信息
    auto recvCount = 300 * 1024 * 1024;                // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_2_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x2rank_fp16_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};               // 三维数组指定超节点-Server-Device信息
    auto recvCount = 400 * 1024 * 1024;                // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_2_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x4rank_INT8_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    auto recvCount = 100 * 1024 * 1024;                  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;   // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_2_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x2x4rank_int8_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto recvCount = 1;                                // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_2_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x1x8rank_int8_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 8);
    auto recvCount = 1;                                // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_2_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x3rank_int16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}};        // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;                               // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_2_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x2rank_int32_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};                // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;                               // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_2_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x3rank_bf16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}};        // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;                               // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}
