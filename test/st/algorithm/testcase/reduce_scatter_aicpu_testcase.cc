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

class ST_REDUCE_SCATTER_AICPU_TEST : public ::testing::Test {
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

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_3rank_int32_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}}};                     // 三维数组指定超节点-Server-Device信息
    auto recvCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_4rank_int8_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}};                 // 三维数组指定超节点-Server-Device信息
    auto recvCount = 1000;                             // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_6rank_fp32_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5}}};           // 三维数组指定超节点-Server-Device信息
    auto recvCount = 1000 * 1000;                      // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_8rank_int16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};      // 三维数组指定超节点-Server-Device信息
    auto recvCount = 1024;                              // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_2rank_fp16_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}}};                       // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;                              // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_8rank_bfp16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};      // 三维数组指定超节点-Server-Device信息
    auto recvCount = 500;                               // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_4rank_int8_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}};                 // 三维数组指定超节点-Server-Device信息
    auto recvCount = 400 * 1024 * 1024;                // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_3rank_int32_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}}};                     // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200 * 1024 * 1024;                 // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_6rank_fp32_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5}}};           // 三维数组指定超节点-Server-Device信息
    auto recvCount = 400 * 1024 * 1024;                // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_2rank_fp16_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 4, 5}}};              // 三维数组指定超节点-Server-Device信息
    auto recvCount = 1 * 1024 * 1024;                  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

// ============ HD（递归折半）算法用例组：仅显式 HCCL_ALGO 配置启用，强制指定算法后校验数据正确性 ============

class ST_REDUCE_SCATTER_AICPU_HD_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_ALGO");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// HD Sole 单机 8 rank：HCCL_ALGO='reducescatter:sole{level0=hdaicpureduce}'（rank 数为 2 的幂）
TEST_F(ST_REDUCE_SCATTER_AICPU_HD_TEST, st_reduce_scatter_a5_aicpu_hd_sole_8rank_int32_sum_test)
{
    setenv("HCCL_ALGO", "reducescatter:sole{level0=hdaicpureduce}", 1);
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};      // 单 Server 8 Device
    auto recvCount = 1024;                              // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

// HD Sole 单机 4 rank 小规模：HCCL_ALGO='reducescatter:sole{level0=hdaicpureduce}'
TEST_F(ST_REDUCE_SCATTER_AICPU_HD_TEST, st_reduce_scatter_a5_aicpu_hd_sole_4rank_int8_max_test)
{
    setenv("HCCL_ALGO", "reducescatter:sole{level0=hdaicpureduce}", 1);
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}};                 // 单 Server 4 Device
    auto recvCount = 512;                              // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

// HD 层级化 Parallel 双机（2 Server × 4 Device）：HCCL_ALGO='reducescatter:parallel{level0=mesh,level1=hd}'
// （机器数为 2 的幂，level1 组大小 N1=2 单步交换）
TEST_F(ST_REDUCE_SCATTER_AICPU_HD_TEST, st_reduce_scatter_a5_aicpu_hd_parallel_2x4rank_int32_sum_test)
{
    setenv("HCCL_ALGO", "reducescatter:parallel{level0=mesh,level1=hd}", 1);
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};    // 双 Server 各 4 Device
    auto recvCount = 1024;                              // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

// 否定型配置回归：HCCL_ALGO='reducescatter:sole{level0=not(nhr)}' 下 HD 不进入候选（守卫不视为显式启用），
// 回退默认算法选路，数据仍须正确
TEST_F(ST_REDUCE_SCATTER_AICPU_HD_TEST, st_reduce_scatter_a5_aicpu_hd_negated_fallback_8rank_int32_sum_test)
{
    setenv("HCCL_ALGO", "reducescatter:sole{level0=not(nhr)}", 1);
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};      // 单 Server 8 Device
    auto recvCount = 1024;                              // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}
