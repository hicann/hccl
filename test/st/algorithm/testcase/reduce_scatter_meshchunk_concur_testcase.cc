/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_aicpu_testcase_common.h"

// ============================================================================
// MeshChunkConcurNHR 定向用例: 显式 HCCL_ALGO 钉住
// "reducescatter:sequence{level0=meshchunkconcur,level1=nhr}" 强制选中
// AicpuReduceScatterSequenceMeshChunkConcurNHR(不依赖 cost 竞争), 校验数据正确性。
// 拓扑要求两级: level0 机内 mesh+CLOS 并发(需 ≥2 Server 供 CLOS 绕路上行),
// level1 NHR 跨 Server 归约。
// ============================================================================

class ST_REDUCE_SCATTER_AICPU_CONCUR_TEST : public ::testing::Test {
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

namespace {
constexpr const char* CONCUR_NHR_HCCL_ALGO = "reducescatter:sequence{level0=meshchunkconcur,level1=nhr}";
} // namespace

// 双机 2×8(fp32 SUM 主路径, 对齐真机 8p×2 形态): level0 机内 MeshChunk 轮转 + mesh/CLOS 双链路
// 带宽加权分流, level1 NHR 跨机
TEST_F(ST_REDUCE_SCATTER_AICPU_CONCUR_TEST, st_reduce_scatter_a5_aicpu_meshchunkconcur_2s8r_fp32_sum_test)
{
    setenv("HCCL_ALGO", CONCUR_NHR_HCCL_ALGO, 1);
    // 仿真模型初始化: 1 超节点 × 2 Server × 8 Device = 16 rank
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto recvCount = 256 * 1024;                       // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;     // 规约算子
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

// 双机 2×8(int8 MIN): dtype 与规约算子泛化
TEST_F(ST_REDUCE_SCATTER_AICPU_CONCUR_TEST, st_reduce_scatter_a5_aicpu_meshchunkconcur_2s8r_int8_min_test)
{
    setenv("HCCL_ALGO", CONCUR_NHR_HCCL_ALGO, 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto recvCount = 512 * 1024;                       // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;     // 规约算子
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

// 双机 2×4(8 rank, fp32 SUM 非对齐尾巴 1023×1023): 切分/偏移按元素粒度路径覆盖
TEST_F(ST_REDUCE_SCATTER_AICPU_CONCUR_TEST, st_reduce_scatter_a5_aicpu_meshchunkconcur_2s4r_fp32_tail_test)
{
    setenv("HCCL_ALGO", CONCUR_NHR_HCCL_ALGO, 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 4);
    auto recvCount = 1023 * 1023;                      // 非对齐数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;     // 规约算子
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}
