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

class ST_REDUCE_TEST :
    public ::testing::TestWithParam<std::tuple<TopoMeta, u64, HcclDataType, HcclReduceOp, uint32_t>> {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }

    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_BUFFSIZE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }

    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_P(ST_REDUCE_TEST, st_reduce_aicpu_test)
{
    auto params = GetParam();
    const auto& topoMeta = std::get<0>(params);
    u64 recvCount = std::get<1>(params);
    HcclDataType dataType = std::get<2>(params);
    HcclReduceOp reduceOp = std::get<3>(params);
    uint32_t root = std::get<4>(params);

    RunReduceTest(topoMeta, recvCount, dataType, reduceOp, root);
}

// 参数化实例化
INSTANTIATE_TEST_SUITE_P(
    ReduceVariants, ST_REDUCE_TEST,
    ::testing::Values(
        // 每个 tuple 表示一组测试参数
        // 1D Mesh
        std::make_tuple(GenerateMeshTopoMeta(2), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(3), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(5), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(6), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(7), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(8), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 1 << 20, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 1),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 2),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 3),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_MAX, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_MIN, 0),
        std::make_tuple(GenerateMeshTopoMeta(2), (1 << 24) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2), 220 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2), (1 << 30) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2), 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        // 1DMeshNHR (大数据量用例)
        std::make_tuple(GenerateMeshTopoMeta(3, 1, 3), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(3, 1, 3), 100, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1023, HCCL_DATA_TYPE_INT16, HCCL_REDUCE_SUM, 0)
        // 2DMeshNHR
        ));

// RHD（递归折半倍增）显式配置二级形态（对齐 RS 侧 rhd_001/002 口径）：
// reduce:sequence{level0=meshconcur,level1=rhd}。
// host-DPU 形态（RunReduceDPUCase 统一置位 ENABLE_HOSTDPU_FOR_LLT）下命中
// DpuReduceSequenceMeshConcurRHD，数据正确性由 CheckReduce 语义校验断言；算法命中
// 证据由 VM checker（rdu 系列矩阵）承载，本组用例提供 CI 级数据面回归。
class ST_REDUCE_RHD_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("ENABLE_HOSTDPU_L2_ONLY");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ALGO"); // 用例体内已自清理，此处防御性兜底
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// rhd_001：d1=6 非 2 次幂——层内前/后处理生效（RHD 区别于 HD 的目标场景）。
TEST_F(ST_REDUCE_RHD_TEST, test_host_dpu_reduce_rhd_001)
{
    setenv("HCCL_ALGO", "reduce:sequence{level0=meshconcur,level1=rhd}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 6, 2);
    RunReduceDPUCase(topoMeta, 256 * 1024, HCCL_DATA_TYPE_INT32, 4, HCCL_REDUCE_SUM, 0);
    unsetenv("HCCL_ALGO");
}

// rhd_002：d1=4 2 次幂——纯主体路径（PRE/POST 空步骤分支）。
TEST_F(ST_REDUCE_RHD_TEST, test_host_dpu_reduce_rhd_002)
{
    setenv("HCCL_ALGO", "reduce:sequence{level0=meshconcur,level1=rhd}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 2);
    RunReduceDPUCase(topoMeta, 256 * 1024, HCCL_DATA_TYPE_INT32, 4, HCCL_REDUCE_SUM, 0);
    unsetenv("HCCL_ALGO");
}

// 设备形态（AICPU 引擎，RunReduceTest 置位 AI_CPU）：命中 AicpuReduceSequenceMeshConcurRHD。
// rhd_dev_001：d1=6 非 2 次幂——PRE/POST 前后处理 + σ 布局生效。
TEST_F(ST_REDUCE_RHD_TEST, test_device_reduce_rhd_dev_001)
{
    setenv("HCCL_ALGO", "reduce:sequence{level0=meshconcur,level1=rhd}", 1);
    RunReduceTest(TopoMeta{{{0}, {0}, {0}, {0}, {0}, {0}}}, 256 * 1024, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0);
    unsetenv("HCCL_ALGO");
}

// rhd_dev_002：d1=4 2 次幂——纯主体路径。
TEST_F(ST_REDUCE_RHD_TEST, test_device_reduce_rhd_dev_002)
{
    setenv("HCCL_ALGO", "reduce:sequence{level0=meshconcur,level1=rhd}", 1);
    RunReduceTest(TopoMeta{{{0}, {0}, {0}, {0}}}, 256 * 1024, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0);
    unsetenv("HCCL_ALGO");
}

// 三级 RHD（AICPU 引擎）：reduce:sequence{level0=meshconcur,level1=rhd,level2=rhd}，命中
// AicpuReduceSequenceMeshConcurRHDRHD。
// rhd_3l_dev_001：2 超节点 × 2 server × 2 rank，L1/L2 域均 2 次幂。
TEST_F(ST_REDUCE_RHD_TEST, test_device_reduce_rhd_3l_dev_001)
{
    setenv("HCCL_ALGO", "reduce:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    RunReduceTest(topoMeta, 256 * 1024, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0);
    unsetenv("HCCL_ALGO");
}

// rhd_3l_dev_002：3 超节点 × 2 server × 2 rank，L2 域 p=3 非 2 次幂（PRE/POST 生效）。
TEST_F(ST_REDUCE_RHD_TEST, test_device_reduce_rhd_3l_dev_002)
{
    setenv("HCCL_ALGO", "reduce:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 2);
    RunReduceTest(topoMeta, 256 * 1024, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0);
    unsetenv("HCCL_ALGO");
}

// 三级 RHD（host-DPU 引擎）：ENABLE_HOSTDPU_L2_ONLY 构造 L2=HOST、内层 DEVICE 的 l1dev
// 形态，满足 topoPrecondCheck 的 L1 设备侧 CLOS 前提，命中 DpuReduceSequenceMeshConcurRHDRHD。
// rhd_3l_dpu_001：2 超节点 × 2 server × 2 rank，L1/L2 域均 2 次幂。
TEST_F(ST_REDUCE_RHD_TEST, test_host_dpu_reduce_rhd_3l_dpu_001)
{
    setenv("HCCL_ALGO", "reduce:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    setenv("ENABLE_HOSTDPU_L2_ONLY", "1", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    RunReduceDPUCase(topoMeta, 256 * 1024, HCCL_DATA_TYPE_INT32, 4, HCCL_REDUCE_SUM, 0);
    unsetenv("HCCL_ALGO");
    unsetenv("ENABLE_HOSTDPU_L2_ONLY");
}

// rhd_3l_dpu_002：3 超节点 × 2 server × 2 rank，L2 域 p=3 非 2 次幂。
TEST_F(ST_REDUCE_RHD_TEST, test_host_dpu_reduce_rhd_3l_dpu_002)
{
    setenv("HCCL_ALGO", "reduce:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    setenv("ENABLE_HOSTDPU_L2_ONLY", "1", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 2);
    RunReduceDPUCase(topoMeta, 256 * 1024, HCCL_DATA_TYPE_INT32, 4, HCCL_REDUCE_SUM, 0);
    unsetenv("HCCL_ALGO");
    unsetenv("ENABLE_HOSTDPU_L2_ONLY");
}
