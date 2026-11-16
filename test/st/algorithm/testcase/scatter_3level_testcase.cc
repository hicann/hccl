/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_testcase_common.h"

class ST_SCATTER_3LEVEL_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
    }
    void TearDown() override
    {
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(ST_SCATTER_3LEVEL_TEST, test_aicpu_scatter_mesh_1d_success_2x8x8_root0_fp32_small_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    RunScatterTest(0, topoMeta, 200, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_3x4x4_fp32_sum_repeatnum_gt1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 4, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_2level_backward_compat_meshnhr)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_2level_backward_compat_meshnhr_2)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 4);
    auto recvCount = 400;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_2x4x4_int32_max_different_scale)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 500;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_2x2x2_fp32_sum_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto recvCount = 500 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_2x2x8_fp32_sum_small_cluster_recv200_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_4x2x4_int32_sum_repeatnum4)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_2x2x4_bfp16_max_dtype)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto recvCount = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_3x2x8_fp32_sum_level2_3cluster)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_2x3x4_int32_sum_asymmetric_all)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

// --- Degenerate Level (dimension=1) edge cases ---

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_3x1x8_fp32_min_l1_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_2x1x4_int8_sum_l1_degenerate_recv8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    auto recvCount = 8 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_4x1x1_fp32_sum_double_degenerate_recv16_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 1, 1);
    auto recvCount = 16 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_2x3x4_fp32_min_recv4_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 4 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_2x4x4_int16_max_recv64k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 64 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

TEST_F(ST_SCATTER_3LEVEL_TEST, st_scatter_3level_2x4x1_fp32_sum_l0_degenerate_recv128k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    auto recvCount = 128 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

// ===== Scatter RHD（AICPU 三级，preferred-only：HCCL_ALGO 显式配置选路）=====
// 三级链式形态 scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}；
// 维度矩阵覆盖 2^N / 非 2^N（d1=5 r=1、d2=6 r=2）/ d1=1 退化（skipLevel1 分支）/ 多 loop 大数据。
class ST_SCATTER_3LEVEL_RHD_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
    }
    void TearDown() override
    {
        unsetenv("HCCL_ALGO");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 三级全 2^N：2x4x4，root0（主路径）
TEST_F(ST_SCATTER_3LEVEL_RHD_TEST, st_scatter_rhd_3level_2x4x4_root0_int32_small_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

// 三级跨 pod root：2x4x4，root16（superPod 1 的 server0 device0）——级内 root 换算
// （L1 仅 root pod 执行、L2 逐组 SetRoot 到 root pod 侧成员）与跨 pod 搬运路径的回归覆盖
TEST_F(ST_SCATTER_3LEVEL_RHD_TEST, st_scatter_rhd_3level_2x4x4_root16_int32_small_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTest(16, topoMeta, recvCount, dataType);
}

// 三级非 2^N（d1=5，r=1）：2x5x4，root0
TEST_F(ST_SCATTER_3LEVEL_RHD_TEST, st_scatter_rhd_3level_2x5x4_root0_fp32_small_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 5, 4);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

// 三级非 2^N（d2=6，r=2）：2x4x6，root1（root 压 d2 边缘）
TEST_F(ST_SCATTER_3LEVEL_RHD_TEST, st_scatter_rhd_3level_2x4x6_root1_int16_small_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 6);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;

    RunScatterTest(1, topoMeta, recvCount, dataType);
}

// 三级 d1=1 退化（skipLevel1 分支必须覆盖）：2x1x8，root0
TEST_F(ST_SCATTER_3LEVEL_RHD_TEST, st_scatter_rhd_3level_2x1x8_root0_fp32_small_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 8);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

// 三级多 loop 大数据：2x2x2，root3（loop 尾段 currDataCount 变小，Scatter 尾块实际口径）
TEST_F(ST_SCATTER_3LEVEL_RHD_TEST, st_scatter_rhd_3level_2x2x2_root3_fp32_big_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    auto recvCount = 500 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunScatterTest(3, topoMeta, recvCount, dataType);
}

// 三级非 2^N（d1=6，r=2，σ 非平凡）× d2=4（L1 repeatNum>1）：首次让"多 block σ 置换"与
// "非平凡 σ"同时出现——RearrangeToSigma 逐 repeat 置换的回归用例（block 1..d2-1 不置换时，
// 错位 slice 由 CheckScatter 数据内容校验直接指认）
TEST_F(ST_SCATTER_3LEVEL_RHD_TEST, st_scatter_rhd_3level_2x6x4_root0_int32_small_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 6, 4);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTest(0, topoMeta, recvCount, dataType);
}

// 回归 A：input 不可变——σ 置换暂存不得写用户 sendbuf（暂存若错借 inputPtr 首份，
// 任务图上 dst 落 INPUT 池；三级非 2^N 时含非全局 root）
TEST_F(ST_SCATTER_3LEVEL_RHD_TEST, st_scatter_rhd_3level_2x6x4_root0_int32_small_data_input_immutable)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 6, 4);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTestInputImmutable(0, topoMeta, recvCount, dataType);
}

// 回归 B：负向 null-input——非 root 传 nullptr sendbuf（Scatter API：sendbuf 仅
// root 有效）。若非全局 root 的 L1 root 借 param.inputPtr 做 σ 暂存，CHK_PTR_NULL
// 空指针直接失败；当前实现全链路不依赖非 root 输入指针
TEST_F(ST_SCATTER_3LEVEL_RHD_TEST, st_scatter_rhd_3level_2x6x4_root0_int32_small_data_null_input_nonroot)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 6, 4);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunScatterTestNullInputNonRoot(0, topoMeta, recvCount, dataType);
}
