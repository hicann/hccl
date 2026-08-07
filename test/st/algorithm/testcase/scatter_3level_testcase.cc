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
