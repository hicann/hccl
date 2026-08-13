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

class ST_SCATTER_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        // 设置展开模式为HOST_TS
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(ST_SCATTER_DPU_TEST, test_aicpu_scatter_mesh1dnhr_success_2x2_root0_fp16_small_data)
{
    TopoMeta topoMeta{{{0}, {0}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_FP16);
}

TEST_F(ST_SCATTER_DPU_TEST, test_aicpu_scatter_mesh1dnhr_success_2x4_root0_fp32_small_data)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}};
    RunScatterTest(1, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_SCATTER_DPU_TEST, test_aicpu_scatter_mesh1dnhr_success_2x2_root0_int32_small_data)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(ST_SCATTER_DPU_TEST, test_aicpu_scatter_mesh1dnhr_success_3x2_root0_uint64_small_data)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_UINT64);
}

TEST_F(ST_SCATTER_DPU_TEST, test_aicpu_scatter_mesh1dnhr_success_3x4_root0_fp32_big_data)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};
    RunScatterTest(3, topoMeta, 400 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_SCATTER_DPU_TEST, test_aicpu_scatter_mesh1dnhr_success_2x8_root0_int8_big_data)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    RunScatterTest(6, topoMeta, 400 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_INT8);
}
