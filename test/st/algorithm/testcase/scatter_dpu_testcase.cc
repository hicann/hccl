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

// ===== Scatter RHD（DPU 二级/三级，preferred-only：HCCL_ALGO 显式配置选路）=====
// host-DPU 形态（ENABLE_HOSTDPU_FOR_LLT）下二级链式形态选中 DpuScatterSequenceMeshConcurRHD
// （老路径：单次 DPURunInfo 共享内存请求 + DpuBatchTransfer 合并段）。
class ST_SCATTER_RHD_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
    }
    void TearDown() override
    {
        unsetenv("HCCL_ALGO");
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// DPU 二级 2^N：2x2，root0（主路径）
TEST_F(ST_SCATTER_RHD_DPU_TEST, st_scatter_rhd_dpu_2level_2x2_root0_int32_small_data)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_INT32);
}

// DPU 二级非 2^N（r=1，d1=3）：3x2，root2（root 压边缘）
TEST_F(ST_SCATTER_RHD_DPU_TEST, st_scatter_rhd_dpu_2level_3x2_root2_fp32_small_data)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}};
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunScatterTest(2, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_FP32);
}

// DPU 二级非 2^N（r=2，d1=6）：6x2，root0
TEST_F(ST_SCATTER_RHD_DPU_TEST, st_scatter_rhd_dpu_2level_6x2_root0_fp16_small_data)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}}};
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_FP16);
}

// σ 暂存借用回归（DPU 侧）：input 不可变——若 σ 暂存错误借用全局 root 的 param.inputPtr
// 首份（2 级 L1 root=全局 root，指针有效但脏写用户 sendbuf 违反契约）
TEST_F(ST_SCATTER_RHD_DPU_TEST, st_scatter_rhd_dpu_2level_6x2_root0_fp16_small_data_input_immutable)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}}};
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunScatterTestInputImmutable(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_FP16);
}

// DPU 三级（2x6x2，d1=6 非 2^N σ、root=3 非 0 槽位）：root 经 tempAlgParams 序列化传
// device 侧实例——若 GenInterTemplateParams1/2 漏填该字段，device 侧恒用默认 0，
// 跨 pod 的 rootAlgRank 与 σ 置换全部错位（本用例为该路径首个 ST 覆盖）
TEST_F(ST_SCATTER_RHD_DPU_TEST, st_scatter_rhd_dpu_3level_2x6x2_root3_int32_small_data)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 6, 2);
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    RunScatterTest(3, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_INT32);
}

// DPU 二级大 p 多 loop：2x8，root6（大数据，loop 尾段）
TEST_F(ST_SCATTER_RHD_DPU_TEST, st_scatter_rhd_dpu_2level_2x8_root6_int8_big_data)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunScatterTest(6, topoMeta, 400 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_INT8);
}

// DPU 二级非 2^N（r=1，d1=9）：9x2，root0——DPU 模板建链同病（HighestHostRoce 内嵌同一
// ±2^k 连接集），修复的 DPU 侧回归用例（症状同设备侧：RunRhdSteps 找不到对端链路）
TEST_F(ST_SCATTER_RHD_DPU_TEST, st_scatter_rhd_dpu_2level_9x2_root0_int32_small_data)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}}};
    setenv("HCCL_ALGO", "scatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_INT32);
}
