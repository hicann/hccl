/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* AlgoNameMapper 真实现 UT——覆盖 Lookup 全映射/miss/Init 幂等/晚注册检测。
 * 测试策略：AlgoNameMapper::Global() 是进程级单例（call_once），首次 Init 即冻结。
 * 用例按 gtest 文件序执行：首个用例做综合 Init（含合法/null/unknown 三类条目），
 * 后续用例依赖该状态做 Lookup/幂等/晚注册验证——与真实调用序列同构。
 * 生产代码 algo_name_mapper.cc 由 CMakeLists 直接编入（非桩）。 */

#include <gtest/gtest.h>
#include <cstring>

#include "algo_name_mapper.h"
#include "cost_model.h"

using namespace ops_hccl;

static AlgElement g_elements[6] = {};

static void FillTestAlgos(AllAlgos& allAlgos, int count)
{
    allAlgos.algElements = g_elements;
    allAlgos.count = count;
    allAlgos.capacity = 6;
}

/* 综合 Init：2 条合法 + null algName + unknown opType + 非法名——一次覆盖 5 条路径 */
TEST(AlgoNameMapperTest, InitComprehensiveAndLookupHit)
{
    g_elements[0].algName = "AicpuAllReduceSoleNHR";
    g_elements[0].executorName = "sole";
    g_elements[0].templateName = nullptr;
    g_elements[0].templateNum = 0;
    g_elements[0].opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    g_elements[1].algName = "DpuAllGatherParallelMeshOneShot";
    g_elements[1].executorName = "parallel";
    g_elements[1].templateName = nullptr;
    g_elements[1].templateNum = 0;
    g_elements[1].opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    g_elements[2].algName = nullptr; /* 空名条目 → skip + WARN */
    g_elements[2].opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    g_elements[3].algName = "AicpuUnknownOpTypeSoleNHR";
    g_elements[3].opType = HcclCMDType::HCCL_CMD_INVALID; /* unknown opType → skip + WARN */

    g_elements[4].algName = "GarbageName"; /* 不含 opType Pascal 前缀 → LookupByPrefix miss + WARN */
    g_elements[4].opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    AllAlgos allAlgos = {};
    FillTestAlgos(allAlgos, 5);
    AlgoNameMapper::Global()->Init(allAlgos);

    /* 验证两条合法映射的 Lookup 命中 */
    AlgoDims dims;
    EXPECT_TRUE(AlgoNameMapper::Global()->Lookup("AicpuAllReduceSoleNHR", dims));
    EXPECT_STRNE(dims.engineUser, "");
    EXPECT_STRNE(dims.executorUser, "");

    EXPECT_TRUE(AlgoNameMapper::Global()->Lookup("DpuAllGatherParallelMeshOneShot", dims));
    EXPECT_STRNE(dims.engineUser, "");
}

TEST(AlgoNameMapperTest, LookupMiss)
{
    AlgoDims dims;
    EXPECT_FALSE(AlgoNameMapper::Global()->Lookup("NonExistentAlgorithm", dims));
    EXPECT_STREQ(dims.engineUser, "");
    EXPECT_STREQ(dims.executorUser, "");
    EXPECT_STREQ(dims.templateUser, "");
}

TEST(AlgoNameMapperTest, LookupNullName)
{
    AlgoDims dims;
    EXPECT_FALSE(AlgoNameMapper::Global()->Lookup(nullptr, dims));
    EXPECT_STREQ(dims.engineUser, "");
}

TEST(AlgoNameMapperTest, InitIdempotent)
{
    /* call_once 已在首个用例触发，此处的不同数据集应被忽略——旧映射仍生效 */
    g_elements[5].algName = "CcumsAllReduceConcurNHR";
    g_elements[5].executorName = "concur";
    g_elements[5].opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    AllAlgos secondSet = {};
    FillTestAlgos(secondSet, 6); /* count=6（含新条目） */

    AlgoNameMapper::Global()->Init(secondSet);

    /* 旧映射仍命中（call_once 冻结的是首个数据集） */
    AlgoDims dims;
    EXPECT_TRUE(AlgoNameMapper::Global()->Lookup("AicpuAllReduceSoleNHR", dims));

    /* 新条目不应被录入（晚注册 + frozenCount 告警） */
    EXPECT_FALSE(AlgoNameMapper::Global()->Lookup("CcumsAllReduceConcurNHR", dims));
}
