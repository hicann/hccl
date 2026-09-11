/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "algo_desc.h"
#include "alg_selector.h"
#include "executor/ops_executor.h"

using namespace ops_hccl;

// 构造一个简单的 SEQUENCE 算法描述：1 层 Mesh AllGather 模板
static AlgoExecDesc MakeSimpleAlgoExecDesc()
{
    AlgoExecDesc desc;
    desc.execPolicy = HcclAlgExecPolicy::SEQUENCE;
    TemplateExecDesc tpl;
    tpl.templateDesc.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    tpl.templateDesc.algType = HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH;
    tpl.subCommIndex = 0;
    desc.children.emplace_back(tpl);
    desc.dataSplitRatio = {1, 1};
    return desc;
}

// ============ AlgSelector 注册/查询 ============

TEST(HcclAlgorithmTest, AlgSelectorRegisterAndGet)
{
    HcclAlgorithm desc;
    desc.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    desc.engineType = HcclAlgEngineType::COMM_ENGINE_AICPU;
    desc.algName = "SelTestAlgo";
    desc.algoExecDesc = MakeSimpleAlgoExecDesc();

    HcclResult ret = AlgSelector::Instance().Register("SelTestAlgo", desc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclAlgorithm got;
    bool found = AlgSelector::Instance().GetAlgorithm("SelTestAlgo", got);
    EXPECT_TRUE(found);
    EXPECT_EQ(got.algName, "SelTestAlgo");
    EXPECT_EQ(got.hcclCmdType, HcclCMDType::HCCL_CMD_ALLGATHER);
    EXPECT_EQ(got.engineType, HcclAlgEngineType::COMM_ENGINE_AICPU);
    EXPECT_EQ(got.algoExecDesc.children.size(), 1u);
}

TEST(HcclAlgorithmTest, AlgSelectorGetUnknown)
{
    HcclAlgorithm got;
    bool found = AlgSelector::Instance().GetAlgorithm("NoSuchAlg", got);
    EXPECT_FALSE(found);
}

TEST(HcclAlgorithmTest, AlgSelectorRegisterOverwrite)
{
    HcclAlgorithm desc1;
    desc1.algName = "OverwriteAlg";
    desc1.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    AlgSelector::Instance().Register("OverwriteAlg", desc1);

    HcclAlgorithm desc2;
    desc2.algName = "OverwriteAlg2";
    desc2.hcclCmdType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    AlgSelector::Instance().Register("OverwriteAlg", desc2);

    HcclAlgorithm got;
    ASSERT_TRUE(AlgSelector::Instance().GetAlgorithm("OverwriteAlg", got));
    EXPECT_EQ(got.hcclCmdType, HcclCMDType::HCCL_CMD_ALLREDUCE);
}

// 注册后返回的 HcclAlgorithm 是独立拷贝；topoMatch 为 shared_ptr 共享对象
TEST(HcclAlgorithmTest, AlgSelectorCopyIsolation)
{
    HcclAlgorithm desc;
    desc.algName = "IsolateAlg";
    desc.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    desc.algoExecDesc = MakeSimpleAlgoExecDesc();
    AlgSelector::Instance().Register("IsolateAlg", desc);

    // 修改本地拷贝不影响注册表中的算法
    desc.algName = "Changed";
    desc.algoExecDesc.children.clear();

    HcclAlgorithm got;
    ASSERT_TRUE(AlgSelector::Instance().GetAlgorithm("IsolateAlg", got));
    EXPECT_EQ(got.algName, "IsolateAlg");
    EXPECT_EQ(got.algoExecDesc.children.size(), 1u);
}

// ============ HcclAlgorithm::GetExecutor（经 ops_executor_stub 提供构造/析构） ============

TEST(HcclAlgorithmTest, GetExecutorReturnsNonNull)
{
    HcclAlgorithm desc;
    desc.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    desc.algoExecDesc = MakeSimpleAlgoExecDesc();

    OpParam param;
    param.root = 0;
    param.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    auto exec = desc.GetExecutor(param);
    ASSERT_NE(exec, nullptr);
}

// Dump() 应可调用且不崩溃（日志 stub 关闭）
TEST(HcclAlgorithmTest, DumpNoCrash)
{
    HcclAlgorithm desc;
    desc.hcclCmdType = HcclCMDType::HCCL_CMD_ALLGATHER;
    desc.algoExecDesc = MakeSimpleAlgoExecDesc();
    desc.Dump();
    SUCCEED();
}
