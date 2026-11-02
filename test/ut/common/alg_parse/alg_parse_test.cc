/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>
#include "alg_parse.h"

using namespace ops_hccl;

HcclResult HcclGetDeviceType(HcclDevType& deviceType)
{
    deviceType = HcclDevType::DEV_TYPE_910_93;
    return HCCL_SUCCESS;
}

class HcclAlgoParserTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// 测试正确用例1：多段并列配置，使用多种 ALGO_TYPES 值
TEST_F(HcclAlgoParserTest, ParseCorrectCase1)
{
    HcclAlgoParser parser;
    std::string input = "allreduce:sequence{mesh2die,nhrmultilink};sole{nhr};allgather:sequence{mesh,nhr,nhr};concur{"
                        "mesh,nhrmultilink}; not(meshoneshot)";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 5u);

    // 检查第一段：allreduce:sequence{mesh2die,nhrmultilink}
    EXPECT_EQ(parser.executorList[0].opType, "allreduce");
    EXPECT_EQ(parser.executorList[0].executorType, "sequence");
    EXPECT_EQ(parser.executorList[0].algoList.size(), 2u);
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "mesh2die");
    EXPECT_EQ(parser.executorList[0].algoList[1].algoType, "nhrmultilink");
    EXPECT_TRUE(parser.executorList[0].enable);

    // 检查第二段：sole{nhr}
    EXPECT_EQ(parser.executorList[1].opType, "");
    EXPECT_EQ(parser.executorList[1].executorType, "sole");
    EXPECT_EQ(parser.executorList[1].algoList.size(), 1u);
    EXPECT_EQ(parser.executorList[1].algoList[0].algoType, "nhr");
    EXPECT_TRUE(parser.executorList[1].enable);

    // 检查第五段：not(meshoneshot) → not(sole{meshoneshot})
    EXPECT_EQ(parser.executorList[4].opType, "");
    EXPECT_EQ(parser.executorList[4].executorType, "sole");
    EXPECT_EQ(parser.executorList[4].algoList[0].algoType, "meshoneshot");
    EXPECT_FALSE(parser.executorList[4].enable);
}

// 测试正确用例2：包含下划线转驼峰
TEST_F(HcclAlgoParserTest, ParseCorrectCase2)
{
    HcclAlgoParser parser;
    std::string input
        = "not(parallel{mesh,nhr});allgather:sequence{mesh,nhr};allreduce:not(sequence{mesh_multi_link,nhr})";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 3u);

    // 检查第三段：allreduce:not(sequence{mesh_multi_link,nhr})
    EXPECT_EQ(parser.executorList[2].opType, "allreduce");
    EXPECT_EQ(parser.executorList[2].algoList[0].algoType, "meshmultilink");
    EXPECT_FALSE(parser.executorList[2].enable);
}

// 测试正确用例3：显式level和取非，使用 mesh2die / nhrmultilink / meshchunktwoshot
TEST_F(HcclAlgoParserTest, ParseCorrectCase3)
{
    HcclAlgoParser parser;
    std::string input
        = "allgather:not(sequence{level0=mesh2die,level1=nhrmultilink}); concur{not(mesh),meshchunktwoshot};mesh_chunk";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 3u);

    // 检查第一段：allgather:not(sequence{level0=mesh2die,level1=nhrmultilink})
    EXPECT_EQ(parser.executorList[0].opType, "allgather");
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "mesh2die");
    EXPECT_EQ(parser.executorList[0].algoList[1].algoType, "nhrmultilink");
    EXPECT_FALSE(parser.executorList[0].enable);

    // 检查第二段：concur{not(mesh),meshchunktwoshot}
    EXPECT_EQ(parser.executorList[1].executorType, "concur");
    EXPECT_EQ(parser.executorList[1].algoList[0].algoType, "mesh");
    EXPECT_FALSE(parser.executorList[1].algoList[0].enable);
    EXPECT_EQ(parser.executorList[1].algoList[1].algoType, "meshchunktwoshot");
    EXPECT_TRUE(parser.executorList[1].algoList[1].enable);

    // 检查第三段：mesh_chunk → sole{meshchunk}
    EXPECT_EQ(parser.executorList[2].executorType, "sole");
    EXPECT_EQ(parser.executorList[2].algoList[0].algoType, "meshchunk");
}

// 测试正确用例4：下划线转驼峰，algoList 使用 meshtwoshot
TEST_F(HcclAlgoParserTest, ParseCorrectCase4)
{
    HcclAlgoParser parser;
    std::string input = "mesh_concur;sequence{meshtwoshot,not(nhrmultilink)};allgather:sequence{mesh,nhr}";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 3u);

    // 检查第一段：mesh_concur → sole{meshconcur}
    EXPECT_EQ(parser.executorList[0].executorType, "sole");
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "meshconcur");

    // 检查第二段：sequence{meshtwoshot,not(nhrmultilink)}
    EXPECT_EQ(parser.executorList[1].executorType, "sequence");
    EXPECT_EQ(parser.executorList[1].algoList[0].algoType, "meshtwoshot");
    EXPECT_TRUE(parser.executorList[1].algoList[0].enable);
    EXPECT_EQ(parser.executorList[1].algoList[1].algoType, "nhrmultilink");
    EXPECT_FALSE(parser.executorList[1].algoList[1].enable);
}

// 测试正确用例5：冗余空格容错
TEST_F(HcclAlgoParserTest, ParseCorrectCase5)
{
    HcclAlgoParser parser;
    std::string input = "mesh2die; sequence{meshchunk  ,not( nhrmultilink)};allgather:sequence{mesh, nhr}";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 3u);

    // 检查第一段：mesh2die → sole{mesh2die}
    EXPECT_EQ(parser.executorList[0].executorType, "sole");
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "mesh2die");

    // 检查第二段：sequence{meshchunk,not(nhrmultilink)}
    EXPECT_EQ(parser.executorList[1].algoList[0].algoType, "meshchunk");
    EXPECT_EQ(parser.executorList[1].algoList[1].algoType, "nhrmultilink");
    EXPECT_FALSE(parser.executorList[1].algoList[1].enable);
}

TEST_F(HcclAlgoParserTest, MeshConfigurationDoesNotExposeWriteSemantics)
{
    HcclAlgoParser parser;
    EXPECT_EQ(parser.Parser("alltoall:sole{mesh}"), HCCL_SUCCESS);
    EXPECT_EQ(GetAlgoNameToTypeMap().count("MeshWrite"), 0u);
    EXPECT_EQ(GetAlgoNameToTypeMap().count("MeshTwoShotWrite"), 0u);
    EXPECT_NE(parser.Parser("alltoall:sole{meshwrite}"), HCCL_SUCCESS);
}

// 测试错误用例1：非法 opType
TEST_F(HcclAlgoParserTest, ParseErrorInvalidOpType)
{
    HcclAlgoParser parser;
    // 非法 opType "invalid_op"
    EXPECT_NE(parser.Parser("invalid_op:sole{meshchunk}"), HCCL_SUCCESS);
}

// 测试错误用例2：括号未闭合
TEST_F(HcclAlgoParserTest, ParseErrorUnclosedParen)
{
    HcclAlgoParser parser;
    // not( 后面缺少 )
    std::string input = "not(parallel{mesh2die,nhrmultilink};allgather:sequence{mesh,nhr}";

    HcclResult ret = parser.Parser(input);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// 测试错误用例3：花括号内出现分号
TEST_F(HcclAlgoParserTest, ParseErrorSemicolonInBrace)
{
    HcclAlgoParser parser;
    // concur{not(mesh2die),nhr;nhr_chunk} → 分号在花括号内
    std::string input
        = "allgather:not(sequence{level0=mesh2die,level1=nhrmultilink}); concur{not(mesh2die),nhr;meshchunk}";

    HcclResult ret = parser.Parser(input);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// 测试空输入
TEST_F(HcclAlgoParserTest, ParseEmptyInput)
{
    HcclAlgoParser parser;
    std::string input = "";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 0u);
}

// 测试shorthand展开，使用三种不同 ALGO_TYPES 值
TEST_F(HcclAlgoParserTest, ParseShorthand)
{
    HcclAlgoParser parser;
    std::string input = "nhrmultilink;meshoneshot;mesh2die";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 3u);

    // 所有shorthand都应展开为sole{...}
    EXPECT_EQ(parser.executorList[0].executorType, "sole");
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "nhrmultilink");
    EXPECT_EQ(parser.executorList[1].executorType, "sole");
    EXPECT_EQ(parser.executorList[1].algoList[0].algoType, "meshoneshot");
    EXPECT_EQ(parser.executorList[2].executorType, "sole");
    EXPECT_EQ(parser.executorList[2].algoList[0].algoType, "mesh2die");
}

// 测试非法 algoType 被拒绝
TEST_F(HcclAlgoParserTest, ParseErrorInvalidAlgoType)
{
    HcclAlgoParser parser;
    // "invalid_algo" 不在 ALGO_TYPES 表中
    EXPECT_NE(parser.Parser("allreduce:sole{invalid_algo}"), HCCL_SUCCESS);
}

// 测试非法 executorType 被拒绝
TEST_F(HcclAlgoParserTest, ParseErrorInvalidExecutorType)
{
    HcclAlgoParser parser;
    // "invalid_exec" 不在 EXECUTOR_TYPES 表中
    EXPECT_NE(parser.Parser("allreduce:invalid_exec{meshchunktwoshot}"), HCCL_SUCCESS);
}

// 测试多个 level 有 not，使用 mesh2die / nhrmultilink / meshchunktwoshot
TEST_F(HcclAlgoParserTest, ParseMultipleNegatedAlgo)
{
    HcclAlgoParser parser;
    std::string input = "allreduce:sequence{mesh2die,not(nhrmultilink),not(meshchunktwoshot)}";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 1u);

    EXPECT_EQ(parser.executorList[0].opType, "allreduce");
    EXPECT_EQ(parser.executorList[0].executorType, "sequence");
    EXPECT_EQ(parser.executorList[0].algoList.size(), 3u);
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "mesh2die");
    EXPECT_TRUE(parser.executorList[0].algoList[0].enable);
    EXPECT_EQ(parser.executorList[0].algoList[1].algoType, "nhrmultilink");
    EXPECT_FALSE(parser.executorList[0].algoList[1].enable);
    EXPECT_EQ(parser.executorList[0].algoList[2].algoType, "meshchunktwoshot");
    EXPECT_FALSE(parser.executorList[0].algoList[2].enable);
}

// 测试正确用例6：HD 算法词表解析——allgather:sole{hd} 单段（HD 模板显式启用 DSL）
TEST_F(HcclAlgoParserTest, ParseHdAlgoCase1)
{
    HcclAlgoParser parser;
    std::string input = "allgather:sole{hd}";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 1u);

    EXPECT_EQ(parser.executorList[0].opType, "allgather");
    EXPECT_EQ(parser.executorList[0].executorType, "sole");
    EXPECT_EQ(parser.executorList[0].algoList.size(), 1u);
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "hd");
    EXPECT_TRUE(parser.executorList[0].enable);
}

// 测试正确用例7：HD 算法词表解析——allgather:parallel{hd,hd} 两段（双槽位绑定 DSL）
TEST_F(HcclAlgoParserTest, ParseHdAlgoCase2)
{
    HcclAlgoParser parser;
    std::string input = "allgather:parallel{hd,hd}";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 1u);

    EXPECT_EQ(parser.executorList[0].opType, "allgather");
    EXPECT_EQ(parser.executorList[0].executorType, "parallel");
    EXPECT_EQ(parser.executorList[0].algoList.size(), 2u);
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "hd");
    EXPECT_EQ(parser.executorList[0].algoList[1].algoType, "hd");
    EXPECT_TRUE(parser.executorList[0].enable);
}

// 测试正确用例8：HD 与既有算法混排——shorthand "hd"、sequence{mesh,hd}、not(parallel{hd,nhr})
TEST_F(HcclAlgoParserTest, ParseHdAlgoCase3)
{
    HcclAlgoParser parser;
    std::string input = "hd;allgather:sequence{mesh,hd};not(parallel{hd,nhr})";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 3u);

    // 检查第一段：hd → sole{hd}
    EXPECT_EQ(parser.executorList[0].opType, "");
    EXPECT_EQ(parser.executorList[0].executorType, "sole");
    EXPECT_EQ(parser.executorList[0].algoList.size(), 1u);
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "hd");
    EXPECT_TRUE(parser.executorList[0].enable);

    // 检查第二段：allgather:sequence{mesh,hd}
    EXPECT_EQ(parser.executorList[1].opType, "allgather");
    EXPECT_EQ(parser.executorList[1].executorType, "sequence");
    EXPECT_EQ(parser.executorList[1].algoList.size(), 2u);
    EXPECT_EQ(parser.executorList[1].algoList[0].algoType, "mesh");
    EXPECT_EQ(parser.executorList[1].algoList[1].algoType, "hd");
    EXPECT_TRUE(parser.executorList[1].enable);

    // 检查第三段：not(parallel{hd,nhr})
    EXPECT_EQ(parser.executorList[2].opType, "");
    EXPECT_EQ(parser.executorList[2].executorType, "parallel");
    EXPECT_EQ(parser.executorList[2].algoList.size(), 2u);
    EXPECT_EQ(parser.executorList[2].algoList[0].algoType, "hd");
    EXPECT_EQ(parser.executorList[2].algoList[1].algoType, "nhr");
    EXPECT_FALSE(parser.executorList[2].enable);
}
// 测试正确用例：HD 算法词表解析——reducescatter:sole{level0=hdaicpureduce}（Sole HD 显式启用 DSL）
TEST_F(HcclAlgoParserTest, ParseHdSoleReduceScatter)
{
    HcclAlgoParser parser;
    std::string input = "reducescatter:sole{level0=hdaicpureduce}";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 1u);

    EXPECT_EQ(parser.executorList[0].opType, "reducescatter");
    EXPECT_EQ(parser.executorList[0].executorType, "sole");
    EXPECT_EQ(parser.executorList[0].algoList.size(), 1u);
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "hdaicpureduce");
    EXPECT_TRUE(parser.executorList[0].algoList[0].enable);
    EXPECT_TRUE(parser.executorList[0].enable);
}

// 测试正确用例：HD 算法词表解析——reducescatter:parallel{level0=mesh,level1=hd}（层级化 MeshHD DSL，
// levelN= 前缀按下标展开：algoList[0] 对应 level0、algoList[1] 对应 level1）
TEST_F(HcclAlgoParserTest, ParseHdParallelReduceScatter)
{
    HcclAlgoParser parser;
    std::string input = "reducescatter:parallel{level0=mesh,level1=hd}";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 1u);

    EXPECT_EQ(parser.executorList[0].opType, "reducescatter");
    EXPECT_EQ(parser.executorList[0].executorType, "parallel");
    EXPECT_EQ(parser.executorList[0].algoList.size(), 2u);
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "mesh");
    EXPECT_EQ(parser.executorList[0].algoList[1].algoType, "hd");
    EXPECT_TRUE(parser.executorList[0].algoList[0].enable);
    EXPECT_TRUE(parser.executorList[0].algoList[1].enable);
    EXPECT_TRUE(parser.executorList[0].enable);
}

// 测试正确用例：HD 与既有算法混排——shorthand "hd"、executor 级否定 not(parallel{mesh,hd})、
// 槽位级否定 level0=not(hdaicpureduce)
TEST_F(HcclAlgoParserTest, ParseHdMixedAndNegated)
{
    HcclAlgoParser parser;
    std::string input = "hd;reducescatter:not(parallel{level0=mesh,level1=hd});sole{level0=not(hdaicpureduce)}";

    HcclResult ret = parser.Parser(input);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(parser.executorList.size(), 3u);

    // 第一段：shorthand "hd" → sole{hd}
    EXPECT_EQ(parser.executorList[0].opType, "");
    EXPECT_EQ(parser.executorList[0].executorType, "sole");
    EXPECT_EQ(parser.executorList[0].algoList.size(), 1u);
    EXPECT_EQ(parser.executorList[0].algoList[0].algoType, "hd");
    EXPECT_TRUE(parser.executorList[0].enable);

    // 第二段：reducescatter:not(parallel{level0=mesh,level1=hd}) → executor 级否定
    EXPECT_EQ(parser.executorList[1].opType, "reducescatter");
    EXPECT_EQ(parser.executorList[1].executorType, "parallel");
    EXPECT_EQ(parser.executorList[1].algoList.size(), 2u);
    EXPECT_EQ(parser.executorList[1].algoList[0].algoType, "mesh");
    EXPECT_EQ(parser.executorList[1].algoList[1].algoType, "hd");
    EXPECT_FALSE(parser.executorList[1].enable);

    // 第三段：sole{level0=not(hdaicpureduce)} → 槽位级否定（否定型配置不视为显式启用）
    EXPECT_EQ(parser.executorList[2].opType, "");
    EXPECT_EQ(parser.executorList[2].executorType, "sole");
    EXPECT_EQ(parser.executorList[2].algoList.size(), 1u);
    EXPECT_EQ(parser.executorList[2].algoList[0].algoType, "hdaicpureduce");
    EXPECT_FALSE(parser.executorList[2].algoList[0].enable);
}

// 测试：HD DSL 配置 → 注册名拼名匹配（选路打标链端到端）：显式配置必须使对应 HD 注册名获得
// 正向优先级——拼名 Aicpu+ReduceScatter+Sole+HDAicpuReduce 与 Aicpu+ReduceScatter+Parallel+Mesh+HD
// 须与注册名逐字符一致；拼名错位/大小写错会使选路静默回退既有算法（数据仍正确、用例仍 PASS），
// 本用例锁定该链路。注意：UpdateCostModelWithAlgo 倒序处理子句且同 opType 去重（后写子句生效），
// 两个绑定须分别用独立配置验证
TEST_F(HcclAlgoParserTest, UpdateCostModelWithAlgoHdNameMatch)
{
    const char* hdAlgNames[2] = {"AicpuReduceScatterSoleHDAicpuReduce", "AicpuReduceScatterParallelMeshHD"};
    const char* hdConfigs[2]
        = {"reducescatter:sole{level0=hdaicpureduce}", "reducescatter:parallel{level0=mesh,level1=hd}"};
    for (int cfg = 0; cfg < 2; cfg++) {
        HcclAlgoParser parser;
        ASSERT_EQ(parser.Parser(hdConfigs[cfg]), HCCL_SUCCESS);

        CostAlgoParams entries[2] = {};
        entries[0].algName = hdAlgNames[0];
        entries[0].count = 1;
        entries[1].algName = hdAlgNames[1];
        entries[1].count = 1;
        CostModel model{entries, 2};

        EXPECT_EQ(UpdateCostModelWithAlgo(parser, model, {"Aicpu"}), HCCL_SUCCESS);
        // 配置指向的绑定获得正向优先级，另一绑定不受影响
        EXPECT_EQ(entries[cfg].hcclAlgoPriority, 1) << "config: " << hdConfigs[cfg];
        EXPECT_EQ(entries[1 - cfg].hcclAlgoPriority, 0) << "config: " << hdConfigs[cfg];
    }
}

// 测试：否定型配置语义——not(nhr) 表示"不支持 nhr、其他的都可选"（HCCL_ALGO 文档语义），
// sole 执行器下所有非 nhr 算法（含 HD）同获正向优先级，最终选路由成本竞争决定；
// 与同族 AicpuAllReduceParallelMeshHD 的 topoCustomCheck 机制行为一致
TEST_F(HcclAlgoParserTest, UpdateCostModelWithAlgoHdNegatedSelectable)
{
    HcclAlgoParser parser;
    ASSERT_EQ(parser.Parser("reducescatter:sole{level0=not(nhr)}"), HCCL_SUCCESS);

    CostAlgoParams entries[1] = {};
    entries[0].algName = "AicpuReduceScatterSoleHDAicpuReduce";
    entries[0].count = 1;
    CostModel model{entries, 1};

    EXPECT_EQ(UpdateCostModelWithAlgo(parser, model, {"Aicpu"}), HCCL_SUCCESS);
    EXPECT_EQ(entries[0].hcclAlgoPriority, 1);
}
