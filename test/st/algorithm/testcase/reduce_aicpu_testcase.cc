/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_testcase_common.h"

// 两级64bit/PROD场景新算法(AicpuReduceSequenceNHRAicpuReduceNHR)的HCCL_ALGO指定token
constexpr char REDUCE_SPEC_ALGO_TOKEN[] = "reduce:sequence{nhraicpureduce,nhr}";

class ST_REDUCE_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        unsetenv("HCCL_ALGO"); // auto子模式兜底: 每条用例从干净算法环境开始
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_ALGO"); // spec子模式的setenv不泄漏到下一条用例
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_INDEPENDENT_OP");
        ResetAlgEnvConfigInitState();
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// ST_RD_2L_001: 两级16rank+INT64+SUM, 自适应选路按cost排序(CheckReduce硬断言, 选路日志复核)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_sum_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8); // 1pod×2srv×8rank=16卡, topoLevelNums=2
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_001 spec(代表组合1/3): 指定新算法验证INT64+SUM执行正确性(Phase3强制收敛)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_sum_spec_test)
{
    setenv("HCCL_ALGO", REDUCE_SPEC_ALGO_TOKEN, 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_002: 两级16rank+UINT64+SUM
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_uint64_sum_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_003: 两级16rank+FP64+SUM
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_fp64_sum_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_004: 两级16rank+INT64+PROD
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_prod_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_PROD, 0);
}

// ST_RD_2L_005: 两级16rank+INT64+MAX
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_max_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_MAX, 0);
}

// ST_RD_2L_005 spec(代表组合2/3): 指定新算法验证INT64+MAX(64bit×非SUM规约)执行正确性
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_max_spec_test)
{
    setenv("HCCL_ALGO", REDUCE_SPEC_ALGO_TOKEN, 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_MAX, 0);
}

// ST_RD_2L_006: 两级16rank+INT64+MIN(与MAX同族)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_min_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_MIN, 0);
}

// ST_RD_2L_007: 两级16rank+FP16+PROD(PROD分支×非64bit dtype)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_fp16_prod_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD, 0);
}

// ST_RD_2L_007 spec(代表组合3/3): 指定新算法验证FP16+PROD执行正确性
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_fp16_prod_spec_test)
{
    setenv("HCCL_ALGO", REDUCE_SPEC_ALGO_TOKEN, 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_PROD, 0);
}

// ST_RD_2L_008: 两级16rank+FP32+PROD
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_fp32_prod_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_PROD, 0);
}

// ST_RD_2L_009: 1pod×3srv×4rank=12rank+recvCount=243非整除尾块
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_12rank_fp64_sum_tail_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 4); // 1pod×3srv×4rank=12卡
    RunReduceTest(topoMeta, 243, HcclDataType::HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_009 spec: 指定新算法验证非整除尾块两级切分(tailSize语义)不越界
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_12rank_fp64_sum_tail_spec_test)
{
    setenv("HCCL_ALGO", REDUCE_SPEC_ALGO_TOKEN, 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 4);
    RunReduceTest(topoMeta, 243, HcclDataType::HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_010: 16M总量(超CCL单轮容量)触发多loop分轮
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_sum_16m_multiloop_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 2 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_010 spec: 指定新算法验证多loop分轮与逐loop断批同步(仅RS-L1 NHRPcie段)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_sum_16m_multiloop_spec_test)
{
    setenv("HCCL_ALGO", REDUCE_SPEC_ALGO_TOKEN, 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 2 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_011 spec(主): root=13(非0)时AG-L0仅root写回(isRootRank_语义)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_sum_root13_spec_test)
{
    setenv("HCCL_ALGO", REDUCE_SPEC_ALGO_TOKEN, 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 13);
}

// ST_RD_2L_011 auto(回归): root=13自适应选路合法性
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_sum_root13_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 13);
}

// CHK_RED_002/003: 中间root下校验非root recvBuf零写入与sendBuf零污染。
// 断言内嵌于 TaskCheckReduceSemantics 对全部Reduce用例生效, 本用例为其显式载体(root矩阵补8)。
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_sum_root8_nonroot_unwritten_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 8);
}

// ST_RD_2L_012: 三级32rank维持原路由选Sole(新算法attrs两级门禁被拒)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_3level_32rank_int64_sum_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8); // 2pod×2srv×8rank=32卡, topoLevelNums=3
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_013: 单机8rank选路维持单机算法(8KB<8MB阈值选SoleMesh)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_1level_8rank_int64_sum_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8); // 单机8卡, topoLevelNums=1
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_014: 两级+FP32+MAX门禁负向(非64bit非PROD, 新算法不进候选池)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_fp32_max_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_MAX, 0);
}

// ST_RD_2L_015 spec(主): count=1单元素切片退化(sliceSize/tailSize最小粒度)不越界
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_sum_count1_spec_test)
{
    setenv("HCCL_ALGO", REDUCE_SPEC_ALGO_TOKEN, 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RD_2L_015 auto(回归): count=1小消息按cost排序(SoleMesh胜出合法)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_int64_sum_count1_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1, HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RED_SYM_004 gate: 非白名单算法SoleMesh强制选路, 对称门禁按确定性白名单否决(恒回退CCL), 结果正确
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_fp32_sum_solemesh_gate_test)
{
    setenv("HCCL_ALGO", "reduce:sole{mesh}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RED_SYM_004 gate: 非白名单算法TwoShot强制选路(b7440772死代码清理回归对象), 恒回退CCL, 结果正确
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_fp32_sum_twoshot_gate_test)
{
    setenv("HCCL_ALGO", "reduce:sole{meshtwoshot}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_COMM_SYM_003: 同comm连续两次同shape调用, task cache/资源复用路径正确性(合并任务图checker)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_2level_16rank_fp32_sum_twice_call_cache_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunReduceTestTwice(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RED_SYM_003 gate: 3级16rank(2pod×2srv×4rank)强制选路ParallelNHRNHR, 对称路径正确性验证
// (替代32rank真机验证——16 rank ST拓扑即可满足ParallelNHRNHR的3级+UBOe+对称选路条件)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_3level_16rank_fp32_sum_parallel_nhrnhr_gate_test)
{
    setenv("HCCL_ALGO", "reduce:parallel{nhr,nhr}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4); // 2pod×2srv×4rank=16卡, topoLevelNums=3
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}

// ST_RED_SYM_003 gate auto: 3级16rank自适应选路(ParallelNHRNHR应被选中)
TEST_F(ST_REDUCE_AICPU_TEST, st_reduce_aicpu_3level_16rank_fp32_sum_auto_test)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    RunReduceTest(topoMeta, 1024, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, 0);
}
