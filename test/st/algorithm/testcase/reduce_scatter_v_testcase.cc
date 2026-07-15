/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "alg_env_config.h"
#include "v_testcase_common.h"

constexpr u32 DATATYPE_SIZE_TABLE_RSV[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_REDUCESCATTERV_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

static HcclResult ReduceScatterVDispatch(u32 rankId, u64 totalCount, VDataDesTag vDataDes,
    HcclComm comm, aclrtStream stream)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_RSV[vDataDes.dataType];
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    u64 recvDataCount = vDataDes.counts[rankId];
    u64 sendBufSize = totalCount * dataTypeSize;
    u64 recvBufSize = recvDataCount * dataTypeSize;
    aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
    aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
    return HcclReduceScatterV(sendBuf, vDataDes.counts.data(), vDataDes.displs.data(),
        recvBuf, recvDataCount, vDataDes.dataType, HcclReduceOp::HCCL_REDUCE_SUM, comm, stream);
}

static HcclResult ReduceScatterVVerify(AllRankTaskQueues &taskQueues, u32 rankSize, VDataDesTag vDataDes)
{
    return CheckReduceScatterV(taskQueues, rankSize, HcclReduceOp::HCCL_REDUCE_SUM, vDataDes);
}

static void SetIndependentOpEnv() { setenv("HCCL_INDEPENDENT_OP", "1", 1); }

static void RunReduceScatterVMultilevel(const TopoMeta &topoInfo, VDataDesTag vDataDes)
{
    RunVMultilevelTest(topoInfo, vDataDes, SetIndependentOpEnv, ReduceScatterVDispatch, ReduceScatterVVerify);
}

TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_aicpu_test)
{
    TopoMeta topoMeta{{{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {155, 155};
    vDataDes.displs = {0, 155};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT16;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_multilevel_2pod_4rank_int32_equal_test)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_multilevel_2pod_6rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {3, 4, 5}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {200, 200, 200, 200, 200, 200};
    vDataDes.displs = {0, 200, 400, 600, 800, 1000};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    
    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_3layer_2pod_2server_4rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0}, {1}}, {{0}, {1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {200, 200, 200, 200};
    vDataDes.displs = {0, 200, 400, 600};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_3layer_2pod_1server_4rank_int32_equal_test)
{
    TopoMeta topoMeta{{{0, 1}}, {{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

// asymmetric topology
TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_asymmetric_int8_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4}, {0, 1}, {0, 1, 2}}, {{0, 1, 2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {700, 700, 600, 600, 500, 500, 400, 400, 300, 300, 200, 200, 100, 100};
    vDataDes.displs = {0, 700, 1400, 2000, 2600, 3100, 3600, 4000, 4400, 4700, 5000, 5200, 5400, 5500};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_REDUCESCATTERV_TEST, st_reduce_scatter_v_a5_asymmetric_fp32_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}, {{0, 1, 2, 3, 4, 5}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100, 100, 100, 100, 100, 500, 500};
    vDataDes.displs = {0, 100, 200, 300, 400, 500, 600, 700, 800, 1300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunReduceScatterVMultilevel(topoMeta, vDataDes);
}