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
#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include <thread>
#include "alg_env_config.h"
 
using namespace HcclSim;
using namespace ops_hccl;
 
class ST_REDUCE_SCATTER_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

static void RunReduceScatterTest(const TopoMeta &topoMeta, u64 recvCount,
                                  HcclDataType dataType,
                                  HcclReduceOp reduceOp = HCCL_REDUCE_SUM)
{
    auto rankSize = 0;
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }
    size_t dataUnitSize = 0;
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:   dataUnitSize = sizeof(int8_t);  break;
        case HCCL_DATA_TYPE_INT16:  dataUnitSize = sizeof(int16_t); break;
        case HCCL_DATA_TYPE_INT32:  dataUnitSize = sizeof(int32_t); break;
        case HCCL_DATA_TYPE_INT64:  dataUnitSize = sizeof(int64_t); break;
        case HCCL_DATA_TYPE_FP16:   dataUnitSize = 2;               break;
        case HCCL_DATA_TYPE_FP32:   dataUnitSize = sizeof(float);   break;
        case HCCL_DATA_TYPE_FP64:   dataUnitSize = sizeof(double);  break;
        case HCCL_DATA_TYPE_BFP16:  dataUnitSize = 2;               break;
        case HCCL_DATA_TYPE_UINT8:  dataUnitSize = sizeof(uint8_t); break;
        default:                    dataUnitSize = 0;               break;
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));
            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;
            u64 recvBufSize = recvCount * dataUnitSize;
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    SimWorld::Global()->Deinit();
}
 
TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_001)
{
    RunReduceScatterTest(TopoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}}, 1, HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_002)
{
    RunReduceScatterTest(TopoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}}, 1 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_003)
{
    RunReduceScatterTest(TopoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}}, 1 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_004)
{
    RunReduceScatterTest(TopoMeta{{{0}, {0}, {0}, {0}}}, 301 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_005)
{
    RunReduceScatterTest(TopoMeta{{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}, 301 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
}
