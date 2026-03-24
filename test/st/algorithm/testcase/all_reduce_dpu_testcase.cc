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

class ST_ALL_REDUCE_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

void RunAllReduceDPUCase(const TopoMeta &topoInfo, const u64 dataCount,
    const HcclDataType dataType, const u32 dataTypeSize, const HcclReduceOp reduceOp)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_950);

    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("ENABLE_HOSTDPU", "1", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);

    // 算子执行参数设置
    u32 rankSize = 0;
    for (auto elem : topoInfo[0]) {
        rankSize += elem.size();
    }

    // 多线程运行ALL REDUCE ONE SHOT算子
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            // 1.SetDevice
            aclrtSetDevice(rankId);

            // 2.创建流
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            // 3.初始化通信域
            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            u64 sendBufSize = dataCount * dataTypeSize;  // 数据量转化为字节数
            u64 recvBufSize = dataCount * dataTypeSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 4.算子下发
            CHK_RET(HcclAllReduce(sendBuf, recvBuf, dataCount, dataType, reduceOp, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllReduce(taskQueues, rankSize, dataType, dataCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_base_test)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    u64 dataCount = 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    u32 dataTypeSize = 1;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}
 
TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_single_dataCount)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    u64 dataCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    u32 dataTypeSize = 2;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}
 
TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_odd_dataCount)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}};
    u64 dataCount = 13;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    u32 dataTypeSize = 4;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}
 
TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_even_dataCount)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    u64 dataCount = 16;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    u32 dataTypeSize = 2;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}
 
TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_small_data)
{
    TopoMeta topoMeta{{{0, 1, 2}, {8, 9, 10}}};
    u64 dataCount = 4096;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    u32 dataTypeSize = 4;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}
 
TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_mid_data)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    u64 dataCount = 1048575;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    u32 dataTypeSize = 2;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}
 
TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_hcclbuff_add_1)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};
    u64 dataCount = 200 * 1024 * 1024 + 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    u32 dataTypeSize = 1;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}

TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_base_test_eight_server_sole_rank)
{
    TopoMeta topoMeta{{{0}, {2}, {3}, {1}, {2}, {6}, {5}, {7}}};
    u64 dataCount = 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    u32 dataTypeSize = 1;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}
 
TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_base_test_eight_server_sole_rank_5data)
{
    TopoMeta topoMeta{{{0}, {2}, {3}, {1}, {2}, {6}, {5}, {7}}};
    u64 dataCount = 5;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    u32 dataTypeSize = 2;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}
 
TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_odd_dataCount_three_server_five_rank)
{
    TopoMeta topoMeta{{{0, 1, 5, 6, 7}, {0, 1, 2, 4, 5}, {0, 1, 5, 6, 7}}};
    u64 dataCount = 13;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    u32 dataTypeSize = 4;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}
 
TEST_F(ST_ALL_REDUCE_DPU_TEST, st_all_reduce_dpu_mid_data_8servers_8ranks)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};
    u64 dataCount = 1048575;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    u32 dataTypeSize = 2;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceDPUCase(topoMeta, dataCount, dataType, dataTypeSize, reduceOp);
}