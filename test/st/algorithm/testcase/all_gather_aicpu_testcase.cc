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

constexpr uint32_t DATATYPE_SIZE_TABLE_ALL_GATHER_ST[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_ALL_GATHER_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }

    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};
u32 AnalyseRankSize(const TopoMeta &topoInfo)
{
    u32 rankSize = 0;
    for (const auto &superPod : topoInfo) {
        for (const auto &podIdx : superPod) {
            rankSize += podIdx.size();
        }
    }
    return rankSize;
}

void RunAllGatherAicpuA5(const TopoMeta &topoInfo, const u64 &sendCount, const HcclDataType &dataType)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_910_95);

    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_ALL_GATHER_ST[dataType];
    auto rankSize = AnalyseRankSize(topoInfo);
    // 算子执行参数设置
    // 多线程运行SCATTER算子
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
            u64 sendBufSize = sendCount * dataTypeSize;  // 数据量转化为字节数
            u64 recvBufSize = sendCount * dataTypeSize * rankSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            HCCL_INFO("[ST_ALL_GATHER_AICPU_TEST]Run HcclAllGather");

            // 4.算子下发
            CHK_RET(HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    // 等待多线程执行完成
    for (auto &thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_2rank_int64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                                // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_3rank_int64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;  // 1GB单卡数据量，scratch单rank分区为100M（200M/2）, 有24M尾块
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT64;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_4rank_int32_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_6rank_int16_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 5}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                                 // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_8rank_int8_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                                 // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_2rank_hif8_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 1 * 1024 * 1024 * 1024;            // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_mesh_1d_4rank_fp8e4m3_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 1 * 1024 * 1024 * 1024;            // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E4M3;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}


TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_2rank_fp64_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP64;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_3rank_fp32_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_4rank_fp16_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_8rank_bfp16_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                               // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_6rank_fp8e5m2_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;                              // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_nhr_6rank_fp8e8m0_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {0}, {0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 1024 * 1024 * 1024;                // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x2rank_small_data_test)
{
    // 1GB较大数据量（多loop），同时包含尾块，数据宽度8B
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;  // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x3rank_small_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 100;  // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT32;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}


TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_2x3rank_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 300 * 1024 * 1024;  // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT16;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_AICPU_TEST, st_all_gather_a5_aicpu_parallel_nhr_mesh1d_3x3rank_big_data_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto sendCount = 300 * 1024 * 1024;  // 单卡数据量

    auto dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;  // 数据类型
    RunAllGatherAicpuA5(topoMeta, sendCount, dataType);
}