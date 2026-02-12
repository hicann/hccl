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

constexpr uint32_t DATATYPE_SIZE_TABLE_REDUCE_SCATTER[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

 
class ST_REDUCE_SCATTER_AICPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

u32 CalsRankSize(const TopoMeta &topoMeta)
{
    u32 rankSize = 0;
    for (const auto &superPod : topoMeta) {
        for (const auto &podIdx : superPod) {
            rankSize += podIdx.size();
        }
    }
    return rankSize;
}
 
void RunReduceScatterAicpuA5(const TopoMeta &topoMeta, const u64 &recvCount, const HcclDataType &dataType,
    const HcclReduceOp &reduceOp)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    // 算子执行参数设置
    auto rankSize = CalsRankSize(topoMeta);  // 参与集合通信的卡数(同topoMeta卡数一致)
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_REDUCE_SCATTER[dataType];
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
            u64 sendBufSize = recvCount * dataTypeSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataTypeSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
 
            // 4.算子下发
            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));
 
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
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);
 
    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_3rank_int32_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 100;                                // 单卡数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}
 
TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_4rank_int8_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 1000;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_6rank_fp32_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 1000*1000;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_8rank_int16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_2rank_fp16_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_mesh_8rank_bfp16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 500;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_2rank_int16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_4rank_int32_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_3rank_fp32_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 400 * 1024 * 1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_8rank_fp16_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_4rank_int8_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 400 * 1024 * 1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_nhr_4rank_bf16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x2rank_int32_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_3x2rank_fp32_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}}}; // 三维数组指定超节点-Server-Device信息
    auto recvCount = 300 * 1024 * 1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x3rank_int16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1}, {0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x2rank_fp16_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    auto recvCount = 400 * 1024 * 1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x4rank_INT8_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    auto recvCount = 100 * 1024 * 1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshnhr_2x3rank_bf16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1}, {0, 1}, {0, 1}}}; // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_4rank_int8_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 400*1024*1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_3rank_int32_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200*1024*1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_6rank_fp32_min_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 400*1024*1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_8rank_int16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200*1024*1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_2rank_fp16_sum_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 4, 5}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 1*1024*1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}

TEST_F(ST_REDUCE_SCATTER_AICPU_TEST, st_reduce_scatter_a5_aicpu_meshchunk_8rank_bfp16_max_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息
    auto recvCount = 200*1024*1024;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatterAicpuA5(topoMeta, recvCount, dataType, reduceOp);
}