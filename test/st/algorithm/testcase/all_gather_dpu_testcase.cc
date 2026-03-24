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

constexpr uint32_t DATATYPE_SIZE_TABLE_AG_DPU_ST[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t),
    sizeof(int32_t), 2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t),
    sizeof(uint32_t), 8, 2, 16, 2, 1, 1, 1, 1};

class ST_ALL_GATHER_DPU_TEST : public ::testing::Test {
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
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

void RunAllGatherDPUA5(const TopoMeta &topoMeta, u64 sendCount, HcclDataType dataType)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    // 算子执行参数设置
    uint32_t rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto &&superPod : topoMeta) {
        for (auto &&server : superPod) {
            rankSize += server.size();
        }
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_AG_DPU_ST[dataType];

    // 多线程运行AllGather算子
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

            // 4.算子下发
            CHK_RET(HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream));

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
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_1_fp32)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 1;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp32)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_210m_fp32)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 210 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_210m_fp32_2_2)
{
    TopoMeta topoMeta {{{0, 1}, {0, 1}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 210 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int8)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int16)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int32)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int64)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT64;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp16)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp64)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP64;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_bfp16)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp8e5m2)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp8e4m3)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E4M3;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp8e8m0)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_hif8)
{
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;  // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;  // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}
