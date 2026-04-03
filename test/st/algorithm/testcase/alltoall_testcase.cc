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
namespace checker {
class ST_ALLTOALL_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        // 取消设置环境变量
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}

    void RunAlltoAllMeshTest(TopoMeta &topoMeta, uint32_t rankSize, HcclDataType dataType, uint64_t dataCount)
    {
        SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
        // 设置展开模式为HOST_TS
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("HCCL_BUFFSIZE", "200", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        

        // 设置收发数据量，收发数据量相同
        u64 sendDataCount = dataCount;
        u64 recvDataCount = dataCount;

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
                // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
                aclrtMalloc(&sendBuf, sendDataCount * SIZE_TABLE[dataType] * rankSize,
                            static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
                aclrtMalloc(&recvBuf, recvDataCount * SIZE_TABLE[dataType] * rankSize,
                            static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

                // 4.算子下发
                CHK_RET(HcclAlltoAll(sendBuf, sendDataCount, dataType, recvBuf, sendDataCount, dataType, comm, stream));

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
        HcclResult res = CheckAll2All(taskQueues, rankSize, dataType, sendDataCount);
        EXPECT_TRUE(res == HCCL_SUCCESS);

        // 资源清理
        SimWorld::Global()->Deinit();
    }

    void RunHostDpuAlltoAllMeshTest(TopoMeta &topoMeta, HcclDataType dataType, uint64_t dataCount)
    {
        SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
        // 设置环境变量
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        

        // 设置发送数据量和接收数据量相同
        u64 sendDataCount = dataCount;
        u64 recvDataCount = dataCount;
        // 计算RankSize
        u32 rankSize = 0;
        for (const auto &superPod : topoMeta) {
            for (const auto &server : superPod) {
                rankSize += server.size();
            }
        }

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
                // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
                aclrtMalloc(&sendBuf, sendDataCount * SIZE_TABLE[dataType] * rankSize,
                            static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
                aclrtMalloc(&recvBuf, recvDataCount * SIZE_TABLE[dataType] * rankSize,
                            static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

                // 4.算子下发
                CHK_RET(HcclAlltoAll(sendBuf, sendDataCount, dataType, recvBuf, sendDataCount, dataType, comm, stream));

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
        HcclResult res = CheckAll2All(taskQueues, rankSize, dataType, sendDataCount);
        EXPECT_TRUE(res == HCCL_SUCCESS);

        // 资源清理
        SimWorld::Global()->Deinit();
    }
};

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_0)
{
    TopoMeta topoMeta{{{0}, {0}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;

    u64 sendDataCount = 0;
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_1)
{
    TopoMeta topoMeta{{{0}, {0}, {0}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT32;

    u64 sendDataCount = 1024;
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_2)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT64;

    u64 sendDataCount = 1024;
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_3)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16;

    // 500M + 1 > UB_MAX_TransDataSize(256M) > HCCL_BUF size(200M)
    u64 sendDataCount = 500 * 1024 * 1024 / sizeof(HcclDataType::HCCL_DATA_TYPE_INT16) + 1;
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_4)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E4M3;

    //  UB_MAX_TransDataSize(256M) > 220M > HCCL_BUF size(200M)
    u64 sendDataCount = 220 * 1024 * 1024 / sizeof(HcclDataType::HCCL_DATA_TYPE_FP8E4M3);
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_5)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0, 1}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;
    //  UB_MAX_TransDataSize(256M)> 220M + 1> HCCL_BUF size(200M)
    u64 sendDataCount = 220 * 1024 * 1024 / sizeof(HcclDataType::HCCL_DATA_TYPE_HIF8) - 1;
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_6)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;

    u64 sendDataCount = 1;
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_7)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    u64 sendDataCount = 1025;
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_8)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0;

    u64 sendDataCount = 0;
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_hostDpu_test_9)
{
    TopoMeta topoMeta{{{0, 1, 2, 3, 4}, {0}, {0}, {0}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP64;

    u64 sendDataCount = 11;
    RunHostDpuAlltoAllMeshTest(topoMeta, dataType, sendDataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_0)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    uint64_t dataCount = 0;
    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_1)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    uint64_t dataCount = 1048576;

    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_2)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    uint64_t dataCount = 1073741824;

    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_3)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    uint64_t dataCount = 10737418240;

    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_4)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    std::vector<HcclDataType> dataTypeList{
        HcclDataType::HCCL_DATA_TYPE_INT16,
        HcclDataType::HCCL_DATA_TYPE_INT32,
        HcclDataType::HCCL_DATA_TYPE_INT64,
        HcclDataType::HCCL_DATA_TYPE_UINT16,
        HcclDataType::HCCL_DATA_TYPE_UINT32,
        HcclDataType::HCCL_DATA_TYPE_UINT64,
        HcclDataType::HCCL_DATA_TYPE_FP32,
        HcclDataType::HCCL_DATA_TYPE_FP64,
        HcclDataType::HCCL_DATA_TYPE_HIF8,
        HcclDataType::HCCL_DATA_TYPE_FP8E4M3,
        HcclDataType::HCCL_DATA_TYPE_FP8E5M2,
        HcclDataType::HCCL_DATA_TYPE_FP8E8M0
    };
    uint64_t dataCount = 1073741824;

    for (uint32_t i = 0; i < dataTypeList.size(); i++) {
        RunAlltoAllMeshTest(topoMeta, rankSize, dataTypeList[i], dataCount);
    }
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_5)
{
    TopoMeta topoMeta {{{0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 3;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    uint64_t dataCount = 67108864;

    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_6)
{
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 4;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    uint64_t dataCount = 134217728;

    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_7)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 5;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    uint64_t dataCount = 536870912;

    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_8)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 6;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    uint64_t dataCount = 134217728;

    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_9)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 7;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;
    uint64_t dataCount = 67108864;

    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

TEST_F(ST_ALLTOALL_TEST, st_alltoall_10)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 8;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E4M3;
    uint64_t dataCount = 67108864;

    RunAlltoAllMeshTest(topoMeta, rankSize, dataType, dataCount);
}

}