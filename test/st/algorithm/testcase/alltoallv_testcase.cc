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
class ST_ALLTOALLV_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        // 取消设置环境变量
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}

    void RunAlltoAllVMeshTest(TopoMeta &topoMeta, uint32_t rankSize, HcclDataType dataType, std::vector<u64> &sendCountMatrix)
    {
        SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
        // 设置展开模式为HOST_TS
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("HCCL_BUFFSIZE", "200", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        setenv("HCCL_ENABLE_OPEN_AICPU", "1", 1);

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

                // 构造数据
                std::vector<u64> sendCounts(rankSize, 0);
                std::vector<u64> recvCounts(rankSize, 0);
                std::vector<u64> sdispls(rankSize, 0);
                std::vector<u64> rdispls(rankSize, 0);

                u64 sendDataCount = 0;
                for (u64 i = 0; i < rankSize; i++) {
                    sendCounts[i] = sendCountMatrix[rankId * rankSize + i];
                    sdispls[i] = sendDataCount;
                    sendDataCount += sendCounts[i];
                }

                u64 recvDataCount = 0;
                for (u64 i = 0; i < rankSize; i++) {
                    recvCounts[i] = sendCountMatrix[i * rankSize + rankId];
                    rdispls[i] = recvDataCount;
                    recvDataCount += recvCounts[i];
                }

                void *sendBuf = nullptr;
                void *recvBuf = nullptr;
                // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
                aclrtMalloc(&sendBuf, sendDataCount * SIZE_TABLE[dataType], static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
                aclrtMalloc(&recvBuf, recvDataCount * SIZE_TABLE[dataType], static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

                // 4.算子下发
                CHK_RET(HcclAlltoAllV(sendBuf,
                    sendCounts.data(),
                    sdispls.data(),
                    dataType,
                    recvBuf,
                    recvCounts.data(),
                    rdispls.data(),
                    dataType,
                    comm,
                    stream));

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
        HcclResult res = CheckAll2AllV(taskQueues, rankSize, dataType, sendCountMatrix);
        EXPECT_TRUE(res == HCCL_SUCCESS);

        // 资源清理
        SimWorld::Global()->Deinit();
    }

    void RunHostDpuAlltoAllVMeshTest(TopoMeta &topoMeta, HcclDataType dataType, std::vector<u64> &sendCountMatrix)
    {
        SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
        // 设置环境变量
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("ENABLE_HOSTDPU", "1", 1);
        setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);

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

                // 构造数据
                std::vector<u64> sendCounts(rankSize, 0);
                std::vector<u64> recvCounts(rankSize, 0);
                std::vector<u64> sdispls(rankSize, 0);
                std::vector<u64> rdispls(rankSize, 0);

                u64 sendDataCount = 0;
                for (u64 i = 0; i < rankSize; i++) {
                    sendCounts[i] = sendCountMatrix[rankId * rankSize + i];
                    sdispls[i] = sendDataCount;
                    sendDataCount += sendCounts[i];
                }

                u64 recvDataCount = 0;
                for (u64 i = 0; i < rankSize; i++) {
                    recvCounts[i] = sendCountMatrix[i * rankSize + rankId];
                    rdispls[i] = recvDataCount;
                    recvDataCount += recvCounts[i];
                }

                void *sendBuf = nullptr;
                void *recvBuf = nullptr;
                // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
                aclrtMalloc(&sendBuf, sendDataCount * SIZE_TABLE[dataType],
                            static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
                aclrtMalloc(&recvBuf, recvDataCount * SIZE_TABLE[dataType],
                            static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

                // 4.算子下发
                CHK_RET(HcclAlltoAllV(sendBuf, sendCounts.data(), sdispls.data(), dataType, recvBuf, recvCounts.data(),
                                      rdispls.data(), dataType, comm, stream));

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
        HcclResult res = CheckAll2AllV(taskQueues, rankSize, dataType, sendCountMatrix);
        EXPECT_TRUE(res == HCCL_SUCCESS);

        // 资源清理
        SimWorld::Global()->Deinit();
    }
};


TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_0)
{
    TopoMeta topoMeta{{{0}, {0}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT32;

    // 构造sendCountMatrix，每个rank发送给其他rank的数据量
    // 矩阵大小：rankSize × rankSize
    std::vector<u64> sendCountMatrix = {
        10, 11,
        1, 2222
    };

    RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_1)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP64;

    std::vector<u64> sendCountMatrix = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_2)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT64;

    std::vector<u64> sendCountMatrix = {
        0, 50, 25, 209715200,
        50, 314572800, 30, 80,
        209715200, 30, 0, 0,
        75, 0, 45, 209715200
    };
    RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_3)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    std::vector<u64> sendCountMatrix = {
        21111, 50, 0, 209715200,
        50, 314572800, 0, 80,
        209715200, 0, 0, 21,
        75, 0, 45, 209235255
    };
    RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_4)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {0}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0;

    std::vector<u64> sendCountMatrix = {
        524288000,  0,  0,  0,  0,
        0,  524288000,  0,  0,  0,
        0, 0, 524288000, 0, 0,
        0, 0, 0, 524288000, 0,
        0, 0, 0, 0, 524288000
    };
    RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_5)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    std::vector<u64> sendCountMatrix = {
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    };

    RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_6)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;
    
    std::vector<u64> sendCountMatrix = {
        524288000, 230686721, 356876, 524288000, 5556876, 230686721,
        1048577, 1025, 230686721, 2097153, 4096, 2097152,
        524288000, 4096, 230686721, 8197, 230686721, 2097152,
        230686721, 1048577, 524288000, 2097152, 230686721, 1048577,
        2097153, 8197, 230686721, 1025, 4096, 524288000,
        524288000, 2097152, 1048576, 512, 256, 230686721
    };

    RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_7)
{
    TopoMeta topoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2;
    
    std::vector<u64> sendCountMatrix = {
        0, 1024, 891289, 671088, 0, 943718, 384, 888, 0,
        9600, 33, 0, 293601, 0, 30, 1024, 0, 5872025,
        0, 0, 0, 0, 377487, 4097, 888, 44, 104857,
        1, 0, 1200, 20, 0, 41943040, 1025, 0, 333,
        33, 104857, 1, 146800, 85244, 335544, 3, 55, 1,
        102, 0, 22, 0, 10, 22, 0, 314572800, 1,
        122, 0, 0, 512, 20546, 2, 7200, 1, 0,
        0, 55, 11, 1, 0, 55, 2560, 0, 35262,
        1, 40, 0, 8192, 1025, 0, 1, 33, 384
    };

    RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_8)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}, {0}, {0}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    
    std::vector<u64> sendCountMatrix = {
        524288000, 3068, 230686721, 0, 4096, 8193,
        0, 1025, 0, 0, 4096, 524288000,
        0, 4096, 230686721, 8197, 5555, 524288000,
        230686721, 0, 512, 0, 3065, 0,
        0, 8197, 0, 1025, 4096, 0,
        2048, 522486, 0, 230686721, 256, 0
    };

    RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_hostDpu_test_9)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}, {5, 6}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2;
    
    std::vector<u64> sendCountMatrix = {
        1, 0, 134217728, 536870912, 67108864, 536870912,
        1, 67108864, 0, 536870912, 67108864, 536870912,
        1, 67108864, 134217728, 0, 67108864, 536870912,
        1, 67108864, 134217728, 536870912, 0, 536870912,
        1, 67108864, 134217728, 536870912, 67108864, 0,
        0, 67108864, 134217728, 536870912, 67108864, 536870912,
    };

RunHostDpuAlltoAllVMeshTest(topoMeta, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_0)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        0, 0,
        0, 0,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_1)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;

    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        1048576, 1048576,
        1048576, 1048576,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_2)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        1073741824, 1073741824,
        1073741824, 1073741824,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_3)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;

    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        10737418240, 10737418240,
        10737418240, 10737418240,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_4)
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

    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        1073741824, 1073741824,
        1073741824, 1073741824,
    };
    for (uint32_t i = 0; i < dataTypeList.size(); i++) {
        RunAlltoAllVMeshTest(topoMeta, rankSize, dataTypeList[i], sendCountMatrix);
    }
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_5)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
    1073741824, 1073741824,
    1073741824, 1073741824,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_6)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        0, 4294967296,
        2147483648, 210763776,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_7)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        4294967296, 0,
        2147483648, 5,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_8)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT64;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        40, 8,
        0, 1024,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_9)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT16;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        100, 50,
        100, 0,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_10)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT32;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        0, 134217728,
        0, 134217728,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_11)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        536870912, 0,
        536870912, 0,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_12)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        0, 0,
        268435456, 268435456,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_13)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP64;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        536870912, 536870912,
        0, 0,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_14)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        10485760, 0,
        0, 0,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_15)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP64;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        0, 5,
        0, 0,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_16)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        0, 0,
        10485760, 0,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_17)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E4M3;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        0, 0,
        0, 10737418240,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_18)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        10737418240, 134217728,
        134217728, 10737418240,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_19)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        1073741824, 101,
        98, 1074790400,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_20)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        268435456, 536870912,
        1023, 134217728,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_21)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        2, 999,
        2, 4096,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_22)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        12, 12,
        8, 16,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_23)
{
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 2;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        128, 1048576,
        2048, 92274176,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_24)
{
    TopoMeta topoMeta {{{0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 3;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        1048576, 1048576, 1048576,
        1048576, 1048576, 1048576,
        1048576, 1048576, 1048576,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_25)
{
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 4;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT8;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        1073741824, 1073741824, 1073741824, 1073741824,
        1073741824, 1073741824, 1073741824, 1073741824,
        1073741824, 1073741824, 1073741824, 1073741824,
        1073741824, 1073741824, 1073741824, 1073741824,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_26)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 5;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        1, 67108864, 134217728, 0, 67108864,
        1, 67108864, 134217728, 536870912, 0,
        0, 67108864, 134217728, 536870912, 67108864,
        1, 0, 134217728, 536870912, 67108864,
        1, 67108864, 0, 536870912, 67108864,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_27)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 6;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        1, 0, 134217728, 536870912, 67108864, 536870912,
        1, 67108864, 0, 536870912, 67108864, 536870912,
        1, 67108864, 134217728, 0, 67108864, 536870912,
        1, 67108864, 134217728, 536870912, 0, 536870912,
        1, 67108864, 134217728, 536870912, 67108864, 0,
        0, 67108864, 134217728, 536870912, 67108864, 536870912,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

TEST_F(ST_ALLTOALLV_TEST, st_alltoallv_28)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6}}};  // 三维数组指定超节点-Server-Device信息
    uint32_t rankSize = 7;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8;
    // 构造sendCountMatrix，每个rank再去构造自己对应的数据
    std::vector<u64> sendCountMatrix = {
        0, 67108864, 134217728, 536870912, 4096, 67108864, 67108864,
        1, 0, 134217728, 536870912, 4096, 67108864, 67108864,
        1, 67108864, 0, 536870912, 4096, 67108864, 67108864,
        1, 67108864, 134217728, 0, 4096, 67108864, 67108864,
        1, 67108864, 134217728, 536870912, 0, 67108864, 67108864,
        1, 67108864, 134217728, 536870912, 4096, 0, 67108864,
        1, 67108864, 134217728, 536870912, 4096, 67108864, 0,
    };
    RunAlltoAllVMeshTest(topoMeta, rankSize, dataType, sendCountMatrix);
}

}