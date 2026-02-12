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
 
class ST_BROADCAST_TEST : public ::testing::Test {
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
       void RunBroadcastTest(TopoMeta topoMeta, u32 rankSize, uint64_t count, HcclDataType dataType, u32 root, u32 dataTypeSize)
    {
        // 仿真模型初始化
        // TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
        SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);
    
        // 设置展开模式为HOST_TS
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
    
        // // 算子执行参数设置
        // auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
        // auto count = 100;  // 接收数据量
        // auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
        // auto root = 0;  // root节点
    
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
    
                void *buf = nullptr;
                u64 bufSize = count * dataTypeSize;
                // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
                aclrtMalloc(&buf, bufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
    
                // 4.算子下发
                CHK_RET(HcclBroadcast(buf, count, dataType, root, comm, stream));
    
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
        HcclResult res = CheckBroadcast(taskQueues, rankSize, dataType, count, root);
        EXPECT_TRUE(res == HCCL_SUCCESS);
    
        // 资源清理
        SimWorld::Global()->Deinit();
    }
};
 
TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_four_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
    auto count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_five_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 5;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    auto root = 1;  // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_four_4G_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = (uint64_t)1024*1024*1024*4;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int8_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_five_4G_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 5;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = (uint64_t)1024*1024*1024*4;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_six_4G_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 6;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = (uint64_t)1024*1024*1024*4;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 2;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_seven_4G_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 7;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = (uint64_t)1024*1024*1024*4;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_three_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 3;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_two_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 2;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_eight_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 8;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_eight_4G_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 8;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = (uint64_t)1024*1024*1024*4;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_eight_test_1)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 8;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 1;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_one_four_test_bigdata)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 550000;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_NHR_one_four_test_01)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 4);

    // 算子执行参数设置
    auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_NHR_one_four_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0},{0},{0},{0}}};

    // 算子执行参数设置
    auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 1;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_NHR_one_four_test_bigdata)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0},{0},{0},{0}}};


    // 算子执行参数设置
    auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 500000;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}


TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_server_two_two_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 2);

    // 算子执行参数设置
    auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_server_two_four_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 4);

    // 算子执行参数设置
    auto rankSize = 8;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_server_two_eight_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);

    // 算子执行参数设置
    auto rankSize = 16;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_server_eight_eight_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 8, 8);

    // 算子执行参数设置
    auto rankSize = 64;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    auto root = 0;  // root节点
    auto dataTypeSize = sizeof(int8_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_diagonal_root_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 8);

    // 算子执行参数设置
    auto rankSize = 32;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型
    auto root = 31;  // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_random_root_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 8);

    // 算子执行参数设置
    auto rankSize = 24;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;  // 数据类型
    auto root = 11;  // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_one_count_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);

    // 算子执行参数设置
    auto rankSize = 16;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 1;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    auto root = 5;  // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

TEST_F(ST_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_bigdata_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);

    // 算子执行参数设置
    auto rankSize = 16;  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 2 * 200 * 1024 * 1024 + 1;  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    auto root = 11;  // root节点
    auto dataTypeSize = sizeof(int8_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);

}

