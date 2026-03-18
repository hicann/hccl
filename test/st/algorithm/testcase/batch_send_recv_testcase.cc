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
#include <random>

using namespace HcclSim;
using namespace ops_hccl;

class ST_BATCH_SEND_RECV_TEST : public ::testing::Test {
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

HcclResult MultiThreadExecOp(u32 rankSize, u64 sendBufSize, u64 sendCount, HcclDataType dataType)
{
    // 多线程运行BATCH_SEND_RECV算子
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
            u64 totalSendBufSize = sendBufSize * rankSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, totalSendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, totalSendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            printf("ADDR rankId:%u, sendBuf:%lx, recvBuf:%lx, totalSendBufSize:%lx\n", rankId, (uint64_t)sendBuf, (uint64_t)recvBuf, (uint64_t)totalSendBufSize);
            std::vector<HcclSendRecvItem> sendRecvInfo;
            sendRecvInfo.reserve(rankSize * 2);
            void* curSendBuf = sendBuf;
            void* curRecvBuf = recvBuf;
            for (u32 i = 0; i < rankSize; i++) {
                curSendBuf = static_cast<void*>(static_cast<u8*>(sendBuf) + sendBufSize * i);
                curRecvBuf = static_cast<void*>(static_cast<u8*>(recvBuf) + sendBufSize * i);
                sendRecvInfo.push_back({HcclSendRecvType::HCCL_SEND, curSendBuf, sendCount, dataType, i});
                sendRecvInfo.push_back({HcclSendRecvType::HCCL_RECV, curRecvBuf, sendCount, dataType, i});
            }

            // 4.算子下发
            CHK_RET(HcclBatchSendRecv(sendRecvInfo.data(), sendRecvInfo.size(), comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return HCCL_SUCCESS;
}

void RunBatchSendRecvTest(TopoMeta topoMeta, u32 rankSize, u64 dataCount, HcclDataType dataType, u64 unitSize)
{
    // 仿真模型初始化
    #ifdef MACRO_DEV_TYPE_NEW
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
    #else
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);
    #endif

    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    // 算子执行参数设置
    u64 dataBufSize = dataCount * unitSize;

    // 多线程运行BATCH_SEND_RECV算子
    HcclResult exeRes = MultiThreadExecOp(rankSize, dataBufSize, dataCount, dataType);
    EXPECT_TRUE(exeRes == HCCL_SUCCESS);
    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckBatchSendRecv(taskQueues, rankSize, dataType, dataCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

HcclResult MultiThreadExecOpMultiTimes(u32 rankSize, u64 sendBufSize, u64 recvCount, HcclDataType dataType)
{
    // 多线程运行BATCH_SEND_RECV算子
    // 运行2次，第一次每个rank给前一个rank发，从后一个rank收；第二次每个rank给前2个rank发，从后2个rank收
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

            // 4.申请内存
            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            u64 sendBufSize = recvCount * sizeof(uint32_t);  // 数据量转化为字节数
            u64 recvBufSize = recvCount * sizeof(uint32_t);
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            void *sendBuf1 = nullptr;
            void *recvBuf1 = nullptr;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf1, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf1, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 5.第一次算子下发
            u32 lastRank = (rankId + rankSize - 1) % rankSize;
            u32 nextRank = (rankId + rankSize + 1) % rankSize;
            u32 lastTwoRank = (rankId + rankSize - 2) % rankSize;
            u32 nextTwoRank = (rankId + rankSize + 2) % rankSize;
            std::vector<HcclSendRecvItem> sendRecvInfo = {
                {HcclSendRecvType::HCCL_SEND, sendBuf, recvCount, dataType, lastRank},
                {HcclSendRecvType::HCCL_RECV, recvBuf, recvCount, dataType, nextRank},
                {HcclSendRecvType::HCCL_SEND, sendBuf1, recvCount, dataType, lastTwoRank},
                {HcclSendRecvType::HCCL_RECV, recvBuf1, recvCount, dataType, nextTwoRank}
            };
            CHK_RET(HcclBatchSendRecv(sendRecvInfo.data(), sendRecvInfo.size(), comm, stream));

            // 6.第二次算子下发
            sendRecvInfo = {
                {HcclSendRecvType::HCCL_SEND, sendBuf, recvCount, dataType, nextRank},
                {HcclSendRecvType::HCCL_RECV, recvBuf, recvCount, dataType, lastRank}
            };
            CHK_RET(HcclBatchSendRecv(sendRecvInfo.data(), sendRecvInfo.size(), comm, stream));

            // 7.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return HCCL_SUCCESS;
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_000)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 2);
    RunBatchSendRecvTest(topoMeta, 2, 1, HcclDataType::HCCL_DATA_TYPE_INT32, 4);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_001)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 3);
    RunBatchSendRecvTest(topoMeta, 3, 100, HcclDataType::HCCL_DATA_TYPE_INT32, 4);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_002)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 6);
    RunBatchSendRecvTest(topoMeta, 6, 1000, HcclDataType::HCCL_DATA_TYPE_FP32, 4);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_003)
{
    u64 dataCount = static_cast<u64>(1024)*1024*1024;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 2);
    RunBatchSendRecvTest(topoMeta, 2, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, 2);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_004)
{
    u64 dataCount = static_cast<u64>(1024)*1024*1024*4;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 4);
    RunBatchSendRecvTest(topoMeta, 4, dataCount, HcclDataType::HCCL_DATA_TYPE_INT8, 1);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_005)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);
    RunBatchSendRecvTest(topoMeta, 8, 1000, HcclDataType::HCCL_DATA_TYPE_INT16, 2);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_006)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);
    RunBatchSendRecvTest(topoMeta, 8, 10000, HcclDataType::HCCL_DATA_TYPE_UINT32, 4);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_007)
{
    u64 dataCount = static_cast<u64>(1024)*1024*1024;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);
    RunBatchSendRecvTest(topoMeta, 8, dataCount, HcclDataType::HCCL_DATA_TYPE_INT64, 8);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_008)
{
    u64 dataCount = static_cast<u64>(1024)*1024*1024*2;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 6);
    RunBatchSendRecvTest(topoMeta, 6, dataCount, HcclDataType::HCCL_DATA_TYPE_BFP16, 2);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_009)
{
    u64 dataCount = static_cast<u64>(1024)*1024*1024*4;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 5);
    RunBatchSendRecvTest(topoMeta, 5, dataCount, HcclDataType::HCCL_DATA_TYPE_UINT64, 8);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_010)
{
    u64 dataCount = static_cast<u64>(1024)*1024*1024*6;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);
    RunBatchSendRecvTest(topoMeta, 8, dataCount, HcclDataType::HCCL_DATA_TYPE_FP8E4M3, 1);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_011)
{
    u64 dataCount = static_cast<u64>(1024)*1024*1024*8;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 7);
    RunBatchSendRecvTest(topoMeta, 7, dataCount, HcclDataType::HCCL_DATA_TYPE_FP32, 4);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_012)
{
    u64 dataCount = static_cast<u64>(1024)*1024;
    TopoMeta topoMeta {{{0, 1, 2, 4, 5}}};
    RunBatchSendRecvTest(topoMeta, 5, dataCount, HcclDataType::HCCL_DATA_TYPE_FP64, 8);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_013)
{
    u64 dataCount = static_cast<u64>(1024)*1024*1024;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    RunBatchSendRecvTest(topoMeta, 16, dataCount, HcclDataType::HCCL_DATA_TYPE_BFP16, 2);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_014)
{
    u64 dataCount = static_cast<u64>(1024)*1024*500;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 8);
    RunBatchSendRecvTest(topoMeta, 32, dataCount, HcclDataType::HCCL_DATA_TYPE_UINT64, 8);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_015)
{
    u64 dataCount = static_cast<u64>(1024)*1024*300;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 8, 8);
    RunBatchSendRecvTest(topoMeta, 64, dataCount, HcclDataType::HCCL_DATA_TYPE_FP8E5M2, 1);
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_016)
{
    u64 dataCount = 0;
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);
    // 仿真模型初始化
    #ifdef MACRO_DEV_TYPE_NEW
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
    #else
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);
    #endif

    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    // 算子执行参数设置
    u64 dataBufSize = dataCount * 1;

    // 多线程运行BATCH_SEND_RECV算子
    HcclResult exeRes = MultiThreadExecOp(8, dataBufSize, dataCount, HcclDataType::HCCL_DATA_TYPE_FP8E8M0);
    EXPECT_TRUE(exeRes == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

HcclResult ZeroItemTest(u32 rankSize, u64 sendBufSize, u64 sendCount, HcclDataType dataType)
{
    // 多线程运行BATCH_SEND_RECV算子
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
            u64 totalSendBufSize = sendBufSize * rankSize;  // 数据量转化为字节数
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, totalSendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, totalSendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            printf("ADDR rankId:%u, sendBuf:%lx, recvBuf:%lx, totalSendBufSize:%lx\n", rankId, (uint64_t)sendBuf, (uint64_t)recvBuf, (uint64_t)totalSendBufSize);
            std::vector<HcclSendRecvItem> sendRecvInfo;
            sendRecvInfo.reserve(rankSize * 2);
            void* curSendBuf = sendBuf;
            void* curRecvBuf = recvBuf;
            for (u32 i = 0; i < rankSize; i++) {
                curSendBuf = static_cast<void*>(static_cast<u8*>(sendBuf) + sendBufSize * i);
                curRecvBuf = static_cast<void*>(static_cast<u8*>(recvBuf) + sendBufSize * i);
                sendRecvInfo.push_back({HcclSendRecvType::HCCL_SEND, curSendBuf, sendCount, dataType, i});
                sendRecvInfo.push_back({HcclSendRecvType::HCCL_RECV, curRecvBuf, sendCount, dataType, i});
            }

            // 4.算子下发
            CHK_RET(HcclBatchSendRecv(sendRecvInfo.data(), 0, comm, stream));

            // 5.销毁通信域
            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    return HCCL_SUCCESS;
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_017)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);
    #ifdef MACRO_DEV_TYPE_NEW
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
    #else
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);
    #endif

    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    // 算子执行参数设置
    u64 dataCount = 100;
    u64 dataBufSize = 100;

    // 多线程运行BATCH_SEND_RECV算子
    HcclResult exeRes = MultiThreadExecOp(8, dataBufSize, dataCount, HcclDataType::HCCL_DATA_TYPE_FP8E8M0);
    
    EXPECT_TRUE(exeRes == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

TEST_F(ST_BATCH_SEND_RECV_TEST, st_batch_send_recv_a5_aicpu_test_run_twice)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    #ifdef MACRO_DEV_TYPE_NEW
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
    #else
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);
    #endif

    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    // 算子执行参数设置
    auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
    u64 recvCount = 100;  // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0;  // 数据类型
    u64 sendBufSize = recvCount * 1;

    HcclResult res = MultiThreadExecOpMultiTimes(rankSize, sendBufSize, recvCount, dataType);
    EXPECT_TRUE(res == HCCL_SUCCESS);
    SimWorld::Global()->Deinit();
}
