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
 
TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_001)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    size_t dataUnitSize = sizeof(int8_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_002)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 100 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_003)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 100 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_004)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 100 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_005)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_006)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_007)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_008)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_009)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_010)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_011)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_013)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_014)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;  // 数据类型
    size_t dataUnitSize = sizeof(float);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_016)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;  // 数据类型
    size_t dataUnitSize = sizeof(int16_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_017)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;  // 数据类型
    size_t dataUnitSize = sizeof(int16_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_018)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
    size_t dataUnitSize = sizeof(int32_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_019)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;  // 数据类型
    size_t dataUnitSize = sizeof(int16_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_020)
{
    // 仿真模型初始化
    TopoMeta topoMeta {{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};  // 三维数组指定超节点-Server-Device信息
    auto rankSize = 0;  // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }

    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);
 
    // 设置展开模式为AI_CPU
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    

    // 算子执行参数设置
    auto recvCount = 1 * 1024 * 1024; // 接收数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;  // 数据类型
    size_t dataUnitSize = sizeof(int8_t);
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;  // 数据量转化为字节数
            u64 recvBufSize = recvCount * dataUnitSize;
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