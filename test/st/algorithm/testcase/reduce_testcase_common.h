/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_TESTCASE_COMMON_H
#define REDUCE_TESTCASE_COMMON_H

#include <thread>
#include <numeric>
#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include "alg_env_config.h"

using namespace HcclSim;
using namespace ops_hccl;

static inline TopoMeta GenerateMeshTopoMeta(u32 xSize, u32 ySize = 1, u32 serverSize = 1)
{
    constexpr u32 MAX_SIZE = 8;
    if (xSize > MAX_SIZE) {
        throw std::runtime_error(
            "xSize = " + std::to_string(xSize) + " cannot be greater than " + std::to_string(MAX_SIZE));
    }
    if (ySize > MAX_SIZE) {
        throw std::runtime_error(
            "ySize = " + std::to_string(xSize) + " cannot be greater than " + std::to_string(MAX_SIZE));
    }
    std::vector<PhyDeviceId> level0Topo;
    for (u32 yIdx = 0; yIdx < ySize; yIdx++) {
        std::vector<PhyDeviceId> currRankIds(xSize);
        std::iota(currRankIds.begin(), currRankIds.end(), yIdx * MAX_SIZE);
        level0Topo.insert(level0Topo.end(), currRankIds.begin(), currRankIds.end());
    }
    std::vector<ServerMeta> serverTopo(serverSize, level0Topo);
    return TopoMeta{serverTopo};
}

static inline size_t GetDataTypeSize(HcclDataType type)
{
    switch (type) {
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return sizeof(int32_t);
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return sizeof(float);
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return sizeof(int16_t);
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            return sizeof(int16_t);
        case HcclDataType::HCCL_DATA_TYPE_BFP16:
            return sizeof(int16_t);
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return sizeof(int8_t);
        // 其他类型...
        case HcclDataType::HCCL_DATA_TYPE_FP64:
            return sizeof(double);
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return sizeof(double);
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return sizeof(double);
        default:
            return 0;
    }
}

static inline u32 GetRankSize(const TopoMeta& topoMeta)
{
    u32 rankSize = 0;
    for (const SuperPodMeta& superPod : topoMeta) {
        for (const ServerMeta& server : superPod) {
            rankSize += static_cast<u32>(server.size());
        }
    }
    return rankSize;
}

static inline void
RunReduceTest(const TopoMeta& topoMeta, u64 recvCount, HcclDataType dataType, HcclReduceOp reduceOp, uint32_t root)
{
    // 初始化仿真环境
    SimWorld::Global()->Init(topoMeta, HcclDevType::DEV_TYPE_950);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    std::vector<std::thread> threads;
    u32 rankSize = GetRankSize(topoMeta);
    for (int rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);
            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            u64 sendBufSize = recvCount * GetDataTypeSize(dataType) * rankSize;
            u64 recvBufSize = recvCount * GetDataTypeSize(dataType);
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            CHK_RET(HcclReduce(sendBuf, recvBuf, recvCount, dataType, reduceOp, root, comm, stream));

            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckReduce(taskQueues, rankSize, dataType, recvCount, reduceOp, root);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}

static inline void RunReduceDPUCase(
    const TopoMeta& topoInfo, const u64 dataCount, const HcclDataType dataType, const u32 dataTypeSize,
    const HcclReduceOp reduceOp, const u32 root)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoInfo, HcclDevType::DEV_TYPE_950);

    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

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

            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            u64 sendBufSize = dataCount * dataTypeSize; // 数据量转化为字节数
            u64 recvBufSize = dataCount * dataTypeSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 4.算子下发
            CHK_RET(HcclReduce(sendBuf, recvBuf, dataCount, dataType, reduceOp, root, comm, stream));

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
    HcclResult res = CheckReduce(taskQueues, rankSize, dataType, dataCount, reduceOp, root);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

#endif // REDUCE_TESTCASE_COMMON_H
