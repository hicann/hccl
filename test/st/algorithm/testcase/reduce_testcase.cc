/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

class ST_REDUCE_TEST
    : public ::testing::TestWithParam<std::tuple<TopoMeta, u64, HcclDataType, HcclReduceOp, uint32_t>> {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }

    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_BUFFSIZE");
    }

    static void SetUpTestCase()
    {}

    static void TearDownTestCase()
    {}

    size_t GetDataTypeSize(HcclDataType type)
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
            default:
                return 0;
        }
    }

    u32 GetRankSize(const TopoMeta &topoMeta)
    {
        u32 rankSize = 0;
        for (const SuperPodMeta &superPod : topoMeta) {
            for (const ServerMeta &server : superPod) {
                rankSize += static_cast<u32>(server.size());
            }
        }
        return rankSize;
    }

    void RunReduceTest(
        const TopoMeta &topoMeta, u64 recvCount, HcclDataType dataType, HcclReduceOp reduceOp, uint32_t root)
    {
        // 初始化仿真环境
        SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);
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

                void *sendBuf = nullptr;
                void *recvBuf = nullptr;
                u64 sendBufSize = recvCount * GetDataTypeSize(dataType) * rankSize;
                u64 recvBufSize = recvCount * GetDataTypeSize(dataType);
                aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
                aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

                CHK_RET(HcclReduce(sendBuf, recvBuf, recvCount, dataType, reduceOp, root, comm, stream));

                CHK_RET(HcclCommDestroy(comm));
                return HCCL_SUCCESS;
            });
        }

        for (auto &thread : threads) {
            thread.join();
        }

        auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
        HcclResult res = CheckReduce(taskQueues, rankSize, dataType, recvCount, reduceOp, root);
        EXPECT_TRUE(res == HCCL_SUCCESS);

        SimWorld::Global()->Deinit();
    }
};

TEST_P(ST_REDUCE_TEST, st_reduce_aicpu_test)
{
    auto params = GetParam();
    const auto &topoMeta = std::get<0>(params);
    u64 recvCount = std::get<1>(params);
    HcclDataType dataType = std::get<2>(params);
    HcclReduceOp reduceOp = std::get<3>(params);
    uint32_t root = std::get<4>(params);

    RunReduceTest(topoMeta, recvCount, dataType, reduceOp, root);
}

TopoMeta GenerateMeshTopoMeta(u32 xSize, u32 ySize = 1, u32 serverSize = 1)
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

// 参数化实例化
INSTANTIATE_TEST_SUITE_P(ReduceVariants, ST_REDUCE_TEST,
    ::testing::Values(
        // 每个 tuple 表示一组测试参数
        // 1D Mesh
        std::make_tuple(GenerateMeshTopoMeta(2), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(3), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(5), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(6), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(7), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(8), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 1 << 20, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 1),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 2),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 3),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_MAX, 0),
        std::make_tuple(GenerateMeshTopoMeta(4), 100, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_MIN, 0),
        std::make_tuple(GenerateMeshTopoMeta(2), (1 << 24) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2), 220 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2), (1 << 30) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2), 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        // NHR
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 2), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 3), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 4), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 2), 1 << 20, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 2), (1 << 30) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 5), 400 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 2),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 6), 111 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 5),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 7), 50 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 4),
        std::make_tuple(GenerateMeshTopoMeta(1, 1, 8), 17 * (1 << 20) - 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 4),
        // 1DMeshNHR
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 2, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 3, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 1),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 2),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 3),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 100, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 1),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 4097, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 2),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), (1 << 30) - 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_MIN, 1),
        std::make_tuple(GenerateMeshTopoMeta(3, 1, 2), 100, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, 3),
        std::make_tuple(GenerateMeshTopoMeta(3, 1, 3), 100, HCCL_DATA_TYPE_BFP16, HCCL_REDUCE_SUM, 8),
        std::make_tuple(GenerateMeshTopoMeta(4, 1, 2), 210 * (1 << 20), HCCL_DATA_TYPE_INT32, HCCL_REDUCE_MIN, 6),
        std::make_tuple(GenerateMeshTopoMeta(4, 1, 4), 4095, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 11),
        std::make_tuple(GenerateMeshTopoMeta(8, 1, 2), 100, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_MAX, 5),
        std::make_tuple(GenerateMeshTopoMeta(3, 1, 3), 1 << 30, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(3, 1, 3), 100, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 1023, HCCL_DATA_TYPE_INT16, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 24 * (1 << 20) - 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 96 * (1 << 20) - 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0),
        std::make_tuple(GenerateMeshTopoMeta(2, 1, 2), 16 * (1 << 20) - 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0)
        // 2DMeshNHR
        ));
