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
#include "selected_algo_log_stub.h"

using namespace HcclSim;
using namespace ops_hccl;

namespace {
constexpr uint64_t SIM_CCL_BUFFER_SIZE = 200ULL * 1024 * 1024;
constexpr uint64_t MultiLoopCount(uint64_t dataTypeSize) { return SIM_CCL_BUFFER_SIZE / dataTypeSize + 1; }
static_assert(
    MultiLoopCount(sizeof(int32_t)) * sizeof(int32_t) > SIM_CCL_BUFFER_SIZE,
    "multi-loop ST count must exceed one simulated CCL buffer");
} // namespace

class DPU_BROADCAST_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        ResetSelectedAlgoLog();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_ALGO");
        unsetenv("ENABLE_HOSTDPU_L2_ONLY");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void RunBroadcastTest(
        TopoMeta topoMeta, u32 rankSize, uint64_t count, HcclDataType dataType, u32 root, u32 dataTypeSize,
        const char* hcclAlgo = nullptr, const bool enableDpuL2Only = false)
    {
        // 仿真模型初始化
        // TopoMeta topoMeta {{{0, 1, 2, 3}}};  // 三维数组指定超节点-Server-Device信息
        SimWorld::Global()->Init(topoMeta, HcclDevType::DEV_TYPE_950);

        // 显式 HCCL_ALGO 只提供优先级；执行后断言 RHD 真实命中，防止静默回退导致假绿。
        if (hcclAlgo != nullptr) {
            setenv("HCCL_ALGO", hcclAlgo, 1);
        } else {
            unsetenv("HCCL_ALGO");
        }
        setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
        setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
        if (enableDpuL2Only) {
            setenv("ENABLE_HOSTDPU_L2_ONLY", "1", 1);
        } else {
            unsetenv("ENABLE_HOSTDPU_L2_ONLY");
        }
        setenv("HCCL_INDEPENDENT_OP", "1", 1);

        // // 算子执行参数设置
        // auto rankSize = 4;  // 参与集合通信的卡数(同topoMeta卡数一致)
        // auto count = 100;  // 接收数据量
        // auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型
        // auto root = 0;  // root节点

        // 多线程运行Broadcast算子
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

                void* buf = nullptr;
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
        if (hcclAlgo != nullptr) {
            ExpectAlgoHit("RHD");
        }

        // 资源清理
        SimWorld::Global()->Deinit();
    }
};

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_2x3_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {3, 4, 5}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    uint32_t rankSize = 0; // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto&& topo0 : topoMeta) {
        for (auto&& topo1 : topo0) {
            rankSize += topo1.size();
        }
    }
    auto count = 100;                                  // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto root = 0;                                     // root节点
    auto dataTypeSize = sizeof(int8_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_2x2_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {2, 3}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    uint32_t rankSize = 0; // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto&& topo0 : topoMeta) {
        for (auto&& topo1 : topo0) {
            rankSize += topo1.size();
        }
    }
    auto count = 100;                                   // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16; // 数据类型
    auto root = 0;                                      // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_3x3_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2}, {3, 4, 5}, {3, 4, 5}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    uint32_t rankSize = 0; // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto&& topo0 : topoMeta) {
        for (auto&& topo1 : topo0) {
            rankSize += topo1.size();
        }
    }
    auto count = 100;                                   // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 0;                                      // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_2x2_400M_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {2, 3}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    uint32_t rankSize = 0; // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto&& topo0 : topoMeta) {
        for (auto&& topo1 : topo0) {
            rankSize += topo1.size();
        }
    }
    auto count = (uint64_t)1024 * 1024 * 400;           // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 0;                                      // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_count1_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {2, 3}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    uint32_t rankSize = 0; // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto&& topo0 : topoMeta) {
        for (auto&& topo1 : topo0) {
            rankSize += topo1.size();
        }
    }
    auto count = 1;                                     // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 0;                                      // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_root1_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {2, 3}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    uint32_t rankSize = 0; // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto&& topo0 : topoMeta) {
        for (auto&& topo1 : topo0) {
            rankSize += topo1.size();
        }
    }
    uint64_t count = 100;                              // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    auto root = 1;                                     // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_root3_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    uint32_t rankSize = 0; // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto&& topo0 : topoMeta) {
        for (auto&& topo1 : topo0) {
            rankSize += topo1.size();
        }
    }
    uint64_t count = 400;                              // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    auto root = 3;                                     // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_1DTwoShot_250M_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1}, {2, 3}}}; // 三维数组指定超节点-Server-Device信息

    // 算子执行参数设置
    uint32_t rankSize = 0; // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto&& topo0 : topoMeta) {
        for (auto&& topo1 : topo0) {
            rankSize += topo1.size();
        }
    }
    uint64_t count = (uint64_t)1024 * 1024 * 250;      // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    auto root = 0;                                     // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_NHR_one_four_test_01)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}}};

    // 算子执行参数设置
    uint32_t rankSize = 0; // 参与集合通信的卡数(同topoMeta卡数一致)
    for (auto&& topo0 : topoMeta) {
        for (auto&& topo1 : topo0) {
            rankSize += topo1.size();
        }
    }
    uint64_t count = 100;                               // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 0;                                      // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_NHR_one_four_test)
{
    // // 仿真模型初始化
    // TopoMeta topoMeta;
    // GenTopoMeta(topoMeta, 1, 1, 4);
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {1}, {2}, {3}}};

    // 算子执行参数设置
    auto rankSize = 4;                                  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;                               // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 0;                                      // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_NHR_one_four_test_bigdata)
{
    // 仿真模型初始化
    TopoMeta topoMeta{{{0}, {1}, {2}, {3}}};

    // 算子执行参数设置
    auto rankSize = 4;                                  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 500000;                            // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 0;                                      // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_server_two_two_test)
{
    // 仿真模型初始化
    // TopoMeta topoMeta {{{0, 1},{0, 1}}};
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 2);

    // 算子执行参数设置
    auto rankSize = 4;                                  // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;                               // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    auto root = 0;                                      // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_server_two_eight_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);

    // 算子执行参数设置
    auto rankSize = 16;                                 // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;                               // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16; // 数据类型
    auto root = 0;                                      // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_server_eight_eight_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 8, 8);

    // 算子执行参数设置
    auto rankSize = 64;                                // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;                              // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto root = 0;                                     // root节点
    auto dataTypeSize = sizeof(int8_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_diagonal_root_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 8);

    // 算子执行参数设置
    auto rankSize = 32;                                 // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;                               // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16; // 数据类型
    auto root = 31;                                     // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_random_root_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 8);

    // 算子执行参数设置
    auto rankSize = 24;                                // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 100;                              // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    auto root = 11;                                    // root节点
    auto dataTypeSize = sizeof(int16_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_one_count_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);

    // 算子执行参数设置
    auto rankSize = 16;                                // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 1;                                // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    auto root = 5;                                     // root节点
    auto dataTypeSize = sizeof(int32_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_a5_aicpu_Mesh1DNHR_bigdata_test)
{
    // 仿真模型初始化
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);

    // 算子执行参数设置
    auto rankSize = 16;                                // 参与集合通信的卡数(同topoMeta卡数一致)
    uint64_t count = 2 * 200 * 1024 * 1024 + 1;        // 数据量
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    auto root = 11;                                    // root节点
    auto dataTypeSize = sizeof(int8_t);
    RunBroadcastTest(topoMeta, rankSize, count, dataType, root, dataTypeSize);
}

// ==================== Broadcast RHD（DPU 二级，老共享内存接口） ====================
// 以下用例显式配置 HCCL_ALGO，并由 RunBroadcastTest 断言实际命中 RHD；
namespace {
constexpr const char* BROADCAST_RHD_DPU_2LEVEL = "broadcast:sequence{level0=meshconcur,level1=rhd}";
} // namespace

// DPU 二级 2 次幂：L0=2、L1=2；原地 Broadcast，覆盖高位 root。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_2level_power)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 2);
    RunBroadcastTest(
        topoMeta, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, 3, sizeof(int32_t), BROADCAST_RHD_DPU_2LEVEL);
}

// DPU 二级非 2 次幂：L1=3；count 不整除，多 loop，覆盖 part1 高位和 block root。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_2level_nonpower_tail_multiloop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 2);
    RunBroadcastTest(
        topoMeta, 6, MultiLoopCount(sizeof(int32_t)), HcclDataType::HCCL_DATA_TYPE_INT32, 2, sizeof(int32_t),
        BROADCAST_RHD_DPU_2LEVEL);
}

TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_2level_root_block)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 2);
    RunBroadcastTest(topoMeta, 6, 1025, HcclDataType::HCCL_DATA_TYPE_INT8, 4, sizeof(int8_t), BROADCAST_RHD_DPU_2LEVEL);
}

// DPU 二级非 2 次幂：L1=3，root=0 覆盖 part1 低位代表。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_2level_nonpower_root_low)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 2);
    RunBroadcastTest(
        topoMeta, 6, 1025, HcclDataType::HCCL_DATA_TYPE_INT32, 0, sizeof(int32_t), BROADCAST_RHD_DPU_2LEVEL);
}

// DPU 二级 2 次幂：root 低位且使用 8 字节数据类型。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_2level_power_root_low_int64)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 2);
    RunBroadcastTest(
        topoMeta, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT64, 0, sizeof(int64_t), BROADCAST_RHD_DPU_2LEVEL);
}

// ==================== Broadcast RHD（DPU 三级） ====================
namespace {
constexpr const char* BROADCAST_RHD_DPU_3LEVEL = "broadcast:sequence{level0=meshconcur,level1=rhd,level2=rhd}";
}

// 2 次幂三级拓扑：2x2x2；root 高位，原地单缓冲 Broadcast。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_3level_power_root_high)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    RunBroadcastTest(
        topoMeta, 8, 100, HcclDataType::HCCL_DATA_TYPE_INT32, 7, sizeof(int32_t), BROADCAST_RHD_DPU_3LEVEL, true);
}

// L1 非 2 次幂：3；count 不整除且超过单 loop 阈值，覆盖 part1 高位 root。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_3level_l1_nonpower_tail_multiloop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 2);
    RunBroadcastTest(
        topoMeta, 12, MultiLoopCount(sizeof(int32_t)), HcclDataType::HCCL_DATA_TYPE_INT32, 2, sizeof(int32_t),
        BROADCAST_RHD_DPU_3LEVEL, true);
}

// L1 非 2 次幂：root 位于 block（L2 高位），覆盖跨层 root 位置组合。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_3level_l1_nonpower_root_block)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 2);
    RunBroadcastTest(
        topoMeta, 12, 1025, HcclDataType::HCCL_DATA_TYPE_INT8, 10, sizeof(int8_t), BROADCAST_RHD_DPU_3LEVEL, true);
}

// L2 非 2 次幂：3；root 低位，覆盖 L2 part1 低位代表。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_3level_l2_nonpower_root_low)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 2);
    RunBroadcastTest(
        topoMeta, 12, 1025, HcclDataType::HCCL_DATA_TYPE_INT32, 0, sizeof(int32_t), BROADCAST_RHD_DPU_3LEVEL, true);
}

// L2 非 2 次幂：root 位于 block，覆盖后处理全量发送。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_3level_l2_nonpower_root_block)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 2);
    RunBroadcastTest(
        topoMeta, 12, 1025, HcclDataType::HCCL_DATA_TYPE_INT8, 8, sizeof(int8_t), BROADCAST_RHD_DPU_3LEVEL, true);
}

// count=1 覆盖最小消息和层内零切片路径。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_3level_count1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    RunBroadcastTest(
        topoMeta, 8, 1, HcclDataType::HCCL_DATA_TYPE_INT8, 5, sizeof(int8_t), BROADCAST_RHD_DPU_3LEVEL, true);
}

// 小消息且 root 低位，覆盖 BFP16 与 2 字节对齐。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_3level_small_bfp16_root_low)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    RunBroadcastTest(
        topoMeta, 8, 257, HcclDataType::HCCL_DATA_TYPE_BFP16, 0, sizeof(uint16_t), BROADCAST_RHD_DPU_3LEVEL, true);
}

// 大数据量触发多 loop；root 高位，继续验证原地单缓冲语义。
TEST_F(DPU_BROADCAST_TEST, st_broadcast_rhd_dpu_3level_multiloop_root_high)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 2);
    RunBroadcastTest(
        topoMeta, 12, MultiLoopCount(sizeof(int8_t)), HcclDataType::HCCL_DATA_TYPE_INT8, 11, sizeof(int8_t),
        BROADCAST_RHD_DPU_3LEVEL, true);
}
