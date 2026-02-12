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

constexpr uint32_t DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_SCATTER_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

uint64_t CountElements(const TopoMeta &topoMeta) {
    uint64_t total = 0;
    for (const auto& vec2d : topoMeta) {
        for (const auto& vec1d : vec2d) {
            // 直接加上最内层 vector 的大小
            total += vec1d.size();
        }
    }
    return total;
}

void RunScatterTest(int root, TopoMeta &topoMeta, int dataCount, HcclDataType dataType) 
{
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_910_95);
    
    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    // 算子执行参数设置
    auto rankSize = CountElements(topoMeta);  // 参与集合通信的卡数(同topoMeta卡数一致)
    auto recvCount = dataCount;  // 接收数据量
    // auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;  // 数据类型

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
            u64 sendDataSize = recvCount * DATATYPE_SIZE_TABLE[dataType] * rankSize;
            u64 recvDataSize = recvCount * DATATYPE_SIZE_TABLE[dataType];
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendDataSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvDataSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 4.算子下发
            CHK_RET(HcclScatter(sendBuf, recvBuf, recvCount, dataType, root, comm, stream));

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
    HcclResult res = CheckScatter(taskQueues, rankSize, dataType, recvCount, root);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit(); 
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh_1d_success_1x3_root0_int32_small_data)
{
    TopoMeta topoMeta {{{0, 1, 2}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh_1d_success_1x4_root0_uint64_small_data)
{
    TopoMeta topoMeta {{{0, 1, 2, 3}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_UINT64);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh_1d_success_1x6_root3_int16_small_data)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5}}};
    RunScatterTest(3, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_INT16);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh_1d_success_1x8_root5_fp32_big_data)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5, 6, 7}}};
    RunScatterTest(5, topoMeta, 400 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh_1d_success_1x6_root1_int8_big_data)
{
    TopoMeta topoMeta {{{0, 1, 2, 3, 4, 5}}};
    RunScatterTest(1, topoMeta, 1024 * 1024 * 1024 + 1, HcclDataType::HCCL_DATA_TYPE_INT8);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh_1d_success_1x2_root0_fp16_small_data)
{
    TopoMeta topoMeta {{{0, 1}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_FP16);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh1dnhr_success_2x2_root0_int32_small_data)
{   
    TopoMeta topoMeta {{{0,1}, {8,9}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh1dnhr_success_3x2_root0_uint64_small_data)
{   
    TopoMeta topoMeta {{{0,1}, {8,9}, {16, 17}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_UINT64);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh1dnhr_success_3x3_root3_int16_small_data)
{   
    TopoMeta topoMeta {{{0,1,2}, {8,9,10}, {16,17,18}}};
    RunScatterTest(3, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_INT16);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh1dnhr_success_4x4_root5_fp32_big_data)
{   
    TopoMeta topoMeta {{{0,1,2,3}, {8,9,10,11}, {16,17,18,19}}};
    RunScatterTest(5, topoMeta, 400 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh1dnhr_success_2x8_root1_int8_big_data)
{   
    TopoMeta topoMeta {{{0,1,2,3,4,5,6,7}, {8,9,10,11,12,13,14,15}}};
    RunScatterTest(1, topoMeta, 400 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_INT8);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_mesh1dnhr_success_4x2_root0_fp16_small_data)
{   
    TopoMeta topoMeta {{{0,1}, {8,9}, {16, 17}, {24,25}}};
    RunScatterTest(1, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_FP16);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_nhr_success_1x3_root0_int32_small_data)
{   
    TopoMeta topoMeta {{{0}, {0}, {0}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_nhr_success_1x4_root0_uint64_small_data)
{   
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}}};
    RunScatterTest(0, topoMeta, 105, HcclDataType::HCCL_DATA_TYPE_UINT64);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_nhr_success_1x6_root3_int16_small_data)
{   
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}}};
    RunScatterTest(3, topoMeta, 400 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_INT16);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_nhr_success_1x8_root5_fp64_big_data)
{   
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}};
    RunScatterTest(5, topoMeta, 400 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_FP64);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_nhr_success_1x6_root1_int8_big_data)
{   
    TopoMeta topoMeta {{{0}, {0}, {0}, {0}, {0}, {0}}};
    RunScatterTest(1, topoMeta, 451 * 1024 * 1024, HcclDataType::HCCL_DATA_TYPE_INT8);
}

TEST_F(ST_SCATTER_TEST, test_aicpu_scatter_nhr_success_1x2_root0_fp16_small_data)
{   
    TopoMeta topoMeta {{{0}, {0}}};
    RunScatterTest(0, topoMeta, 100, HcclDataType::HCCL_DATA_TYPE_FP16);
}