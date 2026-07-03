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

constexpr uint32_t DATATYPE_SIZE_TABLE_ALL_GATHER_3LEVEL[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_ALL_GATHER_3LEVEL_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

void RunAllGather3LevelA5(const TopoMeta &topoMeta, const u64 &sendCount, const HcclDataType &dataType)
{
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    auto rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_ALL_GATHER_3LEVEL[dataType];
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);

            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void *sendBuf = nullptr;
            void *recvBuf = nullptr;
            u64 sendBufSize = sendCount * dataTypeSize;
            u64 recvBufSize = sendCount * dataTypeSize * rankSize;
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            CHK_RET(HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream));

            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllGather(taskQueues, rankSize, dataType, sendCount);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x8x8_int8_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 8, 8);
    auto sendCount = 3 * 1024 + 6;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x2x2_fp32_2_)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto sendCount = 1 * 1024 * 1024 + 73;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}


TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_1x2x8_int32_backward_compat)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x1x4_int8_different_scale)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    auto sendCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}


TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x1x1_fp32_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 1, 1);
    auto sendCount = 262;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x4x1_fp32_send128k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    auto sendCount = 128 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x4x4_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto sendCount = 500;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// L1=1: degenerate L1 + sendCount=8+1
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x3x4_fp32_l1_degenerate_send243_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto sendCount = 200 + 43;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// L0=1 + L1=1: double degenerate, 1x1x4=4 ranks, sendCount=16+1
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x2x8_fp32_send201_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto sendCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x2x2_int32_send200)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 2);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x2x4_bfp16_send300)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto sendCount = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_3x2x3_fp32_send200)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 3);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

// P2: #4 - higher repeatNum (repeatNum=L2=4)
TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x3x4_int32_repeatnum4)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x3x2_bfp16_send4m_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 3, 2);
    auto sendCount = 4 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_3x1x8_fp32_send200)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 8);
    auto sendCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x3x4_int32_send64m_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto sendCount = 64 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x2x2_int32_send128m_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto sendCount = 7 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_4x1x1_fp32_send326m_plus_671)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 1, 1);
    auto sendCount = 326 * 1024 * 1024 + 671;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_8x1x1_fp32_send326m_plus_671)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 8, 1, 1);
    auto sendCount = 326 * 1024 * 1024 + 671;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_8x1x1_bfp16_send326m_plus_27)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 8, 1, 1);
    auto sendCount = 326 * 1024 * 1024 + 27;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_3x1x5_bfp16_send356k_plus_43)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 5);
    auto sendCount = 356 * 1024 + 43;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x8x8_bfp16_send300k_plus_3)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    auto sendCount = 300 * 1024 + 3;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_3LEVEL_TEST, st_allgather_3level_2x2x8_bfp16_send100k_plus_8)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto sendCount = 100 * 1024 + 8;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGather3LevelA5(topoMeta, sendCount, dataType);
}