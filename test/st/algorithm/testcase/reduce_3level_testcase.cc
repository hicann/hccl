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

constexpr uint32_t DATATYPE_SIZE_TABLE_REDUCE_3LEVEL[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_REDUCE_3LEVEL_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_INDEPENDENT_OP");
    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

void RunReduce3LevelA5(const TopoMeta &topoMeta, const u64 &recvCount, const HcclDataType &dataType,
    const HcclReduceOp &reduceOp, const uint32_t root)
{
    SimWorld::Global()->Init(topoMeta, HcclDevType::DEV_TYPE_950);

    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    auto rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_REDUCE_3LEVEL[dataType];
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
            u64 sendBufSize = recvCount * dataTypeSize * rankSize;
            u64 recvBufSize = recvCount * dataTypeSize;
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

// P0: #1 - 3-level basic correctness on 128-card topology, root=0
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x8x8_fp32_sum_basic_root0)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// P0: #2 - 3-level basic correctness, root=middle rank
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x4x4_fp32_sum_root_mid)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 15;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// P0: #3 - 3-level basic correctness, root=last rank
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x2x8_int32_sum_root_last)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 31;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// P0: #4 - backward compatibility, 2-level behavior unchanged (L2=1)
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_2level_backward_compat_1x2x8_int32_sum)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// P1: #5 - different reduce ops (MAX) on different topology scale
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x4x4_int32_max_different_scale)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 500;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// P1: #6 - different reduce ops (MIN) with BFP16
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x2x4_bfp16_min_dtype)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto recvCount = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    uint32_t root = 7;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// P1: #7 - small-scale large-data loop segmentation
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x2x2_fp32_sum_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto recvCount = 500 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// P1: #8 - small cluster, boundary rank verification, recvCount=200+1
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x2x8_fp32_sum_small_cluster_recv200_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// P2: #9 - Level2 has 3 clusters (L2=3)
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_3x2x8_fp32_sum_level2_3cluster)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// P2: #10 - fully asymmetric dimensions
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x3x4_int32_sum_asymmetric_all)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 12;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// --- Degenerate Level (dimension=1) edge cases ---

// L1=1: single server per pod, degenerate L1, MIN op
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_3x1x8_fp32_min_l1_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// L1=1: degenerate L1 + recvCount=8+1, just over aligned boundary
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x1x4_int8_sum_l1_degenerate_recv8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    auto recvCount = 8 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// L0=1 + L1=1: double degenerate, 4x1x1=4 ranks, recvCount=16+1
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_4x1x1_fp32_sum_double_degenerate_recv16_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 1, 1);
    auto recvCount = 16 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// L0=1: degenerate L0 + recvCount=128K+1, large data with remainder element
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x4x1_fp32_sum_l0_degenerate_recv128k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    auto recvCount = 128 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// --- Strange / weird recvCount = aligned_value + 1 cases ---

// recvCount=4+1=5: just over power-of-2, tests remainder element in stride slicing
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x3x4_fp32_min_recv4_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 4 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// recvCount=64K+1=65537: just over 64K boundary, loop slicing remainder on 32-card
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x4x4_int16_max_recv64k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 64 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// recvCount=1M+73: large data triggering multiple loops
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x2x2_fp32_sum_recv1m_plus_73)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto recvCount = 1 * 1024 * 1024 + 73;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 3;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// recvCount=4M+1: very large data on 32-card topology
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_4x3x2_bfp16_sum_recv4m_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 3, 2);
    auto recvCount = 4 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// recvCount=64M+1: stress test large data multi-loop
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x3x4_int32_sum_recv64m_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 64 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// recvCount=1: minimal data, single element
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x2x2_int32_sum_recv1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto recvCount = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 5;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// INT8 + SUM on 128-card topology, small count
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x8x8_int8_sum_recv262)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    auto recvCount = 262;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// 3-level with 3x2x3 topology, MAX op
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_3x2x3_fp32_max_recv200)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 3);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    uint32_t root = 8;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// 4-level degenerate to 2-level (L2=1, L1=1), single server
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_1x1x8_int32_sum_single_server)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 3;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// 100K+8 on 32-card BFP16, tests tail block handling
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x2x8_bfp16_sum_recv100k_plus_8)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 100 * 1024 + 8;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}

// 300K+3 on 128-card BFP16, tests large loop with tail
TEST_F(ST_REDUCE_3LEVEL_TEST, st_reduce_3level_2x8x8_bfp16_sum_recv300k_plus_3)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    auto recvCount = 300 * 1024 + 3;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduce3LevelA5(topoMeta, recvCount, dataType, reduceOp, root);
}
