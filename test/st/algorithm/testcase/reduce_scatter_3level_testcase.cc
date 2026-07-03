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
#include "check_utils.h"
#include "hccl_verifier.h"
#include <thread>
#include "alg_env_config.h"

using namespace HcclSim;
using namespace ops_hccl;

constexpr uint32_t DATATYPE_SIZE_TABLE_REDUCE_SCATTER[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_REDUCE_SCATTER_3LEVEL_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
    }
    void TearDown() override
    {
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_OP_EXPANSION_MODE");

    }
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
};

void RunReduceScatter3LevelA5(const TopoMeta &topoMeta, const u64 &recvCount, const HcclDataType &dataType,
    const HcclReduceOp &reduceOp)
{
    SimWorld::Global()->Init(topoMeta, DevType::DEV_TYPE_950);

    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);

    auto rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_REDUCE_SCATTER[dataType];
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);

            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void *recvBuf = nullptr;
            void *sendBuf = nullptr;
            u64 sendBufSize = recvCount * dataTypeSize * rankSize;
            u64 recvBufSize = recvCount * dataTypeSize;

            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));

            CHK_RET(HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, reduceOp, comm, stream));

            CHK_RET(HcclCommDestroy(comm));
            return HCCL_SUCCESS;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckReduceScatter(taskQueues, rankSize, dataType, recvCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    SimWorld::Global()->Deinit();
}

// P0: #1 - 3-level basic correctness on 128-card topology
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_8x8x2_fp32_sum_basic)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P0: #3 - outputRepeatStride>0, repeatNum=L2=3 verification
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_4x4x3_fp32_sum_repeatnum_gt1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 4, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P0: #16 - backward compatibility, 2-level behavior unchanged
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_2level_backward_compat_meshnhr)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P1: #5 - different topology scale correctness
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_4x4x2_int32_max_different_scale)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 500;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P1: #10 - small-scale large-data loop segmentation
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_4x2x2_fp32_sum_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    auto recvCount = 500 * 1024 * 1024;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P1: #11 - small cluster, boundary rank verification, recvCount=200+1
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_8x2x2_fp32_sum_small_cluster_recv200_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 200 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P2: #4 - higher repeatNum (repeatNum=L2=4)
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_4x2x4_int32_sum_repeatnum4)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P2: #8 - BFP16 data type on 16-card topology
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_4x2x2_bfp16_max_dtype)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    auto recvCount = 300;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}


// P2: #14 - Level2 has 3 clusters (repeatNum=3)
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_8x2x3_fp32_sum_level2_3cluster)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// P2: #15 - fully asymmetric dimensions
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_4x3x2_int32_sum_asymmetric_all)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// --- Degenerate Level (dimension=1) edge cases ---

// L1=1: single server per pod, 8x1x3=24 ranks, degenerate L1, MIN op
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_8x1x3_fp32_min_l1_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// L1=1: degenerate L1 + recvCount=8+1, just over aligned boundary
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_4x1x2_int8_sum_l1_degenerate_recv8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    auto recvCount = 8 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// L0=1 + L1=1: double degenerate, 1x1x4=4 ranks, recvCount=16+1
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_1x1x4_fp32_sum_double_degenerate_recv16_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 1, 1);
    auto recvCount = 16 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// --- Strange / weird recvCount = aligned_value + 1 cases ---

// recvCount=4+1=5: just over power-of-2, tests remainder element in stride slicing
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_4x3x2_fp32_min_recv4_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    auto recvCount = 4 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// recvCount=64K+1=65537: just over 64K boundary, loop slicing remainder on 32-card
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_4x4x2_int16_max_recv64k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    auto recvCount = 64 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}

// L0=1 + recvCount=128K+1: degenerate L0 + large data with remainder element
TEST_F(ST_REDUCE_SCATTER_3LEVEL_TEST, st_reduce_scatter_3level_1x4x2_fp32_sum_l0_degenerate_recv128k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    auto recvCount = 128 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunReduceScatter3LevelA5(topoMeta, recvCount, dataType, reduceOp);
}