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

constexpr uint32_t DATATYPE_SIZE_TABLE_AR[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

class ST_ALL_REDUCE_MULTILEVEL_TEST : public ::testing::Test {
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

void RunAllReduceMultiLevelCase(const TopoMeta &topoInfo, const u64 dataCount,
    const HcclDataType dataType, const HcclReduceOp reduceOp)
{
    // 仿真模型初始化
    SimWorld::Global()->Init(topoInfo, DevType::DEV_TYPE_950);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_AR[dataType];

    // 设置展开模式为HOST_TS
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    // 算子执行参数设置
    u32 rankSize = 0;
    for (const auto &superPod : topoInfo) {
        for (const auto &podIdx : superPod) {
            rankSize += podIdx.size();
        }
    }

    // 多线程运行ALL REDUCE
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
            u64 sendBufSize = dataCount * dataTypeSize;  // 数据量转化为字节数
            u64 recvBufSize = dataCount * dataTypeSize;
            // 打桩实现，仿真运行需标记内存是INPUT和OUTPUT
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));

            // 4.算子下发
            CHK_RET(HcclAllReduce(sendBuf, recvBuf, dataCount, dataType, reduceOp, comm, stream));

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
    HcclResult res = CheckAllReduce(taskQueues, rankSize, dataType, dataCount, reduceOp);
    EXPECT_TRUE(res == HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

// P0: most basic case
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_test_01)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 dataCount = 18;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P0: basic correctness on 128-rank topology
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_8x8x2_fp32_sum_basic)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 8, 8);
    u64 dataCount = 200;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P0: outputRepeatStride>0, repeatNum=L2=3 verification
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x4x3_fp32_sum_repeatnum_gt1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 4, 4);
    u64 dataCount = 200;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P1: different topology scale correctness with MAX op
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x4x2_int32_max_different_scale)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 dataCount = 500;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P1: MIN op on 3-level topology
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x4x2_fp32_min_op)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 dataCount = 500;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P1: small cluster, boundary rank verification, dataCount=200+1
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_8x2x2_fp32_sum_small_cluster_data200_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    u64 dataCount = 200 + 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P1: small-scale large-data loop segmentation
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x4x2_fp32_sum_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    u64 dataCount = 500 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P2: higher repeatNum (repeatNum=L2=4)
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x2x4_int32_sum_repeatnum4)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 4);
    u64 dataCount = 200;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P2: BFP16 data type on 16-rank topology
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x2x2_bfp16_max_dtype)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    u64 dataCount = 300;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P2: FP16 data type on 32-rank topology
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x4x2_fp16_sum_dtype)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 dataCount = 500 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P2: INT8 small data on 3-level topology
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x2x2_int8_sum_corner)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    u64 dataCount = 100;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P2: Level2 has 3 clusters (repeatNum=3)
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_8x2x3_fp32_sum_level2_3cluster)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 8);
    u64 dataCount = 200;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// P2: fully asymmetric dimensions
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x3x2_int32_sum_asymmetric_all)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    u64 dataCount = 200;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}


// --- Degenerate Level (dimension=1) edge cases ---
//
// L1=1: single server per pod, 8x1x3=24 ranks, degenerate L1, MIN op
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_8x1x3_fp32_min_l1_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 1, 8);
    u64 dataCount = 200;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// L1=1: degenerate L1 + dataCount=8+1, just over aligned boundary
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x1x2_int8_sum_l1_degenerate_data8_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    u64 dataCount = 8 + 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// L0=1 + L1=1: double degenerate, 1x1x4=4 ranks, dataCount=16+1
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_1x1x4_fp32_sum_double_degenerate_data16_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 1, 1);
    u64 dataCount = 16 + 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// L0=1: degenerate L0, 2x4x1=8 ranks
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_1x4x2_fp32_sum_l0_degenerate)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    u64 dataCount = 101;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// dataCount=4+1=5: just over power-of-2, tests remainder element in stride slicing
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x3x2_fp32_min_data4_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    u64 dataCount = 4 + 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MIN;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// dataCount=64K+1=65537: just over 64K boundary, loop slicing remainder on 32-rank
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x4x2_int16_max_data64k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 dataCount = 64 * 1024 + 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// L0=1 + dataCount=128K+1: degenerate L0 + large data with remainder element
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_1x4x2_fp32_sum_l0_degenerate_data128k_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    u64 dataCount = 128 * 1024 + 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// dataCount=1: single element on 3-level topology
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x2x2_int32_sum_single_element)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    u64 dataCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// tailSize overflow: dataCount=40M-1, single loop, tailSize exceeds rsResultBuffSize_ by 8 bytes
// maxCountPerLoop=40M (meshCommBuffSize_/dtSize/totalRankAlign*totalRankAlign)
// q=10485759, r=3, tailSize=(q+r)*4=41943048 > rsResultBuffSize_=41943040
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x2x2_fp32_sum_tailsize_overflow_single_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    u64 dataCount = 40 * 1024 * 1024 - 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}

// tailSize overflow: dataCount=80M-1, multi-loop, last loop tailSize exceeds rsResultBuffSize_
// Loop1: currDataCount=40M (aligned), Loop2: currDataCount=40M-1 (overflow, trimmed to 40M-6)
TEST_F(ST_ALL_REDUCE_MULTILEVEL_TEST, st_all_reduce_3level_4x2x2_fp32_sum_tailsize_overflow_multi_loop)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 4);
    u64 dataCount = 80 * 1024 * 1024 - 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    RunAllReduceMultiLevelCase(topoMeta, dataCount, dataType, reduceOp);
}
