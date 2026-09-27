/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Reduce Sequence 执行器（AicpuReduceSequenceMeshConcurNHRNHR）定向用例：
 * 通过 HCCL_ALGO 强制选中该执行器，固化 root AGL0 直写用户 output 的行为：
 * 1) root 位于组内首位/末位 rank（末位覆盖 tailSize 尾块直写布局）；
 * 2) 数据量超过单 loop 容量（多 loop），验证 processedDataCount 累进偏移下各段 output 连续正确；
 * 3) 非 root rank 走原 HCCL_BUFFER 路径，由 CheckReduce 校验 root 输出结果兜底。
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

constexpr uint32_t DATATYPE_SIZE_TABLE_REDUCE_SEQ[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(int8_t),
       sizeof(int16_t),
       sizeof(int32_t),
       2,
       sizeof(float),
       sizeof(int64_t),
       sizeof(uint64_t),
       sizeof(uint8_t),
       sizeof(uint16_t),
       sizeof(uint32_t),
       8,
       2,
       16,
       2,
       1,
       1,
       1,
       1};

class ST_REDUCE_SEQUENCE_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        // 强制命中 AicpuReduceSequenceMeshConcurNHRNHR，不依赖 selector 自动选择
        setenv("HCCL_ALGO", "sequence{meshconcur,nhr,nhr}", 1);
    }
    void TearDown() override
    {
        unsetenv("HCCL_ALGO");
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

void RunReduceSequence(
    const TopoMeta& topoMeta, const u64 recvCount, const HcclDataType dataType, const HcclReduceOp reduceOp,
    const uint32_t root)
{
    SimWorld::Global()->Init(topoMeta, HcclDevType::DEV_TYPE_950);

    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    auto rankSize = CalRankSize(topoMeta);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_REDUCE_SEQ[dataType];
    std::vector<std::thread> threads;
    for (auto rankId = 0; rankId < rankSize; ++rankId) {
        threads.emplace_back([=]() {
            aclrtSetDevice(rankId);

            aclrtStream stream = nullptr;
            aclrtCreateStream(&stream);

            HcclComm comm = nullptr;
            CHK_RET(HcclCommInitClusterInfo("./ranktable.json", rankId, &comm));

            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
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

// P0: root=0（组内首位 rank），基础直写正确性
TEST_F(ST_REDUCE_SEQUENCE_TEST, st_reduce_seq_2x2x8_int32_sum_root_first)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 200;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    RunReduceSequence(topoMeta, recvCount, dataType, reduceOp, root);
}

// P0: root=末位 rank，其 AGL0 切片为尾块（tailSize 直写布局），叠加非对齐余量
TEST_F(ST_REDUCE_SEQUENCE_TEST, st_reduce_seq_2x2x8_int32_sum_root_last_recv100k_plus_13)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 100 * 1024 + 13;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 31;
    RunReduceSequence(topoMeta, recvCount, dataType, reduceOp, root);
}

// P0: root=中间 rank，大数据多 loop（processedDataCount 累进偏移）
TEST_F(ST_REDUCE_SEQUENCE_TEST, st_reduce_seq_2x2x8_int32_sum_root_mid_multiloop_recv64m_plus_1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 64 * 1024 * 1024 + 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 15;
    RunReduceSequence(topoMeta, recvCount, dataType, reduceOp, root);
}

// P1: fp16 最小数据量 + root 既非首位也非末位
TEST_F(ST_REDUCE_SEQUENCE_TEST, st_reduce_seq_2x2x8_fp16_sum_recv1_root_mid)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 1;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 17;
    RunReduceSequence(topoMeta, recvCount, dataType, reduceOp, root);
}

// P1: 非对齐小数据 + MAX 操作（root=0），覆盖余量切片下的直写布局
TEST_F(ST_REDUCE_SEQUENCE_TEST, st_reduce_seq_2x2x8_int32_max_recv200_plus_7)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 8);
    auto recvCount = 200 + 7;
    auto dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    auto reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;
    uint32_t root = 0;
    RunReduceSequence(topoMeta, recvCount, dataType, reduceOp, root);
}
