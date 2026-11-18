/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include "gtest/gtest.h"
#include "sim_world.h"
#include "hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl_rt.h"
#include "hccl_verifier.h"
#include "check_utils.h"
#include "alg_env_config.h"
#include "hccl_sim_pub.h"

using namespace HcclSim;
using namespace ops_hccl;

// 与 all_reduce_testcase 的 1shot 边界用例保持一致的仿真配置
constexpr u64 OL_CAPTURE_1D_DATA_SIZE = 8 * 1024 * 1024;

// 拓扑固定为 2 rank（{{{0, 1}}}），入图计数断言值由该拓扑推导
constexpr u32 OL_CAPTURE_RANK_SIZE = 2;

class ST_ORDER_LAUNCH_CAPTURE_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_STUB_CAPTURE_ACTIVE");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// AICPU allreduce 1shot；captureActive=true 时桩回填 ACTIVE，走 capture 链路
static void RunAllReduceCaptureCase(const bool captureActive)
{
    TopoMeta topoMeta{{{0, 1}}};
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
    u32 dataTypeSize = 2;
    u64 dataCount = OL_CAPTURE_1D_DATA_SIZE / dataTypeSize;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_MAX;

    // 仿真模型初始化
    SimWorld::Global()->Init(topoMeta, HcclDevType::DEV_TYPE_950);

    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
    if (captureActive) {
        setenv("HCCL_STUB_CAPTURE_ACTIVE", "1", 1);
    } else {
        unsetenv("HCCL_STUB_CAPTURE_ACTIVE");
    }

    std::vector<std::thread> threads;
    for (u32 rankId = 0; rankId < OL_CAPTURE_RANK_SIZE; ++rankId) {
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
            u64 sendBufSize = dataCount * dataTypeSize;
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

    for (auto& thread : threads) {
        thread.join();
    }

    // 结果成图校验
    auto taskQueues = SimTaskQueue::Global()->GetAllRankTaskQueues();
    HcclResult res = CheckAllReduce(taskQueues, OL_CAPTURE_RANK_SIZE, dataType, dataCount, reduceOp);
    EXPECT_EQ(res, HCCL_SUCCESS);

    // 资源清理
    SimWorld::Global()->Deinit();
}

// capture NONE：rtStreamAddToModel 零调用（回归锚点）
TEST_F(ST_ORDER_LAUNCH_CAPTURE_TEST, st_order_launch_capture_none_regression)
{
    uint32_t before = AclStubGetRtStreamAddToModelCallCount();
    RunAllReduceCaptureCase(false);
    EXPECT_EQ(AclStubGetRtStreamAddToModelCallCount(), before);
}

// capture ACTIVE：占位流恰每 rank 入图一次（== rankSize），CheckAllReduce 校验数据语义；
// 保序链路 ST 不可达（专用线程桩为 0），回归保护见 test/ut/order_launch
TEST_F(ST_ORDER_LAUNCH_CAPTURE_TEST, st_order_launch_capture_active_addstream)
{
    uint32_t before = AclStubGetRtStreamAddToModelCallCount();
    RunAllReduceCaptureCase(true);
    uint32_t addCount = AclStubGetRtStreamAddToModelCallCount() - before;
    EXPECT_EQ(addCount, OL_CAPTURE_RANK_SIZE);
}
