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

class ST_REDUCE_SCATTER_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("HCCL_ALGO"); // RHD 系用例防御性清理（用例体内已自清理）
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

static void RunReduceScatterTest(
    const TopoMeta& topoMeta, u64 recvCount, HcclDataType dataType, HcclReduceOp reduceOp = HCCL_REDUCE_SUM,
    bool enableHostDpu = true)
{
    auto rankSize = 0;
    for (auto elem : topoMeta[0]) {
        rankSize += elem.size();
    }
    size_t dataUnitSize = 0;
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
            dataUnitSize = sizeof(int8_t);
            break;
        case HCCL_DATA_TYPE_INT16:
            dataUnitSize = sizeof(int16_t);
            break;
        case HCCL_DATA_TYPE_INT32:
            dataUnitSize = sizeof(int32_t);
            break;
        case HCCL_DATA_TYPE_INT64:
            dataUnitSize = sizeof(int64_t);
            break;
        case HCCL_DATA_TYPE_FP16:
            dataUnitSize = 2;
            break;
        case HCCL_DATA_TYPE_FP32:
            dataUnitSize = sizeof(float);
            break;
        case HCCL_DATA_TYPE_FP64:
            dataUnitSize = sizeof(double);
            break;
        case HCCL_DATA_TYPE_BFP16:
            dataUnitSize = 2;
            break;
        case HCCL_DATA_TYPE_UINT8:
            dataUnitSize = sizeof(uint8_t);
            break;
        default:
            dataUnitSize = 0;
            break;
    }

    SimWorld::Global()->Init(topoMeta, HcclDevType::DEV_TYPE_950);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    // enableHostDpu=false 时不置位——设备形态（AICPU 引擎全链路），
    // 覆盖 Aicpu 二级/三级 RHD 绑定的 device 版 InsTempReduceScatterRHD 模板
    if (enableHostDpu) {
        setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    } else {
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
    }
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

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
            u64 sendBufSize = recvCount * dataUnitSize * rankSize;
            u64 recvBufSize = recvCount * dataUnitSize;
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
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

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_001)
{
    RunReduceScatterTest(TopoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}}, 1, HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_002)
{
    RunReduceScatterTest(TopoMeta{{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}}, 1 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_003)
{
    RunReduceScatterTest(TopoMeta{{{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}}}, 1 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_004)
{
    RunReduceScatterTest(TopoMeta{{{0}, {0}, {0}, {0}}}, 301 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
}

TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_005)
{
    RunReduceScatterTest(TopoMeta{{{0, 1}, {0, 1}, {0, 1}, {0, 1}}}, 301 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
}

// RHD（递归折半倍增）显式配置二级形态（需求 5.6.1：reduce_scatter:sequence{level0=meshconcur,
// level1=rhd}）：host-DPU 形态（ENABLE_HOSTDPU_FOR_LLT 由 RunReduceScatterTest 统一设置）下
// 命中 DpuReduceScatterSequenceMeshConcurRHD，数据正确性由语义校验器断言；算法命中证据
// （--expect-algo 口径）由 VM checker rA/rD 系列承载，本用例提供 CI 级数据面回归。
// rhd_001：d1=6 非 2 次幂——层内前/后处理生效（RHD 区别于 HD 的目标场景）。
TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_rhd_001)
{
    setenv("HCCL_ALGO", "reducescatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunReduceScatterTest(TopoMeta{{{0}, {0}, {0}, {0}, {0}, {0}}}, 1 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
    unsetenv("HCCL_ALGO");
}

// rhd_002：d1=4 2 次幂——纯主体路径（PRE/POST 空步骤分支）。
TEST_F(ST_REDUCE_SCATTER_TEST, test_host_dpu_reducescatter_rhd_002)
{
    setenv("HCCL_ALGO", "reducescatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunReduceScatterTest(TopoMeta{{{0}, {0}, {0}, {0}}}, 1 * 1024 * 1024, HCCL_DATA_TYPE_FP32);
    unsetenv("HCCL_ALGO");
}

// 设备形态（AICPU 引擎全链路，ENABLE_HOSTDPU_FOR_LLT 不置位）：
// 命中 AicpuReduceScatterSequenceMeshConcurRHD（device 版 InsTempReduceScatterRHD 五段管线，
// σ 槽位寻址 + write-reduce 边传边归约）。算法命中证据由 VM checker rA 系列承载，
// 本用例提供 CI 级设备侧数据面回归。
// rhd_dev_001：d1=6 非 2 次幂——PRE/POST 前后处理 + σ 布局生效。
TEST_F(ST_REDUCE_SCATTER_TEST, test_device_reducescatter_rhd_dev_001)
{
    setenv("HCCL_ALGO", "reducescatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunReduceScatterTest(
        TopoMeta{{{0}, {0}, {0}, {0}, {0}, {0}}}, 1 * 1024 * 1024, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, false);
    unsetenv("HCCL_ALGO");
}

// rhd_dev_002：d1=4 2 次幂——纯主体路径。
TEST_F(ST_REDUCE_SCATTER_TEST, test_device_reducescatter_rhd_dev_002)
{
    setenv("HCCL_ALGO", "reducescatter:sequence{level0=meshconcur,level1=rhd}", 1);
    RunReduceScatterTest(TopoMeta{{{0}, {0}, {0}, {0}}}, 1 * 1024 * 1024, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, false);
    unsetenv("HCCL_ALGO");
}

// [CI run#5334/#5355] 三级用例（dev_003）在 ST 框架建链超时（HcclGetChannel hcclRet=9）——
// ST SimWorld 不支持多 pod 拓扑（全部既有 ST 用例均为单 pod），三级通道建立不完整。
// 三级覆盖由 VM checker rA3~rA6（12+ 例，2幂/非幂/双非幂全维度）承载，ST 不重复。
