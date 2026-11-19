/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "v_testcase_common.h"
#include "gtest/gtest.h"
#include <algorithm>

#include "alg_env_config.h"

constexpr u32 DATATYPE_SIZE_TABLE_AGV[HCCL_DATA_TYPE_RESERVED]
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

class ST_ALL_GATHER_V_TEST : public ::testing::Test {
protected:
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    static void TearDownTestCase() {}
    static void SetUpTestCase() {}
};

static HcclResult
AllGatherVDispatch(u32 rankId, u64 totalCount, VDataDesTag vDataDes, HcclComm comm, aclrtStream stream)
{
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_AGV[vDataDes.dataType];
    u64 sendBufSize = vDataDes.counts[rankId] * dataTypeSize;
    u64 recvBufSize = totalCount * dataTypeSize;
    aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
    aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
    return HcclAllGatherV(
        sendBuf, vDataDes.counts[rankId], recvBuf, vDataDes.counts.data(), vDataDes.displs.data(), vDataDes.dataType,
        comm, stream);
}

// displs 允许空洞：recvBuf 按 max(displs+counts) 分配。
static HcclResult
AllGatherVDispatchWithDispls(u32 rankId, u64 totalCount, VDataDesTag vDataDes, HcclComm comm, aclrtStream stream)
{
    (void)totalCount;
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_AGV[vDataDes.dataType];
    u64 sendBufSize = vDataDes.counts[rankId] * dataTypeSize;
    u64 recvBufSize = 0;
    for (u32 i = 0; i < vDataDes.counts.size(); i++) {
        recvBufSize = std::max(recvBufSize, (vDataDes.displs[i] + vDataDes.counts[i]) * dataTypeSize);
    }
    if (sendBufSize > 0
        && aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK)) != ACL_SUCCESS) {
        return HCCL_E_INTERNAL;
    }
    if (recvBufSize > 0
        && aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK)) != ACL_SUCCESS) {
        (void)aclrtFree(sendBuf);
        return HCCL_E_INTERNAL;
    }
    return HcclAllGatherV(
        sendBuf, vDataDes.counts[rankId], recvBuf, vDataDes.counts.data(), vDataDes.displs.data(), vDataDes.dataType,
        comm, stream);
}

// 单 rank 直接调用，用于断言入参错误码（in-place / displs 非法）。
static HcclResult RunAllGatherVWithBuffers(const TopoMeta& topoInfo, VDataDesTag vDataDes, bool inplace)
{
    SimWorld::Global()->Init(topoInfo, HcclDevType::DEV_TYPE_950);
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    HcclComm comm = nullptr;
    HcclResult ret = HcclCommInitClusterInfo("./ranktable.json", 0, &comm);
    if (ret != HCCL_SUCCESS) {
        (void)aclrtDestroyStream(stream);
        SimWorld::Global()->Deinit();
        return ret;
    }

    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_AGV[vDataDes.dataType];
    u64 sendSize = vDataDes.counts[0] * dataTypeSize;
    u64 recvSize = 0;
    for (u32 i = 0; i < vDataDes.counts.size(); i++) {
        recvSize = std::max(recvSize, (vDataDes.displs[i] + vDataDes.counts[i]) * dataTypeSize);
    }
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    if (sendSize > 0
        && aclrtMalloc(&sendBuf, sendSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK)) != ACL_SUCCESS) {
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        SimWorld::Global()->Deinit();
        return HCCL_E_INTERNAL;
    }
    if (!inplace && recvSize > 0
        && aclrtMalloc(&recvBuf, recvSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK)) != ACL_SUCCESS) {
        (void)aclrtFree(sendBuf);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        SimWorld::Global()->Deinit();
        return HCCL_E_INTERNAL;
    }
    if (inplace) {
        recvBuf = sendBuf;
    }

    ret = HcclAllGatherV(
        sendBuf, vDataDes.counts[0], recvBuf, vDataDes.counts.data(), vDataDes.displs.data(), vDataDes.dataType, comm,
        stream);
    (void)HcclCommDestroy(comm);
    if (sendBuf != nullptr) {
        (void)aclrtFree(sendBuf);
    }
    if (recvBuf != nullptr && recvBuf != sendBuf) {
        (void)aclrtFree(recvBuf);
    }
    (void)aclrtDestroyStream(stream);
    SimWorld::Global()->Deinit();
    return ret;
}

static void RunAllGatherVMultilevel(const TopoMeta& topoInfo, VDataDesTag vDataDes)
{
    RunVMultilevelTest(topoInfo, vDataDes, nullptr, AllGatherVDispatch, CheckAllGatherV);
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_aicpu_test)
{
    TopoMeta topoMeta{{{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {89478485, 178956970};
    vDataDes.displs = {0, 89478485};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_multilevel_2pod_4rank_int32_equal_test)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_multilevel_2pod_6rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0, 1, 2}, {3, 4, 5}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {200, 200, 200, 200, 200, 200};
    vDataDes.displs = {0, 200, 400, 600, 800, 1000};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

// 3-pod topologies unsupported: simulator has only 2 net layers (no inter-superpod links at layer 2).
// Unequal AllGatherV unsupported: mesh1D slave streams end with WAIT not LOCAL_POST_TO (pre-existing bug).

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_3layer_2pod_2server_8rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}, {{0, 1}, {0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {200, 200, 200, 200, 200, 200, 200, 200};
    vDataDes.displs = {0, 200, 400, 600, 800, 1000, 1200, 1400};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_3layer_2pod_1server_4rank_int32_equal_test)
{
    TopoMeta topoMeta{{{0, 1}}, {{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_3layer_2pod_1server_8rank_fp16_equal_test)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}}, {{0, 1, 2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100, 100, 100, 100, 100, 100, 100};
    vDataDes.displs = {0, 100, 200, 300, 400, 500, 600, 700};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

// asymmetric topology
TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_a5_asymmetric_fp32_test)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1, 2}}, {{0, 1, 2, 3}, {0, 1, 2, 3, 4}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {700, 700, 600, 600, 500, 500, 400, 400, 300, 300, 200, 200, 100, 100};
    vDataDes.displs = {0, 700, 1400, 2000, 2600, 3100, 3600, 4000, 4400, 4700, 5000, 5200, 5400, 5500};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunAllGatherVMultilevel(topoMeta, vDataDes);
}

// AGV 入参/布局专项：displs 空洞。
TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_displs_hole_test)
{
    TopoMeta topoMeta{{{0, 1, 2}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {2, 3, 1};
    vDataDes.displs = {0, 5, 8};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunVMultilevelTest(topoMeta, vDataDes, nullptr, AllGatherVDispatchWithDispls, CheckAllGatherV);
}

// AGV 入参/布局专项：全 0 counts 入口短路。
TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_all_zero_counts_test)
{
    TopoMeta topoMeta{{{0, 1, 2}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {0, 0, 0};
    vDataDes.displs = {0, 0, 0};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    // 全 0 counts 入口短路，无任务图可校验，直接断言返回码。
    EXPECT_EQ(RunAllGatherVWithBuffers(topoMeta, vDataDes, false), HCCL_SUCCESS);
}

// AGV 入参专项：displs 非法返回 HCCL_E_PARA。
TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_invalid_displs_test)
{
    TopoMeta topoMeta{{{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100};
    vDataDes.displs = {1, 0};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    EXPECT_EQ(RunAllGatherVWithBuffers(topoMeta, vDataDes, false), HCCL_E_PARA);
}

// AGV 入参专项：in-place 返回 HCCL_E_PARA。
TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_inplace_test)
{
    TopoMeta topoMeta{{{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {100, 100};
    vDataDes.displs = {0, 100};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    EXPECT_EQ(RunAllGatherVWithBuffers(topoMeta, vDataDes, true), HCCL_E_PARA);
}

// AGV 入参专项：displs 区间重叠返回 HCCL_E_PARA。
TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_overlap_displs_test)
{
    TopoMeta topoMeta{{{0, 1}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {2, 3};
    vDataDes.displs = {0, 1};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    EXPECT_EQ(RunAllGatherVWithBuffers(topoMeta, vDataDes, false), HCCL_E_PARA);
}

// displs 非单调但互不重叠：SE 决策删除单调不减约束后应被接受（不再拒绝）
TEST_F(ST_ALL_GATHER_V_TEST, st_all_gather_v_nonmonotonic_displs_test)
{
    // intervals [100,110) and [0,10): 非单调、互不重叠 → 合法布局
    // 走完整多 rank 路径 + checker 校验，证明布局被接受且正确执行
    TopoMeta topoMeta{{{0, 1, 2}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {2, 3, 1};
    vDataDes.displs = {100, 0, 50}; // 非单调乱序，互不重叠
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunVMultilevelTest(topoMeta, vDataDes, nullptr, AllGatherVDispatchWithDispls, CheckAllGatherV);
}
