/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "v_testcase_common.h"
#include "gtest/gtest.h"

#include "alg_env_config.h"
#include <algorithm>

constexpr u32 DATATYPE_SIZE_TABLE_AGV_DPU[HCCL_DATA_TYPE_RESERVED]
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

// DPU模式(DpuAllGatherVSoleMesh)ST用例: 仿真CCL为200MB, 单轮容量=cclSize/rankSize(slot[myRank]兼任发送暂存),
// 多轮loop阈值按此计算; ENABLE_HOSTDPU_FOR_LLT要求serverNum>=2, 拓扑均取2服务器
class ST_ALL_GATHER_V_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override { ResetAlgEnvConfigInitState(); }
    void TearDown() override
    {
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// hostDPU仿真环境: 全层端点报HOST, hostDpuOnly拓扑下唯一AGV的DPU算法DpuAllGatherVSoleMesh确定性选路
static void SetHostDpuEnv()
{
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
}

static HcclResult
AllGatherVDispatch(u32 rankId, u64 totalCount, VDataDesTag vDataDes, HcclComm comm, aclrtStream stream)
{
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE_AGV_DPU[vDataDes.dataType];
    u64 sendBufSize = vDataDes.counts[rankId] * dataTypeSize;
    u64 recvBufSize = totalCount * dataTypeSize;
    // 0 count rank的sendBuf最小占位1字节, 规避aclrtMalloc对size=0报错
    aclrtMalloc(&sendBuf, std::max<u64>(sendBufSize, 1), static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
    aclrtMalloc(&recvBuf, recvBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
    return HcclAllGatherV(
        sendBuf, vDataDes.counts[rankId], recvBuf, vDataDes.counts.data(), vDataDes.displs.data(), vDataDes.dataType,
        comm, stream);
}

static void RunAllGatherVDpu(const TopoMeta& topoInfo, VDataDesTag vDataDes)
{
    RunVMultilevelTest(topoInfo, vDataDes, SetHostDpuEnv, AllGatherVDispatch, CheckAllGatherV);
}

// 基础正向: 2服务器x2卡, 等长counts
TEST_F(ST_ALL_GATHER_V_DPU_TEST, host_dpu_all_gather_v_2srv_4rank_fp16_equal)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1024, 1024, 1024, 1024};
    vDataDes.displs = {0, 1024, 2048, 3072};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVDpu(topoMeta, vDataDes);
}

// counts不均: 各rank数据量差异大(1M/1/4K/7)
TEST_F(ST_ALL_GATHER_V_DPU_TEST, host_dpu_all_gather_v_2srv_4rank_int32_uneven)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1048576, 1, 4096, 7};
    vDataDes.displs = {0, 1048576, 1048577, 1052673};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunAllGatherVDpu(topoMeta, vDataDes);
}

// 含0 count: 中间rank1与末尾rank3接收量为0(该rank广播空切片, 仍正常接收其余rank切片)
TEST_F(ST_ALL_GATHER_V_DPU_TEST, host_dpu_all_gather_v_2srv_4rank_fp16_zero_count)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1024, 0, 2048, 0};
    vDataDes.displs = {0, 1024, 1024, 3072};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunAllGatherVDpu(topoMeta, vDataDes);
}

// 多轮loop: 4卡单轮容量200MB/4=50MB, 最大rank分片18M个FP32(72MB)超限触发2轮;
// counts不均使各rank进度不同, 覆盖按对端进度落位的场景
TEST_F(ST_ALL_GATHER_V_DPU_TEST, host_dpu_all_gather_v_2srv_4rank_fp32_multiloop_uneven)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {18874368, 4718592, 9437184, 1572864};
    vDataDes.displs = {0, 18874368, 23592960, 33030144};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunAllGatherVDpu(topoMeta, vDataDes);
}

// 8卡(2服务器x4卡)组合: counts不均+含0
TEST_F(ST_ALL_GATHER_V_DPU_TEST, host_dpu_all_gather_v_2srv_8rank_int8_uneven_zero)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1048576, 0, 262144, 7, 65536, 1, 0, 5242880};
    vDataDes.displs = {0, 1048576, 1048576, 1310720, 1310727, 1376263, 1376264, 1376264};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    RunAllGatherVDpu(topoMeta, vDataDes);
}

// 注: InsTempDpuAllGatherVMesh::CalcRes 的 MESH_1D_CLOS(UBX)分支 LLT不可达, 原因同
// reduce_scatter_v_dpu_testcase.cc 文件尾注释; 线程数计算与通道申请正确性留作 UBX 环境联调项
