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

constexpr u32 DATATYPE_SIZE_TABLE_RSV_DPU[HCCL_DATA_TYPE_RESERVED]
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

// DPU模式(DpuReduceScatterVSoleMesh)ST用例: 仿真CCL为200MB, 单轮容量=cclSize/(2*rankSize),
// 多轮loop阈值按此计算; ENABLE_HOSTDPU_FOR_LLT要求serverNum>=2, 拓扑均取2服务器
class ST_REDUCE_SCATTER_V_DPU_TEST : public ::testing::Test {
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

static void SetHostDpuEnv()
{
    setenv("ENABLE_HOSTDPU_FOR_LLT", "1", 1);
    setenv("HCCL_INDEPENDENT_OP", "1", 1);
}

static void RunReduceScatterVDpu(const TopoMeta& topoInfo, VDataDesTag vDataDes, HcclReduceOp reduceOp)
{
    RunVMultilevelTest(
        topoInfo, vDataDes, SetHostDpuEnv,
        [reduceOp](u32 rankId, u64 totalCount, VDataDesTag vDataDes, HcclComm comm, aclrtStream stream) -> HcclResult {
            void* sendBuf = nullptr;
            void* recvBuf = nullptr;
            const u32 dataTypeSize = DATATYPE_SIZE_TABLE_RSV_DPU[vDataDes.dataType];
            u64 sendBufSize = totalCount * dataTypeSize;
            u64 recvBufSize = vDataDes.counts[rankId] * dataTypeSize;
            // 0 count rank的recvBuf最小占位1字节, 规避aclrtMalloc对size=0报错
            aclrtMalloc(&sendBuf, sendBufSize, static_cast<aclrtMemMallocPolicy>(BUFFER_INPUT_MARK));
            aclrtMalloc(&recvBuf, std::max<u64>(recvBufSize, 1), static_cast<aclrtMemMallocPolicy>(BUFFER_OUTPUT_MARK));
            return HcclReduceScatterV(
                sendBuf, vDataDes.counts.data(), vDataDes.displs.data(), recvBuf, vDataDes.counts[rankId],
                vDataDes.dataType, reduceOp, comm, stream);
        },
        [reduceOp](AllRankTaskQueues& taskQueues, u32 rankSize, VDataDesTag vDataDes) -> HcclResult {
            return CheckReduceScatterV(taskQueues, rankSize, reduceOp, vDataDes);
        });
}

// 基础正向: 2服务器x2卡, 等长counts, SUM
TEST_F(ST_REDUCE_SCATTER_V_DPU_TEST, host_dpu_reduce_scatter_v_2srv_4rank_int32_equal_sum)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1024, 1024, 1024, 1024};
    vDataDes.displs = {0, 1024, 2048, 3072};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVDpu(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

// counts不均: 各rank数据量差异大(1M/1/4K/7)
TEST_F(ST_REDUCE_SCATTER_V_DPU_TEST, host_dpu_reduce_scatter_v_2srv_4rank_int32_uneven_sum)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1048576, 1, 4096, 7};
    vDataDes.displs = {0, 1048576, 1048577, 1052673};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT32;

    RunReduceScatterVDpu(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 含0 count: rank1/rank3无接收分片(提前返回, 仍向其他rank发送本侧切片)
TEST_F(ST_REDUCE_SCATTER_V_DPU_TEST, host_dpu_reduce_scatter_v_2srv_4rank_fp16_zero_count_sum)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1024, 0, 2048, 0};
    vDataDes.displs = {0, 1024, 1024, 3072};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16;

    RunReduceScatterVDpu(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 多轮loop: 4卡单轮容量200MB/(2*4)=25MB, 12M个FP32(48MB)超限触发2轮
TEST_F(ST_REDUCE_SCATTER_V_DPU_TEST, host_dpu_reduce_scatter_v_2srv_4rank_fp32_multiloop_sum)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {12582912, 12582912, 12582912, 12582912};
    vDataDes.displs = {0, 12582912, 25165824, 37748736};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunReduceScatterVDpu(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

// INT64数据类型: 触发模板内INT64同步分支(直拷在规约前完成的同步保障)
TEST_F(ST_REDUCE_SCATTER_V_DPU_TEST, host_dpu_reduce_scatter_v_2srv_4rank_int64_sum)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1024, 512, 256, 128};
    vDataDes.displs = {0, 1024, 1536, 1792};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT64;

    RunReduceScatterVDpu(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

// PROD规约: 触发模板内PROD同步分支
TEST_F(ST_REDUCE_SCATTER_V_DPU_TEST, host_dpu_reduce_scatter_v_2srv_4rank_fp32_prod)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1024, 512, 256, 128};
    vDataDes.displs = {0, 1024, 1536, 1792};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    RunReduceScatterVDpu(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_PROD);
}

// INT64+PROD双条件叠加的同步分支
TEST_F(ST_REDUCE_SCATTER_V_DPU_TEST, host_dpu_reduce_scatter_v_2srv_4rank_int64_prod)
{
    TopoMeta topoMeta{{{0, 1}, {2, 3}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1024, 512, 256, 128};
    vDataDes.displs = {0, 1024, 1536, 1792};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT64;

    RunReduceScatterVDpu(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_PROD);
}

// 8卡(2服务器x4卡)组合: counts不均+含0
TEST_F(ST_REDUCE_SCATTER_V_DPU_TEST, host_dpu_reduce_scatter_v_2srv_8rank_int8_uneven_zero_sum)
{
    TopoMeta topoMeta{{{0, 1, 2, 3}, {4, 5, 6, 7}}};
    VDataDesTag vDataDes;
    vDataDes.counts = {1048576, 0, 262144, 7, 65536, 1, 0, 5242880};
    vDataDes.displs = {0, 1048576, 1048576, 1310720, 1310727, 1376263, 1376264, 1376264};
    vDataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;

    RunReduceScatterVDpu(topoMeta, vDataDes, HcclReduceOp::HCCL_REDUCE_SUM);
}

// 注: InsTempDpuReduceScatterVMesh::CalcRes 的 MESH_1D_CLOS(UBX)分支(level0Topo==MESH_1D_CLOS &&
// !level0PcieMix && deviceNumPerModule>1, GetIntraRankNumFromPhysicalLevels + intraRankNum-1线程) LLT不可达:
// ST topo模型 L0 仅能构造 MESH_1D 框内形态(2D网格场景 L0 为3实例, 落入默认CLOS), GenTopoMeta 3层拓扑
// L0 仍为 MESH_1D, 均不满足 topoInstNum==2(1×CLOS+1×1DMESH)的 UBX 形态; 该分支线程数计算与
// 通道申请正确性留作 UBX 环境联调项
