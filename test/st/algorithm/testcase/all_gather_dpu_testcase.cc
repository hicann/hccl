/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "testcase_common.h"
#include "selected_algo_log_stub.h"

using namespace HcclSim;
using namespace ops_hccl;

class ST_ALL_GATHER_DPU_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        // 清空选路命中记录（PR#3933 评审 21ec67：防上一用例残留，见 ExpectAlgoHit）
        ResetSelectedAlgoLog();
    }
    void TearDown() override
    {
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_OP_EXPANSION_MODE");
        unsetenv("HCCL_ALGO");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

HcclResult RunHcclAllGather(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    (void)op;
    return HcclAllGather(sendBuf, recvBuf, count, dataType, comm, stream);
}

HcclResult RunCheckAllGather(
    HcclSim::AllRankTaskQueues& taskQueues, u32 rankSize, HcclDataType dataType, u64 dataCount, HcclReduceOp reduceType)
{
    (void)reduceType;
    return CheckAllGather(taskQueues, rankSize, dataType, dataCount);
}

void RunAllGatherDPUA5(const TopoMeta& topoMeta, u64 sendCount, HcclDataType dataType)
{
    // 算子执行参数设置
    u32 rankSize = CalRankSize(topoMeta); // 参与集合通信的卡数(同topoMeta卡数一致)
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    RunDpuTest(
        topoMeta, rankSize, sendCount, dataType, sendCount * dataTypeSize, sendCount * dataTypeSize * rankSize,
        RunHcclAllGather, RunCheckAllGather, HcclReduceOp::HCCL_REDUCE_RESERVED);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_1_fp32)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 1;                                         // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp32)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_210m_fp32)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 210 * 1024 * 1024;                         // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_210m_fp32_2_2)
{
    TopoMeta topoMeta{{{0, 1}, {0, 1}}};                       // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 210 * 1024 * 1024;                         // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int8)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int16)
{
    TopoMeta topoMeta{{{0}, {0}}};                              // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                           // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT16; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int32)
{
    TopoMeta topoMeta{{{0}, {0}}};                              // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                           // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_int64)
{
    TopoMeta topoMeta{{{0}, {0}}};                              // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                           // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT64; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp16)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp64)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP64; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_bfp16)
{
    TopoMeta topoMeta{{{0}, {0}}};                              // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                           // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp8e5m2)
{
    TopoMeta topoMeta{{{0}, {0}}};                                // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                             // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E5M2; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp8e4m3)
{
    TopoMeta topoMeta{{{0}, {0}}};                                // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                             // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E4M3; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_fp8e8m0)
{
    TopoMeta topoMeta{{{0}, {0}}};                                // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                             // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP8E8M0; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_10m_hif8)
{
    TopoMeta topoMeta{{{0}, {0}}};                             // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// asymmetric topology
TEST_F(ST_ALL_GATHER_DPU_TEST, host_dpu_opbase_all_gather_asymmetric_10m_hif8)
{
    TopoMeta topoMeta{{{0}, {0, 2, 3}}};                       // 三维数组指定超节点-Server-Device信息
    u64 sendCount = 10 * 1024 * 1024;                          // 接收数据量
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_HIF8; // 数据类型
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// ========== 正向测试 ==========
// 对称3级: x=4, y=4, z=2, int8
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x4x2_int8)
{
    setenv("HCCL_ALGO", "allgather:pipeline{mesh,nhr,nhr}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 对称3级: x=2, y=2, z=2, bfp16
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_2x2x2_bfp16)
{
    setenv("HCCL_ALGO", "allgather:pipeline{mesh,nhr,nhr}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 sendCount = 256;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 对称3级: x=8, y=4, z=2, int32
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_8x4x2_int32)
{
    setenv("HCCL_ALGO", "allgather:pipeline{mesh,nhr,nhr}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 8);
    u64 sendCount = 2048;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 大数据量: 1M
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x4x2_fp32_1m)
{
    setenv("HCCL_ALGO", "allgather:pipeline{mesh,nhr,nhr}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 sendCount = 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// ========== 边界测试 ==========
// xRankSize=1 (L0退化)
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_1x4x2_fp32)
{
    setenv("HCCL_ALGO", "allgather:pipeline{mesh,nhr,nhr}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 1);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// zRankSize=1 (L2退化, 单超节点) -> 不走OmniPipe
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x4x1_fp32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 4);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// yRankSize=1 (L1退化)
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x1x2_fp32)
{
    setenv("HCCL_ALGO", "allgather:pipeline{mesh,nhr,nhr}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 4);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 最小数据量
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_2x2x2_fp32_send1)
{
    setenv("HCCL_ALGO", "allgather:pipeline{mesh,nhr,nhr}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 sendCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 大数据量: 25M
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_4x4x2_int32_16m)
{
    setenv("HCCL_ALGO", "allgather:pipeline{mesh,nhr,nhr}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 sendCount = 25 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 非对称TopoMeta
TEST_F(ST_ALL_GATHER_DPU_TEST, omnipipe_dpu_3level_L0asym_int32_512)
{
    setenv("HCCL_ALGO", "allgather:pipeline{mesh,nhr,nhr}", 1);
    TopoMeta topoMeta{{{0, 1}, {0, 1}}, {{0, 1}}};
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// ========== DPU 三级 Sequence (DpuAllGatherSequenceMeshConcurNHRNHR) ==========
// 与 OmniPipe 组的差异:
//   1. HCCL_ALGO 强制选路 —— "allgather:sequence{meshconcur,nhr,nhr}" 对引擎候选展开命中
//      Aicpu/Dpu 两个三级 Sequence, hostDPU 拓扑下 AICPU 版被 attrs 双向互斥过滤
//      (isHostDpuOnly=false 且 topo hostDpuOnly=true), preferred 层收敛后唯一命中 DPU 三级;
//   2. ENABLE_HOSTDPU_L2_ONLY=1 —— 仿真模型默认全层报 HOST, 会让三级 Sequence 的准入③
//      (中间层 locType 须为 DEVICE)拦截; 开关打开后仅 L2 报 HOST、L0/L1 报 DEVICE,
//      对齐真实 228 形态(框间 UB CLOS 落设备侧), 准入①②③全部通过。
class ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetAlgEnvConfigInitState();
        // 清空选路命中记录（PR#3933 评审 21ec67：防上一用例残留，见 ExpectAlgoHit）
        ResetSelectedAlgoLog();
        setenv("HCCL_ALGO", "allgather:sequence{meshconcur,nhr,nhr}", 1);
        setenv("ENABLE_HOSTDPU_L2_ONLY", "1", 1);
    }
    void TearDown() override
    {
        unsetenv("HCCL_ALGO");
        unsetenv("ENABLE_HOSTDPU_L2_ONLY");
        unsetenv("HCCL_ENABLE_OPEN_AICPU");
        unsetenv("ENABLE_HOSTDPU_FOR_LLT");
        unsetenv("HCCL_INDEPENDENT_OP");
        unsetenv("HCCL_OP_EXPANSION_MODE");
    }
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 对称3级正向: x=4, y=4, z=2
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_2x4x4_int8)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 对称3级最小拓扑: 2x2x2 = 8 卡
// d1=1 退化拓扑 2x1x8: 中间层单实例使 levels_[1].rankSize==1, 触发 skipLevel1_,
// OrchestrateLoop 跳过 L1 NHR 仅执行 L2->L0, 验证中间层退化下三级数据面正确
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_2x1x8_bfp16)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 1, 8);
    u64 sendCount = 256;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 对称3级: 8卡/server 形态, 2x4x8 = 64 卡
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_2x4x8_int32)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 8);
    u64 sendCount = 2048;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 最小数据量: sendCount=1
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_2x2x2_fp32_send1)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2);
    u64 sendCount = 1;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 非整除尾块: sendCount=243 (不整除 sliceSize)
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_2x3x4_fp32_send243)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 3, 4);
    u64 sendCount = 243;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// L1 非对称拓扑: pod0=2srv×8rank(16卡) + pod1=1srv×8rank(8卡), L1 层实例大小{16,8}。
// topo_match 按最小实例对齐(GCD 虚拟化), 验证非对称形态下三级匹配/准入行为与数据面正确
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_l1_asym_16_8_fp32)
{
    TopoMeta topoMeta;
    SuperPodMeta pod0 = {ServerMeta{0, 1, 2, 3, 4, 5, 6, 7}, ServerMeta{0, 1, 2, 3, 4, 5, 6, 7}};
    SuperPodMeta pod1 = {ServerMeta{0, 1, 2, 3, 4, 5, 6, 7}};
    topoMeta.push_back(pod0);
    topoMeta.push_back(pod1);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 大数据量: 16M (多 loop)
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_2x4x4_int32_16m)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 sendCount = 16 * 1024 * 1024;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 准入③回退: 不开 ENABLE_HOSTDPU_L2_ONLY 时全层 locType=HOST, 准入③拦截三级,
// preferred 算法被过滤后 WARNING 回退自动选路(选中二级), 数据面仍须正确
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_l1_host_fallback_2x4x4_fp32)
{
    unsetenv("ENABLE_HOSTDPU_L2_ONLY");
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 4);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 两层拓扑回退: 单超节点(z=1)下三级 topo 不匹配, 回退二级, 数据面仍须正确。
// 不开 ENABLE_HOSTDPU_L2_ONLY: 两层拓扑时 CheckHostDPUOnly 检查 L1 层, 开关会使 L1 报 DEVICE
// 导致 hostDPUOnly=false 而回退 AICPU 算法池(其框间通道在此仿真拓扑下不可建链);
// 关闭开关保持全层 HOST, hostDpuOnly=true, 回退 DPU 二级 Sequence(与 omnipipe 4x4x1 先例同路径)。
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_1x4x4_fp32_fallback)
{
    unsetenv("ENABLE_HOSTDPU_L2_ONLY");
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 4);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// 踩踏拓扑回归: 4 超节点 x 2 框(d1=2, d2=4, 满足 d2>=d1+1)。修复前 PostLocalCopy 的
// 紧凑->跨步原地升序重排中, 条目1的目标槽位(i*d1*S=1*2*S)会覆盖条目2(=d1)尚未搬运的源数据,
// 导致 AllGather 结果错误; 修复后 NHR 环按 outputSliceStride 直落最终槽位, 无重排拷贝。
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, sequence_dpu_3level_4x2x2_int32_stomp)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 4, 2, 2);
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
}

// ========== DPU RHD 二级/三级 Sequence 绑定点（PR#3933 评审 MEDIUM：ST 零覆盖定向补齐）==========
// 选路词与 AICPU 侧 RHD（all_gather_aicpu_testcase.cc §RHD）不同源，按注册名拼名口径（alg_parse
// ComposeAlgoName 按 [Engine]AllGatherSequence[level0 词][level1 词]… 精确匹配注册名）：
//   二级 DpuAllGatherSequenceMeshRHD（L0 词=Mesh：L0=server 内 ZAxisDetour、L1=server 间
//   host-DPU RHD）← allgather:sequence{level0=mesh,level1=rhd}；meshconcur 词拼出的是 AICPU 款
//   AicpuAllGatherSequenceMeshConcurRHD，host-DPU 拓扑下被 isHostDpuOnly 硬检查互斥过滤不可达；
//   三级 DpuAllGatherSequenceMeshConcurRHDRHD（L0 词=MeshConcur，执行序 L2→L1→L0）←
//   allgather:sequence{level0=meshconcur,level1=rhd,level2=rhd}，与 AICPU 款同串，host-DPU
//   拓扑（hostDpuOnly=true）下 AICPU 款被硬检查过滤、preferred 唯一收敛 DPU 款。
// 验证范围（如实登记）：本组用例验证 preferred 选路收敛 + AICPU 侧编排面（KernelRun 协议序
// LocalDataCopy→HcommBatchModeEnd→HcommThreadSynchronize→DPURunInfo 序列化→HcommSendRequest→
// HcommWaitResponse→HcommBatchModeStart→PostLocalCopy，及二级 intra ZAxisDetour 数据面）；DPU
// 数据面 RunAllGatherRHDDpu 因 ST 仿真库以 -DAICPU_COMPILE 编入 rhd_dpu.cc 而 #ifndef 编空
// （nhr_dpu.cc:241 同构先例，HcommSendRequest 桩同步回调 HcclLaunchDPUKernel 后数据面为空转），
// 跨 rank 搬运不随本组用例真实执行。
// 本组 4 条用例 RunAllGatherDPUA5 后经 ExpectAlgoHit 断言选路真实命中（PR#3933 评审
// 21ec67：防静默回退——dev-r13 pow2 组网误写实证）。

// 二级 RHD 2^幂退化：1 超节点 × 2 server × 2 rank=4 rank（RHD 域 p=2、σ 恒等，前/后处理空步骤）。
// 不开 ENABLE_HOSTDPU_L2_ONLY：两级拓扑全层报 HOST，CheckHostDPUOnly 校验 L1（末层）得
// hostDpuOnly=true（与 sequence_dpu_3level_1x4x4_fp32_fallback 回退先例同路径），二级 DPU RHD 可达。
// dev-r13 修正：原组网误写 (1, 2, 1)（2 rank，L0 层内单 rank → Level1Nhr=true 硬过滤，
// "skipped by topo filter" 回退 NHR——用例从未真正命中 RHD），对齐头注释 4 rank 意图改
// (1, 2, 2)，RHD 域 p=2 真命中（Level1Nhr=false 同 nonpow3 形态）
TEST_F(ST_ALL_GATHER_DPU_TEST, st_all_gather_dpu_rhd_sequence_2level_pow2_fp32)
{
    setenv("HCCL_ALGO", "allgather:sequence{level0=mesh,level1=rhd}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 2, 2); // 1 超节点 × 2 server × 2 rank=4 rank，RHD 域 p=2
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// 二级 RHD 非幂（d1=3，前/后处理生效）：1 超节点 × 3 server × 2 rank=6 rank（RHD 域 p=3、r=1，
// 相邻配对 (0,1)、代表 0/被合并 1/block 2；AICPU 侧装载/卸载槽位按非幂 run 表落位）
TEST_F(ST_ALL_GATHER_DPU_TEST, st_all_gather_dpu_rhd_sequence_2level_nonpow3_int8)
{
    setenv("HCCL_ALGO", "allgather:sequence{level0=mesh,level1=rhd}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 3, 2); // 1 超节点 × 3 server × 2 rank，RHD 域 p=3
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// 三级 RHD：2 超节点 × 2 server × 2 rank=8 rank（L1 域 p=2、L2 域 p=2 均幂退化，执行序 L2→L1→L0）。
// 沿套件 SetUp 的 ENABLE_HOSTDPU_L2_ONLY=1（仅 L2 报 HOST → hostDpuOnly=true；L1 报 DEVICE 过
// 三级 admit ③），HCCL_ALGO 在用例体内覆盖套件默认 NHR 串（setenv 后 RunDpuTest 生效），强制
// 命中 DpuAllGatherSequenceMeshConcurRHDRHD
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, st_all_gather_dpu_rhd_sequence_3level_2x2x2_int8)
{
    setenv("HCCL_ALGO", "allgather:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 2, 2); // 2 超节点 × 2 server × 2 rank，L1/L2 域均 p=2
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}

// 三级 RHD 非幂（dev-r13 补）：3 超节点 × 2 server × 2 rank=12 rank，L2 域 p=3 r=1（非幂，
// L2=host-DPU RHD 非幂的 DPU 款 PRE/MAIN/POST 路径）、L1 域 p=2 幂退化，执行序 L2→L1→L0。
// 复用 2x2x2 同款选路与套件开关，验证三级形态下 L2 RHD-DPU 非幂数据面（二级 nonpow3 同族）
TEST_F(ST_ALL_GATHER_DPU_3LEVEL_SEQ_TEST, st_all_gather_dpu_rhd_sequence_3level_nonpow3_int8)
{
    setenv("HCCL_ALGO", "allgather:sequence{level0=meshconcur,level1=rhd,level2=rhd}", 1);
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 3, 2, 2); // 3 超节点 × 2 server × 2 rank，L2 域 p=3 非幂
    u64 sendCount = 512;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    RunAllGatherDPUA5(topoMeta, sendCount, dataType);
    ExpectAlgoHit("RHD");
}
