/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <gtest/gtest.h>
#include "alg_param.h"
#include "cost_model.h"
#include "ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.h"
#include "ins_temp_reduce_scatter_aicpu_reduce_nhr.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ins_temp_all_reduce_aicpu_reduce_nhr.h"

using namespace ops_hccl;

namespace {
constexpr const char* MESH_SEQ_ALG_NAME = "AicpuReduceScatterSequenceMeshNHRAicpuReduce";
}

// ============ Pcie 模板静态 CalcCostCoeff（框间段真实建模） ============

TEST(ReduceScatterMeshSeqCostTest, PcieTemplateRealModelInterLevel)
{
    // 框间层实参：rankSize=d1=2, dataRatio=1.0, input=HCCL_BUFFER/output=OUTPUT/scratch=HCCL_BUFFER
    CalcCostCoeffParam inter;
    inter.rankSize = 2;
    inter.dataRatio = 1.0f;
    inter.netType = CommTopo::COMM_TOPO_CLOS;
    inter.portNum = {8};
    inter.algName = MESH_SEQ_ALG_NAME;
    inter.inputBuffer = BufferType::HCCL_BUFFER;
    inter.outputBuffer = BufferType::OUTPUT;
    inter.scratchBuffer = BufferType::HCCL_BUFFER;
    auto r = InsTempReduceScatterAicpuReduceNHRPcie::CalcCostCoeff(inter);
    ASSERT_EQ(r.size(), 1u);
    // A = dataRatio×rankSize×(rs-1)/(portNum×56GB/s)：2 轮 gather 每轮发 (rs-1) 份 slice
    EXPECT_NEAR(r[0].A, 1.0f * 2 * 1 / (8 * 56.0e9f), 1e-14);
    // B = 出拷贝(dataRatio/1400G)；input==scratch 关闭入拷贝，output==scratch 关闭出拷贝
    EXPECT_FLOAT_EQ(r[0].B, 1.0f / 1.4e12f);
    // C = 2us × rs×(nSteps + 2 + SYNC_EQUIV_TASKS=50)；D = 1us × taskNum
    // GetNHRStepNum(2)=1 → taskNum = 2×(1+2+50) = 106
    const int taskNum = 2 * (1 + 2 + 50);
    EXPECT_NEAR(r[0].C, 2e-6 * taskNum, 1e-9);
    EXPECT_NEAR(r[0].D, 1e-6 * taskNum, 1e-9);
    // reduceCopies = (rs-1)×dataRatio = 1×1.0（myIdx 轮规约 1 个 slice）
    EXPECT_FLOAT_EQ(r[0].reduceCopies, 1.0f);
}

TEST(ReduceScatterMeshSeqCostTest, PcieTemplateNoAlgNameGating)
{
    // 共用模板的其它算法（AR 侧 RS 半段）/未知名 → 同口径返回真实模型（不做 algName 门控）
    CalcCostCoeffParam p;
    p.rankSize = 2;
    p.dataRatio = 1.0f;
    p.netType = CommTopo::COMM_TOPO_CLOS;
    p.portNum = {8};
    p.inputBuffer = BufferType::HCCL_BUFFER;
    p.outputBuffer = BufferType::OUTPUT;
    p.scratchBuffer = BufferType::HCCL_BUFFER;
    p.algName = "AicpuAllReduceSequenceMeshNHRAicpuReduce";
    auto r1 = InsTempReduceScatterAicpuReduceNHRPcie::CalcCostCoeff(p);
    ASSERT_EQ(r1.size(), 1u);
    p.algName = nullptr;
    auto r2 = InsTempReduceScatterAicpuReduceNHRPcie::CalcCostCoeff(p);
    ASSERT_EQ(r2.size(), 1u);
    EXPECT_FLOAT_EQ(r1[0].A, r2[0].A);
    EXPECT_FLOAT_EQ(r1[0].A, 1.0f * 2 * 1 / (8 * 56.0e9f));
}

TEST(ReduceScatterMeshSeqCostTest, PcieTemplateCostScalesWithDataRatio)
{
    // 传输/拷贝量均正比于 dataRatio×rankSize(输入全量)：同 rankSize 下 dataRatio 翻倍 → A/B 翻倍，C 不变
    CalcCostCoeffParam base;
    base.rankSize = 4;
    base.dataRatio = 1.0f;
    base.netType = CommTopo::COMM_TOPO_CLOS;
    base.portNum = {8};
    base.algName = MESH_SEQ_ALG_NAME;
    base.inputBuffer = BufferType::HCCL_BUFFER;
    base.outputBuffer = BufferType::OUTPUT;
    base.scratchBuffer = BufferType::HCCL_BUFFER;
    auto r1 = InsTempReduceScatterAicpuReduceNHRPcie::CalcCostCoeff(base);
    CalcCostCoeffParam dbl = base;
    dbl.dataRatio = 2.0f;
    auto r2 = InsTempReduceScatterAicpuReduceNHRPcie::CalcCostCoeff(dbl);
    ASSERT_EQ(r1.size(), 1u);
    ASSERT_EQ(r2.size(), 1u);
    EXPECT_FLOAT_EQ(r2[0].A, 2.0f * r1[0].A);
    EXPECT_FLOAT_EQ(r2[0].B, 2.0f * r1[0].B); // B = dataRatio/1400G（出拷贝）
    EXPECT_FLOAT_EQ(r2[0].C, r1[0].C);        // C 仅随 rankSize 变化
}

TEST(ReduceScatterMeshSeqCostTest, PcieTemplateIsPodNoPortHalve)
{
    // isPod 不触发端口折半(halvePodPort=false, 对齐 RS 家族 ins_temp_reduce_scatter_nhr/hd):
    // A 与非 pod 场景(PcieTemplateRealModelInterLevel)逐位一致, 若误用默认 true 会高估 ~1.74x
    CalcCostCoeffParam inter;
    inter.rankSize = 2;
    inter.dataRatio = 1.0f;
    inter.netType = CommTopo::COMM_TOPO_CLOS;
    inter.portNum = {8};
    inter.algName = MESH_SEQ_ALG_NAME;
    inter.inputBuffer = BufferType::HCCL_BUFFER;
    inter.outputBuffer = BufferType::OUTPUT;
    inter.scratchBuffer = BufferType::HCCL_BUFFER;
    inter.isPod = true;
    auto r = InsTempReduceScatterAicpuReduceNHRPcie::CalcCostCoeff(inter);
    ASSERT_EQ(r.size(), 1u);
    EXPECT_FLOAT_EQ(r[0].A, 1.0f * 2 * 1 / (8 * 56.0e9f)); // 与非 pod 同值, 端口未减半
}

// ============ Sole 模板真实化（整域） ============

TEST(ReduceScatterMeshSeqCostTest, SoleTemplateRealModelWholeDomain)
{
    // Sole executor 实参：rankSize=N=16, dataRatio=1.0, input=INPUT/output=HCCL_BUFFER/scratch=HCCL_BUFFER
    CalcCostCoeffParam sole;
    sole.rankSize = 16;
    sole.dataRatio = 1.0f;
    sole.netType = CommTopo::COMM_TOPO_CLOS;
    sole.portNum = {8};
    sole.algName = "AicpuReduceScatterSoleNHRAicpuReduce";
    sole.inputBuffer = BufferType::INPUT;
    sole.outputBuffer = BufferType::HCCL_BUFFER;
    sole.scratchBuffer = BufferType::HCCL_BUFFER;
    auto r = InsTempReduceScatterAicpuReduceNHR::CalcCostCoeff(sole);
    ASSERT_EQ(r.size(), 1u);
    // A = dataRatio×rankSize×(N-1)/(portNum×56G)：N 轮 gather 总发送 (N-1) 份输入全量
    EXPECT_NEAR(r[0].A, 1.0f * 16 * 15 / (8 * 56.0e9f), 1e-13);
    // B = 入拷贝(dataRatio×rankSize/1400G，N 轮输入全量入 cclBuff)；output==scratch 关闭出拷贝
    EXPECT_FLOAT_EQ(r[0].B, 16.0f / 1.4e12f);
    // taskNum = 16×(GetNHRStepNum(16)=4 + 2 + 50) = 896；C 含每轮同步折算
    const int taskNum = 16 * (4 + 2 + 50);
    EXPECT_NEAR(r[0].C, 2e-6 * taskNum, 1e-8);
    EXPECT_NEAR(r[0].D, 1e-6 * taskNum, 1e-9);
    // reduceCopies = (N-1)×dataRatio = 15（myIdx 轮规约 15 个 slice）
    EXPECT_FLOAT_EQ(r[0].reduceCopies, 15.0f);
}

TEST(ReduceScatterNhrCostTest, MultiLinkAggregatesPhysicalSources)
{
    CalcCostCoeffParam single;
    single.rankSize = 8;
    single.dataRatio = 1.0f;
    single.netType = CommTopo::COMM_TOPO_CLOS;
    single.portNum = {16};
    single.algName = "AicpuReduceScatterSoleNHR";
    single.inputBuffer = BufferType::INPUT;
    single.outputBuffer = BufferType::HCCL_BUFFER;
    single.scratchBuffer = BufferType::HCCL_BUFFER;

    CalcCostCoeffParam multiLink = single;
    multiLink.algName = "AicpuReduceScatterSoleNHRMultiLink";
    multiLink.phyLevelPortNums = {{16}, {16}};

    auto singleCost = InsTempReduceScatterNHR::CalcCostCoeff(single);
    auto multiLinkCost = InsTempReduceScatterNHR::CalcCostCoeff(multiLink);

    ASSERT_EQ(singleCost.size(), 1u);
    ASSERT_EQ(multiLinkCost.size(), 1u);
    EXPECT_FLOAT_EQ(multiLinkCost[0].A, singleCost[0].A / 2.0f);
    EXPECT_FLOAT_EQ(multiLinkCost[0].B, singleCost[0].B);
    EXPECT_FLOAT_EQ(multiLinkCost[0].C, singleCost[0].C);
}

TEST(ReduceScatterNhrCostTest, IncompleteSourcesFallBackToAnchor)
{
    CalcCostCoeffParam anchor;
    anchor.rankSize = 8;
    anchor.dataRatio = 1.0f;
    anchor.netType = CommTopo::COMM_TOPO_CLOS;
    anchor.portNum = {16};
    anchor.algName = "AicpuReduceScatterSoleNHRMultiLink";
    anchor.inputBuffer = BufferType::INPUT;
    anchor.outputBuffer = BufferType::HCCL_BUFFER;
    anchor.scratchBuffer = BufferType::HCCL_BUFFER;

    CalcCostCoeffParam incomplete = anchor;
    incomplete.phyLevelPortNums = {{16}, {}};

    auto anchorCost = InsTempReduceScatterNHR::CalcCostCoeff(anchor);
    auto incompleteCost = InsTempReduceScatterNHR::CalcCostCoeff(incomplete);

    ASSERT_EQ(anchorCost.size(), 1u);
    ASSERT_EQ(incompleteCost.size(), 1u);
    EXPECT_FLOAT_EQ(incompleteCost[0].A, anchorCost[0].A);
}

// ============ AllReduce 模板（单轮整域 gather + 本地规约） ============

TEST(ReduceScatterMeshSeqCostTest, AllReduceTemplateRealModelSole)
{
    // Sole executor 实参：rankSize=N=8, dataRatio=1/N, input=INPUT/output=OUTPUT/scratch=HCCL_BUFFER
    CalcCostCoeffParam ar;
    ar.rankSize = 8;
    ar.dataRatio = 1.0f / 8;
    ar.netType = CommTopo::COMM_TOPO_CLOS;
    ar.portNum = {8};
    ar.algName = "AicpuAllReduceSoleNHRAicpuReduce";
    ar.inputBuffer = BufferType::INPUT;
    ar.outputBuffer = BufferType::OUTPUT;
    ar.scratchBuffer = BufferType::HCCL_BUFFER;
    auto r = InsTempAllReduceAicpuReduceNHR::CalcCostCoeff(ar);
    ASSERT_EQ(r.size(), 1u);
    // A：单轮整域 NHR AllGather，dataRatio×rankSize=1.0 → 发送 (N-1) 份全量
    EXPECT_NEAR(r[0].A, 1.0f * 7 / (8 * 56.0e9f), 1e-14);
    // B = 入拷贝 + 出拷贝各 1 份全量(dataRatio×rankSize=1.0)
    EXPECT_FLOAT_EQ(r[0].B, 2.0f / 1.4e12f);
    // taskNum = (N-1) 规约 + 4×nSteps 收发(GetNHRStepNum(8)=3) + 2 拷贝 = 21；C 含 1 次同步折算
    const int taskNum = (8 - 1) + 4 * 3 + 2;
    EXPECT_NEAR(r[0].C, 2e-6 * (taskNum + 50), 1e-9);
    EXPECT_NEAR(r[0].D, 1e-6 * taskNum, 1e-9);
    // reduceCopies = (N-1)×dataRatio×rankSize = 7×0.125×8（RunReduce 规约 7 次全量）
    EXPECT_FLOAT_EQ(r[0].reduceCopies, 7.0f);
}

// ============ 64bit/PROD host 规约折算（CalcHostReduceSlope） ============

TEST(ReduceScatterMeshSeqCostTest, HostReduceSlopeGating)
{
    // 门控与 alg_data_trans_wrapper 的 LocalReduce host 路由条件对齐: 64bit(8字节) 或 PROD
    const float copies = 7.0f;
    // fp32 SUM: DMA 路径, 不折算
    EXPECT_FLOAT_EQ(CalcHostReduceSlope(copies, 4, HcclReduceOp::HCCL_REDUCE_SUM), 0.0f);
    // fp64 SUM: 1.2ns×7/8
    EXPECT_FLOAT_EQ(CalcHostReduceSlope(copies, 8, HcclReduceOp::HCCL_REDUCE_SUM), 1.2e-9f * 7.0f / 8.0f);
    // fp32 PROD: 1.2ns×7/4
    EXPECT_FLOAT_EQ(CalcHostReduceSlope(copies, 4, HcclReduceOp::HCCL_REDUCE_PROD), 1.2e-9f * 7.0f / 4.0f);
    // dts 未知(0)兜底不折算
    EXPECT_FLOAT_EQ(CalcHostReduceSlope(copies, 0, HcclReduceOp::HCCL_REDUCE_PROD), 0.0f);
    // 无规约段(reduceCopies=0)
    EXPECT_FLOAT_EQ(CalcHostReduceSlope(0.0f, 8, HcclReduceOp::HCCL_REDUCE_SUM), 0.0f);
}
