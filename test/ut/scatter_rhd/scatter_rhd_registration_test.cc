/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Scatter RHD 注册与选路 UT（真注册表 + 真过滤缝隙，非手工复制的 attrs）：
// 链接 scatter_sequence_aicpu_executor_3level.cc，其静态注册把四条注册（NHR 二/三级 + RHD 二/三级）
// 写入运行期单例（AlgAttrsRegistry / CollAlgExecRegistryV2），本文件按生产名查询真 attrs；
// CheckAlgoMatchTopoWithReason（cost_model.cc）为选路过滤的真实缝隙：软策略（topoCustomCheck，
// preferred-only 的默认排除机制）与硬 attrs（min/maxTopoLevelNum，层级不符的排除机制）均在此裁决。
// 配置→候选过滤层（UpdateCostModelWithAlgo）的 RHD 用例在 test/ut/common/alg_parse。
#include <gtest/gtest.h>
#include "alg_attrs_registry.h"
#include "cost_model.h"
#include "alg_parse.h"

using namespace ops_hccl;

namespace {
// 构造选路判定所需的最小拓扑事实（CheckAlgoMatchTopoWithReason 读到的字段）
TopoInfoWithNetLayerDetails MakeTopo(u32 levelNums, Level0Shape level0, u32 rankSize = 8, bool hostDpuOnly = false)
{
    TopoInfoWithNetLayerDetails topo;
    topo.topoLevelNums = levelNums;
    topo.level0Topo = level0;
    topo.userRankSize = rankSize;
    topo.hostDpuOnly = hostDpuOnly;
    return topo;
}
} // namespace

// 生产注册名解析 + attrs 值（真注册表查询）：preferred-only 口径——软策略恒过滤、硬 attrs 兜底
TEST(ScatterRhdRegistrationTest, AicpuRhdAttrsRegisteredFromProduction)
{
    const AlgAttrs* a2 = AlgAttrsRegistry::Instance().Get("AicpuScatterSequenceMeshConcurRHD");
    ASSERT_NE(a2, nullptr);
    EXPECT_EQ(a2->opType, HcclCMDType::HCCL_CMD_SCATTER);
    EXPECT_EQ(a2->engine, OpExecuteConfig::AICPU_TS);
    EXPECT_EQ(a2->topo.minTopoLevelNum, 2u);
    EXPECT_EQ(a2->topo.maxTopoLevelNum, 2u);
    EXPECT_EQ(a2->topo.supportLevel0Topos, LEVEL0_TOPO_MESH_1D);
    ASSERT_NE(a2->topo.topoCustomCheck, nullptr);
    EXPECT_FALSE(a2->topo.topoCustomCheck(nullptr)); // preferred-only：软策略恒过滤
    EXPECT_EQ(a2->op.opCustomCheck, nullptr);        // 二级形态无 4GB 数据量门槛

    const AlgAttrs* a3 = AlgAttrsRegistry::Instance().Get("AicpuScatterSequenceMeshConcurRHDRHD");
    ASSERT_NE(a3, nullptr);
    EXPECT_EQ(a3->opType, HcclCMDType::HCCL_CMD_SCATTER);
    EXPECT_EQ(a3->engine, OpExecuteConfig::AICPU_TS);
    EXPECT_EQ(a3->topo.minTopoLevelNum, 3u);
    EXPECT_EQ(a3->topo.maxTopoLevelNum, 3u);
    EXPECT_EQ(a3->topo.supportLevel0Topos, LEVEL0_TOPO_MESH_1D);
    ASSERT_NE(a3->topo.topoCustomCheck, nullptr);
    EXPECT_FALSE(a3->topo.topoCustomCheck(nullptr));
}

// 既有 NHR 注册零变化（同文件追加注册不改变既有条目）
TEST(ScatterRhdRegistrationTest, ExistingNhrAttrsUnchanged)
{
    const AlgAttrs* n2 = AlgAttrsRegistry::Instance().Get("AicpuScatterSequenceMeshConcurNHR");
    ASSERT_NE(n2, nullptr);
    EXPECT_EQ(n2->topo.minTopoLevelNum, 2u);
    EXPECT_EQ(n2->topo.maxTopoLevelNum, 2u);
    EXPECT_EQ(n2->topo.topoCustomCheck, nullptr); // NHR 无软策略检查（默认选路可见）
    ASSERT_NE(n2->op.opCustomCheck, nullptr);     // 4GB 门槛保留

    const AlgAttrs* n3 = AlgAttrsRegistry::Instance().Get("AicpuScatterSequenceMeshConcurNHRNHR");
    ASSERT_NE(n3, nullptr);
    EXPECT_EQ(n3->topo.minTopoLevelNum, 3u);
    EXPECT_EQ(n3->topo.maxTopoLevelNum, 3u);
}

// 选路方向一/二：显式配置命中 + 未配置默认排除（真缝隙：既有注册零变化 + 默认选路零变化）
TEST(ScatterRhdSelectionTest, ExplicitConfigSelectsRhdOnTwoLevelTopo)
{
    auto topo2 = MakeTopo(2, Level0Shape::MESH_1D);
    // 显式配置（软策略让位 needSoftPolicyCheck=false）+ 二级拓扑 → 命中
    EXPECT_TRUE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHD", &topo2, false).matched);
    // 未配置（默认选路软策略检查）→ topoCustomCheck 恒 false → 被过滤（默认选路零变化）
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHD", &topo2, true).matched);
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHDRHD", &topo2, true).matched);
    // 既有 NHR 默认选路行为不变（软策略检查通过，仍为默认候选）
    EXPECT_TRUE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurNHR", &topo2, true).matched);
    auto topo3 = MakeTopo(3, Level0Shape::MESH_1D);
    EXPECT_TRUE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurNHRNHR", &topo3, true).matched);
}

// 选路方向三：形态与拓扑层级不符被硬 attrs 排除（选路唯一候选被排除 →
// SelectMinCost 空表 → HCCL_E_NOT_SUPPORT；本缝隙断言其过滤半边）；单级与 L0 形态不符同理
TEST(ScatterRhdSelectionTest, LevelAndShapeMismatchRejected)
{
    auto topo3 = MakeTopo(3, Level0Shape::MESH_1D);
    auto topo2 = MakeTopo(2, Level0Shape::MESH_1D);
    auto topo1 = MakeTopo(1, Level0Shape::MESH_1D);
    // 三级拓扑配二级形态 → maxTopoLevelNum 过滤
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHD", &topo3, false).matched);
    // 二级拓扑配三级形态 → minTopoLevelNum 过滤
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHDRHD", &topo2, false).matched);
    // 三级注册在三级拓扑命中
    EXPECT_TRUE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHDRHD", &topo3, false).matched);
    // 单级拓扑 → minTopoLevelNum=2 过滤（单级无 RHD）
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHD", &topo1, false).matched);
    // L0 形态不符（MESH_1D_CLOS 不在 supportLevel0Topos=MESH_1D 内）→ 过滤（硬 attrs 形态门槛）
    auto topo2Clos = MakeTopo(2, Level0Shape::MESH_1D_CLOS);
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHD", &topo2Clos, false).matched);
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHDRHD", &topo2Clos, false).matched);
}

// 注册名逐段落在 GetAlgoNameToTypeMap 键内（名字可被 alg_parse 解析回逐层 AlgoType）
TEST(ScatterRhdRegistrationTest, AlgoTypesParsedFromRegistrationName)
{
    const AlgAttrs* a2 = AlgAttrsRegistry::Instance().Get("AicpuScatterSequenceMeshConcurRHD");
    ASSERT_NE(a2, nullptr);
    EXPECT_EQ(a2->algoTypes.size(), 2u);
    ASSERT_GE(a2->algoTypes.size(), 2u);
    EXPECT_EQ(a2->algoTypes[0], AlgoType::MESH_CONCUR);
    EXPECT_EQ(a2->algoTypes[1], AlgoType::RHD);

    const AlgAttrs* a3 = AlgAttrsRegistry::Instance().Get("AicpuScatterSequenceMeshConcurRHDRHD");
    ASSERT_NE(a3, nullptr);
    EXPECT_EQ(a3->algoTypes.size(), 3u);
    ASSERT_GE(a3->algoTypes.size(), 3u);
    EXPECT_EQ(a3->algoTypes[0], AlgoType::MESH_CONCUR);
    EXPECT_EQ(a3->algoTypes[1], AlgoType::RHD);
    EXPECT_EQ(a3->algoTypes[2], AlgoType::RHD);
}

// DPU 二级注册：真注册表 attrs + host-DPU 形态双向准入 + 默认排除
TEST(ScatterRhdRegistrationTest, DpuRhdAttrsRegisteredFromProduction)
{
    const AlgAttrs* d2 = AlgAttrsRegistry::Instance().Get("DpuScatterSequenceMeshConcurRHD");
    ASSERT_NE(d2, nullptr);
    EXPECT_EQ(d2->opType, HcclCMDType::HCCL_CMD_SCATTER);
    EXPECT_EQ(d2->engine, OpExecuteConfig::HOSTCPU);
    EXPECT_EQ(d2->topo.minTopoLevelNum, 2u);
    EXPECT_EQ(d2->topo.maxTopoLevelNum, 2u);
    EXPECT_EQ(d2->topo.supportLevel0Topos, LEVEL0_TOPO_MESH_1D);
    EXPECT_TRUE(d2->topo.isHostDpuOnly);
    ASSERT_NE(d2->topo.topoCustomCheck, nullptr);
    EXPECT_FALSE(d2->topo.topoCustomCheck(nullptr)); // preferred-only
    EXPECT_EQ(d2->algoTypes.size(), 2u);
    ASSERT_GE(d2->algoTypes.size(), 2u);
    EXPECT_EQ(d2->algoTypes[0], AlgoType::MESH_CONCUR);
    EXPECT_EQ(d2->algoTypes[1], AlgoType::RHD);
}

// DPU 二级选路（真缝隙）：host-DPU 形态命中；非 host-DPU 形态被 isHostDpuOnly 排除；
// 未配置默认排除；AICPU 注册在 host-DPU 形态被反向排除（与既有 NHR DPU 同口径）
TEST(ScatterRhdSelectionTest, DpuRhdSelectionOnHostDpuForm)
{
    auto topo2Dpu = MakeTopo(2, Level0Shape::MESH_1D, 8, true);
    auto topo2NonDpu = MakeTopo(2, Level0Shape::MESH_1D, 8, false);
    // host-DPU 二级形态 + 显式配置 → 命中
    EXPECT_TRUE(CheckAlgoMatchTopoWithReason("DpuScatterSequenceMeshConcurRHD", &topo2Dpu, false).matched);
    // 非 host-DPU 形态 → isHostDpuOnly 过滤
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("DpuScatterSequenceMeshConcurRHD", &topo2NonDpu, false).matched);
    // 未配置（默认选路软策略）→ topoCustomCheck 恒 false 被过滤（默认选路零变化）
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("DpuScatterSequenceMeshConcurRHD", &topo2Dpu, true).matched);
    // host-DPU 形态下 AICPU 注册被反向排除（isHostDpuOnly=false vs topo.hostDpuOnly=true）
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("AicpuScatterSequenceMeshConcurRHD", &topo2Dpu, false).matched);
    // 既有 DpuScatterSequenceMeshNHR 在 host-DPU 形态默认行为不变（软策略检查通过）
    EXPECT_TRUE(CheckAlgoMatchTopoWithReason("DpuScatterSequenceMeshNHR", &topo2Dpu, true).matched);
}

// DPU 三级注册：真注册表 attrs + 三级 host-DPU 形态命中
TEST(ScatterRhdRegistrationTest, DpuRhd3LevelAttrsRegisteredFromProduction)
{
    const AlgAttrs* d3 = AlgAttrsRegistry::Instance().Get("DpuScatterSequenceMeshConcurRHDRHD");
    ASSERT_NE(d3, nullptr);
    EXPECT_EQ(d3->opType, HcclCMDType::HCCL_CMD_SCATTER);
    EXPECT_EQ(d3->engine, OpExecuteConfig::HOSTCPU);
    EXPECT_EQ(d3->topo.minTopoLevelNum, 3u);
    EXPECT_EQ(d3->topo.maxTopoLevelNum, 3u);
    EXPECT_EQ(d3->topo.supportLevel0Topos, LEVEL0_TOPO_MESH_1D);
    EXPECT_TRUE(d3->topo.isHostDpuOnly);
    ASSERT_NE(d3->topo.topoCustomCheck, nullptr);
    EXPECT_FALSE(d3->topo.topoCustomCheck(nullptr)); // preferred-only
    EXPECT_EQ(d3->algoTypes.size(), 3u);
    ASSERT_GE(d3->algoTypes.size(), 3u);
    EXPECT_EQ(d3->algoTypes[0], AlgoType::MESH_CONCUR);
    EXPECT_EQ(d3->algoTypes[1], AlgoType::RHD);
    EXPECT_EQ(d3->algoTypes[2], AlgoType::RHD);
}

// DPU 三级选路（真缝隙）：三级 host-DPU 形态命中；层级不符/形态不符/默认三方向被拒
TEST(ScatterRhdSelectionTest, DpuRhd3LevelSelection)
{
    auto topo3Dpu = MakeTopo(3, Level0Shape::MESH_1D, 8, true);
    auto topo3NonDpu = MakeTopo(3, Level0Shape::MESH_1D, 8, false);
    auto topo2Dpu = MakeTopo(2, Level0Shape::MESH_1D, 8, true);
    // 三级 host-DPU 形态 + 显式配置 → 命中
    EXPECT_TRUE(CheckAlgoMatchTopoWithReason("DpuScatterSequenceMeshConcurRHDRHD", &topo3Dpu, false).matched);
    // 非 host-DPU 形态 → isHostDpuOnly 过滤
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("DpuScatterSequenceMeshConcurRHDRHD", &topo3NonDpu, false).matched);
    // 二级拓扑配三级形态 → minTopoLevelNum 过滤（选路唯一候选被排除 → E_NOT_SUPPORT）
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("DpuScatterSequenceMeshConcurRHDRHD", &topo2Dpu, false).matched);
    // 未配置（默认选路软策略）→ topoCustomCheck 恒 false 被过滤
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("DpuScatterSequenceMeshConcurRHDRHD", &topo3Dpu, true).matched);
    // L0 形态不符（CLOS）→ supportLevel0Topos 过滤
    auto topo3DpuClos = MakeTopo(3, Level0Shape::MESH_1D_CLOS, 8, true);
    EXPECT_FALSE(CheckAlgoMatchTopoWithReason("DpuScatterSequenceMeshConcurRHDRHD", &topo3DpuClos, false).matched);
}
