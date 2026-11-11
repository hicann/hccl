/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 断言 AllToAllV / AllToAllVC Pairwise 选路修复的生产注册行为（PR 核心三项变更）：
//   1) AicpuAllToAllVSolePairwise / AicpuAllToAllVCSolePairwise 新增 topoPriorityCheck，
//      与 topoCustomCheck 共用同一收紧条件（deviceNumPerModule==8、rankSize>=16 且 %16==0、
//      level0Symmetric、<=128P）；
//   2) AicpuAllToAllVSoleMesh 移除对 Pairwise 场景的 topoCustomCheck 硬排除，
//      恢复 supportLevel0Topos = LEVEL0_TOPO_ANY，Pairwise 场景仍保留候选兜底；
//   3) AicpuAllToAllVCSolePairwise 同步收紧准入（level0Symmetric && <=128P），
//      超出范围由 AicpuAllToAllVCSoleMesh（无任何 customCheck）兜底。
// attrs 经 wrap/ins_v2_all_to_all_v_attrs_wrap.cc 以生产源文件直接编入本测试进程，
// 断言对象为 AlgAttrsRegistry 中的生产注册项，非测试内复刻条件。
//
// 覆盖范围声明：本 UT 仅断言 topoCustomCheck / topoPriorityCheck 的注册行为（lambda
// 真值），不覆盖 HCCL_ALGO 否定 Pairwise 后与 topoPriorityCheck 排他的交互语义。
// 该交互发生在 cost_model.cc::ApplyTopoPriority 与 cost_table.cc::ConvergeHcclAlgoTier
// 中，这两个 TU 未编入本 UT 目标（CMakeLists.txt 仅含 alg_parse.cc + registry + wrap）。
// stub（stub/pairwise_alg_attrs_stub.cc）提供 HcclGetHcclAlgo / GetEnv 的可注入
// accessor（SetHcclAlgoOverride / SetEnvOverride），HCCL_ALGOInjectNotAffectTopoPriorityCheck
// 用例消费该入口验证 topoPriorityCheck lambda 不感知 HCCL_ALGO 配置（排他覆盖风险的根因），
// 但完整的 ApplyTopoPriority 排他语义仍需 cost_model 层 UT 覆盖。

#include "gtest/gtest.h"
#include <string>
#include "alg_param.h"
#include "alg_attrs_registry.h"
#include "pairwise_alg_attrs_stub.h"

using namespace ops_hccl;

namespace {
// Pairwise 生产条件常量（ins_v2_all_to_all_v_sole_executor.cc /
// ins_temp_all_to_all_v_pairwise.h）：RANK_NUM_PER_BOARD=8、THREAD_SET_NUM=2、
// PAIRWISE_MAX_RANK_SIZE=128。此处按语义使用字面值并断言生产 lambda 行为，
// 生产条件变更（如上限调整、去掉对称性约束）将使对应断言失败，起到回归锁定作用。
constexpr u32 PAIRWISE_DEV_PER_MODULE = 8;
constexpr u32 PAIRWISE_UNIT_RANK_SIZE = 16;
constexpr u32 PAIRWISE_MAX_RANK_SIZE = 128;

TopoInfoWithNetLayerDetails MakeTopo(u32 rankSize, u32 devPerModule, bool symmetric)
{
    TopoInfoWithNetLayerDetails t;
    t.userRankSize = rankSize;
    t.deviceNumPerModule = devPerModule;
    t.level0Symmetric = symmetric;
    t.level0Topo = Level0Shape::MESH_1D; // Pairwise 注册的 supportLevel0Topos
    t.topoLevelNums = 2;                 // 跨框场景（minTopoLevelNum = TOPO_LEVEL_NUM_2）
    return t;
}

const AlgAttrs* MustGetAttrs(const std::string& name)
{
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(name);
    EXPECT_NE(nullptr, attrs) << "attrs not registered: " << name;
    return attrs;
}
} // namespace

// V / VC / Mesh 四个算法均注册在 (opType, engine) = (ALLTOALLV|ALLTOALLVC, AICPU_TS) 组，
// topoPriorityCheck 的排他粒度即该组，先锁定组归属。
TEST(PairwiseAlgAttrsTest, RegisteredInExpectedOpTypeAndEngineGroup)
{
    const std::pair<const char*, HcclCMDType> expectations[] = {
        {"AicpuAllToAllVSolePairwise", HcclCMDType::HCCL_CMD_ALLTOALLV},
        {"AicpuAllToAllVSoleMesh", HcclCMDType::HCCL_CMD_ALLTOALLV},
        {"AicpuAllToAllVCSolePairwise", HcclCMDType::HCCL_CMD_ALLTOALLVC},
        {"AicpuAllToAllVCSoleMesh", HcclCMDType::HCCL_CMD_ALLTOALLVC},
    };
    for (const auto& item : expectations) {
        const AlgAttrs* attrs = MustGetAttrs(item.first);
        ASSERT_NE(nullptr, attrs);
        EXPECT_EQ(item.second, attrs->opType) << item.first;
        EXPECT_EQ(OpExecuteConfig::AICPU_TS, attrs->engine) << item.first;
    }
}

// 变更 1a：V Pairwise 场景（对称、16 倍数且 <=128P）customCheck 与 priorityCheck 同时命中，
// 命中 priorityCheck 后同组未声明 priority 的 Mesh 被排他。
TEST(PairwiseAlgAttrsTest, VPairwiseCustomAndPriorityCheckAcceptPairwiseScenario)
{
    const AlgAttrs* attrs = MustGetAttrs("AicpuAllToAllVSolePairwise");
    ASSERT_NE(nullptr, attrs);
    ASSERT_NE(nullptr, attrs->topo.topoCustomCheck);
    ASSERT_NE(nullptr, attrs->topo.topoPriorityCheck);

    const u32 positiveRankSizes[] = {PAIRWISE_UNIT_RANK_SIZE, 64, PAIRWISE_MAX_RANK_SIZE};
    for (u32 rankSize : positiveRankSizes) {
        auto t = MakeTopo(rankSize, PAIRWISE_DEV_PER_MODULE, true);
        EXPECT_TRUE(attrs->topo.topoCustomCheck(&t)) << "rankSize=" << rankSize;
        EXPECT_TRUE(attrs->topo.topoPriorityCheck(&t)) << "rankSize=" << rankSize;
    }
}

// 变更 1b：V Pairwise 非准入场景（单框 8P、24P 非整除、4 卡模块、>128P、非对称）
// customCheck 与 priorityCheck 同时不命中：Pairwise 被硬过滤 / 不参与排他，Mesh 保留候选。
TEST(PairwiseAlgAttrsTest, VPairwiseChecksRejectNonPairwiseScenario)
{
    const AlgAttrs* attrs = MustGetAttrs("AicpuAllToAllVSolePairwise");
    ASSERT_NE(nullptr, attrs);
    ASSERT_NE(nullptr, attrs->topo.topoCustomCheck);
    ASSERT_NE(nullptr, attrs->topo.topoPriorityCheck);

    const TopoInfoWithNetLayerDetails negatives[] = {
        MakeTopo(PAIRWISE_DEV_PER_MODULE, PAIRWISE_DEV_PER_MODULE, true),    // 8P：低于 16P 下限
        MakeTopo(24, PAIRWISE_DEV_PER_MODULE, true),                         // 24P：非 16 倍数
        MakeTopo(PAIRWISE_UNIT_RANK_SIZE, 4, true),                          // 每模块 4 卡
        MakeTopo(2 * PAIRWISE_MAX_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, true), // 256P：超 128P 上限
        MakeTopo(PAIRWISE_MAX_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, false),    // 128P 非对称
    };
    for (const auto& t : negatives) {
        EXPECT_FALSE(attrs->topo.topoCustomCheck(&t)) << "rankSize=" << t.userRankSize;
        EXPECT_FALSE(attrs->topo.topoPriorityCheck(&t)) << "rankSize=" << t.userRankSize;
    }
}

// 变更 3：VC Pairwise 与 V 同构 —— customCheck / priorityCheck 共用同一收紧条件，
// 新增 level0Symmetric 与 <=128P 守卫。
TEST(PairwiseAlgAttrsTest, VCPairwiseCustomAndPriorityCheckAcceptPairwiseScenario)
{
    const AlgAttrs* attrs = MustGetAttrs("AicpuAllToAllVCSolePairwise");
    ASSERT_NE(nullptr, attrs);
    ASSERT_NE(nullptr, attrs->topo.topoCustomCheck);
    ASSERT_NE(nullptr, attrs->topo.topoPriorityCheck);

    const u32 positiveRankSizes[] = {PAIRWISE_UNIT_RANK_SIZE, 64, PAIRWISE_MAX_RANK_SIZE};
    for (u32 rankSize : positiveRankSizes) {
        auto t = MakeTopo(rankSize, PAIRWISE_DEV_PER_MODULE, true);
        EXPECT_TRUE(attrs->topo.topoCustomCheck(&t)) << "rankSize=" << rankSize;
        EXPECT_TRUE(attrs->topo.topoPriorityCheck(&t)) << "rankSize=" << rankSize;
    }
}

TEST(PairwiseAlgAttrsTest, VCPairwiseChecksRejectNonPairwiseScenario)
{
    const AlgAttrs* attrs = MustGetAttrs("AicpuAllToAllVCSolePairwise");
    ASSERT_NE(nullptr, attrs);
    ASSERT_NE(nullptr, attrs->topo.topoCustomCheck);
    ASSERT_NE(nullptr, attrs->topo.topoPriorityCheck);

    const TopoInfoWithNetLayerDetails negatives[] = {
        MakeTopo(PAIRWISE_DEV_PER_MODULE, PAIRWISE_DEV_PER_MODULE, true),
        MakeTopo(24, PAIRWISE_DEV_PER_MODULE, true),
        MakeTopo(PAIRWISE_UNIT_RANK_SIZE, 4, true),
        MakeTopo(2 * PAIRWISE_MAX_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, true),
        MakeTopo(PAIRWISE_MAX_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, false),
    };
    for (const auto& t : negatives) {
        EXPECT_FALSE(attrs->topo.topoCustomCheck(&t)) << "rankSize=" << t.userRankSize;
        EXPECT_FALSE(attrs->topo.topoPriorityCheck(&t)) << "rankSize=" << t.userRankSize;
    }
}

// 变更 2：V Mesh 移除对 Pairwise 场景的 topoCustomCheck 硬排除 ——
// Pairwise 场景（16P 对称 MESH_1D）Mesh 仍返回 true 参与候选兜底；
// Mesh 不声明 topoPriorityCheck，HCCL_ALGO 正向指定 / tuner 接管时可绕过排他。
TEST(PairwiseAlgAttrsTest, VMeshCustomCheckNoLongerExcludesPairwiseScenario)
{
    const AlgAttrs* attrs = MustGetAttrs("AicpuAllToAllVSoleMesh");
    ASSERT_NE(nullptr, attrs);
    ASSERT_NE(nullptr, attrs->topo.topoCustomCheck);
    EXPECT_EQ(nullptr, attrs->topo.topoPriorityCheck);
    EXPECT_EQ(LEVEL0_TOPO_ANY, attrs->topo.supportLevel0Topos);

    auto t = MakeTopo(PAIRWISE_UNIT_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, true);
    EXPECT_TRUE(attrs->topo.topoCustomCheck(&t));
}

// V Mesh 保留既有 MESH_1D_CLOS 的 PCIe 混连守卫（非本次变更，回归锁定）。
TEST(PairwiseAlgAttrsTest, VMeshCustomCheckKeepsMesh1dClosPcieMixGuard)
{
    const AlgAttrs* attrs = MustGetAttrs("AicpuAllToAllVSoleMesh");
    ASSERT_NE(nullptr, attrs);
    ASSERT_NE(nullptr, attrs->topo.topoCustomCheck);

    TopoInfoWithNetLayerDetails closNoMix = MakeTopo(PAIRWISE_UNIT_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, true);
    closNoMix.level0Topo = Level0Shape::MESH_1D_CLOS;
    closNoMix.level0PcieMix = false;
    EXPECT_FALSE(attrs->topo.topoCustomCheck(&closNoMix));

    TopoInfoWithNetLayerDetails closMix = MakeTopo(PAIRWISE_UNIT_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, true);
    closMix.level0Topo = Level0Shape::MESH_1D_CLOS;
    closMix.level0PcieMix = true;
    EXPECT_TRUE(attrs->topo.topoCustomCheck(&closMix));
}

// VC Mesh 无任何 topoCustomCheck / topoPriorityCheck（注册块本次未触碰，回归锁定）：
// Pairwise 准入收紧后，超 128P / 非对称场景由其 LEVEL0_TOPO_ANY 兜底。
TEST(PairwiseAlgAttrsTest, VCMeshHasNoExclusionAndAnyTopoSupport)
{
    const AlgAttrs* attrs = MustGetAttrs("AicpuAllToAllVCSoleMesh");
    ASSERT_NE(nullptr, attrs);
    EXPECT_EQ(nullptr, attrs->topo.topoCustomCheck);
    EXPECT_EQ(nullptr, attrs->topo.topoPriorityCheck);
    EXPECT_EQ(LEVEL0_TOPO_ANY, attrs->topo.supportLevel0Topos);
}

// 排他净效果验证：ApplyTopoPriority 的排他条件为「keepSet.count(i)==0 &&
// hcclAlgoPriority<=0」（cost_model.cc:487-488），keepSet 收集条件为
// topoPriorityCheck 命中（cost_model.cc:467）。本用例锁定排他生效的前提条件组合：
//   1) Pairwise 在 Pairwise 场景 topoPriorityCheck 返回 true → 进入 keepSet；
//   2) Mesh 不声明 topoPriorityCheck（nullptr）→ 不进入 keepSet；
//   3) Mesh 在 Pairwise 场景仍通过 topoCustomCheck（存活到排他阶段）。
// 条件 1/2/3 均在本用例断言。排他删除条件还依赖 Mesh 的 hcclAlgoPriority <= 0，
// 但 hcclAlgoPriority 是 CostAlgoParams（cost_model.h:111）的字段、默认值为 0，
// 不在 AlgAttrs 注册层、且 cost_model.cc 未编入本 UT 目标，故该值无法在此断言。
// 若将来 Mesh 在注册时设置了非 0 的 hcclAlgoPriority（>0），排他删除条件不再成立，
// 本用例无法捕获——该路径需在 cost_model 层的 UT 中覆盖。
TEST(PairwiseAlgAttrsTest, TopoPriorityExclusionNetEffectOnPairwiseScenario)
{
    const AlgAttrs* pairwise = MustGetAttrs("AicpuAllToAllVSolePairwise");
    const AlgAttrs* mesh = MustGetAttrs("AicpuAllToAllVSoleMesh");
    ASSERT_NE(nullptr, pairwise);
    ASSERT_NE(nullptr, mesh);

    auto t = MakeTopo(PAIRWISE_UNIT_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, true);

    // 条件 1：Pairwise 命中 topoPriorityCheck → 进入 keepSet
    ASSERT_NE(nullptr, pairwise->topo.topoPriorityCheck);
    EXPECT_TRUE(pairwise->topo.topoPriorityCheck(&t));

    // 条件 2：Mesh 不声明 topoPriorityCheck → 不进入 keepSet
    EXPECT_EQ(nullptr, mesh->topo.topoPriorityCheck);

    // 条件 3：Mesh 在 Pairwise 场景仍通过 topoCustomCheck（存活到排他阶段）
    ASSERT_NE(nullptr, mesh->topo.topoCustomCheck);
    EXPECT_TRUE(mesh->topo.topoCustomCheck(&t));
}

// VC 侧排他净效果同构：Pairwise 命中 priorityCheck，Mesh 不声明 priorityCheck 且
// VC Mesh 无 customCheck（LEVEL0_TOPO_ANY 任意拓扑存活）。hcclAlgoPriority 限制同 V 侧。
TEST(PairwiseAlgAttrsTest, VCTopoPriorityExclusionNetEffectOnPairwiseScenario)
{
    const AlgAttrs* pairwise = MustGetAttrs("AicpuAllToAllVCSolePairwise");
    const AlgAttrs* mesh = MustGetAttrs("AicpuAllToAllVCSoleMesh");
    ASSERT_NE(nullptr, pairwise);
    ASSERT_NE(nullptr, mesh);

    auto t = MakeTopo(PAIRWISE_UNIT_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, true);

    ASSERT_NE(nullptr, pairwise->topo.topoPriorityCheck);
    EXPECT_TRUE(pairwise->topo.topoPriorityCheck(&t));

    EXPECT_EQ(nullptr, mesh->topo.topoPriorityCheck);

    // VC Mesh 无 topoCustomCheck（LEVEL0_TOPO_ANY），任何拓扑均存活到排他阶段
    EXPECT_EQ(nullptr, mesh->topo.topoCustomCheck);
    EXPECT_EQ(LEVEL0_TOPO_ANY, mesh->topo.supportLevel0Topos);
}

// 非 V 的 AllToAll Pairwise 刻意不收紧（无 128P / 对称守卫）、不声明 topoPriorityCheck：
// AllToAll cost 按数据量计算，同引擎内可用 cost 竞争区分算法优先级（注册处注释已声明该差异）。
TEST(PairwiseAlgAttrsTest, AllToAllPairwiseKeepsCostCompetitionCondition)
{
    const AlgAttrs* attrs = MustGetAttrs("AicpuAllToAllSolePairwise");
    ASSERT_NE(nullptr, attrs);
    ASSERT_NE(nullptr, attrs->topo.topoCustomCheck);
    EXPECT_EQ(nullptr, attrs->topo.topoPriorityCheck);

    // >128P 且满足基础条件（8 卡模块、16 倍数）仍参与 cost 竞争
    auto overLimit = MakeTopo(2 * PAIRWISE_MAX_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, true);
    EXPECT_TRUE(attrs->topo.topoCustomCheck(&overLimit));

    // 非对称仍参与 cost 竞争
    auto asymmetric = MakeTopo(PAIRWISE_UNIT_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, false);
    EXPECT_TRUE(attrs->topo.topoCustomCheck(&asymmetric));

    // 基础条件（8 卡模块、>=16P、%16==0）仍为准入硬条件
    auto singleBoard = MakeTopo(PAIRWISE_DEV_PER_MODULE, PAIRWISE_DEV_PER_MODULE, true);
    EXPECT_FALSE(attrs->topo.topoCustomCheck(&singleBoard));
    auto notDivisible = MakeTopo(24, PAIRWISE_DEV_PER_MODULE, true);
    EXPECT_FALSE(attrs->topo.topoCustomCheck(&notDivisible));
}

// 验证 topoPriorityCheck lambda 不感知 HCCL_ALGO 用户配置：即使 stub 注入了
// "用户显式否定 Pairwise" 的 HCCL_ALGO 值，topoPriorityCheck 返回值仍仅取决于拓扑条件。
// 这是 ApplyTopoPriority 排他可能覆盖用户否定意图的根因——topoPriorityCheck 在
// cost_model.cc:467 被 keepSet 收集时不检查 hcclAlgoPriority，排他删除条件
// (cost_model.cc:487-488) 的 hcclAlgoPriority<=0 在 AlgAttrs 层也不可见。
// 因此此处断言的是"topoPriorityCheck 不读 HCCL_ALGO"这一事实，而非排他语义本身。
// TODO: 排他语义（hcclAlgoPriority<=0 删除条件 + ConvergeHcclAlgoTier 否定恢复）
// 需在 cost_model / cost_table 层 UT 覆盖，本 PR 不含该层测试。
TEST(PairwiseAlgAttrsTest, HCCL_ALGOInjectNotAffectTopoPriorityCheck)
{
    const AlgAttrs* pairwise = MustGetAttrs("AicpuAllToAllVSolePairwise");
    ASSERT_NE(nullptr, pairwise);
    ASSERT_NE(nullptr, pairwise->topo.topoPriorityCheck);

    auto t = MakeTopo(PAIRWISE_UNIT_RANK_SIZE, PAIRWISE_DEV_PER_MODULE, true);

    // 基线：默认（未配置 HCCL_ALGO）Pairwise 场景命中
    ResetAlgAttrsStub();
    EXPECT_TRUE(pairwise->topo.topoPriorityCheck(&t));

    // 注入"用户显式否定 Pairwise"后，topoPriorityCheck 仍命中——
    // lambda 不读取 HcclGetHcclAlgo / GetEnv 的返回值
    SetHcclAlgoOverride("not(pairwise)");
    SetEnvOverride("HCCL_ALGO", "not(pairwise)");
    EXPECT_TRUE(pairwise->topo.topoPriorityCheck(&t));

    // 恢复默认，避免影响后续用例
    ResetAlgAttrsStub();
}
