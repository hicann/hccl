/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "ins_v2_all_to_all_sequence_executor.h"
#include "topo_match_two_level.h"
#include "ccu_alltoall_stage_template_registry.h"

namespace ops_hccl {

namespace ut_stub {
    void InjectProbeAlgAttrs(const std::string& name, const AlgAttrs& attrs);
    void ResetEngineCtx();
    void* LastEngineCtx();
    void ResetLaunchRecords();
    size_t LaunchRecordCount();
    const std::vector<uint64_t>& LaunchRecordArgs(size_t idx);
} // namespace ut_stub

namespace {
    TopoInfoWithNetLayerDetails MakeTwoLevelTopo(u32 myRank, u32 userRankSize)
    {
        TopoInfoWithNetLayerDetails topo;
        topo.userRank = myRank;
        topo.userRankSize = userRankSize;

        PhysicalLevelInfo level0;
        for (u32 i = 0; i < userRankSize / 2; i++) {
            level0.localRanks.push_back(i);
        }
        level0.view = PhysicalLevelView::GLOBAL;
        level0.instSizeListByLayer = {userRankSize / 2, userRankSize / 2};
        level0.hasTopoInst = true;
        level0.topoType = CommTopo::COMM_TOPO_CLOS;
        level0.locType = ENDPOINT_LOC_TYPE_DEVICE;
        level0.portNums = {1};

        PhysicalLevelInfo level1;
        for (u32 i = 0; i < userRankSize; i++) {
            level1.localRanks.push_back(i);
        }
        level1.view = PhysicalLevelView::GLOBAL;
        level1.instSizeListByLayer = {userRankSize};
        level1.hasTopoInst = true;
        level1.topoType = CommTopo::COMM_TOPO_CLOS;
        level1.locType = ENDPOINT_LOC_TYPE_DEVICE;
        level1.portNums = {8};

        topo.physicalLevels = {std::move(level0), std::move(level1)};
        topo.physicalLevelNum = static_cast<u32>(topo.physicalLevels.size());
        topo.topoLevelNums = 2; // 独立于 physicalLevels，用例按需覆盖
        return topo;
    }

    AlgAttrs MakeTwoLevelAttrs()
    {
        AlgAttrs attrs;
        attrs.name = "ut";
        attrs.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
        attrs.engine = OpExecuteConfig::AICPU_TS;
        attrs.algoTypes = {AlgoType::MESH, AlgoType::NHR};
        return attrs;
    }

    // GCD=1 非对称 + HOST 锚点层：HOSTCPU 豁免后 MatchTopo 成功而 d0=1
    TopoInfoWithNetLayerDetails MakeGcdOneHostTopo(u32 myRank, u32 userRankSize)
    {
        TopoInfoWithNetLayerDetails topo;
        topo.userRank = myRank;
        topo.userRankSize = userRankSize;

        PhysicalLevelInfo level0; // 非对称 instList → GCD=1
        for (u32 i = 0; i < userRankSize; i++) {
            level0.localRanks.push_back(i);
        }
        level0.view = PhysicalLevelView::GLOBAL;
        level0.instSizeListByLayer = {userRankSize / 2, userRankSize - userRankSize / 2};
        level0.hasTopoInst = true;
        level0.topoType = CommTopo::COMM_TOPO_CLOS;
        level0.locType = ENDPOINT_LOC_TYPE_DEVICE;

        PhysicalLevelInfo level1; // HOST 锚点层
        for (u32 i = 0; i < userRankSize; i++) {
            level1.localRanks.push_back(i);
        }
        level1.view = PhysicalLevelView::GLOBAL;
        level1.instSizeListByLayer = {userRankSize};
        level1.hasTopoInst = true;
        level1.topoType = CommTopo::COMM_TOPO_CLOS;
        level1.locType = ENDPOINT_LOC_TYPE_HOST;

        topo.physicalLevels = {std::move(level0), std::move(level1)};
        topo.physicalLevelNum = static_cast<u32>(topo.physicalLevels.size());
        topo.topoLevelNums = 2;
        return topo;
    }
} // namespace

class CcuA2ASequenceExecutorTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        param_ = OpParam{};
        param_.engine = CommEngine::COMM_ENGINE_CCU;
        topo_ = MakeTwoLevelTopo(3, 16);
        attrs_ = MakeTwoLevelAttrs();
        ut_stub::ResetEngineCtx();
    }

    void TearDown() override { ut_stub::ResetEngineCtx(); }

    OpParam param_{};
    TopoInfoWithNetLayerDetails topo_{};
    AlgAttrs attrs_{};
    std::vector<std::vector<u32>> subCommRanks_ = {{3, 11}};
};

using TestSeqExecutor = InsV2AlltoAllSequenceExecutor<TopoMatchTwoLevel>;

// GetStageAlgos 契约：CCU 二级列表与注册表可创建性一致
TEST_F(CcuA2ASequenceExecutorTest, GetStageAlgosContract)
{
    TestSeqExecutor executor;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    ASSERT_EQ(executor.CalcAlgHierarchyInfoV2(&topo_, algHierarchyInfo, attrs_), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(algHierarchyInfo.infos.size(), 2u);

    std::vector<std::string> ccuAlgos;
    EXPECT_EQ(executor.GetStageAlgos(param_, 2, ccuAlgos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuAlgos, (std::vector<std::string>{"FirstMeshStage", "LastMeshStage"}));

    OpParam nonCcu{};
    std::vector<std::string> aicpuAlgos;
    EXPECT_EQ(executor.GetStageAlgos(nonCcu, 2, aicpuAlgos), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(aicpuAlgos, (std::vector<std::string>{"MeshStage", "MeshStage"}));

    std::vector<std::string> rejected;
    EXPECT_EQ(executor.GetStageAlgos(param_, 3, rejected), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(executor.GetStageAlgos(param_, 1, rejected), HcclResult::HCCL_E_INTERNAL);

    // 伪造三级层级穿透前置校验 → "CCU 仅 2 级"拦截
    executor.algHierarchyInfo_.infos.resize(3);
    EXPECT_EQ(executor.GetStageAlgos(param_, 3, rejected), HcclResult::HCCL_E_INTERNAL);

    for (const auto& algoName : ccuAlgos) {
        auto stage
            = CcuAlltoAllStageTemplateRegistry::Instance().Create(algoName, param_, topo_.userRank, subCommRanks_, 1);
        EXPECT_NE(stage, nullptr) << "algoName=" << algoName;
    }
}

// BuildStageTemplates 二级全链装配
TEST_F(CcuA2ASequenceExecutorTest, BuildStageTemplatesTwoLevelCcu)
{
    TestSeqExecutor executor;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    ASSERT_EQ(executor.CalcAlgHierarchyInfoV2(&topo_, algHierarchyInfo, attrs_), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(algHierarchyInfo.infos.size(), 2u);

    ASSERT_EQ(executor.BuildStageTemplates(param_), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(executor.stageAlgos_, (std::vector<std::string>{"FirstMeshStage", "LastMeshStage"}));
    ASSERT_EQ(executor.ccuStageTemplates_.size(), 2u);
    EXPECT_NE(executor.ccuStageTemplates_[0], nullptr);
    EXPECT_NE(executor.ccuStageTemplates_[1], nullptr);
}

// topoLevelNums=3 打平为算法 2 级
TEST_F(CcuA2ASequenceExecutorTest, BuildStageTemplatesFlattensThreeLevelTopo)
{
    TestSeqExecutor executor;
    topo_.topoLevelNums = 3; // physicalLevels 仍两级（两字段独立）
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    ASSERT_EQ(executor.CalcAlgHierarchyInfoV2(&topo_, algHierarchyInfo, attrs_), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(executor.totalStages_, 2u); // totalStages_ = infos.size()
    EXPECT_EQ(executor.BuildStageTemplates(param_), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(executor.stageAlgos_, (std::vector<std::string>{"FirstMeshStage", "LastMeshStage"}));
    ASSERT_EQ(executor.ccuStageTemplates_.size(), 2u);
    EXPECT_NE(executor.ccuStageTemplates_[0], nullptr);
    EXPECT_NE(executor.ccuStageTemplates_[1], nullptr);
}

// 真三级物理层压缩：算法级数恒 2，装配照常
TEST_F(CcuA2ASequenceExecutorTest, BuildStageTemplatesThreePhysicalLevelsCompressed)
{
    TestSeqExecutor executor;
    TopoInfoWithNetLayerDetails topo = MakeTwoLevelTopo(3, 16);
    PhysicalLevelInfo level2; // 模拟超节点间集群层
    for (u32 i = 0; i < topo.userRankSize; i++) {
        level2.localRanks.push_back(i);
    }
    level2.view = PhysicalLevelView::GLOBAL;
    level2.instSizeListByLayer = {topo.userRankSize};
    level2.hasTopoInst = true;
    level2.topoType = CommTopo::COMM_TOPO_CLOS;
    level2.locType = ENDPOINT_LOC_TYPE_DEVICE;
    topo.physicalLevels.push_back(std::move(level2));
    topo.physicalLevelNum = 3;
    topo.topoLevelNums = 3;

    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    ASSERT_EQ(executor.CalcAlgHierarchyInfoV2(&topo, algHierarchyInfo, attrs_), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(algHierarchyInfo.infos.size(), 2u);

    EXPECT_EQ(executor.totalStages_, 2u);
    EXPECT_EQ(executor.BuildStageTemplates(param_), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(executor.stageAlgos_, (std::vector<std::string>{"FirstMeshStage", "LastMeshStage"}));
}

// 退化级数（<2）拒绝，引擎无关
TEST_F(CcuA2ASequenceExecutorTest, BuildStageTemplatesDegenerateLevelRejected)
{
    TestSeqExecutor executor;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    ASSERT_EQ(executor.CalcAlgHierarchyInfoV2(&topo_, algHierarchyInfo, attrs_), HcclResult::HCCL_SUCCESS);
    OpParam aicpuParam{};
    aicpuParam.engine = CommEngine::COMM_ENGINE_AICPU_TS;

    executor.totalStages_ = 0; // 注入退化级数
    EXPECT_EQ(executor.BuildStageTemplates(param_), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(executor.BuildStageTemplates(aicpuParam), HcclResult::HCCL_E_INTERNAL);

    executor.totalStages_ = 1;
    EXPECT_EQ(executor.BuildStageTemplates(param_), HcclResult::HCCL_E_INTERNAL);
    EXPECT_EQ(executor.BuildStageTemplates(aicpuParam), HcclResult::HCCL_E_INTERNAL);
}

// CcuSched 界内返回两段实算；CCU 两段 C 合计 = 2μs×(1+2) = 6e-6，B/D 恒 0
TEST_F(CcuA2ASequenceExecutorTest, CalcCostCoeffCcuSchedInBoundsReturnsTwoSegParams)
{
    TestSeqExecutor executor;
    AlgAttrs attrs = MakeTwoLevelAttrs();
    attrs.name = "CcuSchedAllToAllSequenceMeshMesh";
    ut_stub::InjectProbeAlgAttrs(attrs.name, attrs);

    std::vector<CostModelParam> coeffs = executor.CalcCostCoeff(nullptr, &topo_, attrs.name.c_str(), param_);
    ASSERT_EQ(coeffs.size(), 2u);
    EXPECT_GT(coeffs[0].A, 0.0f);
    EXPECT_GT(coeffs[1].A, 0.0f);
    EXPECT_FLOAT_EQ(coeffs[0].C + coeffs[1].C, 6.0e-6f);
    EXPECT_FLOAT_EQ(coeffs[0].B + coeffs[1].B, 0.0f);
    EXPECT_FLOAT_EQ(coeffs[0].D + coeffs[1].D, 0.0f);
}

// d0=1（HOSTCPU 豁免后 MatchTopo 成功）→ 下界剔除
TEST_F(CcuA2ASequenceExecutorTest, CalcCostCoeffCcuSchedD0OutOfBoundReturnsEmpty)
{
    TestSeqExecutor executor;
    AlgAttrs attrs = MakeTwoLevelAttrs();
    attrs.name = "CcuSchedAllToAllSequenceMeshMesh";
    attrs.engine = OpExecuteConfig::HOSTCPU; // 豁免 GCD=1 拒绝
    ut_stub::InjectProbeAlgAttrs(attrs.name, attrs);

    TopoInfoWithNetLayerDetails gcdOneTopo = MakeGcdOneHostTopo(3, 7); // d0=GCD(3,4)=1
    std::vector<CostModelParam> coeffs = executor.CalcCostCoeff(nullptr, &gcdOneTopo, attrs.name.c_str(), param_);
    EXPECT_TRUE(coeffs.empty());
}

// d0=256 越 CCU 128 通道上限 → 剔除
TEST_F(CcuA2ASequenceExecutorTest, CalcCostCoeffCcuSchedD0ExceedsCcuChannelLimitReturnsEmpty)
{
    TestSeqExecutor executor;
    AlgAttrs attrs = MakeTwoLevelAttrs();
    attrs.name = "CcuSchedAllToAllSequenceMeshMesh";
    ut_stub::InjectProbeAlgAttrs(attrs.name, attrs);

    TopoInfoWithNetLayerDetails topo = MakeTwoLevelTopo(3, 512);
    std::vector<CostModelParam> coeffs = executor.CalcCostCoeff(nullptr, &topo, attrs.name.c_str(), param_);
    EXPECT_TRUE(coeffs.empty());
}

// d0=129 恰达上限 → 放行
TEST_F(CcuA2ASequenceExecutorTest, CalcCostCoeffCcuSchedBoundaryD0Allowed)
{
    TestSeqExecutor executor;
    AlgAttrs attrs = MakeTwoLevelAttrs();
    attrs.name = "CcuSchedAllToAllSequenceMeshMesh";
    ut_stub::InjectProbeAlgAttrs(attrs.name, attrs);

    TopoInfoWithNetLayerDetails topo = MakeTwoLevelTopo(3, 258);
    std::vector<CostModelParam> coeffs = executor.CalcCostCoeff(nullptr, &topo, attrs.name.c_str(), param_);
    EXPECT_EQ(coeffs.size(), 2u);
}

// d1=130 越 CCU 128 通道上限 → 剔除（非对称 instList：GCD=2 → d0=2、d1=260/2=130）
TEST_F(CcuA2ASequenceExecutorTest, CalcCostCoeffCcuSchedD1ExceedsCcuChannelLimitReturnsEmpty)
{
    TestSeqExecutor executor;
    AlgAttrs attrs = MakeTwoLevelAttrs();
    attrs.name = "CcuSchedAllToAllSequenceMeshMesh";
    ut_stub::InjectProbeAlgAttrs(attrs.name, attrs);

    TopoInfoWithNetLayerDetails topo;
    topo.userRank = 3;
    topo.userRankSize = 260;

    PhysicalLevelInfo level0;
    for (u32 i = 0; i < 260; i++) {
        level0.localRanks.push_back(i);
    }
    level0.view = PhysicalLevelView::GLOBAL;
    level0.instSizeListByLayer = {2, 258}; // 非对称 → GCD=2 → d0=2、d1=130
    level0.hasTopoInst = true;
    level0.topoType = CommTopo::COMM_TOPO_CLOS;
    level0.locType = ENDPOINT_LOC_TYPE_DEVICE;
    level0.portNums = {1};

    PhysicalLevelInfo level1;
    for (u32 i = 0; i < 260; i++) {
        level1.localRanks.push_back(i);
    }
    level1.view = PhysicalLevelView::GLOBAL;
    level1.instSizeListByLayer = {260};
    level1.hasTopoInst = true;
    level1.topoType = CommTopo::COMM_TOPO_CLOS;
    level1.locType = ENDPOINT_LOC_TYPE_DEVICE;
    level1.portNums = {8};

    topo.physicalLevels = {std::move(level0), std::move(level1)};
    topo.physicalLevelNum = 2;
    topo.topoLevelNums = 2;

    std::vector<CostModelParam> coeffs = executor.CalcCostCoeff(nullptr, &topo, attrs.name.c_str(), param_);
    EXPECT_TRUE(coeffs.empty());
}

// d1=1（单实例全量 → d0=N、d1=1）→ MatchTopo d1==1 拒绝 → 空系数
TEST_F(CcuA2ASequenceExecutorTest, CalcCostCoeffCcuSchedD1OutOfBoundReturnsEmpty)
{
    TestSeqExecutor executor;
    AlgAttrs attrs = MakeTwoLevelAttrs();
    attrs.name = "CcuSchedAllToAllSequenceMeshMesh";
    ut_stub::InjectProbeAlgAttrs(attrs.name, attrs);

    TopoInfoWithNetLayerDetails topo;
    topo.userRank = 3;
    topo.userRankSize = 16;

    PhysicalLevelInfo level0;
    for (u32 i = 0; i < 16; i++) {
        level0.localRanks.push_back(i);
    }
    level0.view = PhysicalLevelView::GLOBAL;
    level0.instSizeListByLayer = {16}; // 单实例 → 对称 → d0=16、d1=1
    level0.hasTopoInst = true;
    level0.topoType = CommTopo::COMM_TOPO_CLOS;
    level0.locType = ENDPOINT_LOC_TYPE_DEVICE;

    PhysicalLevelInfo level1;
    for (u32 i = 0; i < 16; i++) {
        level1.localRanks.push_back(i);
    }
    level1.view = PhysicalLevelView::GLOBAL;
    level1.instSizeListByLayer = {16};
    level1.hasTopoInst = true;
    level1.topoType = CommTopo::COMM_TOPO_CLOS;
    level1.locType = ENDPOINT_LOC_TYPE_DEVICE;

    topo.physicalLevels = {std::move(level0), std::move(level1)};
    topo.physicalLevelNum = 2;
    topo.topoLevelNums = 2;

    std::vector<CostModelParam> coeffs = executor.CalcCostCoeff(nullptr, &topo, attrs.name.c_str(), param_);
    EXPECT_TRUE(coeffs.empty());
}

// portNum 缺失 → 守卫拦截
TEST_F(CcuA2ASequenceExecutorTest, CalcCostCoeffCcuSchedPortNumEmptyReturnsEmpty)
{
    TestSeqExecutor executor;
    AlgAttrs attrs = MakeTwoLevelAttrs();
    attrs.name = "CcuSchedAllToAllSequenceMeshMesh";
    ut_stub::InjectProbeAlgAttrs(attrs.name, attrs);

    topo_.physicalLevels[0].portNums.clear();
    std::vector<CostModelParam> coeffs = executor.CalcCostCoeff(nullptr, &topo_, attrs.name.c_str(), param_);
    EXPECT_TRUE(coeffs.empty());
}

// AICPU 分支同样两段实算
TEST_F(CcuA2ASequenceExecutorTest, CalcCostCoeffNonCcuSchedNameReturnsTwoSegParams)
{
    TestSeqExecutor executor;
    AlgAttrs attrs = MakeTwoLevelAttrs();
    attrs.name = "AicpuAllToAllSequenceMeshMesh";
    ut_stub::InjectProbeAlgAttrs(attrs.name, attrs);

    std::vector<CostModelParam> coeffs = executor.CalcCostCoeff(nullptr, &topo_, attrs.name.c_str(), param_);
    EXPECT_EQ(coeffs.size(), 2u);
}

// 未注册名 → 探测剔除
TEST_F(CcuA2ASequenceExecutorTest, CalcCostCoeffUnregisteredNameReturnsEmpty)
{
    TestSeqExecutor executor;
    std::vector<CostModelParam> coeffs = executor.CalcCostCoeff(nullptr, &topo_, "NotRegisteredAlgo", param_);
    EXPECT_TRUE(coeffs.empty());
}

// 契约硬失败：threadNum=0/两级缺 kernel → HCCL_E_INTERNAL
TEST_F(CcuA2ASequenceExecutorTest, FastLaunchRejectsInvalidCtx)
{
    TestSeqExecutor executor;
    std::vector<char> buf(CcuFastLaunchCtx::GetCtxSize(2, 2), 0);
    auto* ctx = reinterpret_cast<CcuFastLaunchCtx*>(buf.data());

    ctx->threadNum = 0;
    EXPECT_EQ(executor.FastLaunch(param_, ctx), HcclResult::HCCL_E_INTERNAL);

    ctx->threadNum = 1;
    ctx->ccuKernelNum[0] = 0; // 首级缺 kernel
    EXPECT_EQ(executor.FastLaunch(param_, ctx), HcclResult::HCCL_E_INTERNAL);

    ctx->ccuKernelNum[0] = 1;
    ctx->ccuKernelNum[1] = 0; // 末级缺 kernel
    EXPECT_EQ(executor.FastLaunch(param_, ctx), HcclResult::HCCL_E_INTERNAL);
}

// RestoreStageCcuKernels：正常切片归位/数量不足/切片越界三分支
TEST_F(CcuA2ASequenceExecutorTest, RestoreStageCcuKernelsSlicesAndRejects)
{
    TestSeqExecutor executor;
    executor.totalStages_ = 2;
    AlgResourceCtxSerializable resCtx{};
    CcuKernelHandle handle{};
    resCtx.ccuKernels = {handle, handle};

    // 正常：{1,1} 切片归位
    resCtx.ccuKernelNum = {1, 1};
    EXPECT_EQ(executor.RestoreStageCcuKernels(resCtx), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(executor.stageCcuKernels_.size(), 2u);
    EXPECT_EQ(executor.stageCcuKernels_[1].size(), 1u);
    EXPECT_EQ(executor.stageCcuKernels_[0].size(), 1u);

    // 数量不足
    resCtx.ccuKernelNum = {1};
    EXPECT_EQ(executor.RestoreStageCcuKernels(resCtx), HcclResult::HCCL_E_INTERNAL);

    // 切片越界
    resCtx.ccuKernelNum = {3, 0};
    EXPECT_EQ(executor.RestoreStageCcuKernels(resCtx), HcclResult::HCCL_E_INTERNAL);
}

// FastLaunchSaveCtx：0 kernel just-success + 两模板 save 平铺序
TEST_F(CcuA2ASequenceExecutorTest, FastLaunchSaveCtxSavesTwoStageLayout)
{
    TestSeqExecutor executor;
    executor.threads_ = {ThreadHandle{}};
    OpParam param{};
    param.opMode = OpMode::OPBASE; // ≠ OFFLOAD

    std::vector<std::vector<CcuKernelSubmitInfo>> emptyInfos(2);
    EXPECT_EQ(executor.FastLaunchSaveCtx(param, emptyInfos, 0), HcclResult::HCCL_SUCCESS);

    CcuKernelSubmitInfo firstInfo{};
    firstInfo.cachedArgs[0] = 0x1111;
    CcuKernelSubmitInfo lastInfo{};
    lastInfo.cachedArgs[0] = 0x2222;
    std::vector<std::vector<CcuKernelSubmitInfo>> infos = {{firstInfo}, {lastInfo}};
    EXPECT_EQ(executor.FastLaunchSaveCtx(param, infos, 0), HcclResult::HCCL_SUCCESS);

    auto* saved = static_cast<CcuFastLaunchCtx*>(ut_stub::LastEngineCtx());
    ASSERT_NE(saved, nullptr);
    EXPECT_EQ(saved->threadNum, 1u);
    EXPECT_EQ(saved->ccuKernelNum[0], 1u);
    EXPECT_EQ(saved->ccuKernelNum[1], 1u);
    auto* submitInfos = saved->GetCcuKernelSubmitInfoPtr();
    EXPECT_EQ(submitInfos[0].cachedArgs[0], 0x1111u); // 首 Stage 在前
    EXPECT_EQ(submitInfos[1].cachedArgs[0], 0x2222u); // 末 Stage 在后
}

// FastLaunch 成功路径：{1,1} 切片 + 两 stage 下发 + 槽位刷新（与 Rejects 正反对称）
TEST_F(CcuA2ASequenceExecutorTest, FastLaunchSuccessDispatchesBothStages)
{
    TestSeqExecutor executor;
    uint64_t inBuf[64] = {};
    uint64_t outBuf[64] = {};
    uint64_t cclBuf[64] = {};
    param_.inputPtr = &inBuf;
    param_.outputPtr = &outBuf;
    param_.hcclBuff.addr = &cclBuf;

    std::vector<char> buf(CcuFastLaunchCtx::GetCtxSize(2, 2), 0);
    auto* ctx = reinterpret_cast<CcuFastLaunchCtx*>(buf.data());
    ctx->threadNum = 2;
    ctx->ccuKernelNum[0] = 1;
    ctx->ccuKernelNum[1] = 1;

    auto* submitInfos = ctx->GetCcuKernelSubmitInfoPtr();
    for (u32 i = 0; i < 13; i++) {
        submitInfos[0].cachedArgs[i] = 0x1000 + i; // 首 Stage 哨兵
    }
    for (u32 i = 0; i < 12; i++) {
        submitInfos[1].cachedArgs[i] = 0x2000 + i; // 末 Stage 哨兵
    }

    EXPECT_EQ(executor.FastLaunch(param_, ctx), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(ut_stub::LaunchRecordCount(), 2u);

    // 首 Stage：13 槽，槽 0/1 刷新为 input/cclBuff 地址
    const auto& a0 = ut_stub::LaunchRecordArgs(0);
    ASSERT_EQ(a0.size(), 13u);
    EXPECT_EQ(a0[0], static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&inBuf)));
    EXPECT_EQ(a0[1], static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&cclBuf)));
    for (u32 i = 2; i < 13; i++) {
        EXPECT_EQ(a0[i], 0x1000 + i) << "first slot " << i;
    }

    // 末 Stage：12 槽，槽 0/1 刷新为 cclBuff/output 地址
    const auto& a1 = ut_stub::LaunchRecordArgs(1);
    ASSERT_EQ(a1.size(), 12u);
    EXPECT_EQ(a1[0], static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&cclBuf)));
    EXPECT_EQ(a1[1], static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&outBuf)));
    for (u32 i = 2; i < 12; i++) {
        EXPECT_EQ(a1[i], 0x2000 + i) << "last slot " << i;
    }
}

} // namespace ops_hccl
