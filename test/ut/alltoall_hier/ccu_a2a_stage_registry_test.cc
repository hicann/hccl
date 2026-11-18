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
#include <memory>
#include <string>
#include <vector>

#include "ccu_alltoall_stage_template_registry.h"
#include "ccu_temp_all_to_all_hier_stage_base.h"
#include "ccu_temp_all_to_all_hier_first_stage.h"
#include "template_utils.h"

namespace ops_hccl {

// 与执行器 GetStageAlgos 硬编码一致的注册名
constexpr char CCU_A2A_FIRST_STAGE_ALGO[] = "FirstMeshStage";
constexpr char CCU_A2A_LAST_STAGE_ALGO[] = "LastMeshStage";

namespace ut_stub {
    void InjectChannelScenario(bool dualDie, uint32_t die0Bw, uint32_t die1Bw);
    void InjectMisorderedPeers(const std::vector<uint32_t>& peers);
    void InjectDie1MissingPeer(uint32_t peer);
    void ResetChannelScenario();
    uint32_t ChannelDieOf(const HcclChannelDesc& channel);
    uint32_t ChannelBwOf(const HcclChannelDesc& channel);
    void ResetLaunchRecords();
    size_t LaunchRecordCount();
    const std::vector<uint64_t>& LaunchRecordArgs(size_t idx);
} // namespace ut_stub

// 基类为抽象类，以最小派生类实例化
class FakeStage : public CcuTempAlltoAllHierStageBase {
public:
    FakeStage() = default;

    std::string GetAlgoType() const override { return "FakeStage"; }

    std::string Describe() const override { return "FakeStage for SetStageRole UT"; }
};

class CcuA2AStageRegistryTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        param_ = OpParam{};
        param_.engine = CommEngine::COMM_ENGINE_CCU;
        subCommRanks_ = {{0, 1, 2, 3}};
        rankId_ = 2;
        stageIndex_ = 1;
        ut_stub::ResetChannelScenario();
        ut_stub::ResetLaunchRecords();
    }

    OpParam param_{};
    std::vector<std::vector<u32>> subCommRanks_;
    u32 rankId_{0};
    u32 stageIndex_{0};
};

TEST_F(CcuA2AStageRegistryTest, CreateFirstMeshStage)
{
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->GetAlgoType(), std::string(CCU_A2A_FIRST_STAGE_ALGO));
    EXPECT_EQ(stage->GetStageIndex(), stageIndex_);
    EXPECT_EQ(stage->GetTemplateRankSize(), 4u); // subCommRanks[0].size() 覆盖基类求和语义
    EXPECT_EQ(stage->GetMyAlgRank(), 2u);        // rankId 在组内的下标
}

TEST_F(CcuA2AStageRegistryTest, CreateLastMeshStage)
{
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_LAST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->GetAlgoType(), std::string(CCU_A2A_LAST_STAGE_ALGO));
    EXPECT_EQ(stage->GetStageIndex(), stageIndex_);
    EXPECT_EQ(stage->GetTemplateRankSize(), 4u);
    EXPECT_EQ(stage->GetMyAlgRank(), 2u);
}

TEST_F(CcuA2AStageRegistryTest, CreateUnregisteredAlgoNameReturnsNull)
{
    auto stage
        = CcuAlltoAllStageTemplateRegistry::Instance().Create("NotExist", param_, rankId_, subCommRanks_, stageIndex_);
    EXPECT_EQ(stage, nullptr);
}

// rankId 不在组内时 myAlgRank_ 置 INVALID（CalcRes 校验捕获）
TEST_F(CcuA2AStageRegistryTest, CreateWithRankNotInGroupKeepsInvalidAlgRank)
{
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, 9u, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->GetMyAlgRank(), INVALID_VALUE_RANKID);
    EXPECT_EQ(stage->GetTemplateRankSize(), 4u);
}

TEST_F(CcuA2AStageRegistryTest, CreateWithEmptySubCommRanksKeepsDefaults)
{
    std::vector<std::vector<u32>> emptyRanks;
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, emptyRanks, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->GetTemplateRankSize(), 0u);
    EXPECT_EQ(stage->GetMyAlgRank(), INVALID_VALUE_RANKID);
}

// ValidateStageTopo 契约拦截：异常输入在通道申请前拒绝
TEST_F(CcuA2AStageRegistryTest, CalcResRejectsInvalidStageTopo)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.userRankSize = 8;
    AlgResourceRequest req;

    // 空向量
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, std::vector<std::vector<u32>>{}, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_E_INTERNAL);

    // 空组 {{}}
    std::vector<std::vector<u32>> emptyGroup = {{}};
    stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, emptyGroup, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_E_INTERNAL);

    // 多组
    std::vector<std::vector<u32>> multiGroup = {{0, 1, 2, 3}, {4, 5, 6, 7}};
    stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, multiGroup, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_E_INTERNAL);

    // rank 不在组内
    stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, 9u, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_E_INTERNAL);

    // userRankSize 非整除组大小
    topo.userRankSize = 10;
    stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_E_INTERNAL);

    // topoInfo 空指针
    stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->CalcRes(nullptr, param_, nullptr, req), HcclResult::HCCL_E_PTR);

    // 末 Stage 共用防线
    topo.userRankSize = 8;
    stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_LAST_STAGE_ALGO, param_, rankId_, std::vector<std::vector<u32>>{}, stageIndex_);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_E_INTERNAL);
}

TEST(CcuA2ASetStageRoleTest, FirstStageRole)
{
    FakeStage stage;
    stage.SetStageRole(1, 2);
    EXPECT_TRUE(stage.IsFirstStage());
    EXPECT_FALSE(stage.IsLastStage());
    EXPECT_FALSE(stage.IsMiddleStage());
    EXPECT_EQ(stage.GetStageIndex(), 1u);
}

TEST(CcuA2ASetStageRoleTest, LastStageRole)
{
    FakeStage stage;
    stage.SetStageRole(0, 2);
    EXPECT_FALSE(stage.IsFirstStage());
    EXPECT_TRUE(stage.IsLastStage());
    EXPECT_FALSE(stage.IsMiddleStage());
    EXPECT_EQ(stage.GetStageIndex(), 0u);
}

// 中间 stage（防御性，真实路径不可达）
TEST(CcuA2ASetStageRoleTest, MiddleStageRole)
{
    FakeStage stage;
    stage.SetStageRole(1, 3);
    EXPECT_FALSE(stage.IsFirstStage());
    EXPECT_FALSE(stage.IsLastStage());
    EXPECT_TRUE(stage.IsMiddleStage());
    EXPECT_EQ(stage.GetStageIndex(), 1u);
}

// 单 kernel：通道数==groupSize-1 且按组序
TEST_F(CcuA2AStageRegistryTest, CalcResSingleKernelAssemblesChannelsInGroupOrder)
{
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = rankId_;
    topo.userRankSize = 8;
    AlgResourceRequest req;
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(req.ccuKernelNum, (std::vector<u32>{1}));
    ASSERT_EQ(req.ccuKernelInfos.size(), 1u);
    ASSERT_EQ(req.ccuKernelInfos[0].channels.size(), 3u);
    EXPECT_EQ(req.ccuKernelInfos[0].channels[0].remoteRank, 0u);
    EXPECT_EQ(req.ccuKernelInfos[0].channels[1].remoteRank, 1u);
    EXPECT_EQ(req.ccuKernelInfos[0].channels[2].remoteRank, 3u);
    EXPECT_DOUBLE_EQ(req.dieSplitRatio, 1.0);
}

// 双 kernel：两桶各持全部对端、低带宽侧换至 MINOR
TEST_F(CcuA2AStageRegistryTest, CalcResDualKernelSwapsLowBwToMinor)
{
    ut_stub::InjectChannelScenario(true, 10, 30); // die1 高带宽，期望为 MAJOR
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = rankId_;
    topo.userRankSize = 8;
    AlgResourceRequest req;
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(req.ccuKernelNum, (std::vector<u32>{2}));
    ASSERT_EQ(req.ccuKernelInfos.size(), 2u);
    for (u32 i = 0; i < 2; i++) {
        ASSERT_EQ(req.ccuKernelInfos[i].channels.size(), 3u);
        const u32 peers[3] = {0, 1, 3};
        for (u32 p = 0; p < 3; p++) {
            EXPECT_EQ(req.ccuKernelInfos[i].channels[p].remoteRank, peers[p]) << "kernel " << i << " peer " << p;
        }
    }
    EXPECT_EQ(ut_stub::ChannelDieOf(req.ccuKernelInfos[0].channels[0]), 1u);
    EXPECT_EQ(ut_stub::ChannelBwOf(req.ccuKernelInfos[0].channels[0]), 30u);
    EXPECT_EQ(ut_stub::ChannelDieOf(req.ccuKernelInfos[1].channels[0]), 0u);
    EXPECT_EQ(ut_stub::ChannelBwOf(req.ccuKernelInfos[1].channels[0]), 10u);
    EXPECT_NEAR(req.dieSplitRatio, 0.75, 1e-9); // 30/(30+10)
}

// CalcSegSize 分段边界
TEST_F(CcuA2AStageRegistryTest, CalcSegSizeBoundaryCases)
{
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    auto* first = static_cast<CcuTempAlltoAllHierFirstStage*>(stage.get());
    u64 segSize = 0;
    u64 segOffset = 0;

    // 非 2+6：单 kernel 全量
    first->is2Plus6_ = false;
    EXPECT_EQ(first->CalcSegSize(0, 1000, segSize, segOffset), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(segSize, 1000u);
    EXPECT_EQ(segOffset, 0u);

    // 小 slice(<2*128)：MAJOR 全量、MINOR 空段
    first->is2Plus6_ = true;
    first->dieSplitRatio_ = 0.5;
    EXPECT_EQ(first->CalcSegSize(0, 200, segSize, segOffset), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(segSize, 200u);
    EXPECT_EQ(segOffset, 0u);
    EXPECT_EQ(first->CalcSegSize(1, 200, segSize, segOffset), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(segSize, 0u);
    EXPECT_EQ(segOffset, 200u);

    // ==2*ALIGN：128/128 均分
    EXPECT_EQ(first->CalcSegSize(0, 256, segSize, segOffset), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(segSize, 128u);
    EXPECT_EQ(segOffset, 0u);
    EXPECT_EQ(first->CalcSegSize(1, 256, segSize, segOffset), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(segSize, 128u);
    EXPECT_EQ(segOffset, 128u);

    // 非 128 倍数：和恒等+128B 对齐
    EXPECT_EQ(first->CalcSegSize(0, 1000, segSize, segOffset), HcclResult::HCCL_SUCCESS);
    u64 majorSize = segSize;
    EXPECT_EQ(segOffset, 0u);
    EXPECT_EQ(majorSize % 128, 0u);
    EXPECT_EQ(first->CalcSegSize(1, 1000, segSize, segOffset), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(segSize + majorSize, 1000u);
    EXPECT_EQ(segOffset, majorSize);

    // 夹紧下限
    first->dieSplitRatio_ = 0.01;
    EXPECT_EQ(first->CalcSegSize(0, 1000, segSize, segOffset), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(segSize, 128u);

    // 夹紧上限
    first->dieSplitRatio_ = 0.99;
    EXPECT_EQ(first->CalcSegSize(0, 1000, segSize, segOffset), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(segSize, 768u);
}

// 双 kernel 组序契约拦截
TEST_F(CcuA2AStageRegistryTest, CalcResDualKernelRejectsMisorderedChannels)
{
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    TopoInfoWithNetLayerDetails topo{};
    topo.userRank = rankId_;
    topo.userRankSize = 8;
    AlgResourceRequest req;

    // 组序错位：{0,1,5} 逐位失配
    ut_stub::InjectMisorderedPeers({0, 1, 5});
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_E_INTERNAL);

    // 桶缺对端：die1 桶尺寸失配
    ut_stub::ResetChannelScenario();
    ut_stub::InjectDie1MissingPeer(3);
    EXPECT_EQ(stage->CalcRes(nullptr, param_, &topo, req), HcclResult::HCCL_E_INTERNAL);
}

// FastLaunch 槽位刷新：仅槽 0/1 刷新、其余保持
TEST_F(CcuA2AStageRegistryTest, FastLaunchRefreshesAddrSlotsOnly)
{
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);
    uint64_t inBuf = 0;
    uint64_t cclBuf = 0;

    // 0 kernel：just success
    TemplateFastLaunchCtx emptyCtx;
    emptyCtx.threads = {ThreadHandle{}};
    emptyCtx.buffInfo.inputPtr = &inBuf;
    emptyCtx.buffInfo.hcclBuff.addr = &cclBuf;
    EXPECT_EQ(stage->FastLaunch(param_, emptyCtx), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ut_stub::LaunchRecordCount(), 0u);

    // 1 kernel：槽 0/1 刷新、其余保持哨兵
    TemplateFastLaunchCtx ctx;
    ctx.threads = {ThreadHandle{}};
    ctx.buffInfo.inputPtr = &inBuf;
    ctx.buffInfo.hcclBuff.addr = &cclBuf;
    CcuKernelSubmitInfo info{};
    for (u32 i = 0; i < 13; i++) {
        info.cachedArgs[i] = 0x1000 + i; // 哨兵
    }
    ctx.ccuKernelSubmitInfos = {info};
    EXPECT_EQ(stage->FastLaunch(param_, ctx), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(ut_stub::LaunchRecordCount(), 1u);
    const auto& args = ut_stub::LaunchRecordArgs(0);
    ASSERT_EQ(args.size(), 13u);
    EXPECT_EQ(args[0], static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&inBuf)));
    EXPECT_EQ(args[1], static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&cclBuf)));
    for (u32 i = 2; i < 13; i++) {
        EXPECT_EQ(args[i], 0x1000 + i) << "slot " << i;
    }
}

// KernelRun 编排：首 Stage 2+6 双 kernel 分段下发
TEST_F(CcuA2AStageRegistryTest, KernelRunDualKernelLaunchesSegRecords)
{
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_FIRST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);

    u64 sendCount = 16;
    param_.all2AllVDataDes.sendCounts = &sendCount;

    uint64_t inBuf[64] = {};
    uint64_t cclBuf[64] = {};
    TemplateDataParams tap{};
    tap.buffInfo.inputPtr = inBuf;
    tap.buffInfo.inputSize = sizeof(inBuf); // GetToken 需要 (ptr,size) 二元组
    tap.buffInfo.hcclBuff.addr = cclBuf;
    tap.buffInfo.hcclBuff.size = sizeof(cclBuf);
    tap.dataType = HCCL_DATA_TYPE_INT32;
    tap.sliceSize = 256;
    tap.outputSliceStride = 4;        // otherDim
    tap.inputSliceStride = 4 * 4 * 4; // cclCount(4) * dts(4) * otherDim(4)

    TemplateResource res{};
    CcuKernelHandle handle{};
    res.ccuKernels = {handle, handle};
    res.threads = {ThreadHandle{}, ThreadHandle{}};
    res.dieSplitRatio = 0.75f;

    EXPECT_EQ(stage->KernelRun(param_, tap, res), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(ut_stub::LaunchRecordCount(), 2u);

    const auto& a0 = ut_stub::LaunchRecordArgs(0);
    ASSERT_EQ(a0.size(), 13u);
    EXPECT_EQ(a0[3], 128u); // segSize（2+6 夹紧均分）
    EXPECT_EQ(a0[4], 0u);   // segOffset
    EXPECT_EQ(a0[7], 64u);  // blockStride = 16 * 4
    EXPECT_EQ(a0[8], 16u);  // slotStride = 4 * 4

    const auto& a1 = ut_stub::LaunchRecordArgs(1);
    ASSERT_EQ(a1.size(), 13u);
    EXPECT_EQ(a1[3], 128u);
    EXPECT_EQ(a1[4], 128u);
    EXPECT_EQ(a0[3] + a1[3], 256u); // 两段和恒等

    EXPECT_EQ(res.submitInfos.size(), 2u); // FillCachedArgs 编排
}

// KernelRun 编排：末 Stage 单 kernel 全量下发
TEST_F(CcuA2AStageRegistryTest, KernelRunLastStageSingleKernelLaunchesFullSlice)
{
    auto stage = CcuAlltoAllStageTemplateRegistry::Instance().Create(
        CCU_A2A_LAST_STAGE_ALGO, param_, rankId_, subCommRanks_, stageIndex_);
    ASSERT_NE(stage, nullptr);

    u64 sendCount = 16;
    param_.all2AllVDataDes.sendCounts = &sendCount;

    uint64_t cclBuf[64] = {};
    uint64_t outBuf[64] = {};
    TemplateDataParams tap{};
    tap.buffInfo.hcclBuff.addr = cclBuf;
    tap.buffInfo.hcclBuff.size = sizeof(cclBuf); // GetToken 需要 (ptr,size) 二元组
    tap.buffInfo.outputPtr = outBuf;
    tap.buffInfo.outputSize = sizeof(outBuf);
    tap.dataType = HCCL_DATA_TYPE_INT32;
    tap.sliceSize = 256;
    tap.outputSliceStride = 4;
    tap.inputSliceStride = 64;

    TemplateResource res{};
    CcuKernelHandle handle{};
    res.ccuKernels = {handle};
    res.threads = {ThreadHandle{}};

    EXPECT_EQ(stage->KernelRun(param_, tap, res), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(ut_stub::LaunchRecordCount(), 1u);

    const auto& args = ut_stub::LaunchRecordArgs(0);
    ASSERT_EQ(args.size(), 12u);
    EXPECT_EQ(args[3], 256u); // sliceSize 全量
    EXPECT_EQ(args[6], 64u);  // blockStride
    EXPECT_EQ(args[7], 16u);  // slotStride

    EXPECT_EQ(res.submitInfos.size(), 1u);
}

} // namespace ops_hccl
