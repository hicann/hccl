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
#include "topo_match_two_level.h"
#include "topo_match_test_common.h"

using namespace ops_hccl;
using ops_hccl::ut_helper::MakeLevel;
using ops_hccl::ut_helper::MakeProfile;
using ops_hccl::ut_helper::MakeTopoInfo;
using ops_hccl::ut_helper::Range;

class TopoMatchTwoLevelTest : public ::testing::Test {
protected:
    TopoMatchTwoLevel matcher_;
};

// T1: 对称 2 层 GLOBAL，d0=8, d1=2
TEST_F(TopoMatchTwoLevelTest, SymmetricTwoLevelGlobal)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos.size(), 2u);
    ASSERT_EQ(info.infos[0][0], Range(8));
    // group1 = {3, 11}（step=8, count=2, offset=3）
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0].size(), 1u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// T2: 冗余物理层压缩（3 层 effIdx，algo 2 层；无 Mesh 锚点，分段压缩高层）
TEST_F(TopoMatchTwoLevelTest, RedundantLayerCompression)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos[0][0], Range(8));
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    // pIndices 压缩：phys0=effIdx[0]=0, phys1=effIdx[2]=2
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
}

// T3: MeshConcur CLOS 双层（physicalIdx[0] = {Mesh层, CLOS层}）
TEST_F(TopoMatchTwoLevelTest, MeshConcurDualLayerClos)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_1DMESH),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCUR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // MeshConcur 层 → {Mesh层(idx0), CLOS层(idx1)}
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0].size(), 2u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][1], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    // 普通层 → {idx1}
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1].size(), 1u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// T4: MeshConcur 上层非 CLOS 但 localRanks 是超集 → SUCCESS（不再强求 topoType==CLOS）
TEST_F(TopoMatchTwoLevelTest, MeshConcurUpperNotClosButSuperset)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_1DMESH),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_1DMESH),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCUR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // phys1 localRanks={0..15} ⊇ {0..7} → 找到，physicalIdx[0]={0,1}
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0].size(), 2u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][1], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// T5: 非对称 GCD 打平，2 级 GLOBAL。instList={16,4}, myRank=2 在 16-instance{0..15}
TEST_F(TopoMatchTwoLevelTest, AsymmetricGcd)
{
    auto topo = MakeTopoInfo(
        2, 20,
        {
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 4}),
            MakeLevel(Range(20), PhysicalLevelView::GLOBAL, {20}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // d0=GCD(16,4)=4; myRank=2 在 localRanks[0..15] 的 index 2, startIdx=(2/4)*4=0 → group0={0,1,2,3}
    ASSERT_EQ(info.infos[0][0], (std::vector<u32>{0, 1, 2, 3}));
    // d1=20/4=5, group1=BuildRepresentativeGroup(4, 5, 2)={2,6,10,14,18}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{2, 6, 10, 14, 18}));
}

// T6: 非对称 GCD=1 → NOT_SUPPORT。instList={3,5}, myRank=3 在 5-instance{0..4}
TEST_F(TopoMatchTwoLevelTest, AsymmetricGcdOne)
{
    auto topo = MakeTopoInfo(
        3, 8,
        {
            MakeLevel(Range(5), PhysicalLevelView::GLOBAL, {3, 5}),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T7: LOCAL level0，d0=localRanks.size()=4
TEST_F(TopoMatchTwoLevelTest, LocalLevel0)
{
    auto topo = MakeTopoInfo(
        2, 16,
        {
            MakeLevel(Range(4), PhysicalLevelView::LOCAL),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos[0][0], Range(4));
    // d1=4, group1=BuildRepresentativeGroup(4, 4, 2)={2,6,10,14}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{2, 6, 10, 14}));
}

// T8: userRankSize 不被 d0 整除 → NOT_SUPPORT
TEST_F(TopoMatchTwoLevelTest, UserRankSizeNotDivisibleByD0)
{
    auto topo = MakeTopoInfo(
        2, 10,
        {
            MakeLevel(Range(4), PhysicalLevelView::LOCAL),
            MakeLevel(Range(10), PhysicalLevelView::GLOBAL, {10}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T9: hostdpu 顶层锚 HOST
TEST_F(TopoMatchTwoLevelTest, HostdpuAnchorTopHost)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_1DMESH, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos[0][0], Range(8));
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    // 顶层 phys1 = HOST 层
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// T10: hostdpu HOST 层在低位，下层不够 → NOT_SUPPORT
TEST_F(TopoMatchTwoLevelTest, HostdpuHostLayerTooLow)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_1DMESH, ENDPOINT_LOC_TYPE_DEVICE),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T11: d0<=1（2 层）→ NOT_SUPPORT
TEST_F(TopoMatchTwoLevelTest, D0LeOne)
{
    auto topo = MakeTopoInfo(
        0, 2,
        {
            MakeLevel(Range(1), PhysicalLevelView::LOCAL),
            MakeLevel(Range(2), PhysicalLevelView::GLOBAL, {2}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T12: 有效层 < 2 → NOT_SUPPORT
TEST_F(TopoMatchTwoLevelTest, EffectiveLayersLessThanTwo)
{
    auto topo = MakeTopoInfo(3, 8, {MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8})});
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T13: MeshConcur 底层非 1DMESH（有冗余层触发 FindAnchors，找不到 Mesh 层）→ NOT_SUPPORT
TEST_F(TopoMatchTwoLevelTest, MeshConcurBottomNot1DMesh)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCUR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T14: MeshConcur mesh 层在最高位（无上层超集）→ NOT_SUPPORT
TEST_F(TopoMatchTwoLevelTest, MeshConcurMeshAtTopNoUpper)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_1DMESH),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCUR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    // 1:1 映射：algo0(MeshConcur)→phys1(1DMESH, 在最高位), algo1→phys0
    // FindUpperEncompassingLevel 从 phys1 往上无层 → NOT_SUPPORT
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T15: 3 层拓扑每 server 1 卡，d0=1 → NOT_SUPPORT
TEST_F(TopoMatchTwoLevelTest, ThreeLayersOneCardPerServerD0One)
{
    auto topo = MakeTopoInfo(
        0, 4,
        {
            MakeLevel(Range(1), PhysicalLevelView::LOCAL),
            MakeLevel(Range(2), PhysicalLevelView::GLOBAL, {2}),
            MakeLevel(Range(4), PhysicalLevelView::GLOBAL, {4}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    // 3 effIdx > 2 → FindAnchors 无 Mesh 锚点 → 分段压缩 pIndices={0,2}; phys0 localRanks.size()=1 → d0<=1
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T16: 3 层冗余 GLOBAL 非对称 level0 GCD>1 → SUCCESS
TEST_F(TopoMatchTwoLevelTest, ThreeLayersAsymmetricGcd)
{
    auto topo = MakeTopoInfo(
        2, 24,
        {
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 8}),
            MakeLevel(Range(24), PhysicalLevelView::GLOBAL, {24}),
            MakeLevel(Range(24), PhysicalLevelView::GLOBAL, {24}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // d0=GCD(16,8)=8; 24%8=0, d1=3; group0={0..7}; group1={2,10,18}
    ASSERT_EQ(info.infos[0][0], (std::vector<u32>{0, 1, 2, 3, 4, 5, 6, 7}));
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{2, 10, 18}));
    // 3 effIdx>2 分段压缩：phys0=effIdx[0]=0, phys1=effIdx[2]=2
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
}

// T17: LOCAL level0 + GLOBAL 非对称 level1（level1 非对称不影响 d0，TwoLevel 只 GCD level0）
TEST_F(TopoMatchTwoLevelTest, LocalLevel0GlobalAsymLevel1)
{
    auto topo = MakeTopoInfo(
        2, 16,
        {
            MakeLevel(Range(4), PhysicalLevelView::LOCAL),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {8, 4, 4}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // d0=4(LOCAL), d1=4; group0={0..3}; group1={2,6,10,14}
    ASSERT_EQ(info.infos[0][0], Range(4));
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{2, 6, 10, 14}));
}

// T18: 非对称 + hostdpu → SUCCESS
TEST_F(TopoMatchTwoLevelTest, AsymmetricWithHostdpu)
{
    auto topo = MakeTopoInfo(
        2, 24,
        {
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(24), PhysicalLevelView::GLOBAL, {24}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // hostdpu 锚 algo1→phys1(HOST+24==size); d0=GCD(16,8)=8; d1=3; group0={0..7}; group1={2,10,18}
    ASSERT_EQ(info.infos[0][0], (std::vector<u32>{0, 1, 2, 3, 4, 5, 6, 7}));
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{2, 10, 18}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// T19: 非对称 GCD>1 但 userRankSize 不被 d0 整除 → NOT_SUPPORT
TEST_F(TopoMatchTwoLevelTest, AsymmetricGcdSizeNotDivisible)
{
    auto topo = MakeTopoInfo(
        2, 10,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 4}),
            MakeLevel(Range(10), PhysicalLevelView::GLOBAL, {10}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    // d0=GCD(8,4)=4; 10%4=2≠0 → NOT_SUPPORT
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T20: AIV 过滤 UBG 层后仍能 match（3 层过滤剩 2 层）
TEST_F(TopoMatchTwoLevelTest, AivFilterUbgStillMatch)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(
                Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE,
                {COMM_PROTOCOL_UBG}),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE, {}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE, {}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR}, OpExecuteConfig::AIV);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // phys0(UBG) 被排除，effIdx={1,2}; d0=8, d1=2; physIdx={1,2}
    ASSERT_EQ(info.infos[0][0], Range(8));
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
}

// T21: AIV 过滤 UBG 层后仅剩 1 层 → NOT_SUPPORT
TEST_F(TopoMatchTwoLevelTest, AivFilterUbgOnlyOneRemains)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(
                Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE,
                {COMM_PROTOCOL_UBG}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE, {}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR}, OpExecuteConfig::AIV);
    AlgHierarchyInfoForAllLevel info;
    // phys0(UBG) 排除，effIdx={1} < 2 → NOT_SUPPORT
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// T22: hostdpu 顶层 DEVICE+全集，倒数第二层 HOST+全集；AnchorHostDpu 跳过 DEVICE 选 HOST
TEST_F(TopoMatchTwoLevelTest, HostdpuTopDeviceLowerHost)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // AnchorHostDpu 从高到低找 HOST：phys2(DEVICE) 跳过，phys1(HOST+16==size) 锚定 algo1
    // phys2 在锚点之上但非 HOST，不被选；algo1→phys1, algo0→phys0
    ASSERT_EQ(info.infos[0][0], Range(8));
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// T23: 两层 localRanks 完全相同（均为全集），映射成 TwoLevel
TEST_F(TopoMatchTwoLevelTest, TwoIdenticalFullLocalRanksLayers)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // d0=16, d1=1; group0={0..15}; group1=BuildRepresentativeGroup(16,1,3)={3}
    ASSERT_EQ(info.infos[0][0], Range(16));
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
}

// T24: P3 场景适配 — [Mesh, Mesh] 两层算法，仅一个 1DMESH 层
// algo0(Mesh) 锚 phys0(1DMESH); candidateLow 阻止 algo1 在 phys0 附近锚定;
// algo1 找不到 1DMESH 不锚定，尾段覆盖 → pIndices=[0,2]，全部有效
TEST_F(TopoMatchTwoLevelTest, P3SingleMeshAnchorCandidateLowGuards)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}, true, COMM_TOPO_1DMESH),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::MESH});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // algo0→phys0(1DMESH), candidateLow=1 for algo1; phys1/phys2 均非 1DMESH → algo1 不锚定
    // 尾段: MatchLayerIdxBySegment(1,1,1,2) → pIndices[1]=2
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
    ASSERT_EQ(info.infos[0][0], Range(8));
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
}
