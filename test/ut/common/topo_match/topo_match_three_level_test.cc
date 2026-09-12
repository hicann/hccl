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
#include "topo_match_three_level.h"
#include "topo_match_test_common.h"

using namespace ops_hccl;
using ops_hccl::ut_helper::MakeLevel;
using ops_hccl::ut_helper::MakeProfile;
using ops_hccl::ut_helper::MakeTopoInfo;
using ops_hccl::ut_helper::Range;
using ops_hccl::ut_helper::RangeFrom;

class TopoMatchThreeLevelTest : public ::testing::Test {
protected:
    TopoMatchThreeLevel matcher_;
};

// H1: 对称 3 层 d2>=2，myRank 在 instance1（group1 跨实例修复）
// phys0={16..23},{8,8,8,8}; phys1={16..31},{16,16}; phys2={0..31},{32}; d0=8,d1=2,d2=2
TEST_F(TopoMatchThreeLevelTest, SymmetricD2Ge2Instance1)
{
    auto topo = MakeTopoInfo(
        20, 32,
        {
            MakeLevel(RangeFrom(16, 8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}),
            MakeLevel(RangeFrom(16, 16), PhysicalLevelView::GLOBAL, {16, 16}),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos.size(), 3u);
    ASSERT_EQ(info.infos[0][0], RangeFrom(16, 8));
    // level1Base=(20/16)*16=16, offset=16+20%8=20 → group1={20,28}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{20, 28}));
    // group2=BuildRepresentativeGroup(16, 2, 20%16=4)={4,20}
    ASSERT_EQ(info.infos[2][0], (std::vector<u32>{4, 20}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 3u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0].size(), 1u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1].size(), 1u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2].size(), 1u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
}

// H2: d2>=2，myRank 在 instance0
// phys0={0..7},{8,8,8,8}; phys1={0..15},{16,16}; phys2={0..31},{32}
TEST_F(TopoMatchThreeLevelTest, SymmetricD2Ge2Instance0)
{
    auto topo = MakeTopoInfo(
        3, 32,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 16}),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos.size(), 3u);
    ASSERT_EQ(info.infos[0][0], Range(8));
    // level1Base=0, group1={3,11}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    // group2={3,19}
    ASSERT_EQ(info.infos[2][0], (std::vector<u32>{3, 19}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 3u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
}

// H3: 对称 3 层 d2=1 → 最上层 size 为 1，不支持
// phys0={0..7},{8,8}; phys1={0..15},{16}; phys2={0..15},{16}; d0=8,d1=2,d2=1
TEST_F(TopoMatchThreeLevelTest, SymmetricD2Eq1SingleInstance)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// H4: 非对称 level0 → NOT_SUPPORT
TEST_F(TopoMatchThreeLevelTest, AsymmetricLevel0)
{
    auto topo = MakeTopoInfo(
        2, 32,
        {
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 4, 8, 4}),
            MakeLevel(Range(24), PhysicalLevelView::GLOBAL, {24}),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// H5: 非对称 level1 → NOT_SUPPORT
TEST_F(TopoMatchThreeLevelTest, AsymmetricLevel1)
{
    auto topo = MakeTopoInfo(
        2, 32,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}),
            MakeLevel(Range(24), PhysicalLevelView::GLOBAL, {16, 8}),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// H6: level1TotalSize 不被 d0 整除 → NOT_SUPPORT
TEST_F(TopoMatchThreeLevelTest, Level1TotalSizeNotDivisibleByD0)
{
    auto topo = MakeTopoInfo(
        2, 32,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}),
            MakeLevel(Range(12), PhysicalLevelView::GLOBAL, {12}),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// H7: userRankSize 不被 d0*d1 整除 → NOT_SUPPORT
TEST_F(TopoMatchThreeLevelTest, UserRankSizeNotDivisibleByD0D1)
{
    auto topo = MakeTopoInfo(
        2, 24,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
            MakeLevel(Range(24), PhysicalLevelView::GLOBAL, {24}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// H8: 有效层 < 3 → NOT_SUPPORT
TEST_F(TopoMatchThreeLevelTest, EffectiveLayersLessThanThree)
{
    auto topo = MakeTopoInfo(
        3, 16,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_E_NOT_SUPPORT);
}

// H9: MeshConcur + 上层超集双层
// phys0=1DMESH{0..7},{8,8,8,8}; phys1=CLOS{0..15},{16,16}; phys2=CLOS{0..31},{32}; d0=8,d1=2,d2=2
TEST_F(TopoMatchThreeLevelTest, MeshConcurDualLayerClos)
{
    auto topo = MakeTopoInfo(
        3, 32,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}, true, COMM_TOPO_1DMESH),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 16}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}, true, COMM_TOPO_CLOS),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH_CONCUR, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(info.infos.size(), 3u);
    ASSERT_EQ(info.infos[0][0], Range(8));
    // d0=8, d1=2, d2=2; group1={3,11}; group2={3,19}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    ASSERT_EQ(info.infos[2][0], (std::vector<u32>{3, 19}));
    // MeshConcur → {idx0(1DMESH), idx1(上层超集)}
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0].size(), 2u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][1], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1].size(), 1u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2].size(), 1u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
}

// H10: 4 层物理 + Mesh 锚点（1DMESH 在 phys1，非 phys0）
// phys0=CLOS{0..7},{8,8,8,8}; phys1=1DMESH{0..7},{8,8,8,8}; phys2=CLOS{0..15},{16}; phys3=CLOS{0..31},{32}
// algo0=MESH 锚 phys1(1DMESH); 分段: algo0→phys1, algo1→phys2, algo2→phys3
TEST_F(TopoMatchThreeLevelTest, FourLayersMeshAnchorAtPhys1)
{
    auto topo = MakeTopoInfo(
        3, 32,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}, true, COMM_TOPO_1DMESH),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}, true, COMM_TOPO_CLOS),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // algo0=MESH 锚 phys1(1DMESH); 前段 algo0 之前无 algo; 分段: algo0→phys1, algo1→phys2, algo2→phys3
    // phys0 被 algo0 锚点吸收（段内压缩到 phys1）；d0=8,d1=2,d2=2
    ASSERT_EQ(info.infos.size(), 3u);
    ASSERT_EQ(info.infos[0][0], Range(8)); // phys1 localRanks
    // level1Base=0, group1={3,11}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    // group2={3,19}
    ASSERT_EQ(info.infos[2][0], (std::vector<u32>{3, 19}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 3u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_3);
}

// H11: 4 层 + hostdpu，HOST 层在 phys2（非最高层 phys3）
// phys0=1DMESH{0..7},{8,8,8,8},DEV; phys1=CLOS{0..15},{16},DEV; phys2=CLOS{0..31},{32},HOST;
// phys3=CLOS{0..31},{32},DEV hostdpu 锚 algo2→phys2(HOST); algo0=MESH 锚 phys0; 分段 algo1→phys1
TEST_F(TopoMatchThreeLevelTest, FourLayersHostdpuHostNotTop)
{
    auto topo = MakeTopoInfo(
        3, 32,
        {
            MakeLevel(
                Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}, true, COMM_TOPO_1DMESH, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // hostdpu 锚 algo2→phys2(HOST); Mesh 锚 algo0→phys0(1DMESH); 中间段 algo1→phys1
    // d0=8(phys0), d1=2(phys1=16), d2=2(32/(8*2))
    ASSERT_EQ(info.infos.size(), 3u);
    ASSERT_EQ(info.infos[0][0], Range(8));
    // group1={3,11}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    // group2=BuildRepresentativeGroup(16, 2, 3%16=3)={3,19}
    ASSERT_EQ(info.infos[2][0], (std::vector<u32>{3, 19}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 3u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
}

// H12: 5 层物理 + Mesh 锚点 + 多冗余层压缩。instList 用 {16,16}
// phys0=1DMESH{0..7},{16,16}; phys1=CLOS{0..15},{16,16}(冗余); phys2=CLOS{0..15},{16,16}(冗余);
// phys3=CLOS{0..15},{16,16}; phys4=CLOS{0..31},{32}
TEST_F(TopoMatchThreeLevelTest, FiveLayersMultipleRedundant)
{
    auto topo = MakeTopoInfo(
        3, 32,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {16, 16}, true, COMM_TOPO_1DMESH),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 16}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 16}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 16}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}, true, COMM_TOPO_CLOS),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // algo0=MESH→phys0(1DMESH); 分段: algo1→phys1, algo2→phys4(最高层压缩)
    // d0=8, d1=2, d2=2
    ASSERT_EQ(info.infos.size(), 3u);
    ASSERT_EQ(info.infos[0][0], Range(8));
    // group1={3,11}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    // group2={3,19}
    ASSERT_EQ(info.infos[2][0], (std::vector<u32>{3, 19}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 3u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_4);
}

// H13: 6 层 + hostdpu + Mesh，HOST 在 phys4（非最高 phys5）
// phys0=1DMESH{0..7},{8,8,8,8},DEV; phys1=CLOS{0..15},{16},DEV; phys2=CLOS{0..15},{16},DEV(冗余);
// phys3=CLOS{0..15},{16},DEV; phys4=CLOS{0..31},{32},HOST; phys5=CLOS{0..31},{32},DEV
TEST_F(TopoMatchThreeLevelTest, SixLayersHostdpuHostNotTop)
{
    auto topo = MakeTopoInfo(
        3, 32,
        {
            MakeLevel(
                Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}, true, COMM_TOPO_1DMESH, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_HOST),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}, true, COMM_TOPO_CLOS, ENDPOINT_LOC_TYPE_DEVICE),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR}, OpExecuteConfig::HOSTCPU);
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // hostdpu 锚 algo2→phys4(HOST, 从高到低找首个HOST); Mesh 锚 algo0→phys0; 中间段 algo1→phys3(段内压缩最高)
    // d0=8(phys0), d1=2(phys1=16), d2=2(32/16)
    ASSERT_EQ(info.infos.size(), 3u);
    ASSERT_EQ(info.infos[0][0], Range(8));
    // group1={3,11}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    // group2={3,19}
    ASSERT_EQ(info.infos[2][0], (std::vector<u32>{3, 19}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 3u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_3);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_4);
}

// H14: 多层物理，其中一层非对称但不是 phys0 也不是 phys1 → SUCCESS
// phys0={0..7},{8,8,8,8}(对称); phys1={0..15},{16,16}(对称); phys2={0..23},{16,8}(非对称,被压缩吸收);
// phys3={0..31},{32}(top)
TEST_F(TopoMatchThreeLevelTest, AsymmetricLayerNotPhys0Or1)
{
    auto topo = MakeTopoInfo(
        3, 32,
        {
            MakeLevel(Range(8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8}),
            MakeLevel(Range(16), PhysicalLevelView::GLOBAL, {16, 16}),
            MakeLevel(Range(24), PhysicalLevelView::GLOBAL, {16, 8}),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32}),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::NHR});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // 4 effIdx > 3 → FindAnchors: algo0=MESH 无 1DMESH → 跳过; 分段压缩: algo0→phys0, algo1→phys1, algo2→phys3
    // phys2(非对称)被压缩吸收，不参与对称校验 → SUCCESS; d0=8,d1=2,d2=2
    ASSERT_EQ(info.infos.size(), 3u);
    ASSERT_EQ(info.infos[0][0], Range(8));
    // group1={3,11}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{3, 11}));
    // group2={3,19}
    ASSERT_EQ(info.infos[2][0], (std::vector<u32>{3, 19}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 3u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_3);
}

// H15: P3 原始场景 — [Mesh, NHR, Mesh] 三层算法，5 层物理，两个相邻 1DMESH
// 旧代码: algo0 锚 phys2, algo2 锚 phys3(相邻) → pIndices[1]=INVALID_UINT → 越界读
// 修复后: candidateLow=2+(2-0)=4, phys4 非 1DMESH → algo2 不锚定 → 尾段覆盖 algo1,2 → pIndices=[2,3,4]
TEST_F(TopoMatchThreeLevelTest, P3AdjacentMeshAnchorsNoInvalidUint)
{
    auto topo = MakeTopoInfo(
        30, 128,
        {
            MakeLevel(RangeFrom(24, 8), PhysicalLevelView::GLOBAL, {8, 8, 8, 8, 8, 8, 8, 8}, true, COMM_TOPO_CLOS),
            MakeLevel(RangeFrom(16, 16), PhysicalLevelView::GLOBAL, {16, 16, 16, 16}, true, COMM_TOPO_CLOS),
            MakeLevel(Range(32), PhysicalLevelView::GLOBAL, {32, 32, 32, 32}, true, COMM_TOPO_1DMESH),
            MakeLevel(Range(64), PhysicalLevelView::GLOBAL, {64, 64}, true, COMM_TOPO_1DMESH),
            MakeLevel(Range(128), PhysicalLevelView::GLOBAL, {128}, true, COMM_TOPO_CLOS),
        });
    AlgAttrs profile = MakeProfile({AlgoType::MESH, AlgoType::NHR, AlgoType::MESH});
    AlgHierarchyInfoForAllLevel info;
    ASSERT_EQ(matcher_.MatchTopo(&topo, info, profile), HcclResult::HCCL_SUCCESS);
    // algo0(Mesh) 锚 phys2(1DMESH); candidateLow for algo2 = 2+(2-0)=4, phys4 非 1DMESH → 不锚定
    // 尾段: MatchLayerIdxBySegment(1,2,3,4) → pIndices=[2,3,4]
    // d0=32(phys2), d1=64/32=2(phys3), d2=128/32/2=2(phys4)
    ASSERT_EQ(info.infos.size(), 3u);
    ASSERT_EQ(info.infos[0][0], Range(32));
    // level1Base=0, group1=BuildRepresentativeGroup(32, 2, 30)={30,62}
    ASSERT_EQ(info.infos[1][0], (std::vector<u32>{30, 62}));
    // group2=BuildRepresentativeGroup(64, 2, 30)={30,94}
    ASSERT_EQ(info.infos[2][0], (std::vector<u32>{30, 94}));
    ASSERT_EQ(info.physicalIdxForAlgoLevels.size(), 3u);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[0][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_2);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[1][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_3);
    ASSERT_EQ(info.physicalIdxForAlgoLevels[2][0], PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_4);
}
