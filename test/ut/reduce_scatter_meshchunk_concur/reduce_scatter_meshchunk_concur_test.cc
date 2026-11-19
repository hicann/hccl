/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <vector>

#include "ins_temp_reduce_scatter_mesh_1D_meshchunk_concur.h"
#include "cost_model.h"

using namespace ops_hccl;

namespace {

// 构造最小化 OpParam（仅填模板构造所需字段）
OpParam MakeOpParam()
{
    OpParam param{};
    param.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    param.DataDes.count = 1024;
    param.reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    return param;
}

// 构造 4 rank 通信域
std::vector<std::vector<u32>> MakeSubCommRanks(u32 rankSize = 4)
{
    std::vector<u32> ranks;
    for (u32 i = 0; i < rankSize; i++) {
        ranks.push_back(i);
    }
    return {ranks};
}

// 构造带 portGroupSize 的 ChannelInfo
ChannelInfo MakeChannelInfo(u32 portGroupSize)
{
    ChannelInfo info{};
    info.portGroupSize = portGroupSize;
    return info;
}

} // namespace

// ============================================================
// ScratchSlotBaseOff: 槽位寻址正确性
// ============================================================
TEST(MeshChunkConcurTest, ScratchSlotBaseOffSoleDegeneratesToBase)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    TemplateDataParams p{};
    // sole 参数: stride 全 0 → 退化为 hcclBuffBaseOff
    p.buffInfo.hcclBuffBaseOff = 4096;
    p.outputRepeatStride = 0;
    p.outputSliceStride = 0;
    EXPECT_EQ(tmpl->ScratchSlotBaseOff(p, 0, 0), 4096u);
    EXPECT_EQ(tmpl->ScratchSlotBaseOff(p, 0, 3), 4096u);
    EXPECT_EQ(tmpl->ScratchSlotBaseOff(p, 2, 1), 4096u);
}

TEST(MeshChunkConcurTest, ScratchSlotBaseOffSequenceBlockLayout)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    // 序列参数: L0=4 slots, block stride = 4*256 = 1024
    TemplateDataParams p{};
    p.buffInfo.hcclBuffBaseOff = 0;
    p.outputRepeatStride = 1024;                                   // L0 * C * dt
    p.outputSliceStride = 256;                                     // C * dt
    EXPECT_EQ(tmpl->ScratchSlotBaseOff(p, 0, 0), 0u);              // block0 slot0
    EXPECT_EQ(tmpl->ScratchSlotBaseOff(p, 0, 1), 256u);            // block0 slot1
    EXPECT_EQ(tmpl->ScratchSlotBaseOff(p, 0, 3), 768u);            // block0 slot3
    EXPECT_EQ(tmpl->ScratchSlotBaseOff(p, 1, 0), 1024u);           // block1 slot0
    EXPECT_EQ(tmpl->ScratchSlotBaseOff(p, 1, 2), 1024 + 512u);     // block1 slot2
    EXPECT_EQ(tmpl->ScratchSlotBaseOff(p, 2, 3), 2 * 1024 + 768u); // block2 slot3
}

// ============================================================
// CalcScratchMultiple: 序列 vs sole 语义区分
// ============================================================
TEST(MeshChunkConcurTest, CalcScratchMultipleSequenceContext)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    // 序列框内 (INPUT, HCCL_BUFFER) → templateRankSize_
    EXPECT_EQ(tmpl->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER), 4u);
}

TEST(MeshChunkConcurTest, CalcScratchMultipleSoleContext)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    // sole (INPUT, OUTPUT) → templateRankSize_ - 1
    EXPECT_EQ(tmpl->CalcScratchMultiple(BufferType::INPUT, BufferType::OUTPUT), 3u);
}

// ============================================================
// GetThreadNum: 线程数随通道数扩展
// ============================================================
TEST(MeshChunkConcurTest, GetThreadNumDefaultSingleChannel)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    // 默认 channelsPerRank_=1: (4-1)*1 = 3(含主线 0, queIdx 范围 [0,2])
    EXPECT_EQ(tmpl->GetThreadNum(), 3u);
}

TEST(MeshChunkConcurTest, GetThreadNumMultiChannel)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    // 设置 2 通道/对端 (mesh + clos)
    std::map<u32, std::vector<ChannelInfo>> channelMap;
    channelMap[1] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[2] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[3] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    ASSERT_EQ(tmpl->SetchannelsPerRank(channelMap), HCCL_SUCCESS);

    // channelsPerRank_=2: (4-1)*2 = 6(含主线 0, queIdx 范围 [0,5])
    EXPECT_EQ(tmpl->GetThreadNum(), 6u);
}

// ============================================================
// CalcDataSplitByPortGroup: 基类均分 + Concur 带宽加权
// ============================================================
TEST(MeshChunkConcurTest, CalcDataSplitSingleChannelFullChunk)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    // 1 通道 → 全量数据走该通道
    std::vector<ChannelInfo> channels = {MakeChannelInfo(1)};
    std::vector<u64> elemCount, size, offset;
    ASSERT_EQ(tmpl->CalcDataSplitByPortGroup(1000, 4, channels, elemCount, size, offset), HCCL_SUCCESS);
    EXPECT_EQ(elemCount.size(), 1u);
    EXPECT_EQ(elemCount[0], 1000u);
    EXPECT_EQ(size[0], 4000u);
    EXPECT_EQ(offset[0], 0u);
}

TEST(MeshChunkConcurTest, CalcDataSplitBandwidthWeighted)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    // 设置 2 通道: mesh(portGroupSize=3) + clos(portGroupSize=8)
    std::map<u32, std::vector<ChannelInfo>> channelMap;
    channelMap[1] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[2] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[3] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    ASSERT_EQ(tmpl->SetchannelsPerRank(channelMap), HCCL_SUCCESS);

    // Concur 子类: level0Weight = rankSize-1 = 3, level1Bandwidth = 8, totalWeight = 11
    std::vector<ChannelInfo> channels = {MakeChannelInfo(3), MakeChannelInfo(8)};
    std::vector<u64> elemCount, size, offset;
    u64 total = 1100; // 选 1100 使切分结果整数
    ASSERT_EQ(tmpl->CalcDataSplitByPortGroup(total, 4, channels, elemCount, size, offset), HCCL_SUCCESS);

    EXPECT_EQ(elemCount.size(), 2u);
    // level0 (mesh): 1100 * 3/11 = 300
    // level1 (clos): 1100 * 8/11 = 800
    EXPECT_EQ(elemCount[0], 300u);
    EXPECT_EQ(elemCount[1], 800u);
    EXPECT_EQ(size[0], 1200u);
    EXPECT_EQ(size[1], 3200u);
    EXPECT_EQ(offset[0], 0u);
    EXPECT_EQ(offset[1], 1200u);
    // 总量守恒
    EXPECT_EQ(elemCount[0] + elemCount[1], total);
}

// ============================================================
// CalcDataSplitByPortGroup: 极小切片防退化(<2K 挂死修复)
// ============================================================
TEST(MeshChunkConcurTest, CalcDataSplitTinySliceBorrowFromLargest)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    // 2 通道(mesh=3, clos=8), 权重 3/11 vs 8/11: 2 元素时 mesh 加权取整为 0
    std::map<u32, std::vector<ChannelInfo>> channelMap;
    channelMap[1] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[2] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[3] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    ASSERT_EQ(tmpl->SetchannelsPerRank(channelMap), HCCL_SUCCESS);

    std::vector<ChannelInfo> channels = {MakeChannelInfo(3), MakeChannelInfo(8)};
    std::vector<u64> elemCount, size, offset;
    // 原始切分 [0, 2](mesh 取整清零) → 借元素保底每通道 ≥1: [1, 1]
    ASSERT_EQ(tmpl->CalcDataSplitByPortGroup(2, 4, channels, elemCount, size, offset), HCCL_SUCCESS);
    EXPECT_EQ(elemCount.size(), 2u);
    EXPECT_EQ(elemCount[0], 1u);
    EXPECT_EQ(elemCount[1], 1u);
    EXPECT_EQ(size[0], 4u);
    EXPECT_EQ(size[1], 4u);
    EXPECT_EQ(offset[0], 0u);
    EXPECT_EQ(offset[1], 4u);
    // 总量守恒
    EXPECT_EQ(elemCount[0] + elemCount[1], 2u);
}

TEST(MeshChunkConcurTest, CalcDataSplitInsufficientElementsNotSupport)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    std::map<u32, std::vector<ChannelInfo>> channelMap;
    channelMap[1] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[2] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[3] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    ASSERT_EQ(tmpl->SetchannelsPerRank(channelMap), HCCL_SUCCESS);

    std::vector<ChannelInfo> channels = {MakeChannelInfo(3), MakeChannelInfo(8)};
    std::vector<u64> elemCount, size, offset;
    // 1 元素 < 通道数 2: 任何切分必留空槽(空槽与静态线程/notify 结构不兼容), 干净失败
    EXPECT_EQ(tmpl->CalcDataSplitByPortGroup(1, 4, channels, elemCount, size, offset), HCCL_E_NOT_SUPPORT);
}

TEST(MeshChunkConcurTest, CalcDataSplitZeroSliceNotSupport)
{
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(MakeOpParam(), 0, MakeSubCommRanks(4));

    std::map<u32, std::vector<ChannelInfo>> channelMap;
    channelMap[1] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[2] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    channelMap[3] = {MakeChannelInfo(3), MakeChannelInfo(8)};
    ASSERT_EQ(tmpl->SetchannelsPerRank(channelMap), HCCL_SUCCESS);

    std::vector<ChannelInfo> channels = {MakeChannelInfo(3), MakeChannelInfo(8)};
    std::vector<u64> elemCount, size, offset;
    // 0 元素 peer 切片(基类小数据按元素均分可产生 0 字节 peer 块): 同空槽问题, 干净失败
    EXPECT_EQ(tmpl->CalcDataSplitByPortGroup(0, 4, channels, elemCount, size, offset), HCCL_E_NOT_SUPPORT);
}

// ============================================================
// CalcRes: 极小数据 host 侧守卫(内核启动前干净失败, 防 350s notify 连环挂)
// ============================================================
TEST(MeshChunkConcurTest, CalcResTinyDataNotSupport)
{
    // 4 rank(3 peer)/stub 双通道: unit 5 元素 → 最小 peer 切片 5/3=1 < 通道数 2 → NOT_SUPPORT
    OpParam param = MakeOpParam();
    param.DataDes.count = 5;
    auto tmpl = std::make_unique<InsTempReduceScatterMesh1DMeshChunkConcur>(param, 0, MakeSubCommRanks(4));
    AlgResourceRequest resRequest{};
    EXPECT_EQ(tmpl->CalcRes(nullptr, param, nullptr, resRequest), HCCL_E_NOT_SUPPORT);
}

// ============================================================
// 模板静态属性: props 标识
// ============================================================
TEST(MeshChunkConcurTest, TemplatePropsAlgoType)
{
    // 变体模板报精确类型(同族惯例: MESH_CONCUR/MESH_CHUNK_TWOSHOT 等), 供类型过滤场景对齐
    EXPECT_EQ(InsTempReduceScatterMesh1DMeshChunkConcur::props.algoType, AlgoType::MESH_CHUNK_CONCUR);
}

// ============================================================
// CalcCostCoeff: rankSize 守卫 + A 项(双路分流取 max × 1.39 净系数)
// ============================================================
TEST(MeshChunkConcurTest, CalcCostCoeffRankSizeGuard)
{
    CalcCostCoeffParam param;
    param.rankSize = 1;
    EXPECT_TRUE(InsTempReduceScatterMesh1DMeshChunkConcur::CalcCostCoeff(param).empty());
    param.rankSize = 9;
    EXPECT_TRUE(InsTempReduceScatterMesh1DMeshChunkConcur::CalcCostCoeff(param).empty());
}

TEST(MeshChunkConcurTest, CalcCostCoeffTermAWeightedMaxAndNetRatio)
{
    // 8p 单机 + phyLevel 提供逐层形态/端口(2 条 CLOS 通道各 4 口) + isPod=true:
    // 锁定 A0(传 isPod)/A1(固定传 false) 的传参约定与 1.39 净系数
    CalcCostCoeffParam param;
    param.rankSize = 8;
    param.dataRatio = 1.0f;
    param.isPod = true;
    param.inputBuffer = BufferType::INPUT;
    param.scratchBuffer = BufferType::HCCL_BUFFER;
    param.phyLevelIdxs = {PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0, PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1};
    param.phyLevelNetTypes = {CommTopo::COMM_TOPO_1DMESH, CommTopo::COMM_TOPO_CLOS};
    param.phyLevelPortNums = {{}, {4, 4}};

    // 期望值与实现走同一 CostModelManager 入口对拍, 锁定分流比/max/净系数/isPod 传参逻辑:
    // mesh 权重 = rankSize-1(7), CLOS = 8 口, A = max(A0, A1) × 1.39
    const u32 level0Bandwidth = 7;
    const int level1PortTotal = 8;
    const float level0Ratio
        = static_cast<float>(level0Bandwidth) / (level0Bandwidth + static_cast<u32>(level1PortTotal));
    float expA0 = 0.0f;
    CostModelManager::Global()->CalcMeshParam(
        param.dataRatio * level0Ratio, CommTopo::COMM_TOPO_1DMESH, 0, param.rankSize, expA0, param.isPod);
    float expA1 = 0.0f;
    CostModelManager::Global()->CalcMeshParam(
        param.dataRatio * (1.0f - level0Ratio), CommTopo::COMM_TOPO_CLOS, level1PortTotal, param.rankSize, expA1,
        false);
    const float expA = std::max(expA0, expA1) * 1.39f;

    auto params = InsTempReduceScatterMesh1DMeshChunkConcur::CalcCostCoeff(param);
    ASSERT_EQ(params.size(), 1u);
    EXPECT_FLOAT_EQ(params[0].A, expA);
    // B: input != scratch → PreCopy 项生效(与实现同一入口对拍)
    float expB = 0.0f;
    CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, expB);
    EXPECT_FLOAT_EQ(params[0].B, expB);
    EXPECT_GT(params[0].C, 0.0f);
    EXPECT_GT(params[0].D, 0.0f);
}

TEST(MeshChunkConcurTest, CalcCostCoeffFallbackPortAndZeroPreCopy)
{
    // phyLevel 信息缺失 → level1PortTotal 走 fallback=8, A 与显式 8 口结果逐位一致
    CalcCostCoeffParam explicitParam;
    explicitParam.rankSize = 8;
    explicitParam.dataRatio = 1.0f;
    explicitParam.inputBuffer = BufferType::INPUT;
    explicitParam.scratchBuffer = BufferType::HCCL_BUFFER;
    explicitParam.phyLevelIdxs = {PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_0, PhysicalLevelIndex::PHYSICAL_LEVEL_IDX_1};
    explicitParam.phyLevelNetTypes = {CommTopo::COMM_TOPO_1DMESH, CommTopo::COMM_TOPO_CLOS};
    explicitParam.phyLevelPortNums = {{}, {4, 4}};
    auto explicitParams = InsTempReduceScatterMesh1DMeshChunkConcur::CalcCostCoeff(explicitParam);
    ASSERT_EQ(explicitParams.size(), 1u);

    CalcCostCoeffParam fallbackParam; // 无 phyLevel 信息且 portNum 为空
    fallbackParam.rankSize = 8;
    fallbackParam.dataRatio = 1.0f;
    fallbackParam.inputBuffer = BufferType::INPUT;
    fallbackParam.scratchBuffer = BufferType::HCCL_BUFFER;
    auto fallbackParams = InsTempReduceScatterMesh1DMeshChunkConcur::CalcCostCoeff(fallbackParam);
    ASSERT_EQ(fallbackParams.size(), 1u);
    EXPECT_FLOAT_EQ(fallbackParams[0].A, explicitParams[0].A);

    // input == scratch(对称内存免 PreCopy) → B = 0
    fallbackParam.inputBuffer = fallbackParam.scratchBuffer;
    auto noPreCopyParams = InsTempReduceScatterMesh1DMeshChunkConcur::CalcCostCoeff(fallbackParam);
    ASSERT_EQ(noPreCopyParams.size(), 1u);
    EXPECT_FLOAT_EQ(noPreCopyParams[0].B, 0.0f);
}
