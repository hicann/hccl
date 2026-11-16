/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BatchSendRecv AIV 的 host 侧 UT：被测对象是生产的模板（aiv_temp_batch_send_recv_mesh_1D.cc）与
// executor（ins_v2_aiv_batch_send_recv_sole_executor.cc）。
//
// 观测手段只有两个，都不依赖私有成员：
//   1) ExecuteKernelLaunchBsr 桩：executor 每轮把 ExtraArgsBsr 交给模板、模板再交给 launch，
//      所以桩里收到的 (AivOpArgs, ExtraArgsBsr) 就是"这一轮实际下发的内容"——分轮结果、片序、
//      桶偏移、字节数、flagBlock、tag 全在这里成对可见；
//   2) CalcChannelRequest* 桩：CalcRes 传下来的对端集合就是"本调用要为哪些对端建链"。
//
// 覆盖重点（这些一旦错，device 侧要么静默读脏数据、要么 flag 永远等不到）：
//   - tag 只由轮次决定，值域与其它 AIV 算子隔离，越界回绕后仍留在本值域；
//   - 收发两侧对同一 pair 切出完全一致的 (remoteRank, sliceIdx, byteCount)，flag 格号才能配上；
//   - 桶偏移按对端分桶、recv 侧落在自己那一桶，跨轮不重叠；
//   - 自收发的目的地址来自配对 recv 项，配对缺失要在 host 侧就拦住；
//   - rankSize 取到上限 256 时每 pair 每轮只剩 1 片，2*rankSize*flagBlock 仍不越任务表；
//   - 零任务 rank（itemNum==0 或全零字节）不下发任何一轮，也不消耗轮次。

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "aiv_temp_batch_send_recv_mesh_1D.h"
#include "alg_attrs_registry.h"
#include "alg_param.h"
#include "channel_request.h"
#include "hccl_aiv_utils.h"

namespace ops_hccl {
namespace bsr_host_test {

    // 一轮下发的完整快照。ExtraArgsBsr 12KB，按轮次拷贝，用例规模下无压力。
    struct LaunchRecord {
        AivOpArgs args;
        ExtraArgsBsr extra;
    };

    inline std::vector<LaunchRecord> launches;

    // 让 executor 的失败注入可控
    inline HcclResult launchResult = HcclResult::HCCL_SUCCESS;

} // namespace bsr_host_test

// 生产 hccl_aiv_utils.h 只声明该入口；UT 用它观测每轮下发内容
HcclResult ExecuteKernelLaunchBsr(const AivOpArgs& opArgs, const ExtraArgsBsr& extraArgsBsr)
{
    bsr_host_test::launches.push_back({opArgs, extraArgsBsr});
    return bsr_host_test::launchResult;
}

} // namespace ops_hccl

// executor 是模板类且成员定义在 .cc 里，直接编入本 TU：这样 Orchestrate/DispatchOneRound 的
// 定义对测试可见，无需在测试里另写显式实例化。
#include "ins_v2_aiv_batch_send_recv_sole_executor.cc"

namespace {

using namespace ops_hccl;

constexpr u32 TAG_LOWER_BOUND = 1025U;   // AIV_BSR_TAG_BASE + 1
constexpr u32 OTHER_OPS_TAG_MAX = 1000U; // 其它 AIV 算子的 tag 上界（TAG_RESET_COUNT）
constexpr u32 SEND_TYPE = 0;
constexpr u32 RECV_TYPE = 1;
constexpr u32 SELF_SEND_TYPE = 2;
constexpr u32 SELF_RECV_TYPE = 3;
constexpr u64 BUCKET_ALIGN = 32;

// 用户 buffer 用真实内存：被测代码只做地址加减，但让地址唯一可区分断言才算得清。
class BufferPool {
public:
    void* Add(size_t bytes = 64)
    {
        blocks_.emplace_back(bytes, 0);
        return blocks_.back().data();
    }

private:
    std::vector<std::vector<uint8_t>> blocks_;
};

HcclSendRecvItem MakeItem(void* buf, u64 count, HcclDataType dataType, u32 remoteRank, HcclSendRecvType type)
{
    HcclSendRecvItem item;
    item.buf = buf;
    item.count = count;
    item.dataType = dataType;
    item.remoteRank = remoteRank;
    item.sendRecvType = type;
    return item;
}

OpParam MakeParam(std::vector<HcclSendRecvItem>& items, u32 numBlocksLimit = 48)
{
    OpParam param;
    param.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    param.numBlocksLimit = numBlocksLimit;
    param.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    param.batchSendRecvDataDes.sendRecvItemsPtr = items.data();
    param.batchSendRecvDataDes.itemNum = static_cast<u32>(items.size());
    return param;
}

// 模板用例直接构造 OpParam 时必须给核预算：numBlocksLimit 由上层填，0 不是"不限"而是"核不够"，
// CalNumBlocks 会直接判 NOT_SUPPORT。
OpParam MakeTemplateParam(u32 numBlocksLimit = 48)
{
    OpParam param;
    param.opMode = OpMode::OPBASE;
    param.numBlocksLimit = numBlocksLimit;
    return param;
}

std::vector<u32> AllRanks(u32 rankSize)
{
    std::vector<u32> ranks;
    for (u32 r = 0; r < rankSize; ++r) {
        ranks.push_back(r);
    }
    return ranks;
}

AlgResourceCtxSerializable MakeResCtx(u32 myRank, u32 rankSize, u64 cclSize, void* aivCommInfo = nullptr)
{
    AlgResourceCtxSerializable ctx;
    ctx.topoInfo.userRank = myRank;
    ctx.topoInfo.userRankSize = rankSize;
    ctx.cclMem.size = cclSize;
    ctx.cclMem.addr = reinterpret_cast<void*>(0x1000);
    ctx.aivCommInfoPtr = aivCommInfo;
    // infos[0] 是该层唯一的子组，含全体 rank：模板的 tempRankSize_ 就取它的 size()
    ctx.algHierarchyInfo.infos.resize(1);
    ctx.algHierarchyInfo.infos[0].push_back(AllRanks(rankSize));
    return ctx;
}

using Executor = InsV2AivBatchSendRecvSoleExecutor<TopoMatchOneLevel, AivTempBatchSendRecvMesh1D>;

class BsrHostTest : public ::testing::Test {
protected:
    void SetUp() override { ResetObservations(); }

    // 合并后的用例在一个函数里跑多段场景，段间要显式清观测值
    static void ResetObservations()
    {
        bsr_host_test::launches.clear();
        bsr_host_test::launchResult = HcclResult::HCCL_SUCCESS;
        bsr_host_test::meshChannelRequests = 0;
        bsr_host_test::meshClosChannelRequests = 0;
        bsr_host_test::closProtocolsReturned.clear();
        bsr_host_test::channelRequestRanks.clear();
    }

    const bsr_host_test::LaunchRecord& LaunchAt(size_t idx) const
    {
        EXPECT_LT(idx, bsr_host_test::launches.size());
        return bsr_host_test::launches[idx];
    }

    // 把一轮的下发项摘成可比较的元组，便于断言"收发两侧一致"
    static std::vector<std::tuple<u32, u32, u32, u32>> SliceKeys(const bsr_host_test::LaunchRecord& record)
    {
        std::vector<std::tuple<u32, u32, u32, u32>> keys;
        for (u64 i = 0; i < record.extra.itemNum; ++i) {
            const auto& item = record.extra.items[i];
            keys.emplace_back(
                static_cast<u32>(item.sendRecvType), static_cast<u32>(item.sliceIdx), item.byteCount,
                static_cast<u32>(item.peer.remoteRank));
        }
        return keys;
    }
};

// ---------------------------------------------------------------------------------------------
// 模板：tag 生成
// ---------------------------------------------------------------------------------------------

// tag 全在 kernel 内按轮次算：host 只负责把 roundIdx 换成 sliceId 下发。三件事一起守——
// 值域与其它 AIV 算子不相交、越界回绕后仍留在本值域、同一次调用内相邻轮不重复。
TEST_F(BsrHostTest, TagAdvancesPerRoundAndStaysInsideItsOwnDomain)
{
    OpParam param = MakeTemplateParam();
    const std::vector<std::vector<u32>> subCommRanks = {AllRanks(2)};
    AivTempBatchSendRecvMesh1D tmpl(param, 0, subCommRanks);
    ExtraArgsBsr extra{};
    tmpl.SetBatchSendRecvItems(extra);

    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.hcclBuff.size = 1 << 20;
    TemplateResource resource;

    // 覆盖：起点、环内、环末、回绕首轮、回绕之后、多圈之后
    const std::vector<u32> roundIdxs = {0,
                                        1,
                                        2,
                                        AIV_BSR_TAG_SPAN - 2,
                                        AIV_BSR_TAG_SPAN - 1,
                                        AIV_BSR_TAG_SPAN,
                                        AIV_BSR_TAG_SPAN + 1,
                                        3 * AIV_BSR_TAG_SPAN};
    std::vector<u32> tags;
    for (u32 roundIdx : roundIdxs) {
        bsr_host_test::launches.clear();
        tmpl.SetRoundIdx(roundIdx);
        ASSERT_EQ(tmpl.KernelRun(param, tempAlgParams, resource), HcclResult::HCCL_SUCCESS);
        const u32 tag = LaunchAt(0).args.sliceId;
        EXPECT_EQ(tag, AIV_BSR_TAG_BASE + 1U + roundIdx % AIV_BSR_TAG_SPAN) << "roundIdx=" << roundIdx;
        // 值域封闭：回绕后不允许掉回其它算子的 [1, 1000]
        EXPECT_GT(tag, OTHER_OPS_TAG_MAX) << "roundIdx=" << roundIdx;
        EXPECT_LE(tag, AIV_BSR_TAG_MAX) << "roundIdx=" << roundIdx;
        // tag<<16 不能进 int32 符号位（device 侧按 int32 比较）
        EXPECT_LE(static_cast<u32>(tag << 16), 0x7FFFFFFFU) << "roundIdx=" << roundIdx;
        tags.push_back(tag);
    }
    // 首轮的 tag 下界严格大于其它算子的上界，这就是隔离本身
    EXPECT_EQ(tags[0], TAG_LOWER_BOUND);
    EXPECT_GT(AIV_BSR_TAG_BASE, OTHER_OPS_TAG_MAX);

    // 相邻轮不同 tag：flag 值 = (tag<<16)|sliceIdx，同 tag 会让上一轮残留在本轮同号片上被误判为"已到达"
    constexpr u32 ROUNDS = 512;
    std::vector<u32> seq;
    for (u32 r = 0; r < ROUNDS; ++r) {
        bsr_host_test::launches.clear();
        tmpl.SetRoundIdx(r);
        ASSERT_EQ(tmpl.KernelRun(param, tempAlgParams, resource), HcclResult::HCCL_SUCCESS);
        seq.push_back(LaunchAt(0).args.sliceId);
    }
    EXPECT_EQ(std::adjacent_find(seq.begin(), seq.end()), seq.end());
}

// ---------------------------------------------------------------------------------------------
// 模板：核数与下发入参
// ---------------------------------------------------------------------------------------------

TEST_F(BsrHostTest, KernelRunCarriesRoundDataAndRejectsAbnormalInput)
{
    // 1) 正常路径：12 个下发字段逐个核对
    {
        OpParam param;
        param.opMode = OpMode::OFFLOAD;
        param.numBlocksLimit = 24;
        void* aivCommInfo = reinterpret_cast<void*>(0x2000);
        const std::vector<std::vector<u32>> subCommRanks = {AllRanks(4)};
        AivTempBatchSendRecvMesh1D tmpl(param, 3, subCommRanks);
        ExtraArgsBsr extra{};
        extra.itemNum = 2;
        extra.flagBlock = 64;
        extra.items[0].byteCount = 96;
        extra.items[1].byteCount = 32;
        tmpl.SetBatchSendRecvItems(extra);
        tmpl.SetDfxTotalBytes(128);
        tmpl.SetRoundIdx(7);

        TemplateDataParams tempAlgParams;
        tempAlgParams.buffInfo.hcclBuff.size = 8 << 20;
        TemplateResource resource;
        resource.aivCommInfoPtr = aivCommInfo;
        ASSERT_EQ(tmpl.KernelRun(param, tempAlgParams, resource), HcclResult::HCCL_SUCCESS);

        const auto& record = LaunchAt(0);
        EXPECT_EQ(record.args.cmdType, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
        EXPECT_EQ(record.args.rank, 3U);
        EXPECT_EQ(record.args.rankSize, 4U);
        EXPECT_EQ(record.args.count, 8U << 20);
        EXPECT_EQ(record.args.dataType, HCCL_DATA_TYPE_INT8);
        EXPECT_EQ(record.args.buffersIn, aivCommInfo);
        EXPECT_EQ(record.args.isOpBase, false);
        EXPECT_EQ(record.args.numBlocks, 24U);
        EXPECT_EQ(record.args.totalDataCount, 128U);
        EXPECT_EQ(record.args.sliceId, AIV_BSR_TAG_BASE + 1U + 7U);
        EXPECT_EQ(record.extra.itemNum, 2U);
        EXPECT_EQ(record.extra.flagBlock, 64U);
    }

    ResetObservations();

    // 2) 任务表溢出必须在模板层就拦住，不能靠 device 侧截断
    {
        OpParam param = MakeTemplateParam();
        const std::vector<std::vector<u32>> subCommRanks = {AllRanks(2)};
        AivTempBatchSendRecvMesh1D tmpl(param, 0, subCommRanks);
        ExtraArgsBsr extra{};
        extra.itemNum = AIV_BSR_ITEM_MAX + 1U;
        tmpl.SetBatchSendRecvItems(extra);

        TemplateDataParams tempAlgParams;
        TemplateResource resource;
        EXPECT_EQ(tmpl.KernelRun(param, tempAlgParams, resource), HcclResult::HCCL_E_PARA);
        EXPECT_TRUE(bsr_host_test::launches.empty());
    }

    // 3) launch 失败原样上抛，不吞错
    {
        OpParam param = MakeTemplateParam();
        const std::vector<std::vector<u32>> subCommRanks = {AllRanks(2)};
        AivTempBatchSendRecvMesh1D tmpl(param, 0, subCommRanks);
        ExtraArgsBsr extra{};
        tmpl.SetBatchSendRecvItems(extra);

        bsr_host_test::launchResult = HcclResult::HCCL_E_INTERNAL;
        TemplateDataParams tempAlgParams;
        TemplateResource resource;
        EXPECT_EQ(tmpl.KernelRun(param, tempAlgParams, resource), HcclResult::HCCL_E_INTERNAL);
    }

    // 4) 核预算不足 2 个（1 发 1 收）直接判不支持，且不改写调用方的出参
    {
        OpParam param;
        const std::vector<std::vector<u32>> subCommRanks = {AllRanks(2)};
        AivTempBatchSendRecvMesh1D tmpl(param, 0, subCommRanks);

        u32 numBlocks = 0xFFFF;
        EXPECT_EQ(tmpl.CalNumBlocks(numBlocks, 0, 0), HcclResult::HCCL_E_NOT_SUPPORT);
        EXPECT_EQ(tmpl.CalNumBlocks(numBlocks, 0, 1), HcclResult::HCCL_E_NOT_SUPPORT);
        EXPECT_EQ(numBlocks, 0xFFFFU);
        EXPECT_EQ(tmpl.CalNumBlocks(numBlocks, 0, 2), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(numBlocks, 2U);
        EXPECT_EQ(tmpl.CalNumBlocks(numBlocks, 1U << 30, 48), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(numBlocks, 48U);
    }
}

// ---------------------------------------------------------------------------------------------
// 模板：增量建链的对端集合与拓扑分支
// ---------------------------------------------------------------------------------------------

TEST_F(BsrHostTest, CalcResLinksOnlyActivePeersAcrossTopologies)
{
    const std::vector<std::vector<u32>> subCommRanks = {AllRanks(4)};

    // 1) 只为本次真正通信的对端建链：count==0 的项（含指向 rank3 的）都要被滤掉，自收发不经链
    {
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(), 4, HCCL_DATA_TYPE_FP32, 1, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(), 8, HCCL_DATA_TYPE_FP32, 1, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(), 4, HCCL_DATA_TYPE_FP32, 2, HcclSendRecvType::HCCL_RECV));
        // count==0 的自项与 count==0 的跨 rank 项都不该占链：后者特意指向 rank3，而 rank3 不出现在任何
        // 有效项里，于是"有没有按 count 过滤"直接体现在 activeRanks 是否多出 3。
        items.push_back(MakeItem(pool.Add(), 0, HCCL_DATA_TYPE_FP32, 0, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(), 0, HCCL_DATA_TYPE_FP32, 3, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(), 4, HCCL_DATA_TYPE_FP32, 0, HcclSendRecvType::HCCL_SEND));

        OpParam param = MakeParam(items);
        TopoInfoWithNetLayerDetails topoInfo;
        topoInfo.userRank = 0;
        topoInfo.userRankSize = 4;
        AivTempBatchSendRecvMesh1D tmpl(param, 0, subCommRanks);

        AlgResourceRequest request;
        ASSERT_EQ(tmpl.CalcRes(nullptr, param, &topoInfo, request), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(request.slaveThreadNum, 0U);
        EXPECT_EQ(request.notifyNumOnMainThread, 0U);
        ASSERT_EQ(request.channels.size(), 1U);
        // 自己 + 去重后的活跃对端，按首次出现顺序
        const std::vector<u32> expected = {0, 1, 2};
        ASSERT_EQ(bsr_host_test::channelRequestRanks.size(), 1U);
        EXPECT_EQ(bsr_host_test::channelRequestRanks[0], expected);
    }

    ResetObservations();

    // 2) 越界对端要在建链前拦住，且不留半截通道申请
    {
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(), 4, HCCL_DATA_TYPE_FP32, 4, HcclSendRecvType::HCCL_SEND));

        OpParam param = MakeParam(items);
        TopoInfoWithNetLayerDetails topoInfo;
        topoInfo.userRank = 0;
        topoInfo.userRankSize = 4;
        AivTempBatchSendRecvMesh1D tmpl(param, 0, subCommRanks);
        AlgResourceRequest request;
        EXPECT_EQ(tmpl.CalcRes(nullptr, param, &topoInfo, request), HcclResult::HCCL_E_PARA);
        EXPECT_EQ(bsr_host_test::meshChannelRequests + bsr_host_test::meshClosChannelRequests, 0U);
    }

    ResetObservations();

    // 3) level0 是 CLOS 且非 PCIe 混插时走 CLOS；PCIe 混插时 CLOS 多 jetty 不可用，退回 Mesh1D
    {
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(), 4, HCCL_DATA_TYPE_FP32, 1, HcclSendRecvType::HCCL_SEND));
        OpParam param = MakeParam(items);
        TopoInfoWithNetLayerDetails topoInfo;
        topoInfo.userRank = 0;
        topoInfo.userRankSize = 2;
        const std::vector<std::vector<u32>> ranks2 = {AllRanks(2)};

        topoInfo.level0Topo = Level0Shape::MESH_1D_CLOS;
        topoInfo.level0PcieMix = false;
        AivTempBatchSendRecvMesh1D tmpl(param, 0, ranks2);
        AlgResourceRequest request;
        ASSERT_EQ(tmpl.CalcRes(nullptr, param, &topoInfo, request), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(bsr_host_test::meshClosChannelRequests, 1U);
        EXPECT_EQ(bsr_host_test::meshChannelRequests, 0U);

        topoInfo.level0PcieMix = true;
        AivTempBatchSendRecvMesh1D tmplMix(param, 0, ranks2);
        AlgResourceRequest requestMix;
        ASSERT_EQ(tmplMix.CalcRes(nullptr, param, &topoInfo, requestMix), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(bsr_host_test::meshClosChannelRequests, 1U);
        EXPECT_EQ(bsr_host_test::meshChannelRequests, 1U);
    }

    ResetObservations();

    // 4) CLOS 分支只保留 UB_MEM 通道：非 UB 链路 AIV 引擎用不了
    {
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(), 4, HCCL_DATA_TYPE_FP32, 1, HcclSendRecvType::HCCL_SEND));
        OpParam param = MakeParam(items);
        TopoInfoWithNetLayerDetails topoInfo;
        topoInfo.userRank = 0;
        topoInfo.userRankSize = 2;
        topoInfo.level0Topo = Level0Shape::MESH_1D_CLOS;
        topoInfo.level0PcieMix = false;
        bsr_host_test::closProtocolsReturned = {COMM_PROTOCOL_UB_MEM, COMM_PROTOCOL_OTHER, COMM_PROTOCOL_UB_MEM};

        const std::vector<std::vector<u32>> ranks2 = {AllRanks(2)};
        AivTempBatchSendRecvMesh1D tmpl(param, 0, ranks2);
        AlgResourceRequest request;
        ASSERT_EQ(tmpl.CalcRes(nullptr, param, &topoInfo, request), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(request.channels.size(), 1U);
        ASSERT_EQ(request.channels[0].size(), 2U);
        for (const auto& channel : request.channels[0]) {
            EXPECT_EQ(channel.channelProtocol, COMM_PROTOCOL_UB_MEM);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// executor：切片与分轮
// ---------------------------------------------------------------------------------------------

TEST_F(BsrHostTest, SendAndRecvSidesSliceIdenticallyForTheSamePair)
{
    // 同一 pair 两侧各自跑一次 executor：切出来的片必须一一对应，否则两边的 flag 格号对不上，
    // WaitFlag 会一直等一个永远不会被写的格。
    const u32 rankSize = 2;
    const u64 cclSize = 4096;
    BufferPool pool;

    std::vector<HcclSendRecvItem> rank0Items;
    rank0Items.push_back(MakeItem(pool.Add(512), 512, HCCL_DATA_TYPE_INT8, 1, HcclSendRecvType::HCCL_SEND));
    rank0Items.push_back(MakeItem(pool.Add(300), 300, HCCL_DATA_TYPE_INT8, 1, HcclSendRecvType::HCCL_RECV));

    std::vector<HcclSendRecvItem> rank1Items;
    rank1Items.push_back(MakeItem(pool.Add(300), 300, HCCL_DATA_TYPE_INT8, 0, HcclSendRecvType::HCCL_SEND));
    rank1Items.push_back(MakeItem(pool.Add(512), 512, HCCL_DATA_TYPE_INT8, 0, HcclSendRecvType::HCCL_RECV));

    {
        OpParam param = MakeParam(rank0Items);
        auto ctx = MakeResCtx(0, rankSize, cclSize);
        Executor exec;
        ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);
    }
    const size_t rank0Rounds = bsr_host_test::launches.size();
    std::vector<std::vector<std::tuple<u32, u32, u32, u32>>> rank0Layout;
    for (size_t r = 0; r < rank0Rounds; ++r) {
        rank0Layout.push_back(SliceKeys(LaunchAt(r)));
    }

    bsr_host_test::launches.clear();
    {
        OpParam param = MakeParam(rank1Items);
        auto ctx = MakeResCtx(1, rankSize, cclSize);
        Executor exec;
        ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);
    }
    ASSERT_EQ(bsr_host_test::launches.size(), rank0Rounds);

    // 两侧每轮的 SEND 片集合与对侧的 RECV 片集合必须等价（type 0<->1 互换）。
    // remoteRank 也是"对侧视角"：rank0 看到的对端是 1，rank1 看到的是 0，镜像时换成对侧 myRank。
    const u32 mirroredPeer = 0;
    for (size_t r = 0; r < rank0Rounds; ++r) {
        auto keys1 = SliceKeys(LaunchAt(r));
        ASSERT_EQ(keys1.size(), rank0Layout[r].size()) << "round " << r;
        std::vector<std::tuple<u32, u32, u32, u32>> mirrored;
        for (const auto& key : rank0Layout[r]) {
            const u32 type = std::get<0>(key);
            const u32 peerType = (type == SEND_TYPE) ? RECV_TYPE : ((type == RECV_TYPE) ? SEND_TYPE : type);
            mirrored.emplace_back(peerType, std::get<1>(key), std::get<2>(key), mirroredPeer);
        }
        std::sort(mirrored.begin(), mirrored.end());
        std::sort(keys1.begin(), keys1.end());
        EXPECT_EQ(keys1, mirrored) << "round " << r;
    }
}

TEST_F(BsrHostTest, SlicesAndRoundsFollowBucketCapacityAndBucketLayout)
{
    // 1) 长任务按桶容量切轮：100KB / 32KB = 3 满轮 + 1 尾轮，每轮片号从 0 重来、桶内偏移回桶首
    {
        const u32 rankSize = 2;
        const u64 bucketCap = 32 * 1024;
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(
            MakeItem(pool.Add(100 * 1024), 100 * 1024, HCCL_DATA_TYPE_INT8, 1, HcclSendRecvType::HCCL_SEND));

        OpParam param = MakeParam(items);
        auto ctx = MakeResCtx(0, rankSize, 64 * 1024); // bucketCap = 32KB
        Executor exec;
        ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);

        ASSERT_EQ(bsr_host_test::launches.size(), 4U);
        const std::vector<u32> expectedBytes = {32768, 32768, 32768, 4096};
        for (size_t r = 0; r < bsr_host_test::launches.size(); ++r) {
            const auto& record = LaunchAt(r);
            ASSERT_EQ(record.extra.itemNum, 1U) << "round " << r;
            EXPECT_EQ(record.extra.items[0].byteCount, expectedBytes[r]) << "round " << r;
            EXPECT_EQ(record.extra.items[0].sliceIdx, 0) << "round " << r;
            // 桶内偏移每轮回到桶首：跨轮不重叠，残留 flag 才不会被同号片误命中
            EXPECT_EQ(record.extra.items[0].peer.cclOffset, bucketCap) << "round " << r;
            EXPECT_EQ(record.args.sliceId, AIV_BSR_TAG_BASE + 1U + r);
        }
    }

    ResetObservations();

    // 2) 多对端：发方向按对端分桶、桶间不重叠，收方向落在本 rank 自己那一桶；同 pair 内按 count 降序、
    //    sliceIdx 组内从 0 递增（两侧排序一致是 sliceIdx 对得上的前提）
    {
        const u32 rankSize = 4;
        const u64 cclSize = 4096; // bucketCap = 1024
        const u64 bucketCap = 1024;
        const u32 myRank = 1;
        BufferPool pool;

        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(256), 256, HCCL_DATA_TYPE_INT8, 3, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(256), 256, HCCL_DATA_TYPE_INT8, 0, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(256), 256, HCCL_DATA_TYPE_INT8, 2, HcclSendRecvType::HCCL_RECV));

        OpParam param = MakeParam(items);
        auto ctx = MakeResCtx(myRank, rankSize, cclSize);
        Executor exec;
        ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(bsr_host_test::launches.size(), 1U);

        const auto& record = LaunchAt(0);
        ASSERT_EQ(record.extra.itemNum, 3U);
        // 下发顺序：发方向按 remoteRank 升序在前，收方向随后
        EXPECT_EQ(record.extra.items[0].peer.remoteRank, 0U);
        EXPECT_EQ(record.extra.items[0].peer.cclOffset, 0U);
        EXPECT_EQ(record.extra.items[1].peer.remoteRank, 3U);
        EXPECT_EQ(record.extra.items[1].peer.cclOffset, 3U * bucketCap);
        // recv 侧读的是对端按本 rank 编号分好的桶
        EXPECT_EQ(record.extra.items[2].sendRecvType, RECV_TYPE);
        EXPECT_EQ(record.extra.items[2].peer.remoteRank, 2U);
        EXPECT_EQ(record.extra.items[2].peer.cclOffset, static_cast<u32>(myRank * bucketCap));

        // 桶区间互不重叠且都在 cclBuffer 内
        std::vector<std::pair<u64, u64>> ranges;
        for (u64 i = 0; i < record.extra.itemNum; ++i) {
            const u64 begin = record.extra.items[i].peer.cclOffset;
            const u64 end = begin + record.extra.items[i].byteCount;
            EXPECT_LE(end, cclSize);
            ranges.emplace_back(begin, end);
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            EXPECT_LE(ranges[i - 1].second, ranges[i].first);
        }
    }

    ResetObservations();

    // 3) 同 pair 内按 count 降序排，sliceIdx 组内从 0 递增；片起点按 32B 对齐
    {
        const u32 rankSize = 8;
        const u64 bucketCap = 512; // 4096 / 8
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(64), 5, HCCL_DATA_TYPE_INT8, 5, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(64), 1, HCCL_DATA_TYPE_INT8, 2, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(64), 3, HCCL_DATA_TYPE_INT8, 2, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(64), 2, HCCL_DATA_TYPE_INT8, 5, HcclSendRecvType::HCCL_SEND));

        OpParam param = MakeParam(items);
        auto ctx = MakeResCtx(3, rankSize, 4096);
        Executor exec;
        ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(bsr_host_test::launches.size(), 1U);

        const auto& record = LaunchAt(0);
        ASSERT_EQ(record.extra.itemNum, 4U);
        const std::vector<u32> peers = {2, 2, 5, 5};
        const std::vector<u32> bytes = {3, 1, 5, 2};
        const std::vector<u32> sliceIdx = {0, 1, 0, 1};
        // 每个 item 长度都不是 32B 倍数，仍是各自一片；下一片落位抬到对齐边界，
        // 所以桶内偏移是 0/32（而不是 0/3、0/5），保证每片起点恒为 32B 倍数
        const std::vector<u32> expectCclOffset = {2 * bucketCap, 2 * bucketCap + 32, 5 * bucketCap, 5 * bucketCap + 32};
        for (u64 i = 0; i < record.extra.itemNum; ++i) {
            EXPECT_EQ(record.extra.items[i].peer.remoteRank, peers[i]) << "item " << i;
            EXPECT_EQ(record.extra.items[i].byteCount, bytes[i]) << "item " << i;
            EXPECT_EQ(record.extra.items[i].sliceIdx, sliceIdx[i]) << "item " << i;
            EXPECT_EQ(record.extra.items[i].peer.cclOffset, expectCclOffset[i]) << "item " << i;
            EXPECT_EQ(record.extra.items[i].peer.cclOffset % BUCKET_ALIGN, 0U) << "item " << i;
        }
    }
}

// 中转区容量侧的三条边界：向下对齐、对齐后不足一个单位就报错、超 UB 上限要 clamp。
TEST_F(BsrHostTest, CclCapacityEdgesAreClampedOrRejected)
{
    const u32 rankSize = 2;

    // 1) 1000/2 = 500 → 向下对齐到 480
    {
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(100), 100, HCCL_DATA_TYPE_INT8, 1, HcclSendRecvType::HCCL_SEND));
        OpParam param = MakeParam(items);
        auto ctx = MakeResCtx(0, rankSize, 1000);
        Executor exec;
        ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(bsr_host_test::launches.size(), 1U);
        EXPECT_EQ(LaunchAt(0).extra.items[0].peer.cclOffset, 480U);
    }

    ResetObservations();

    // 2) 62/2 = 31 → 对齐后为 0，凑不出一个对齐单位
    {
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(8), 8, HCCL_DATA_TYPE_INT8, 1, HcclSendRecvType::HCCL_SEND));
        OpParam param = MakeParam(items);
        auto ctx = MakeResCtx(0, rankSize, 62);
        Executor exec;
        EXPECT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_E_PARA);
        EXPECT_TRUE(bsr_host_test::launches.empty());
    }

    ResetObservations();

    // 3) 512MB 的 cclBuffer 只用前 256MB（UB 单次传输上限），第二桶从 128MB 处开始
    {
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(8), 8, HCCL_DATA_TYPE_INT8, 1, HcclSendRecvType::HCCL_SEND));
        OpParam param = MakeParam(items);
        auto ctx = MakeResCtx(0, rankSize, 512ULL * 1024 * 1024);
        Executor exec;
        ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(bsr_host_test::launches.size(), 1U);
        EXPECT_EQ(LaunchAt(0).extra.items[0].peer.cclOffset, UB_MAX_DATA_SIZE / rankSize);
    }
}

// rankSize 取到准入上限 256：itemCapPerPair = 512/(2*256) 退化成 1，flagBlock 也是 1。
// 此时每轮最多 2*255 片（收发各 255 个对端），仍必须落在 512 项任务表内。
TEST_F(BsrHostTest, MaxRankSizeKeepsOneItemPerPairPerRound)
{
    const u32 rankSize = MAX_RANK_SIZE_V;
    // 每桶一个对齐单位，于是每个 pair 的每条任务独占一轮，把 itemCapPerPair 的极值真正压出来
    const u64 cclSize = static_cast<u64>(rankSize) * BUCKET_ALIGN;
    BufferPool pool;
    std::vector<HcclSendRecvItem> items;
    for (u32 peer = 1; peer < rankSize; ++peer) {
        for (u32 i = 0; i < 2; ++i) {
            items.push_back(
                MakeItem(pool.Add(BUCKET_ALIGN), BUCKET_ALIGN, HCCL_DATA_TYPE_INT8, peer, HcclSendRecvType::HCCL_SEND));
            items.push_back(
                MakeItem(pool.Add(BUCKET_ALIGN), BUCKET_ALIGN, HCCL_DATA_TYPE_INT8, peer, HcclSendRecvType::HCCL_RECV));
        }
    }

    OpParam param = MakeParam(items);
    auto ctx = MakeResCtx(0, rankSize, cclSize);
    Executor exec;
    ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);
    // 每 pair 每方向 2 条 → 2 轮
    ASSERT_EQ(bsr_host_test::launches.size(), 2U);

    for (size_t r = 0; r < bsr_host_test::launches.size(); ++r) {
        const auto& record = LaunchAt(r);
        // 512/(2*256) = 1：每 pair 每轮每方向只能有 1 片，flagBlock 同步为 1（device 侧格数按它分块）
        EXPECT_EQ(record.extra.flagBlock, 1U) << "round " << r;
        EXPECT_EQ(2ULL * rankSize * record.extra.flagBlock, AIV_BSR_ITEM_MAX) << "round " << r;
        // 255 个对端 × 2 方向 = 510 片，恰好压在 512 上限之内的最大值
        EXPECT_EQ(record.extra.itemNum, 2ULL * (rankSize - 1)) << "round " << r;
        EXPECT_LE(record.extra.itemNum, AIV_BSR_ITEM_MAX) << "round " << r;

        std::map<std::pair<u32, u32>, u32> seen;
        for (u64 i = 0; i < record.extra.itemNum; ++i) {
            const auto& item = record.extra.items[i];
            const auto key
                = std::make_pair(static_cast<u32>(item.sendRecvType), static_cast<u32>(item.peer.remoteRank));
            // 同一轮内同一 (方向, 对端) 不重复：flag 格号 = remoteRank*flagBlock + sliceIdx，重复就是同格
            EXPECT_EQ(seen.count(key), 0U) << "round " << r << " item " << i;
            seen[key] = 1;
            EXPECT_EQ(item.sliceIdx, 0U) << "round " << r << " item " << i;
        }
    }
}

// 多对端、每对端多条任务：单轮必须收敛在任务表容量内，且同一轮内 (方向, 对端, 片号) 唯一。
TEST_F(BsrHostTest, EveryRoundStaysWithinTableCapacity)
{
    const u32 rankSize = 16;
    BufferPool pool;
    std::vector<HcclSendRecvItem> items;
    // 每个对端收、发各 40 条小任务：单轮上限 itemCapPerPair = 512/(2*16) = 16，必然分多轮。
    // 收发都铺满才压得住 2*rankSize*flagBlock <= AIV_BSR_ITEM_MAX 这条边界。
    for (u32 peer = 0; peer < rankSize; ++peer) {
        if (peer == 0) {
            continue;
        }
        for (u32 i = 0; i < 40; ++i) {
            items.push_back(MakeItem(pool.Add(32), 32, HCCL_DATA_TYPE_INT8, peer, HcclSendRecvType::HCCL_SEND));
            items.push_back(MakeItem(pool.Add(32), 32, HCCL_DATA_TYPE_INT8, peer, HcclSendRecvType::HCCL_RECV));
        }
    }

    OpParam param = MakeParam(items);
    // 中转区取得足够大（每桶 1KB，每片 32B ⇒ 桶能放 32 片），让"每对端片数上限"而不是桶容量
    // 成为分轮的第一约束，这样 2*rankSize*flagBlock <= 512 才真正被压到。
    auto ctx = MakeResCtx(0, rankSize, 16 * 1024);
    Executor exec;
    ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);
    ASSERT_FALSE(bsr_host_test::launches.empty());

    u64 totalSlices = 0;
    for (const auto& record : bsr_host_test::launches) {
        EXPECT_LE(record.extra.itemNum, AIV_BSR_ITEM_MAX);
        EXPECT_EQ(record.extra.flagBlock, AIV_BSR_ITEM_MAX / (2ULL * rankSize));
        totalSlices += record.extra.itemNum;
        // 同一轮内 (方向, 对端, 片号) 唯一：收发各有独立的 flag 信箱格序，故方向参与去重键
        std::map<std::tuple<u32, u32, u32>, u32> seen;
        for (u64 i = 0; i < record.extra.itemNum; ++i) {
            const auto& item = record.extra.items[i];
            const auto key = std::make_tuple(
                static_cast<u32>(item.sendRecvType), static_cast<u32>(item.peer.remoteRank),
                static_cast<u32>(item.sliceIdx));
            EXPECT_EQ(seen.count(key), 0U);
            seen[key] = 1;
        }
    }
    EXPECT_EQ(totalSlices, items.size());
}

// ---------------------------------------------------------------------------------------------
// executor：自收发
// ---------------------------------------------------------------------------------------------

TEST_F(BsrHostTest, SelfSendCopiesIntoPairedRecvBuffer)
{
    const u32 myRank = 0;
    const u32 rankSize = 2;

    // 1) 正常配对：SELF_SEND 的目的地址取自配对 recv 项，SELF_RECV 不搬数据
    {
        BufferPool pool;
        void* sendBuf = pool.Add(64);
        void* recvBuf = pool.Add(64);
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(sendBuf, 4, HCCL_DATA_TYPE_FP32, myRank, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(recvBuf, 4, HCCL_DATA_TYPE_FP32, myRank, HcclSendRecvType::HCCL_RECV));

        OpParam param = MakeParam(items);
        auto ctx = MakeResCtx(myRank, rankSize, 4096);
        Executor exec;
        ASSERT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(bsr_host_test::launches.size(), 1U);

        const auto& record = LaunchAt(0);
        ASSERT_EQ(record.extra.itemNum, 2U);
        const auto& selfSend = record.extra.items[0];
        const auto& selfRecv = record.extra.items[1];
        EXPECT_EQ(selfSend.sendRecvType, SELF_SEND_TYPE);
        EXPECT_EQ(selfSend.bufAddr, reinterpret_cast<u64>(sendBuf));
        // 目的地址取自配对的 recv 项：host 不再自己拷，这个地址填错就是搬进错地方
        EXPECT_EQ(selfSend.dstAddr, reinterpret_cast<u64>(recvBuf));
        EXPECT_EQ(selfSend.byteCount, 4U * 4U);
        EXPECT_EQ(selfRecv.sendRecvType, SELF_RECV_TYPE);
        EXPECT_EQ(selfRecv.bufAddr, reinterpret_cast<u64>(recvBuf));
        // SELF_RECV 不搬数据，两个视图都不激活
        EXPECT_EQ(selfRecv.dstAddr, 0U);
    }

    ResetObservations();

    // 2) 只有 SELF_SEND 没有配对 recv：必须在 host 侧拦住，不能下发一条找不到目的地的搬运
    {
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(64), 4, HCCL_DATA_TYPE_FP32, myRank, HcclSendRecvType::HCCL_SEND));
        OpParam param = MakeParam(items);
        auto ctx = MakeResCtx(myRank, rankSize, 4096);
        Executor exec;
        EXPECT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_E_PARA);
        EXPECT_TRUE(bsr_host_test::launches.empty());
    }

    // 3) 配对项 count 不一致：两侧 flag 与字节数都会对不上
    {
        BufferPool pool;
        std::vector<HcclSendRecvItem> items;
        items.push_back(MakeItem(pool.Add(64), 4, HCCL_DATA_TYPE_FP32, myRank, HcclSendRecvType::HCCL_SEND));
        items.push_back(MakeItem(pool.Add(64), 8, HCCL_DATA_TYPE_FP32, myRank, HcclSendRecvType::HCCL_RECV));
        OpParam param = MakeParam(items);
        auto ctx = MakeResCtx(myRank, rankSize, 4096);
        Executor exec;
        EXPECT_EQ(exec.Orchestrate(param, ctx), HcclResult::HCCL_E_PARA);
    }
}

// ---------------------------------------------------------------------------------------------
// executor：零任务 rank
// ---------------------------------------------------------------------------------------------

// 某个 rank 完全没有任务（itemNum==0），或任务全为 0 字节时，它不该下发任何一轮：一旦下发，对端不存在
// 对应片，flag 与屏障都会等一个永远不会到来的值。这条是"各 rank 任务数不对称"的 host 侧对应物，
// kernel 侧"无活跃对端就不进屏障区"由 BarrierCoversOnlyActivePeers 覆盖。
TEST_F(BsrHostTest, EmptyTaskListDispatchesNothingAndKeepsTagAtRoundZero)
{
    const u32 rankSize = 4;
    const u32 myRank = 2; // 对端 0/1/3 都有任务，本 rank 一条都没有
    BufferPool pool;
    auto ctx = MakeResCtx(myRank, rankSize, 4096);

    // 1) itemNum==0。注意 BuildTaskDeques 首句是 CHK_PTR_NULL，上层即使无任务也会给非空指针，
    //    所以这里传哨兵地址而不是空 vector 的 data()。
    HcclSendRecvItem sentinel = MakeItem(pool.Add(), 0, HCCL_DATA_TYPE_INT8, 0, HcclSendRecvType::HCCL_SEND);
    OpParam noItem;
    noItem.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    noItem.numBlocksLimit = 48;
    noItem.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    noItem.batchSendRecvDataDes.sendRecvItemsPtr = &sentinel;
    noItem.batchSendRecvDataDes.itemNum = 0;

    Executor exec;
    ASSERT_EQ(exec.Orchestrate(noItem, ctx), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(bsr_host_test::launches.empty());

    // 2) itemNum>0 但每条 count 都是 0：切片器不产生任何片，同样不下发，也不该被当成错误
    HcclSendRecvItem zeroBytes[2];
    zeroBytes[0] = MakeItem(pool.Add(), 0, HCCL_DATA_TYPE_INT8, 0, HcclSendRecvType::HCCL_SEND);
    zeroBytes[1] = MakeItem(pool.Add(), 0, HCCL_DATA_TYPE_INT8, 1, HcclSendRecvType::HCCL_RECV);
    OpParam zeroByteParam;
    zeroByteParam.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    zeroByteParam.numBlocksLimit = 48;
    zeroByteParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    zeroByteParam.batchSendRecvDataDes.sendRecvItemsPtr = zeroBytes;
    zeroByteParam.batchSendRecvDataDes.itemNum = 2;

    ASSERT_EQ(exec.Orchestrate(zeroByteParam, ctx), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(bsr_host_test::launches.empty());

    // 3) 空调用不消耗轮次：同一个实例接着跑一个真任务，tag 仍从第 0 轮（1025）起算
    std::vector<HcclSendRecvItem> items;
    items.push_back(MakeItem(pool.Add(64), 64, HCCL_DATA_TYPE_INT8, 0, HcclSendRecvType::HCCL_SEND));
    OpParam realParam = MakeParam(items);
    ASSERT_EQ(exec.Orchestrate(realParam, ctx), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(bsr_host_test::launches.size(), 1U);
    EXPECT_EQ(LaunchAt(0).args.sliceId, AIV_BSR_TAG_BASE + 1U);
}

// ---------------------------------------------------------------------------------------------
// 注册属性：AIV 路径的准入门槛
// ---------------------------------------------------------------------------------------------

TEST_F(BsrHostTest, RegisteredAttrsGateAivPath)
{
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get("AivBatchSendRecvSoleMesh");
    ASSERT_NE(attrs, nullptr);
    // 与 device 侧 MAX_RANK_SIZE_V 同源（AIV 地址表与 flag 信箱都按 256 设计）；rankSize 超限的回退由选路
    // 框架按这个上限过滤，executor 自身不再重复判断。
    EXPECT_EQ(attrs->topo.maxSupportRankSize, 256U);
    EXPECT_EQ(attrs->topo.maxSupportRankSize, MAX_RANK_SIZE_V);
    EXPECT_EQ(attrs->topo.supportLevel0Topos, LEVEL0_TOPO_ANY);
    EXPECT_EQ(attrs->topo.maxTopoLevelNum, TOPO_LEVEL_NUM_2);
    EXPECT_TRUE(attrs->topo.isSupportLevel0PcieMix);
    // opType 由生产注册表的 ParseAlgName 从算法名解析，桩不做该解析，故此处不校验 opType。

    ASSERT_TRUE(static_cast<bool>(attrs->topo.topoCustomCheck));
    TopoInfoWithNetLayerDetails topoInfo;
    // 纯 UB 组网且 level2 不是 UB RTP：AIV 可用
    EXPECT_TRUE(attrs->topo.topoCustomCheck(&topoInfo));
    // 带 UB RTP 的 level2 或顶层 UBoE：AIV 路径不可用
    topoInfo.level2UbRtp = true;
    EXPECT_FALSE(attrs->topo.topoCustomCheck(&topoInfo));
    topoInfo.level2UbRtp = false;
    topoInfo.topLevelUboe = true;
    EXPECT_FALSE(attrs->topo.topoCustomCheck(&topoInfo));
}

} // namespace
