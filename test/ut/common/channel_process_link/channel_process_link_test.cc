/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ProcessLinksForChannel空channel源头拦截UT（Issue #838）：
// 与对端遍历所有netLayer均未建链时，在函数内报错返回HCCL_E_INTERNAL，
// 而非静默SUCCESS导致空channel列表沿调用链传递、错误延迟到kernel执行期。

#include <gtest/gtest.h>

#include "channel.h"
#include "stub/hcomm_stub.h"

using ops_hccl::ProcessLinksForChannel; // channel.cc中为namespace内非static导出，此处直接引用

namespace {
constexpr u32 MY_RANK = 0;
constexpr u32 REMOTE_RANK = 1;
constexpr u32 SINGLE_LAYER = 1;
constexpr u32 MULTI_LAYER = 2;

CommLink MakeLink()
{
    CommLink link = {}; // 全零endpoint与stub返回的topoInst endpoint相等，GetTopoTypeByLink正常返回
    link.linkAttr.linkProtocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
    return link;
}
} // namespace

class ProcessLinksForChannelTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        g_rankGraph = MockRankGraph{};
        g_rankGraph.netLayers = {0};
    }
};

// 单层拓扑下与对端无任何link：源头报错而非静默SUCCESS（Issue #838核心场景）
TEST_F(ProcessLinksForChannelTest, SingleLayerNoLinkReturnsError)
{
    std::vector<HcclChannelDesc> channels;
    ASSERT_EQ(
        ProcessLinksForChannel(nullptr, MY_RANK, REMOTE_RANK, channels, CommTopo::COMM_TOPO_1DMESH, SINGLE_LAYER),
        HCCL_E_INTERNAL);
    EXPECT_TRUE(channels.empty());
}

// 多层拓扑下所有层均无link：逐层continue耗尽后同样报错
TEST_F(ProcessLinksForChannelTest, MultiLayerAllNoLinkReturnsError)
{
    g_rankGraph.netLayers = {0, 1};
    std::vector<HcclChannelDesc> channels;
    ASSERT_EQ(
        ProcessLinksForChannel(nullptr, MY_RANK, REMOTE_RANK, channels, CommTopo::COMM_TOPO_1DMESH, MULTI_LAYER),
        HCCL_E_INTERNAL);
    EXPECT_TRUE(channels.empty());
}

// 单层拓扑下存在link：正常建链，行为不变
TEST_F(ProcessLinksForChannelTest, SingleLayerLinkCreatesChannel)
{
    g_rankGraph.links[{0, REMOTE_RANK}] = {MakeLink()};
    std::vector<HcclChannelDesc> channels;
    ASSERT_EQ(
        ProcessLinksForChannel(nullptr, MY_RANK, REMOTE_RANK, channels, CommTopo::COMM_TOPO_1DMESH, SINGLE_LAYER),
        HCCL_SUCCESS);
    ASSERT_EQ(channels.size(), 1u);
    EXPECT_EQ(channels[0].remoteRank, REMOTE_RANK);
    EXPECT_EQ(channels[0].channelProtocol, CommProtocol::COMM_PROTOCOL_UB_CTP);
}

// 多层拓扑下层0无link、层1有link：continue到高层建链成功，多层查找语义保留、不误伤
TEST_F(ProcessLinksForChannelTest, MultiLayerFallbackToHigherLayerSucceeds)
{
    g_rankGraph.netLayers = {0, 1};
    g_rankGraph.links[{1, REMOTE_RANK}] = {MakeLink()};
    std::vector<HcclChannelDesc> channels;
    ASSERT_EQ(
        ProcessLinksForChannel(nullptr, MY_RANK, REMOTE_RANK, channels, CommTopo::COMM_TOPO_1DMESH, MULTI_LAYER),
        HCCL_SUCCESS);
    ASSERT_EQ(channels.size(), 1u);
    EXPECT_EQ(channels[0].remoteRank, REMOTE_RANK);
}

// 防御性校验：非法dataType在入口被拦截，不进入DATATYPE_SIZE_TABLE越界查表
TEST_F(ProcessLinksForChannelTest, CalcChannelRequestNhrMultiJettyUbxRejectsInvalidDataType)
{
    ops_hccl::OpParam param{};
    param.DataDes.dataType = HCCL_DATA_TYPE_RESERVED;
    ops_hccl::TopoInfoWithNetLayerDetails topoInfo{};
    std::vector<std::vector<u32>> subcommInfo;
    std::vector<HcclChannelDesc> channels;
    // 白名单校验先于解引用comm/topoInfo，nullptr不影响本用例
    EXPECT_EQ(
        ops_hccl::CalcChannelRequestNhrMultiJettyUbx(nullptr, param, &topoInfo, subcommInfo, channels, 0), HCCL_E_PARA);
    EXPECT_TRUE(channels.empty());
}

// channels已含前序对端条目时：判定基准为增量（channelCountBefore）而非绝对非空，
// 第二个对端无link仍报错且前序条目保持不变，锁定delta判定语义不误报
TEST_F(ProcessLinksForChannelTest, PreexistingEntriesNotMisreportedWhenNextPeerNoLink)
{
    constexpr u32 PREV_RANK = 1;
    constexpr u32 NEXT_RANK = 2;
    g_rankGraph.links[{0, PREV_RANK}] = {MakeLink()};
    std::vector<HcclChannelDesc> channels;
    ASSERT_EQ(
        ProcessLinksForChannel(nullptr, MY_RANK, PREV_RANK, channels, CommTopo::COMM_TOPO_1DMESH, SINGLE_LAYER),
        HCCL_SUCCESS);
    ASSERT_EQ(channels.size(), 1u); // 前序对端已建链1条

    // 第二个对端无link：应报错（若误用"channels非空"判定则会静默放行）
    ASSERT_EQ(
        ProcessLinksForChannel(nullptr, MY_RANK, NEXT_RANK, channels, CommTopo::COMM_TOPO_1DMESH, SINGLE_LAYER),
        HCCL_E_INTERNAL);
    ASSERT_EQ(channels.size(), 1u); // 前序条目不被破坏
    EXPECT_EQ(channels[0].remoteRank, PREV_RANK);
}
