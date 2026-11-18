/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// InsV2AlltoAllVSerialSoleExecutor 的窗口编排（SerialAlltoAllWindow）UT：
// 覆盖 Prepare 的 window/mesh 路由判定（uniform 校验 / meshSlot 容量比较）、
// RunWindow 的 XOR 配对顺序、RunPeer 按 cap_ 分块。
// 该窗口类位于 executor 源文件的匿名命名空间，本 TU 以 #include 生产源文件的方式纳入，
// 同时屏蔽 REGISTER_EXEC_V2 / REGISTER_ALG_ATTRS，避免实例化 executor/模板注册。

#include "nda_test_access.h"

#include <cstring>

#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs_registry.h"

#undef REGISTER_EXEC_V2
#define REGISTER_EXEC_V2(type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate)
#undef REGISTER_ALG_ATTRS
#define REGISTER_ALG_ATTRS(algoName, ...)

#include "ins_v2_all_to_all_v_serial_sole_executor.cc"

namespace nda_ut {
namespace {

    using Window = SerialAlltoAllWindow<InsTempDpuAlltoAllMeshNda>;

    OpParam MakeExecutorOpParam(
        HcclCMDType opType, HcclDataType dataType, const u64* sendCounts, const u64* recvCounts, const u64* sdispls,
        const u64* rdispls)
    {
        OpParam op{};
        op.opType = opType;
        std::strncpy(op.algTag, "ut", sizeof(op.algTag));
        op.all2AllVDataDes.sendType = dataType;
        op.all2AllVDataDes.recvType = dataType;
        op.all2AllVDataDes.sendCounts = const_cast<u64*>(sendCounts);
        op.all2AllVDataDes.recvCounts = const_cast<u64*>(recvCounts);
        op.all2AllVDataDes.sdispls = const_cast<u64*>(sdispls);
        op.all2AllVDataDes.rdispls = const_cast<u64*>(rdispls);
        return op;
    }

    AlgResourceCtxSerializable MakeExecutorContext(
        u32 rank, u32 size, void* cclAddr, u64 cclSize, const std::vector<ChannelInfo>& channels,
        const std::vector<ThreadHandle>& threads)
    {
        AlgResourceCtxSerializable ctx{};
        ctx.topoInfo.userRank = rank;
        ctx.topoInfo.userRankSize = size;
        ctx.cclMem.addr = cclAddr;
        ctx.cclMem.size = cclSize;
        ctx.channels.push_back(channels);
        ctx.threads = threads;
        return ctx;
    }

    // 为每个对端构造一条 channel，remoteCclMem.addr 用 baseAddr + peer*stride 区分，便于断言配对顺序
    std::vector<ChannelInfo> MakePeerChannels(u32 size, u32 rank, u64 remoteCclSize, u8* baseAddr, u64 stride = 0x100)
    {
        std::vector<ChannelInfo> channels;
        for (u32 peer = 0; peer < size; ++peer) {
            if (peer == rank) {
                continue;
            }
            void* remoteAddr
                = stride == 0 ? static_cast<void*>(baseAddr) : static_cast<void*>(baseAddr + peer * stride);
            channels.push_back(MakeChannel(
                peer, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 100 + peer, remoteAddr,
                remoteCclSize));
        }
        return channels;
    }

    class SerialAlltoAllWindowTest : public ::testing::Test {
    protected:
        void SetUp() override { ResetStubState(); }
        void TearDown() override { ResetStubState(); }
    };

    struct UnusedTopoMatch {
        HcclResult MatchTopo(TopoInfoWithNetLayerDetails*, AlgHierarchyInfoForAllLevel&, const AlgAttrs&)
        {
            return HCCL_E_NOT_SUPPORT;
        }
    };

    class MeshRecordingTemplate : public InsTempDpuAlltoAllMeshNda {
    public:
        using InsTempDpuAlltoAllMeshNda::InsTempDpuAlltoAllMeshNda;
        inline static std::vector<TemplateDataParams> rounds;

        HcclResult KernelRun(const OpParam&, const TemplateDataParams& data, TemplateResource&) override
        {
            rounds.push_back(data);
            return HCCL_SUCCESS;
        }
    };

    class SerialMeshCapacityTest : public ::testing::Test {
    protected:
        void SetUp() override
        {
            ResetStubState();
            MeshRecordingTemplate::rounds.clear();
        }

        HcclResult RunMesh(u64 count, bool symmetric, u32 rankSize = 4)
        {
            std::vector<u64> counts(rankSize, count);
            std::vector<u64> displs(rankSize);
            std::vector<u64> peerDispls(rankSize, 0);
            std::vector<u32> ranks(rankSize);
            for (u32 peer = 0; peer < rankSize; ++peer) {
                displs[peer] = peer * count;
                ranks[peer] = peer;
            }
            OpParam param = MakeExecutorOpParam(
                HcclCMDType::HCCL_CMD_ALLTOALLVC, HCCL_DATA_TYPE_FP32, counts.data(), counts.data(), displs.data(),
                displs.data());
            param.supportSymmetricMemory = symmetric;
            param.varMemSize = ALL_TO_ALL_V_VECTOR_NUM * counts.size() * sizeof(u64);
            param.all2AllVDataDes.peerRdispls = peerDispls.data();
            std::vector<u8> memory(1024);
            const auto channels = MakePeerChannels(rankSize, 0, memory.size(), memory.data(), 0);
            auto context = MakeExecutorContext(0, rankSize, memory.data(), memory.size(), channels, {1});
            context.algHierarchyInfo.infos = {{ranks}};
            InsV2AlltoAllVSerialSoleExecutor<UnusedTopoMatch, MeshRecordingTemplate> executor;
            return executor.RunMesh(param, context);
        }
    };

    TEST_F(SerialMeshCapacityTest, SymmetricOversizeStopsBeforeKernel)
    {
        EXPECT_EQ(RunMesh(33, true), HCCL_E_PARA);
        EXPECT_TRUE(MeshRecordingTemplate::rounds.empty());
        EXPECT_TRUE(LocalCopyCalls().empty());
        EXPECT_TRUE(WriteCalls().empty());
        EXPECT_TRUE(HandshakeCalls().empty());
        EXPECT_TRUE(BatchCalls().empty());
    }

    TEST_F(SerialMeshCapacityTest, SymmetricBoundaryKeepsSingleRoundWithCclSemantics)
    {
        ASSERT_EQ(RunMesh(32, true), HCCL_SUCCESS);
        ASSERT_EQ(MeshRecordingTemplate::rounds.size(), 1U);
        const auto& data = MeshRecordingTemplate::rounds[0];
        EXPECT_EQ(data.count, 32U);
        EXPECT_EQ(data.inputSliceStride, 128U);
        EXPECT_FALSE(data.enableRemoteMemAccess);
    }

    TEST_F(SerialMeshCapacityTest, SymmetricSmallKeepsExistingCompactStride)
    {
        ASSERT_EQ(RunMesh(7, true), HCCL_SUCCESS);
        ASSERT_EQ(MeshRecordingTemplate::rounds.size(), 1U);
        EXPECT_EQ(MeshRecordingTemplate::rounds[0].inputSliceStride, 28U);
        EXPECT_EQ(MeshRecordingTemplate::rounds[0].count, 7U);
        EXPECT_FALSE(MeshRecordingTemplate::rounds[0].enableRemoteMemAccess);
    }

    TEST_F(SerialMeshCapacityTest, OrdinaryMeshKeepsTwoRoundsAndTailOffsets)
    {
        ASSERT_EQ(RunMesh(33, false), HCCL_SUCCESS);
        ASSERT_EQ(MeshRecordingTemplate::rounds.size(), 2U);
        const auto& first = MeshRecordingTemplate::rounds[0];
        const auto& tail = MeshRecordingTemplate::rounds[1];
        EXPECT_EQ(first.count, 32U);
        EXPECT_EQ(tail.count, 1U);
        EXPECT_EQ(first.inputSliceStride, 128U);
        EXPECT_EQ(tail.inputSliceStride, 128U);
        EXPECT_EQ(tail.sendCounts, std::vector<u64>(4, 1));
        EXPECT_EQ(tail.sdispls, (std::vector<u64>{32, 65, 98, 131}));
        EXPECT_FALSE(tail.enableRemoteMemAccess);
    }

    TEST_F(SerialMeshCapacityTest, SingleRankSelfCopyDoesNotRequirePeerCclSlot)
    {
        ASSERT_EQ(RunMesh(129, true, 1), HCCL_SUCCESS);
        ASSERT_EQ(MeshRecordingTemplate::rounds.size(), 1U);
        EXPECT_EQ(MeshRecordingTemplate::rounds[0].count, 129U);
    }

    TEST_F(SerialAlltoAllWindowTest, PrepareDegenerateSingleRankFallsBackToMesh)
    {
        u64 sendCounts[1] = {0};
        u64 recvCounts[1] = {0};
        u64 sdispls[1] = {0};
        u64 rdispls[1] = {0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALLV, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);
        AlgResourceCtxSerializable ctx = MakeExecutorContext(0, 1, reinterpret_cast<void*>(0x1000), 8192, {}, {1});
        Window window(op, ctx);
        bool handled = true;
        EXPECT_EQ(window.Prepare(handled), HCCL_SUCCESS);
        EXPECT_FALSE(handled);
    }

    TEST_F(SerialAlltoAllWindowTest, PrepareMissingChannelsFallsBackToMesh)
    {
        u64 sendCounts[2] = {0, 0};
        u64 recvCounts[2] = {0, 0};
        u64 sdispls[2] = {0, 0};
        u64 rdispls[2] = {0, 0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALLV, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);
        AlgResourceCtxSerializable ctx = MakeExecutorContext(0, 2, reinterpret_cast<void*>(0x1000), 8192, {}, {1});
        ctx.channels.clear(); // 外层 channels 为空（size 0）→ 不满足 channels.size()==1
        Window window(op, ctx);
        bool handled = true;
        EXPECT_EQ(window.Prepare(handled), HCCL_SUCCESS);
        EXPECT_FALSE(handled);
    }

    TEST_F(SerialAlltoAllWindowTest, PrepareEmptyThreadsFallsBackToMesh)
    {
        u64 sendCounts[2] = {0, 1};
        u64 recvCounts[2] = {0, 1};
        u64 sdispls[2] = {0, 0};
        u64 rdispls[2] = {0, 0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALLV, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);
        u8 base[4] = {};
        std::vector<ChannelInfo> channels = MakePeerChannels(2, 0, 8192, base);
        AlgResourceCtxSerializable ctx = MakeExecutorContext(0, 2, base, 8192, channels, {});
        Window window(op, ctx);
        bool handled = true;
        EXPECT_EQ(window.Prepare(handled), HCCL_SUCCESS);
        EXPECT_FALSE(handled);
    }

    TEST_F(SerialAlltoAllWindowTest, PrepareMissingPeerChannelFailsBeforeCommunication)
    {
        const u32 rank = 0;
        const u32 size = 4;
        u64 sendCounts[4] = {0, 1, 1, 1};
        u64 recvCounts[4] = {0, 1, 1, 1};
        u64 sdispls[4] = {0, 0, 0, 0};
        u64 rdispls[4] = {0, 0, 0, 0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALLV, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);

        u8 base[4] = {};
        std::vector<ChannelInfo> channels;
        channels.push_back(
            MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 11, base, 1024));
        AlgResourceCtxSerializable ctx = MakeExecutorContext(rank, size, base, 1024, channels, {1});
        Window window(op, ctx);
        bool handled = true;
        EXPECT_EQ(window.Prepare(handled), HCCL_E_PARA);
        EXPECT_FALSE(handled);
        EXPECT_TRUE(LocalCopyCalls().empty());
        EXPECT_TRUE(WriteCalls().empty());
        EXPECT_TRUE(HandshakeCalls().empty());
        EXPECT_TRUE(BatchCalls().empty());
    }

    TEST_F(SerialAlltoAllWindowTest, PrepareAlltoallUniformSmallRoutesMesh)
    {
        const u64 fixedCount = 128;
        u64 sendCounts[2] = {fixedCount, fixedCount};
        u64 recvCounts[2] = {fixedCount, fixedCount};
        u64 sdispls[2] = {0, 0};
        u64 rdispls[2] = {0, 0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALL, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);
        u8 base[4] = {};
        std::vector<ChannelInfo> channels = MakePeerChannels(2, 0, 8192, base);
        AlgResourceCtxSerializable ctx = MakeExecutorContext(0, 2, base, 8192, channels, {1});
        Window window(op, ctx);
        bool handled = true;
        EXPECT_EQ(window.Prepare(handled), HCCL_SUCCESS);
        EXPECT_FALSE(handled); // uniform 且固定 count 不超过 meshSlot → mesh 路径
    }

    TEST_F(SerialAlltoAllWindowTest, PrepareAlltoallNonUniformRoutesWindow)
    {
        const u64 fixedCount = 128;
        u64 sendCounts[2] = {fixedCount, fixedCount};
        u64 recvCounts[2] = {fixedCount, fixedCount};
        u64 sdispls[2] = {0, 0};
        u64 rdispls[2] = {0, 0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALL, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);
        u8 base[4] = {};
        // 对端 remoteCclMem.size 与本端 cclMem.size 不一致 → non-uniform → 窗口路径
        std::vector<ChannelInfo> channels = MakePeerChannels(2, 0, 4096, base);
        AlgResourceCtxSerializable ctx = MakeExecutorContext(0, 2, base, 8192, channels, {1});
        Window window(op, ctx);
        bool handled = false;
        EXPECT_EQ(window.Prepare(handled), HCCL_SUCCESS);
        EXPECT_TRUE(handled);
    }

    TEST_F(SerialAlltoAllWindowTest, PrepareAlltoallvRoutesWindow)
    {
        u64 sendCounts[2] = {0, 300};
        u64 recvCounts[2] = {0, 300};
        u64 sdispls[2] = {0, 0};
        u64 rdispls[2] = {0, 0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALLV, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);
        u8 base[4] = {};
        std::vector<ChannelInfo> channels = MakePeerChannels(2, 0, 8192, base);
        AlgResourceCtxSerializable ctx = MakeExecutorContext(0, 2, base, 8192, channels, {1});
        Window window(op, ctx);
        bool handled = false;
        EXPECT_EQ(window.Prepare(handled), HCCL_SUCCESS);
        EXPECT_TRUE(handled);
    }

    TEST_F(SerialAlltoAllWindowTest, RunWindowVisitsPeersInXorOrder)
    {
        const u32 rank = 1;
        const u32 size = 4;
        u64 sendCounts[4] = {1, 0, 1, 1};
        u64 recvCounts[4] = {1, 0, 1, 1};
        u64 sdispls[4] = {0, 0, 0, 0};
        u64 rdispls[4] = {0, 0, 0, 0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALLV, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);

        u8 base[4] = {};
        u8* peerBase = reinterpret_cast<u8*>(0x20000);
        std::vector<ChannelInfo> channels = MakePeerChannels(size, rank, 8192, peerBase);
        AlgResourceCtxSerializable ctx = MakeExecutorContext(rank, size, base, 8192, channels, {1});
        Window window(op, ctx);
        bool handled = false;
        ASSERT_EQ(window.Prepare(handled), HCCL_SUCCESS);
        ASSERT_TRUE(handled);

        EXPECT_EQ(window.RunWindow(), HCCL_SUCCESS);

        // rank=1, size=4 → limit=4，step 1..3 → peer = 1^step：0、3、2
        ASSERT_EQ(WriteCalls().size(), 3u);
        const u32 expectedPeers[3] = {0, 3, 2};
        for (u32 i = 0; i < 3; ++i) {
            ASSERT_EQ(WriteCalls()[i].txDst.size(), 1u);
            const void* expectedAddr = static_cast<void*>(peerBase + expectedPeers[i] * 0x100);
            EXPECT_EQ(WriteCalls()[i].txDst[0].addr_, expectedAddr);
        }
    }

    TEST_F(SerialAlltoAllWindowTest, RunPeerChunksLargeTransferByCap)
    {
        const u32 rank = 0;
        const u32 size = 2;
        const u64 total = 300; // cap_ = 512 / 4 = 128 → 3 个 chunk（128/128/44）
        u64 sendCounts[2] = {0, total};
        u64 recvCounts[2] = {0, total};
        u64 sdispls[2] = {0, 0};
        u64 rdispls[2] = {0, 0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALLV, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);

        u8 base[4] = {};
        std::vector<ChannelInfo> channels = MakePeerChannels(size, rank, 1024, base, 0);
        AlgResourceCtxSerializable ctx = MakeExecutorContext(rank, size, base, 1024, channels, {1});
        Window window(op, ctx);
        bool handled = false;
        ASSERT_EQ(window.Prepare(handled), HCCL_SUCCESS);
        ASSERT_TRUE(handled);

        EXPECT_EQ(window.RunPeer(1), HCCL_SUCCESS);

        // 300 / cap(128) = 3 次窗口 chunk，每次一个 SendRecvWriteNda
        ASSERT_EQ(WriteCalls().size(), 3u);
        for (u32 i = 0; i < 3; ++i) {
            ASSERT_EQ(WriteCalls()[i].txSrc.size(), 1u);
        }
        EXPECT_EQ(WriteCalls()[0].txSrc[0].count_, 128u);
        EXPECT_EQ(WriteCalls()[1].txSrc[0].count_, 128u);
        EXPECT_EQ(WriteCalls()[2].txSrc[0].count_, 44u);
    }

    TEST_F(SerialAlltoAllWindowTest, RunSelfTakesMinOfSendAndRecvCounts)
    {
        const u32 rank = 0;
        const u32 size = 2;
        // 本端 sendCounts(5) > recvCounts(3)：取小后只拷 3 个元素，避免越过 output 的 recv 槽位
        u64 sendCounts[2] = {5, 0};
        u64 recvCounts[2] = {3, 0};
        u64 sdispls[2] = {0, 0};
        u64 rdispls[2] = {0, 0};
        OpParam op = MakeExecutorOpParam(
            HcclCMDType::HCCL_CMD_ALLTOALLV, HCCL_DATA_TYPE_FP32, sendCounts, recvCounts, sdispls, rdispls);

        u8 base[4] = {};
        std::vector<ChannelInfo> channels = MakePeerChannels(size, rank, 1024, base, 0);
        AlgResourceCtxSerializable ctx = MakeExecutorContext(rank, size, base, 1024, channels, {1});
        Window window(op, ctx);
        bool handled = false;
        ASSERT_EQ(window.Prepare(handled), HCCL_SUCCESS);
        ASSERT_TRUE(handled);

        EXPECT_EQ(window.RunSelf(), HCCL_SUCCESS);

        ASSERT_EQ(LocalCopyCalls().size(), 1u);
        EXPECT_EQ(LocalCopyCalls()[0].src.count_, 3u);
        EXPECT_EQ(LocalCopyCalls()[0].dst.count_, 3u);
    }

} // namespace
} // namespace nda_ut
