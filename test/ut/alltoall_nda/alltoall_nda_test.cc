/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nda_test_access.h"

#include <cstring>

namespace nda_ut {
namespace {

    class InsTempDpuAlltoAllMeshNdaTest : public ::testing::Test {
    protected:
        void SetUp() override { ResetStubState(); }
        void TearDown() override { ResetStubState(); }
    };

    // ---------------------------------------------------------------------------
    // RunWindowChunk：串行窗口 chunk 的 NDA 中转（input→sendBuffer）与收尾（recvBuffer→output）
    // ---------------------------------------------------------------------------
    TEST_F(InsTempDpuAlltoAllMeshNdaTest, RunWindowChunkNdaSendRecvCopiesAndUsesNdaWrite)
    {
        uint8_t input[64] = {};
        uint8_t output[64] = {};
        uint8_t sendBuffer[64] = {};
        uint8_t recvBuffer[64] = {};
        uint8_t remote[64] = {};
        ChannelInfo channel
            = MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 7, remote, 64);

        DataSlice in(input, 0, 100, 25);
        DataSlice out(output, 0, 100, 25);
        DataSlice send(sendBuffer, 0, 100, 25);
        DataSlice recv(recvBuffer, 200, 100, 25);

        EXPECT_EQ(
            InsTempDpuAlltoAllMeshNda::RunWindowChunk(
                channel, 3, HCCL_DATA_TYPE_FP32, RunWindowChunkSlices{in, out, send, recv}),
            HCCL_SUCCESS);

        ASSERT_EQ(LocalCopyCalls().size(), 2u);
        EXPECT_EQ(LocalCopyCalls()[0].src.addr_, in.addr_);
        EXPECT_EQ(LocalCopyCalls()[0].dst.addr_, send.addr_);
        EXPECT_EQ(LocalCopyCalls()[1].src.addr_, recv.addr_);
        EXPECT_EQ(LocalCopyCalls()[1].src.offset_, 200u);
        EXPECT_EQ(LocalCopyCalls()[1].dst.addr_, out.addr_);

        ASSERT_EQ(WriteCalls().size(), 1u);
        EXPECT_EQ(WriteCalls()[0].kind, WriteKind::kSendRecvWriteNda);
        ASSERT_EQ(WriteCalls()[0].txSrc.size(), 1u);
        EXPECT_EQ(WriteCalls()[0].txSrc[0].addr_, send.addr_); // NDA 走 CCL_IN 中转
        ASSERT_EQ(WriteCalls()[0].txDst.size(), 1u);
        EXPECT_EQ(WriteCalls()[0].txDst[0].addr_, remote);
        EXPECT_EQ(WriteCalls()[0].txDst[0].offset_, 200u); // 落点复用 recvBuffer 的偏移
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, RunWindowChunkNdaSendOnlyUsesSendWriteForNda)
    {
        uint8_t input[64] = {};
        uint8_t sendBuffer[64] = {};
        uint8_t remote[64] = {};
        ChannelInfo channel
            = MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 7, remote, 64);

        DataSlice in(input, 0, 80, 20);
        DataSlice out(nullptr, 0, 0, 0);
        DataSlice send(sendBuffer, 0, 80, 20);
        DataSlice recv(nullptr, 0, 0, 0);

        EXPECT_EQ(
            InsTempDpuAlltoAllMeshNda::RunWindowChunk(
                channel, 3, HCCL_DATA_TYPE_FP32, RunWindowChunkSlices{in, out, send, recv}),
            HCCL_SUCCESS);

        ASSERT_EQ(LocalCopyCalls().size(), 1u); // 仅输入中转，无收尾拷贝
        ASSERT_EQ(WriteCalls().size(), 1u);
        EXPECT_EQ(WriteCalls()[0].kind, WriteKind::kSendWriteNda);
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, RunWindowChunkNdaRecvOnlyUsesRecvWriteForNda)
    {
        uint8_t output[64] = {};
        uint8_t recvBuffer[64] = {};
        uint8_t remote[64] = {};
        ChannelInfo channel
            = MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 7, remote, 64);

        DataSlice in(nullptr, 0, 0, 0);
        DataSlice out(output, 0, 60, 15);
        DataSlice send(nullptr, 0, 0, 0);
        DataSlice recv(recvBuffer, 0, 60, 15);

        EXPECT_EQ(
            InsTempDpuAlltoAllMeshNda::RunWindowChunk(
                channel, 3, HCCL_DATA_TYPE_FP32, RunWindowChunkSlices{in, out, send, recv}),
            HCCL_SUCCESS);

        ASSERT_EQ(LocalCopyCalls().size(), 1u); // 仅收尾拷贝
        ASSERT_EQ(WriteCalls().size(), 1u);
        EXPECT_EQ(WriteCalls()[0].kind, WriteKind::kRecvWriteNda);
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, RunWindowChunkNonNdaSkipsStagingAndUsesPlainWrite)
    {
        uint8_t input[64] = {};
        uint8_t output[64] = {};
        uint8_t sendBuffer[64] = {};
        uint8_t recvBuffer[64] = {};
        uint8_t remote[64] = {};
        ChannelInfo channel = MakeChannel(
            1, CommProtocol::COMM_PROTOCOL_PCIE, EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE, 7, remote, 64);

        DataSlice in(input, 0, 100, 25);
        DataSlice out(output, 0, 100, 25);
        DataSlice send(sendBuffer, 0, 100, 25);
        DataSlice recv(recvBuffer, 200, 100, 25);

        EXPECT_EQ(
            InsTempDpuAlltoAllMeshNda::RunWindowChunk(
                channel, 3, HCCL_DATA_TYPE_FP32, RunWindowChunkSlices{in, out, send, recv}),
            HCCL_SUCCESS);

        ASSERT_EQ(LocalCopyCalls().size(), 1u); // 非 NDA 无输入中转
        ASSERT_EQ(WriteCalls().size(), 1u);
        EXPECT_EQ(WriteCalls()[0].kind, WriteKind::kSendRecvWrite);
        ASSERT_EQ(WriteCalls()[0].txSrc.size(), 1u);
        EXPECT_EQ(WriteCalls()[0].txSrc[0].addr_, in.addr_); // 非 NDA 直接从输入内存写对端
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, RunWindowChunkNonNdaSendOnlyUsesPlainSendWrite)
    {
        uint8_t input[64] = {};
        uint8_t remote[64] = {};
        ChannelInfo channel = MakeChannel(
            1, CommProtocol::COMM_PROTOCOL_PCIE, EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE, 7, remote, 64);

        DataSlice in(input, 0, 80, 20);
        DataSlice out(nullptr, 0, 0, 0);
        DataSlice send(nullptr, 0, 0, 0);
        DataSlice recv(nullptr, 0, 0, 0);

        EXPECT_EQ(
            InsTempDpuAlltoAllMeshNda::RunWindowChunk(
                channel, 3, HCCL_DATA_TYPE_FP32, RunWindowChunkSlices{in, out, send, recv}),
            HCCL_SUCCESS);

        ASSERT_EQ(LocalCopyCalls().size(), 0u);
        ASSERT_EQ(WriteCalls().size(), 1u);
        EXPECT_EQ(WriteCalls()[0].kind, WriteKind::kSendWrite);
        EXPECT_EQ(WriteCalls()[0].txSrc[0].addr_, in.addr_);
    }

    // ---------------------------------------------------------------------------
    // SendRecvNda：单对端 NDA 握手
    // ---------------------------------------------------------------------------
    TEST_F(InsTempDpuAlltoAllMeshNdaTest, SendRecvNdaSendOnlyUsesSendWriteForNdaWithoutHandshake)
    {
        uint8_t input[64] = {};
        uint8_t hcclBuff[128] = {};
        uint8_t remote[64] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        SetupTemplate(tpl, 0, 1, 4, 32, 64);

        TemplateDataParams params = MakeParams(2, {0, 8}, {0, 0}, {0, 0}, {0, 0}, input, nullptr, hcclBuff, 0);
        ChannelInfo link
            = MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 7, remote, 64);

        EXPECT_EQ(tpl.SendRecvNda(1, link, params), HCCL_SUCCESS);

        ASSERT_EQ(WriteCalls().size(), 1u);
        EXPECT_EQ(WriteCalls()[0].kind, WriteKind::kSendWriteNda);
        EXPECT_EQ(HandshakeCalls().size(), 0u); // 单向发送不握手
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, SendRecvNdaRecvOnlyUsesRecvWriteForNdaWithoutHandshake)
    {
        uint8_t output[64] = {};
        uint8_t hcclBuff[128] = {};
        uint8_t remote[64] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        SetupTemplate(tpl, 0, 1, 4, 32, 64);

        TemplateDataParams params = MakeParams(2, {0, 0}, {0, 8}, {0, 0}, {0, 0}, nullptr, output, hcclBuff, 0);
        ChannelInfo link
            = MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 7, remote, 64);

        EXPECT_EQ(tpl.SendRecvNda(1, link, params), HCCL_SUCCESS);

        ASSERT_EQ(WriteCalls().size(), 1u);
        EXPECT_EQ(WriteCalls()[0].kind, WriteKind::kRecvWriteNda);
        EXPECT_EQ(HandshakeCalls().size(), 0u); // 单向接收不握手
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, SendRecvNdaSendRecvPerformsFullHandshake)
    {
        uint8_t input[64] = {};
        uint8_t output[64] = {};
        uint8_t hcclBuff[128] = {};
        uint8_t remote[64] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        SetupTemplate(tpl, 0, 1, 4, 32, 64);

        TemplateDataParams params = MakeParams(2, {0, 8}, {0, 8}, {0, 0}, {0, 0}, input, output, hcclBuff, 0);
        ChannelInfo link
            = MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 7, remote, 64);

        EXPECT_EQ(tpl.SendRecvNda(1, link, params), HCCL_SUCCESS);

        ASSERT_EQ(HandshakeCalls().size(), 6u);
        // Record(ACK) -> Wait(ACK) -> Write(DATA_SIGNAL) -> Wait(DATA_SIGNAL) -> Record(FIN_ACK) -> Wait(FIN_ACK)
        EXPECT_EQ(HandshakeCalls()[0].kind, HandshakeCall::kNotifyRecord);
        EXPECT_EQ(HandshakeCalls()[0].notifyIdx, NOTIFY_IDX_ACK);
        EXPECT_EQ(HandshakeCalls()[1].kind, HandshakeCall::kNotifyWait);
        EXPECT_EQ(HandshakeCalls()[1].notifyIdx, NOTIFY_IDX_ACK);
        EXPECT_EQ(HandshakeCalls()[2].kind, HandshakeCall::kWriteWithNotify);
        EXPECT_EQ(HandshakeCalls()[2].notifyIdx, NOTIFY_IDX_DATA_SIGNAL);
        EXPECT_EQ(HandshakeCalls()[3].kind, HandshakeCall::kNotifyWait);
        EXPECT_EQ(HandshakeCalls()[3].notifyIdx, NOTIFY_IDX_DATA_SIGNAL);
        EXPECT_EQ(HandshakeCalls()[4].kind, HandshakeCall::kNotifyRecord);
        EXPECT_EQ(HandshakeCalls()[4].notifyIdx, NOTIFY_IDX_FIN_ACK);
        EXPECT_EQ(HandshakeCalls()[5].kind, HandshakeCall::kNotifyWait);
        EXPECT_EQ(HandshakeCalls()[5].notifyIdx, NOTIFY_IDX_FIN_ACK);
        // 每个原语都透传了 thread 与 channel 句柄
        for (const auto& call : HandshakeCalls()) {
            EXPECT_EQ(call.thread, 1u);
            EXPECT_EQ(call.channel, 7u);
        }
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, SendRecvNdaNotifyRecordFailurePropagates)
    {
        uint8_t input[64] = {};
        uint8_t output[64] = {};
        uint8_t hcclBuff[128] = {};
        uint8_t remote[64] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        SetupTemplate(tpl, 0, 1, 4, 32, 64);

        TemplateDataParams params = MakeParams(2, {0, 8}, {0, 8}, {0, 0}, {0, 0}, input, output, hcclBuff, 0);
        ChannelInfo link
            = MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 7, remote, 64);

        SetNotifyRecordResult(HCCL_E_INTERNAL);
        EXPECT_NE(tpl.SendRecvNda(1, link, params), HCCL_SUCCESS);
    }

    // ---------------------------------------------------------------------------
    // SendRecvNdaMesh：批量 MESH 四阶段握手
    // ---------------------------------------------------------------------------
    TEST_F(InsTempDpuAlltoAllMeshNdaTest, SendRecvNdaMeshPerformsFourPhaseHandshakeAcrossPeers)
    {
        uint8_t hcclBuff[256] = {};
        uint8_t remote[64] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        SetupTemplate(tpl, 0, 1, 4, 32, 64);

        TemplateDataParams params
            = MakeParams(4, {0, 8, 8, 8}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, nullptr, nullptr, hcclBuff, 0);
        ChannelInfo c1
            = MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 11, remote, 64);
        ChannelInfo c2
            = MakeChannel(2, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 22, remote, 64);
        ChannelInfo c3
            = MakeChannel(3, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 33, remote, 64);

        std::vector<std::pair<u32, const ChannelInfo*>> peers = {{1, &c1}, {2, &c2}, {3, &c3}};
        EXPECT_EQ(tpl.SendRecvNdaMesh(peers, params), HCCL_SUCCESS);

        // 3 对端批量握手，共 18 次原语。phase 2/3 按对端交错（每个对端先 Wait 再 Write/Record）：
        // Record(ACK)×3 → (Wait(ACK)+Write(DATA_SIGNAL))×3 → (Wait(DATA_SIGNAL)+Record(FIN_ACK))×3 → Wait(FIN_ACK)×3
        ASSERT_EQ(HandshakeCalls().size(), 18u);
        for (u32 p = 0; p < 3; ++p) {
            EXPECT_EQ(HandshakeCalls()[p].kind, HandshakeCall::kNotifyRecord);
            EXPECT_EQ(HandshakeCalls()[p].notifyIdx, NOTIFY_IDX_ACK);
        }
        for (u32 p = 0; p < 3; ++p) {
            EXPECT_EQ(HandshakeCalls()[3 + p * 2].kind, HandshakeCall::kNotifyWait);
            EXPECT_EQ(HandshakeCalls()[3 + p * 2].notifyIdx, NOTIFY_IDX_ACK);
            EXPECT_EQ(HandshakeCalls()[3 + p * 2 + 1].kind, HandshakeCall::kWriteWithNotify);
            EXPECT_EQ(HandshakeCalls()[3 + p * 2 + 1].notifyIdx, NOTIFY_IDX_DATA_SIGNAL);
        }
        for (u32 p = 0; p < 3; ++p) {
            EXPECT_EQ(HandshakeCalls()[9 + p * 2].kind, HandshakeCall::kNotifyWait);
            EXPECT_EQ(HandshakeCalls()[9 + p * 2].notifyIdx, NOTIFY_IDX_DATA_SIGNAL);
            EXPECT_EQ(HandshakeCalls()[9 + p * 2 + 1].kind, HandshakeCall::kNotifyRecord);
            EXPECT_EQ(HandshakeCalls()[9 + p * 2 + 1].notifyIdx, NOTIFY_IDX_FIN_ACK);
        }
        for (u32 p = 0; p < 3; ++p) {
            EXPECT_EQ(HandshakeCalls()[15 + p].kind, HandshakeCall::kNotifyWait);
            EXPECT_EQ(HandshakeCalls()[15 + p].notifyIdx, NOTIFY_IDX_FIN_ACK);
        }
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, SendRecvNdaMeshRejectsUnsupportedLocation)
    {
        uint8_t hcclBuff[256] = {};
        uint8_t remote[64] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        SetupTemplate(tpl, 0, 1, 4, 32, 64);

        TemplateDataParams params
            = MakeParams(4, {0, 8, 8, 8}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, nullptr, nullptr, hcclBuff, 0);
        ChannelInfo bad = MakeChannel(
            1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED, 11, remote, 64);

        std::vector<std::pair<u32, const ChannelInfo*>> peers = {{1, &bad}};
        EXPECT_EQ(tpl.SendRecvNdaMesh(peers, params), HCCL_E_NOT_SUPPORT);
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, SendRecvNdaMeshSkipsZeroLengthWriteForRecvOnlyPeer)
    {
        uint8_t hcclBuff[256] = {};
        uint8_t remote[64] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        SetupTemplate(tpl, 0, 1, 4, 32, 64);

        // peer1 收发都有（sendCounts=8），peer2 纯收（sendCounts=0, recvCounts=8）
        TemplateDataParams params
            = MakeParams(3, {0, 8, 0}, {0, 8, 8}, {0, 0, 0}, {0, 0, 0}, nullptr, nullptr, hcclBuff, 0);
        ChannelInfo c1
            = MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 11, remote, 64);
        ChannelInfo c2
            = MakeChannel(2, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 22, remote, 64);

        std::vector<std::pair<u32, const ChannelInfo*>> peers = {{1, &c1}, {2, &c2}};
        EXPECT_EQ(tpl.SendRecvNdaMesh(peers, params), HCCL_SUCCESS);

        // Record(ACK)×2 → peer1 Wait(ACK)+Write(DATA_SIGNAL) → peer2 Wait(ACK)+Record(DATA_SIGNAL)
        // → Wait(DATA_SIGNAL)+Record(FIN_ACK)×2 → Wait(FIN_ACK)×2，共 12 次原语
        ASSERT_EQ(HandshakeCalls().size(), 12u);
        EXPECT_EQ(HandshakeCalls()[3].kind, HandshakeCall::kWriteWithNotify); // peer1 正常写
        EXPECT_EQ(HandshakeCalls()[5].kind, HandshakeCall::kNotifyRecord);    // peer2 无数据，补 DATA_SIGNAL
        EXPECT_EQ(HandshakeCalls()[5].notifyIdx, NOTIFY_IDX_DATA_SIGNAL);
    }

    // ---------------------------------------------------------------------------
    // PreCopyDataToCclInBuf / PostCopyDataToRecvBuf：AppendCopy 连续切片合并
    // ---------------------------------------------------------------------------
    TEST_F(InsTempDpuAlltoAllMeshNdaTest, PreCopyCoalescesContinuousRocSlicesIntoOneLocalCopy)
    {
        uint8_t input[256] = {};
        uint8_t hcclBuff[256] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        const u64 n = 8;
        SetupTemplate(tpl, 0, 1, 4, n * 4, 0); // block = n*4 使目的地址连续

        TemplateDataParams params
            = MakeParams(3, {0, n, n}, {0, 0, 0}, {0, n, 2 * n}, {0, 0, 0}, input, nullptr, hcclBuff, 0);
        std::map<u32, std::vector<ChannelInfo>> channels;
        channels[1] = {MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST)};
        channels[2] = {MakeChannel(2, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST)};
        std::vector<ThreadHandle> threads = {1};

        EXPECT_EQ(tpl.PreCopyDataToCclInBuf(MakeRankIds(3), params, channels, threads), HCCL_SUCCESS);

        ASSERT_EQ(LocalCopyCalls().size(), 1u); // 两个连续切片合并为一次拷贝
        EXPECT_EQ(LocalCopyCalls()[0].src.addr_, input);
        EXPECT_EQ(LocalCopyCalls()[0].src.offset_, n * 4u);
        EXPECT_EQ(LocalCopyCalls()[0].src.size_, 2 * n * 4u);
        EXPECT_EQ(LocalCopyCalls()[0].dst.addr_, hcclBuff);
        EXPECT_EQ(LocalCopyCalls()[0].dst.offset_, n * 4u);
        EXPECT_EQ(LocalCopyCalls()[0].dst.size_, 2 * n * 4u);
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, PreCopySplitsNonContinuousSlicesIntoSeparateLocalCopies)
    {
        uint8_t input[256] = {};
        uint8_t hcclBuff[256] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        const u64 n = 8;
        SetupTemplate(tpl, 0, 1, 4, 2 * n * 4, 0); // block = 2*n*4 使目的地址不连续

        TemplateDataParams params
            = MakeParams(3, {0, n, n}, {0, 0, 0}, {0, n, 2 * n}, {0, 0, 0}, input, nullptr, hcclBuff, 0);
        std::map<u32, std::vector<ChannelInfo>> channels;
        channels[1] = {MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST)};
        channels[2] = {MakeChannel(2, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST)};
        std::vector<ThreadHandle> threads = {1};

        EXPECT_EQ(tpl.PreCopyDataToCclInBuf(MakeRankIds(3), params, channels, threads), HCCL_SUCCESS);

        ASSERT_EQ(LocalCopyCalls().size(), 2u); // 目的不连续拆成两次拷贝
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, PostCopyCoalescesContinuousRocRecvSlices)
    {
        uint8_t output[256] = {};
        uint8_t hcclBuff[256] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        const u64 n = 8;
        SetupTemplate(tpl, 0, 1, 4, n * 4, n * 4); // halfMaxTmpMemSize_ = n*4

        TemplateDataParams params
            = MakeParams(3, {0, 0, 0}, {0, n, n}, {0, 0, 0}, {0, n, 2 * n}, nullptr, output, hcclBuff, 0);
        std::map<u32, std::vector<ChannelInfo>> channels;
        channels[1] = {MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST)};
        channels[2] = {MakeChannel(2, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST)};
        std::vector<ThreadHandle> threads = {1};

        EXPECT_EQ(tpl.PostCopyDataToRecvBuf(MakeRankIds(3), params, channels, threads), HCCL_SUCCESS);

        ASSERT_EQ(LocalCopyCalls().size(), 1u);
        EXPECT_EQ(LocalCopyCalls()[0].dst.addr_, output);
        EXPECT_EQ(LocalCopyCalls()[0].dst.offset_, n * 4u);
        EXPECT_EQ(LocalCopyCalls()[0].dst.size_, 2 * n * 4u);
    }

    // ---------------------------------------------------------------------------
    // SendRecvData：ROCE 对端路由（ALLTOALL → 批量 MESH，ALLTOALLV → 逐对端）
    // ---------------------------------------------------------------------------
    TEST_F(InsTempDpuAlltoAllMeshNdaTest, SendRecvDataRoutesAlltoallRocPeersToMeshBatch)
    {
        uint8_t input[256] = {};
        uint8_t output[256] = {};
        uint8_t hcclBuff[256] = {};
        uint8_t remote[64] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        SetupTemplate(tpl, 0, 1, 4, 32, 64);

        OpParam param{};
        param.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
        std::strncpy(param.algTag, "ut", sizeof(param.algTag));

        TemplateDataParams params
            = MakeParams(4, {0, 8, 8, 8}, {0, 8, 8, 8}, {0, 0, 0, 0}, {0, 0, 0, 0}, input, output, hcclBuff, 0);
        TemplateResource resource;
        resource.channels[1] = {
            MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 11, remote, 64)};
        resource.channels[2] = {
            MakeChannel(2, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 22, remote, 64)};
        resource.channels[3] = {
            MakeChannel(3, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 33, remote, 64)};

        EXPECT_EQ(tpl.SendRecvData(param, MakeRankIds(4), params, resource), HCCL_SUCCESS);

        // batch 窗口：End 一次进入批量，Start 一次恢复
        ASSERT_EQ(BatchCalls().size(), 2u);
        EXPECT_FALSE(BatchCalls()[0].start);
        EXPECT_TRUE(BatchCalls()[1].start);
        // 3 对端批量 MESH
        // 握手（Record(ACK)/Wait(ACK)/Write(DATA_SIGNAL)/Wait(DATA_SIGNAL)/Record(FIN_ACK)/Wait(FIN_ACK) 各 3 次）
        ASSERT_EQ(HandshakeCalls().size(), 18u);
        EXPECT_EQ(WriteCalls().size(), 0u);
    }

    TEST_F(InsTempDpuAlltoAllMeshNdaTest, SendRecvDataRoutesAlltoallvRocPeersPerPeer)
    {
        uint8_t input[256] = {};
        uint8_t output[256] = {};
        uint8_t hcclBuff[256] = {};
        uint8_t remote[64] = {};
        InsTempDpuAlltoAllMeshNda tpl;
        SetupTemplate(tpl, 0, 1, 4, 32, 64);

        OpParam param{};
        param.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
        std::strncpy(param.algTag, "ut", sizeof(param.algTag));

        TemplateDataParams params
            = MakeParams(3, {0, 8, 8}, {0, 8, 8}, {0, 0, 0}, {0, 0, 0}, input, output, hcclBuff, 0);
        TemplateResource resource;
        resource.channels[1] = {
            MakeChannel(1, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 11, remote, 64)};
        resource.channels[2] = {
            MakeChannel(2, CommProtocol::COMM_PROTOCOL_ROCE, EndpointLocType::ENDPOINT_LOC_TYPE_HOST, 22, remote, 64)};

        EXPECT_EQ(tpl.SendRecvData(param, MakeRankIds(3), params, resource), HCCL_SUCCESS);

        ASSERT_EQ(BatchCalls().size(), 2u);
        EXPECT_FALSE(BatchCalls()[0].start);
        EXPECT_TRUE(BatchCalls()[1].start);
        // 逐对端 SendRecvNda，每个对端一次完整 6 步握手
        ASSERT_EQ(HandshakeCalls().size(), 12u);
        EXPECT_EQ(HandshakeCalls()[0].notifyIdx, NOTIFY_IDX_ACK);
        EXPECT_EQ(HandshakeCalls()[6].notifyIdx, NOTIFY_IDX_ACK); // 第二个对端重新从 ACK 开始
    }

} // namespace
} // namespace nda_ut
