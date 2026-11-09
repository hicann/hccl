/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "wrapper_test_common.h"

namespace wrapper_ut {

TEST_F(DpuWriteReduceTest, ExecutesSupportedTypeAndReduceOpMatrix)
{
    struct Type {
        HcclDataType hccl;
        HcommDataType hcomm;
        u64 size;
    };
    const std::array<Type, 6> types
        = {{{HCCL_DATA_TYPE_INT8, HCOMM_DATA_TYPE_INT8, 1},
            {HCCL_DATA_TYPE_INT16, HCOMM_DATA_TYPE_INT16, 2},
            {HCCL_DATA_TYPE_INT32, HCOMM_DATA_TYPE_INT32, 4},
            {HCCL_DATA_TYPE_FP16, HCOMM_DATA_TYPE_FP16, 2},
            {HCCL_DATA_TYPE_FP32, HCOMM_DATA_TYPE_FP32, 4},
            {HCCL_DATA_TYPE_BFP16, HCOMM_DATA_TYPE_BFP16, 2}}};
    const std::array<std::pair<HcclReduceOp, HcommReduceOp>, 3> ops
        = {{{HCCL_REDUCE_SUM, HCOMM_REDUCE_SUM},
            {HCCL_REDUCE_MAX, HCOMM_REDUCE_MAX},
            {HCCL_REDUCE_MIN, HCOMM_REDUCE_MIN}}};
    std::array<u8, 32> src{}, dst{};
    auto channel = MakeChannel(0x1001);
    for (const auto& type : types) {
        for (const auto& op : ops) {
            SCOPED_TRACE(testing::Message() << type.hccl << ":" << op.first);
            g_calls.clear();
            DpuTransferCtx transfer{};
            transfer.txCh = &channel;
            AddSendSlice(transfer, src.data(), 2, dst.data(), 4, 4 * type.size, 4);
            ASSERT_EQ(DpuWriteReduceTransfer(transfer, type.hccl, op.first), HCCL_SUCCESS);
            ASSERT_EQ(g_calls.size(), 4);
            const auto& write = g_calls[1];
            EXPECT_EQ(write.api, StubApi::WRITE_REDUCE);
            EXPECT_EQ(write.dataType, type.hcomm);
            EXPECT_EQ(write.reduceOp, op.second);
            EXPECT_EQ(write.count, 4);
            EXPECT_EQ(write.src, src.data() + 2);
            EXPECT_EQ(write.dst, dst.data() + 4);
            EXPECT_EQ(write.channel, channel.handle);
            EXPECT_EQ(write.notifyIdx, NOTIFY_IDX_DATA_SIGNAL);
        }
    }
}

TEST_F(DpuWriteReduceTest, ExecutesExactTraceForDirectionsSlicesAndChannelAliases)
{
    struct Case {
        u32 sends;
        u32 recvs;
        bool sameHandle;
        bool zeroSlices;
    };
    const std::array<Case, 8> cases
        = {{{0, 0, false, false},
            {1, 0, false, false},
            {0, 1, false, false},
            {1, 1, false, false},
            {1, 1, true, false},
            {2, 2, false, true},
            {2, 0, false, true},
            {0, 2, false, true}}};
    std::array<u8, 64> src{}, dst{};
    ExecTimeoutManager::Instance().SetExecTimeout(37);
    for (const auto& c : cases) {
        SCOPED_TRACE(testing::Message() << c.sends << ":" << c.recvs << ":" << c.sameHandle << ":" << c.zeroSlices);
        g_calls.clear();
        auto tx = MakeChannel(0x1001);
        auto rx = MakeChannel(c.sameHandle ? 0x1001 : 0x1002);
        DpuTransferCtx transfer{};
        transfer.txCh = c.sends ? &tx : nullptr;
        transfer.rxCh = c.recvs ? &rx : nullptr;
        for (u32 i = 0; i < c.sends; ++i) {
            if (c.zeroSlices)
                AddSendSlice(transfer, src.data(), 0, dst.data(), 0, 0, 0);
            AddSendSlice(transfer, src.data(), 2 + i * 8, dst.data(), 4 + i * 8, 8, 4);
        }
        for (u32 i = 0; i < c.recvs; ++i) {
            if (c.zeroSlices)
                AddRecvSlice(transfer, src.data(), dst.data(), 0, 0);
            AddRecvSlice(transfer, src.data(), dst.data(), 8, 4);
        }
        ASSERT_EQ(DpuWriteReduceTransfer(transfer, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM), HCCL_SUCCESS);
        std::vector<std::tuple<StubApi, ChannelHandle, u32>> expected;
        if (c.recvs)
            expected.emplace_back(StubApi::RECORD, rx.handle, NOTIFY_IDX_STEP_SYNC);
        if (c.sends)
            expected.emplace_back(StubApi::WAIT, tx.handle, NOTIFY_IDX_STEP_SYNC);
        for (u32 i = 0; i < c.sends; ++i)
            expected.emplace_back(StubApi::WRITE_REDUCE, tx.handle, NOTIFY_IDX_DATA_SIGNAL);
        if (c.sends)
            expected.emplace_back(StubApi::DRAIN, tx.handle, 0);
        if (c.recvs && !(c.sends && c.sameHandle))
            expected.emplace_back(StubApi::DRAIN, rx.handle, 0);
        if (c.sends || c.recvs)
            expected.emplace_back(StubApi::FENCE, 0, 0);
        for (u32 i = 0; i < c.recvs; ++i)
            expected.emplace_back(StubApi::WAIT, rx.handle, NOTIFY_IDX_DATA_SIGNAL);
        ASSERT_EQ(g_calls.size(), expected.size());
        u32 writeIdx = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            const auto& call = g_calls[i];
            EXPECT_EQ(std::make_tuple(call.api, call.channel, call.notifyIdx), expected[i]);
            if (call.api == StubApi::WAIT)
                EXPECT_EQ(call.timeout, 37);
            if (call.api == StubApi::WRITE_REDUCE) {
                EXPECT_EQ(call.src, src.data() + 2 + writeIdx * 8);
                EXPECT_EQ(call.dst, dst.data() + 4 + writeIdx * 8);
                EXPECT_EQ(call.count, 4);
                ++writeIdx;
            }
        }
    }
}

TEST_F(DpuWriteReduceTest, StopsAtEveryFailedPrimitiveIncludingRepeatedCalls)
{
    std::array<u8, 32> src{}, dst{};
    auto tx = MakeChannel(0x1001), rx = MakeChannel(0x1002);
    DpuTransferCtx transfer{};
    transfer.txCh = &tx;
    transfer.rxCh = &rx;
    for (u32 i = 0; i < 2; ++i) {
        AddSendSlice(transfer, src.data(), i * 8, dst.data(), i * 8, 8, 4);
        AddRecvSlice(transfer, src.data(), dst.data(), 8, 4);
    }
    ASSERT_EQ(DpuWriteReduceTransfer(transfer, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM), HCCL_SUCCESS);
    const auto success = g_calls;
    for (size_t position = 0; position < success.size(); ++position) {
        for (auto failureCode : {HCCL_E_INTERNAL, HCCL_E_PARA}) {
            SCOPED_TRACE(testing::Message() << "position=" << position << " code=" << failureCode);
            g_calls.clear();
            const auto api = success[position].api;
            const u32 occurrence
                = std::count_if(success.begin(), success.begin() + position + 1, [api](const StubCall& call) {
                      return call.api == api;
                  });
            InjectFailure(api, occurrence);
            g_failureCode = failureCode;
            EXPECT_EQ(DpuWriteReduceTransfer(transfer, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM), failureCode);
            ASSERT_EQ(g_calls.size(), position + 1);
            for (size_t i = 0; i < g_calls.size(); ++i) {
                EXPECT_EQ(g_calls[i].api, success[i].api);
                EXPECT_EQ(g_calls[i].channel, success[i].channel);
                EXPECT_EQ(g_calls[i].notifyIdx, success[i].notifyIdx);
            }
        }
    }
}

TEST(TemplateDataParamsTest, DpuRunInfoRoundTripPreservesSerializedFieldsForReduceOps)
{
    const std::array<HcclReduceOp, 5> reduceOps
        = {HCCL_REDUCE_SUM, HCCL_REDUCE_PROD, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN, HCCL_REDUCE_RESERVED};
    for (HcclReduceOp reduceOp : reduceOps) {
        DPURunInfo source{};
        source.templateName = "InsTempReduceScatterNhrDpu";
        FillTemplateDataParams(source.tempAlgParams, reduceOp);
        ChannelInfo channel = MakeChannel(0x7100);
        channel.remoteRank = 7;
        channel.remoteCclMem = {HCCL_MEM_TYPE_HOST, reinterpret_cast<void*>(static_cast<uintptr_t>(0x7200)), 0x7300};
        source.channels[7] = {channel};
        source.myRank = 3;
        source.subCommRanks = {{3, 5, 7}};
        source.execTimeout = 19;

        std::vector<char> serialized = source.Serialize();
        DPURunInfo restored{};
        restored.DeSerialize(serialized);

        EXPECT_EQ(restored.templateName, source.templateName);
        ExpectTemplateDataParamsEqual(restored.tempAlgParams, source.tempAlgParams);
        ASSERT_EQ(restored.channels.size(), 1);
        ASSERT_EQ(restored.channels.at(7).size(), 1);
        const ChannelInfo& restoredChannel = restored.channels.at(7).front();
        EXPECT_EQ(restoredChannel.isValid, channel.isValid);
        EXPECT_EQ(restoredChannel.remoteRank, channel.remoteRank);
        EXPECT_EQ(restoredChannel.protocol, channel.protocol);
        EXPECT_EQ(restoredChannel.locationType, channel.locationType);
        EXPECT_EQ(restoredChannel.handle, channel.handle);
        EXPECT_EQ(restoredChannel.remoteCclMem.type, channel.remoteCclMem.type);
        EXPECT_EQ(restoredChannel.remoteCclMem.addr, channel.remoteCclMem.addr);
        EXPECT_EQ(restoredChannel.remoteCclMem.size, channel.remoteCclMem.size);
        EXPECT_EQ(restored.myRank, source.myRank);
        EXPECT_EQ(restored.subCommRanks, source.subCommRanks);
        EXPECT_EQ(restored.execTimeout, source.execTimeout);
    }
}

TEST_F(DpuWriteReduceTest, RejectsInvalidSliceMetadataBeforeCommunication)
{
    std::array<u8, 64> localCcl{};
    std::array<u8, 64> remote{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    params.repeatNum = 1;
    AicpuNHRStepInfo step{};
    step.toRank = 40;
    step.fromRank = 20;
    step.txSliceIdxs = {1};
    step.rxSliceIdxs = {2};
    const std::map<u32, std::vector<ChannelInfo>> channels
        = {{20, {MakeDpuChannel(0x1020, remote.data())}}, {40, {MakeDpuChannel(0x1040, remote.data())}}};
    for (u32 sliceIdx : {1U, 2U}) {
        for (u64 count : std::array<u64, 2>{3, std::numeric_limits<u64>::max()}) {
            auto badParams = params;
            badParams.allRankProcessedDataCount[sliceIdx] = count;
            EXPECT_EQ(DpuBatchWriteReduceTransferNHR(step, channels, badParams, 0, 4), HCCL_E_PARA);
        }
    }
    for (bool send : {false, true}) {
        for (u32 fault = 0; fault < 6; ++fault) {
            SCOPED_TRACE(testing::Message() << "send=" << send << " metadataFault=" << fault);
            DpuTransferCtx transfer{};
            auto channel = MakeDpuChannel(0x1001, remote.data());
            auto& src = send ? transfer.txSrcSlices : transfer.rxSrcSlices;
            auto& dst = send ? transfer.txDstSlices : transfer.rxDstSlices;
            if (send)
                transfer.txCh = &channel;
            else
                transfer.rxCh = &channel;
            src.emplace_back(localCcl.data(), 0, 8, 4);
            dst.emplace_back(remote.data(), 0, 8, 4);
            if (fault == 0)
                dst.clear();
            if (fault == 1)
                dst[0].size_ = 4;
            if (fault == 2)
                dst[0].count_ = 3;
            if (fault == 3) {
                src[0].size_ = 0;
                dst[0].size_ = 0;
            }
            if (fault == 4)
                src[0].addr_ = nullptr;
            if (fault == 5)
                dst[0].addr_ = nullptr;
            EXPECT_EQ(CheckWriteReduceTransfer(transfer, HCCL_DATA_TYPE_FP16), fault >= 4 ? HCCL_E_PTR : HCCL_E_PARA);
        }
    }
    EXPECT_TRUE(g_calls.empty());
}

TEST_F(DpuWriteReduceTest, RejectsInvalidDataTypesBeforeSliceChecks)
{
    std::array<u8, 64> localCcl{};
    std::array<u8, 64> remote{};
    const ChannelInfo channel = MakeDpuChannel(0x1001, remote.data());
    std::array<DpuTransferCtx, 3> transfers{};
    transfers[1].txCh = &channel;
    transfers[1].txSrcSlices.emplace_back(localCcl.data(), 0, 8, 4);
    transfers[1].txDstSlices.emplace_back(remote.data(), 0, 8, 4);
    transfers[2].rxCh = &channel;
    transfers[2].rxSrcSlices.emplace_back(remote.data(), 0, 8, 4);
    transfers[2].rxDstSlices.emplace_back(localCcl.data(), 0, 8, 4);

    for (HcclDataType dataType :
         {static_cast<HcclDataType>(-1), HCCL_DATA_TYPE_RESERVED,
          static_cast<HcclDataType>(HCCL_DATA_TYPE_RESERVED + 1)}) {
        for (size_t direction = 0; direction < transfers.size(); ++direction) {
            SCOPED_TRACE(testing::Message() << "dataType=" << dataType << " direction=" << direction);
            EXPECT_EQ(CheckWriteReduceTransfer(transfers[direction], dataType), HCCL_E_PARA);
            if (direction != 0) {
                auto nullDestination = transfers[direction];
                auto& dstSlices = direction == 1 ? nullDestination.txDstSlices : nullDestination.rxDstSlices;
                dstSlices[0].addr_ = nullptr;
                EXPECT_EQ(CheckWriteReduceTransfer(nullDestination, HCCL_DATA_TYPE_FP16), HCCL_E_PTR);
                EXPECT_EQ(CheckWriteReduceTransfer(nullDestination, dataType), HCCL_E_PARA);
            }
            EXPECT_TRUE(g_calls.empty());
        }
    }
}

TEST_F(DpuWriteReduceTest, RejectsInvalidStepSliceIndicesBeforeCommunication)
{
    std::array<u8, 64> localCcl{};
    std::array<u8, 64> remote{};
    const std::map<u32, std::vector<ChannelInfo>> channels
        = {{20, {MakeDpuChannel(0x1020, remote.data())}}, {40, {MakeDpuChannel(0x1040, remote.data())}}};
    for (bool send : {false, true}) {
        for (u32 fault = 0; fault < 6; ++fault) {
            SCOPED_TRACE(testing::Message() << "send=" << send << " indexFault=" << fault);
            auto params = MakeNhrParams(localCcl.data());
            AicpuNHRStepInfo step{};
            step.toRank = 40;
            step.fromRank = 20;
            auto& indices = send ? step.txSliceIdxs : step.rxSliceIdxs;
            indices = {fault == 0 ? 4U : fault == 1 ? std::numeric_limits<u32>::max() : 1U};
            if (fault == 2)
                params.allRankSliceSize.resize(1);
            if (fault == 3)
                params.allRankDispls.resize(1);
            if (fault == 4)
                params.allRankProcessedDataCount.resize(1);
            const u32 rankSize = fault == 5 ? 1 : 4;
            EXPECT_EQ(DpuBatchWriteReduceTransferNHR(step, channels, params, 0, rankSize), HCCL_E_PARA);
            EXPECT_TRUE(g_calls.empty());
        }
    }
}

TEST_F(DpuWriteReduceTest, RejectsUnsupportedTypesOpsAndNullBufferAtTemplateEntry)
{
    const std::array<HcclDataType, 6> unsupportedDataTypes
        = {HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_UINT32,
           HCCL_DATA_TYPE_FP64,  HCCL_DATA_TYPE_HIF8,   HCCL_DATA_TYPE_RESERVED};
    std::array<u8, 64> localCcl{};
    std::array<u8, 64> remote{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    params.repeatNum = 1;
    const std::map<u32, std::vector<ChannelInfo>> channels
        = {{20, {MakeDpuChannel(0x1020, remote.data())}},
           {30, {MakeDpuChannel(0x1030, remote.data())}},
           {40, {MakeDpuChannel(0x1040, remote.data())}}};
    InsTempReduceScatterNhrDpu algorithm;

    for (HcclDataType dataType : unsupportedDataTypes) {
        params.dataType = dataType;
        const u64 typeSize = dataType == HCCL_DATA_TYPE_RESERVED ? 0 : DATATYPE_SIZE_TABLE[dataType];
        // Keep count/size metadata consistent so it cannot mask a missing type check.
        for (u32 sliceIdx = 0; sliceIdx < params.allRankSliceSize.size(); ++sliceIdx) {
            params.allRankSliceSize[sliceIdx] = params.allRankProcessedDataCount[sliceIdx] * typeSize;
            params.allRankDispls[sliceIdx] = sliceIdx * 4ULL * typeSize;
        }
        EXPECT_EQ(ValidateReduceScatterNhrParams(params, 4), HCCL_E_PARA);
        EXPECT_EQ(algorithm.DPUKernelRun(params, channels, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);
    }
    params.dataType = static_cast<HcclDataType>(-1);
    EXPECT_EQ(ValidateReduceScatterNhrParams(params, 4), HCCL_E_PARA);
    EXPECT_EQ(algorithm.DPUKernelRun(params, channels, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);
    params = MakeNhrParams(localCcl.data());
    for (auto op : {HCCL_REDUCE_PROD, HCCL_REDUCE_RESERVED, static_cast<HcclReduceOp>(-1)}) {
        params.reduceOp = op;
        EXPECT_EQ(algorithm.DPUKernelRun(params, channels, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);
    }
    params = MakeNhrParams(localCcl.data());
    params.buffInfo.hcclBuff.addr = nullptr;
    EXPECT_EQ(algorithm.DPUKernelRun(params, channels, 10, {{10, 20, 30, 40}}), HCCL_E_PTR);
    EXPECT_TRUE(g_calls.empty());
}

TEST_F(DpuWriteReduceTest, RejectsInvalidDpuChannelsBeforeCommunication)
{
    std::array<u8, 64> localCcl{}, remote{};
    auto params = MakeNhrParams(localCcl.data());
    params.repeatNum = 1;
    for (bool send : {false, true}) {
        for (u32 fault = 0; fault < 8; ++fault) {
            SCOPED_TRACE(testing::Message() << "send=" << send << " fault=" << fault);
            AicpuNHRStepInfo step{};
            step.toRank = 40;
            step.fromRank = 40;
            if (send)
                step.txSliceIdxs = {1};
            else
                step.rxSliceIdxs = {2};
            std::map<u32, std::vector<ChannelInfo>> channels;
            if (fault != 0)
                channels[40] = {};
            if (fault > 1) {
                auto channel = MakeDpuChannel(0x1040, remote.data());
                if (fault == 2)
                    channel.isValid = false;
                if (fault == 3)
                    channel.handle = 0;
                if (fault == 4)
                    channel.locationType = EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
                if (fault == 5)
                    channel.protocol = CommProtocol::COMM_PROTOCOL_RESERVED;
                if (fault == 6)
                    channel.remoteCclMem.addr = nullptr;
                if (fault == 7)
                    channel.remoteCclMem.size = 0;
                channels[40] = {channel};
            }
            EXPECT_EQ(
                DpuBatchWriteReduceTransferNHR(step, channels, params, 0, 4), fault == 6 ? HCCL_E_PTR : HCCL_E_PARA);
            EXPECT_TRUE(g_calls.empty());
        }
    }
}

TEST_F(DpuWriteReduceTest, ExistingBatchTransferStillUsesPlainWrite)
{
    std::array<u8, 8> src{};
    std::array<u8, 8> dst{};
    ChannelInfo channel = MakeChannel(0x1001);
    DpuTransferCtx transfer{};
    transfer.txCh = &channel;
    AddSendSlice(transfer, src.data(), 0, dst.data(), 0, 8, 4);
    std::vector<DpuTransferCtx> transfers{transfer};

    ASSERT_EQ(DpuBatchTransfer(transfers), HCCL_SUCCESS);
    EXPECT_EQ(
        std::count_if(
            g_calls.begin(), g_calls.end(),
            [](const StubCall& call) {
                return call.api == StubApi::WRITE;
            }),
        1);
    EXPECT_EQ(
        std::count_if(
            g_calls.begin(), g_calls.end(),
            [](const StubCall& call) {
                return call.api == StubApi::WRITE_REDUCE;
            }),
        0);
}

TEST(ReduceScatterNhrDpuStepTest, BuildsStableStepsForPowerOfTwoAndNonPowerOfTwoRanks)
{
    for (u32 rankSize : {1U, 2U, 3U, 4U, 5U, 6U, 8U, 9U}) {
        std::vector<u32> ranks(rankSize);
        for (u32 rank = 0; rank < rankSize; ++rank) {
            ranks[rank] = rank * 2 + 10;
        }
        u32 expectedSteps = 0;
        for (u32 covered = 1; covered < rankSize; covered <<= 1) {
            ++expectedSteps;
        }
        for (u32 localRank = 0; localRank < rankSize; ++localRank) {
            InsTempReduceScatterNhrDpu algorithm(OpParam{}, ranks[localRank], {ranks});
            EXPECT_EQ(GetNHRStepNum(rankSize), expectedSteps);
            u32 totalSlices = 0;
            for (u32 step = 0; step < expectedSteps; ++step) {
                AicpuNHRStepInfo info{};
                ASSERT_EQ(algorithm.GetStepInfo(step, expectedSteps, info), HCCL_SUCCESS);
                totalSlices += info.nSlices;
                EXPECT_EQ(info.step, step);
                EXPECT_EQ(info.myRank, localRank);
                EXPECT_LT(info.toRank, rankSize);
                EXPECT_LT(info.fromRank, rankSize);
                EXPECT_EQ(info.txSliceIdxs.size(), info.nSlices);
                EXPECT_EQ(info.rxSliceIdxs.size(), info.nSlices);
                EXPECT_TRUE(std::all_of(info.txSliceIdxs.begin(), info.txSliceIdxs.end(), [rankSize](u32 sliceIdx) {
                    return sliceIdx < rankSize;
                }));
                EXPECT_TRUE(std::all_of(info.rxSliceIdxs.begin(), info.rxSliceIdxs.end(), [rankSize](u32 sliceIdx) {
                    return sliceIdx < rankSize;
                }));
            }
            EXPECT_EQ(totalSlices, rankSize - 1);
        }
    }

    InsTempReduceScatterNhrDpu algorithm(OpParam{}, 10, {{10, 20, 30, 40}});
    std::array<AicpuNHRStepInfo, 2> steps{};
    ASSERT_EQ(algorithm.GetStepInfo(0, steps.size(), steps[0]), HCCL_SUCCESS);
    ASSERT_EQ(algorithm.GetStepInfo(1, steps.size(), steps[1]), HCCL_SUCCESS);
    EXPECT_EQ(steps[0].toRank, 3);
    EXPECT_EQ(steps[0].fromRank, 1);
    EXPECT_EQ(steps[0].txSliceIdxs, (std::vector<u32>{3, 1}));
    EXPECT_EQ(steps[0].rxSliceIdxs, (std::vector<u32>{0, 2}));
    EXPECT_EQ(steps[1].toRank, 2);
    EXPECT_EQ(steps[1].fromRank, 2);
    EXPECT_EQ(steps[1].txSliceIdxs, (std::vector<u32>{2}));
    EXPECT_EQ(steps[1].rxSliceIdxs, (std::vector<u32>{0}));
}

TEST(ReduceScatterNhrDpuStepTest, RejectsEmptyRanksAndUnknownLocalRank)
{
    InsTempReduceScatterNhrDpu algorithm;
    AicpuNHRStepInfo step{};
    EXPECT_EQ(algorithm.GetStepInfo(0, 0, step), HCCL_E_PARA);
    EXPECT_EQ(algorithm.DPUKernelRun(TemplateDataParams{}, {}, 0, {}), HCCL_E_PARA);
    EXPECT_EQ(algorithm.DPUKernelRun(TemplateDataParams{}, {}, 0, {{}}), HCCL_E_PARA);
    EXPECT_EQ(algorithm.DPUKernelRun(TemplateDataParams{}, {}, 7, {{1, 3, 5}}), HCCL_E_PARA);
}

TEST_F(DpuWriteReduceTest, NhrTemplateDataRoundTripRunsStepsThroughWriteReduceWrapper)
{
    std::array<u8, 256> localCcl{};
    std::array<u8, 256> remote20{};
    std::array<u8, 256> remote30{};
    std::array<u8, 256> remote40{};
    DPURunInfo runInfo{};
    runInfo.templateName = "InsTempReduceScatterNhrDpu";
    runInfo.tempAlgParams = MakeNhrParams(localCcl.data());
    runInfo.tempAlgParams.dataType = HCCL_DATA_TYPE_INT16;
    runInfo.channels[20] = {MakeDpuChannel(0x1020, remote20.data())};
    runInfo.channels[30] = {MakeDpuChannel(0x1030, remote30.data())};
    runInfo.channels[40] = {MakeDpuChannel(0x1040, remote40.data())};
    runInfo.myRank = 10;
    runInfo.subCommRanks = {{10, 20, 30, 40}};

    std::vector<char> serialized = runInfo.Serialize();
    DPURunInfo restored{};
    restored.DeSerialize(serialized);
    InsTempReduceScatterNhrDpu algorithm;
    ASSERT_EQ(
        algorithm.DPUKernelRun(restored.tempAlgParams, restored.channels, restored.myRank, restored.subCommRanks),
        HCCL_SUCCESS);

    std::vector<StubCall> writes;
    std::copy_if(g_calls.begin(), g_calls.end(), std::back_inserter(writes), [](const StubCall& call) {
        return call.api == StubApi::WRITE_REDUCE;
    });
    ASSERT_EQ(writes.size(), 4);
    EXPECT_EQ(writes[0].channel, 0x1040);
    EXPECT_EQ(writes[0].src, localCcl.data() + 12);
    EXPECT_EQ(writes[0].dst, remote40.data() + 12);
    EXPECT_EQ(writes[1].channel, 0x1030);
    EXPECT_EQ(writes[1].src, localCcl.data() + 24);
    EXPECT_EQ(writes[1].dst, remote30.data() + 24);
    EXPECT_EQ(writes[2].channel, 0x1040);
    EXPECT_EQ(writes[2].src, localCcl.data() + 44);
    EXPECT_EQ(writes[2].dst, remote40.data() + 44);
    EXPECT_EQ(writes[3].channel, 0x1030);
    EXPECT_EQ(writes[3].src, localCcl.data() + 56);
    EXPECT_EQ(writes[3].dst, remote30.data() + 56);
    for (const StubCall& write : writes) {
        EXPECT_EQ(write.count, 4);
        EXPECT_EQ(write.dataType, HCOMM_DATA_TYPE_INT16);
        EXPECT_EQ(write.reduceOp, HCOMM_REDUCE_SUM);
    }
}

TEST_F(DpuWriteReduceTest, NhrDpuBatchWriteReduceTransferHandlesSendRecvPeersAndRepeatOffsets)
{
    std::array<u8, 128> localCcl{};
    std::array<u8, 128> remote20{};
    std::array<u8, 128> remote40{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    params.reduceOp = HCCL_REDUCE_MIN;
    ASSERT_EQ(ValidateReduceScatterNhrParams(params, 4), HCCL_SUCCESS);
    struct TransferCase {
        bool hasTx;
        bool hasRx;
        bool samePeer;
    };
    for (const TransferCase& transferCase :
         {TransferCase{true, false, false}, TransferCase{false, true, false}, TransferCase{true, true, true},
          TransferCase{true, true, false}, TransferCase{false, false, false}}) {
        g_calls.clear();
        AicpuNHRStepInfo step{};
        step.toRank = 40;
        step.fromRank = transferCase.samePeer ? 40 : 20;
        // 零长度slice即使在只有单向数据时也不应要求另一方向的channel。
        step.txSliceIdxs = transferCase.hasTx ? std::vector<u32>{3, 1} : std::vector<u32>{3};
        step.rxSliceIdxs = transferCase.hasRx ? std::vector<u32>{0, 2} : std::vector<u32>{3};
        std::map<u32, std::vector<ChannelInfo>> channels;
        if (transferCase.hasTx || (transferCase.hasRx && transferCase.samePeer)) {
            channels[40] = {MakeDpuChannel(0x1040, remote40.data())};
        }
        if (transferCase.hasRx && !transferCase.samePeer) {
            channels[20] = {MakeDpuChannel(0x1020, remote20.data())};
        }

        ASSERT_EQ(DpuBatchWriteReduceTransferNHR(step, channels, params, 1, 4), HCCL_SUCCESS);
        std::vector<StubCall> writes;
        std::copy_if(g_calls.begin(), g_calls.end(), std::back_inserter(writes), [](const StubCall& call) {
            return call.api == StubApi::WRITE_REDUCE;
        });
        ASSERT_EQ(writes.size(), transferCase.hasTx ? 1 : 0);
        if (transferCase.hasTx) {
            EXPECT_EQ(writes[0].channel, 0x1040);
            EXPECT_EQ(writes[0].src, localCcl.data() + 44);
            EXPECT_EQ(writes[0].dst, remote40.data() + 44);
            EXPECT_EQ(writes[0].count, 4);
            EXPECT_EQ(writes[0].dataType, HCOMM_DATA_TYPE_FP16);
            EXPECT_EQ(writes[0].reduceOp, HCOMM_REDUCE_MIN);
        }
        EXPECT_EQ(
            std::count_if(
                g_calls.begin(), g_calls.end(),
                [](const StubCall& call) {
                    return call.api == StubApi::WAIT && call.notifyIdx == NOTIFY_IDX_DATA_SIGNAL;
                }),
            transferCase.hasRx ? 2 : 0);
        const u32 expectedDrains = transferCase.samePeer ? 1 : transferCase.hasTx + transferCase.hasRx;
        EXPECT_EQ(
            std::count_if(
                g_calls.begin(), g_calls.end(),
                [](const StubCall& call) {
                    return call.api == StubApi::DRAIN;
                }),
            expectedDrains);
        if (!transferCase.hasTx && !transferCase.hasRx) {
            EXPECT_TRUE(g_calls.empty());
        }
    }
}

TEST_F(DpuWriteReduceTest, NhrScratchRepeatRangesUseRankCountTimesSliceSize)
{
    std::array<u8, 64> localCcl{};
    std::array<u8, 64> remote20{};
    std::array<u8, 64> remote30{};
    std::array<u8, 64> remote40{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    params.buffInfo.hcclBuff.size = localCcl.size();
    // 第二轮最大slice终点为4 + 4 * 8 + 20 + 8 = 64，与outputRepeatStride=64无关。
    ASSERT_EQ(ValidateReduceScatterNhrParams(params, 4), HCCL_SUCCESS);
    std::map<u32, std::vector<ChannelInfo>> channels;
    channels[20] = {MakeDpuChannel(0x1020, remote20.data())};
    channels[30] = {MakeDpuChannel(0x1030, remote30.data())};
    channels[40] = {MakeDpuChannel(0x1040, remote40.data())};
    channels[20][0].remoteCclMem.size = remote20.size();
    channels[30][0].remoteCclMem.size = remote30.size();
    channels[40][0].remoteCclMem.size = remote40.size();
    InsTempReduceScatterNhrDpu algorithm;
    ASSERT_EQ(algorithm.DPUKernelRun(params, channels, 10, {{10, 20, 30, 40}}), HCCL_SUCCESS);

    const auto lastWrite = std::find_if(g_calls.rbegin(), g_calls.rend(), [](const StubCall& call) {
        return call.api == StubApi::WRITE_REDUCE;
    });
    ASSERT_NE(lastWrite, g_calls.rend());
    EXPECT_EQ(lastWrite->src, localCcl.data() + 56);
    EXPECT_EQ(lastWrite->dst, remote30.data() + 56);
    EXPECT_EQ(lastWrite->count, 4);

    g_calls.clear();
    channels[20][0].remoteCclMem.size = 63;
    EXPECT_EQ(algorithm.DPUKernelRun(params, channels, 10, {{10, 20, 30, 40}}), HCCL_SUCCESS);
    EXPECT_FALSE(g_calls.empty());
    g_calls.clear();
    channels[20][0].remoteCclMem.size = remote20.size();
    for (u32 remoteRank : {20U, 40U}) {
        channels[remoteRank][0].remoteCclMem.size = 0;
        EXPECT_EQ(algorithm.DPUKernelRun(params, channels, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);
        EXPECT_TRUE(g_calls.empty());
        g_calls.clear();
        channels[remoteRank][0].remoteCclMem.size = 64;
    }
    params.buffInfo.hcclBuff.size = 63;
    EXPECT_EQ(algorithm.DPUKernelRun(params, channels, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);
    EXPECT_TRUE(g_calls.empty());
}

TEST_F(DpuWriteReduceTest, NhrScratchRejectsRepeatStrideAndBaseOverflow)
{
    std::array<u8, 64> localCcl{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    params.dataType = HCCL_DATA_TYPE_INT8;
    params.allRankSliceSize = {UINT64_MAX / 4 + 1, 0, 0, 0};
    params.allRankProcessedDataCount = params.allRankSliceSize;
    params.allRankDispls = {0, 0, 0, 0};
    EXPECT_EQ(ValidateReduceScatterNhrParams(params, 4), HCCL_E_PARA);

    params = MakeNhrParams(localCcl.data());
    params.buffInfo.hcclBuff.size = UINT64_MAX;
    params.buffInfo.hcclBuffBaseOff = UINT64_MAX - 31;
    params.allRankDispls = {0, 0, 0, 0};
    EXPECT_EQ(ValidateReduceScatterNhrParams(params, 4), HCCL_E_PARA);

    params = MakeNhrParams(localCcl.data());
    params.dataType = HCCL_DATA_TYPE_INT8;
    params.repeatNum = 3;
    params.buffInfo.hcclBuff.size = UINT64_MAX;
    params.buffInfo.hcclBuffBaseOff = 0;
    params.allRankSliceSize = {UINT64_MAX / 8 + 1, 0, 0, 0};
    params.allRankProcessedDataCount = params.allRankSliceSize;
    params.allRankDispls = {0, 0, 0, 0};
    EXPECT_EQ(ValidateReduceScatterNhrParams(params, 4), HCCL_E_PARA);
    params = MakeNhrParams(localCcl.data());
    params.allRankDispls[0] = UINT64_MAX - 3;
    EXPECT_EQ(ValidateReduceScatterNhrParams(params, 4), HCCL_E_PARA);
    params = MakeNhrParams(localCcl.data());
    params.buffInfo.hcclBuffBaseOff = UINT64_MAX - 7;
    params.allRankDispls[0] = 8;
    EXPECT_EQ(ValidateReduceScatterNhrParams(params, 4), HCCL_E_PARA);
    EXPECT_TRUE(g_calls.empty());
}

TEST_F(DpuWriteReduceTest, NhrTemplateSkipsZeroSlicesAndSingleRankWithoutChannels)
{
    std::array<u8, 64> localCcl{};
    TemplateDataParams zeroParams = MakeNhrParams(localCcl.data());
    zeroParams.allRankSliceSize = {0, 0, 0, 0};
    zeroParams.allRankProcessedDataCount = {0, 0, 0, 0};
    InsTempReduceScatterNhrDpu algorithm;
    EXPECT_EQ(algorithm.DPUKernelRun(zeroParams, {}, 10, {{10, 20, 30, 40}}), HCCL_SUCCESS);
    EXPECT_TRUE(g_calls.empty());

    auto noRepeat = MakeNhrParams(nullptr);
    noRepeat.repeatNum = 0;
    EXPECT_EQ(algorithm.DPUKernelRun(noRepeat, {}, 10, {{10, 20, 30, 40}}), HCCL_SUCCESS);
    EXPECT_TRUE(g_calls.empty());

    TemplateDataParams singleRank{};
    singleRank.buffInfo.hcclBuff = {HCCL_MEM_TYPE_HOST, localCcl.data(), localCcl.size()};
    singleRank.repeatNum = 1;
    singleRank.dataType = HCCL_DATA_TYPE_INT32;
    singleRank.reduceOp = HCCL_REDUCE_MAX;
    singleRank.allRankSliceSize = {16};
    singleRank.allRankDispls = {0};
    singleRank.allRankProcessedDataCount = {4};
    EXPECT_EQ(algorithm.DPUKernelRun(singleRank, {}, 10, {{10}}), HCCL_SUCCESS);
    EXPECT_TRUE(g_calls.empty());
}

TEST_F(DpuWriteReduceTest, NhrTemplateRejectsInvalidMetadataAndChannelsBeforeTransfer)
{
    std::array<u8, 128> localCcl{};
    std::array<u8, 128> remote{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    std::map<u32, std::vector<ChannelInfo>> channels;
    channels[20] = {MakeDpuChannel(0x1020, remote.data())};
    channels[30] = {MakeDpuChannel(0x1030, remote.data())};
    channels[40] = {MakeDpuChannel(0x1040, remote.data())};
    InsTempReduceScatterNhrDpu algorithm;

    TemplateDataParams badVectors = params;
    badVectors.allRankDispls.pop_back();
    EXPECT_EQ(algorithm.DPUKernelRun(badVectors, channels, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);

    TemplateDataParams badCount = params;
    badCount.allRankProcessedDataCount[1] = 3;
    EXPECT_EQ(algorithm.DPUKernelRun(badCount, channels, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);
    badCount = params;
    badCount.allRankProcessedDataCount[3] = 1;
    EXPECT_EQ(algorithm.DPUKernelRun(badCount, channels, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);

    auto missingChannel = channels;
    missingChannel.erase(40);
    EXPECT_EQ(algorithm.DPUKernelRun(params, missingChannel, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);

    auto emptyChannel = channels;
    emptyChannel[20].clear();
    EXPECT_EQ(algorithm.DPUKernelRun(params, emptyChannel, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);

    auto nullRemote = channels;
    nullRemote[40][0].remoteCclMem.addr = nullptr;
    EXPECT_EQ(algorithm.DPUKernelRun(params, nullRemote, 10, {{10, 20, 30, 40}}), HCCL_E_PTR);

    auto shortRemote = channels;
    shortRemote[40][0].remoteCclMem.size = 16;
    EXPECT_EQ(algorithm.DPUKernelRun(params, shortRemote, 10, {{10, 20, 30, 40}}), HCCL_E_PARA);
    EXPECT_TRUE(g_calls.empty());
}

TEST_F(DpuWriteReduceTest, NhrTemplateStopsAtFailedStepAndDoesNotRunLaterSteps)
{
    std::array<u8, 128> localCcl{};
    std::array<u8, 128> remote20{};
    std::array<u8, 128> remote30{};
    std::array<u8, 128> remote40{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    std::map<u32, std::vector<ChannelInfo>> channels;
    channels[20] = {MakeDpuChannel(0x1020, remote20.data())};
    channels[30] = {MakeDpuChannel(0x1030, remote30.data())};
    channels[40] = {MakeDpuChannel(0x1040, remote40.data())};
    InjectFailure(StubApi::WRITE_REDUCE);

    InsTempReduceScatterNhrDpu algorithm;
    EXPECT_EQ(algorithm.DPUKernelRun(params, channels, 10, {{10, 20, 30, 40}}), HCCL_E_INTERNAL);
    EXPECT_EQ(
        std::count_if(
            g_calls.begin(), g_calls.end(),
            [](const StubCall& call) {
                return call.api == StubApi::WRITE_REDUCE;
            }),
        1);
}

} // namespace wrapper_ut
