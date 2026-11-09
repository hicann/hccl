/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// DPU 侧没有 OpParam，dataType/reduceOp 全靠任务书过河：序列化契约的往返与截断行为。

#include "nhr_test_access.h"

namespace nhr_ut {
namespace {

    TemplateDataParams MakeFullParams(HcclDataType dataType, HcclReduceOp reduceOp)
    {
        TemplateDataParams params;
        params.buffInfo.inputPtr = reinterpret_cast<void*>(0x1100);
        params.buffInfo.outputPtr = reinterpret_cast<void*>(0x2200);
        params.buffInfo.hcclBuff.type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
        params.buffInfo.hcclBuff.addr = reinterpret_cast<void*>(0x3300);
        params.buffInfo.hcclBuff.size = 8192;
        params.buffInfo.inBuffType = BufferType::INPUT;
        params.buffInfo.outBuffType = BufferType::OUTPUT;
        params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        params.buffInfo.inputSize = 1024;
        params.buffInfo.outputSize = 512;
        params.buffInfo.hcclBuffSize = 4096;
        params.buffInfo.inBuffBaseOff = 11;
        params.buffInfo.outBuffBaseOff = 22;
        params.buffInfo.hcclBuffBaseOff = 33;

        params.count = 64;
        params.sliceSize = 256;
        params.inputSliceStride = 512;
        params.outputSliceStride = 256;
        params.repeatNum = 3;
        params.inputRepeatStride = 2048;
        params.outputRepeatStride = 768;
        params.tailSize = 7;
        params.enableRemoteMemAccess = true;
        params.processedDataCount = 999; // 不参与序列化，DPU 侧不使用
        params.root = 5;
        params.dataType = dataType;
        params.reduceOp = reduceOp;
        params.allRankSliceSize = {256, 0, 128};
        params.allRankDispls = {0, 256, 512};
        params.allRankProcessedDataCount = {64, 0, 32};
        params.sendCounts = {1, 2, 3};
        params.recvCounts = {4, 5, 6};
        params.sdispls = {7, 8, 9};
        params.rdispls = {10, 11, 12};
        params.peerRdispls = {13, 14, 15};
        params.opType = BatchSendRecvOpType::SEND;
        params.omniLastStepRead_ = true;
        params.localCopyFlag = 0x5A;

        params.stepSliceInfo.stepCount = {{1, 2}, {3, 4}};
        params.stepSliceInfo.stepSliceSize = {{5, 6}};
        params.stepSliceInfo.stepInputSliceStride = {7};
        params.stepSliceInfo.stepOutputSliceStride = {8};
        params.stepSliceInfo.inputOmniPipeSliceStride = {{9}, {10}};
        params.stepSliceInfo.outputOmniPipeSliceStride = {{11}};

        params.omniReadDstStepSliceInfo.stepCount = {{12}};
        params.omniReadDstStepSliceInfo.stepSliceSize = {{13}};
        params.omniReadDstStepSliceInfo.stepInputSliceStride = {14};
        params.omniReadDstStepSliceInfo.stepOutputSliceStride = {15};
        params.omniReadDstStepSliceInfo.inputOmniPipeSliceStride = {{16}};
        params.omniReadDstStepSliceInfo.outputOmniPipeSliceStride = {{17}};
        return params;
    }

    void ExpectFullParamsEqual(const TemplateDataParams& actual, const TemplateDataParams& expected)
    {
        EXPECT_EQ(actual.buffInfo.inputPtr, expected.buffInfo.inputPtr);
        EXPECT_EQ(actual.buffInfo.outputPtr, expected.buffInfo.outputPtr);
        EXPECT_EQ(actual.buffInfo.hcclBuff.addr, expected.buffInfo.hcclBuff.addr);
        EXPECT_EQ(actual.buffInfo.hcclBuff.size, expected.buffInfo.hcclBuff.size);
        EXPECT_EQ(actual.buffInfo.inBuffBaseOff, expected.buffInfo.inBuffBaseOff);
        EXPECT_EQ(actual.buffInfo.outBuffBaseOff, expected.buffInfo.outBuffBaseOff);
        EXPECT_EQ(actual.buffInfo.hcclBuffBaseOff, expected.buffInfo.hcclBuffBaseOff);

        EXPECT_EQ(actual.count, expected.count);
        EXPECT_EQ(actual.sliceSize, expected.sliceSize);
        EXPECT_EQ(actual.inputSliceStride, expected.inputSliceStride);
        EXPECT_EQ(actual.outputSliceStride, expected.outputSliceStride);
        EXPECT_EQ(actual.repeatNum, expected.repeatNum);
        EXPECT_EQ(actual.inputRepeatStride, expected.inputRepeatStride);
        EXPECT_EQ(actual.outputRepeatStride, expected.outputRepeatStride);
        EXPECT_EQ(actual.tailSize, expected.tailSize);
        EXPECT_EQ(actual.enableRemoteMemAccess, expected.enableRemoteMemAccess);
        EXPECT_EQ(actual.root, expected.root);
        EXPECT_EQ(actual.dataType, expected.dataType);
        EXPECT_EQ(actual.reduceOp, expected.reduceOp);
        EXPECT_EQ(actual.allRankSliceSize, expected.allRankSliceSize);
        EXPECT_EQ(actual.allRankDispls, expected.allRankDispls);
        EXPECT_EQ(actual.allRankProcessedDataCount, expected.allRankProcessedDataCount);
        EXPECT_EQ(actual.sendCounts, expected.sendCounts);
        EXPECT_EQ(actual.recvCounts, expected.recvCounts);
        EXPECT_EQ(actual.sdispls, expected.sdispls);
        EXPECT_EQ(actual.rdispls, expected.rdispls);
        EXPECT_EQ(actual.peerRdispls, expected.peerRdispls);
        EXPECT_EQ(actual.opType, expected.opType);
        EXPECT_EQ(actual.omniLastStepRead_, expected.omniLastStepRead_);
        EXPECT_EQ(actual.localCopyFlag, expected.localCopyFlag);
        EXPECT_EQ(actual.stepSliceInfo.stepCount, expected.stepSliceInfo.stepCount);
        EXPECT_EQ(actual.stepSliceInfo.stepSliceSize, expected.stepSliceInfo.stepSliceSize);
        EXPECT_EQ(actual.stepSliceInfo.stepInputSliceStride, expected.stepSliceInfo.stepInputSliceStride);
        EXPECT_EQ(actual.stepSliceInfo.stepOutputSliceStride, expected.stepSliceInfo.stepOutputSliceStride);
        EXPECT_EQ(actual.stepSliceInfo.inputOmniPipeSliceStride, expected.stepSliceInfo.inputOmniPipeSliceStride);
        EXPECT_EQ(actual.stepSliceInfo.outputOmniPipeSliceStride, expected.stepSliceInfo.outputOmniPipeSliceStride);
        EXPECT_EQ(actual.omniReadDstStepSliceInfo.stepCount, expected.omniReadDstStepSliceInfo.stepCount);
        EXPECT_EQ(
            actual.omniReadDstStepSliceInfo.inputOmniPipeSliceStride,
            expected.omniReadDstStepSliceInfo.inputOmniPipeSliceStride);
        EXPECT_EQ(
            actual.omniReadDstStepSliceInfo.outputOmniPipeSliceStride,
            expected.omniReadDstStepSliceInfo.outputOmniPipeSliceStride);
    }

    TEST(NhrSerializationTest, TemplateDataParamsRoundTripKeepsAllFields)
    {
        const TemplateDataParams src = MakeFullParams(HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
        std::vector<char> data = src.Serialize();

        TemplateDataParams dst;
        dst.DeSerialize(data);
        ExpectFullParamsEqual(dst, src);
    }

    TEST(NhrSerializationTest, ReduceOpSurvivesAllDataTypeAndOpCombinations)
    {
        const std::vector<HcclDataType> dataTypes = {HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_INT32};
        const std::vector<HcclReduceOp> reduceOps
            = {HCCL_REDUCE_SUM, HCCL_REDUCE_PROD, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN};

        for (HcclDataType dataType : dataTypes) {
            for (HcclReduceOp reduceOp : reduceOps) {
                SCOPED_TRACE(
                    testing::Message() << "dataType=" << static_cast<int>(dataType)
                                       << " reduceOp=" << static_cast<int>(reduceOp));
                const TemplateDataParams src = MakeFullParams(dataType, reduceOp);
                std::vector<char> data = src.Serialize();
                TemplateDataParams dst;
                dst.DeSerialize(data);
                EXPECT_EQ(dst.dataType, dataType);
                EXPECT_EQ(dst.reduceOp, reduceOp);
            }
        }
    }

    TEST(NhrSerializationTest, ProcessedDataCountIsNotPartOfTheWireFormat)
    {
        // processedDataCount 不在序列化清单内：DPU 侧不使用，round-trip 后保持默认值
        const TemplateDataParams src = MakeFullParams(HCCL_DATA_TYPE_FP16, HCCL_REDUCE_MIN);
        ASSERT_EQ(src.processedDataCount, 999u);
        std::vector<char> data = src.Serialize();
        TemplateDataParams dst;
        dst.DeSerialize(data);
        EXPECT_EQ(dst.processedDataCount, 0u);
    }

    TEST(NhrSerializationTest, SerializationIsByteStable)
    {
        const TemplateDataParams src = MakeFullParams(HCCL_DATA_TYPE_INT32, HCCL_REDUCE_PROD);
        const std::vector<char> first = src.Serialize();
        const std::vector<char> second = src.Serialize();
        EXPECT_EQ(first, second);

        TemplateDataParams dst;
        dst.DeSerialize(const_cast<std::vector<char>&>(first));
        EXPECT_EQ(dst.Serialize(), first);
    }

    TEST(NhrSerializationTest, TruncatedTailKeepsReduceOpUntouched)
    {
        // 尾部截断（reduceOp 字段不完整）当前实现不抛异常，读取失败时不改写目标值
        const TemplateDataParams src = MakeFullParams(HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM);
        std::vector<char> data = src.Serialize();
        ASSERT_GT(data.size(), sizeof(HcclReduceOp));
        data.resize(data.size() - sizeof(HcclReduceOp));

        TemplateDataParams dst;
        dst.reduceOp = HCCL_REDUCE_MAX;
        dst.DeSerialize(data);
        EXPECT_EQ(dst.reduceOp, HCCL_REDUCE_MAX);
        EXPECT_EQ(dst.dataType, src.dataType);
        EXPECT_EQ(dst.allRankSliceSize, src.allRankSliceSize);
    }

    TEST(NhrSerializationTest, DpuRunInfoRoundTrip)
    {
        DPURunInfo src;
        src.templateName = "InsTempReduceScatterNhrDpu";
        src.tempAlgParams = MakeFullParams(HCCL_DATA_TYPE_FP16, HCCL_REDUCE_MAX);
        src.myRank = 3;
        src.subCommRanks = {{0, 2, 4, 6}, {1}};
        src.execTimeout = 42;

        ChannelInfo channel;
        channel.isValid = true;
        channel.remoteRank = 2;
        channel.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
        channel.notifyNum = 3;
        channel.handle = 0xABC;
        src.channels[2] = {channel};
        ChannelInfo second = channel;
        second.remoteRank = 0;
        second.handle = 0xDEF;
        src.channels[0] = {second, second};

        std::vector<char> data = src.Serialize();
        DPURunInfo dst;
        dst.DeSerialize(data);

        EXPECT_EQ(dst.templateName, src.templateName);
        EXPECT_EQ(dst.myRank, src.myRank);
        EXPECT_EQ(dst.execTimeout, src.execTimeout);
        EXPECT_EQ(dst.subCommRanks, src.subCommRanks);
        ASSERT_EQ(dst.channels.size(), src.channels.size());
        ASSERT_EQ(dst.channels.count(2), 1u);
        ASSERT_EQ(dst.channels.at(2).size(), 1u);
        EXPECT_EQ(dst.channels.at(2)[0].handle, channel.handle);
        EXPECT_EQ(dst.channels.at(2)[0].remoteRank, channel.remoteRank);
        EXPECT_EQ(dst.channels.at(2)[0].notifyNum, channel.notifyNum);
        EXPECT_EQ(dst.channels.at(2)[0].protocol, channel.protocol);
        EXPECT_EQ(dst.channels.at(2)[0].isValid, channel.isValid);
        ASSERT_EQ(dst.channels.at(0).size(), 2u);
        EXPECT_EQ(dst.channels.at(0)[0].handle, second.handle);
        EXPECT_EQ(dst.channels.at(0)[1].handle, second.handle);
        ExpectFullParamsEqual(dst.tempAlgParams, src.tempAlgParams);
    }

    TEST(NhrSerializationTest, DpuRunInfoDefaultTimeoutSurvivesRoundTrip)
    {
        DPURunInfo src;
        src.templateName = "InsTempReduceScatterNhrDpu";
        src.myRank = 0;
        src.subCommRanks = {{0, 1}};
        std::vector<char> data = src.Serialize();

        DPURunInfo dst;
        dst.execTimeout = 0;
        dst.DeSerialize(data);
        EXPECT_EQ(dst.execTimeout, src.execTimeout);
        EXPECT_EQ(dst.channels.empty(), true);
    }

} // namespace
} // namespace nhr_ut
