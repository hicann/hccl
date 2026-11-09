/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOSTDPU_WRAPPER_TEST_COMMON_H
#define HOSTDPU_WRAPPER_TEST_COMMON_H
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <tuple>
#include "exec_timeout_manager.h"
#include <gtest/gtest.h>
#include "dpu_alg_nhr_opt_wrapper.h"
#include "ins_temp_reduce_scatter_nhr_dpu.h"
namespace wrapper_ut {
using namespace ops_hccl;
enum class StubApi { RECORD, WAIT, WRITE, WRITE_REDUCE, DRAIN, FENCE };

struct StubCall {
    StubApi api;
    ChannelHandle channel;
    void* dst;
    const void* src;
    u64 count;
    HcommDataType dataType;
    HcommReduceOp reduceOp;
    u32 notifyIdx;
    u32 timeout = 0;
};

inline std::vector<StubCall> g_calls;
inline StubApi g_failureApi = StubApi::FENCE;
inline u32 g_failureOccurrence = 0;
inline HcclResult g_failureCode = HCCL_SUCCESS;

// host 侧 KernelRun 路径的调用捕获：host 原语不进入 g_calls，避免影响 DPU write-reduce 的时序断言。
inline u32 g_batchModeStartCount = 0;
inline u32 g_batchModeEndCount = 0;
inline u32 g_threadSyncCount = 0;
inline u32 g_localCopyCount = 0;
inline u64 g_sendRequestShmemHandle = 0;
inline std::string g_sendRequestTag;
inline std::vector<char> g_sendRequestPayload;
inline u32 g_sendMsgId = 0x5A5A; // HcommSendRequest 返回、HcommWaitResponse 回填的 msgId

// CalcRes 通道请求桩的输入/输出捕获。
inline u32 g_channelRequestCallCount = 0;
inline std::vector<std::vector<u32>> g_channelRequestSubCommRanks;
inline std::vector<HcclChannelDesc> g_channelRequestDescs;
inline HcclResult g_channelRequestCode = HCCL_SUCCESS;

// host 原语失败注入：occurrence 为 0 表示不注入，否则在第 N 次调用返回对应错误码。
inline HcclResult g_localCopyCode = HCCL_SUCCESS;
inline u32 g_localCopyFailOccurrence = 0;
inline HcclResult g_batchModeStartCode = HCCL_E_INTERNAL;
inline u32 g_batchModeStartFailOccurrence = 0;

// CalcCostCoeff 的三路 cost 桩：记录入参并返回固定值，便于断言分支与参数传递。
inline float g_meshA = 1.5f;
inline float g_localCopyB = 2.5f;
inline float g_dpuLatencyC = 3.5f;
inline float g_meshN = 0.0f;
inline CommTopo g_meshNetType = CommTopo::COMM_TOPO_1DMESH;
inline int g_meshPortNum = -1;
inline u32 g_meshRankSize = 0;
inline EngineType g_meshEngine = EngineType::AICPU;
inline float g_localCopyN = 0.0f;
inline EngineType g_localCopyEngine = EngineType::AICPU;
inline int g_dpuStepNum = 0;
inline int g_dpuSyncNum = 0;
inline int g_dpuChannelNum = 0;
inline int g_dpuSndRcvNum = 0;

inline int32_t RecordCall(const StubCall& call)
{
    g_calls.push_back(call);
    const u32 occurrence
        = static_cast<u32>(std::count_if(g_calls.begin(), g_calls.end(), [&call](const StubCall& recorded) {
              return recorded.api == call.api;
          }));
    if (call.api == g_failureApi && occurrence == g_failureOccurrence) {
        return static_cast<int32_t>(g_failureCode);
    }
    return static_cast<int32_t>(HCCL_SUCCESS);
}

inline void InjectFailure(StubApi api, u32 occurrence = 1)
{
    g_failureApi = api;
    g_failureOccurrence = occurrence;
    g_failureCode = HCCL_E_INTERNAL;
}

inline ChannelInfo MakeChannel(ChannelHandle handle)
{
    ChannelInfo channel{};
    channel.isValid = true;
    channel.handle = handle;
    channel.locationType = EndpointLocType::ENDPOINT_LOC_TYPE_HOST;
    channel.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    return channel;
}

inline ChannelInfo MakeDpuChannel(ChannelHandle handle, void* remoteCclAddress)
{
    ChannelInfo channel = MakeChannel(handle);
    channel.remoteCclMem = {HCCL_MEM_TYPE_HOST, remoteCclAddress, 4096};
    return channel;
}

inline TemplateDataParams MakeNhrParams(void* localCclAddress)
{
    TemplateDataParams params{};
    params.buffInfo.hcclBuff = {HCCL_MEM_TYPE_HOST, localCclAddress, 4096};
    params.buffInfo.hcclBuffBaseOff = 4;
    params.repeatNum = 2;
    params.outputRepeatStride = 64;
    params.dataType = HCCL_DATA_TYPE_FP16;
    params.reduceOp = HCCL_REDUCE_SUM;
    params.allRankSliceSize = {8, 8, 8, 0};
    params.allRankDispls = {0, 8, 20, 28};
    params.allRankProcessedDataCount = {4, 4, 4, 0};
    return params;
}

inline void
AddSendSlice(DpuTransferCtx& transfer, void* src, u64 srcOffset, void* dst, u64 dstOffset, u64 size, u64 count)
{
    transfer.txSrcSlices.emplace_back(src, srcOffset, size, count);
    transfer.txDstSlices.emplace_back(dst, dstOffset, size, count);
}

inline void AddRecvSlice(DpuTransferCtx& transfer, void* src, void* dst, u64 size, u64 count)
{
    transfer.rxSrcSlices.emplace_back(src, 0, size, count);
    transfer.rxDstSlices.emplace_back(dst, 0, size, count);
}

inline void FillStepSliceInfo(StepSliceInfo& info, u64 base)
{
    info.stepCount = {{base + 1, base + 2}, {base + 3}};
    info.stepSliceSize = {{base + 4}, {base + 5, base + 6}};
    info.stepInputSliceStride = {base + 7, base + 8};
    info.stepOutputSliceStride = {base + 9};
    info.inputOmniPipeSliceStride = {{base + 10}, {base + 11}};
    info.outputOmniPipeSliceStride = {{base + 12, base + 13}};
}

inline void ExpectStepSliceInfoEqual(const StepSliceInfo& actual, const StepSliceInfo& expected)
{
    EXPECT_EQ(actual.stepCount, expected.stepCount);
    EXPECT_EQ(actual.stepSliceSize, expected.stepSliceSize);
    EXPECT_EQ(actual.stepInputSliceStride, expected.stepInputSliceStride);
    EXPECT_EQ(actual.stepOutputSliceStride, expected.stepOutputSliceStride);
    EXPECT_EQ(actual.inputOmniPipeSliceStride, expected.inputOmniPipeSliceStride);
    EXPECT_EQ(actual.outputOmniPipeSliceStride, expected.outputOmniPipeSliceStride);
}

inline void FillTemplateDataParams(TemplateDataParams& params, HcclReduceOp reduceOp)
{
    params.buffInfo.inputPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x1100));
    params.buffInfo.outputPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x2200));
    params.buffInfo.hcclBuff = {HCCL_MEM_TYPE_HOST, reinterpret_cast<void*>(static_cast<uintptr_t>(0x3300)), 0x4400};
    params.buffInfo.inBuffType = BufferType::INPUT;
    params.buffInfo.outBuffType = BufferType::OUTPUT;
    params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    params.buffInfo.inputSize = 101;
    params.buffInfo.outputSize = 102;
    params.buffInfo.hcclBuffSize = 103;
    params.buffInfo.inBuffBaseOff = 104;
    params.buffInfo.outBuffBaseOff = 105;
    params.buffInfo.hcclBuffBaseOff = 106;
    params.count = 201;
    params.sliceSize = 202;
    params.inputSliceStride = 203;
    params.outputSliceStride = 204;
    params.repeatNum = 205;
    params.inputRepeatStride = 206;
    params.outputRepeatStride = 207;
    params.tailSize = 208;
    params.enableRemoteMemAccess = true;
    params.allRankSliceSize = {301, 302};
    params.allRankDispls = {303, 304};
    params.sendCounts = {305, 306};
    params.recvCounts = {307};
    params.sdispls = {308};
    params.rdispls = {309, 310};
    params.peerRdispls = {311};
    params.allRankProcessedDataCount = {312, 313};
    params.root = 401;
    params.dataType = HCCL_DATA_TYPE_FP16;
    FillStepSliceInfo(params.stepSliceInfo, 500);
    params.opType = BatchSendRecvOpType::SEND;
    FillStepSliceInfo(params.omniReadDstStepSliceInfo, 600);
    params.omniLastStepRead_ = true;
    params.localCopyFlag = 701;
    params.reduceOp = reduceOp;
}

inline void ExpectTemplateDataParamsEqual(const TemplateDataParams& actual, const TemplateDataParams& expected)
{
    EXPECT_EQ(actual.buffInfo.inputPtr, expected.buffInfo.inputPtr);
    EXPECT_EQ(actual.buffInfo.outputPtr, expected.buffInfo.outputPtr);
    EXPECT_EQ(actual.buffInfo.hcclBuff.type, expected.buffInfo.hcclBuff.type);
    EXPECT_EQ(actual.buffInfo.hcclBuff.addr, expected.buffInfo.hcclBuff.addr);
    EXPECT_EQ(actual.buffInfo.hcclBuff.size, expected.buffInfo.hcclBuff.size);
    EXPECT_EQ(actual.buffInfo.inBuffType, expected.buffInfo.inBuffType);
    EXPECT_EQ(actual.buffInfo.outBuffType, expected.buffInfo.outBuffType);
    EXPECT_EQ(actual.buffInfo.hcclBuffType, expected.buffInfo.hcclBuffType);
    EXPECT_EQ(actual.buffInfo.inputSize, expected.buffInfo.inputSize);
    EXPECT_EQ(actual.buffInfo.outputSize, expected.buffInfo.outputSize);
    EXPECT_EQ(actual.buffInfo.hcclBuffSize, expected.buffInfo.hcclBuffSize);
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
    EXPECT_EQ(actual.allRankSliceSize, expected.allRankSliceSize);
    EXPECT_EQ(actual.allRankDispls, expected.allRankDispls);
    EXPECT_EQ(actual.sendCounts, expected.sendCounts);
    EXPECT_EQ(actual.recvCounts, expected.recvCounts);
    EXPECT_EQ(actual.sdispls, expected.sdispls);
    EXPECT_EQ(actual.rdispls, expected.rdispls);
    EXPECT_EQ(actual.peerRdispls, expected.peerRdispls);
    EXPECT_EQ(actual.allRankProcessedDataCount, expected.allRankProcessedDataCount);
    EXPECT_EQ(actual.root, expected.root);
    EXPECT_EQ(actual.dataType, expected.dataType);
    EXPECT_EQ(actual.reduceOp, expected.reduceOp);
    ExpectStepSliceInfoEqual(actual.stepSliceInfo, expected.stepSliceInfo);
    EXPECT_EQ(actual.opType, expected.opType);
    ExpectStepSliceInfoEqual(actual.omniReadDstStepSliceInfo, expected.omniReadDstStepSliceInfo);
    EXPECT_EQ(actual.omniLastStepRead_, expected.omniLastStepRead_);
    EXPECT_EQ(actual.localCopyFlag, expected.localCopyFlag);
}

inline void ResetHostPathCapture()
{
    g_batchModeStartCount = 0;
    g_batchModeEndCount = 0;
    g_threadSyncCount = 0;
    g_localCopyCount = 0;
    g_sendRequestShmemHandle = 0;
    g_sendRequestTag.clear();
    g_sendRequestPayload.clear();
    g_channelRequestCallCount = 0;
    g_channelRequestSubCommRanks.clear();
    g_channelRequestDescs.clear();
    g_channelRequestCode = HCCL_SUCCESS;
    g_localCopyCode = HCCL_SUCCESS;
    g_localCopyFailOccurrence = 0;
    g_batchModeStartCode = HCCL_E_INTERNAL;
    g_batchModeStartFailOccurrence = 0;
    g_meshA = 1.5f;
    g_localCopyB = 2.5f;
    g_dpuLatencyC = 3.5f;
    g_meshN = 0.0f;
    g_meshNetType = CommTopo::COMM_TOPO_1DMESH;
    g_meshPortNum = -1;
    g_meshRankSize = 0;
    g_meshEngine = EngineType::AICPU;
    g_localCopyN = 0.0f;
    g_localCopyEngine = EngineType::AICPU;
    g_dpuStepNum = 0;
    g_dpuSyncNum = 0;
    g_dpuChannelNum = 0;
    g_dpuSndRcvNum = 0;
}

class DpuWriteReduceTest : public testing::Test {
protected:
    void SetUp() override
    {
        previousTimeout_ = ExecTimeoutManager::Instance().GetExecTimeout();
        ExecTimeoutManager::Instance().SetExecTimeout(37);
        g_calls.clear();
        ResetHostPathCapture();
        g_failureApi = StubApi::FENCE;
        g_failureOccurrence = 0;
        g_failureCode = HCCL_SUCCESS;
    }
    void TearDown() override
    {
        ExecTimeoutManager::Instance().SetExecTimeout(previousTimeout_);
        g_calls.clear();
        ResetHostPathCapture();
        g_failureOccurrence = 0;
        g_failureCode = HCCL_SUCCESS;
    }

private:
    u32 previousTimeout_ = 0;
};

} // namespace wrapper_ut
#endif
