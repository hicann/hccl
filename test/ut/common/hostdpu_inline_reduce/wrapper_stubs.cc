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
using namespace wrapper_ut;
extern "C" {

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    return RecordCall(
        {StubApi::RECORD, channel, nullptr, nullptr, 0, HCOMM_DATA_TYPE_RESERVED, HCOMM_REDUCE_RESERVED,
         remoteNotifyIdx});
}

int32_t HcommChannelNotifyWaitOnThread(ThreadHandle, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout)
{
    return RecordCall(
        {StubApi::WAIT, channel, nullptr, nullptr, 0, HCOMM_DATA_TYPE_RESERVED, HCOMM_REDUCE_RESERVED, localNotifyIdx,
         timeout});
}

int32_t HcommWriteWithNotifyNbiOnThread(
    ThreadHandle, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    return RecordCall(
        {StubApi::WRITE, channel, dst, src, len, HCOMM_DATA_TYPE_RESERVED, HCOMM_REDUCE_RESERVED, remoteNotifyIdx});
}

int32_t HcommWriteReduceWithNotifyOnThread(
    ThreadHandle, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    return RecordCall({StubApi::WRITE_REDUCE, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx});
}

int32_t HcommFenceOnThread(ThreadHandle)
{
    return RecordCall({StubApi::FENCE, 0, nullptr, nullptr, 0, HCOMM_DATA_TYPE_RESERVED, HCOMM_REDUCE_RESERVED, 0});
}

int32_t HcommThreadNotifyRecordOnThread(ThreadHandle, ThreadHandle, uint32_t) { return HCCL_SUCCESS; }

int32_t HcommThreadNotifyWaitOnThread(ThreadHandle, uint32_t, uint32_t) { return HCCL_SUCCESS; }

} // extern "C"

namespace ops_hccl {

HcclResult HcommChannelDrainOnThreadWithCompat(ThreadHandle, ChannelHandle channel, bool)
{
    return static_cast<HcclResult>(
        RecordCall({StubApi::DRAIN, channel, nullptr, nullptr, 0, HCOMM_DATA_TYPE_RESERVED, HCOMM_REDUCE_RESERVED, 0}));
}

} // namespace ops_hccl

#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "cost_model.h"
#include "hcomm_primitives_dl.h"

// The DPU entry tests below never enter these host-only paths, while the
// KernelRun hand-off test drives them explicitly; capture instead of failing.
extern "C" {

int32_t HcommBatchModeStart(const char*)
{
    ++g_batchModeStartCount;
    if (g_batchModeStartFailOccurrence != 0 && g_batchModeStartCount == g_batchModeStartFailOccurrence) {
        return static_cast<int32_t>(g_batchModeStartCode);
    }
    return HCCL_SUCCESS;
}

int32_t HcommBatchModeEnd(const char*)
{
    ++g_batchModeEndCount;
    return HCCL_SUCCESS;
}

int32_t HcommThreadSynchronize(ThreadHandle)
{
    ++g_threadSyncCount;
    return HCCL_SUCCESS;
}

int32_t HcommSendRequest(uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId)
{
    g_sendRequestShmemHandle = handle;
    g_sendRequestTag.assign(msgTag == nullptr ? "" : msgTag);
    if (src != nullptr && sizeByte > 0) {
        const char* payload = static_cast<const char*>(src);
        g_sendRequestPayload.assign(payload, payload + sizeByte);
    } else {
        g_sendRequestPayload.clear();
    }
    if (msgId != nullptr) {
        *msgId = g_sendMsgId;
    }
    return HCCL_SUCCESS;
}

int32_t HcommWaitResponse(uint64_t, void*, size_t, uint32_t* msgId)
{
    if (msgId != nullptr) {
        *msgId = g_sendMsgId;
    }
    return HCCL_SUCCESS;
}

} // extern "C"

namespace ops_hccl {

HcclResult LocalCopy(const ThreadHandle&, const DataSlice&, const DataSlice&)
{
    ++g_localCopyCount;
    if (g_localCopyFailOccurrence != 0 && g_localCopyCount == g_localCopyFailOccurrence) {
        return g_localCopyCode;
    }
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNhrHighestHostRoce(
    HcclComm, const OpParam&, const TopoInfoWithNetLayerDetails*, const std::vector<std::vector<u32>>& subCommRanks,
    std::vector<HcclChannelDesc>& channels)
{
    ++g_channelRequestCallCount;
    g_channelRequestSubCommRanks = subCommRanks;
    if (g_channelRequestCode != HCCL_SUCCESS) {
        return g_channelRequestCode;
    }
    channels = g_channelRequestDescs;
    return HCCL_SUCCESS;
}

// CalcCostCoeff 只依赖这三路 cost 计算；cost_model.cc 未链接进本 UT，这里给出可观测的桩实现。
CostModelManager* CostModelManager::Global()
{
    static CostModelManager instance;
    return &instance;
}

CostModelManager::CostModelManager() {}

void CostModelManager::CalcMeshParam(
    float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool, EngineType engine)
{
    g_meshN = n;
    g_meshNetType = netType;
    g_meshPortNum = portNum;
    g_meshRankSize = rankSize;
    g_meshEngine = engine;
    A = g_meshA;
}

void CostModelManager::CalcLocalCopyParams(float n, EngineType scene, float& B)
{
    g_localCopyN = n;
    g_localCopyEngine = scene;
    B = g_localCopyB;
}

void CostModelManager::CalcDpuLatencyParams(int stepNum, int syncNum, int channelNum, int sndRcvNum, float& C)
{
    g_dpuStepNum = stepNum;
    g_dpuSyncNum = syncNum;
    g_dpuChannelNum = channelNum;
    g_dpuSndRcvNum = sndRcvNum;
    C = g_dpuLatencyC;
}

} // namespace ops_hccl
