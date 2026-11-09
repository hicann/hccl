/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nhr_test_access.h"

#include "alg_v2_template_base.h"
#include "channel.h"

namespace {

std::vector<nhr_ut::CopyCall> g_copyCalls;
nhr_ut::BatchModeStats g_batchModeStats;
nhr_ut::CostCapture g_costCapture;

s32 g_batchModeStartResult = 0;
s32 g_batchModeEndResult = 0;
s32 g_threadSyncResult = 0;
s32 g_sendRequestResult = 0;
s32 g_waitResponseResult = 0;
u32 g_sendMsgId = 1;
u32 g_waitResponseMsgId = 1;

} // namespace

namespace nhr_ut {

std::vector<CopyCall>& CopyCalls() { return g_copyCalls; }

BatchModeStats& BatchModeCounts() { return g_batchModeStats; }

CostCapture& CostCalls() { return g_costCapture; }

void SetBatchModeStartResult(s32 result) { g_batchModeStartResult = result; }

void SetBatchModeEndResult(s32 result) { g_batchModeEndResult = result; }

void SetThreadSyncResult(s32 result) { g_threadSyncResult = result; }

void SetSendRequestResult(s32 result) { g_sendRequestResult = result; }

void SetWaitResponseResult(s32 result) { g_waitResponseResult = result; }

void SetSendMsgId(u32 msgId) { g_sendMsgId = msgId; }

void SetWaitResponseMsgId(u32 msgId) { g_waitResponseMsgId = msgId; }

void ResetStubState()
{
    g_copyCalls.clear();
    g_batchModeStats = BatchModeStats{};
    g_costCapture = CostCapture{};
    g_batchModeStartResult = 0;
    g_batchModeEndResult = 0;
    g_threadSyncResult = 0;
    g_sendRequestResult = 0;
    g_waitResponseResult = 0;
    g_sendMsgId = 1;
    g_waitResponseMsgId = 1;
}

} // namespace nhr_ut

namespace ops_hccl {

// 记录 host 侧拷贝指令，供寻址断言使用
HcclResult LocalCopy(const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice)
{
    (void)thread;
    nhr_ut::CopyCall call;
    call.srcAddr = srcSlice.addr_;
    call.srcOffset = srcSlice.offset_;
    call.srcSize = srcSlice.size_;
    call.srcCount = srcSlice.count_;
    call.dstAddr = dstSlice.addr_;
    call.dstOffset = dstSlice.offset_;
    call.dstSize = dstSlice.size_;
    call.dstCount = dstSlice.count_;
    nhr_ut::CopyCalls().push_back(call);
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNhrHighestHostRoce(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_SUCCESS;
}

// CalcCostCoeff 依赖的三路 cost 计算：cost_model.cc 未链接进本 UT，这里给出可观测的桩实现。
CostModelManager* CostModelManager::Global()
{
    static CostModelManager instance;
    return &instance;
}

CostModelManager::CostModelManager() {}

void CostModelManager::CalcMeshParam(
    float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod, EngineType engine)
{
    nhr_ut::CostCapture& capture = nhr_ut::CostCalls();
    capture.meshN = n;
    capture.meshNetType = netType;
    capture.meshPortNum = portNum;
    capture.meshRankSize = rankSize;
    capture.meshIsPod = isPod;
    capture.meshEngine = engine;
    A = capture.meshA;
}

void CostModelManager::CalcLocalCopyParams(float n, EngineType scene, float& B)
{
    nhr_ut::CostCapture& capture = nhr_ut::CostCalls();
    capture.localCopyN = n;
    capture.localCopyEngine = scene;
    B = capture.localCopyB;
}

void CostModelManager::CalcDpuLatencyParams(int stepNum, int syncNum, int channelNum, int sndRcvNum, float& C)
{
    nhr_ut::CostCapture& capture = nhr_ut::CostCalls();
    capture.dpuStepNum = stepNum;
    capture.dpuSyncNum = syncNum;
    capture.dpuChannelNum = channelNum;
    capture.dpuSndRcvNum = sndRcvNum;
    C = capture.dpuLatencyC;
}

} // namespace ops_hccl

// 与 hcomm_primitives_dl.h 中 extern "C" 的 DECL_WEAK_FUNC 声明保持同签名
extern "C" {

int32_t HcommBatchModeStart(const char* batchTag)
{
    (void)batchTag;
    nhr_ut::BatchModeCounts().startCount++;
    return g_batchModeStartResult;
}

int32_t HcommBatchModeEnd(const char* batchTag)
{
    (void)batchTag;
    nhr_ut::BatchModeCounts().endCount++;
    return g_batchModeEndResult;
}

int32_t HcommThreadSynchronize(ThreadHandle thread)
{
    (void)thread;
    return g_threadSyncResult;
}

int32_t HcommSendRequest(uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId)
{
    (void)handle;
    (void)msgTag;
    (void)src;
    (void)sizeByte;
    if (msgId != nullptr) {
        *msgId = g_sendMsgId;
    }
    return g_sendRequestResult;
}

int32_t HcommWaitResponse(uint64_t handle, void* dst, size_t sizeByte, uint32_t* msgId)
{
    (void)handle;
    (void)dst;
    (void)sizeByte;
    if (msgId != nullptr) {
        *msgId = g_waitResponseMsgId;
    }
    return g_waitResponseResult;
}

} // extern "C"
