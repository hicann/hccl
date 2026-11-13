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
#include "dpu_alg_nhr_opt_wrapper.h"

namespace {

std::vector<nhr_ut::CopyCall> g_copyCalls;
nhr_ut::BatchModeStats g_batchModeStats;
nhr_ut::CostCapture g_costCapture;
std::vector<nhr_ut::NotifyRecordCall> g_notifyRecordCalls;
std::vector<nhr_ut::NotifyWaitCall> g_notifyWaitCalls;
std::vector<nhr_ut::DpuTransferCall> g_dpuTransferCalls;
std::vector<nhr_ut::HandshakeEvent> g_handshakeEvents;

s32 g_batchModeStartResult = 0;
s32 g_batchModeEndResult = 0;
s32 g_threadSyncResult = 0;
s32 g_sendRequestResult = 0;
s32 g_waitResponseResult = 0;
s32 g_notifyRecordResult = 0;
s32 g_notifyWaitResult = 0;
s32 g_dpuTransferResult = 0;
u32 g_sendMsgId = 1;
u32 g_waitResponseMsgId = 1;

} // namespace

namespace nhr_ut {

std::vector<CopyCall>& CopyCalls() { return g_copyCalls; }

BatchModeStats& BatchModeCounts() { return g_batchModeStats; }

CostCapture& CostCalls() { return g_costCapture; }

std::vector<NotifyRecordCall>& NotifyRecordCalls() { return g_notifyRecordCalls; }

std::vector<NotifyWaitCall>& NotifyWaitCalls() { return g_notifyWaitCalls; }

std::vector<DpuTransferCall>& DpuTransferCalls() { return g_dpuTransferCalls; }

std::vector<HandshakeEvent>& HandshakeEvents() { return g_handshakeEvents; }

void SetBatchModeStartResult(s32 result) { g_batchModeStartResult = result; }

void SetBatchModeEndResult(s32 result) { g_batchModeEndResult = result; }

void SetThreadSyncResult(s32 result) { g_threadSyncResult = result; }

void SetSendRequestResult(s32 result) { g_sendRequestResult = result; }

void SetWaitResponseResult(s32 result) { g_waitResponseResult = result; }

void SetSendMsgId(u32 msgId) { g_sendMsgId = msgId; }

void SetWaitResponseMsgId(u32 msgId) { g_waitResponseMsgId = msgId; }

void SetNotifyRecordResult(s32 result) { g_notifyRecordResult = result; }

void SetNotifyWaitResult(s32 result) { g_notifyWaitResult = result; }

void SetDpuTransferResult(s32 result) { g_dpuTransferResult = result; }

void ResetStubState()
{
    g_copyCalls.clear();
    g_batchModeStats = BatchModeStats{};
    g_costCapture = CostCapture{};
    g_notifyRecordCalls.clear();
    g_notifyWaitCalls.clear();
    g_dpuTransferCalls.clear();
    g_handshakeEvents.clear();
    g_batchModeStartResult = 0;
    g_batchModeEndResult = 0;
    g_threadSyncResult = 0;
    g_sendRequestResult = 0;
    g_waitResponseResult = 0;
    g_notifyRecordResult = 0;
    g_notifyWaitResult = 0;
    g_dpuTransferResult = 0;
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
    nhr_ut::HandshakeEvents().push_back(nhr_ut::HandshakeEvent::kLocalCopy);
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

// CalcCostCoeff 依赖的 cost_model.cc 未链接进本 UT，相关三路 cost 计算在本文件后部给出可观测的桩实现。
// ValidateReduceScatterNhrParams 只做参数校验、无下发行为，这里保持空桩，
// 与旧版 AICPU_COMPILE 编译期裁剪的链接语义一致。
// RunNHR 的 DPU 下发路径由下方 DpuBatchWriteReduceTransferNHR 桩记录（DpuTransferCalls /
// HandshakeEvents），供 isDpuThreadEnabled 分支断言下发次数、线程句柄透传与失败返回值传播。
HcclResult ValidateReduceScatterNhrParams(const TemplateDataParams& tempAlgParams, u32 rankSize)
{
    (void)tempAlgParams;
    (void)rankSize;
    return HCCL_SUCCESS;
}

// 记录每次 write-reduce 下发的线程句柄与 repeat 下标，并通过 g_dpuTransferResult 注入返回值，
// 使 RunNHR 的下发次数、线程透传与失败传播可在 UT 中断言。
HcclResult DpuBatchWriteReduceTransferNHR(
    const AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const TemplateDataParams& tempAlgParam, u32 repeat, u32 templateRankSize, ThreadHandle thread,
    bool isDrainSupported)
{
    (void)stepInfo;
    (void)channels;
    (void)tempAlgParam;
    (void)templateRankSize;
    (void)isDrainSupported;
    nhr_ut::DpuTransferCall call;
    call.thread = thread;
    call.repeat = repeat;
    nhr_ut::DpuTransferCalls().push_back(call);
    nhr_ut::HandshakeEvents().push_back(nhr_ut::HandshakeEvent::kDpuTransfer);
    return static_cast<HcclResult>(g_dpuTransferResult);
}

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

// 记录 (thread, dstThread, notifyIdx)，供 isDpuThreadEnabled 分支断言握手顺序与句柄透传。
int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    nhr_ut::NotifyRecordCall call;
    call.thread = thread;
    call.dstThread = dstThread;
    call.notifyIdx = dstNotifyIdx;
    nhr_ut::NotifyRecordCalls().push_back(call);
    nhr_ut::HandshakeEvents().push_back(nhr_ut::HandshakeEvent::kNotifyRecord);
    return g_notifyRecordResult;
}

int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
{
    nhr_ut::NotifyWaitCall call;
    call.thread = thread;
    call.notifyIdx = notifyIdx;
    call.timeOut = timeOut;
    nhr_ut::NotifyWaitCalls().push_back(call);
    nhr_ut::HandshakeEvents().push_back(nhr_ut::HandshakeEvent::kNotifyWait);
    return g_notifyWaitResult;
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
