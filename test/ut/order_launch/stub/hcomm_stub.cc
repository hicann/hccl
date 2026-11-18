/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_stub.h"

#include "acl/acl_rt.h"
#include "hccl_res_dl.h"
#include "hcomm_primitives_dl.h"
#include "rt_external.h"

namespace OrderLaunchUtStub {
std::vector<AddToModelCall> g_addCalls;
int32_t g_notifyWaitCalls = 0;
int32_t g_notifyRecordCalls = 0;
int32_t g_acqRet = 0;
uint64_t g_acqThread = 0;
uint64_t g_deviceAcqThread = 0;
bool g_supportDedicated = true;
int32_t g_captureRet = 0;    // ACL_SUCCESS
int32_t g_captureStatus = 0; // ACL_MODEL_RI_CAPTURE_STATUS_NONE
void* g_rtModel = nullptr;
int32_t g_threadResRet = 0;
void* g_threadResStream = nullptr;
int32_t g_addRet = 0; // RT_ERROR_NONE
int32_t g_notifyRecordRet = 0;
int32_t g_notifyWaitRet = 0;
int32_t g_exportRet = 0;
uint32_t g_callSeq = 0;
uint32_t g_addSeq = 0;
uint32_t g_firstNotifyRecordSeq = 0;
uint32_t g_lastNotifyWaitSeq = 0;

void Reset()
{
    g_addCalls.clear();
    g_notifyWaitCalls = 0;
    g_notifyRecordCalls = 0;
    g_acqRet = 0;
    g_acqThread = 0x1234;            // 默认非零句柄
    g_deviceAcqThread = g_acqThread; // DEVICE 类默认同值
    g_supportDedicated = true;
    g_captureRet = 0; // ACL_SUCCESS
    g_captureStatus = static_cast<int32_t>(aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE);
    g_rtModel = reinterpret_cast<void*>(0x5678);
    g_threadResRet = 0;
    g_threadResStream = reinterpret_cast<void*>(0x9abc);
    g_addRet = 0;
    g_notifyRecordRet = 0;
    g_notifyWaitRet = 0;
    g_exportRet = 0;
    g_callSeq = 0;
    g_addSeq = 0;
    g_firstNotifyRecordSeq = 0;
    g_lastNotifyWaitSeq = 0;
}
} // namespace OrderLaunchUtStub

using namespace OrderLaunchUtStub;

// 强符号覆写生产弱默认，返回值由桩旋钮控制
extern "C" {

HcclResult HcclDedicatedThreadAcquire(
    HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread)
{
    (void)comm;
    (void)notifyNumPerThread;
    if (thread != nullptr) {
        // DEVICE 类句柄独立旋钮
        *thread = (useType == HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE) ? g_deviceAcqThread : g_acqThread;
    }
    return static_cast<HcclResult>(g_acqRet);
}

bool HcommIsSupportHcclDedicatedThreadAcquire(void) { return g_supportDedicated; }

HcclResult HcclThreadExportToCommEngine(
    HcclComm comm, uint32_t threadNum, const ThreadHandle* threads, CommEngine dstCommEngine,
    ThreadHandle* exportedThreads)
{
    (void)comm;
    (void)dstCommEngine;
    if (g_exportRet != 0) {
        return static_cast<HcclResult>(g_exportRet);
    }
    if (threads == nullptr || exportedThreads == nullptr) {
        return HCCL_E_PTR;
    }
    for (uint32_t i = 0; i < threadNum; ++i) {
        exportedThreads[i] = threads[i];
    }
    return HCCL_SUCCESS;
}

int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    (void)thread;
    (void)dstThread;
    (void)dstNotifyIdx;
    ++g_notifyRecordCalls;
    uint32_t seq = ++g_callSeq;
    if (g_firstNotifyRecordSeq == 0) {
        g_firstNotifyRecordSeq = seq;
    }
    return g_notifyRecordRet;
}

int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
{
    (void)thread;
    (void)notifyIdx;
    (void)timeOut;
    ++g_notifyWaitCalls;
    g_lastNotifyWaitSeq = ++g_callSeq;
    return g_notifyWaitRet;
}

aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* modelRI)
{
    (void)stream;
    if (status == nullptr || modelRI == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *status = static_cast<aclmdlRICaptureStatus>(g_captureStatus);
    *modelRI = reinterpret_cast<aclmdlRI>(g_rtModel);
    return g_captureRet;
}

rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl)
{
    uint32_t seq = ++g_callSeq;
    if (g_addSeq == 0) {
        g_addSeq = seq;
    }
    g_addCalls.push_back(AddToModelCall{reinterpret_cast<void*>(stm), reinterpret_cast<void*>(captureMdl)});
    return static_cast<rtError_t>(g_addRet);
}
} // extern "C"
