/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "hcomm_primitives_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
int32_t (*hcommLocalCopyOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommLocalReduceOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = nullptr;
int32_t (*hcommThreadNotifyRecordOnThreadPtr)(ThreadHandle, ThreadHandle, uint32_t) = nullptr;
int32_t (*hcommThreadNotifyWaitOnThreadPtr)(ThreadHandle, uint32_t, uint32_t) = nullptr;
int32_t (*hcommAclrtNotifyRecordOnThreadPtr)(ThreadHandle, uint64_t) = nullptr;
int32_t (*hcommAclrtNotifyWaitOnThreadPtr)(ThreadHandle, uint64_t, uint32_t) = nullptr;
int32_t (*hcommWriteOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommWriteReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = nullptr;
int32_t (*hcommWriteWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t) = nullptr;
int32_t (*hcommWriteReduceWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp, uint32_t) = nullptr;
int32_t (*hcommReadOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommReadReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = nullptr;
int32_t (*hcommWriteNbiOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommWriteWithNotifyNbiOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t) = nullptr;
int32_t (*hcommReadNbiOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = nullptr;
int32_t (*hcommChannelNotifyRecordOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t) = nullptr;
int32_t (*hcommChannelNotifyWaitOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t, uint32_t) = nullptr;
int32_t (*hcommBatchModeStartPtr)(const char*) = nullptr;
int32_t (*hcommBatchModeEndPtr)(const char*) = nullptr;
int32_t (*hcommAcquireCommPtr)(const char*) = nullptr;
int32_t (*hcommReleaseCommPtr)(const char*) = nullptr;
HcclResult (*hcommSymWinGetPeerPointerPtr)(CommSymWindow, size_t, uint32_t, void**) = nullptr;
int32_t (*hcommThreadSynchronizePtr)(ThreadHandle) = nullptr;
int32_t (*hcommSendRequestPtr)(MsgHandle, const char*, const void*, size_t, uint32_t*) = nullptr;
int32_t (*hcommWaitResponsePtr)(MsgHandle, void*, size_t, uint32_t*) = nullptr;
int32_t (*hcommFenceOnThreadPtr)(ThreadHandle) = nullptr;
int32_t (*hcommChannelFenceOnThreadPtr)(ThreadHandle, ChannelHandle) = nullptr;

// 添加支持标志（静态，默认 false）
static bool g_hcommLocalCopyOnThreadSupported = false;
static bool g_hcommLocalReduceOnThreadSupported = false;
static bool g_hcommThreadNotifyRecordOnThreadSupported = false;
static bool g_hcommThreadNotifyWaitOnThreadSupported = false;
static bool g_hcommAclrtNotifyRecordOnThreadSupported = false;
static bool g_hcommAclrtNotifyWaitOnThreadSupported = false;
static bool g_hcommWriteOnThreadSupported = false;
static bool g_hcommWriteReduceOnThreadSupported = false;
static bool g_hcommWriteWithNotifyOnThreadSupported = false;
static bool g_hcommWriteReduceWithNotifyOnThreadSupported = false;
static bool g_hcommReadOnThreadSupported = false;
static bool g_hcommReadReduceOnThreadSupported = false;
static bool g_hcommWriteNbiOnThreadSupported = false;
static bool g_hcommWriteWithNotifyNbiOnThreadSupported = false;
static bool g_hcommReadNbiOnThreadSupported = false;
static bool g_hcommChannelNotifyRecordOnThreadSupported = false;
static bool g_hcommChannelNotifyWaitOnThreadSupported = false;
static bool g_hcommBatchModeStartSupported = false;
static bool g_hcommBatchModeEndSupported = false;
static bool g_hcommAcquireCommSupported = false;
static bool g_hcommReleaseCommSupported = false;
static bool g_hcommSymWinGetPeerPointerSupported = false;
static bool g_hcommThreadSynchronizeSupported = false;
static bool g_hcommSendRequestSupported = false;
static bool g_hcommWaitResponseSupported = false;
static bool g_hcommFenceOnThreadSupported = false;
static bool g_hcommChannelFenceOnThreadSupported = false;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static int32_t StubHcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommLocalCopyOnThread not supported");
    return -1;
}

static int32_t StubHcommLocalReduceOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t count,
                                            HcommDataType dataType, HcommReduceOp reduceOp) {
    (void)thread; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp;
    HCCL_ERROR("[HcclWrapper] HcommLocalReduceOnThread not supported");
    return -1;
}

static int32_t StubHcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx) {
    (void)thread; (void)dstThread; (void)dstNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommThreadNotifyRecordOnThread not supported");
    return -1;
}

static int32_t StubHcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut) {
    (void)thread; (void)notifyIdx; (void)timeOut;
    HCCL_ERROR("[HcclWrapper] HcommThreadNotifyWaitOnThread not supported");
    return -1;
}

static int32_t StubHcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId) {
    (void)thread; (void)dstNotifyId;
    HCCL_ERROR("[HcclWrapper] HcommAclrtNotifyRecordOnThread not supported");
    return -1;
}

static int32_t StubHcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut) {
    (void)thread; (void)notifyId; (void)timeOut;
    HCCL_ERROR("[HcclWrapper] HcommAclrtNotifyWaitOnThread not supported");
    return -1;
}

static int32_t StubHcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommWriteOnThread not supported");
    return -1;
}

static int32_t StubHcommWriteReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                            uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp;
    HCCL_ERROR("[HcclWrapper] HcommWriteReduceOnThread not supported");
    return -1;
}

static int32_t StubHcommWriteWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                                uint64_t len, uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommWriteWithNotifyOnThread not supported");
    return -1;
}

static int32_t StubHcommWriteReduceWithNotifyOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                                      uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp,
                                                      uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommWriteReduceWithNotifyOnThread not supported");
    return -1;
}

static int32_t StubHcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommReadOnThread not supported");
    return -1;
}

static int32_t StubHcommReadReduceOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
                                           uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)count; (void)dataType; (void)reduceOp;
    HCCL_ERROR("[HcclWrapper] HcommReadReduceOnThread not supported");
    return -1;
}

static int32_t StubHcommWriteNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommWriteNbiOnThread not supported");
    return -1;
}

static int32_t StubHcommWriteWithNotifyNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommWriteWithNotifyNbiOnThread not supported");
    return -1;
}

static int32_t StubHcommReadNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)thread; (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommReadNbiOnThread not supported");
    return -1;
}

static int32_t StubHcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyRecordOnThread not supported");
    return -1;
}

static int32_t StubHcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout) {
    (void)thread; (void)channel; (void)localNotifyIdx; (void)timeout;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyWaitOnThread not supported");
    return -1;
}

static int32_t StubHcommBatchModeStart(const char* batchTag) {
    (void)batchTag;
    HCCL_ERROR("[HcclWrapper] HcommBatchModeStart not supported");
    return -1;
}

static int32_t StubHcommBatchModeEnd(const char* batchTag) {
    (void)batchTag;
    HCCL_ERROR("[HcclWrapper] HcommBatchModeEnd not supported");
    return -1;
}

static int32_t StubHcommAcquireComm(const char* commId) {
    (void)commId;
    HCCL_ERROR("[HcclWrapper] HcommAcquireComm not supported");
    return -1;
}

static int32_t StubHcommReleaseComm(const char* commId) {
    (void)commId;
    HCCL_ERROR("[HcclWrapper] HcommReleaseComm not supported");
    return -1;
}

static HcclResult StubHcommSymWinGetPeerPointer(CommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr) {
    (void)winHandle; (void)offset; (void)peerRank; (void)ptr;
    HCCL_ERROR("[HcclWrapper] HcommSymWinGetPeerPointer not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static int32_t StubHcommThreadSynchronize(ThreadHandle thread) {
    (void)thread;
    HCCL_ERROR("[HcclWrapper] HcommThreadSynchronize not supported");
    return -1;
}

static int32_t StubHcommSendRequest(MsgHandle handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId) {
    (void)handle; (void)msgTag; (void)src; (void)sizeByte; (void)msgId;
    HCCL_ERROR("[HcclWrapper] HcommSendRequest not supported");
    return -1;
}

static int32_t StubHcommWaitResponse(MsgHandle handle, void* dst, size_t sizeByte, uint32_t* msgId) {
    (void)handle; (void)dst; (void)sizeByte; (void)msgId;
    HCCL_ERROR("[HcclWrapper] HcommWaitResponse not supported");
    return -1;
}

static int32_t StubHcommFenceOnThread(ThreadHandle thread) {
    (void)thread;
    HCCL_ERROR("[HcclWrapper] HcommFenceOnThread not supported");
    return -1;
}

static int32_t StubHcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel) {
    (void)thread; (void)channel;
    HCCL_ERROR("[HcclWrapper] HcommChannelFenceOnThread not supported");
    return -1;
}

// ---------- 初始化函数 ----------
void HcommPrimitivesDlInit(void* libHcommHandle) {
    // 辅助宏：解析符号，失败则指向对应桩函数，同时设置支持标志
    #define SET_PTR(ptr, handle, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(handle, name); \
            if (ptr == nullptr) { \
                ptr = stub; \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcommLocalCopyOnThreadPtr, libHcommHandle, "HcommLocalCopyOnThread", StubHcommLocalCopyOnThread, g_hcommLocalCopyOnThreadSupported);
    SET_PTR(hcommLocalReduceOnThreadPtr, libHcommHandle, "HcommLocalReduceOnThread", StubHcommLocalReduceOnThread, g_hcommLocalReduceOnThreadSupported);
    SET_PTR(hcommThreadNotifyRecordOnThreadPtr, libHcommHandle, "HcommThreadNotifyRecordOnThread", StubHcommThreadNotifyRecordOnThread, g_hcommThreadNotifyRecordOnThreadSupported);
    SET_PTR(hcommThreadNotifyWaitOnThreadPtr, libHcommHandle, "HcommThreadNotifyWaitOnThread", StubHcommThreadNotifyWaitOnThread, g_hcommThreadNotifyWaitOnThreadSupported);
    SET_PTR(hcommAclrtNotifyRecordOnThreadPtr, libHcommHandle, "HcommAclrtNotifyRecordOnThread", StubHcommAclrtNotifyRecordOnThread, g_hcommAclrtNotifyRecordOnThreadSupported);
    SET_PTR(hcommAclrtNotifyWaitOnThreadPtr, libHcommHandle, "HcommAclrtNotifyWaitOnThread", StubHcommAclrtNotifyWaitOnThread, g_hcommAclrtNotifyWaitOnThreadSupported);
    SET_PTR(hcommWriteOnThreadPtr, libHcommHandle, "HcommWriteOnThread", StubHcommWriteOnThread, g_hcommWriteOnThreadSupported);
    SET_PTR(hcommWriteReduceOnThreadPtr, libHcommHandle, "HcommWriteReduceOnThread", StubHcommWriteReduceOnThread, g_hcommWriteReduceOnThreadSupported);
    SET_PTR(hcommWriteWithNotifyOnThreadPtr, libHcommHandle, "HcommWriteWithNotifyOnThread", StubHcommWriteWithNotifyOnThread, g_hcommWriteWithNotifyOnThreadSupported);
    SET_PTR(hcommWriteReduceWithNotifyOnThreadPtr, libHcommHandle, "HcommWriteReduceWithNotifyOnThread", StubHcommWriteReduceWithNotifyOnThread, g_hcommWriteReduceWithNotifyOnThreadSupported);
    SET_PTR(hcommReadOnThreadPtr, libHcommHandle, "HcommReadOnThread", StubHcommReadOnThread, g_hcommReadOnThreadSupported);
    SET_PTR(hcommReadReduceOnThreadPtr, libHcommHandle, "HcommReadReduceOnThread", StubHcommReadReduceOnThread, g_hcommReadReduceOnThreadSupported);
    SET_PTR(hcommWriteNbiOnThreadPtr, libHcommHandle, "HcommWriteNbiOnThread", StubHcommWriteNbiOnThread, g_hcommWriteNbiOnThreadSupported);
    SET_PTR(hcommWriteWithNotifyNbiOnThreadPtr, libHcommHandle, "HcommWriteWithNotifyNbiOnThread", StubHcommWriteWithNotifyNbiOnThread, g_hcommWriteWithNotifyNbiOnThreadSupported);
    SET_PTR(hcommReadNbiOnThreadPtr, libHcommHandle, "HcommReadNbiOnThread", StubHcommReadNbiOnThread, g_hcommReadNbiOnThreadSupported);
    SET_PTR(hcommChannelNotifyRecordOnThreadPtr, libHcommHandle, "HcommChannelNotifyRecordOnThread", StubHcommChannelNotifyRecordOnThread, g_hcommChannelNotifyRecordOnThreadSupported);
    SET_PTR(hcommChannelNotifyWaitOnThreadPtr, libHcommHandle, "HcommChannelNotifyWaitOnThread", StubHcommChannelNotifyWaitOnThread, g_hcommChannelNotifyWaitOnThreadSupported);
    SET_PTR(hcommBatchModeStartPtr, libHcommHandle, "HcommBatchModeStart", StubHcommBatchModeStart, g_hcommBatchModeStartSupported);
    SET_PTR(hcommBatchModeEndPtr, libHcommHandle, "HcommBatchModeEnd", StubHcommBatchModeEnd, g_hcommBatchModeEndSupported);
    SET_PTR(hcommAcquireCommPtr, libHcommHandle, "HcommAcquireComm", StubHcommAcquireComm, g_hcommAcquireCommSupported);
    SET_PTR(hcommReleaseCommPtr, libHcommHandle, "HcommReleaseComm", StubHcommReleaseComm, g_hcommReleaseCommSupported);
    SET_PTR(hcommSymWinGetPeerPointerPtr, libHcommHandle, "HcommSymWinGetPeerPointer", StubHcommSymWinGetPeerPointer, g_hcommSymWinGetPeerPointerSupported);
    SET_PTR(hcommThreadSynchronizePtr, libHcommHandle, "HcommThreadSynchronize", StubHcommThreadSynchronize, g_hcommThreadSynchronizeSupported);
    SET_PTR(hcommSendRequestPtr, libHcommHandle, "HcommSendRequest", StubHcommSendRequest, g_hcommSendRequestSupported);
    SET_PTR(hcommWaitResponsePtr, libHcommHandle, "HcommWaitResponse", StubHcommWaitResponse, g_hcommWaitResponseSupported);
    SET_PTR(hcommFenceOnThreadPtr, libHcommHandle, "HcommFenceOnThread", StubHcommFenceOnThread, g_hcommFenceOnThreadSupported);
    SET_PTR(hcommChannelFenceOnThreadPtr, libHcommHandle, "HcommChannelFenceOnThread", StubHcommChannelFenceOnThread, g_hcommChannelFenceOnThreadSupported);

    #undef SET_PTR
}

void HcommPrimitivesDlFini(void) {
    hcommLocalCopyOnThreadPtr = StubHcommLocalCopyOnThread;
    hcommLocalReduceOnThreadPtr = StubHcommLocalReduceOnThread;
    hcommThreadNotifyRecordOnThreadPtr = StubHcommThreadNotifyRecordOnThread;
    hcommThreadNotifyWaitOnThreadPtr = StubHcommThreadNotifyWaitOnThread;
    hcommAclrtNotifyRecordOnThreadPtr = StubHcommAclrtNotifyRecordOnThread;
    hcommAclrtNotifyWaitOnThreadPtr = StubHcommAclrtNotifyWaitOnThread;
    hcommWriteOnThreadPtr = StubHcommWriteOnThread;
    hcommWriteReduceOnThreadPtr = StubHcommWriteReduceOnThread;
    hcommWriteWithNotifyOnThreadPtr = StubHcommWriteWithNotifyOnThread;
    hcommWriteReduceWithNotifyOnThreadPtr = StubHcommWriteReduceWithNotifyOnThread;
    hcommReadOnThreadPtr = StubHcommReadOnThread;
    hcommReadReduceOnThreadPtr = StubHcommReadReduceOnThread;
    hcommWriteNbiOnThreadPtr = StubHcommWriteNbiOnThread;
    hcommWriteWithNotifyNbiOnThreadPtr = StubHcommWriteWithNotifyNbiOnThread;
    hcommReadNbiOnThreadPtr = StubHcommReadNbiOnThread;
    hcommChannelNotifyRecordOnThreadPtr = StubHcommChannelNotifyRecordOnThread;
    hcommChannelNotifyWaitOnThreadPtr = StubHcommChannelNotifyWaitOnThread;
    hcommBatchModeStartPtr = StubHcommBatchModeStart;
    hcommBatchModeEndPtr = StubHcommBatchModeEnd;
    hcommAcquireCommPtr = StubHcommAcquireComm;
    hcommReleaseCommPtr = StubHcommReleaseComm;
    hcommSymWinGetPeerPointerPtr = StubHcommSymWinGetPeerPointer;
    hcommThreadSynchronizePtr = StubHcommThreadSynchronize;
    hcommSendRequestPtr = StubHcommSendRequest;
    hcommWaitResponsePtr = StubHcommWaitResponse;
    hcommFenceOnThreadPtr = StubHcommFenceOnThread;
    hcommChannelFenceOnThreadPtr = StubHcommChannelFenceOnThread;
}

// ---------- 对外提供的查询接口（判断函数是否存在）----------
extern "C" bool HcommIsSupportHcommLocalCopyOnThread(void) {
    return g_hcommLocalCopyOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommLocalReduceOnThread(void) {
    return g_hcommLocalReduceOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommThreadNotifyRecordOnThread(void) {
    return g_hcommThreadNotifyRecordOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommThreadNotifyWaitOnThread(void) {
    return g_hcommThreadNotifyWaitOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommAclrtNotifyRecordOnThread(void) {
    return g_hcommAclrtNotifyRecordOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommAclrtNotifyWaitOnThread(void) {
    return g_hcommAclrtNotifyWaitOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteOnThread(void) {
    return g_hcommWriteOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteReduceOnThread(void) {
    return g_hcommWriteReduceOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteWithNotifyOnThread(void) {
    return g_hcommWriteWithNotifyOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteReduceWithNotifyOnThread(void) {
    return g_hcommWriteReduceWithNotifyOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommReadOnThread(void) {
    return g_hcommReadOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommReadReduceOnThread(void) {
    return g_hcommReadReduceOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteNbiOnThread(void) {
    return g_hcommWriteNbiOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommWriteWithNotifyNbiOnThread(void) {
    return g_hcommWriteWithNotifyNbiOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommReadNbiOnThread(void) {
    return g_hcommReadNbiOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyRecordOnThread(void) {
    return g_hcommChannelNotifyRecordOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyWaitOnThread(void) {
    return g_hcommChannelNotifyWaitOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommBatchModeStart(void) {
    return g_hcommBatchModeStartSupported;
}
extern "C" bool HcommIsSupportHcommBatchModeEnd(void) {
    return g_hcommBatchModeEndSupported;
}
extern "C" bool HcommIsSupportHcommAcquireComm(void) {
    return g_hcommAcquireCommSupported;
}
extern "C" bool HcommIsSupportHcommReleaseComm(void) {
    return g_hcommReleaseCommSupported;
}
extern "C" bool HcommIsSupportHcommSymWinGetPeerPointer(void) {
    return g_hcommSymWinGetPeerPointerSupported;
}
extern "C" bool HcommIsSupportHcommThreadSynchronize(void) {
    return g_hcommThreadSynchronizeSupported;
}
extern "C" bool HcommIsSupportHcommSendRequest(void) {
    return g_hcommSendRequestSupported;
}
extern "C" bool HcommIsSupportHcommWaitResponse(void) {
    return g_hcommWaitResponseSupported;
}
extern "C" bool HcommIsSupportHcommFenceOnThread(void) {
    return g_hcommFenceOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommChannelFenceOnThread(void) {
    return g_hcommChannelFenceOnThreadSupported;
}