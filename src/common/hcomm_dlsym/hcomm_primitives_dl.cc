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
int32_t (*hcommLocalCopyOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommLocalReduceOnThreadPtr)(ThreadHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = NULL;
int32_t (*hcommThreadNotifyRecordOnThreadPtr)(ThreadHandle, ThreadHandle, uint32_t) = NULL;
int32_t (*hcommThreadNotifyWaitOnThreadPtr)(ThreadHandle, uint32_t, uint32_t) = NULL;
int32_t (*hcommAclrtNotifyRecordOnThreadPtr)(ThreadHandle, uint64_t) = NULL;
int32_t (*hcommAclrtNotifyWaitOnThreadPtr)(ThreadHandle, uint64_t, uint32_t) = NULL;
int32_t (*hcommWriteOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommWriteReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = NULL;
int32_t (*hcommWriteWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, uint32_t) = NULL;
int32_t (*hcommWriteReduceWithNotifyOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp, uint32_t) = NULL;
int32_t (*hcommReadOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommReadReduceOnThreadPtr)(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp) = NULL;
int32_t (*hcommWriteNbiPtr)(ChannelHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommWriteWithNotifyNbiPtr)(ChannelHandle, void*, const void*, uint64_t, uint32_t) = NULL;
int32_t (*hcommReadNbiPtr)(ChannelHandle, void*, const void*, uint64_t) = NULL;
int32_t (*hcommChannelNotifyRecordOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t) = NULL;
int32_t (*hcommChannelNotifyRecordPtr)(ChannelHandle, uint32_t) = NULL;
int32_t (*hcommChannelNotifyWaitOnThreadPtr)(ThreadHandle, ChannelHandle, uint32_t, uint32_t) = NULL;
int32_t (*hcommChannelNotifyWaitPtr)(ChannelHandle, uint32_t, uint32_t) = NULL;
int32_t (*hcommBatchModeStartPtr)(const char*) = NULL;
int32_t (*hcommBatchModeEndPtr)(const char*) = NULL;
int32_t (*hcommAcquireCommPtr)(const char*) = NULL;
int32_t (*hcommReleaseCommPtr)(const char*) = NULL;
HcclResult (*hcommSymWinGetPeerPointerPtr)(CommSymWindow, size_t, uint32_t, void**) = NULL;
int32_t (*hcommThreadSynchronizePtr)(ThreadHandle) = NULL;
int32_t (*hcommSendRequestPtr)(MsgHandle, const char*, const void*, size_t, uint32_t*) = NULL;
int32_t (*hcommWaitResponsePtr)(MsgHandle, void*, size_t, uint32_t*) = NULL;
int32_t (*hcommFlushPtr)() = NULL;
int32_t (*hcommChannelFencePtr)(ChannelHandle) = NULL;

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
static bool g_hcommWriteNbiSupported = false;
static bool g_hcommWriteWithNotifyNbiSupported = false;
static bool g_hcommReadNbiSupported = false;
static bool g_hcommChannelNotifyRecordOnThreadSupported = false;
static bool g_hcommChannelNotifyRecordSupported = false;
static bool g_hcommChannelNotifyWaitOnThreadSupported = false;
static bool g_hcommChannelNotifyWaitSupported = false;
static bool g_hcommBatchModeStartSupported = false;
static bool g_hcommBatchModeEndSupported = false;
static bool g_hcommAcquireCommSupported = false;
static bool g_hcommReleaseCommSupported = false;
static bool g_hcommSymWinGetPeerPointerSupported = false;
static bool g_hcommThreadSynchronizeSupported = false;
static bool g_hcommSendRequestSupported = false;
static bool g_hcommWaitResponseSupported = false;
static bool g_hcommFlushSupported = false;
static bool g_hcommChannelFenceSupported = false;

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

static int32_t StubHcommWriteNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommWriteNbi not supported");
    return -1;
}

static int32_t StubHcommWriteWithNotifyNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx) {
    (void)channel; (void)dst; (void)src; (void)len; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommWriteWithNotifyNbi not supported");
    return -1;
}

static int32_t StubHcommReadNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len) {
    (void)channel; (void)dst; (void)src; (void)len;
    HCCL_ERROR("[HcclWrapper] HcommReadNbi not supported");
    return -1;
}

static int32_t StubHcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx) {
    (void)thread; (void)channel; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyRecordOnThread not supported");
    return -1;
}

static int32_t StubHcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx) {
    (void)channel; (void)remoteNotifyIdx;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyRecord not supported");
    return -1;
}

static int32_t StubHcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout) {
    (void)thread; (void)channel; (void)localNotifyIdx; (void)timeout;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyWaitOnThread not supported");
    return -1;
}

static int32_t StubHcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout) {
    (void)channel; (void)localNotifyIdx; (void)timeout;
    HCCL_ERROR("[HcclWrapper] HcommChannelNotifyWait not supported");
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

static int32_t StubHcommFlush() {
    HCCL_ERROR("[HcclWrapper] HcommFlush not supported");
    return -1;
}

static int32_t StubHcommChannelFence(ChannelHandle channel) {
    (void)channel;
    HCCL_ERROR("[HcclWrapper] HcommChannelFence not supported");
    return -1;
}

// ---------- 初始化函数 ----------
void HcommPrimitivesDlInit(void* libHcommHandle) {
    // 辅助宏：解析符号，失败则指向对应桩函数，同时设置支持标志
    #define SET_PTR(ptr, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) { \
                ptr = stub; \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcommLocalCopyOnThreadPtr, "HcommLocalCopyOnThread", StubHcommLocalCopyOnThread, g_hcommLocalCopyOnThreadSupported);
    SET_PTR(hcommLocalReduceOnThreadPtr, "HcommLocalReduceOnThread", StubHcommLocalReduceOnThread, g_hcommLocalReduceOnThreadSupported);
    SET_PTR(hcommThreadNotifyRecordOnThreadPtr, "HcommThreadNotifyRecordOnThread", StubHcommThreadNotifyRecordOnThread, g_hcommThreadNotifyRecordOnThreadSupported);
    SET_PTR(hcommThreadNotifyWaitOnThreadPtr, "HcommThreadNotifyWaitOnThread", StubHcommThreadNotifyWaitOnThread, g_hcommThreadNotifyWaitOnThreadSupported);
    SET_PTR(hcommAclrtNotifyRecordOnThreadPtr, "HcommAclrtNotifyRecordOnThread", StubHcommAclrtNotifyRecordOnThread, g_hcommAclrtNotifyRecordOnThreadSupported);
    SET_PTR(hcommAclrtNotifyWaitOnThreadPtr, "HcommAclrtNotifyWaitOnThread", StubHcommAclrtNotifyWaitOnThread, g_hcommAclrtNotifyWaitOnThreadSupported);
    SET_PTR(hcommWriteOnThreadPtr, "HcommWriteOnThread", StubHcommWriteOnThread, g_hcommWriteOnThreadSupported);
    SET_PTR(hcommWriteReduceOnThreadPtr, "HcommWriteReduceOnThread", StubHcommWriteReduceOnThread, g_hcommWriteReduceOnThreadSupported);
    SET_PTR(hcommWriteWithNotifyOnThreadPtr, "HcommWriteWithNotifyOnThread", StubHcommWriteWithNotifyOnThread, g_hcommWriteWithNotifyOnThreadSupported);
    SET_PTR(hcommWriteReduceWithNotifyOnThreadPtr, "HcommWriteReduceWithNotifyOnThread", StubHcommWriteReduceWithNotifyOnThread, g_hcommWriteReduceWithNotifyOnThreadSupported);
    SET_PTR(hcommReadOnThreadPtr, "HcommReadOnThread", StubHcommReadOnThread, g_hcommReadOnThreadSupported);
    SET_PTR(hcommReadReduceOnThreadPtr, "HcommReadReduceOnThread", StubHcommReadReduceOnThread, g_hcommReadReduceOnThreadSupported);
    SET_PTR(hcommWriteNbiPtr, "HcommWriteNbi", StubHcommWriteNbi, g_hcommWriteNbiSupported);
    SET_PTR(hcommWriteWithNotifyNbiPtr, "HcommWriteWithNotifyNbi", StubHcommWriteWithNotifyNbi, g_hcommWriteWithNotifyNbiSupported);
    SET_PTR(hcommReadNbiPtr, "HcommReadNbi", StubHcommReadNbi, g_hcommReadNbiSupported);
    SET_PTR(hcommChannelNotifyRecordOnThreadPtr, "HcommChannelNotifyRecordOnThread", StubHcommChannelNotifyRecordOnThread, g_hcommChannelNotifyRecordOnThreadSupported);
    SET_PTR(hcommChannelNotifyRecordPtr, "HcommChannelNotifyRecord", StubHcommChannelNotifyRecord, g_hcommChannelNotifyRecordSupported);
    SET_PTR(hcommChannelNotifyWaitOnThreadPtr, "HcommChannelNotifyWaitOnThread", StubHcommChannelNotifyWaitOnThread, g_hcommChannelNotifyWaitOnThreadSupported);
    SET_PTR(hcommChannelNotifyWaitPtr, "HcommChannelNotifyWait", StubHcommChannelNotifyWait, g_hcommChannelNotifyWaitSupported);
    SET_PTR(hcommBatchModeStartPtr, "HcommBatchModeStart", StubHcommBatchModeStart, g_hcommBatchModeStartSupported);
    SET_PTR(hcommBatchModeEndPtr, "HcommBatchModeEnd", StubHcommBatchModeEnd, g_hcommBatchModeEndSupported);
    SET_PTR(hcommAcquireCommPtr, "HcommAcquireComm", StubHcommAcquireComm, g_hcommAcquireCommSupported);
    SET_PTR(hcommReleaseCommPtr, "HcommReleaseComm", StubHcommReleaseComm, g_hcommReleaseCommSupported);
    SET_PTR(hcommSymWinGetPeerPointerPtr, "HcommSymWinGetPeerPointer", StubHcommSymWinGetPeerPointer, g_hcommSymWinGetPeerPointerSupported);
    SET_PTR(hcommThreadSynchronizePtr, "HcommThreadSynchronize", StubHcommThreadSynchronize, g_hcommThreadSynchronizeSupported);
    SET_PTR(hcommSendRequestPtr, "HcommSendRequest", StubHcommSendRequest, g_hcommSendRequestSupported);
    SET_PTR(hcommWaitResponsePtr, "HcommWaitResponse", StubHcommWaitResponse, g_hcommWaitResponseSupported);
    SET_PTR(hcommFlushPtr, "HcommFlush", StubHcommFlush, g_hcommFlushSupported);
    SET_PTR(hcommChannelFencePtr, "HcommChannelFence", StubHcommChannelFence, g_hcommChannelFenceSupported);

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
    hcommWriteNbiPtr = StubHcommWriteNbi;
    hcommWriteWithNotifyNbiPtr = StubHcommWriteWithNotifyNbi;
    hcommReadNbiPtr = StubHcommReadNbi;
    hcommChannelNotifyRecordOnThreadPtr = StubHcommChannelNotifyRecordOnThread;
    hcommChannelNotifyRecordPtr = StubHcommChannelNotifyRecord;
    hcommChannelNotifyWaitOnThreadPtr = StubHcommChannelNotifyWaitOnThread;
    hcommChannelNotifyWaitPtr = StubHcommChannelNotifyWait;
    hcommBatchModeStartPtr = StubHcommBatchModeStart;
    hcommBatchModeEndPtr = StubHcommBatchModeEnd;
    hcommAcquireCommPtr = StubHcommAcquireComm;
    hcommReleaseCommPtr = StubHcommReleaseComm;
    hcommSymWinGetPeerPointerPtr = StubHcommSymWinGetPeerPointer;
    hcommThreadSynchronizePtr = StubHcommThreadSynchronize;
    hcommSendRequestPtr = StubHcommSendRequest;
    hcommWaitResponsePtr = StubHcommWaitResponse;
    hcommFlushPtr = StubHcommFlush;
    hcommChannelFencePtr = StubHcommChannelFence;
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
extern "C" bool HcommIsSupportHcommWriteNbi(void) {
    return g_hcommWriteNbiSupported;
}
extern "C" bool HcommIsSupportHcommWriteWithNotifyNbi(void) {
    return g_hcommWriteWithNotifyNbiSupported;
}
extern "C" bool HcommIsSupportHcommReadNbi(void) {
    return g_hcommReadNbiSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyRecordOnThread(void) {
    return g_hcommChannelNotifyRecordOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyRecord(void) {
    return g_hcommChannelNotifyRecordSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyWaitOnThread(void) {
    return g_hcommChannelNotifyWaitOnThreadSupported;
}
extern "C" bool HcommIsSupportHcommChannelNotifyWait(void) {
    return g_hcommChannelNotifyWaitSupported;
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
extern "C" bool HcommIsSupportHcommFlush(void) {
    return g_hcommFlushSupported;
}
extern "C" bool HcommIsSupportHcommChannelFence(void) {
    return g_hcommChannelFenceSupported;
}