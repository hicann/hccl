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
#include "hccl_res_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
HcclResult (*hcclGetHcclBufferPtr)(HcclComm, void**, uint64_t*) = nullptr;
HcclResult (*hcclGetRemoteIpcHcclBufPtr)(HcclComm, uint64_t, void**, uint64_t*) = nullptr;
HcclResult (*hcclThreadAcquirePtr)(HcclComm, CommEngine, uint32_t, uint32_t, ThreadHandle*) = nullptr;
HcclResult (*hcclThreadAcquireWithStreamPtr)(HcclComm, CommEngine, aclrtStream, uint32_t, ThreadHandle*) = nullptr;
HcclResult (*hcclChannelAcquirePtr)(HcclComm, CommEngine, const HcclChannelDesc*, uint32_t, ChannelHandle*) = nullptr;
HcclResult (*hcclChannelGetHcclBufferPtr)(HcclComm, ChannelHandle, void**, uint64_t*) = nullptr;
HcclResult (*hcclEngineCtxCreatePtr)(HcclComm, const char*, CommEngine, uint64_t, void**) = nullptr;
HcclResult (*hcclEngineCtxGetPtr)(HcclComm, const char*, CommEngine, void**, uint64_t*) = nullptr;
HcclResult (*hcclEngineCtxCopyPtr)(HcclComm, CommEngine, const char*, const void*, uint64_t, uint64_t) = nullptr;
int32_t    (*hcclTaskRegisterPtr)(HcclComm, const char*, Callback) = nullptr;
int32_t    (*hcclTaskUnRegisterPtr)(HcclComm, const char*) = nullptr;
HcclResult (*hcclDevMemAcquirePtr)(HcclComm, const char*, uint64_t*, void**, bool*) = nullptr;
HcclResult (*hcclThreadExportToCommEnginePtr)(HcclComm, uint32_t, const ThreadHandle*, CommEngine, ThreadHandle*) = nullptr;
HcclResult (*hcclChannelGetRemoteMemsPtr)(HcclComm, ChannelHandle, uint32_t*, CommMem**, char***) = nullptr;
HcclResult (*hcclCommMemRegPtr)(HcclComm, const char*, const CommMem*, HcclMemHandle*) = nullptr;
HcclResult (*hcclEngineCtxDestroyPtr)(HcclComm, const char*, CommEngine) = nullptr;

// 添加支持标志（静态，默认 false）
static bool g_hcclGetHcclBufferSupported = false;
static bool g_hcclGetRemoteIpcHcclBufSupported = false;
static bool g_hcclThreadAcquireSupported = false;
static bool g_hcclThreadAcquireWithStreamSupported = false;
static bool g_hcclChannelAcquireSupported = false;
static bool g_hcclChannelGetHcclBufferSupported = false;
static bool g_hcclEngineCtxCreateSupported = false;
static bool g_hcclEngineCtxGetSupported = false;
static bool g_hcclEngineCtxCopySupported = false;
static bool g_hcclTaskRegisterSupported = false;
static bool g_hcclTaskUnRegisterSupported = false;
static bool g_hcclDevMemAcquireSupported = false;
static bool g_hcclThreadExportToCommEngineSupported = false;
static bool g_hcclChannelGetRemoteMemsSupported = false;
static bool g_hcclCommMemRegSupported = false;
static bool g_hcclEngineCtxDestroySupported = false;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static HcclResult StubHcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size) {
    (void)comm; (void)buffer; (void)size;
    HCCL_ERROR("[HcclWrapper] HcclGetHcclBuffer not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclGetRemoteIpcHcclBuf(HcclComm comm, uint64_t remoteRank, void** addr, uint64_t* size) {
    (void)comm; (void)remoteRank; (void)addr; (void)size;
    HCCL_ERROR("[HcclWrapper] HcclGetRemoteIpcHcclBuf not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum,
                                        uint32_t notifyNumPerThread, ThreadHandle* threads) {
    (void)comm; (void)engine; (void)threadNum; (void)notifyNumPerThread; (void)threads;
    HCCL_ERROR("[HcclWrapper] HcclThreadAcquire not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream,
                                                  uint32_t notifyNum, ThreadHandle* thread) {
    (void)comm; (void)engine; (void)stream; (void)notifyNum; (void)thread;
    HCCL_ERROR("[HcclWrapper] HcclThreadAcquireWithStream not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs,
                                         uint32_t channelNum, ChannelHandle* channels) {
    (void)comm; (void)engine; (void)channelDescs; (void)channelNum; (void)channels;
    HCCL_ERROR("[HcclWrapper] HcclChannelAcquire not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void** buffer, uint64_t* size) {
    (void)comm; (void)channel; (void)buffer; (void)size;
    HCCL_ERROR("[HcclWrapper] HcclChannelGetHcclBuffer not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine,
                                          uint64_t size, void** ctx) {
    (void)comm; (void)ctxTag; (void)engine; (void)size; (void)ctx;
    HCCL_ERROR("[HcclWrapper] HcclEngineCtxCreate not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine,
                                       void** ctx, uint64_t* size) {
    (void)comm; (void)ctxTag; (void)engine; (void)ctx; (void)size;
    HCCL_ERROR("[HcclWrapper] HcclEngineCtxGet not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char* ctxTag,
                                        const void* srcCtx, uint64_t size, uint64_t dstCtxOffset) {
    (void)comm; (void)engine; (void)ctxTag; (void)srcCtx; (void)size; (void)dstCtxOffset;
    HCCL_ERROR("[HcclWrapper] HcclEngineCtxCopy not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static int32_t StubHcclTaskRegister(HcclComm comm, const char* msgTag, Callback cb) {
    (void)comm; (void)msgTag; (void)cb;
    HCCL_ERROR("[HcclWrapper] HcclTaskRegister not supported");
    return -1;
}

static int32_t StubHcclTaskUnRegister(HcclComm comm, const char* msgTag) {
    (void)comm; (void)msgTag;
    HCCL_ERROR("[HcclWrapper] HcclTaskUnRegister not supported");
    return -1;
}

static HcclResult StubHcclDevMemAcquire(HcclComm comm, const char* memTag, uint64_t* size,
                                        void** addr, bool* newCreated) {
    (void)comm; (void)memTag; (void)size; (void)addr; (void)newCreated;
    HCCL_ERROR("[HcclWrapper] HcclDevMemAcquire not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclThreadExportToCommEngine(HcclComm comm, uint32_t threadNum,
                                                   const ThreadHandle* threads, CommEngine dstCommEngine,
                                                   ThreadHandle* exportedThreads) {
    (void)comm; (void)threadNum; (void)threads; (void)dstCommEngine; (void)exportedThreads;
    HCCL_ERROR("[HcclWrapper] HcclThreadExportToCommEngine not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel,
                                               uint32_t* memNum, CommMem** remoteMems, char*** memTags) {
    (void)comm; (void)channel; (void)memNum; (void)remoteMems; (void)memTags;
    HCCL_ERROR("[HcclWrapper] HcclChannelGetRemoteMems not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclCommMemReg(HcclComm comm, const char* memTag, const CommMem* mem,
                                     HcclMemHandle* memHandle) {
    (void)comm; (void)memTag; (void)mem; (void)memHandle;
    HCCL_ERROR("[HcclWrapper] HcclCommMemReg not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclEngineCtxDestroy(HcclComm comm, const char* ctxTag, CommEngine engine) {
    (void)comm; (void)ctxTag; (void)engine;
    HCCL_ERROR("[HcclWrapper] HcclEngineCtxDestroy not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// 初始化
void HcclResDlInit(void* libHcommHandle) {
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

    SET_PTR(hcclGetHcclBufferPtr, libHcommHandle, "HcclGetHcclBuffer", StubHcclGetHcclBuffer, g_hcclGetHcclBufferSupported);
    SET_PTR(hcclGetRemoteIpcHcclBufPtr, libHcommHandle, "HcclGetRemoteIpcHcclBuf", StubHcclGetRemoteIpcHcclBuf, g_hcclGetRemoteIpcHcclBufSupported);
    SET_PTR(hcclThreadAcquirePtr, libHcommHandle, "HcclThreadAcquire", StubHcclThreadAcquire, g_hcclThreadAcquireSupported);
    SET_PTR(hcclThreadAcquireWithStreamPtr, libHcommHandle, "HcclThreadAcquireWithStream", StubHcclThreadAcquireWithStream, g_hcclThreadAcquireWithStreamSupported);
    SET_PTR(hcclChannelAcquirePtr, libHcommHandle, "HcclChannelAcquire", StubHcclChannelAcquire, g_hcclChannelAcquireSupported);
    SET_PTR(hcclChannelGetHcclBufferPtr, libHcommHandle, "HcclChannelGetHcclBuffer", StubHcclChannelGetHcclBuffer, g_hcclChannelGetHcclBufferSupported);
    SET_PTR(hcclEngineCtxCreatePtr, libHcommHandle, "HcclEngineCtxCreate", StubHcclEngineCtxCreate, g_hcclEngineCtxCreateSupported);
    SET_PTR(hcclEngineCtxGetPtr, libHcommHandle, "HcclEngineCtxGet", StubHcclEngineCtxGet, g_hcclEngineCtxGetSupported);
    SET_PTR(hcclEngineCtxCopyPtr, libHcommHandle, "HcclEngineCtxCopy", StubHcclEngineCtxCopy, g_hcclEngineCtxCopySupported);
    SET_PTR(hcclTaskRegisterPtr, libHcommHandle, "HcclTaskRegister", StubHcclTaskRegister, g_hcclTaskRegisterSupported);
    SET_PTR(hcclTaskUnRegisterPtr, libHcommHandle, "HcclTaskUnRegister", StubHcclTaskUnRegister, g_hcclTaskUnRegisterSupported);
    SET_PTR(hcclDevMemAcquirePtr, libHcommHandle, "HcclDevMemAcquire", StubHcclDevMemAcquire, g_hcclDevMemAcquireSupported);
    SET_PTR(hcclThreadExportToCommEnginePtr, libHcommHandle, "HcclThreadExportToCommEngine", StubHcclThreadExportToCommEngine, g_hcclThreadExportToCommEngineSupported);
    SET_PTR(hcclChannelGetRemoteMemsPtr, libHcommHandle, "HcclChannelGetRemoteMems", StubHcclChannelGetRemoteMems, g_hcclChannelGetRemoteMemsSupported);
    SET_PTR(hcclCommMemRegPtr, libHcommHandle, "HcclCommMemReg", StubHcclCommMemReg, g_hcclCommMemRegSupported);
    SET_PTR(hcclEngineCtxDestroyPtr, libHcommHandle, "HcclEngineCtxDestroy", StubHcclEngineCtxDestroy, g_hcclEngineCtxDestroySupported);

    #undef SET_PTR
}

void HcclResDlFini(void) {
    // 重置为桩函数，防止fini后误用
    hcclGetHcclBufferPtr = StubHcclGetHcclBuffer;
    hcclGetRemoteIpcHcclBufPtr = StubHcclGetRemoteIpcHcclBuf;
    hcclThreadAcquirePtr = StubHcclThreadAcquire;
    hcclThreadAcquireWithStreamPtr = StubHcclThreadAcquireWithStream;
    hcclChannelAcquirePtr = StubHcclChannelAcquire;
    hcclChannelGetHcclBufferPtr = StubHcclChannelGetHcclBuffer;
    hcclEngineCtxCreatePtr = StubHcclEngineCtxCreate;
    hcclEngineCtxGetPtr = StubHcclEngineCtxGet;
    hcclEngineCtxCopyPtr = StubHcclEngineCtxCopy;
    hcclTaskRegisterPtr = StubHcclTaskRegister;
    hcclTaskUnRegisterPtr = StubHcclTaskUnRegister;
    hcclDevMemAcquirePtr = StubHcclDevMemAcquire;
    hcclThreadExportToCommEnginePtr = StubHcclThreadExportToCommEngine;
    hcclChannelGetRemoteMemsPtr = StubHcclChannelGetRemoteMems;
    hcclCommMemRegPtr = StubHcclCommMemReg;
    hcclEngineCtxDestroyPtr = StubHcclEngineCtxDestroy;
}

// ---------- 对外API实现（通过函数指针转发）----------
HcclResult HcclGetHcclBuffer(HcclComm comm, void **buffer, uint64_t *size) {
    return hcclGetHcclBufferPtr(comm, buffer, size);
}
HcclResult HcclGetRemoteIpcHcclBuf(HcclComm comm, uint64_t remoteRank, void **addr, uint64_t *size) {
    return hcclGetRemoteIpcHcclBufPtr(comm, remoteRank, addr, size);
}
HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum,
                              uint32_t notifyNumPerThread, ThreadHandle *threads) {
    return hcclThreadAcquirePtr(comm, engine, threadNum, notifyNumPerThread, threads);
}
HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine, aclrtStream stream,
                                       uint32_t notifyNum, ThreadHandle *thread) {
    return hcclThreadAcquireWithStreamPtr(comm, engine, stream, notifyNum, thread);
}
HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc *channelDescs,
                              uint32_t channelNum, ChannelHandle *channels) {
    return hcclChannelAcquirePtr(comm, engine, channelDescs, channelNum, channels);
}
HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void **buffer, uint64_t *size) {
    return hcclChannelGetHcclBufferPtr(comm, channel, buffer, size);
}
HcclResult HcclEngineCtxCreate(HcclComm comm, const char *ctxTag, CommEngine engine,
                               uint64_t size, void **ctx) {
    return hcclEngineCtxCreatePtr(comm, ctxTag, engine, size, ctx);
}
HcclResult HcclEngineCtxGet(HcclComm comm, const char *ctxTag, CommEngine engine,
                            void **ctx, uint64_t *size) {
    return hcclEngineCtxGetPtr(comm, ctxTag, engine, ctx, size);
}
HcclResult HcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char *ctxTag,
                             const void *srcCtx, uint64_t size, uint64_t dstCtxOffset) {
    return hcclEngineCtxCopyPtr(comm, engine, ctxTag, srcCtx, size, dstCtxOffset);
}
int32_t HcclTaskRegister(HcclComm comm, const char *msgTag, Callback cb) {
    return hcclTaskRegisterPtr(comm, msgTag, cb);
}
int32_t HcclTaskUnRegister(HcclComm comm, const char *msgTag) {
    return hcclTaskUnRegisterPtr(comm, msgTag);
}
HcclResult HcclDevMemAcquire(HcclComm comm, const char *memTag, uint64_t *size,
                             void **addr, bool *newCreated) {
    return hcclDevMemAcquirePtr(comm, memTag, size, addr, newCreated);
}
HcclResult HcclThreadExportToCommEngine(HcclComm comm, uint32_t threadNum,
                                        const ThreadHandle *threads, CommEngine dstCommEngine,
                                        ThreadHandle *exportedThreads) {
    return hcclThreadExportToCommEnginePtr(comm, threadNum, threads, dstCommEngine, exportedThreads);
}
HcclResult HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel,
                                   uint32_t *memNum, CommMem **remoteMems, char ***memTags) {
    return hcclChannelGetRemoteMemsPtr(comm, channel, memNum, remoteMems, memTags);
}
HcclResult HcclCommMemReg(HcclComm comm, const char *memTag, const CommMem *mem,
                          HcclMemHandle *memHandle) {
    return hcclCommMemRegPtr(comm, memTag, mem, memHandle);
}
HcclResult HcclEngineCtxDestroy(HcclComm comm, const char *ctxTag, CommEngine engine) {
    return hcclEngineCtxDestroyPtr(comm, ctxTag, engine);
}

// ---------- 对外提供的查询接口（判断函数是否存在）----------
extern "C" bool HcommIsSupportHcclGetHcclBuffer(void) {
    return g_hcclGetHcclBufferSupported;
}
extern "C" bool HcommIsSupportHcclGetRemoteIpcHcclBuf(void) {
    return g_hcclGetRemoteIpcHcclBufSupported;
}
extern "C" bool HcommIsSupportHcclThreadAcquire(void) {
    return g_hcclThreadAcquireSupported;
}
extern "C" bool HcommIsSupportHcclThreadAcquireWithStream(void) {
    return g_hcclThreadAcquireWithStreamSupported;
}
extern "C" bool HcommIsSupportHcclChannelAcquire(void) {
    return g_hcclChannelAcquireSupported;
}
extern "C" bool HcommIsSupportHcclChannelGetHcclBuffer(void) {
    return g_hcclChannelGetHcclBufferSupported;
}
extern "C" bool HcommIsSupportHcclEngineCtxCreate(void) {
    return g_hcclEngineCtxCreateSupported;
}
extern "C" bool HcommIsSupportHcclEngineCtxGet(void) {
    return g_hcclEngineCtxGetSupported;
}
extern "C" bool HcommIsSupportHcclEngineCtxCopy(void) {
    return g_hcclEngineCtxCopySupported;
}
extern "C" bool HcommIsSupportHcclTaskRegister(void) {
    return g_hcclTaskRegisterSupported;
}
extern "C" bool HcommIsSupportHcclTaskUnRegister(void) {
    return g_hcclTaskUnRegisterSupported;
}
extern "C" bool HcommIsSupportHcclDevMemAcquire(void) {
    return g_hcclDevMemAcquireSupported;
}
extern "C" bool HcommIsSupportHcclThreadExportToCommEngine(void) {
    return g_hcclThreadExportToCommEngineSupported;
}
extern "C" bool HcommIsSupportHcclChannelGetRemoteMems(void) {
    return g_hcclChannelGetRemoteMemsSupported;
}
extern "C" bool HcommIsSupportHcclCommMemReg(void) {
    return g_hcclCommMemRegSupported;
}
extern "C" bool HcommIsSupportHcclEngineCtxDestroy(void) {
    return g_hcclEngineCtxDestroySupported;
}