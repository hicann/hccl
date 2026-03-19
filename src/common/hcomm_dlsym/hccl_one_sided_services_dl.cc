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
#include "hccl_one_sided_services_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
HcclResult (*hcclRegisterMemPtr)(HcclComm, u32, int, void*, u64, HcclMemDesc*) = NULL;
HcclResult (*hcclDeregisterMemPtr)(HcclComm, HcclMemDesc*) = NULL;
HcclResult (*hcclExchangeMemDescPtr)(HcclComm, u32, HcclMemDescs*, int, HcclMemDescs*, u32*) = NULL;
HcclResult (*hcclEnableMemAccessPtr)(HcclComm, HcclMemDesc*, HcclMem*) = NULL;
HcclResult (*hcclDisableMemAccessPtr)(HcclComm, HcclMemDesc*) = NULL;
HcclResult (*hcclBatchPutPtr)(HcclComm, u32, HcclOneSideOpDesc*, u32, rtStream_t) = NULL;
HcclResult (*hcclBatchGetPtr)(HcclComm, u32, HcclOneSideOpDesc*, u32, rtStream_t) = NULL;
HcclResult (*hcclRemapRegistedMemoryPtr)(HcclComm*, HcclMem*, u64, u64) = NULL;
HcclResult (*hcclRegisterGlobalMemPtr)(const HcclMem*, void**) = NULL;
HcclResult (*hcclDeregisterGlobalMemPtr)(void*) = NULL;
HcclResult (*hcclCommBindMemPtr)(HcclComm, void*) = NULL;
HcclResult (*hcclCommUnbindMemPtr)(HcclComm, void*) = NULL;
HcclResult (*hcclCommPreparePtr)(HcclComm, const HcclPrepareConfig*, const int) = NULL;

// 添加支持标志（静态，默认 false）
static bool g_hcclRegisterMemSupported = false;
static bool g_hcclDeregisterMemSupported = false;
static bool g_hcclExchangeMemDescSupported = false;
static bool g_hcclEnableMemAccessSupported = false;
static bool g_hcclDisableMemAccessSupported = false;
static bool g_hcclBatchPutSupported = false;
static bool g_hcclBatchGetSupported = false;
static bool g_hcclRemapRegistedMemorySupported = false;
static bool g_hcclRegisterGlobalMemSupported = false;
static bool g_hcclDeregisterGlobalMemSupported = false;
static bool g_hcclCommBindMemSupported = false;
static bool g_hcclCommUnbindMemSupported = false;
static bool g_hcclCommPrepareSupported = false;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static HcclResult StubHcclRegisterMem(HcclComm comm, u32 remoteRank, int type, void* addr, u64 size, HcclMemDesc* desc) {
    (void)comm; (void)remoteRank; (void)type; (void)addr; (void)size; (void)desc;
    HCCL_ERROR("[HcclWrapper] HcclRegisterMem not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclDeregisterMem(HcclComm comm, HcclMemDesc* desc) {
    (void)comm; (void)desc;
    HCCL_ERROR("[HcclWrapper] HcclDeregisterMem not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclExchangeMemDesc(HcclComm comm, u32 remoteRank, HcclMemDescs* local, int timeout, HcclMemDescs* remote, u32* actualNum) {
    (void)comm; (void)remoteRank; (void)local; (void)timeout; (void)remote; (void)actualNum;
    HCCL_ERROR("[HcclWrapper] HcclExchangeMemDesc not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclEnableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc, HcclMem* remoteMem) {
    (void)comm; (void)remoteMemDesc; (void)remoteMem;
    HCCL_ERROR("[HcclWrapper] HcclEnableMemAccess not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclDisableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc) {
    (void)comm; (void)remoteMemDesc;
    HCCL_ERROR("[HcclWrapper] HcclDisableMemAccess not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclBatchPut(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream) {
    (void)comm; (void)remoteRank; (void)desc; (void)descNum; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclBatchPut not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclBatchGet(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream) {
    (void)comm; (void)remoteRank; (void)desc; (void)descNum; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclBatchGet not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclRemapRegistedMemory(HcclComm* comm, HcclMem* memInfoArray, u64 commSize, u64 arraySize) {
    (void)comm; (void)memInfoArray; (void)commSize; (void)arraySize;
    HCCL_ERROR("[HcclWrapper] HcclRemapRegistedMemory not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclRegisterGlobalMem(const HcclMem* mem, void** memHandle) {
    (void)mem; (void)memHandle;
    HCCL_ERROR("[HcclWrapper] HcclRegisterGlobalMem not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclDeregisterGlobalMem(void* memHandle) {
    (void)memHandle;
    HCCL_ERROR("[HcclWrapper] HcclDeregisterGlobalMem not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommBindMem(HcclComm comm, void* memHandle) {
    (void)comm; (void)memHandle;
    HCCL_ERROR("[HcclWrapper] HcclCommBindMem not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommUnbindMem(HcclComm comm, void* memHandle) {
    (void)comm; (void)memHandle;
    HCCL_ERROR("[HcclWrapper] HcclCommUnbindMem not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommPrepare(HcclComm comm, const HcclPrepareConfig* prepareConfig, const int timeout) {
    (void)comm; (void)prepareConfig; (void)timeout;
    HCCL_ERROR("[HcclWrapper] HcclCommPrepare not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// ---------- 初始化函数 ----------
void HcclOneSidedServicesDlInit(void* libHcommHandle) {
    #define SET_PTR(ptr, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) { \
                ptr = stub; \
                support_flag = false; \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcclRegisterMemPtr, "HcclRegisterMem", StubHcclRegisterMem, g_hcclRegisterMemSupported);
    SET_PTR(hcclDeregisterMemPtr, "HcclDeregisterMem", StubHcclDeregisterMem, g_hcclDeregisterMemSupported);
    SET_PTR(hcclExchangeMemDescPtr, "HcclExchangeMemDesc", StubHcclExchangeMemDesc, g_hcclExchangeMemDescSupported);
    SET_PTR(hcclEnableMemAccessPtr, "HcclEnableMemAccess", StubHcclEnableMemAccess, g_hcclEnableMemAccessSupported);
    SET_PTR(hcclDisableMemAccessPtr, "HcclDisableMemAccess", StubHcclDisableMemAccess, g_hcclDisableMemAccessSupported);
    SET_PTR(hcclBatchPutPtr, "HcclBatchPut", StubHcclBatchPut, g_hcclBatchPutSupported);
    SET_PTR(hcclBatchGetPtr, "HcclBatchGet", StubHcclBatchGet, g_hcclBatchGetSupported);
    SET_PTR(hcclRemapRegistedMemoryPtr, "HcclRemapRegistedMemory", StubHcclRemapRegistedMemory, g_hcclRemapRegistedMemorySupported);
    SET_PTR(hcclRegisterGlobalMemPtr, "HcclRegisterGlobalMem", StubHcclRegisterGlobalMem, g_hcclRegisterGlobalMemSupported);
    SET_PTR(hcclDeregisterGlobalMemPtr, "HcclDeregisterGlobalMem", StubHcclDeregisterGlobalMem, g_hcclDeregisterGlobalMemSupported);
    SET_PTR(hcclCommBindMemPtr, "HcclCommBindMem", StubHcclCommBindMem, g_hcclCommBindMemSupported);
    SET_PTR(hcclCommUnbindMemPtr, "HcclCommUnbindMem", StubHcclCommUnbindMem, g_hcclCommUnbindMemSupported);
    SET_PTR(hcclCommPreparePtr, "HcclCommPrepare", StubHcclCommPrepare, g_hcclCommPrepareSupported);

    #undef SET_PTR
}

void HcclOneSidedServicesDlFini(void) {
    hcclRegisterMemPtr = StubHcclRegisterMem;
    g_hcclRegisterMemSupported = false;
    hcclDeregisterMemPtr = StubHcclDeregisterMem;
    g_hcclDeregisterMemSupported = false;
    hcclExchangeMemDescPtr = StubHcclExchangeMemDesc;
    g_hcclExchangeMemDescSupported = false;
    hcclEnableMemAccessPtr = StubHcclEnableMemAccess;
    g_hcclEnableMemAccessSupported = false;
    hcclDisableMemAccessPtr = StubHcclDisableMemAccess;
    g_hcclDisableMemAccessSupported = false;
    hcclBatchPutPtr = StubHcclBatchPut;
    g_hcclBatchPutSupported = false;
    hcclBatchGetPtr = StubHcclBatchGet;
    g_hcclBatchGetSupported = false;
    hcclRemapRegistedMemoryPtr = StubHcclRemapRegistedMemory;
    g_hcclRemapRegistedMemorySupported = false;
    hcclRegisterGlobalMemPtr = StubHcclRegisterGlobalMem;
    g_hcclRegisterGlobalMemSupported = false;
    hcclDeregisterGlobalMemPtr = StubHcclDeregisterGlobalMem;
    g_hcclDeregisterGlobalMemSupported = false;
    hcclCommBindMemPtr = StubHcclCommBindMem;
    g_hcclCommBindMemSupported = false;
    hcclCommUnbindMemPtr = StubHcclCommUnbindMem;
    g_hcclCommUnbindMemSupported = false;
    hcclCommPreparePtr = StubHcclCommPrepare;
    g_hcclCommPrepareSupported = false;
}

// ---------- 对外API实现（通过函数指针转发）----------
HcclResult HcclRegisterMem(HcclComm comm, u32 remoteRank, int type, void* addr, u64 size, HcclMemDesc* desc) {
    return hcclRegisterMemPtr(comm, remoteRank, type, addr, size, desc);
}
HcclResult HcclDeregisterMem(HcclComm comm, HcclMemDesc* desc) {
    return hcclDeregisterMemPtr(comm, desc);
}
HcclResult HcclExchangeMemDesc(HcclComm comm, u32 remoteRank, HcclMemDescs* local, int timeout, HcclMemDescs* remote, u32* actualNum) {
    return hcclExchangeMemDescPtr(comm, remoteRank, local, timeout, remote, actualNum);
}
HcclResult HcclEnableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc, HcclMem* remoteMem) {
    return hcclEnableMemAccessPtr(comm, remoteMemDesc, remoteMem);
}
HcclResult HcclDisableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc) {
    return hcclDisableMemAccessPtr(comm, remoteMemDesc);
}
HcclResult HcclBatchPut(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream) {
    return hcclBatchPutPtr(comm, remoteRank, desc, descNum, stream);
}
HcclResult HcclBatchGet(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream) {
    return hcclBatchGetPtr(comm, remoteRank, desc, descNum, stream);
}
HcclResult HcclRemapRegistedMemory(HcclComm *comm, HcclMem *memInfoArray, u64 commSize, u64 arraySize) {
    return hcclRemapRegistedMemoryPtr(comm, memInfoArray, commSize, arraySize);
}
HcclResult HcclRegisterGlobalMem(const HcclMem* mem, void** memHandle) {
    return hcclRegisterGlobalMemPtr(mem, memHandle);
}
HcclResult HcclDeregisterGlobalMem(void* memHandle) {
    return hcclDeregisterGlobalMemPtr(memHandle);
}
HcclResult HcclCommBindMem(HcclComm comm, void* memHandle) {
    return hcclCommBindMemPtr(comm, memHandle);
}
HcclResult HcclCommUnbindMem(HcclComm comm, void* memHandle) {
    return hcclCommUnbindMemPtr(comm, memHandle);
}
HcclResult HcclCommPrepare(HcclComm comm, const HcclPrepareConfig* prepareConfig, const int timeout) {
    return hcclCommPreparePtr(comm, prepareConfig, timeout);
}

// ---------- 查询函数实现 ----------
extern "C" bool HcommIsSupportHcclRegisterMem(void) { return g_hcclRegisterMemSupported; }
extern "C" bool HcommIsSupportHcclDeregisterMem(void) { return g_hcclDeregisterMemSupported; }
extern "C" bool HcommIsSupportHcclExchangeMemDesc(void) { return g_hcclExchangeMemDescSupported; }
extern "C" bool HcommIsSupportHcclEnableMemAccess(void) { return g_hcclEnableMemAccessSupported; }
extern "C" bool HcommIsSupportHcclDisableMemAccess(void) { return g_hcclDisableMemAccessSupported; }
extern "C" bool HcommIsSupportHcclBatchPut(void) { return g_hcclBatchPutSupported; }
extern "C" bool HcommIsSupportHcclBatchGet(void) { return g_hcclBatchGetSupported; }
extern "C" bool HcommIsSupportHcclRemapRegistedMemory(void) { return g_hcclRemapRegistedMemorySupported; }
extern "C" bool HcommIsSupportHcclRegisterGlobalMem(void) { return g_hcclRegisterGlobalMemSupported; }
extern "C" bool HcommIsSupportHcclDeregisterGlobalMem(void) { return g_hcclDeregisterGlobalMemSupported; }
extern "C" bool HcommIsSupportHcclCommBindMem(void) { return g_hcclCommBindMemSupported; }
extern "C" bool HcommIsSupportHcclCommUnbindMem(void) { return g_hcclCommUnbindMemSupported; }
extern "C" bool HcommIsSupportHcclCommPrepare(void) { return g_hcclCommPrepareSupported; }