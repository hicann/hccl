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
#include "hccl_mc2_ex_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（静态）
static HcclResult (*hcclGetCommHandleByCtxPtr)(void*, void**) = NULL;
static HcclResult (*hcclReleaseCommPtr)(void*) = NULL;
static HcclResult (*hcclGetTaskStatusPtr)(void*, void*) = NULL;
static HcclResult (*hcclCheckFinishByStreamPtr)(void*) = NULL;
static HcclResult (*hcclPrintTaskExceptionAllCommPtr)(void*) = NULL;
static HcclResult (*hcclLaunchCcoreWaitPtr)(void*, uint64_t, uint32_t, uint64_t, bool) = NULL;
static HcclResult (*hcclLaunchCcorePostPtr)(void*, uint64_t, uint32_t, uint64_t) = NULL;
static HcclResult (*hcclLaunchOpPtr)(void*, void*) = NULL;
static HcclResult (*hcclGetOpArgsPtr)(void**) = NULL;
static HcclResult (*hcclFreeOpArgsPtr)(void*) = NULL;
static HcclResult (*hcclSetOpSrcDataTypePtr)(void*, uint8_t) = NULL;
static HcclResult (*hcclSetOpDstDataTypePtr)(void*, uint8_t) = NULL;
static HcclResult (*hcclSetOpReduceTypePtr)(void*, uint32_t) = NULL;
static HcclResult (*hcclSetOpCountPtr)(void*, uint64_t) = NULL;
static HcclResult (*hcclSetOpAlgConfigPtr)(void*, char*) = NULL;
static HcclResult (*hcclSetOpCommEnginePtr)(void*, uint8_t) = NULL;
static HcclResult (*hcclCommResPreparePtr)(HcclComm, char*, void*, void**) = NULL;

// 支持标志（静态，默认 false）
static bool g_hcclGetCommHandleByCtxSupported = false;
static bool g_hcclReleaseCommSupported = false;
static bool g_hcclGetTaskStatusSupported = false;
static bool g_hcclCheckFinishByStreamSupported = false;
static bool g_hcclPrintTaskExceptionAllCommSupported = false;
static bool g_hcclLaunchCcoreWaitSupported = false;
static bool g_hcclLaunchCcorePostSupported = false;
static bool g_hcclLaunchOpSupported = false;
static bool g_hcclGetOpArgsSupported = false;
static bool g_hcclFreeOpArgsSupported = false;
static bool g_hcclSetOpSrcDataTypeSupported = false;
static bool g_hcclSetOpDstDataTypeSupported = false;
static bool g_hcclSetOpReduceTypeSupported = false;
static bool g_hcclSetOpCountSupported = false;
static bool g_hcclSetOpAlgConfigSupported = false;
static bool g_hcclSetOpCommEngineSupported = false;
static bool g_hcclCommResPrepareSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHcclGetCommHandleByCtx(void* ctx, void** opHandle) {
    (void)ctx; (void)opHandle;
    HCCL_ERROR("[HcclWrapper] HcclGetCommHandleByCtx not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclReleaseComm(void* opHandle) {
    (void)opHandle;
    HCCL_ERROR("[HcclWrapper] HcclReleaseComm not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetTaskStatus(void* opHandle, void* status) {
    (void)opHandle; (void)status;
    HCCL_ERROR("[HcclWrapper] HcclGetTaskStatus not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCheckFinishByStream(void* opHandle) {
    (void)opHandle;
    HCCL_ERROR("[HcclWrapper] HcclCheckFinishByStream not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclPrintTaskExceptionAllComm(void* opHandle) {
    (void)opHandle;
    HCCL_ERROR("[HcclWrapper] HcclPrintTaskExceptionAllComm not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclLaunchCcoreWait(void* opHandle, uint64_t waitAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast) {
    (void)opHandle; (void)waitAddr; (void)turnNum; (void)turnNumAddr; (void)isLast;
    HCCL_ERROR("[HcclWrapper] HcclLaunchCcoreWait not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclLaunchCcorePost(void* opHandle, uint64_t recordAddr, uint32_t turnNum, uint64_t turnNumAddr) {
    (void)opHandle; (void)recordAddr; (void)turnNum; (void)turnNumAddr;
    HCCL_ERROR("[HcclWrapper] HcclLaunchCcorePost not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclLaunchOp(void* opHandle, void* data) {
    (void)opHandle; (void)data;
    HCCL_ERROR("[HcclWrapper] HcclLaunchOp not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetOpArgs(void** opArgs) {
    (void)opArgs;
    HCCL_ERROR("[HcclWrapper] HcclGetOpArgs not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclFreeOpArgs(void* opArgs) {
    (void)opArgs;
    HCCL_ERROR("[HcclWrapper] HcclFreeOpArgs not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSetOpSrcDataType(void* opArgs, uint8_t srcDataType) {
    (void)opArgs; (void)srcDataType;
    HCCL_ERROR("[HcclWrapper] HcclSetOpSrcDataType not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSetOpDstDataType(void* opArgs, uint8_t dstDataType) {
    (void)opArgs; (void)dstDataType;
    HCCL_ERROR("[HcclWrapper] HcclSetOpDstDataType not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSetOpReduceType(void* opArgs, uint32_t reduceType) {
    (void)opArgs; (void)reduceType;
    HCCL_ERROR("[HcclWrapper] HcclSetOpReduceType not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSetOpCount(void* opArgs, uint64_t count) {
    (void)opArgs; (void)count;
    HCCL_ERROR("[HcclWrapper] HcclSetOpCount not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSetOpAlgConfig(void* opArgs, char* algConfig) {
    (void)opArgs; (void)algConfig;
    HCCL_ERROR("[HcclWrapper] HcclSetOpAlgConfig not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSetOpCommEngine(void* opArgs, uint8_t commEngine) {
    (void)opArgs; (void)commEngine;
    HCCL_ERROR("[HcclWrapper] HcclSetOpCommEngine not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommResPrepare(HcclComm comm, char* opName, void* opArgs, void** addr) {
    (void)comm; (void)opName; (void)opArgs; (void)addr;
    HCCL_ERROR("[HcclWrapper] HcclCommResPrepare not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// ---------- 初始化函数 ----------
void HcclMc2ExDlInit(void* libHcommHandle) {
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

    SET_PTR(hcclGetCommHandleByCtxPtr, "HcclGetCommHandleByCtx", StubHcclGetCommHandleByCtx, g_hcclGetCommHandleByCtxSupported);
    SET_PTR(hcclReleaseCommPtr, "HcclReleaseComm", StubHcclReleaseComm, g_hcclReleaseCommSupported);
    SET_PTR(hcclGetTaskStatusPtr, "HcclGetTaskStatus", StubHcclGetTaskStatus, g_hcclGetTaskStatusSupported);
    SET_PTR(hcclCheckFinishByStreamPtr, "HcclCheckFinishByStream", StubHcclCheckFinishByStream, g_hcclCheckFinishByStreamSupported);
    SET_PTR(hcclPrintTaskExceptionAllCommPtr, "HcclPrintTaskExceptionAllComm", StubHcclPrintTaskExceptionAllComm, g_hcclPrintTaskExceptionAllCommSupported);
    SET_PTR(hcclLaunchCcoreWaitPtr, "HcclLaunchCcoreWait", StubHcclLaunchCcoreWait, g_hcclLaunchCcoreWaitSupported);
    SET_PTR(hcclLaunchCcorePostPtr, "HcclLaunchCcorePost", StubHcclLaunchCcorePost, g_hcclLaunchCcorePostSupported);
    SET_PTR(hcclLaunchOpPtr, "HcclLaunchOp", StubHcclLaunchOp, g_hcclLaunchOpSupported);
    SET_PTR(hcclGetOpArgsPtr, "HcclGetOpArgs", StubHcclGetOpArgs, g_hcclGetOpArgsSupported);
    SET_PTR(hcclFreeOpArgsPtr, "HcclFreeOpArgs", StubHcclFreeOpArgs, g_hcclFreeOpArgsSupported);
    SET_PTR(hcclSetOpSrcDataTypePtr, "HcclSetOpSrcDataType", StubHcclSetOpSrcDataType, g_hcclSetOpSrcDataTypeSupported);
    SET_PTR(hcclSetOpDstDataTypePtr, "HcclSetOpDstDataType", StubHcclSetOpDstDataType, g_hcclSetOpDstDataTypeSupported);
    SET_PTR(hcclSetOpReduceTypePtr, "HcclSetOpReduceType", StubHcclSetOpReduceType, g_hcclSetOpReduceTypeSupported);
    SET_PTR(hcclSetOpCountPtr, "HcclSetOpCount", StubHcclSetOpCount, g_hcclSetOpCountSupported);
    SET_PTR(hcclSetOpAlgConfigPtr, "HcclSetOpAlgConfig", StubHcclSetOpAlgConfig, g_hcclSetOpAlgConfigSupported);
    SET_PTR(hcclSetOpCommEnginePtr, "HcclSetOpCommEngine", StubHcclSetOpCommEngine, g_hcclSetOpCommEngineSupported);
    SET_PTR(hcclCommResPreparePtr, "HcclCommResPrepare", StubHcclCommResPrepare, g_hcclCommResPrepareSupported);

    #undef SET_PTR
}

void HcclMc2ExDlFini(void) {
    #define RESET_PTR(ptr, stub, support_flag) do { ptr = stub; support_flag = false; } while(0)

    RESET_PTR(hcclGetCommHandleByCtxPtr, StubHcclGetCommHandleByCtx, g_hcclGetCommHandleByCtxSupported);
    RESET_PTR(hcclReleaseCommPtr, StubHcclReleaseComm, g_hcclReleaseCommSupported);
    RESET_PTR(hcclGetTaskStatusPtr, StubHcclGetTaskStatus, g_hcclGetTaskStatusSupported);
    RESET_PTR(hcclCheckFinishByStreamPtr, StubHcclCheckFinishByStream, g_hcclCheckFinishByStreamSupported);
    RESET_PTR(hcclPrintTaskExceptionAllCommPtr, StubHcclPrintTaskExceptionAllComm, g_hcclPrintTaskExceptionAllCommSupported);
    RESET_PTR(hcclLaunchCcoreWaitPtr, StubHcclLaunchCcoreWait, g_hcclLaunchCcoreWaitSupported);
    RESET_PTR(hcclLaunchCcorePostPtr, StubHcclLaunchCcorePost, g_hcclLaunchCcorePostSupported);
    RESET_PTR(hcclLaunchOpPtr, StubHcclLaunchOp, g_hcclLaunchOpSupported);
    RESET_PTR(hcclGetOpArgsPtr, StubHcclGetOpArgs, g_hcclGetOpArgsSupported);
    RESET_PTR(hcclFreeOpArgsPtr, StubHcclFreeOpArgs, g_hcclFreeOpArgsSupported);
    RESET_PTR(hcclSetOpSrcDataTypePtr, StubHcclSetOpSrcDataType, g_hcclSetOpSrcDataTypeSupported);
    RESET_PTR(hcclSetOpDstDataTypePtr, StubHcclSetOpDstDataType, g_hcclSetOpDstDataTypeSupported);
    RESET_PTR(hcclSetOpReduceTypePtr, StubHcclSetOpReduceType, g_hcclSetOpReduceTypeSupported);
    RESET_PTR(hcclSetOpCountPtr, StubHcclSetOpCount, g_hcclSetOpCountSupported);
    RESET_PTR(hcclSetOpAlgConfigPtr, StubHcclSetOpAlgConfig, g_hcclSetOpAlgConfigSupported);
    RESET_PTR(hcclSetOpCommEnginePtr, StubHcclSetOpCommEngine, g_hcclSetOpCommEngineSupported);
    RESET_PTR(hcclCommResPreparePtr, StubHcclCommResPrepare, g_hcclCommResPrepareSupported);

    #undef RESET_PTR
}

// ---------- 对外API实现 ----------
HcclResult HcclGetCommHandleByCtx(void* ctx, void** opHandle) {
    return hcclGetCommHandleByCtxPtr(ctx, opHandle);
}
HcclResult HcclReleaseComm(void* opHandle) {
    return hcclReleaseCommPtr(opHandle);
}
HcclResult HcclGetTaskStatus(void* opHandle, void* status) {
    return hcclGetTaskStatusPtr(opHandle, status);
}
HcclResult HcclCheckFinishByStream(void* opHandle) {
    return hcclCheckFinishByStreamPtr(opHandle);
}
HcclResult HcclPrintTaskExceptionAllComm(void* opHandle) {
    return hcclPrintTaskExceptionAllCommPtr(opHandle);
}
HcclResult HcclLaunchCcoreWait(void* opHandle, uint64_t waitAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast) {
    return hcclLaunchCcoreWaitPtr(opHandle, waitAddr, turnNum, turnNumAddr, isLast);
}
HcclResult HcclLaunchCcorePost(void* opHandle, uint64_t recordAddr, uint32_t turnNum, uint64_t turnNumAddr) {
    return hcclLaunchCcorePostPtr(opHandle, recordAddr, turnNum, turnNumAddr);
}
HcclResult HcclLaunchOp(void* opHandle, void* data) {
    return hcclLaunchOpPtr(opHandle, data);
}

HcclResult HcclGetOpArgs(void** opArgs) {
    return hcclGetOpArgsPtr(opArgs);
}
HcclResult HcclFreeOpArgs(void* opArgs) {
    return hcclFreeOpArgsPtr(opArgs);
}
HcclResult HcclSetOpSrcDataType(void* opArgs, uint8_t srcDataType) {
    return hcclSetOpSrcDataTypePtr(opArgs, srcDataType);
}
HcclResult HcclSetOpDstDataType(void* opArgs, uint8_t dstDataType) {
    return hcclSetOpDstDataTypePtr(opArgs, dstDataType);
}
HcclResult HcclSetOpReduceType(void* opArgs, uint32_t reduceType) {
    return hcclSetOpReduceTypePtr(opArgs, reduceType);
}
HcclResult HcclSetOpCount(void* opArgs, uint64_t count) {
    return hcclSetOpCountPtr(opArgs, count);
}
HcclResult HcclSetOpAlgConfig(void* opArgs, char* algConfig) {
    return hcclSetOpAlgConfigPtr(opArgs, algConfig);
}
HcclResult HcclSetOpCommEngine(void* opArgs, uint8_t commEngine) {
    return hcclSetOpCommEnginePtr(opArgs, commEngine);
}
HcclResult HcclCommResPrepare(HcclComm comm, char* opName, void* opArgs, void** addr) {
    return hcclCommResPreparePtr(comm, opName, opArgs, addr);
}

// ---------- 查询函数实现 ----------
extern "C" bool HcommIsSupportHcclGetCommHandleByCtx(void) {
    return g_hcclGetCommHandleByCtxSupported;
}
extern "C" bool HcommIsSupportHcclReleaseComm(void) {
    return g_hcclReleaseCommSupported;
}
extern "C" bool HcommIsSupportHcclGetTaskStatus(void) {
    return g_hcclGetTaskStatusSupported;
}
extern "C" bool HcommIsSupportHcclCheckFinishByStream(void) {
    return g_hcclCheckFinishByStreamSupported;
}
extern "C" bool HcommIsSupportHcclPrintTaskExceptionAllComm(void) {
    return g_hcclPrintTaskExceptionAllCommSupported;
}
extern "C" bool HcommIsSupportHcclLaunchCcoreWait(void) {
    return g_hcclLaunchCcoreWaitSupported;
}
extern "C" bool HcommIsSupportHcclLaunchCcorePost(void) {
    return g_hcclLaunchCcorePostSupported;
}
extern "C" bool HcommIsSupportHcclLaunchOp(void) {
    return g_hcclLaunchOpSupported;
}
extern "C" bool HcommIsSupportHcclGetOpArgs(void) {
    return g_hcclGetOpArgsSupported;
}
extern "C" bool HcommIsSupportHcclFreeOpArgs(void) {
    return g_hcclFreeOpArgsSupported;
}
extern "C" bool HcommIsSupportHcclSetOpSrcDataType(void) {
    return g_hcclSetOpSrcDataTypeSupported;
}
extern "C" bool HcommIsSupportHcclSetOpDstDataType(void) {
    return g_hcclSetOpDstDataTypeSupported;
}
extern "C" bool HcommIsSupportHcclSetOpReduceType(void) {
    return g_hcclSetOpReduceTypeSupported;
}
extern "C" bool HcommIsSupportHcclSetOpCount(void) {
    return g_hcclSetOpCountSupported;
}
extern "C" bool HcommIsSupportHcclSetOpAlgConfig(void) {
    return g_hcclSetOpAlgConfigSupported;
}
extern "C" bool HcommIsSupportHcclSetOpCommEngine(void) {
    return g_hcclSetOpCommEngineSupported;
}
extern "C" bool HcommIsSupportHcclCommResPrepare(void) {
    return g_hcclCommResPrepareSupported;
}