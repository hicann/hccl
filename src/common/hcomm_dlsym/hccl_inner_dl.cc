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
#include "hccl_inner_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
HcclResult (*hcclAllReduceInnerPtr)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclBroadcastInnerPtr)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclReduceScatterInnerPtr)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclReduceScatterVInnerPtr)(void*, const void*, const void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclScatterInnerPtr)(void*, void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclAllGatherInnerPtr)(void*, void*, uint64_t, HcclDataType, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclAllGatherVInnerPtr)(void*, uint64_t, void*, const void*, const void*, HcclDataType, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclSendInnerPtr)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclRecvInnerPtr)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclAlltoAllVCInnerPtr)(const void*, const void*, HcclDataType, const void*, HcclDataType, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclAlltoAllVInnerPtr)(const void*, const void*, const void*, HcclDataType, const void*, const void*, const void*, HcclDataType, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclAlltoAllInnerPtr)(const void*, uint64_t, HcclDataType, const void*, uint64_t, HcclDataType, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclReduceInnerPtr)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, uint32_t, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclBatchSendRecvInnerPtr)(HcclSendRecvItem*, uint32_t, HcclComm, aclrtStream) = nullptr;
HcclResult (*hcclCreateOpResCtxInnerPtr)(HcclComm, uint8_t, HcclDataType, HcclDataType, HcclReduceOp, uint64_t, char*, uint32_t, void**) = nullptr;

// 添加支持标志（静态，默认 false）
static bool g_hcclAllReduceInnerSupported = false;
static bool g_hcclBroadcastInnerSupported = false;
static bool g_hcclReduceScatterInnerSupported = false;
static bool g_hcclReduceScatterVInnerSupported = false;
static bool g_hcclScatterInnerSupported = false;
static bool g_hcclAllGatherInnerSupported = false;
static bool g_hcclAllGatherVInnerSupported = false;
static bool g_hcclSendInnerSupported = false;
static bool g_hcclRecvInnerSupported = false;
static bool g_hcclAlltoAllVCInnerSupported = false;
static bool g_hcclAlltoAllVInnerSupported = false;
static bool g_hcclAlltoAllInnerSupported = false;
static bool g_hcclReduceInnerSupported = false;
static bool g_hcclBatchSendRecvInnerSupported = false;
static bool g_hcclCreateOpResCtxInnerSupported = false;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static HcclResult StubHcclAllReduceInner(void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType,
                                         HcclReduceOp op, HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)recvBuf; (void)count; (void)dataType; (void)op; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclAllReduceInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclBroadcastInner(void* buf, uint64_t count, HcclDataType dataType, uint32_t root,
                                         HcclComm comm, aclrtStream stream) {
    (void)buf; (void)count; (void)dataType; (void)root; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclBroadcastInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclReduceScatterInner(void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
                                             HcclReduceOp op, HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)recvBuf; (void)recvCount; (void)dataType; (void)op; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclReduceScatterInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclReduceScatterVInner(void* sendBuf, const void* sendCounts, const void* sendDispls,
                                              void* recvBuf, uint64_t recvCount, HcclDataType dataType,
                                              HcclReduceOp op, HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)sendCounts; (void)sendDispls; (void)recvBuf; (void)recvCount; (void)dataType; (void)op; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclReduceScatterVInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclScatterInner(void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
                                       uint32_t root, HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)recvBuf; (void)recvCount; (void)dataType; (void)root; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclScatterInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclAllGatherInner(void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType,
                                         HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)recvBuf; (void)sendCount; (void)dataType; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclAllGatherInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclAllGatherVInner(void* sendBuf, uint64_t sendCount, void* recvBuf,
                                          const void* recvCounts, const void* recvDispls,
                                          HcclDataType dataType, HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)sendCount; (void)recvBuf; (void)recvCounts; (void)recvDispls; (void)dataType; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclAllGatherVInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSendInner(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                                    HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)count; (void)dataType; (void)destRank; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclSendInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclRecvInner(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                                    HcclComm comm, aclrtStream stream) {
    (void)recvBuf; (void)count; (void)dataType; (void)srcRank; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclRecvInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclAlltoAllVCInner(const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType,
                                          const void* recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)sendCountMatrix; (void)sendType; (void)recvBuf; (void)recvType; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclAlltoAllVCInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclAlltoAllVInner(const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType,
                                         const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType recvType,
                                         HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)sendCounts; (void)sdispls; (void)sendType; (void)recvBuf; (void)recvCounts; (void)rdispls; (void)recvType; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclAlltoAllVInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclAlltoAllInner(const void* sendBuf, uint64_t sendCount, HcclDataType sendType,
                                        const void* recvBuf, uint64_t recvCount, HcclDataType recvType,
                                        HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)sendCount; (void)sendType; (void)recvBuf; (void)recvCount; (void)recvType; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclAlltoAllInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclReduceInner(void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType,
                                      HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream) {
    (void)sendBuf; (void)recvBuf; (void)count; (void)dataType; (void)op; (void)root; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclReduceInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclBatchSendRecvInner(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream) {
    (void)sendRecvInfo; (void)itemNum; (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclBatchSendRecvInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCreateOpResCtxInner(HcclComm comm, uint8_t opType, HcclDataType srcDataType, HcclDataType dstDataType,
                                              HcclReduceOp reduceType, uint64_t count, char* algConfig, uint32_t commEngine, void** opResCtx) {
    (void)comm; (void)opType; (void)srcDataType; (void)dstDataType; (void)reduceType; (void)count; (void)algConfig; (void)commEngine; (void)opResCtx;
    HCCL_ERROR("[HcclWrapper] HcclCreateOpResCtxInner not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// 初始化
void HcclInnerDlInit(void* libHcommHandle) {
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

    SET_PTR(hcclAllReduceInnerPtr, libHcommHandle, "HcclAllReduceInner", StubHcclAllReduceInner, g_hcclAllReduceInnerSupported);
    SET_PTR(hcclBroadcastInnerPtr, libHcommHandle, "HcclBroadcastInner", StubHcclBroadcastInner, g_hcclBroadcastInnerSupported);
    SET_PTR(hcclReduceScatterInnerPtr, libHcommHandle, "HcclReduceScatterInner", StubHcclReduceScatterInner, g_hcclReduceScatterInnerSupported);
    SET_PTR(hcclReduceScatterVInnerPtr, libHcommHandle, "HcclReduceScatterVInner", StubHcclReduceScatterVInner, g_hcclReduceScatterVInnerSupported);
    SET_PTR(hcclScatterInnerPtr, libHcommHandle, "HcclScatterInner", StubHcclScatterInner, g_hcclScatterInnerSupported);
    SET_PTR(hcclAllGatherInnerPtr, libHcommHandle, "HcclAllGatherInner", StubHcclAllGatherInner, g_hcclAllGatherInnerSupported);
    SET_PTR(hcclAllGatherVInnerPtr, libHcommHandle, "HcclAllGatherVInner", StubHcclAllGatherVInner, g_hcclAllGatherVInnerSupported);
    SET_PTR(hcclSendInnerPtr, libHcommHandle, "HcclSendInner", StubHcclSendInner, g_hcclSendInnerSupported);
    SET_PTR(hcclRecvInnerPtr, libHcommHandle, "HcclRecvInner", StubHcclRecvInner, g_hcclRecvInnerSupported);
    SET_PTR(hcclAlltoAllVCInnerPtr, libHcommHandle, "HcclAlltoAllVCInner", StubHcclAlltoAllVCInner, g_hcclAlltoAllVCInnerSupported);
    SET_PTR(hcclAlltoAllVInnerPtr, libHcommHandle, "HcclAlltoAllVInner", StubHcclAlltoAllVInner, g_hcclAlltoAllVInnerSupported);
    SET_PTR(hcclAlltoAllInnerPtr, libHcommHandle, "HcclAlltoAllInner", StubHcclAlltoAllInner, g_hcclAlltoAllInnerSupported);
    SET_PTR(hcclReduceInnerPtr, libHcommHandle, "HcclReduceInner", StubHcclReduceInner, g_hcclReduceInnerSupported);
    SET_PTR(hcclBatchSendRecvInnerPtr, libHcommHandle, "HcclBatchSendRecvInner", StubHcclBatchSendRecvInner, g_hcclBatchSendRecvInnerSupported);
    SET_PTR(hcclCreateOpResCtxInnerPtr, libHcommHandle, "HcclCreateOpResCtxInner", StubHcclCreateOpResCtxInner, g_hcclCreateOpResCtxInnerSupported);

    #undef SET_PTR
}

void HcclInnerDlFini(void) {
    // 重置为桩函数，防止fini后误用
    hcclAllReduceInnerPtr = StubHcclAllReduceInner;
    g_hcclAllReduceInnerSupported = false;
    hcclBroadcastInnerPtr = StubHcclBroadcastInner;
    g_hcclBroadcastInnerSupported = false;
    hcclReduceScatterInnerPtr = StubHcclReduceScatterInner;
    g_hcclReduceScatterInnerSupported = false;
    hcclReduceScatterVInnerPtr = StubHcclReduceScatterVInner;
    g_hcclReduceScatterVInnerSupported = false;
    hcclScatterInnerPtr = StubHcclScatterInner;
    g_hcclScatterInnerSupported = false;
    hcclAllGatherInnerPtr = StubHcclAllGatherInner;
    g_hcclAllGatherInnerSupported = false;
    hcclAllGatherVInnerPtr = StubHcclAllGatherVInner;
    g_hcclAllGatherVInnerSupported = false;
    hcclSendInnerPtr = StubHcclSendInner;
    g_hcclSendInnerSupported = false;
    hcclRecvInnerPtr = StubHcclRecvInner;
    g_hcclRecvInnerSupported = false;
    hcclAlltoAllVCInnerPtr = StubHcclAlltoAllVCInner;
    g_hcclAlltoAllVCInnerSupported = false;
    hcclAlltoAllVInnerPtr = StubHcclAlltoAllVInner;
    g_hcclAlltoAllVInnerSupported = false;
    hcclAlltoAllInnerPtr = StubHcclAlltoAllInner;
    g_hcclAlltoAllInnerSupported = false;
    hcclReduceInnerPtr = StubHcclReduceInner;
    g_hcclReduceInnerSupported = false;
    hcclBatchSendRecvInnerPtr = StubHcclBatchSendRecvInner;
    g_hcclBatchSendRecvInnerSupported = false;
    hcclCreateOpResCtxInnerPtr = StubHcclCreateOpResCtxInner;
    g_hcclCreateOpResCtxInnerSupported = false;
}

// ---------- 对外提供的查询接口（判断函数是否存在）----------
extern "C" bool HcommIsSupportHcclAllReduceInner(void) {
    return g_hcclAllReduceInnerSupported;
}
extern "C" bool HcommIsSupportHcclBroadcastInner(void) {
    return g_hcclBroadcastInnerSupported;
}
extern "C" bool HcommIsSupportHcclReduceScatterInner(void) {
    return g_hcclReduceScatterInnerSupported;
}
extern "C" bool HcommIsSupportHcclReduceScatterVInner(void) {
    return g_hcclReduceScatterVInnerSupported;
}
extern "C" bool HcommIsSupportHcclScatterInner(void) {
    return g_hcclScatterInnerSupported;
}
extern "C" bool HcommIsSupportHcclAllGatherInner(void) {
    return g_hcclAllGatherInnerSupported;
}
extern "C" bool HcommIsSupportHcclAllGatherVInner(void) {
    return g_hcclAllGatherVInnerSupported;
}
extern "C" bool HcommIsSupportHcclSendInner(void) {
    return g_hcclSendInnerSupported;
}
extern "C" bool HcommIsSupportHcclRecvInner(void) {
    return g_hcclRecvInnerSupported;
}
extern "C" bool HcommIsSupportHcclAlltoAllVCInner(void) {
    return g_hcclAlltoAllVCInnerSupported;
}
extern "C" bool HcommIsSupportHcclAlltoAllVInner(void) {
    return g_hcclAlltoAllVInnerSupported;
}
extern "C" bool HcommIsSupportHcclAlltoAllInner(void) {
    return g_hcclAlltoAllInnerSupported;
}
extern "C" bool HcommIsSupportHcclReduceInner(void) {
    return g_hcclReduceInnerSupported;
}
extern "C" bool HcommIsSupportHcclBatchSendRecvInner(void) {
    return g_hcclBatchSendRecvInnerSupported;
}
extern "C" bool HcommIsSupportHcclCreateOpResCtxInner(void) {
    return g_hcclCreateOpResCtxInnerSupported;
}