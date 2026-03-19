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
#include "hccl_comm_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
HcclResult (*hcclGetRankIdPtr)(HcclComm, uint32_t*) = NULL;
HcclResult (*hcclGetRankSizePtr)(HcclComm, uint32_t*) = NULL;
HcclResult (*hcclCommInitClusterInfoPtr)(const char*, uint32_t, HcclComm*) = NULL;
HcclResult (*hcclCommInitClusterInfoConfigPtr)(const char*, uint32_t, HcclCommConfig*, HcclComm*) = NULL;
HcclResult (*hcclCreateSubCommConfigPtr)(HcclComm*, uint32_t, uint32_t*, uint64_t, uint32_t, HcclCommConfig*, HcclComm*) = NULL;
HcclResult (*hcclGetRootInfoPtr)(HcclRootInfo*) = NULL;
HcclResult (*hcclCommInitRootInfoPtr)(uint32_t, const HcclRootInfo*, uint32_t, HcclComm*) = NULL;
HcclResult (*hcclCommInitRootInfoConfigPtr)(uint32_t, const HcclRootInfo*, uint32_t, const HcclCommConfig*, HcclComm*) = NULL;
HcclResult (*hcclSetConfigPtr)(HcclConfig, HcclConfigValue) = NULL;
HcclResult (*hcclGetConfigPtr)(HcclConfig, HcclConfigValue*) = NULL;
HcclResult (*hcclGetCommNamePtr)(HcclComm, char*) = NULL;
HcclResult (*hcclCommGetHandleWithNamePtr)(const char*, HcclComm*) = NULL;
HcclResult (*hcclBarrierPtr)(HcclComm, aclrtStream) = NULL;
HcclResult (*hcclCommDestroyPtr)(HcclComm) = NULL;
HcclResult (*hcclCommInitAllPtr)(uint32_t, int32_t*, HcclComm*) = NULL;
HcclResult (*hcclGetCommAsyncErrorPtr)(HcclComm, HcclResult*) = NULL;
const char* (*hcclGetErrorStringPtr)(HcclResult) = NULL;
uint32_t (*hcclGetCommConfigCapabilityPtr)(void) = NULL;
HcclResult (*hcclCommSuspendPtr)(HcclComm) = NULL;
HcclResult (*hcclCommResumePtr)(HcclComm) = NULL;
HcclResult (*hcclCommSetMemoryRangePtr)(HcclComm, void*, size_t, size_t, uint64_t) = NULL;
HcclResult (*hcclCommUnsetMemoryRangePtr)(HcclComm, void*) = NULL;
HcclResult (*hcclCommActivateCommMemoryPtr)(HcclComm, void*, size_t, size_t, aclrtDrvMemHandle, uint64_t) = NULL;
HcclResult (*hcclCommDeactivateCommMemoryPtr)(HcclComm, void*) = NULL;
HcclResult (*hcclCommWorkingDevNicSetPtr)(HcclComm, uint32_t*, bool*, uint32_t) = NULL;
HcclResult (*hcclGroupStartPtr)(void) = NULL;
HcclResult (*hcclGroupEndPtr)(void) = NULL;
HcclResult (*hcclCommSymWinRegisterPtr)(HcclComm, void*, uint64_t, CommSymWindow*, uint32_t) = NULL;
HcclResult (*hcclCommSymWinDeregisterPtr)(CommSymWindow) = NULL;
HcclResult (*hcclCommSymWinGetPtr)(HcclComm, void*, size_t, CommSymWindow*, size_t*) = NULL;
static HcclResult (*hcclGetRawCommHandlePtr)(const char*, HcclComm*) = NULL;
static HcclResult (*hcclGetCcuTaskInfoPtr)(HcclComm, void*, void*) = NULL;
static HcclResult (*commGetLocalCCLBufPtr)(HcclComm, void**, uint64_t*) = NULL;
static HcclResult (*commGetRemoteCCLBufPtr)(HcclComm, uint32_t, void**, uint64_t*) = NULL;
static HcclResult (*commGetKFCWorkSpacePtr)(HcclComm, void**, uint64_t*) = NULL;
static HcclResult (*commGetCCLBufSizeCfgPtr)(HcclComm, uint64_t*) = NULL;
HcclResult (*hcclCommInitClusterInfoMemConfigPtr)(const char *rankTableString, uint32_t rank,
                                            HcclCommConfig *config, HcclComm *comm) = NULL;
static HcclResult (*hcclSnapshotSavePtr)(void*, uint32_t, uint32_t) = NULL;
static HcclResult (*hcclSnapshotGetBufSizePtr)(uint32_t, uint32_t*) = NULL;
static HcclResult (*hcclSnapshotRecoverAllCommsPtr)(const char*, const char*, void*, uint32_t) = NULL;

// 添加支持标志（静态，默认 false）
static bool g_hcclGetRankIdSupported = false;
static bool g_hcclGetRankSizeSupported = false;
static bool g_hcclCommInitClusterInfoSupported = false;
static bool g_hcclCommInitClusterInfoConfigSupported = false;
static bool g_hcclCreateSubCommConfigSupported = false;
static bool g_hcclGetRootInfoSupported = false;
static bool g_hcclCommInitRootInfoSupported = false;
static bool g_hcclCommInitRootInfoConfigSupported = false;
static bool g_hcclSetConfigSupported = false;
static bool g_hcclGetConfigSupported = false;
static bool g_hcclGetCommNameSupported = false;
static bool g_hcclCommGetHandleWithNameSupported = false;
static bool g_hcclBarrierSupported = false;
static bool g_hcclCommDestroySupported = false;
static bool g_hcclCommInitAllSupported = false;
static bool g_hcclGetCommAsyncErrorSupported = false;
static bool g_hcclGetErrorStringSupported = false;
static bool g_hcclGetCommConfigCapabilitySupported = false;
static bool g_hcclCommSuspendSupported = false;
static bool g_hcclCommResumeSupported = false;
static bool g_hcclCommSetMemoryRangeSupported = false;
static bool g_hcclCommUnsetMemoryRangeSupported = false;
static bool g_hcclCommActivateCommMemorySupported = false;
static bool g_hcclCommDeactivateCommMemorySupported = false;
static bool g_hcclCommWorkingDevNicSetSupported = false;
static bool g_hcclGroupStartSupported = false;
static bool g_hcclGroupEndSupported = false;
static bool g_hcclCommSymWinRegisterSupported = false;
static bool g_hcclCommSymWinDeregisterSupported = false;
static bool g_hcclCommSymWinGetSupported = false;
static bool g_hcclGetRawCommHandleSupported = false;
static bool g_hcclGetCcuTaskInfoSupported = false;
static bool g_commGetLocalCCLBufSupported = false;
static bool g_commGetRemoteCCLBufSupported = false;
static bool g_commGetKFCWorkSpaceSupported = false;
static bool g_commGetCCLBufSizeCfgSupported = false;
static bool g_hcclCommInitClusterInfoMemConfigSupported = false;
static bool g_hcclSnapshotSaveSupported = false;
static bool g_hcclSnapshotGetBufSizeSupported = false;
static bool g_hcclSnapshotRecoverAllCommsSupported = false;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static HcclResult StubHcclGetRankId(HcclComm comm, uint32_t* rank) {
    (void)comm; (void)rank;
    HCCL_ERROR("[HcclWrapper] HcclGetRankId not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclGetRankSize(HcclComm comm, uint32_t* rankSize) {
    (void)comm; (void)rankSize;
    HCCL_ERROR("[HcclWrapper] HcclGetRankSize not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm) {
    (void)clusterInfo; (void)rank; (void)comm;
    HCCL_ERROR("[HcclWrapper] HcclCommInitClusterInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm) {
    (void)clusterInfo; (void)rank; (void)config; (void)comm;
    HCCL_ERROR("[HcclWrapper] HcclCommInitClusterInfoConfig not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCreateSubCommConfig(HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId, HcclCommConfig* config, HcclComm* subComm) {
    (void)comm; (void)rankNum; (void)rankIds; (void)subCommId; (void)subCommRankId; (void)config; (void)subComm;
    HCCL_ERROR("[HcclWrapper] HcclCreateSubCommConfig not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetRootInfo(HcclRootInfo* rootInfo) {
    (void)rootInfo;
    HCCL_ERROR("[HcclWrapper] HcclGetRootInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm) {
    (void)nRanks; (void)rootInfo; (void)rank; (void)comm;
    HCCL_ERROR("[HcclWrapper] HcclCommInitRootInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm) {
    (void)nRanks; (void)rootInfo; (void)rank; (void)config; (void)comm;
    HCCL_ERROR("[HcclWrapper] HcclCommInitRootInfoConfig not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSetConfig(HcclConfig config, HcclConfigValue configValue) {
    (void)config; (void)configValue;
    HCCL_ERROR("[HcclWrapper] HcclSetConfig not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetConfig(HcclConfig config, HcclConfigValue* configValue) {
    (void)config; (void)configValue;
    HCCL_ERROR("[HcclWrapper] HcclGetConfig not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetCommName(HcclComm comm, char* commName) {
    (void)comm; (void)commName;
    HCCL_ERROR("[HcclWrapper] HcclGetCommName not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommGetHandleWithName(const char* commName, HcclComm* comm) {
    (void)commName; (void)comm;
    HCCL_ERROR("[HcclWrapper] HcclCommGetHandleWithName not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclBarrier(HcclComm comm, aclrtStream stream) {
    (void)comm; (void)stream;
    HCCL_ERROR("[HcclWrapper] HcclBarrier not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommDestroy(HcclComm comm) {
    (void)comm;
    HCCL_ERROR("[HcclWrapper] HcclCommDestroy not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms) {
    (void)ndev; (void)devices; (void)comms;
    HCCL_ERROR("[HcclWrapper] HcclCommInitAll not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError) {
    (void)comm; (void)asyncError;
    HCCL_ERROR("[HcclWrapper] HcclGetCommAsyncError not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static const char* StubHcclGetErrorString(HcclResult code) {
    (void)code;
    HCCL_ERROR("[HcclWrapper] HcclGetErrorString not supported");
    return "";
}
static uint32_t StubHcclGetCommConfigCapability(void) {
    HCCL_ERROR("[HcclWrapper] HcclGetCommConfigCapability not supported");
    return 0;
}
static HcclResult StubHcclCommSuspend(HcclComm comm) {
    (void)comm;
    HCCL_ERROR("[HcclWrapper] HcclCommSuspend not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommResume(HcclComm comm) {
    (void)comm;
    HCCL_ERROR("[HcclWrapper] HcclCommResume not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommSetMemoryRange(HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags) {
    (void)comm; (void)baseVirPtr; (void)size; (void)alignment; (void)flags;
    HCCL_ERROR("[HcclWrapper] HcclCommSetMemoryRange not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommUnsetMemoryRange(HcclComm comm, void* baseVirPtr) {
    (void)comm; (void)baseVirPtr;
    HCCL_ERROR("[HcclWrapper] HcclCommUnsetMemoryRange not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommActivateCommMemory(HcclComm comm, void* virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags) {
    (void)comm; (void)virPtr; (void)size; (void)offset; (void)handle; (void)flags;
    HCCL_ERROR("[HcclWrapper] HcclCommActivateCommMemory not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommDeactivateCommMemory(HcclComm comm, void* virPtr) {
    (void)comm; (void)virPtr;
    HCCL_ERROR("[HcclWrapper] HcclCommDeactivateCommMemory not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommWorkingDevNicSet(HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks) {
    (void)comm; (void)ranks; (void)useBackup; (void)nRanks;
    HCCL_ERROR("[HcclWrapper] HcclCommWorkingDevNicSet not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGroupStart(void) {
    HCCL_ERROR("[HcclWrapper] HcclGroupStart not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGroupEnd(void) {
    HCCL_ERROR("[HcclWrapper] HcclGroupEnd not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, CommSymWindow* winHandle, uint32_t flag) {
    (void)comm; (void)addr; (void)size; (void)winHandle; (void)flag;
    HCCL_ERROR("[HcclWrapper] HcclCommSymWinRegister not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommSymWinDeregister(CommSymWindow winHandle) {
    (void)winHandle;
    HCCL_ERROR("[HcclWrapper] HcclCommSymWinDeregister not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, CommSymWindow* winHandle, size_t* offset) {
    (void)comm; (void)ptr; (void)size; (void)winHandle; (void)offset;
    HCCL_ERROR("[HcclWrapper] HcclCommSymWinGet not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetRawCommHandle(const char* commName, HcclComm* commHandle) {
    (void)commName; (void)commHandle;
    HCCL_ERROR("[HcclWrapper] HcclGetRawCommHandle not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclGetCcuTaskInfo(HcclComm comm, void* tilingData, void* ccuTaskGroup) {
    (void)comm; (void)tilingData; (void)ccuTaskGroup;
    HCCL_ERROR("[HcclWrapper] HcclGetCcuTaskInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubCommGetLocalCCLBuf(HcclComm comm, void** addr, uint64_t* size) {
    (void)comm; (void)addr; (void)size;
    HCCL_ERROR("[HcclWrapper] CommGetLocalCCLBuf not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubCommGetRemoteCCLBuf(HcclComm comm, uint32_t remoteRank, void** addr, uint64_t* size) {
    (void)comm; (void)remoteRank; (void)addr; (void)size;
    HCCL_ERROR("[HcclWrapper] CommGetRemoteCCLBuf not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubCommGetKFCWorkSpace(HcclComm comm, void** addr, uint64_t* size) {
    (void)comm; (void)addr; (void)size;
    HCCL_ERROR("[HcclWrapper] CommGetKFCWorkSpace not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubCommGetCCLBufSizeCfg(HcclComm comm, uint64_t* cclBufSize) {
    (void)comm; (void)cclBufSize;
    HCCL_ERROR("[HcclWrapper] CommGetCCLBufSizeCfg not supported");
    return HCCL_E_NOT_SUPPORTED;
}
HcclResult StubHcclCommInitClusterInfoMemConfig(const char *rankTableString, uint32_t rank,
                                            HcclCommConfig *config, HcclComm *comm)
{
    (void)rankTableString; (void)rank; (void)config; (void)comm;
    HCCL_ERROR("[HcclWrapper] HcclCommInitClusterInfoMemConfig not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSnapshotSave(void* snapshotBuf, uint32_t size, uint32_t step) {
    (void)snapshotBuf; (void)size; (void)step;
    HCCL_ERROR("[HcclWrapper] HcclSnapshotSave not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSnapshotGetBufSize(uint32_t step, uint32_t* size) {
    (void)step; (void)size;
    HCCL_ERROR("[HcclWrapper] HcclSnapshotGetBufSize not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcclSnapshotRecoverAllComms(const char* clusterInfo, const char* changedInfo,
                                                   void* snapshotBuf, uint32_t snapshotBufSize) {
    (void)clusterInfo; (void)changedInfo; (void)snapshotBuf; (void)snapshotBufSize;
    HCCL_ERROR("[HcclWrapper] HcclSnapshotRecoverAllComms not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// ---------- 初始化函数 ----------
void HcclCommDlInit(void* libHcommHandle) {
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

    SET_PTR(hcclGetRankIdPtr, "HcclGetRankId", StubHcclGetRankId, g_hcclGetRankIdSupported);
    SET_PTR(hcclGetRankSizePtr, "HcclGetRankSize", StubHcclGetRankSize, g_hcclGetRankSizeSupported);
    SET_PTR(hcclCommInitClusterInfoPtr, "HcclCommInitClusterInfo", StubHcclCommInitClusterInfo, g_hcclCommInitClusterInfoSupported);
    SET_PTR(hcclCommInitClusterInfoConfigPtr, "HcclCommInitClusterInfoConfig", StubHcclCommInitClusterInfoConfig, g_hcclCommInitClusterInfoConfigSupported);
    SET_PTR(hcclCreateSubCommConfigPtr, "HcclCreateSubCommConfig", StubHcclCreateSubCommConfig, g_hcclCreateSubCommConfigSupported);
    SET_PTR(hcclGetRootInfoPtr, "HcclGetRootInfo", StubHcclGetRootInfo, g_hcclGetRootInfoSupported);
    SET_PTR(hcclCommInitRootInfoPtr, "HcclCommInitRootInfo", StubHcclCommInitRootInfo, g_hcclCommInitRootInfoSupported);
    SET_PTR(hcclCommInitRootInfoConfigPtr, "HcclCommInitRootInfoConfig", StubHcclCommInitRootInfoConfig, g_hcclCommInitRootInfoConfigSupported);
    SET_PTR(hcclSetConfigPtr, "HcclSetConfig", StubHcclSetConfig, g_hcclSetConfigSupported);
    SET_PTR(hcclGetConfigPtr, "HcclGetConfig", StubHcclGetConfig, g_hcclGetConfigSupported);
    SET_PTR(hcclGetCommNamePtr, "HcclGetCommName", StubHcclGetCommName, g_hcclGetCommNameSupported);
    SET_PTR(hcclCommGetHandleWithNamePtr, "HcclCommGetHandleWithName", StubHcclCommGetHandleWithName, g_hcclCommGetHandleWithNameSupported);
    SET_PTR(hcclBarrierPtr, "HcclBarrier", StubHcclBarrier, g_hcclBarrierSupported);
    SET_PTR(hcclCommDestroyPtr, "HcclCommDestroy", StubHcclCommDestroy, g_hcclCommDestroySupported);
    SET_PTR(hcclCommInitAllPtr, "HcclCommInitAll", StubHcclCommInitAll, g_hcclCommInitAllSupported);
    SET_PTR(hcclGetCommAsyncErrorPtr, "HcclGetCommAsyncError", StubHcclGetCommAsyncError, g_hcclGetCommAsyncErrorSupported);
    SET_PTR(hcclGetErrorStringPtr, "HcclGetErrorString", StubHcclGetErrorString, g_hcclGetErrorStringSupported);
    SET_PTR(hcclGetCommConfigCapabilityPtr, "HcclGetCommConfigCapability", StubHcclGetCommConfigCapability, g_hcclGetCommConfigCapabilitySupported);
    SET_PTR(hcclCommSuspendPtr, "HcclCommSuspend", StubHcclCommSuspend, g_hcclCommSuspendSupported);
    SET_PTR(hcclCommResumePtr, "HcclCommResume", StubHcclCommResume, g_hcclCommResumeSupported);
    SET_PTR(hcclCommSetMemoryRangePtr, "HcclCommSetMemoryRange", StubHcclCommSetMemoryRange, g_hcclCommSetMemoryRangeSupported);
    SET_PTR(hcclCommUnsetMemoryRangePtr, "HcclCommUnsetMemoryRange", StubHcclCommUnsetMemoryRange, g_hcclCommUnsetMemoryRangeSupported);
    SET_PTR(hcclCommActivateCommMemoryPtr, "HcclCommActivateCommMemory", StubHcclCommActivateCommMemory, g_hcclCommActivateCommMemorySupported);
    SET_PTR(hcclCommDeactivateCommMemoryPtr, "HcclCommDeactivateCommMemory", StubHcclCommDeactivateCommMemory, g_hcclCommDeactivateCommMemorySupported);
    SET_PTR(hcclCommWorkingDevNicSetPtr, "HcclCommWorkingDevNicSet", StubHcclCommWorkingDevNicSet, g_hcclCommWorkingDevNicSetSupported);
    SET_PTR(hcclGroupStartPtr, "HcclGroupStart", StubHcclGroupStart, g_hcclGroupStartSupported);
    SET_PTR(hcclGroupEndPtr, "HcclGroupEnd", StubHcclGroupEnd, g_hcclGroupEndSupported);
    SET_PTR(hcclCommSymWinRegisterPtr, "HcclCommSymWinRegister", StubHcclCommSymWinRegister, g_hcclCommSymWinRegisterSupported);
    SET_PTR(hcclCommSymWinDeregisterPtr, "HcclCommSymWinDeregister", StubHcclCommSymWinDeregister, g_hcclCommSymWinDeregisterSupported);
    SET_PTR(hcclCommSymWinGetPtr, "HcclCommSymWinGet", StubHcclCommSymWinGet, g_hcclCommSymWinGetSupported);
    SET_PTR(hcclGetRawCommHandlePtr, "HcclGetRawCommHandle", StubHcclGetRawCommHandle, g_hcclGetRawCommHandleSupported);
    SET_PTR(hcclGetCcuTaskInfoPtr, "HcclGetCcuTaskInfo", StubHcclGetCcuTaskInfo, g_hcclGetCcuTaskInfoSupported);
    SET_PTR(commGetLocalCCLBufPtr, "CommGetLocalCCLBuf", StubCommGetLocalCCLBuf, g_commGetLocalCCLBufSupported);
    SET_PTR(commGetRemoteCCLBufPtr, "CommGetRemoteCCLBuf", StubCommGetRemoteCCLBuf, g_commGetRemoteCCLBufSupported);
    SET_PTR(commGetKFCWorkSpacePtr, "CommGetKFCWorkSpace", StubCommGetKFCWorkSpace, g_commGetKFCWorkSpaceSupported);
    SET_PTR(commGetCCLBufSizeCfgPtr, "CommGetCCLBufSizeCfg", StubCommGetCCLBufSizeCfg, g_commGetCCLBufSizeCfgSupported);
    SET_PTR(hcclCommInitClusterInfoMemConfigPtr, "HcclCommInitClusterInfoMemConfig", StubHcclCommInitClusterInfoMemConfig, g_hcclCommInitClusterInfoMemConfigSupported);
    SET_PTR(hcclSnapshotSavePtr, "HcclSnapshotSave", StubHcclSnapshotSave, g_hcclSnapshotSaveSupported);
    SET_PTR(hcclSnapshotGetBufSizePtr, "HcclSnapshotGetBufSize", StubHcclSnapshotGetBufSize, g_hcclSnapshotGetBufSizeSupported);
    SET_PTR(hcclSnapshotRecoverAllCommsPtr, "HcclSnapshotRecoverAllComms",
            StubHcclSnapshotRecoverAllComms, g_hcclSnapshotRecoverAllCommsSupported);

    #undef SET_PTR
}

void HcclCommDlFini(void) {
    hcclGetRankIdPtr = StubHcclGetRankId;
    hcclGetRankSizePtr = StubHcclGetRankSize;
    hcclCommInitClusterInfoPtr = StubHcclCommInitClusterInfo;
    g_hcclCommInitClusterInfoSupported = false;
    hcclCommInitClusterInfoConfigPtr = StubHcclCommInitClusterInfoConfig;
    g_hcclCommInitClusterInfoConfigSupported = false;
    hcclCreateSubCommConfigPtr = StubHcclCreateSubCommConfig;
    g_hcclCreateSubCommConfigSupported = false;
    hcclGetRootInfoPtr = StubHcclGetRootInfo;
    g_hcclGetRootInfoSupported = false;
    hcclCommInitRootInfoPtr = StubHcclCommInitRootInfo;
    g_hcclCommInitRootInfoSupported = false;
    hcclCommInitRootInfoConfigPtr = StubHcclCommInitRootInfoConfig;
    g_hcclCommInitRootInfoConfigSupported = false;
    hcclSetConfigPtr = StubHcclSetConfig;
    g_hcclSetConfigSupported = false;
    hcclGetConfigPtr = StubHcclGetConfig;
    g_hcclGetConfigSupported = false;
    hcclGetCommNamePtr = StubHcclGetCommName;
    g_hcclGetCommNameSupported = false;
    hcclCommGetHandleWithNamePtr = StubHcclCommGetHandleWithName;
    g_hcclCommGetHandleWithNameSupported = false;
    hcclBarrierPtr = StubHcclBarrier;
    g_hcclBarrierSupported = false;
    hcclCommDestroyPtr = StubHcclCommDestroy;
    g_hcclCommDestroySupported = false;
    hcclCommInitAllPtr = StubHcclCommInitAll;
    g_hcclCommInitAllSupported = false;
    hcclGetCommAsyncErrorPtr = StubHcclGetCommAsyncError;
    g_hcclGetCommAsyncErrorSupported = false;
    hcclGetErrorStringPtr = StubHcclGetErrorString;
    g_hcclGetErrorStringSupported = false;
    hcclGetCommConfigCapabilityPtr = StubHcclGetCommConfigCapability;
    g_hcclGetCommConfigCapabilitySupported = false;
    hcclCommSuspendPtr = StubHcclCommSuspend;
    g_hcclCommSuspendSupported = false;
    hcclCommResumePtr = StubHcclCommResume;
    g_hcclCommResumeSupported = false;
    hcclCommSetMemoryRangePtr = StubHcclCommSetMemoryRange;
    g_hcclCommSetMemoryRangeSupported = false;
    hcclCommUnsetMemoryRangePtr = StubHcclCommUnsetMemoryRange;
    g_hcclCommUnsetMemoryRangeSupported = false;
    hcclCommActivateCommMemoryPtr = StubHcclCommActivateCommMemory;
    g_hcclCommActivateCommMemorySupported = false;
    hcclCommDeactivateCommMemoryPtr = StubHcclCommDeactivateCommMemory;
    g_hcclCommDeactivateCommMemorySupported = false;
    hcclCommWorkingDevNicSetPtr = StubHcclCommWorkingDevNicSet;
    g_hcclCommWorkingDevNicSetSupported = false;
    hcclGroupStartPtr = StubHcclGroupStart;
    g_hcclGroupStartSupported = false;
    hcclGroupEndPtr = StubHcclGroupEnd;
    g_hcclGroupEndSupported = false;
    hcclCommSymWinRegisterPtr = StubHcclCommSymWinRegister;
    g_hcclCommSymWinRegisterSupported = false;
    hcclCommSymWinDeregisterPtr = StubHcclCommSymWinDeregister;
    g_hcclCommSymWinDeregisterSupported = false;
    hcclCommSymWinGetPtr = StubHcclCommSymWinGet;
    g_hcclCommSymWinGetSupported = false;
    hcclGetRawCommHandlePtr = StubHcclGetRawCommHandle;
    g_hcclGetRawCommHandleSupported = false;
    hcclGetCcuTaskInfoPtr = StubHcclGetCcuTaskInfo;
    g_hcclGetCcuTaskInfoSupported = false;
    commGetLocalCCLBufPtr = StubCommGetLocalCCLBuf;
    g_commGetLocalCCLBufSupported = false;
    commGetRemoteCCLBufPtr = StubCommGetRemoteCCLBuf;
    g_commGetRemoteCCLBufSupported = false;
    commGetKFCWorkSpacePtr = StubCommGetKFCWorkSpace;
    g_commGetKFCWorkSpaceSupported = false;
    commGetCCLBufSizeCfgPtr = StubCommGetCCLBufSizeCfg;
    g_commGetCCLBufSizeCfgSupported = false;
    hcclCommInitClusterInfoMemConfigPtr = StubHcclCommInitClusterInfoMemConfig;
    g_hcclCommInitClusterInfoMemConfigSupported = false;
    hcclSnapshotSavePtr = StubHcclSnapshotSave;
    g_hcclSnapshotSaveSupported = false;
    hcclSnapshotGetBufSizePtr = StubHcclSnapshotGetBufSize;
    g_hcclSnapshotGetBufSizeSupported = false;
    hcclSnapshotRecoverAllCommsPtr = StubHcclSnapshotRecoverAllComms;
    g_hcclSnapshotRecoverAllCommsSupported = false;
}

// ---------- 对外API实现（通过函数指针转发）----------
HcclResult HcclGetRankId(HcclComm comm, uint32_t* rank) {
    return hcclGetRankIdPtr(comm, rank);
}
HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize) {
    return hcclGetRankSizePtr(comm, rankSize);
}
HcclResult HcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm) {
    return hcclCommInitClusterInfoPtr(clusterInfo, rank, comm);
}
HcclResult HcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm) {
    return hcclCommInitClusterInfoConfigPtr(clusterInfo, rank, config, comm);
}
HcclResult HcclCreateSubCommConfig(HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId, HcclCommConfig* config, HcclComm* subComm) {
    return hcclCreateSubCommConfigPtr(comm, rankNum, rankIds, subCommId, subCommRankId, config, subComm);
}
HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo) {
    return hcclGetRootInfoPtr(rootInfo);
}
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm) {
    return hcclCommInitRootInfoPtr(nRanks, rootInfo, rank, comm);
}
HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm) {
    return hcclCommInitRootInfoConfigPtr(nRanks, rootInfo, rank, config, comm);
}
HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue) {
    return hcclSetConfigPtr(config, configValue);
}
HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue* configValue) {
    return hcclGetConfigPtr(config, configValue);
}
HcclResult HcclGetCommName(HcclComm comm, char* commName) {
    return hcclGetCommNamePtr(comm, commName);
}
HcclResult HcclCommGetHandleWithName(const char* commName, HcclComm* comm) {
    return hcclCommGetHandleWithNamePtr(commName, comm);
}
HcclResult HcclBarrier(HcclComm comm, aclrtStream stream) {
    return hcclBarrierPtr(comm, stream);
}
HcclResult HcclCommDestroy(HcclComm comm) {
    return hcclCommDestroyPtr(comm);
}
HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms) {
    return hcclCommInitAllPtr(ndev, devices, comms);
}
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError) {
    return hcclGetCommAsyncErrorPtr(comm, asyncError);
}
const char* HcclGetErrorString(HcclResult code) {
    return hcclGetErrorStringPtr(code);
}
uint32_t HcclGetCommConfigCapability(void) {
    return hcclGetCommConfigCapabilityPtr();
}
HcclResult HcclCommSuspend(HcclComm comm) {
    return hcclCommSuspendPtr(comm);
}
HcclResult HcclCommResume(HcclComm comm) {
    return hcclCommResumePtr(comm);
}
HcclResult HcclCommSetMemoryRange(HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags) {
    return hcclCommSetMemoryRangePtr(comm, baseVirPtr, size, alignment, flags);
}
HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void* baseVirPtr) {
    return hcclCommUnsetMemoryRangePtr(comm, baseVirPtr);
}
HcclResult HcclCommActivateCommMemory(HcclComm comm, void* virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags) {
    return hcclCommActivateCommMemoryPtr(comm, virPtr, size, offset, handle, flags);
}
HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void* virPtr) {
    return hcclCommDeactivateCommMemoryPtr(comm, virPtr);
}
HcclResult HcclCommWorkingDevNicSet(HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks) {
    return hcclCommWorkingDevNicSetPtr(comm, ranks, useBackup, nRanks);
}
HcclResult HcclGroupStart(void) {
    return hcclGroupStartPtr();
}
HcclResult HcclGroupEnd(void) {
    return hcclGroupEndPtr();
}
HcclResult HcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, CommSymWindow* winHandle, uint32_t flag) {
    return hcclCommSymWinRegisterPtr(comm, addr, size, winHandle, flag);
}
HcclResult HcclCommSymWinDeregister(CommSymWindow winHandle) {
    return hcclCommSymWinDeregisterPtr(winHandle);
}
HcclResult HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, CommSymWindow* winHandle, size_t* offset) {
    return hcclCommSymWinGetPtr(comm, ptr, size, winHandle, offset);
}
HcclResult HcclGetRawCommHandle(const char* commName, HcclComm* commHandle) {
    return hcclGetRawCommHandlePtr(commName, commHandle);
}
HcclResult HcclGetCcuTaskInfo(HcclComm comm, void* tilingData, void* ccuTaskGroup) {
    return hcclGetCcuTaskInfoPtr(comm, tilingData, ccuTaskGroup);
}
HcclResult CommGetLocalCCLBuf(HcclComm comm, void **addr, uint64_t *size) {
    return commGetLocalCCLBufPtr(comm, addr, size);
}
HcclResult CommGetRemoteCCLBuf(HcclComm comm, uint32_t remoteRank, void **addr, uint64_t *size) {
    return commGetRemoteCCLBufPtr(comm, remoteRank, addr, size);
}
HcclResult CommGetKFCWorkSpace(HcclComm comm, void **addr, uint64_t *size) {
    return commGetKFCWorkSpacePtr(comm, addr, size);
}
HcclResult CommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufSize) {
    return commGetCCLBufSizeCfgPtr(comm, cclBufSize);
}
HcclResult HcclCommInitClusterInfoMemConfig(const char *rankTableString, uint32_t rank,
                                            HcclCommConfig *config, HcclComm *comm)
{
    return hcclCommInitClusterInfoMemConfigPtr(rankTableString, rank, config, comm);
}
HcclResult HcclSnapshotSave(void* snapshotBuf, uint32_t size, uint32_t step) {
    return hcclSnapshotSavePtr(snapshotBuf, size, step);
}
HcclResult HcclSnapshotGetBufSize(uint32_t step, uint32_t* size) {
    return hcclSnapshotGetBufSizePtr(step, size);
}
HcclResult HcclSnapshotRecoverAllComms(const char* clusterInfo, const char* changedInfo,
                                       void* snapshotBuf, uint32_t snapshotBufSize) {
    return hcclSnapshotRecoverAllCommsPtr(clusterInfo, changedInfo, snapshotBuf, snapshotBufSize);
}

// ---------- 查询函数实现 ----------
extern "C" bool HcommIsSupportHcclGetRankId(void) {
    return g_hcclGetRankIdSupported;
}
extern "C" bool HcommIsSupportHcclGetRankSize(void) {
    return g_hcclGetRankSizeSupported;
}
extern "C" bool HcommIsSupportHcclCommInitClusterInfo(void) { return g_hcclCommInitClusterInfoSupported; }
extern "C" bool HcommIsSupportHcclCommInitClusterInfoConfig(void) { return g_hcclCommInitClusterInfoConfigSupported; }
extern "C" bool HcommIsSupportHcclCreateSubCommConfig(void) { return g_hcclCreateSubCommConfigSupported; }
extern "C" bool HcommIsSupportHcclGetRootInfo(void) { return g_hcclGetRootInfoSupported; }
extern "C" bool HcommIsSupportHcclCommInitRootInfo(void) { return g_hcclCommInitRootInfoSupported; }
extern "C" bool HcommIsSupportHcclCommInitRootInfoConfig(void) { return g_hcclCommInitRootInfoConfigSupported; }
extern "C" bool HcommIsSupportHcclSetConfig(void) { return g_hcclSetConfigSupported; }
extern "C" bool HcommIsSupportHcclGetConfig(void) { return g_hcclGetConfigSupported; }
extern "C" bool HcommIsSupportHcclGetCommName(void) { return g_hcclGetCommNameSupported; }
extern "C" bool HcommIsSupportHcclCommGetHandleWithName(void) { return g_hcclCommGetHandleWithNameSupported; }
extern "C" bool HcommIsSupportHcclBarrier(void) { return g_hcclBarrierSupported; }
extern "C" bool HcommIsSupportHcclCommDestroy(void) { return g_hcclCommDestroySupported; }
extern "C" bool HcommIsSupportHcclCommInitAll(void) { return g_hcclCommInitAllSupported; }
extern "C" bool HcommIsSupportHcclGetCommAsyncError(void) { return g_hcclGetCommAsyncErrorSupported; }
extern "C" bool HcommIsSupportHcclGetErrorString(void) { return g_hcclGetErrorStringSupported; }
extern "C" bool HcommIsSupportHcclGetCommConfigCapability(void) { return g_hcclGetCommConfigCapabilitySupported; }
extern "C" bool HcommIsSupportHcclCommSuspend(void) { return g_hcclCommSuspendSupported; }
extern "C" bool HcommIsSupportHcclCommResume(void) { return g_hcclCommResumeSupported; }
extern "C" bool HcommIsSupportHcclCommSetMemoryRange(void) { return g_hcclCommSetMemoryRangeSupported; }
extern "C" bool HcommIsSupportHcclCommUnsetMemoryRange(void) { return g_hcclCommUnsetMemoryRangeSupported; }
extern "C" bool HcommIsSupportHcclCommActivateCommMemory(void) { return g_hcclCommActivateCommMemorySupported; }
extern "C" bool HcommIsSupportHcclCommDeactivateCommMemory(void) { return g_hcclCommDeactivateCommMemorySupported; }
extern "C" bool HcommIsSupportHcclCommWorkingDevNicSet(void) { return g_hcclCommWorkingDevNicSetSupported; }
extern "C" bool HcommIsSupportHcclGroupStart(void) { return g_hcclGroupStartSupported; }
extern "C" bool HcommIsSupportHcclGroupEnd(void) { return g_hcclGroupEndSupported; }
extern "C" bool HcommIsSupportHcclCommSymWinRegister(void) { return g_hcclCommSymWinRegisterSupported; }
extern "C" bool HcommIsSupportHcclCommSymWinDeregister(void) { return g_hcclCommSymWinDeregisterSupported; }
extern "C" bool HcommIsSupportHcclCommSymWinGet(void) { return g_hcclCommSymWinGetSupported; }
extern "C" bool HcommIsSupportHcclGetRawCommHandle(void) { return g_hcclGetRawCommHandleSupported; }
extern "C" bool HcommIsSupportHcclGetCcuTaskInfo(void) { return g_hcclGetCcuTaskInfoSupported; }
extern "C" bool HcommIsSupportCommGetLocalCCLBuf(void) { return g_commGetLocalCCLBufSupported; }
extern "C" bool HcommIsSupportCommGetRemoteCCLBuf(void) { return g_commGetRemoteCCLBufSupported; }
extern "C" bool HcommIsSupportCommGetKFCWorkSpace(void) { return g_commGetKFCWorkSpaceSupported; }
extern "C" bool HcommIsSupportCommGetCCLBufSizeCfg(void) { return g_commGetCCLBufSizeCfgSupported; }
extern "C" bool HcommIsSupportHcclCommInitClusterInfoMemConfig(void) { return g_hcclCommInitClusterInfoMemConfigSupported; }
extern "C" bool HcommIsSupportHcclSnapshotSave(void) {
    return g_hcclSnapshotSaveSupported;
}
extern "C" bool HcommIsSupportHcclSnapshotGetBufSize(void) {
    return g_hcclSnapshotGetBufSizeSupported;
}
extern "C" bool HcommIsSupportHcclSnapshotRecoverAllComms(void) {
    return g_hcclSnapshotRecoverAllCommsSupported;
}