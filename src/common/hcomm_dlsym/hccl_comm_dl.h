/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_DL_H
#define HCCL_COMM_DL_H

#include "hccl/hccl_comm.h"   // 原头文件，包含所有类型和 inline 函数
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED (static_cast<HcclResult>(-2))
#endif

// 对外 API 的函数声明（包装函数）
HcclResult HcclGetRankId(HcclComm comm, uint32_t* rank);
HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize);
HcclResult HcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm);
HcclResult HcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm);
HcclResult HcclCreateSubCommConfig(HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId, HcclCommConfig* config, HcclComm* subComm);
HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo);
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm);
HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm);
HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue);
HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue* configValue);
HcclResult HcclGetCommName(HcclComm comm, char* commName);
HcclResult HcclCommGetHandleWithName(const char* commName, HcclComm* comm);
HcclResult HcclBarrier(HcclComm comm, aclrtStream stream);
HcclResult HcclCommDestroy(HcclComm comm);
HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms);
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError);
const char* HcclGetErrorString(HcclResult code);
uint32_t HcclGetCommConfigCapability(void);
HcclResult HcclCommSuspend(HcclComm comm);
HcclResult HcclCommResume(HcclComm comm);
HcclResult HcclCommSetMemoryRange(HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags);
HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void* baseVirPtr);
HcclResult HcclCommActivateCommMemory(HcclComm comm, void* virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags);
HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void* virPtr);
HcclResult HcclCommWorkingDevNicSet(HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks);
HcclResult HcclGroupStart(void);
HcclResult HcclGroupEnd(void);
HcclResult HcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, CommSymWindow* winHandle, uint32_t flag);
HcclResult HcclCommSymWinDeregister(CommSymWindow winHandle);
HcclResult HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, CommSymWindow* winHandle, size_t* offset);
HcclResult HcclGetRawCommHandle(const char* commName, HcclComm* commHandle);
HcclResult HcclGetCcuTaskInfo(HcclComm comm, void* tilingData, void* ccuTaskGroup);
HcclResult CommGetLocalCCLBuf(HcclComm comm, void **addr, uint64_t *size);
HcclResult CommGetRemoteCCLBuf(HcclComm comm, uint32_t remoteRank, void **addr, uint64_t *size);
HcclResult CommGetKFCWorkSpace(HcclComm comm, void **addr, uint64_t *size);
HcclResult CommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufSize);
HcclResult HcclCommInitClusterInfoMemConfig(const char *rankTableString, uint32_t rank,
                                            HcclCommConfig *config, HcclComm *comm);
HcclResult HcclSnapshotSave(void* snapshotBuf, uint32_t size, uint32_t step);
HcclResult HcclSnapshotGetBufSize(uint32_t step, uint32_t* size);
HcclResult HcclSnapshotRecoverAllComms(const char* clusterInfo, const char* changedInfo,
                                       void* snapshotBuf, uint32_t snapshotBufSize);

// 查询函数声明
bool HcommIsSupportHcclCommInitClusterInfo(void);
bool HcommIsSupportHcclCommInitClusterInfoConfig(void);
bool HcommIsSupportHcclCreateSubCommConfig(void);
bool HcommIsSupportHcclGetRootInfo(void);
bool HcommIsSupportHcclCommInitRootInfo(void);
bool HcommIsSupportHcclCommInitRootInfoConfig(void);
bool HcommIsSupportHcclSetConfig(void);
bool HcommIsSupportHcclGetConfig(void);
bool HcommIsSupportHcclGetCommName(void);
bool HcommIsSupportHcclCommGetHandleWithName(void);
bool HcommIsSupportHcclBarrier(void);
bool HcommIsSupportHcclCommDestroy(void);
bool HcommIsSupportHcclCommInitAll(void);
bool HcommIsSupportHcclGetCommAsyncError(void);
bool HcommIsSupportHcclGetErrorString(void);
bool HcommIsSupportHcclGetCommConfigCapability(void);
bool HcommIsSupportHcclCommSuspend(void);
bool HcommIsSupportHcclCommResume(void);
bool HcommIsSupportHcclCommSetMemoryRange(void);
bool HcommIsSupportHcclCommUnsetMemoryRange(void);
bool HcommIsSupportHcclCommActivateCommMemory(void);
bool HcommIsSupportHcclCommDeactivateCommMemory(void);
bool HcommIsSupportHcclCommWorkingDevNicSet(void);
bool HcommIsSupportHcclGroupStart(void);
bool HcommIsSupportHcclGroupEnd(void);
bool HcommIsSupportHcclCommSymWinRegister(void);
bool HcommIsSupportHcclCommSymWinDeregister(void);
bool HcommIsSupportHcclCommSymWinGet(void);
bool HcommIsSupportHcclGetRawCommHandle(void);
bool HcommIsSupportHcclGetCcuTaskInfo(void);
bool HcommIsSupportCommGetLocalCCLBuf(void);
bool HcommIsSupportCommGetRemoteCCLBuf(void);
bool HcommIsSupportCommGetKFCWorkSpace(void);
bool HcommIsSupportCommGetCCLBufSizeCfg(void);
bool HcommIsSupportHcclCommInitClusterInfoMemConfig(void);
bool HcommIsSupportHcclSnapshotSave(void);
bool HcommIsSupportHcclSnapshotGetBufSize(void);
bool HcommIsSupportHcclSnapshotRecoverAllComms(void);

void HcclCommDlInit(void* libHcommHandle);        // 本模块独立初始化
void HcclCommDlFini(void);                         // 本模块独立销毁

#ifdef __cplusplus
}
#endif

#endif // HCCL_COMM_DL_H