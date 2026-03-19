/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_MC2_EX_DL_H
#define HCCL_MC2_EX_DL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 对外 API 包装函数声明
HcclResult HcclGetCommHandleByCtx(void *ctx, void **opHandle);
HcclResult HcclReleaseComm(void* opHandle);
HcclResult HcclGetTaskStatus(void* opHandle, void *status);
HcclResult HcclCheckFinishByStream(void* opHandle);
HcclResult HcclPrintTaskExceptionAllComm(void* opHandle);
HcclResult HcclLaunchCcoreWait(void* opHandle, uint64_t waitAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast);
HcclResult HcclLaunchCcorePost(void* opHandle, uint64_t recordAddr, uint32_t turnNum, uint64_t turnNumAddr);
HcclResult HcclLaunchOp(void* opHandle, void* data);

HcclResult HcclGetOpArgs(void **opArgs);
HcclResult HcclFreeOpArgs(void *opArgs);
HcclResult HcclSetOpSrcDataType(void *opArgs, uint8_t srcDataType);
HcclResult HcclSetOpDstDataType(void *opArgs, uint8_t dstDataType);
HcclResult HcclSetOpReduceType(void *opArgs, uint32_t reduceType);
HcclResult HcclSetOpCount(void *opArgs, uint64_t count);
HcclResult HcclSetOpAlgConfig(void *opArgs, char *algConfig);
HcclResult HcclSetOpCommEngine(void *opArgs, uint8_t commEngine);
HcclResult HcclCommResPrepare(HcclComm comm, char *opName, void *opArgs, void **addr);

// 查询函数声明
bool HcommIsSupportHcclGetCommHandleByCtx(void);
bool HcommIsSupportHcclReleaseComm(void);
bool HcommIsSupportHcclGetTaskStatus(void);
bool HcommIsSupportHcclCheckFinishByStream(void);
bool HcommIsSupportHcclPrintTaskExceptionAllComm(void);
bool HcommIsSupportHcclLaunchCcoreWait(void);
bool HcommIsSupportHcclLaunchCcorePost(void);
bool HcommIsSupportHcclLaunchOp(void);

bool HcommIsSupportHcclGetOpArgs(void);
bool HcommIsSupportHcclFreeOpArgs(void);
bool HcommIsSupportHcclSetOpSrcDataType(void);
bool HcommIsSupportHcclSetOpDstDataType(void);
bool HcommIsSupportHcclSetOpReduceType(void);
bool HcommIsSupportHcclSetOpCount(void);
bool HcommIsSupportHcclSetOpAlgConfig(void);
bool HcommIsSupportHcclSetOpCommEngine(void);
bool HcommIsSupportHcclCommResPrepare(void);

// 动态库管理接口
void HcclMc2ExDlInit(void* libHcommHandle);
void HcclMc2ExDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_MC2_EX_DL_H