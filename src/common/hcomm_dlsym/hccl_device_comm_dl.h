/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DEVICE_COMM_DL_H
#define HCCL_DEVICE_COMM_DL_H

#include "dlsym_common.h"
#include "hccl_comm.h"   // 原始头文件，包含所有类型和声明

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(HcclResult, HcclCommGetStatus, const char* commId, HcclCommStatus *status);
DECL_SUPPORT_FLAG(HcclCommGetStatus);

DECL_WEAK_FUNC(HcclResult, HcclGroupStatusGet, bool *isGroupEnabled);
DECL_SUPPORT_FLAG(HcclGroupStatusGet);

DECL_WEAK_FUNC(HcclResult, HcclAicpuKernelLaunch, HcclComm comm, const HcclOpDesc *opInfo,
    const HcclKernelFuncInfo *funcInfo, ThreadHandle aicpuThreadHandle, aclrtStream userStream,
    const HcclKernelLaunchCfg *kernelLaunchCfg);
DECL_SUPPORT_FLAG(HcclAicpuKernelLaunch);

DECL_WEAK_FUNC(HcclResult, HcclCommRegCommStateCallback, const char *regName, HcclCommStateCallback cb, void *args);
DECL_SUPPORT_FLAG(HcclCommRegCommStateCallback);

void HcclDeviceCommDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCCL_DEVICE_COMM_DL_H
