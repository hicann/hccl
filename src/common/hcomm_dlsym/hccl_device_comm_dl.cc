/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_device_comm_dl.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

DEFINE_WEAK_FUNC(HcclResult, HcclCommGetStatus, const char* commId, HcclCommStatus *status);

DEFINE_WEAK_FUNC(HcclResult, HcclGroupStatusGet, bool *isGroupEnabled);

DEFINE_WEAK_FUNC(HcclResult, HcclAicpuKernelLaunch, HcclComm comm, const HcclOpDesc *opInfo,
    const HcclKernelFuncInfo *funcInfo, ThreadHandle aicpuThreadHandle, aclrtStream userStream,
    const HcclKernelLaunchCfg *kernelLaunchCfg);

// 初始化
void HcclDeviceCommDlInit(void* libHcommHandle) {
    INIT_SUPPORT_FLAG(libHcommHandle, HcclCommGetStatus);
    INIT_SUPPORT_FLAG(libHcommHandle, HcclGroupStatusGet);
    INIT_SUPPORT_FLAG(libHcommHandle, HcclAicpuKernelLaunch);
}