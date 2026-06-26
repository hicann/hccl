/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
  * CANN Open Software License Agreement Version 2.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the root of the software repository for the full text of the License.
  */

#ifndef HCCL_HOST_COMM_DL_H
#define HCCL_HOST_COMM_DL_H

#include "dlsym_common.h"
#include "hccl_comm.h"   // 原始头文件，包含所有类型和声明

/* beta.1 起 hccl_comm.h 已提供 HcclOpExpansionMode/HcclConfigType，仅 < 9.1.0_beta.1 (8.5.0/9.0.0) 需要桩 */
#if CANN_VERSION_NUM < CANN_VERSION(9, 0, 0)
typedef enum {
    HCCL_OP_EXPANSION_MODE_INVALID = -1,
    HCCL_OP_EXPANSION_MODE_AI_CPU = 0,
    HCCL_OP_EXPANSION_MODE_AIV = 1,
    HCCL_OP_EXPANSION_MODE_HOST = 2,
    HCCL_OP_EXPANSION_MODE_HOST_TS = 3,
    HCCL_OP_EXPANSION_CCU_MS = 4,
    HCCL_OP_EXPANSION_CCU_SCHED = 5,
    HCCL_OP_EXPANSION_AIV_ONLY = 6
} HcclOpExpansionMode;

typedef enum {
    HCCL_CONFIG_TYPE_INVALID = -1,
    HCCL_CONFIG_TYPE_OP_EXPANSION_MODE = 0
} HcclConfigType;

typedef HcclOpExpansionMode HcclConfigTypeOpExpansionMode;

#endif /* CANN_VERSION_NUM < CANN_VERSION(9, 1, 0, 1) */

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(HcclResult, HcclCommGetStatus, const char* commId, HcclCommStatus *status);
DECL_SUPPORT_FLAG(HcclCommGetStatus);

DECL_WEAK_FUNC(HcclResult, HcclConfigGetInfo, HcclComm comm, HcclConfigType cfgType,
    uint32_t infoLen, void *info);
DECL_SUPPORT_FLAG(HcclConfigGetInfo);

DECL_WEAK_FUNC(HcclResult, HcclGroupStatusGet, bool *isGroupEnabled);
DECL_SUPPORT_FLAG(HcclGroupStatusGet);

DECL_WEAK_FUNC(HcclResult, HcclAicpuKernelLaunch, HcclComm comm, const HcclOpDesc *opInfo,
    const HcclKernelFuncInfo *funcInfo, ThreadHandle aicpuThreadHandle, aclrtStream userStream,
    const HcclKernelLaunchCfg *kernelLaunchCfg);
DECL_SUPPORT_FLAG(HcclAicpuKernelLaunch);

void HcclCommDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCCL_HOST_COMM_DL_H
