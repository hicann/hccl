/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DIAG_DL_H
#define HCOMM_DIAG_DL_H

#include "dlsym_common.h"
#include "hcomm_diag.h"   // 原始头文件，包含所有声明和类型定义
#include "hccl_res.h"     // CommAbiHeader, CommEngine for HcclDfxOpInfo stub

#if CANN_VERSION_NUM >= 90000000
#include "hccl_diag.h"    // 9.0.0 提供 HcclDfxOpInfo, HCOMM_ALG_TAG_LENGTH
#endif

/* 8.5.0 桩: hccl_diag.h 中 9.0.0 新增类型 */
#if CANN_VERSION_NUM < 90000000
#define HCOMM_ALG_TAG_LENGTH 288

struct HcclDfxOpInfo {
    CommAbiHeader header;
    uint64_t beginTime;
    uint64_t endTime;
    uint32_t opMode;
    uint32_t opType;
    uint32_t reduceOp;
    uint32_t dataType;
    uint32_t outputType;
    uint64_t dataCount;
    uint32_t root;
    char algTag[HCOMM_ALG_TAG_LENGTH];
    CommEngine engine;
    uint64_t cpuTsThread;
    uint32_t cpuWaitAicpuNotifyIdx;
    uint32_t cpuWaitAicpuNotifyId;
    int8_t reserve[128];
};
#endif /* CANN_VERSION_NUM < 90000000 */

#ifdef __cplusplus
extern "C" {
#endif

DECL_SUPPORT_FLAG(HcommRegOpInfo);
DECL_SUPPORT_FLAG(HcommRegOpTaskException);

// 动态库管理接口
void HcommDiagDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DIAG_DL_H