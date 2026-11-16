/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_STUCK_DETECT_HCCL_HOST_COMM_DL_H
#define UT_STUCK_DETECT_HCCL_HOST_COMM_DL_H

#include "hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HCCL_COMM_STATE_PHASE_INVALID = -1,
    HCCL_COMM_STATE_PHASE_DESTROY_PRE = 0,
    HCCL_COMM_STATE_PHASE_DESTROY_POST = 1,
    HCCL_COMM_STATE_PHASE_RESUME_PRE = 2,
    HCCL_COMM_STATE_PHASE_RESUME_POST = 3
} HcclCommStatePhase;

typedef HcclResult (*HcclCommStateCallback)(HcclComm comm, HcclCommStatePhase state, void* args);

bool HcommIsSupportHcclCommRegCommStateCallback(void);
HcclResult HcclCommRegCommStateCallback(const char* regName, HcclCommStateCallback cb, void* args);

// 通信域身份查询（真实实现由 libhcomm 提供弱符号）
HcclResult HcclGetCommName(HcclComm comm, char* commName);
HcclResult HcclGetRankId(HcclComm comm, uint32_t* rank);

#ifdef __cplusplus
}
#endif

#endif // UT_STUCK_DETECT_HCCL_HOST_COMM_DL_H
