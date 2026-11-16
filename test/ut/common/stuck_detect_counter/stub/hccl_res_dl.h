/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_STUCK_DETECT_HCCL_RES_DL_H
#define UT_STUCK_DETECT_HCCL_RES_DL_H

#include "hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

bool HcommIsSupportHcclStuckDetectGetCounterAddr(void);
HcclResult HcclStuckDetectGetCounterAddr(
    HcclComm comm, uint64_t* headOpCounterAddr, uint64_t* tailOpCounterAddr, uint64_t* addOneMemAddr,
    uint64_t* counterToken);

#ifdef __cplusplus
}
#endif

#endif // UT_STUCK_DETECT_HCCL_RES_DL_H
