/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_FALLBACK_C_H
#define CCU_FALLBACK_C_H

#include <stdbool.h>
#include <stdint.h>
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult CheckCcuResNegotiationC(HcclComm comm, const void* param, bool localResAvailable);

HcclResult
CheckCcuParamAndFallbackC(HcclComm comm, void* param, void** topoInfo, char* algNameBuf, uint32_t algNameBufLen);

#ifdef __cplusplus
}
#endif

#endif // CCU_FALLBACK_C_H
