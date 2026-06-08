/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RES_EXPT_DL_H
#define HCCL_RES_EXPT_DL_H

#include "dlsym_common.h"
/* hccl_res_expt.h 仅 9.0.0+ 提供，8.5.0 缺失；弱符号声明仅依赖 hccl_types.h 中的基础类型 */
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "hccl_res_expt.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(HcclResult, HcclCommAddExchangeInfo, HcclComm comm, const void *data, uint32_t length);
DECL_WEAK_FUNC(HcclResult, HcclCommGetExchangeInfo, HcclComm comm, uint32_t remoteRank, uint32_t length, void *data,
    uint32_t *actualLength);
DECL_SUPPORT_FLAG(HcclCommAddExchangeInfo);
DECL_SUPPORT_FLAG(HcclCommGetExchangeInfo);

void HcclResExptDlInit(void *libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCCL_RES_EXPT_DL_H