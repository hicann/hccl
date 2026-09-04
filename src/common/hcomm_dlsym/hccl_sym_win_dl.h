/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SYM_WIN_DL_H
#define HCCL_SYM_WIN_DL_H

#include "dlsym_common.h"
#include "hccl_comm.h" // 原始头文件，包含所有类型和声明

/* 8.5.0 桩: HcclCommSymWindow (来自 hccl_types.h) */
#if CANN_VERSION_NUM < CANN_VERSION(9, 0, 0)
typedef void* HcclCommSymWindow;
#endif

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(
    HcclResult, HcclSymWinGetPeerPointer, HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);
DECL_SUPPORT_FLAG(HcclSymWinGetPeerPointer);
DECL_WEAK_FUNC(
    HcclResult, HcclSymWinGetRemoteAddr, HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);
DECL_SUPPORT_FLAG(HcclSymWinGetRemoteAddr);

void HcclSymWinDlInit(void* libHcommHandle);
HcclResult GetSymWinRemoteMem(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);

#ifdef __cplusplus
}
#endif

#endif // HCCL_SYM_WIN_DL_H
