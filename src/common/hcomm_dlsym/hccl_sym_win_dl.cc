/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_sym_win_dl.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

DEFINE_WEAK_FUNC(
    HcclResult, HcclSymWinGetPeerPointer, HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);
DEFINE_WEAK_FUNC(
    HcclResult, HcclSymWinGetRemoteAddr, HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);

// 初始化
void HcclSymWinDlInit(void* libHcommHandle)
{
    INIT_SUPPORT_FLAG(libHcommHandle, HcclSymWinGetPeerPointer);
    INIT_SUPPORT_FLAG(libHcommHandle, HcclSymWinGetRemoteAddr);
}

HcclResult GetSymWinRemoteMem(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    if (HcommIsSupportHcclSymWinGetRemoteAddr()) {
        return static_cast<HcclResult>(HcclSymWinGetRemoteAddr(winHandle, offset, peerRank, ptr));
    }
    return static_cast<HcclResult>(HcclSymWinGetPeerPointer(winHandle, offset, peerRank, ptr));
}
