/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_primitives.h"
#include "alg_data_trans_wrapper.h"
#include "hccl_sym_win_dl.h"

extern "C" int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}

namespace ops_hccl {
// recursive_executor data_ops.cc 复用 src 的 LocalCopy；UT 独立编译时以桩提供定义。
HcclResult LocalCopy(const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice)
{
    (void)thread;
    (void)srcSlice;
    (void)dstSlice;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

// template_utils.cc 调用 GetSymWinRemoteMem；UT 独立编译时以桩提供定义。
HcclResult GetSymWinRemoteMem(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    (void)winHandle;
    (void)offset;
    (void)peerRank;
    if (ptr != nullptr) {
        *ptr = nullptr;
    }
    return HCCL_SUCCESS;
}
