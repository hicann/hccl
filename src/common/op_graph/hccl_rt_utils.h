/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RT_UTILS_H
#define HCCL_RT_UTILS_H

#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"

namespace hccl {

// 本文件对应 InfoStore hcom_acl_adapter.h 中 hrtMemSyncCopy / hrtMemAsyncCopy 的封装。
// ADDR 模式用 aclrtMemcpyAsyncWithOffset 实现二级指针地址拷贝。

enum class HcclRtMemcpyKind {
    HCCL_RT_MEMCPY_KIND_HOST_TO_HOST = 0,
    HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE,
    HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST,
    HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE,
    HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE, // 二级指针地址拷贝
    HCCL_RT_MEMCPY_KIND_RESERVED,
};

// 同步内存拷贝: 按 kind 映射到 aclrtMemcpy 对应类型 (对应 InfoStore hrtMemSyncCopy)
HcclResult HcclRtMemSyncCopy(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind);

// 异步内存拷贝: 按 kind 映射到 aclrtMemcpyAsync / 带偏移二级指针变体 (对应 InfoStore hrtMemAsyncCopy)
HcclResult HcclRtMemAsyncCopy(
    void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind, aclrtStream stream);

// 对应 InfoStore hrtMemcpyAddrAsync: 带偏移的二级指针地址异步拷贝 (D2D ADDR)
// 参数序与 InfoStore 一致: (dst, dstMax, dstOffset, src, count, srcOffset, stream)
// 与 HcclRtMemAsyncCopy 的 ADDR 分支 (offset 双双传 0) 互不影响, 本函数专供多迭代 loop 的逐迭代偏移搬运
// dstMax 分侧传: ② 输入侧 = CCL buf 尺寸 commInputSize; ⑤ 输出侧 = 输出全量 count*unitSize
HcclResult HcclRtMemAddrAsyncCopyWithOffset(
    void* dst, uint64_t destMax, uint64_t dstOffset, const void* src, uint64_t count, uint64_t srcOffset,
    aclrtStream stream);

} // namespace hccl

#endif // HCCL_RT_UTILS_H
