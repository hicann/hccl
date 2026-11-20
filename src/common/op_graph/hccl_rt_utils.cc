/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_rt_utils.h"
#include "log.h"

namespace hccl {

static HcclResult MemcpyKindTranslate(HcclRtMemcpyKind kind, aclrtMemcpyKind* rtKind)
{
    switch (kind) {
        case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_HOST:
            *rtKind = ACL_MEMCPY_HOST_TO_HOST;
            return HCCL_SUCCESS;
        case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE:
            *rtKind = ACL_MEMCPY_HOST_TO_DEVICE;
            return HCCL_SUCCESS;
        case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST:
            *rtKind = ACL_MEMCPY_DEVICE_TO_HOST;
            return HCCL_SUCCESS;
        case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE:
            *rtKind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            return HCCL_SUCCESS;
        // HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE 仅由异步路径支持 (HcclRtMemAsyncCopy
        // 先行拦截, 走 aclrtMemcpyAsyncWithOffset 的二级指针语义); 同步路径不做
        // 裸 D2D 降级, 落缺省分支显式报不支持 — 避免同一枚举在两条路径语义分叉
        default:
            HCCL_ERROR("[MemcpyKindTranslate] Not support the memory copy type[%d].", static_cast<int>(kind));
            return HCCL_E_PARA;
    }
}

HcclResult HcclRtMemSyncCopy(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind)
{
    // 零长度拷贝短路在数据指针判空之前: 空输出定形 (0 元素, addr=nullptr) 按零计数语义
    // 跳过访问时, 允许 (空指针, count=0) 以无操作成功返回
    if (count == 0) {
        HCCL_WARNING("[HcclRtMemSyncCopy] count is zero");
        return HCCL_SUCCESS;
    }
    if (dst == nullptr || src == nullptr) {
        HCCL_ERROR("[HcclRtMemSyncCopy] invalid param, dst=%p src=%p.", dst, src);
        return HCCL_E_PARA;
    }

    aclrtMemcpyKind rtKind;
    HcclResult ret = MemcpyKindTranslate(kind, &rtKind);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    aclError aclRet = aclrtMemcpy(dst, destMax, src, count, rtKind);
    HCCL_DEBUG(
        "Call aclrtMemcpy, return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d]",
        static_cast<int>(aclRet), dst, destMax, src, count, static_cast<int>(rtKind));
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HcclRtMemSyncCopy] aclrtMemcpy failed, "
            "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d].",
            static_cast<int>(aclRet), dst, destMax, src, count, static_cast<int>(rtKind));
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclRtMemAsyncCopy(
    void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind, aclrtStream stream)
{
    // stream 合法性与拷贝长度无关, 判空保持最前; 其余同 HcclRtMemSyncCopy:
    // 零长度短路在数据指针判空之前, 允许空输出定形场景 (空指针, count=0) 无操作成功
    if (stream == nullptr) {
        HCCL_ERROR("[HcclRtMemAsyncCopy] invalid param, stream is null.");
        return HCCL_E_PARA;
    }
    if (count == 0) {
        HCCL_WARNING("[HcclRtMemAsyncCopy] count is zero");
        return HCCL_SUCCESS;
    }
    if (dst == nullptr || src == nullptr) {
        HCCL_ERROR("[HcclRtMemAsyncCopy] invalid param, dst=%p src=%p.", dst, src);
        return HCCL_E_PARA;
    }

    if (kind == HcclRtMemcpyKind::HCCL_RT_MEMCPY_ADDR_DEVICE_TO_DEVICE) {
        aclError aclRet = aclrtMemcpyAsyncWithOffset(
            reinterpret_cast<void**>(dst), destMax, 0, reinterpret_cast<const void**>(const_cast<void*>(src)), count, 0,
            ACL_MEMCPY_INNER_DEVICE_TO_DEVICE, stream);
        HCCL_DEBUG(
            "Call HcclRtMemAsyncCopy(ADDR), return value[%d], dstAddr[%p], destMax[%llu], "
            "srcAddr[%p], count[%llu]",
            static_cast<int>(aclRet), dst, destMax, src, count);
        if (aclRet == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
            HCCL_WARNING("[HcclRtMemAsyncCopy] ADDR kind not supported, ret[%d].", static_cast<int>(aclRet));
            return HCCL_E_NOT_SUPPORT;
        }
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR(
                "[HcclRtMemAsyncCopy] rt memory async copy failed, "
                "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], stream[%p].",
                static_cast<int>(aclRet), dst, destMax, src, count, stream);
            return HCCL_E_RUNTIME;
        }
    } else {
        aclrtMemcpyKind rtKind;
        HcclResult ret = MemcpyKindTranslate(kind, &rtKind);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        aclError aclRet = aclrtMemcpyAsync(dst, destMax, src, count, rtKind, stream);
        HCCL_DEBUG(
            "Call HcclRtMemAsyncCopy, return value[%d], dstAddr[%p], destMax[%llu], "
            "srcAddr[%p], count[%llu], rtKind[%d]",
            static_cast<int>(aclRet), dst, destMax, src, count, static_cast<int>(rtKind));
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR(
                "[HcclRtMemAsyncCopy] rt memory async copy failed, "
                "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d], stream[%p].",
                static_cast<int>(aclRet), dst, destMax, src, count, static_cast<int>(rtKind), stream);
            return HCCL_E_RUNTIME;
        }
    }
    return HCCL_SUCCESS;
}

// 对应 InfoStore hrtMemcpyAddrAsync (hcom_acl_adapter.cc): 带偏移的二级指针地址异步拷贝
HcclResult HcclRtMemAddrAsyncCopyWithOffset(
    void* dst, uint64_t destMax, uint64_t dstOffset, const void* src, uint64_t count, uint64_t srcOffset,
    aclrtStream stream)
{
    // stream 判空保持最前 (合法性与长度无关); 零长度短路在数据指针判空之前,
    // 允许空输出定形场景 (空指针, count=0) 无操作成功
    if (stream == nullptr) {
        HCCL_ERROR("[HcclRtMemAddrAsyncCopyWithOffset] invalid param, stream is null.");
        return HCCL_E_PARA;
    }
    if (count == 0) {
        HCCL_WARNING("[HcclRtMemAddrAsyncCopyWithOffset] count is zero");
        return HCCL_SUCCESS;
    }
    if (dst == nullptr || src == nullptr) {
        HCCL_ERROR("[HcclRtMemAddrAsyncCopyWithOffset] invalid param, dst=%p src=%p.", dst, src);
        return HCCL_E_PARA;
    }

    aclError aclRet = aclrtMemcpyAsyncWithOffset(
        reinterpret_cast<void**>(dst), destMax, dstOffset, reinterpret_cast<const void**>(const_cast<void*>(src)),
        count, srcOffset, ACL_MEMCPY_INNER_DEVICE_TO_DEVICE, stream);
    HCCL_DEBUG(
        "Call HcclRtMemAddrAsyncCopyWithOffset, return value[%d], dstAddr[%p], destMax[%llu], dstOffset[%llu], "
        "srcAddr[%p], count[%llu], srcOffset[%llu]",
        static_cast<int>(aclRet), dst, destMax, dstOffset, src, count, srcOffset);
    if (aclRet == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[HcclRtMemAddrAsyncCopyWithOffset] not supported, ret[%d].", static_cast<int>(aclRet));
        return HCCL_E_NOT_SUPPORT;
    }
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HcclRtMemAddrAsyncCopyWithOffset] rt memory async addr copy with offset failed, "
            "return[%d], para: dstAddr[%p], destMax[%llu], dstOffset[%llu], srcAddr[%p], count[%llu], "
            "srcOffset[%llu], stream[%p].",
            static_cast<int>(aclRet), dst, destMax, dstOffset, src, count, srcOffset, stream);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
