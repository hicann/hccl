/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADAPTER_ACL_H
#define ADAPTER_ACL_H

#include "log.h"
#include "dev_type.h"
#include "hccl_common.h"
#include "acl_base.h"
#include "acl_rt.h"

/* ACL_DEV_ATTR_DEVICE_FORM_FACTOR是新版acl_rt.h才有的枚举值, 老CANN上引用会编译不过。
 * 枚举对预处理器不可见, 因此探测与它同批引入、且只为它服务的宏ACL_DEVICE_FORM_FACTOR_POD。
 */
#ifndef HCCL_SUPPORT_DEV_FORM_FACTOR
#ifdef ACL_DEVICE_FORM_FACTOR_POD
#define HCCL_SUPPORT_DEV_FORM_FACTOR 1
#else
#define HCCL_SUPPORT_DEV_FORM_FACTOR 0
#endif
#endif

namespace ops_hccl {

#define ACLCHECK(cmd)                                                                                           \
    do {                                                                                                        \
        aclError ret = cmd;                                                                                     \
        if (ret != ACL_SUCCESS) {                                                                               \
            HCCL_ERROR("acl interface return err %s:%d, retcode: %d.\n", __FILE__, __LINE__, ret);              \
            if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {                                                        \
                HCCL_ERROR("memory allocation error, check whether the current memory space is sufficient.\n"); \
            }                                                                                                   \
            return HCCL_E_RUNTIME;                                                                              \
        }                                                                                                       \
    } while (0)

HcclResult haclrtGetPairDeviceLinkType(s32 phyDevId, s32 otherPhyDevId, LinkTypeInServer& linkType);

HcclResult
haclrtGetCaptureInfo(aclrtStream stream, aclmdlRICaptureStatus& captureStatus, u64& modelId, bool& isCapture);

HcclResult haclrtGetDeviceIndexByPhyId(u32 devicePhyId, u32& deviceLogicId);

/**
 * @param quiet 取不到时是否降噪。默认false保持原有行为(按ERROR打); 对"取不到就走降级值"的可选属性
 *              传true, 失败改按WARNING打。传true的调用方必须自己判返回值并给出降级值。
 */
HcclResult hcalrtGetDeviceInfo(u32 deviceId, aclrtDevAttr devAttr, s64& val, bool quiet = false);

HcclResult LoadBinaryFromFile(
    const char* binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode, aclrtBinHandle& binHandle);

HcclResult haclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind);

HcclResult haclrtMemset(void* dst, size_t destMax, int32_t value, size_t count);
} // namespace ops_hccl

#endif // ADAPTER_ACL_H
