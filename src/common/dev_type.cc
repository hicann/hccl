/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "dev_type.h"
#include "log.h"
#include "acl_rt.h"
#include <dlfcn.h>

namespace {
static thread_local HcclDevType g_deviceType = HcclDevType::DEV_TYPE_COUNT;

HcclResult HcclGetSocVer(std::string &socName)
{
#ifndef AICPU_COMPILE
    const char *socNamePtr = aclrtGetSocName();
    CHK_PRT_RET((socNamePtr == nullptr), HCCL_ERROR("[Get][SocVer]errNo[0x%016llx] aclrtGet socName failed",
        HCCL_ERROR_CODE(HCCL_E_RUNTIME)), HCCL_E_RUNTIME);

    socName = socNamePtr;
#endif
    return HCCL_SUCCESS;
}
} // namespace

// 获取芯片类型
#ifdef __cplusplus
extern "C" {
#endif
HcclResult __HcclGetDeviceType(HcclDevType &devType)
{
    if (LIKELY((g_deviceType != HcclDevType::DEV_TYPE_COUNT))) {
        devType = g_deviceType;
        return HCCL_SUCCESS;
    }

    std::string socName;
    CHK_RET(HcclGetSocVer(socName));

    //  根据芯片版本号获取芯片类型
    HCCL_DEBUG("[HcclGetDeviceType]socName = %s.", socName.c_str());
    if (socName.find("Ascend950") != std::string::npos) {
        devType = HcclDevType::DEV_TYPE_950;
        g_deviceType = devType;
        HCCL_DEBUG("[HcclGetDeviceType]DeviceType = %d.", static_cast<s32>(g_deviceType));
        return HCCL_SUCCESS;
    }

    if (socName.find("Ascend910_96") != std::string::npos
        || socName.find("Ascend960") != std::string::npos
        || socName.find("ascend960") != std::string::npos) {
        devType = HcclDevType::DEV_TYPE_960;
        g_deviceType = devType;
        HCCL_DEBUG("[HcclGetDeviceType]DeviceType = %d.", static_cast<s32>(g_deviceType));
        return HCCL_SUCCESS;
    }

    auto iter = HCCL_SOC_VER_CONVERT.find(socName);
    if (iter == HCCL_SOC_VER_CONVERT.end()) {
        HCCL_ERROR("[Get][DeviceType]errNo[0x%016llx] rtGetSocVersion get illegal chipver, chip_ver[%s].", \
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), socName.c_str());
        return HCCL_E_RUNTIME;
    }
    devType = iter->second;
    g_deviceType = devType;
    return HCCL_SUCCESS;
}
hccl_weak_alias(__HcclGetDeviceType, HcclGetDeviceType);
#ifdef __cplusplus
}  // extern "C"
#endif
