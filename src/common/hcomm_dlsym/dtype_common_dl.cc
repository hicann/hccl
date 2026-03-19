/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "dtype_common_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hrtGetDeviceTypePtr)(DevType &devType) = NULL;

// 添加支持标志（静态，默认 false）
static bool g_hrtGetDeviceTypeSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHrtGetDeviceType(DevType &devType) {
    (void)devType;
    HCCL_ERROR("[HcclWrapper] hrtGetDeviceType not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// 初始化
void DtypeCommonDlInit(void* libHcommHandle) {
    #define SET_PTR(ptr, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) { \
                ptr = stub; \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hrtGetDeviceTypePtr, "hrtGetDeviceType", StubHrtGetDeviceType, g_hrtGetDeviceTypeSupported);

    #undef SET_PTR
}

void DtypeCommonDlFini(void) {
    hrtGetDeviceTypePtr = StubHrtGetDeviceType;
    g_hrtGetDeviceTypeSupported = false;
}

// ---------- 对外提供的查询接口 ----------
extern "C" bool HcommIsSupportHrtGetDeviceType(void) {
    return g_hrtGetDeviceTypeSupported;
}