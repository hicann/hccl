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
#include "hcomm_diag_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hcommRegOpInfoPtr)(const char*, void*, size_t) = nullptr;
HcclResult (*hcommRegOpTaskExceptionPtr)(const char*, HcommGetOpInfoCallback) = nullptr;

// 添加支持标志（静态，默认 false）
static bool g_hcommRegOpInfoSupported = false;
static bool g_hcommRegOpTaskExceptionSupported = false;

// ---------- 桩函数定义 ----------
static HcclResult StubHcommRegOpInfo(const char* commId, void* opInfo, size_t size) {
    (void)commId; (void)opInfo; (void)size;
    HCCL_ERROR("[HcclWrapper] HcommRegOpInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcommRegOpTaskException(const char* commId, HcommGetOpInfoCallback callback) {
    (void)commId; (void)callback;
    HCCL_ERROR("[HcclWrapper] HcommRegOpTaskException not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// 初始化
void HcommDiagDlInit(void* libHcommHandle) {
    #define SET_PTR(ptr, handle, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(handle, name); \
            if (ptr == nullptr) { \
                ptr = stub; \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcommRegOpInfoPtr, libHcommHandle, "HcommRegOpInfo", StubHcommRegOpInfo, g_hcommRegOpInfoSupported);
    SET_PTR(hcommRegOpTaskExceptionPtr, libHcommHandle, "HcommRegOpTaskException", StubHcommRegOpTaskException, g_hcommRegOpTaskExceptionSupported);

    #undef SET_PTR
}

void HcommDiagDlFini(void) {
    hcommRegOpInfoPtr = StubHcommRegOpInfo;
    g_hcommRegOpInfoSupported = false;
    hcommRegOpTaskExceptionPtr = StubHcommRegOpTaskException;
    g_hcommRegOpTaskExceptionSupported = false;
}

// ---------- 对外提供的查询接口 ----------
extern "C" bool HcommIsSupportHcommRegOpInfo(void) {
    return g_hcommRegOpInfoSupported;
}
extern "C" bool HcommIsSupportHcommRegOpTaskException(void) {
    return g_hcommRegOpTaskExceptionSupported;
}