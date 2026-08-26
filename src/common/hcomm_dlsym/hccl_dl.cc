/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_dl.h"
#include <dlfcn.h>

#if !defined(weak_alias)
#define _weak_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((weak, alias(#name)))
#define weak_alias(name, aliasname) _weak_alias(name, aliasname)
#endif

#ifdef __cplusplus
extern "C" {
#endif
int __HcclDlclose(void* handle) { return dlclose(handle); }

void* __HcclDlsym(void* handle, const char* funcName) { return dlsym(handle, funcName); }

void* __HcclDlopen(const char* libName, int mode) { return dlopen(libName, mode); }
weak_alias(__HcclDlopen, HcclDlopen);
weak_alias(__HcclDlclose, HcclDlclose);
weak_alias(__HcclDlsym, HcclDlsym);

#ifdef __cplusplus
} // extern "C"
#endif
