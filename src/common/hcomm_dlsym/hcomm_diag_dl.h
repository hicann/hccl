/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DIAG_DL_H
#define HCOMM_DIAG_DL_H

#include "hcomm_diag.h"   // 原始头文件，包含所有声明和类型定义

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  (static_cast<HcclResult>(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcommRegOpInfoPtr)(const char*, void*, size_t);
extern HcclResult (*hcommRegOpTaskExceptionPtr)(const char*, HcommGetOpInfoCallback);

// 宏：将原始API名映射为函数指针调用
#define HcommRegOpInfo                (*hcommRegOpInfoPtr)
#define HcommRegOpTaskException        (*hcommRegOpTaskExceptionPtr)

// 查询函数声明
bool HcommIsSupportHcommRegOpInfo(void);
bool HcommIsSupportHcommRegOpTaskException(void);

// 动态库管理接口
void HcommDiagDlInit(void* libHcommHandle);
void HcommDiagDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_DIAG_DL_H