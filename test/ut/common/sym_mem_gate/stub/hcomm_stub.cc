/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_stub.h"

namespace ut_stub {
int g_utHcommVersion = 0;
int g_utSymWinGetRet = HCCL_SUCCESS;
bool g_utSymWinReturnNull = false;
unsigned g_utSymWinGetCalls = 0;
unsigned g_utSymWinFailFirstN = 0;

void UtSymWinStubReset()
{
    g_utHcommVersion = 0; // 0 由 GetHcommVersion 桩按 CANN_VERSION(9,1,0) 兜底
    g_utSymWinGetRet = HCCL_SUCCESS;
    g_utSymWinReturnNull = false;
    g_utSymWinGetCalls = 0;
    g_utSymWinFailFirstN = 0;
}
} // namespace ut_stub

// GetHcommVersion：运行期版本门禁（RefreshSymmetricMemory 的基础门禁输入）
extern "C" int GetHcommVersion(void)
{
    if (ut_stub::g_utHcommVersion > 0) {
        return ut_stub::g_utHcommVersion;
    }
    return CANN_VERSION(9, 1, 0);
}

// HcclCommSymWinGet：DECL_WEAK_FUNC 弱符号，此处强定义覆盖 dlsym 实现。
// HcclCommSymWindow 为不透明句柄，测试返回哨兵值；探测是否成功由用例控制。
extern "C" HcclResult
HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset)
{
    (void)comm;
    (void)ptr;
    (void)size;
    ++ut_stub::g_utSymWinGetCalls;
    HcclResult ret = static_cast<HcclResult>(ut_stub::g_utSymWinGetRet);
    if (ut_stub::g_utSymWinFailFirstN > 0 && ut_stub::g_utSymWinGetCalls <= ut_stub::g_utSymWinFailFirstN) {
        ret = HCCL_E_INTERNAL;
    }
    if (winHandle != nullptr && ret == HCCL_SUCCESS) {
        *winHandle = ut_stub::g_utSymWinReturnNull ? nullptr : reinterpret_cast<HcclCommSymWindow>(0x1234U);
    }
    if (offset != nullptr) {
        *offset = 0;
    }
    return ret;
}
