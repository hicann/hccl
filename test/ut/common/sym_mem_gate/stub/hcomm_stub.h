/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SYM_MEM_GATE_HCOMM_STUB_H
#define SYM_MEM_GATE_HCOMM_STUB_H

#include "hccl.h"
#include "hccl_host_comm_dl.h"

namespace ut_stub {
// HCOMM 版本桩：默认 9.1.0（对称内存最低门槛），用例按需下调验证版本门禁
extern int g_utHcommVersion;
// HcclCommSymWinGet 桩：探测结果控制
extern int g_utSymWinGetRet;          // 探测返回值，HCCL_SUCCESS 表示成功
extern bool g_utSymWinReturnNull;     // 成功时是否仍返回空句柄（模拟异常路径）
extern unsigned g_utSymWinGetCalls;   // 探测调用计数（区分 input/output 侧）
extern unsigned g_utSymWinFailFirstN; // 前 N 次探测强制失败（区分 input 侧/output 侧失败）

void UtSymWinStubReset(); // 恢复默认桩行为
} // namespace ut_stub

#endif // SYM_MEM_GATE_HCOMM_STUB_H
