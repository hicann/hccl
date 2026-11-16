/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_STUCK_DETECT_STUB_H
#define UT_STUCK_DETECT_STUB_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "hccl_host_comm_dl.h"

namespace ut_stub {

// 计数地址查询桩的可注入状态
extern bool g_supportCounterAddr;
extern HcclResult g_queryRet;
extern uint64_t g_headAddr;
extern uint64_t g_tailAddr;
extern uint64_t g_addOneAddr;
extern uint64_t g_token;
extern int g_queryCalls;

// 通信域状态回调桩的可注入状态
extern bool g_supportStateCallback;
extern HcclResult g_regRet;
extern int g_regCalls;
extern HcclCommStateCallback g_registeredCb;
// 查询桩内触发一次状态回调，模拟 Get→查询→Put 期间发生销毁/快恢
extern HcclCommStatePhase g_eraseDuringQueryPhase;

// 通信域身份桩：按句柄指针返回名称/rank，可修改以模拟“地址被新通信域复用”
extern std::unordered_map<void*, std::string> g_commNames;
extern std::unordered_map<void*, uint32_t> g_commRanks;
extern HcclResult g_commNameRet;
extern HcclResult g_rankRet;
extern int g_commNameCalls;

// 与 HcclGetCommName 桩一致地推导某句柄当前名称，供用例构造缓存条目
inline std::string CommNameOf(HcclComm comm)
{
    auto iter = g_commNames.find(comm);
    if (iter != g_commNames.end()) {
        return iter->second;
    }
    return std::string("ut_comm_") + std::to_string(reinterpret_cast<uintptr_t>(comm));
}

inline void ResetCommIdentity()
{
    g_commNames.clear();
    g_commRanks.clear();
    g_commNameRet = HCCL_SUCCESS;
    g_rankRet = HCCL_SUCCESS;
    g_commNameCalls = 0;
}

// AICPU 计数下发桩：记录 LocalReduce 调用
struct ReduceCall {
    uint64_t thread{0};
    uint64_t dst{0};
    uint64_t src{0};
    uint64_t count{0};
    int dataType{0};
    int reduceOp{0};
};

extern int32_t g_reduceRet;
extern std::vector<ReduceCall> g_reduceCalls;

} // namespace ut_stub

#endif // UT_STUCK_DETECT_STUB_H
