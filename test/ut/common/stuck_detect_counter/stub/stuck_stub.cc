/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stuck_stub.h"

#include <cstdio>

#include "hccl_res_dl.h"
#include "hcomm_dlsym.h"
#include "hcomm_primitives.h"

namespace ut_stub {

bool g_supportCounterAddr = true;
HcclResult g_queryRet = HCCL_SUCCESS;
uint64_t g_headAddr = 0x1000;
uint64_t g_tailAddr = 0x2000;
uint64_t g_addOneAddr = 0x3000;
uint64_t g_token = 0xABCD;
int g_queryCalls = 0;

#ifdef UT_SCENARIO_UNSUPPORTED
bool g_supportStateCallback = false;
#else
bool g_supportStateCallback = true;
#endif

#ifdef UT_SCENARIO_REGFAIL
HcclResult g_regRet = HCCL_E_INTERNAL;
#else
HcclResult g_regRet = HCCL_SUCCESS;
#endif

int g_regCalls = 0;
HcclCommStateCallback g_registeredCb = nullptr;
HcclCommStatePhase g_eraseDuringQueryPhase = HCCL_COMM_STATE_PHASE_INVALID;

std::unordered_map<void*, std::string> g_commNames;
std::unordered_map<void*, uint32_t> g_commRanks;
HcclResult g_commNameRet = HCCL_SUCCESS;
HcclResult g_rankRet = HCCL_SUCCESS;
int g_commNameCalls = 0;

int32_t g_reduceRet = 0;
std::vector<ReduceCall> g_reduceCalls;

} // namespace ut_stub

extern "C" {

void HcommDlInit(void) {}

bool HcommIsSupportHcclStuckDetectGetCounterAddr(void) { return ut_stub::g_supportCounterAddr; }

HcclResult HcclStuckDetectGetCounterAddr(
    HcclComm comm, uint64_t* headOpCounterAddr, uint64_t* tailOpCounterAddr, uint64_t* addOneMemAddr,
    uint64_t* counterToken)
{
    ++ut_stub::g_queryCalls;
    if (ut_stub::g_queryRet != HCCL_SUCCESS) {
        return ut_stub::g_queryRet;
    }
    // 模拟查询过程中并发触发销毁/快恢回调
    if (ut_stub::g_eraseDuringQueryPhase != HCCL_COMM_STATE_PHASE_INVALID && ut_stub::g_registeredCb != nullptr) {
        (void)ut_stub::g_registeredCb(comm, ut_stub::g_eraseDuringQueryPhase, nullptr);
    }
    if (headOpCounterAddr != nullptr) {
        *headOpCounterAddr = ut_stub::g_headAddr;
    }
    if (tailOpCounterAddr != nullptr) {
        *tailOpCounterAddr = ut_stub::g_tailAddr;
    }
    if (addOneMemAddr != nullptr) {
        *addOneMemAddr = ut_stub::g_addOneAddr;
    }
    if (counterToken != nullptr) {
        *counterToken = ut_stub::g_token;
    }
    return HCCL_SUCCESS;
}

bool HcommIsSupportHcclCommRegCommStateCallback(void) { return ut_stub::g_supportStateCallback; }

HcclResult HcclCommRegCommStateCallback(const char* regName, HcclCommStateCallback cb, void* args)
{
    (void)regName;
    (void)args;
    ++ut_stub::g_regCalls;
    if (ut_stub::g_regRet != HCCL_SUCCESS) {
        return ut_stub::g_regRet;
    }
    ut_stub::g_registeredCb = cb;
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm comm, char* commName)
{
    ++ut_stub::g_commNameCalls;
    if (ut_stub::g_commNameRet != HCCL_SUCCESS) {
        return ut_stub::g_commNameRet;
    }
    if (commName == nullptr) {
        return HCCL_E_INTERNAL;
    }
    const std::string name = ut_stub::CommNameOf(comm);
    (void)std::snprintf(commName, ops_hccl::COMM_INDENTIFIER_MAX_LENGTH, "%s", name.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t* rank)
{
    if (ut_stub::g_rankRet != HCCL_SUCCESS) {
        return ut_stub::g_rankRet;
    }
    if (rank == nullptr) {
        return HCCL_E_INTERNAL;
    }
    auto iter = ut_stub::g_commRanks.find(comm);
    *rank = (iter != ut_stub::g_commRanks.end()) ? iter->second : 0U;
    return HCCL_SUCCESS;
}

} // extern "C"

int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    ut_stub::ReduceCall call;
    call.thread = thread;
    call.dst = reinterpret_cast<uint64_t>(dst);
    call.src = reinterpret_cast<uint64_t>(src);
    call.count = count;
    call.dataType = static_cast<int>(dataType);
    call.reduceOp = static_cast<int>(reduceOp);
    ut_stub::g_reduceCalls.push_back(call);
    return ut_stub::g_reduceRet;
}
