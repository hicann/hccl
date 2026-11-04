/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "ccu_fallback_c.h"
#include "ccu_launch_dl.h"
#include "hcomm_stub.h"
#include "hccl_ccu_res_dl.h"
#include "hccl_host_comm_dl.h"
#include "hccl_res_dl.h"
#include "hcomm_dlsym.h"
#include "hcomm_primitives_dl.h"
#include "op_common.h"
#include "securec.h"

using namespace ops_hccl;

namespace ut_stub {
thread_local MockResources g_resources;

ResourceNumbers KernelNumbers()
{
    ResourceNumbers numbers;
    for (auto type : RES_TYPES) {
        numbers[type] = 1;
    }
    numbers[HCOMM_CCU_RES_TYPE_VARIABLE] = KERNEL_NEED;
    numbers[HCOMM_CCU_RES_TYPE_ADDRESS] = KERNEL_NEED;
    return numbers;
}

ContextKey MakeKey(HcclComm comm, const char* tag, CommEngine engine)
{
    return {reinterpret_cast<uintptr_t>(comm), tag, engine};
}

void KernelForResourceQuery() {}
} // namespace ut_stub

using namespace ut_stub;

// 以下仅模拟HCOMM资源接口，不实现实例选择、扩容和回退决策。
extern "C" {
CcuResult HcommCcuInsResDescCreate(uint32_t dieId, HcommCcuResDescHandle* desc)
{
    if (desc == nullptr) {
        return CCU_E_PTR;
    }
    *desc = g_resources.nextDesc++;
    g_resources.descriptors.emplace(*desc, Descriptor{dieId, {}});
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescDestroy(HcommCcuResDescHandle desc)
{
    ++g_resources.releaseCounts[desc];
    if (g_resources.descriptors.erase(desc) == 0) {
        ADD_FAILURE() << "需求描述符被重复释放或使用了无效句柄: " << desc;
        return CCU_E_NOT_FOUND;
    }
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescSetNum(HcommCcuResDescHandle desc, HcommCcuResType type, uint32_t number)
{
    auto it = g_resources.descriptors.find(desc);
    if (it == g_resources.descriptors.end()) {
        return CCU_E_NOT_FOUND;
    }
    it->second.numbers[type] = number;
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsResDescQueryNum(HcommCcuResDescHandle desc, HcommCcuResType type, uint32_t* number)
{
    if (number == nullptr) {
        return CCU_E_PTR;
    }
    auto it = g_resources.descriptors.find(desc);
    if (it == g_resources.descriptors.end()) {
        return CCU_E_NOT_FOUND;
    }
    *number = it->second.numbers[type];
    return CCU_SUCCESS;
}

CcuResult HcommCcuKernelQueryResReq(const void*, const void**, uint32_t, HcommCcuResDescHandle desc)
{
    auto it = g_resources.descriptors.find(desc);
    if (it == g_resources.descriptors.end()) {
        return CCU_E_NOT_FOUND;
    }
    it->second.numbers = KernelNumbers();
    return CCU_SUCCESS;
}

CcuResult HcommCcuQueryRemainResDesc(HcommCcuResDescHandle desc)
{
    auto it = g_resources.descriptors.find(desc);
    if (it == g_resources.descriptors.end()) {
        return CCU_E_NOT_FOUND;
    }
    g_resources.remainQueryDies.push_back(it->second.dieId);
    if (it->second.dieId != 0) {
        return CCU_E_UNAVAIL;
    }
    for (auto type : RES_TYPES) {
        it->second.numbers[type] = g_resources.remain;
    }
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsQueryResDesc(CcuInsHandle ins, HcommCcuResDescHandle desc)
{
    g_resources.queriedInstances.push_back(ins);
    auto descIt = g_resources.descriptors.find(desc);
    auto insIt = g_resources.instances.find(ins);
    if (descIt == g_resources.descriptors.end() || insIt == g_resources.instances.end()) {
        return CCU_E_NOT_FOUND;
    }
    descIt->second.numbers = insIt->second;
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsCreate(const HcommCcuResDescHandle* descs, uint32_t count, CcuInsHandle* ins)
{
    if (descs == nullptr || ins == nullptr || count != 1) {
        return CCU_E_PARA;
    }
    auto it = g_resources.descriptors.find(descs[0]);
    if (it == g_resources.descriptors.end()) {
        ADD_FAILURE() << "创建额外实例时，需求描述符已经失效";
        return CCU_E_NOT_FOUND;
    }
    g_resources.createRequests.push_back(it->second.numbers);
    *ins = EXTRA_INS;
    g_resources.instances[*ins] = it->second.numbers;
    return CCU_SUCCESS;
}

CcuResult HcommCcuInsDestroy(CcuInsHandle ins)
{
    g_resources.destroyedInstances.push_back(ins);
    if (g_resources.destroyResult != CCU_SUCCESS) {
        return g_resources.destroyResult;
    }
    g_resources.instances.erase(ins);
    if (ins == g_resources.oldKernelOwner) {
        g_resources.oldKernels.clear();
    }
    return CCU_SUCCESS;
}

HcclResult HcclCommQueryAssignedCcuIns(HcclComm, CcuInsHandle* handles, uint32_t* count)
{
    if (handles == nullptr || count == nullptr) {
        return HCCL_E_PTR;
    }
    *handles = ORIGINAL_INS;
    *count = 1;
    return HCCL_SUCCESS;
}

HcclResult HcclCommAssignCcuIns(HcclComm, CcuInsHandle)
{
    ++g_resources.assignCalls;
    return HCCL_E_PARA;
}

HcclResult HcclCommRegCommStateCallback(const char* name, HcclCommStateCallback cb, void* args)
{
    ++g_resources.callbackRegistrations;
    if (g_resources.callbackRegistrationResult != HCCL_SUCCESS) {
        return g_resources.callbackRegistrationResult;
    }
    g_resources.callbackName = name;
    g_resources.cleanupCallback = cb;
    g_resources.callbackArgs = args;
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm comm, char* name)
{
    if (comm == nullptr || name == nullptr) {
        return HCCL_E_PTR;
    }
    return strcpy_s(name, sizeof("commA"), "commA") == EOK ? HCCL_SUCCESS : HCCL_E_INTERNAL;
}

HcclResult HcclGetDeviceType(HcclDevType& device)
{
    ++g_resources.deviceQueries;
    device = g_resources.device;
    return HCCL_SUCCESS;
}

HcclResult HcclGetHcclBuffer(HcclComm, void** addr, uint64_t* size)
{
    if (addr == nullptr || size == nullptr) {
        return HCCL_E_PTR;
    }
    *addr = g_resources.cclBuffer.data();
    *size = g_resources.cclBuffer.size();
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxGet(HcclComm comm, const char* tag, CommEngine engine, void** ctx, uint64_t* size)
{
    if (tag == nullptr || ctx == nullptr || size == nullptr) {
        return HCCL_E_PTR;
    }
    g_resources.contextQueries.emplace_back(tag);
    auto it = g_resources.contexts.find(MakeKey(comm, tag, engine));
    if (it == g_resources.contexts.end()) {
        return HCCL_E_NOT_FOUND;
    }
    *ctx = it->second.data();
    *size = it->second.size();
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxCreate(HcclComm comm, const char* tag, CommEngine engine, uint64_t size, void** ctx)
{
    if (g_resources.contextCreateResult != HCCL_SUCCESS) {
        return g_resources.contextCreateResult;
    }
    if (tag == nullptr || ctx == nullptr || size == 0) {
        return HCCL_E_PARA;
    }
    auto inserted = g_resources.contexts.emplace(MakeKey(comm, tag, engine), std::vector<uint8_t>(size));
    if (!inserted.second) {
        return HCCL_E_PARA;
    }
    *ctx = inserted.first->second.data();
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxDestroy(HcclComm comm, const char* tag, CommEngine engine)
{
    return g_resources.contexts.erase(MakeKey(comm, tag, engine)) != 0 ? HCCL_SUCCESS : HCCL_E_NOT_FOUND;
}

CcuResult HcommCcuKernelRegisterStart(CcuInsHandle ins)
{
    g_resources.registerTargets.push_back(ins);
    return CCU_SUCCESS;
}

CcuResult HcommCcuKernelRegister(
    CcuInsHandle, uint32_t, const char*, const void*, const void**, uint32_t, CcuKernelHandle* kernel)
{
    if (kernel == nullptr) {
        return CCU_E_PTR;
    }
    if (g_resources.registerResult == CCU_SUCCESS) {
        *kernel = OLD_KERNEL + 1;
    }
    return g_resources.registerResult;
}

CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle) { return CCU_SUCCESS; }

HcclResult CheckCcuResNegotiationC(HcclComm, const void*, bool localResourcesAvailable)
{
    ++g_resources.negotiationCalls;
    g_resources.localResourcesAvailable = localResourcesAvailable;
    return g_resources.negotiationResult;
}

#define CCU_EXPANSION_SUPPORTED(name) \
    bool HcommIsSupport##name(void) { return true; }
CCU_EXPANSION_SUPPORTED(HcommCcuInsResDescCreate)
CCU_EXPANSION_SUPPORTED(HcommCcuInsResDescDestroy)
CCU_EXPANSION_SUPPORTED(HcommCcuInsResDescSetNum)
CCU_EXPANSION_SUPPORTED(HcommCcuInsResDescQueryNum)
CCU_EXPANSION_SUPPORTED(HcommCcuInsCreate)
CCU_EXPANSION_SUPPORTED(HcommCcuInsDestroy)
CCU_EXPANSION_SUPPORTED(HcommCcuInsQueryResDesc)
CCU_EXPANSION_SUPPORTED(HcommCcuQueryRemainResDesc)
CCU_EXPANSION_SUPPORTED(HcommCcuKernelQueryResReq)
CCU_EXPANSION_SUPPORTED(HcclCommAssignCcuIns)
CCU_EXPANSION_SUPPORTED(HcclCommQueryAssignedCcuIns)
CCU_EXPANSION_SUPPORTED(HcclChannelDestroy)
#undef CCU_EXPANSION_SUPPORTED

bool HcommIsSupportHcclChannelQuery(void) { return false; }
// op_common.cc HcclGetChannel/CheckSupportNda 引用（真实符号由 hcomm_primitives_dl.cc 生成，本 UT 用桩），按不支持 NDA
// 特性查询处理
bool HcommIsSupportHcommEndpointCheckFeature(void) { return false; }
bool HcommIsSupportHcclThreadAcquireWithConfig(void) { return false; }
bool HcommIsSupportHcclDedicatedThreadAcquire(void) { return false; }
int GetHcommVersion(void) { return CANN_VERSION_NUM; }
bool HcommIsSupportHcclCommRegCommStateCallback(void) { return g_resources.callbackSupported; }
bool IsHcommDefaultTimeoutSupported(void) { return false; }
} // extern "C"
