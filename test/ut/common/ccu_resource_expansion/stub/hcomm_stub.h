/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RESOURCE_EXPANSION_HCOMM_STUB_H
#define CCU_RESOURCE_EXPANSION_HCOMM_STUB_H

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "ccu_res_dl.h"
#include "dev_type.h"
#include "hccl_host_comm_dl.h"

namespace ut_stub {
constexpr CcuInsHandle ORIGINAL_INS = 100;
constexpr CcuInsHandle EXTRA_INS = 200;
constexpr CcuKernelHandle OLD_KERNEL = 300;
constexpr uint32_t KERNEL_NEED = 600;
constexpr uint32_t ORIGINAL_CAPACITY = 32;
constexpr uint32_t DEVICE_REMAIN = 4096;
constexpr char EXTRA_TAG[] = "hccl_ccu_extra_ins_commA";
constexpr std::array<HcommCcuResType, 6> RES_TYPES = {
    HCOMM_CCU_RES_TYPE_LOOP,    HCOMM_CCU_RES_TYPE_CCU_BUF, HCOMM_CCU_RES_TYPE_VARIABLE,
    HCOMM_CCU_RES_TYPE_ADDRESS, HCOMM_CCU_RES_TYPE_EVENT,   HCOMM_CCU_RES_TYPE_CCU_THREAD,
};

// 模拟HCCL专用context的句柄记录，不代替被测资源申请逻辑。
struct ExpansionCtx {
    CcuInsHandle insHandle = 0;
};

using ResourceNumbers = std::map<HcommCcuResType, uint32_t>;
using ContextKey = std::tuple<uintptr_t, std::string, CommEngine>;

struct Descriptor {
    uint32_t dieId = 0;
    ResourceNumbers numbers;
};

struct MockResources {
    HcclDevType device = HcclDevType::DEV_TYPE_950;
    uint32_t remain = DEVICE_REMAIN;
    HcclResult callbackRegistrationResult = HCCL_SUCCESS;
    bool callbackSupported = true;
    HcclCommStateCallback cleanupCallback = nullptr;
    void* callbackArgs = nullptr;
    std::string callbackName;
    uint32_t callbackRegistrations = 0;
    HcclResult contextCreateResult = HCCL_SUCCESS;
    CcuResult destroyResult = CCU_SUCCESS;
    CcuResult registerResult = CCU_SUCCESS;
    HcommCcuResDescHandle nextDesc = 1;
    std::map<HcommCcuResDescHandle, Descriptor> descriptors;
    std::map<HcommCcuResDescHandle, uint32_t> releaseCounts;
    std::map<ContextKey, std::vector<uint8_t>> contexts;
    std::map<CcuInsHandle, ResourceNumbers> instances;
    std::vector<ResourceNumbers> createRequests;
    std::vector<CcuInsHandle> queriedInstances;
    std::vector<CcuInsHandle> registerTargets;
    std::vector<CcuInsHandle> destroyedInstances;
    std::vector<uint32_t> remainQueryDies;
    std::vector<std::string> contextQueries;
    std::vector<CcuKernelHandle> oldKernels{OLD_KERNEL};
    CcuInsHandle oldKernelOwner = ORIGINAL_INS;
    uint32_t assignCalls = 0;
    uint32_t deviceQueries = 0;
    uint32_t negotiationCalls = 0;
    bool localResourcesAvailable = false;
    HcclResult negotiationResult = HCCL_SUCCESS;
    std::array<uint8_t, 16> cclBuffer{};
};

extern thread_local MockResources g_resources;

ResourceNumbers KernelNumbers();
ContextKey MakeKey(HcclComm comm, const char* tag, CommEngine engine);
void KernelForResourceQuery();
} // namespace ut_stub

#endif // CCU_RESOURCE_EXPANSION_HCOMM_STUB_H
