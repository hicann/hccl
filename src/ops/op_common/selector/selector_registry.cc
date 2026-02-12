/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selector_registry.h"

#include <string>
#include <iostream>
#include <map>
#include <mutex>

namespace ops_hccl {

SelectorRegistry *SelectorRegistry::Global()
{
    static SelectorRegistry *globalSelectorRegistry = new SelectorRegistry;
    return globalSelectorRegistry;
}

HcclResult SelectorRegistry::Register(u32 priority, AutoSelectorBase *selector)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (impls_.count(priority) != 0) {
        HCCL_ERROR("[Algo][Selector] priority %llu already registered.", priority);
        return HcclResult::HCCL_E_PARA;
    }

    impls_[priority] = selector;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult SelectorRegistry::RegisterByOpType(const HcclCMDType opType, u32 priority, AutoSelectorBase *selector)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (opTypeImpls_[opType].count(priority) != 0) {
        HCCL_ERROR("[Algo][Selector] opType %d priority %llu already registered.", opType, priority);
        return HcclResult::HCCL_E_PARA;
    }
    opTypeImpls_[opType][priority] = selector;
    return HcclResult::HCCL_SUCCESS;
}

std::map<u32, AutoSelectorBase *> SelectorRegistry::GetSelectorsByOpType(const HcclCMDType opType)
{
    if (opTypeImpls_.count(opType) == 0) {
        HCCL_WARNING("[Algo][Selector] opType %d has no selector registered.", opType);
        return {};
    }
    return opTypeImpls_[opType];
}

std::map<u32, AutoSelectorBase *> SelectorRegistry::GetAllSelectors()
{
    return impls_;
}

} // namespace Hccl
