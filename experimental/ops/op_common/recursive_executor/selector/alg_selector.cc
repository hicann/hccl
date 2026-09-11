/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_selector.h"

namespace ops_hccl {

AlgSelector& AlgSelector::Instance()
{
    static AlgSelector instance;
    return instance;
}

HcclResult AlgSelector::Register(const std::string& algName, HcclAlgorithm algo)
{
    std::lock_guard<std::mutex> lock(mu_);
    algMap_[algName] = std::move(algo);
    return HCCL_SUCCESS;
}

bool AlgSelector::GetAlgorithm(const std::string& algName, HcclAlgorithm& algo) const
{
    std::lock_guard<std::mutex> lock(mu_);
    auto it = algMap_.find(algName);
    if (it == algMap_.end()) {
        return false;
    }
    algo = it->second;
    return true;
}

} // namespace ops_hccl
