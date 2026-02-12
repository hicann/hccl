/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {

CollAlgExecRegistryV2 &CollAlgExecRegistryV2::Instance()
{
    static CollAlgExecRegistryV2 globalExecRegistry;
    return globalExecRegistry;
}

HcclResult CollAlgExecRegistryV2::Register(const HcclCMDType type, const std::string &tag,
    const CollExecCreatorV2 &collExecCreator)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (execCreators_[type].count(tag) != 0) {
        HCCL_ERROR("[CollAlgExecRegistryV2]Exec tag[%s] already registered.", tag.c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    execCreators_[type].emplace(tag, collExecCreator);
    return HcclResult::HCCL_SUCCESS;
}

std::shared_ptr<InsCollAlgBase> CollAlgExecRegistryV2::GetAlgExec(const HcclCMDType type,
    const std::string &tag)
{
    if (execCreators_.count(type) == 0 || execCreators_[type].count(tag) == 0) {
        HCCL_DEBUG("[CollAlgExecRegistryV2]Creator for executor tag[%s] has not registered.", tag.c_str());
        return nullptr;
    }
    HCCL_DEBUG("[CollAlgExecRegistryV2][GetAlgExec]get executor by algName[%s].", tag.c_str());
    return std::shared_ptr<InsCollAlgBase>(execCreators_[type][tag]());
}

}