/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_exec_registry.h"

namespace ops_hccl {

CollAlgExecRegistry &CollAlgExecRegistry::Instance()
{
    static CollAlgExecRegistry globalExecRegistry;
    return globalExecRegistry;
}

HcclResult CollAlgExecRegistry::Register(const std::string &tag, const CollExecCreator &collAlgExecCreator)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (execCreators_.find(tag) != execCreators_.end()) {
        HCCL_WARNING("[CollAlgExecRegistry]Exec tag[%s] already registered.", tag.c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    execCreators_.emplace(tag, collAlgExecCreator);
    return HcclResult::HCCL_SUCCESS;
}

std::unique_ptr<ExecutorBase> CollAlgExecRegistry::GetAlgExec(
    const std::string &tag)
{
    if (execCreators_.find(tag) == execCreators_.end()) {
        HCCL_DEBUG("[CollAlgExecRegistry]Creator for executor tag[%s] has not registered.", tag.c_str());
        return nullptr;
    }
    HCCL_DEBUG("[CollAlgExecRegistry][GetAlgExec]get executor by algName[%s].", tag.c_str());
    return std::unique_ptr<ExecutorBase>(execCreators_[tag]());
}

}