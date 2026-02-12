/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_v2_template_register.h"

namespace ops_hccl {

InsAlgTemplateRegistry &InsAlgTemplateRegistry::Instance()
{
    static InsAlgTemplateRegistry globalTempRegistry;
    return globalTempRegistry;
}

HcclResult InsAlgTemplateRegistry::Register(const std::string &name, const InsAlgTemplateCreator &algTemplateCreator)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if ((tempCreators_.find(name) != tempCreators_.end()) && (tempCreators_[name] != nullptr)) {
        HCCL_ERROR("[InsAlgTemplateRegistry]template name[%s] already registered.", name.c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    tempCreators_[name] = algTemplateCreator;
    return HcclResult::HCCL_SUCCESS;
}

std::unique_ptr<InsAlgTemplateBase> InsAlgTemplateRegistry::GetAlgTemplate(const std::string &name)
{
    if (tempCreators_.find(name) == tempCreators_.end()) {
        HCCL_ERROR("[InsAlgTemplateRegistry]template name[%s] not found.", name.c_str());
        return nullptr;
    }

    if (tempCreators_[name] == nullptr) {
        HCCL_DEBUG("[InsAlgTemplateRegistry]Creator for template name[%s] has not registered.", name.c_str());
        return nullptr;
    }
    HCCL_DEBUG("[InsAlgTemplateRegistry][GetAlgTemplate]get template by name[%s]", name.c_str());
    return std::unique_ptr<InsAlgTemplateBase>(tempCreators_[name]());
}

}