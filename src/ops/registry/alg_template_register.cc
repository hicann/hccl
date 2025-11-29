/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"

namespace ops_hccl {

AlgTemplateRegistry &AlgTemplateRegistry::Instance()
{
    static AlgTemplateRegistry globalTempRegistry;
    return globalTempRegistry;
}

AlgTemplateRegistry::AlgTemplateRegistry()
{
    tempCreators_.resize(TemplateType::TEMPLATE_CUSTOM_MAX_NUM, nullptr);
}

HcclResult AlgTemplateRegistry::Register(const TemplateType type, const AlgTemplateCreator &algTemplateCreator)
{
    if ((type >= TemplateType::TEMPLATE_NATIVE_MAX_NUM && type <= TemplateType::TEMPLATE_CUSTOM_BEGIN)
        || type >= TemplateType::TEMPLATE_CUSTOM_MAX_NUM) {
        HCCL_ERROR("[AlgTemplateRegistry]template type[%d] out of range.", type);
        return HcclResult::HCCL_E_INTERNAL;
    }

    const std::lock_guard<std::mutex> lock(mu_);
    if (tempCreators_[type] != nullptr) {
        HCCL_ERROR("[AlgTemplateRegistry]template type[%d] already registered.", type);
        return HcclResult::HCCL_E_INTERNAL;
    }
    tempCreators_[type] = algTemplateCreator;
    return HcclResult::HCCL_SUCCESS;
}

std::unique_ptr<AlgTemplateBase> AlgTemplateRegistry::GetAlgTemplate(const TemplateType type)
{
    if ((type >= TemplateType::TEMPLATE_NATIVE_MAX_NUM && type <= TemplateType::TEMPLATE_CUSTOM_BEGIN)
        || type >= TemplateType::TEMPLATE_CUSTOM_MAX_NUM) {
        HCCL_ERROR("[AlgTemplateRegistry]template type[%d] out of range.", type);
        return nullptr;
    }

    if (tempCreators_[type] == nullptr) {
        HCCL_DEBUG("[AlgTemplateRegistry]Creator for template type[%d] has not registered.", type);
        return nullptr;
    }
    HCCL_DEBUG("[AlgTemplateRegistry][GetAlgTemplate]get template by type[%d]", type);
    return std::unique_ptr<AlgTemplateBase>(tempCreators_[type]());
}

}