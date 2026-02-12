/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include "alg_v2_template_base.h"

namespace ops_hccl {

using InsAlgTemplateCreator = std::function<InsAlgTemplateBase *()>;

template <typename P>
static InsAlgTemplateBase *DefaultTemplateCreatorV2()
{
    static_assert(std::is_base_of<InsAlgTemplateBase, P>::value,
        "Template type must derived from Hccl::InsAlgTemplateBase");
    return new (std::nothrow) P();
}

class InsAlgTemplateRegistry {
public:
    static InsAlgTemplateRegistry &Instance();
    InsAlgTemplateRegistry() {};
    HcclResult Register(const std::string &name, const InsAlgTemplateCreator &algTemplateCreator);
    std::unique_ptr<InsAlgTemplateBase> GetAlgTemplate(const std::string &name);

private:
    std::map<std::string, InsAlgTemplateCreator> tempCreators_;
    mutable std::mutex mu_;
};

#define REGISTER_TEMPLATE_V2_HELPER(ctr, name, insAlgTempBase)       \
    static HcclResult g_func_##insAlgTempBase##_##ctr             \
        = InsAlgTemplateRegistry::Instance().Register(name, DefaultTemplateCreatorV2<insAlgTempBase>)

#define REGISTER_TEMPLATE_V2(name, insAlgTempBase) REGISTER_TEMPLATE_V2_HELPER(__COUNTER__, name, insAlgTempBase)

}
