/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RE_TEMPLATE_FACTORY_H
#define RE_TEMPLATE_FACTORY_H

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include "base_template.h"
#include "log.h"

namespace ops_hccl {

using TemplateCreator
    = std::function<std::unique_ptr<BaseTemplate>(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)>;

class TemplateRegistry {
public:
    static TemplateRegistry& Instance()
    {
        static TemplateRegistry instance;
        return instance;
    }

    void Register(HcclCMDType cmdType, HcclAlgoType algType, TemplateCreator creator)
    {
        registry_[{static_cast<int>(cmdType), static_cast<int>(algType)}] = std::move(creator);
    }

    std::unique_ptr<BaseTemplate>
    Create(HcclCMDType cmdType, HcclAlgoType algType, u32 myRank, std::vector<u32> ranks, TemplateDesc desc) const
    {
        auto it = registry_.find({static_cast<int>(cmdType), static_cast<int>(algType)});
        if (it == registry_.end()) {
            HCCL_ERROR(
                "[GetTemplate] unsupported template: cmdType[%d], algType[%d].", static_cast<int>(cmdType),
                static_cast<int>(algType));
            return nullptr;
        }
        return it->second(myRank, std::move(ranks), std::move(desc));
    }

private:
    struct Key {
        int cmdType;
        int algType;
        bool operator<(const Key& o) const { return cmdType != o.cmdType ? cmdType < o.cmdType : algType < o.algType; }
    };
    std::map<Key, TemplateCreator> registry_;
};

inline std::unique_ptr<BaseTemplate>
GetTemplate(const TemplateDesc& templateDesc, const std::vector<u32>& ranks, u32 myRank)
{
    return TemplateRegistry::Instance().Create(
        templateDesc.hcclCmdType, templateDesc.algType, myRank, ranks, templateDesc);
}

} // namespace ops_hccl

#define REGISTER_RE_TEMPLATE(cmdType, algType, TemplateClass)                                              \
    namespace {                                                                                            \
        struct TemplateClass##Registrar {                                                                  \
            TemplateClass##Registrar()                                                                     \
            {                                                                                              \
                ::ops_hccl::TemplateRegistry::Instance().Register(                                         \
                    cmdType, algType,                                                                      \
                    [](u32 myRank, std::vector<u32> ranks,                                                 \
                       ::ops_hccl::TemplateDesc desc) -> std::unique_ptr<::ops_hccl::BaseTemplate> {       \
                        return std::make_unique<TemplateClass>(myRank, std::move(ranks), std::move(desc)); \
                    });                                                                                    \
            }                                                                                              \
        };                                                                                                 \
        static TemplateClass##Registrar g_##TemplateClass##_registrar;                                     \
    } // anonymous namespace

#endif // RE_TEMPLATE_FACTORY_H
