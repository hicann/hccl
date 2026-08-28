/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ALG_ATTRS_REGISTRY_H
#define HCCL_ALG_ATTRS_REGISTRY_H

#include "alg_attrs.h"
#include "cost_model.h"
#include <vector>
#include <map>
#include <string>

namespace ops_hccl {

// Forward declarations for DFX logging (implementations in alg_parse.cc)
std::string HcclCMDTypeToString(HcclCMDType opType);
std::string OpExecuteConfigToString(OpExecuteConfig engine);

class AlgAttrsRegistry {
public:
    static AlgAttrsRegistry& Instance();

    void Register(const AlgAttrs& attrs);
    const AlgAttrs* Get(const std::string& name) const;
    const std::vector<AlgAttrs>& GetAll() const;

    static void ParseAlgName(const std::string& algoName, AlgAttrs& a);
    static std::string GetAlgAttrsSummary(const AlgAttrs& a);

private:
    AlgAttrsRegistry() = default;
    std::vector<AlgAttrs> attrs_;
    std::map<std::string, size_t> nameToIndex_;
};

} // namespace ops_hccl

#ifndef AICPU_COMPILE
#define REGISTER_ALG_ATTRS(algoName, ...)                                                                         \
    static const bool s_attrs_##algoName = [] {                                                                   \
        ::ops_hccl::AlgAttrs a;                                                                                   \
        auto& topo = a.topo;                                                                                      \
        auto& op = a.op;                                                                                          \
        ::ops_hccl::AlgAttrsRegistry::ParseAlgName(#algoName, a);                                                 \
        __VA_ARGS__;                                                                                              \
        HCCL_INFO(                                                                                                \
            "[DFX_REGISTER_ALG_ATTRS] name=%s, opType=%s, engine=%s, "                                            \
            "algoTypes=%s, minTopoLevelNum=%d, maxTopoLevelNum=%d, "                                              \
            "isSupportProd=%d, isSupportInplace=%d, "                                                             \
            "isSupportFloatOrderPreserved=%d, hasTopoCustomCheck=%d, "                                            \
            "hasOpCustomCheck=%d.",                                                                               \
            a.name.c_str(), ::ops_hccl::HcclCMDTypeToString(a.opType).c_str(),                                    \
            ::ops_hccl::OpExecuteConfigToString(a.engine).c_str(),                                                \
            ::ops_hccl::AlgAttrsRegistry::GetAlgAttrsSummary(a).c_str(), a.topo.minTopoLevelNum,                  \
            a.topo.maxTopoLevelNum, a.op.isSupportProd, a.op.isSupportInplace, a.op.isSupportFloatOrderPreserved, \
            a.topo.topoCustomCheck != nullptr ? 1 : 0, a.op.opCustomCheck != nullptr ? 1 : 0);                    \
        ::ops_hccl::AlgAttrsRegistry::Instance().Register(a);                                                     \
        return true;                                                                                              \
    }()

#else // AICPU_COMPILE

#define REGISTER_ALG_ATTRS(algoName, ...)

#endif // AICPU_COMPILE

#endif // HCCL_ALG_ATTRS_REGISTRY_H
