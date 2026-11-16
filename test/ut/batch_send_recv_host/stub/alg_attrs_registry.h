/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_ALG_ATTRS_REGISTRY_H
#define BSR_HOST_TEST_ALG_ATTRS_REGISTRY_H

// AlgAttrsRegistry 的桩：注册照常发生（保留 REGISTER_ALG_ATTRS 的语义），这样 UT 能取回算法注册的
// 拓扑准入属性，断言 rank 上限、层数上限与 topoCustomCheck 的开关条件是否与设计一致。

#include <map>
#include <string>

#include "alg_param.h"

namespace ops_hccl {

constexpr uint8_t LEVEL0_TOPO_ANY = 0xFF;

class AlgAttrsRegistry {
public:
    static AlgAttrsRegistry& Instance()
    {
        static AlgAttrsRegistry instance;
        return instance;
    }

    void Register(const AlgAttrs& attrs) { attrs_[attrs.name] = attrs; }

    const AlgAttrs* Get(const std::string& name) const
    {
        auto it = attrs_.find(name);
        return (it == attrs_.end()) ? nullptr : &it->second;
    }

    static void ParseAlgName(const char* name, AlgAttrs& attrs) { attrs.name = name; }

    void Clear() { attrs_.clear(); }

private:
    std::map<std::string, AlgAttrs> attrs_;
};

} // namespace ops_hccl

#define REGISTER_ALG_ATTRS(algoName, ...)                         \
    static const bool s_attrs_##algoName = [] {                   \
        ::ops_hccl::AlgAttrs a;                                   \
        auto& topo = a.topo;                                      \
        auto& op = a.op;                                          \
        (void)op;                                                 \
        ::ops_hccl::AlgAttrsRegistry::ParseAlgName(#algoName, a); \
        __VA_ARGS__;                                              \
        ::ops_hccl::AlgAttrsRegistry::Instance().Register(a);     \
        return true;                                              \
    }()

#endif // BSR_HOST_TEST_ALG_ATTRS_REGISTRY_H
