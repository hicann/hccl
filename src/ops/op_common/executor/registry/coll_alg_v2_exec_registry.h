/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_V2_EXEC_REGISTRY_H
#define COLL_ALG_V2_EXEC_REGISTRY_H

#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include "executor_v2_base.h"
#include "cost_model.h"

namespace ops_hccl {

using CollExecCreatorV2 = std::function<InsCollAlgBase*()>;

template <typename P>
static InsCollAlgBase* DefaultExecCreatorV2()
{
    static_assert(
        std::is_base_of<InsCollAlgBase, P>::value, "Executor type must derived from Hccl::DefaultExecCreatorV2");
    return new (std::nothrow) P();
}

class CollAlgExecRegistryV2 {
public:
    static CollAlgExecRegistryV2& Instance();
    HcclResult Register(const HcclCMDType type, const std::string& tag, const CollExecCreatorV2& collExecCreator);
    std::unique_ptr<InsCollAlgBase> GetAlgExec(const HcclCMDType type, const std::string& tag);

private:
    std::map<HcclCMDType, std::map<std::string, const CollExecCreatorV2>> execCreators_;
    mutable std::mutex mu_;
};

#define ALG_GET_COUNT_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define ALG_GET_COUNT(...) ALG_GET_COUNT_IMPL(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1)

#define ALG_TEMPLATE_STRINGIFY_1(x) #x
#define ALG_TEMPLATE_STRINGIFY_2(x, ...) #x, ALG_TEMPLATE_STRINGIFY_1(__VA_ARGS__)
#define ALG_TEMPLATE_STRINGIFY_3(x, ...) #x, ALG_TEMPLATE_STRINGIFY_2(__VA_ARGS__)
#define ALG_TEMPLATE_STRINGIFY_4(x, ...) #x, ALG_TEMPLATE_STRINGIFY_3(__VA_ARGS__)
#define ALG_TEMPLATE_STRINGIFY_5(x, ...) #x, ALG_TEMPLATE_STRINGIFY_4(__VA_ARGS__)
#define ALG_TEMPLATE_STRINGIFY_6(x, ...) #x, ALG_TEMPLATE_STRINGIFY_5(__VA_ARGS__)
#define ALG_TEMPLATE_STRINGIFY_7(x, ...) #x, ALG_TEMPLATE_STRINGIFY_6(__VA_ARGS__)
#define ALG_TEMPLATE_STRINGIFY_8(x, ...) #x, ALG_TEMPLATE_STRINGIFY_7(__VA_ARGS__)

#define ALG_TEMPLATE_STRINGIFY_DISPATCH(N, ...) ALG_TEMPLATE_STRINGIFY_##N(__VA_ARGS__)
#define ALG_TEMPLATE_STRINGIFY_EXPAND(N, ...) ALG_TEMPLATE_STRINGIFY_DISPATCH(N, __VA_ARGS__)
#define ALG_STRINGIFY_VA(...) ALG_TEMPLATE_STRINGIFY_EXPAND(ALG_GET_COUNT(__VA_ARGS__), __VA_ARGS__)

#define REGISTER_EXECUTOR_IMPL_HELPER(ctr, type, name, insCollAlgBase)                                                \
    static HcclResult g_func_##name##_##ctr                                                                           \
        = CollAlgExecRegistryV2::Instance().Register(type, std::string(#name), DefaultExecCreatorV2<insCollAlgBase>); \
    static HcclResult g_alg_##name##_##ctr = AddAlgToAllAlgos(type, #name, #insCollAlgBase, nullptr, 0)

#define REGISTER_EXECUTOR_IMPL_HELPER_1(ctr, type, name, insCollAlgBase) \
    REGISTER_EXECUTOR_IMPL_HELPER(ctr, type, name, insCollAlgBase)

#define REGISTER_EXECUTOR_IMPL(type, name, insCollAlgBase) \
    REGISTER_EXECUTOR_IMPL_HELPER_1(__COUNTER__, type, name, insCollAlgBase)

#define REGISTER_EXECUTOR_IMPL_HELPER_NO_TOPOMATCH(ctr, type, name, insCollAlgBase, InsAlgTemplate) \
    static const char* g_alg_templates_##name##_##ctr[] = {#InsAlgTemplate};                        \
    static HcclResult g_func_##name##_##ctr = CollAlgExecRegistryV2::Instance().Register(           \
        type, std::string(#name), DefaultExecCreatorV2<insCollAlgBase<InsAlgTemplate>>);            \
    static HcclResult g_alg_##name##_##ctr                                                          \
        = AddAlgToAllAlgos(type, #name, #insCollAlgBase, g_alg_templates_##name##_##ctr, 1)

#define REGISTER_EXECUTOR_IMPL_HELPER_NO_TOPOMATCH_1(ctr, type, name, insCollAlgBase, InsAlgTemplate) \
    REGISTER_EXECUTOR_IMPL_HELPER_NO_TOPOMATCH(ctr, type, name, insCollAlgBase, InsAlgTemplate)

#define REGISTER_EXECUTOR_IMPL_NO_TOPOMATCH(type, name, insCollAlgBase, InsAlgTemplate) \
    REGISTER_EXECUTOR_IMPL_HELPER_NO_TOPOMATCH_1(__COUNTER__, type, name, insCollAlgBase, InsAlgTemplate)

#define REGISTER_EXECUTOR_HELPER(ctr, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate)        \
    static const char* g_alg_templates_##name##_##ctr[] = {#InsAlgTemplate};                           \
    static HcclResult g_func_##name##_##ctr = CollAlgExecRegistryV2::Instance().Register(              \
        type, std::string(#name), DefaultExecCreatorV2<insCollAlgBase<AlgTopoMatch, InsAlgTemplate>>); \
    static HcclResult g_alg_##name##_##ctr                                                             \
        = AddAlgToAllAlgos(type, #name, #insCollAlgBase, g_alg_templates_##name##_##ctr, 1)

#define REGISTER_EXECUTOR_HELPER_1(ctr, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate) \
    REGISTER_EXECUTOR_HELPER(ctr, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate)

#define REGISTER_EXEC_V2(type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate) \
    REGISTER_EXECUTOR_HELPER_1(__COUNTER__, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate)

#define REGISTER_EXECUTOR_BY_TOPO_HELPER(ctr, type, name, insCollAlgBase, AlgTopoMatch)   \
    static HcclResult g_func_##name##_##ctr = CollAlgExecRegistryV2::Instance().Register( \
        type, std::string(#name), DefaultExecCreatorV2<insCollAlgBase<AlgTopoMatch>>);    \
    static HcclResult g_alg_##name##_##ctr = AddAlgToAllAlgos(type, #name, #insCollAlgBase, nullptr, 0)

#define REGISTER_EXECUTOR_BY_TOPO_HELPER_1(ctr, type, name, insCollAlgBase, AlgTopoMatch) \
    REGISTER_EXECUTOR_BY_TOPO_HELPER(ctr, type, name, insCollAlgBase, AlgTopoMatch)

#define REGISTER_EXECUTOR_BY_TOPO(type, name, insCollAlgBase, AlgTopoMatch) \
    REGISTER_EXECUTOR_BY_TOPO_HELPER_1(__COUNTER__, type, name, insCollAlgBase, AlgTopoMatch)

#define REGISTER_EXECUTOR_BY_TWO_TEMPS_HELPER(                                                  \
    ctr, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1)            \
    static const char* g_alg_templates_##name##_##ctr[] = {#InsAlgTemplate0, #InsAlgTemplate1}; \
    static HcclResult g_func_##name##_##ctr = CollAlgExecRegistryV2::Instance().Register(       \
        type, std::string(#name),                                                               \
        DefaultExecCreatorV2<insCollAlgBase<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>>);  \
    static HcclResult g_alg_##name##_##ctr                                                      \
        = AddAlgToAllAlgos(type, #name, #insCollAlgBase, g_alg_templates_##name##_##ctr, 2)

#define REGISTER_EXECUTOR_BY_TWO_TEMPS_HELPER_1(                                     \
    ctr, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1) \
    REGISTER_EXECUTOR_BY_TWO_TEMPS_HELPER(                                           \
        ctr, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1)

#define REGISTER_EXECUTOR_BY_TWO_TEMPS(type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1) \
    REGISTER_EXECUTOR_BY_TWO_TEMPS_HELPER_1(                                                                       \
        __COUNTER__, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1)

#define REGISTER_EXECUTOR_BY_FOUR_TEMPS_HELPER(                                                                        \
    ctr, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3) \
    static const char* g_alg_templates_##name##_##ctr[]                                                                \
        = {#InsAlgTemplate0, #InsAlgTemplate1, #InsAlgTemplate2, #InsAlgTemplate3};                                    \
    static HcclResult g_func_##name##_##ctr = CollAlgExecRegistryV2::Instance().Register(                              \
        type, std::string(#name),                                                                                      \
        DefaultExecCreatorV2<                                                                                          \
            insCollAlgBase<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>>);        \
    static HcclResult g_alg_##name##_##ctr                                                                             \
        = AddAlgToAllAlgos(type, #name, #insCollAlgBase, g_alg_templates_##name##_##ctr, 4)

#define REGISTER_EXECUTOR_BY_FOUR_TEMPS_HELPER_1(                                                                      \
    ctr, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3) \
    REGISTER_EXECUTOR_BY_FOUR_TEMPS_HELPER(                                                                            \
        ctr, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2,              \
        InsAlgTemplate3)

#define REGISTER_EXECUTOR_BY_FOUR_TEMPS(                                                                          \
    type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3) \
    REGISTER_EXECUTOR_BY_FOUR_TEMPS_HELPER_1(                                                                     \
        __COUNTER__, type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, \
        InsAlgTemplate3)

// 通过 __VA_ARGS__ 展开
// 防御：__VA_ARGS__ 为空时编译报错（ALG_GET_COUNT 空 参数返回 1，触发 static_assert）
#define ALG_CHECK_NONEMPTY_VA_1 static_assert(false, "REGISTER_EXEC_V2_MULTI requires at least 1 template arg");
#define ALG_CHECK_NONEMPTY_VA_2
#define ALG_CHECK_NONEMPTY_VA_3
#define ALG_CHECK_NONEMPTY_VA_4
#define ALG_CHECK_NONEMPTY_VA_5
#define ALG_CHECK_NONEMPTY_VA_6
#define ALG_CHECK_NONEMPTY_VA_7
#define ALG_CHECK_NONEMPTY_VA_8
#define ALG_CHECK_NONEMPTY_VA_DISPATCH(count) ALG_CHECK_NONEMPTY_VA_##count
#define ALG_CHECK_NONEMPTY_VA_EXPAND(count) ALG_CHECK_NONEMPTY_VA_DISPATCH(count)

#define REGISTER_EXECUTOR_IMPL_MULTI(ctr, type, name, insCollAlgBase, AlgTopoMatch, ...)            \
    ALG_CHECK_NONEMPTY_VA_EXPAND(ALG_GET_COUNT(__VA_ARGS__))                                        \
    static const char* g_alg_templates_##name##_##ctr[] = {ALG_STRINGIFY_VA(__VA_ARGS__)};          \
    static HcclResult g_func_##name##_##ctr = CollAlgExecRegistryV2::Instance().Register(           \
        type, std::string(#name), DefaultExecCreatorV2<insCollAlgBase<AlgTopoMatch, __VA_ARGS__>>); \
    static HcclResult g_alg_##name##_##ctr                                                          \
        = AddAlgToAllAlgos(type, #name, #insCollAlgBase, g_alg_templates_##name##_##ctr, ALG_GET_COUNT(__VA_ARGS__))

#define REGISTER_EXECUTOR_HELPER_MULTI(ctr, type, name, insCollAlgBase, AlgTopoMatch, ...) \
    REGISTER_EXECUTOR_IMPL_MULTI(ctr, type, name, insCollAlgBase, AlgTopoMatch, __VA_ARGS__)

// 支持任意数量的后续参数
#define REGISTER_EXEC_V2_MULTI(type, name, insCollAlgBase, AlgTopoMatch, ...) \
    REGISTER_EXECUTOR_HELPER_MULTI(__COUNTER__, type, name, insCollAlgBase, AlgTopoMatch, __VA_ARGS__)
} // namespace ops_hccl
#endif
