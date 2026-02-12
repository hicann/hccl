/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_REGISTRY
#define HCCLV2_COLL_ALG_SELECTOR_REGISTRY

#include <mutex>
#include <map>

#include "auto_selector_base.h"

namespace ops_hccl {

class SelectorRegistry {
public:
    static SelectorRegistry      *Global();
    HcclResult                    Register(u32 priority, AutoSelectorBase *selector);
    HcclResult RegisterByOpType(const HcclCMDType opType, u32 priority, AutoSelectorBase *selector);
    std::map<u32, AutoSelectorBase *> GetAllSelectors();
    std::map<u32, AutoSelectorBase *> GetSelectorsByOpType(const HcclCMDType opType);

private:
    std::map<u32, AutoSelectorBase *> impls_;
    std::map<HcclCMDType, std::map<u32, AutoSelectorBase *>> opTypeImpls_;
    mutable std::mutex            mu_;
};

#define REGISTER_SELECTOR_HELPER(ctr, priority, name, selector)                                                        \
    static HcclResult g_func_##priority##_##name##_##ctr                                                               \
        = SelectorRegistry::Global()->Register(priority, new selector())

#define REGISTER_SELECTOR_HELPER_1(ctr, priority, name, selector)                                                      \
    REGISTER_SELECTOR_HELPER(ctr, priority, name, selector)

#define REGISTER_SELECTOR(priority, selector) REGISTER_SELECTOR_HELPER_1(__COUNTER__, priority, selector, selector)
}

#define REGISTER_SELECTOR_BY_OPTYPE_HELPER(ctr, optype, priority, name, selector)    \
    static HcclResult g_func_##priority##_##name##_##ctr                                                               \
        = SelectorRegistry::Global()->RegisterByOpType(optype, priority, new selector())

#define REGISTER_SELECTOR_BY_OPTYPE_HELPER_1(ctr, optype, priority, name, selector)  \
    REGISTER_SELECTOR_BY_OPTYPE_HELPER(ctr, optype, priority, name, selector)

#define REGISTER_SELECTOR_BY_OPTYPE(optype, priority, selector)              \
    REGISTER_SELECTOR_BY_OPTYPE_HELPER_1(__COUNTER__, optype, priority, selector, selector)// namespace ops_hccl

#endif
