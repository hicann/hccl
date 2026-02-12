/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "execute_selector.h"
#include "auto_selector_base.h"
#include "selector_registry.h"

namespace ops_hccl {

ExecuteSelector::ExecuteSelector()
{
}

HcclResult ExecuteSelector::Run(OpParam &opParam, TopoInfo* topoInfo,
                                std::string &selectAlgName, OpExecuteConfig &opExecuteConfig)
{
    HCCL_DEBUG("[Algo][Selector] Run.");
    std::map<u32, AutoSelectorBase *> selectors = SelectorRegistry::Global()->GetAllSelectors();

    if (opParam.isMc2) {
        auto iter = selectors.find(18);
        if (iter == selectors.end()) {
            HCCL_ERROR("[Algo][Selector] CCU selector is not registried.");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        if(iter->second->Select(opParam, topoInfo, selectAlgName, opExecuteConfig) == SelectorStatus::MATCH) {
            HCCL_INFO("[Algo][Selector] The ccu selector[priority of %u] is matched, the selected algo type is %s",
                iter->first, selectAlgName.c_str());
            return HcclResult::HCCL_SUCCESS;
        }
        HCCL_ERROR("[Algo][Selector] CCU selector can not match for optype[%d].", opParam.opType);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    selectors = SelectorRegistry::Global()->GetSelectorsByOpType(opParam.opType);
    HCCL_INFO("[Algo][Selector] The selector nums of optype[%d] is [%zu].", opParam.opType, selectors.size());
    for (auto iter : selectors) {
        HCCL_DEBUG("[Algo][Selector] The selector[priority of %llu] is running.", iter.first);
        if (iter.second->Select(opParam, topoInfo, selectAlgName, opExecuteConfig) == SelectorStatus::MATCH) {
            HCCL_INFO("[Algo][Selector] The selector[priority of %llu] is matched, the selected algo type is %s",
                      iter.first, selectAlgName.c_str());
            return HcclResult::HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[Algo][Selector] No selector is matched.");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

} // namespace Hccl
