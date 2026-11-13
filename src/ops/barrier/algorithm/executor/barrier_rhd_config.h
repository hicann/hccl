/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the license.
 */

#ifndef BARRIER_RHD_CONFIG_H
#define BARRIER_RHD_CONFIG_H

#include <algorithm>
#include <string>
#include <vector>
#include "alg_parse.h"

namespace ops_hccl {

inline bool IsBarrierRhdExplicitlyConfigured(
    const HcclAlgoParser& parser, const std::string& algName, const std::vector<const char*>& barrierNames,
    const std::vector<std::string>& enginePrefixes)
{
    std::vector<CostAlgoParams> entries;
    for (const auto* name : barrierNames) {
        CostAlgoParams entry{};
        entry.algName = name;
        entry.count = 1;
        entries.push_back(entry);
    }
    CostModel model{entries.data(), static_cast<int>(entries.size())};
    // 按公共解析器的优先级，从后往前找到第一条命中 Barrier 候选的子句。
    HcclAlgoParser clause;
    for (auto it = parser.executorList.rbegin(); it != parser.executorList.rend(); ++it) {
        if (!it->opType.empty() && it->opType != "barrier") {
            continue;
        }
        clause.executorList.assign(1, *it);
        if (UpdateCostModelWithAlgo(clause, model, enginePrefixes) != HCCL_SUCCESS) {
            return false;
        }
        if (std::none_of(entries.begin(), entries.end(), [](const auto& entry) {
                return entry.hcclAlgoPriority != 0;
            })) {
            continue;
        }
        // 前缀和 not() 匹配也可能打正向优先级，RHD 只接受完整的正向匹配。
        return it->enable && !it->algoList.empty()
               && std::all_of(
                   it->algoList.begin(), it->algoList.end(),
                   [](const auto& algo) {
                       return algo.enable;
                   })
               && std::any_of(entries.begin(), entries.end(), [&algName](const auto& entry) {
                      return entry.hcclAlgoPriority > 0 && algName == entry.algName;
                  });
    }
    return false;
}

} // namespace ops_hccl

#endif // BARRIER_RHD_CONFIG_H
