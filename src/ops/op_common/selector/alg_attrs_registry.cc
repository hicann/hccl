/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_attrs_registry.h"
#include "log.h"
#include "alg_parse.h"
#include <algorithm>

namespace ops_hccl {

std::string AlgAttrsRegistry::GetAlgAttrsSummary(const AlgAttrs& a)
{
    if (a.algoTypes.empty()) {
        return "[]";
    }
    std::string s = "[";
    for (size_t i = 0; i < a.algoTypes.size(); ++i) {
        if (i > 0) {
            s += ",";
        }
        s += AlgoTypeToString(a.algoTypes[i]);
    }
    s += "]";
    return s;
}

void AlgAttrsRegistry::ParseAlgName(const std::string& algoName, AlgAttrs& a)
{
    a.name = algoName;

    // 1. Parse engine — reuse GetEnginePrefixEntries from alg_parse.h
    int engineCount = 0;
    const EnginePrefixEntry* engines = GetEnginePrefixEntries(engineCount);
    a.engine = OpExecuteConfig::AICPU_TS;
    for (int i = 0; i < engineCount; ++i) {
        size_t len = strlen(engines[i].pascal);
        if (algoName.size() > len && algoName.substr(0, len) == engines[i].pascal) {
            a.engine = engines[i].engine;
            break;
        }
    }

    // 2. Parse opType — reuse GetOpTypePatternEntries from alg_parse.h
    int opTypeCount = 0;
    const OpTypePatternEntry* opTypes = GetOpTypePatternEntries(opTypeCount);
    a.opType = HcclCMDType::HCCL_CMD_INVALID;
    for (int i = 0; i < opTypeCount; ++i) {
        if (algoName.find(opTypes[i].pascal) != std::string::npos) {
            a.opType = opTypes[i].opType;
            break;
        }
    }

    // 3. Parse algoTypes — reuse GetAlgoNameToTypeMap from alg_parse.h
    const auto& nameToEnum = GetAlgoNameToTypeMap();
    std::vector<std::string> sortedAlgoNames;
    for (const auto& kv : nameToEnum) {
        sortedAlgoNames.push_back(kv.first);
    }
    std::sort(sortedAlgoNames.begin(), sortedAlgoNames.end(), [](const std::string& x, const std::string& y) {
        return x.size() > y.size();
    });

    // Remove engine prefix
    std::string remaining = algoName;
    for (int i = 0; i < engineCount; ++i) {
        size_t len = strlen(engines[i].pascal);
        if (remaining.size() >= len && remaining.substr(0, len) == engines[i].pascal) {
            remaining = remaining.substr(len);
            break;
        }
    }

    // Remove opType (only at position 0)
    for (int i = 0; i < opTypeCount; ++i) {
        size_t len = strlen(opTypes[i].pascal);
        if (remaining.size() >= len && remaining.substr(0, len) == opTypes[i].pascal) {
            remaining = remaining.substr(len);
            break;
        }
    }

    // Remove executorType (only at position 0)
    int execCount = 0;
    const AlgoDimEntry* execs = GetAlgoExecutors(execCount);
    for (int pass = 0; pass < 2; ++pass) {
        bool found = false;
        for (int i = 0; i < execCount; ++i) {
            size_t len = strlen(execs[i].pascal);
            if (remaining.size() >= len && remaining.substr(0, len) == execs[i].pascal) {
                remaining = remaining.substr(len);
                found = true;
                break;
            }
        }
        if (!found) {
            break;
        }
    }

    // Greedy match algo types
    a.algoTypes.clear();
    size_t pos = 0;
    while (pos < remaining.size()) {
        bool matched = false;
        for (const auto& name : sortedAlgoNames) {
            if (remaining.compare(pos, name.size(), name) == 0) {
                auto it = nameToEnum.find(name);
                if (it != nameToEnum.end()) {
                    a.algoTypes.push_back(it->second);
                }
                pos += name.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            // 算法名解析失败：可能是新算法名格式不兼容，先跳过不阻塞功能，后续需整改算法名
            HCCL_WARNING(
                "[AlgAttrsRegistry] ParseAlgName: skipped unmatched algoName=%s at pos=%zu, "
                "remaining='%s'. Please check algorithm naming convention.",
                algoName.c_str(), pos, remaining.c_str() + pos);
            a.algoTypes.clear();
            return;
        }
    }
}

AlgAttrsRegistry& AlgAttrsRegistry::Instance()
{
    static AlgAttrsRegistry instance;
    return instance;
}

void AlgAttrsRegistry::Register(const AlgAttrs& attrs)
{
    if (nameToIndex_.find(attrs.name) != nameToIndex_.end()) {
        HCCL_WARNING("[AlgAttrsRegistry] duplicate attrs name=%s, will overwrite.", attrs.name.c_str());
        attrs_[nameToIndex_[attrs.name]] = attrs;
        return;
    }
    nameToIndex_[attrs.name] = attrs_.size();
    attrs_.push_back(attrs);
    HCCL_DEBUG("[AlgAttrsRegistry] registered attrs name=%s, total=%zu.", attrs.name.c_str(), attrs_.size());
}

const AlgAttrs* AlgAttrsRegistry::Get(const std::string& name) const
{
    auto it = nameToIndex_.find(name);
    if (it == nameToIndex_.end()) {
        return nullptr;
    }
    return &attrs_[it->second];
}

const std::vector<AlgAttrs>& AlgAttrsRegistry::GetAll() const { return attrs_; }

} // namespace ops_hccl
