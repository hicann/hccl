/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "algo_name_mapper.h"

#include "alg_parse.h"
#include "log.h"
#include "tuner_setup.h"

namespace ops_hccl {

/* ===== 单例 ===== */
AlgoNameMapper* AlgoNameMapper::Global()
{
    static AlgoNameMapper* instance = new AlgoNameMapper;
    return instance;
}

/* ===== 构建 2D 表 ===== */
void AlgoNameMapper::BuildMap2D()
{
    int execCount = 0;
    int tplCount = 0;
    const AlgoDimEntry* execs = GetAlgoExecutors(execCount);
    const AlgoDimEntry* tpls = GetAlgoTemplates(tplCount);
    for (int ex = 0; ex < execCount; ex++) {
        for (int t = 0; t < tplCount; t++) {
            std::string key = std::string(execs[ex].pascal) + tpls[t].pascal;
            map2D_[key] = {execs[ex].key, tpls[t].key};
        }
    }
    HCCL_DEBUG("[AlgoNameMapper] 2D map built, %zu entries.", map2D_.size());
}

/* ===== 2D 查表（仅 init 时调用）===== */
bool AlgoNameMapper::Lookup2D(const std::string& algName, const std::string& opTypePascal, AlgoDims& dims) const
{
    /* 1. 定位 optype，拆出 engine 和 execTpl */
    size_t pos = algName.find(opTypePascal);
    if (pos == std::string::npos) {
        return false;
    }

    /* 2. engine = optype 前面，查 alg_parse 引擎表 */
    std::string enginePascal = algName.substr(0, pos);
    dims.engineUser = nullptr;
    int engineCount = 0;
    const AlgoDimEntry* engines = GetAlgoEngines(engineCount);
    for (int i = 0; i < engineCount; i++) {
        if (enginePascal == engines[i].pascal) {
            dims.engineUser = engines[i].key;
            break;
        }
    }
    if (dims.engineUser == nullptr) {
        return false;
    }

    /* 3. execTpl = optype 后面，查 2D 表 */
    std::string execTpl = algName.substr(pos + opTypePascal.size());
    auto it = map2D_.find(execTpl);
    if (it == map2D_.end()) {
        return false;
    }
    dims.executorUser = it->second.first;
    dims.templateUser = it->second.second;
    return true;
}

/* ===== init：建表 + 缓存所有算法（一次性）===== */
void AlgoNameMapper::Init(const AllAlgos& allAlgos)
{
    BuildMap2D(); /* 30 条，<0.1ms */

    for (int i = 0; i < allAlgos.count; ++i) {
        const char* algName = allAlgos.algElements[i].algName;
        HcclCMDType opType = allAlgos.algElements[i].opType;
        const char* opTypePascal = HcclOpTypeToPascal(opType);
        if (opTypePascal == nullptr) {
            HCCL_WARNING(
                "[AlgoNameMapper] unknown opType=%d, skip algName=%s.",
                static_cast<int>(allAlgos.algElements[i].opType), algName);
            continue;
        }

        AlgoDims dims = {};
        if (Lookup2D(algName, opTypePascal, dims)) {
            cache_[algName] = dims;
        } else {
            HCCL_WARNING("[AlgoNameMapper] lookup failed, algName=%s.", algName);
        }
    }
    HCCL_DEBUG("[AlgoNameMapper] init done, cached %zu algorithms.", cache_.size());
}

/* ===== enrich：填 3D 名到 entry 数组（每次 op）===== */
void AlgoNameMapper::Enrich(hcclTunerAlgoEntry_t* entries, int count)
{
    if (entries == nullptr) {
        return;
    }
    for (int i = 0; i < count; i++) {
        auto it = cache_.find(entries[i].algName);
        if (it != cache_.end()) {
            entries[i].engineName = it->second.engineUser;
            entries[i].executorName = it->second.executorUser;
            entries[i].templateName = it->second.templateUser;
        } else {
            entries[i].engineName = "";
            entries[i].executorName = "";
            entries[i].templateName = "";
        }
        entries[i].structSize = sizeof(hcclTunerAlgoEntry_t);
    }
}

} /* namespace ops_hccl */
