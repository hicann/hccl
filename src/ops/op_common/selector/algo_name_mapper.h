/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALGO_NAME_MAPPER_H
#define ALGO_NAME_MAPPER_H

#include <string>
#include <unordered_map>

#include "hccl_algo_dims.h"
#include "hccl_tuner_plugin.h"
#include "cost_model.h"

namespace ops_hccl {

/* 查询结果：算法的 3D 用户名。
 * templateUser 为拼接串：单级="mesh"，多级="meshconcurnhrnhr"（executor 后剩余串小写化）。
 * std::string 随 cache_ 持久化，c_str() 返回的指针在 cache_ 生命周期内稳定。 */
struct AlgoDims {
    const char* engineUser;   /* "aicpu" */
    const char* executorUser; /* "sole" */
    std::string templateUser; /* "meshoneshot" 或 "meshconcurnhrnhr" */
};

class AlgoNameMapper {
public:
    static AlgoNameMapper* Global();

    /* init：缓存所有算法（HCCL 启动时调用一次） */
    void Init(const AllAlgos& allAlgos);

    /* enrich：填 3D 名到 entry 数组（每次 op，CostTableGen 之后调用） */
    void Enrich(hcclTunerAlgoEntry_t* entries, int count);

private:
    AlgoNameMapper() = default;

    /* 算法缓存（init 时填充，Enrich 直接读） */
    std::unordered_map<std::string, AlgoDims> cache_;

    /* executor 前缀匹配 + 剩余即 template 拼接串（init 时调用，替代旧 Lookup2D） */
    bool LookupByPrefix(const std::string& algName, const std::string& opTypePascal, AlgoDims& dims) const;
};

} /* namespace ops_hccl */

#endif /* ALGO_NAME_MAPPER_H */
