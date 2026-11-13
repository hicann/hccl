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

#include <mutex>
#include <string>
#include <unordered_map>

#include "hccl_algo_dims.h"
#include "cost_model.h"

namespace ops_hccl {

/* 查询结果：算法的 3D 用户名。
 * templateUser 为拼接串：单级="mesh"，多级="meshconcurnhrnhr"（executor 后剩余串小写化）。
 * 三个字段均为借出指针：engineUser/executorUser 指向静态维度表，templateUser 指向 cache_ 节点持有的
 * 字符串。cache_ 经 call_once 冻结后无写入/擦除，节点内字符串在进程生命周期内稳定，指针可长期保存；
 * 调用方不得释放/修改，指针跨进程无意义（勿序列化后解引用）。 */
struct AlgoDims {
    const char* engineUser;   /* "aicpu" */
    const char* executorUser; /* "sole" */
    const char* templateUser; /* "meshoneshot" 或 "meshconcurnhrnhr" */
};

class AlgoNameMapper {
public:
    static AlgoNameMapper* Global();

    /* init：缓存所有算法（幂等，内部 call_once，任意调用方先于 Lookup 执行即可） */
    void Init(const AllAlgos& allAlgos);

    /* lookup：按 algName 查 3D 名（miss 时输出空串，与旧 Enrich 未命中行为一致），返回是否命中。
     * 供 costModel 生成时物化进条目（per-comm 一次），替代旧每 op 的 Enrich 热路径查找 */
    bool Lookup(const char* algName, AlgoDims& dims) const;

private:
    AlgoNameMapper() = default;

    /* cache_ 节点条目：templateStr 为节点持有的拼接串（多级算法如 "meshconcurnhrnhr"） */
    struct CacheEntry {
        const char* engineUser;
        const char* executorUser;
        std::string templateStr;
    };

    /* 算法缓存（init 时填充，Lookup 直接读；call_once 冻结后仅读、无 erase） */
    std::unordered_map<std::string, CacheEntry> cache_;
    std::once_flag initFlag_;
    int frozenCount_ = 0; /* 冻结时的算法总数：Init 重入对比 count 增长，晚注册可见化 */

    /* executor 前缀匹配 + 剩余即 template 拼接串（init 时调用，替代旧 Lookup2D）。
     * 拼接串经出参交由 cache_ 节点持有——构造期局部串无持久持有者，不入 AlgoDims */
    bool LookupByPrefix(
        const std::string& algName, const std::string& opTypePascal, const char*& engineUser, const char*& executorUser,
        std::string& templateStr) const;
};

} /* namespace ops_hccl */

#endif /* ALGO_NAME_MAPPER_H */
