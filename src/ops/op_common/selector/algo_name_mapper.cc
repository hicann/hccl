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

#include <algorithm>
#include <cctype>
#include <vector>

#include "alg_parse.h"
#include "log.h"

namespace ops_hccl {

/* ===== 单例 ===== */
AlgoNameMapper* AlgoNameMapper::Global()
{
    static AlgoNameMapper* instance = new AlgoNameMapper;
    return instance;
}

/* ===== 小写化辅助 ===== */
static std::string ToLowerStr(const std::string& s)
{
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return r;
}

/* ===== executor 前缀匹配 + 剩余即 template 拼接串（仅 init 时调用）===== */
bool AlgoNameMapper::LookupByPrefix(
    const std::string& algName, const std::string& opTypePascal, const char*& engineUser, const char*& executorUser,
    std::string& templateStr) const
{
    /* 1. 定位 optype，拆出 engine 和 execTpl */
    size_t pos = algName.find(opTypePascal);
    if (pos == std::string::npos) {
        return false;
    }

    /* 2. engine = optype 前面，查 alg_parse 引擎表 */
    std::string enginePascal = algName.substr(0, pos);
    engineUser = nullptr;
    int engineCount = 0;
    const AlgoDimEntry* engines = GetAlgoEngines(engineCount);
    for (int i = 0; i < engineCount; i++) {
        if (enginePascal == engines[i].pascal) {
            engineUser = engines[i].key;
            break;
        }
    }
    if (engineUser == nullptr) {
        return false;
    }

    /* 3. execTpl = optype 后面，按 executor pascal 长度降序做前缀匹配 */
    std::string execTpl = algName.substr(pos + opTypePascal.size());

    /* 收集 executor 并按 pascal 长度降序排序，避免短前缀误匹配（如 Concur 匹配 Concurrent） */
    int execCount = 0;
    const AlgoDimEntry* execs = GetAlgoExecutors(execCount);
    std::vector<AlgoDimEntry> sortedExecs(execs, execs + execCount);
    std::sort(sortedExecs.begin(), sortedExecs.end(), [](const AlgoDimEntry& a, const AlgoDimEntry& b) {
        return std::string(a.pascal).size() > std::string(b.pascal).size();
    });

    for (int i = 0; i < execCount; i++) {
        const std::string pascal = sortedExecs[i].pascal;
        if (execTpl.size() >= pascal.size() && execTpl.compare(0, pascal.size(), pascal) == 0) {
            executorUser = sortedExecs[i].key;
            /* 4. 剩余串小写化即为 template 拼接串（不切分，支持任意级数） */
            templateStr = ToLowerStr(execTpl.substr(pascal.size()));
            return true;
        }
    }
    return false;
}

/* ===== init：缓存所有算法（幂等，call_once，per-process 一次）=====
 * 前提（已核实）：allAlgos 为 GetAllAlgos() 的进程级全局
 * 单例（cost_model.cc:38-42 static AllAlgos，算法注册经 AddAlgToAllAlgos 全局追加，
 * 非 per-comm 裁剪快照），且算法注册全部完成于首次 InitModel 之前——call_once 冻结的
 * 首次快照即全量。晚注册算法三重可见化（Init 重入 count 增长 WARN + 本文件 lookup
 * failed + cost_model.cc 物化 miss WARNING），3D 名为空仅影响该算法的 tuner 匹配不影响核心选择。
 * 不采纳"call_once 外补插"方案：需加锁破坏 call_once 免锁并发，且破坏悬垂修复
 * "cache_ 唯一写点"的安全论证链。 */
void AlgoNameMapper::Init(const AllAlgos& allAlgos)
{
    std::call_once(initFlag_, [this, &allAlgos]() {
        for (int i = 0; i < allAlgos.count; ++i) {
            const char* algName = allAlgos.algElements[i].algName;
            if (algName == nullptr) {
                /* 空名条目：与 Lookup 的 nullptr 分支对齐——
                 * 不判空则 cache_[nullptr] 由空指针构造 std::string（strlen(nullptr) 崩溃） */
                HCCL_WARNING("[AlgoNameMapper] null algName at index=%d, skip.", i);
                continue;
            }
            HcclCMDType opType = allAlgos.algElements[i].opType;
            const char* opTypePascal = HcclOpTypeToPascal(opType);
            if (opTypePascal == nullptr) {
                HCCL_WARNING(
                    "[AlgoNameMapper] unknown opType=%d, skip algName=%s.",
                    static_cast<int>(allAlgos.algElements[i].opType), algName);
                continue;
            }

            const char* engineUser = "";
            const char* executorUser = "";
            std::string templateStr;
            if (LookupByPrefix(algName, opTypePascal, engineUser, executorUser, templateStr)) {
                /* 拼接串由 cache_ 节点持有；节点经 call_once 冻结后地址与内容不再变化 */
                cache_[algName] = CacheEntry{engineUser, executorUser, std::move(templateStr)};
            } else {
                HCCL_WARNING("[AlgoNameMapper] lookup failed, algName=%s.", algName);
            }
        }
        HCCL_DEBUG("[AlgoNameMapper] init done, cached %zu algorithms.", cache_.size());
        frozenCount_ = allAlgos.count;
    });
    /* 前提被打破的可见化：冻结后又有算法注册（count 增长）——其 3D 名永久为空、依赖
     * 3D 名的 tuner 规则永不命中。call_once 的完成同步保证此处读 frozenCount_ 安全。 */
    if (allAlgos.count > frozenCount_) {
        HCCL_WARNING(
            "[AlgoNameMapper] %d algo(s) registered after cache freeze (frozen=%d, now=%d), "
            "their 3D names stay empty and tuner rules on them never match.",
            allAlgos.count - frozenCount_, frozenCount_, allAlgos.count);
    }
}

/* ===== lookup：查 3D 名（miss 输出空串，与旧 Enrich 未命中行为一致）===== */
bool AlgoNameMapper::Lookup(const char* algName, AlgoDims& dims) const
{
    if (algName == nullptr) {
        dims.engineUser = "";
        dims.executorUser = "";
        dims.templateUser = "";
        return false;
    }
    auto it = cache_.find(algName);
    if (it != cache_.end()) {
        dims.engineUser = it->second.engineUser;
        dims.executorUser = it->second.executorUser;
        /* 借出节点内字符串指针：cache_ 冻结后无写入/擦除，进程生命周期内稳定 */
        dims.templateUser = it->second.templateStr.c_str();
        return true;
    }
    dims.engineUser = "";
    dims.executorUser = "";
    dims.templateUser = "";
    return false;
}

} /* namespace ops_hccl */
