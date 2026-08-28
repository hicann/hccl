/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <set>
#include <map>

#include <alg_parse.h>

namespace ops_hccl {

static const std::map<std::string, std::string> ENGINE_TYPES
    = {{"aicpu", "Aicpu"}, {"aiv", "Aiv"}, {"dpu", "Dpu"}, {"ccums", "CcuMS"}, {"ccusched", "CcuSched"}};
static const std::map<std::string, std::string> OP_TYPES
    = {{"allgather", "AllGather"},
       {"allgatherv", "AllGatherV"},
       {"allreduce", "AllReduce"},
       {"alltoall", "AllToAll"},
       {"alltoallv", "AllToAllV"},
       {"alltoallvc", "AllToAllVC"},
       {"broadcast", "Broadcast"},
       {"reduce", "Reduce"},
       {"reducescatter", "ReduceScatter"},
       {"reducescatterv", "ReduceScatterV"},
       {"scatter", "Scatter"}};
static const std::map<std::string, std::string> EXECUTOR_TYPES
    = {{"sole", "Sole"},         {"sequence", "Sequence"}, {"parallel", "Parallel"},
       {"pipeline", "PipeLine"}, {"concur", "Concur"},     {"strictordered", "StrictOrdered"}};
static const std::map<std::string, std::string> ALGO_TYPES
    = {{"mesh", "Mesh"},
       {"mesh2die", "Mesh2Die"},
       {"meshoneshot", "MeshOneShot"},
       {"meshtwoshot", "MeshTwoShot"},
       {"meshconcur", "MeshConcur"},
       {"meshmultilink", "MeshMultiLink"},
       {"meshchunk", "MeshChunk"},
       {"meshchunktwoshot", "MeshChunkTwoShot"},
       {"nhr", "NHR"},
       {"nhrmultilink", "NHRMultiLink"},
       {"nhraicpureduce", "NHRAicpuReduce"},
       {"nhrsinglechannel", "MeshSingleChannel"},
       {"meshconcurrent", "MeshConcurrent"}};

// 小写字符串转换（自由函数，供全局使用）
static std::string ToLowerStr(const std::string& s)
{
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return r;
}

// ===========================================================================
// 用户算法配置解析器（内部实现类）
// ===========================================================================
std::string UnderscoreToCamelCase(const std::string& name)
{
    std::string result;
    bool nextUpper = false;
    for (char c : name) {
        if (c == '_') {
            nextUpper = true;
        } else {
            if (nextUpper) {
                result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                nextUpper = false;
            } else {
                result += c;
            }
        }
    }
    return result;
}

// ===========================================================================
// 用户算法配置解析器（内部实现类）
// ===========================================================================
class AlgoParserImpl {
public:
    explicit AlgoParserImpl(const std::string& input) : input_(input), pos_(0) {}

    HcclResult Parse(std::vector<HcclAlgoExecutor>& result)
    {
        SkipWs();
        while (!AtEnd()) {
            HcclAlgoExecutor exec;
            CHK_RET(ParseSegment(exec));
            CompactAlgoList(exec.algoList);
            result.push_back(std::move(exec));
            SkipWs();
            if (Eat(';')) {
                SkipWs();
                continue;
            }
            break;
        }
        SkipWs();
        if (!AtEnd()) {
            HCCL_ERROR(
                "[HcclAlgoParser] parse algo config failed, standard format: \"opType:executorType{level0=algoType, "
                "level1=algoType, level2=algoType};...\", "
                "trailing chars at pos %zu: [%s]",
                pos_, input_.substr(pos_).c_str());
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

private:
    const std::string& input_;
    size_t pos_;

    // 哨兵值：标记 level 未显式指定
    static constexpr uint32_t LEVEL_UNSPECIFIED = UINT32_MAX;

    // ---- 辅助函数 ----

    // 有序插入：将 algo 按 level 插入到 algoList 的正确位置
    // level 由 algoList 的 index 隐式表达，插入后保持 index 与 level 对应
    void InsertAlgoOrdered(std::vector<HcclAlgo>& algoList, HcclAlgo algo, uint32_t level) const
    {
        if (level == LEVEL_UNSPECIFIED) {
            // 未指定 level，追加到末尾
            algoList.push_back(std::move(algo));
            return;
        }
        // 扩展到足够大小（中间空位用默认 HcclAlgo 填充，后续会被覆盖或保留）
        if (level >= algoList.size()) {
            algoList.resize(level + 1);
        }
        algoList[level] = std::move(algo);
    }

    // 补全 algoList：消除空位，确保 index 连续且按 level 升序
    // 空位是指 algoType 为空的默认 HcclAlgo 条目，由 resize 产生
    void CompactAlgoList(std::vector<HcclAlgo>& algoList) const
    {
        // 先将所有有效条目（algoType 非空）按 index(=level) 收集
        std::vector<std::pair<uint32_t, HcclAlgo>> validItems;
        for (uint32_t i = 0; i < static_cast<uint32_t>(algoList.size()); i++) {
            if (!algoList[i].algoType.empty()) {
                validItems.emplace_back(i, std::move(algoList[i]));
            }
        }
        // 已按 index 升序（遍历顺序），直接紧凑写入
        algoList.clear();
        for (auto& p : validItems) {
            algoList.push_back(std::move(p.second));
        }
    }

    // ---- 基础词法 ----
    char Peek() const { return pos_ < input_.size() ? input_[pos_] : '\0'; }
    char Peek2() const { return pos_ + 1 < input_.size() ? input_[pos_ + 1] : '\0'; }
    // 尝试消费字符 c：当前字符匹配则前进一步返回 true，否则返回 false
    bool Eat(char c)
    {
        if (Peek() == c) {
            pos_++;
            return true;
        }
        return false;
    }
    bool AtEnd() const { return pos_ >= input_.size(); }

    void SkipWs()
    {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            pos_++;
        }
    }

    // 解析标识符：字母/数字/下划线/连字符（规则8）
    bool ParseIdentifier(std::string& name)
    {
        SkipWs();
        size_t start = pos_;
        while (pos_ < input_.size()) {
            char c = input_[pos_];
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
                pos_++;
            } else {
                break;
            }
        }
        if (pos_ == start)
            return false;
        name = input_.substr(start, pos_ - start);
        return true;
    }

    // ---- 文法规则 ----

    // segment := [opType ':'] executor_expr
    HcclResult ParseSegment(HcclAlgoExecutor& exec)
    {
        size_t savePos = pos_;
        std::string name;
        if (ParseIdentifier(name)) {
            SkipWs();
            if (Peek() == ':' && Peek2() != ':') {
                Eat(':');
                std::string opType = ToLowerStr(name);
                if (OP_TYPES.find(opType) == OP_TYPES.end()) {
                    HCCL_ERROR("[HcclAlgoParser] invalid opType [%s] at pos %zu", opType.c_str(), pos_);
                    return HCCL_E_PARA;
                }
                exec.opType = opType;
                SkipWs();
            } else {
                pos_ = savePos;
            }
        }
        return ParseExecutorExpr(exec);
    }

    // executor_expr := 'not' '(' inner ')' | executor_unit | template_name(shorthand)
    HcclResult ParseExecutorExpr(HcclAlgoExecutor& exec)
    {
        SkipWs();
        size_t savePos = pos_;
        std::string name;
        if (ParseIdentifier(name)) {
            SkipWs();
            if (ToLowerStr(name) == "not" && Peek() == '(') {
                Eat('(');
                SkipWs();
                CHK_RET(ParseExecutorUnitOrAtom(exec));
                SkipWs();
                if (!Eat(')')) {
                    HCCL_ERROR("[HcclAlgoParser] expected ')' after not(...) at pos %zu", pos_);
                    return HCCL_E_PARA;
                }
                exec.enable = false;
                return HCCL_SUCCESS;
            }
            pos_ = savePos;
        }
        return ParseExecutorUnitOrAtom(exec);
    }

    // executor_unit | template_name(shorthand)
    HcclResult ParseExecutorUnitOrAtom(HcclAlgoExecutor& exec)
    {
        SkipWs();
        std::string name;
        if (!ParseIdentifier(name)) {
            HCCL_ERROR("[HcclAlgoParser] expected executor or template at pos %zu", pos_);
            return HCCL_E_PARA;
        }
        SkipWs();
        if (Peek() == '{') {
            std::string executorType = ToLowerStr(name);
            if (EXECUTOR_TYPES.find(executorType) == EXECUTOR_TYPES.end()) {
                HCCL_ERROR("[HcclAlgoParser] invalid executorType [%s] at pos %zu", executorType.c_str(), pos_);
                return HCCL_E_PARA;
            }
            exec.executorType = executorType;
            Eat('{');
            CHK_RET(ParseTemplateList(exec.algoList));
            SkipWs();
            if (!Eat('}')) {
                HCCL_ERROR("[HcclAlgoParser] expected '}' at pos %zu", pos_);
                return HCCL_E_PARA;
            }
            return HCCL_SUCCESS;
        }
        // template shorthand: name => sole{name}
        std::string algoType = ToLowerStr(UnderscoreToCamelCase(name));
        if (ALGO_TYPES.find(algoType) == ALGO_TYPES.end()) {
            HCCL_ERROR("[HcclAlgoParser] invalid algoType [%s] at pos %zu", algoType.c_str(), pos_);
            return HCCL_E_PARA;
        }
        exec.executorType = "sole";
        HcclAlgo algo;
        algo.algoType = algoType;
        algo.enable = true;
        InsertAlgoOrdered(exec.algoList, std::move(algo), LEVEL_UNSPECIFIED);
        return HCCL_SUCCESS;
    }

    // tpl_list := tpl_item (',' tpl_item)*
    HcclResult ParseTemplateList(std::vector<HcclAlgo>& algoList)
    {
        SkipWs();
        if (Peek() == '}')
            return HCCL_SUCCESS;
        while (true) {
            HcclAlgo algo;
            uint32_t level = LEVEL_UNSPECIFIED;
            CHK_RET(ParseTemplateItem(algo, level));
            InsertAlgoOrdered(algoList, std::move(algo), level);
            SkipWs();
            if (!Eat(','))
                break;
            SkipWs();
        }
        return HCCL_SUCCESS;
    }

    // tpl_item := ['level' digit '=' ] tpl_expr
    HcclResult ParseTemplateItem(HcclAlgo& algo, uint32_t& level)
    {
        SkipWs();
        size_t savePos = pos_;
        std::string name;
        if (ParseIdentifier(name)) {
            // 判定 "levelN"
            if (name.size() > 5 && ToLowerStr(name.substr(0, 5)) == "level") {
                bool allDigit = true;
                for (size_t i = 5; i < name.size(); i++) {
                    if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
                        allDigit = false;
                        break;
                    }
                }
                if (allDigit) {
                    SkipWs();
                    if (Eat('=')) {
                        try {
                            unsigned long parsed = std::stoul(name.substr(5));
                            if (parsed > UINT32_MAX) {
                                HCCL_ERROR("[HcclAlgoParser] level %lu exceeds uint32_t range", parsed);
                                return HCCL_E_PARA;
                            }
                            level = static_cast<uint32_t>(parsed);
                        } catch (...) {
                            HCCL_ERROR("[HcclAlgoParser] invalid level: %s", name.c_str());
                            return HCCL_E_PARA;
                        }
                        SkipWs();
                        return ParseTemplateExpr(algo);
                    }
                }
            }
            pos_ = savePos;
        }
        return ParseTemplateExpr(algo);
    }

    // tpl_expr := 'not' '(' template_name ')' | template_name
    HcclResult ParseTemplateExpr(HcclAlgo& algo)
    {
        SkipWs();
        size_t savePos = pos_;
        std::string name;
        if (ParseIdentifier(name)) {
            SkipWs();
            if (ToLowerStr(name) == "not" && Peek() == '(') {
                Eat('(');
                SkipWs();
                CHK_RET(ParseTemplateAtom(algo));
                algo.enable = false;
                SkipWs();
                if (!Eat(')')) {
                    HCCL_ERROR("[HcclAlgoParser] expected ')' after not(template) at pos %zu", pos_);
                    return HCCL_E_PARA;
                }
                return HCCL_SUCCESS;
            }
            pos_ = savePos;
        }
        return ParseTemplateAtom(algo);
    }

    // template_name
    HcclResult ParseTemplateAtom(HcclAlgo& algo)
    {
        SkipWs();
        std::string name;
        if (!ParseIdentifier(name)) {
            HCCL_ERROR("[HcclAlgoParser] expected template name at pos %zu", pos_);
            return HCCL_E_PARA;
        }
        std::string algoType = ToLowerStr(UnderscoreToCamelCase(name));
        if (ALGO_TYPES.find(algoType) == ALGO_TYPES.end()) {
            HCCL_ERROR("[HcclAlgoParser] invalid algoType [%s] at pos %zu", algoType.c_str(), pos_);
            return HCCL_E_PARA;
        }
        algo.algoType = algoType;
        return HCCL_SUCCESS;
    }
};

// ===========================================================================
// HcclAlgoExecutorParser 实现
// ===========================================================================

HcclResult HcclAlgoParser::Parser(const std::string& algoConfig)
{
    executorList.clear();
    if (algoConfig.empty()) {
        return HCCL_SUCCESS;
    }
    AlgoParserImpl parser(algoConfig);
    HcclResult ret = parser.Parse(executorList);
    if (ret != HCCL_SUCCESS) {
        executorList.clear();
        return ret;
    }
    HCCL_INFO("[HcclAlgoParser] parse ok, %s", ToString().c_str());
    return HCCL_SUCCESS;
}

std::string HcclAlgoParser::ToString() const
{
    std::string s = "HcclAlgoExecutorParser{ executorList=[";
    for (size_t i = 0; i < executorList.size(); i++) {
        if (i > 0)
            s += "; ";
        const auto& exec = executorList[i];
        if (!exec.opType.empty())
            s += exec.opType + ":";
        if (!exec.enable)
            s += "not(";
        s += exec.executorType + "{";
        for (size_t j = 0; j < exec.algoList.size(); j++) {
            if (j > 0)
                s += ",";
            s += "level" + std::to_string(j) + "=";
            if (!exec.algoList[j].enable)
                s += "not(";
            s += exec.algoList[j].algoType;
            if (!exec.algoList[j].enable)
                s += ")";
        }
        s += "}";
        if (!exec.enable)
            s += ")";
    }
    s += "] }";
    return s;
}

// ===========================================================================
// CostModel 刷新：UpdateCostModelWithAlgo
// ===========================================================================
// 拼接算法名：[EngineType][OpType(cap)][ExecutorType(cap)][AlgoType0(cap)][AlgoType1(cap)]...
// 驼峰命名：首字段（EngineType）小写开头，后续字段首字母大写
static std::string ComposeAlgoName(
    const std::string& engineType, const std::string& opType, const std::string& executorType,
    const std::vector<HcclAlgo>& algoList)
{
    std::string name = engineType; // 首字段保持小写（camelCase）
    name += OP_TYPES.at(opType);
    name += EXECUTOR_TYPES.at(executorType);
    for (const auto& algo : algoList) {
        name += ALGO_TYPES.at(algo.algoType);
    }
    return name;
}

// 拼接算法名前缀（algoList 为空时使用）：[EngineType][OpType(cap)][ExecutorType(cap)]
static std::string
ComposeAlgoPrefix(const std::string& engineType, const std::string& opType, const std::string& executorType)
{
    return engineType + OP_TYPES.at(opType) + EXECUTOR_TYPES.at(executorType);
}

// 前缀匹配
static bool StartsWith(const std::string& str, const std::string& prefix)
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

// algoType 驼峰名列表（按长度降序，避免 "Mesh" 误匹配 "Mesh2Die"）
static const std::vector<std::string> SORTED_ALGO_NAMES = []() {
    std::vector<std::string> names;
    for (const auto& pair : ALGO_TYPES) {
        names.push_back(pair.second);
    }
    std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
        return a.size() > b.size();
    });
    return names;
}();

// 逐个算法名检查是否匹配 algoList 模式（规则 4.6.2，支持多个 level 同时有 not）
// 前缀 engine+opType+executorType 精确匹配，后续 algoType 段逐个匹配：
//   enable=true  → 该位置必须是指定 algoType
//   enable=false → 该位置可以是任意 algoType，但不能是指定 algoType
static bool MatchesAlgoPattern(
    const std::string& key, const std::string& engine, const std::string& opType, const std::string& executorType,
    const std::vector<HcclAlgo>& algoList)
{
    std::string prefix = engine + OP_TYPES.at(opType) + EXECUTOR_TYPES.at(executorType);
    if (!StartsWith(key, prefix))
        return false;
    std::string remaining = key.substr(prefix.size());
    size_t pos = 0;
    for (const auto& algo : algoList) {
        // 尝试在当前位置匹配一个 algoType（长名优先）
        std::string matched;
        for (const auto& name : SORTED_ALGO_NAMES) {
            if (remaining.compare(pos, name.size(), name) == 0) {
                matched = name;
                break;
            }
        }
        if (matched.empty())
            return false;
        if (algo.enable) {
            if (matched != ALGO_TYPES.at(algo.algoType))
                return false;
        } else {
            if (matched == ALGO_TYPES.at(algo.algoType))
                return false;
        }
        pos += matched.size();
    }
    return pos == remaining.size();
}

// 判断算法名是否属于指定 OpType
// costModel 算法名使用 ENGINE_TYPES.second（驼峰）+ OP_TYPES.second + EXECUTOR_TYPES.second 格式
static bool IsAlgoOfOpType(const std::string& algoKey, const std::string& opType)
{
    std::string camelOpType = OP_TYPES.at(opType);
    for (const auto& enginePair : ENGINE_TYPES) {
        std::string prefix = enginePair.second + camelOpType;
        if (!StartsWith(algoKey, prefix) || algoKey.size() == prefix.size()) {
            continue;
        }
        std::string rest = algoKey.substr(prefix.size());
        for (const auto& execPair : EXECUTOR_TYPES) {
            if (StartsWith(rest, execPair.second)) {
                return true;
            }
        }
    }
    return false;
}

// 判断算法名是否包含 send 或 recv（大小写不敏感）
// 规则 4.6：send/recv 算法名不参与 count=0 的排除逻辑
static bool ContainsSendRecv(const std::string& algoKey)
{
    std::string lower = ToLowerStr(algoKey);
    return lower.find("send") != std::string::npos || lower.find("recv") != std::string::npos;
}

// 匹配过程共享上下文，用于减少函数参数个数
struct AlgoMatchCtx {
    CostModel& model;
    const std::vector<std::string>& engineTypes;
    const HcclAlgoExecutor& exec;
    const std::map<std::string, int>& keyToIdx;
    std::set<std::string>& matchedOpTypes;
};

// 单次 opType 匹配的结果
struct MatchResult {
    std::vector<std::string> matchedNames;
    bool negatedFound = false;
};

// 对单个 opType 遍历所有 engine 收集匹配结果（三种匹配模式）
static void CollectMatchedNamesForOpType(
    AlgoMatchCtx& ctx, const std::string& opType, bool isExecNegated, bool hasNegatedAlgo, MatchResult& result)
{
    for (const auto& engine : ctx.engineTypes) {
        if (hasNegatedAlgo) {
            for (int i = 0; i < ctx.model.count; i++) {
                if (ctx.model.costAlgoParams[i].count == 0)
                    continue;
                std::string key(ctx.model.costAlgoParams[i].algName ? ctx.model.costAlgoParams[i].algName : "");
                if (MatchesAlgoPattern(key, engine, opType, ctx.exec.executorType, ctx.exec.algoList)) {
                    result.matchedNames.push_back(key);
                }
            }
        } else if (ctx.exec.algoList.empty()) {
            std::string prefix = ComposeAlgoPrefix(engine, opType, ctx.exec.executorType);
            for (int i = 0; i < ctx.model.count; i++) {
                std::string key(ctx.model.costAlgoParams[i].algName ? ctx.model.costAlgoParams[i].algName : "");
                if (!StartsWith(key, prefix))
                    continue;
                if (isExecNegated) {
                    ctx.model.costAlgoParams[i].count = 0;
                    result.negatedFound = true;
                } else if (ctx.model.costAlgoParams[i].count != 0) {
                    result.matchedNames.push_back(key);
                }
            }
        } else {
            std::string fullName = ComposeAlgoName(engine, opType, ctx.exec.executorType, ctx.exec.algoList);
            auto it = ctx.keyToIdx.find(fullName);
            if (it == ctx.keyToIdx.end())
                continue;
            int algoIdx = it->second;
            if (isExecNegated) {
                ctx.model.costAlgoParams[algoIdx].count = 0;
                result.negatedFound = true;
            } else if (ctx.model.costAlgoParams[algoIdx].count != 0) {
                result.matchedNames.push_back(fullName);
            }
        }
    }
}

// 匹配后处理：标记 OpType 并排除未匹配算法
static void
ProcessMatchedResults(AlgoMatchCtx& ctx, const std::string& opType, bool isExecNegated, const MatchResult& result)
{
    if (!isExecNegated && !result.matchedNames.empty()) {
        ctx.matchedOpTypes.insert(opType);
        for (int i = 0; i < ctx.model.count; i++) {
            std::string key(ctx.model.costAlgoParams[i].algName ? ctx.model.costAlgoParams[i].algName : "");
            if (!IsAlgoOfOpType(key, opType))
                continue;
            bool isMatched = false;
            for (const auto& name : result.matchedNames) {
                if (key == name) {
                    isMatched = true;
                    break;
                }
            }
            if (!isMatched && !ContainsSendRecv(key)) {
                ctx.model.costAlgoParams[i].count = 0;
            }
        }
    } else if (isExecNegated && result.negatedFound) {
        ctx.matchedOpTypes.insert(opType);
    }
}

// 排除不在engineTypes列表中的算法
static HcclResult ExcludeAlgosNotInEngines(CostModel& model, const std::vector<std::string>& engineTypes)
{
    if (engineTypes.empty()) {
        HCCL_ERROR("engineTypes is empty");
        return HCCL_E_PARA;
    }

    std::string engines;
    for (const auto& engine : engineTypes) {
        if (!engines.empty())
            engines += ", ";
        engines += engine;
    }
    HCCL_INFO("[UpdateCostModelWithAlgo] engineTypes: [%s]", engines.c_str());

    for (int i = 0; i < model.count; i++) {
        if (model.costAlgoParams[i].count == 0)
            continue;
        std::string key(model.costAlgoParams[i].algName ? model.costAlgoParams[i].algName : "");
        if (ContainsSendRecv(key))
            continue; // send/recv 不处理
        bool inEngine = false;
        for (const auto& engine : engineTypes) {
            if (StartsWith(key, engine)) {
                inEngine = true;
                break;
            }
        }
        if (!inEngine) {
            model.costAlgoParams[i].count = 0;
        }
    }
    return HCCL_SUCCESS;
}

// 主函数
HcclResult
UpdateCostModelWithAlgo(const HcclAlgoParser& algoParser, CostModel& model, const std::vector<std::string>& engineTypes)
{
    CHK_RET(ExcludeAlgosNotInEngines(model, engineTypes));

    std::map<std::string, int> keyToIdx;
    for (int i = 0; i < model.count; i++) {
        if (model.costAlgoParams[i].algName != nullptr) {
            keyToIdx[model.costAlgoParams[i].algName] = i;
        }
    }
    std::set<std::string> matchedOpTypes;

    for (int idx = static_cast<int>(algoParser.executorList.size()) - 1; idx >= 0; idx--) {
        const auto& exec = algoParser.executorList[idx];
        if (exec.executorType.empty())
            continue;
        if (EXECUTOR_TYPES.find(exec.executorType) == EXECUTOR_TYPES.end())
            continue;
        bool algoValid = true;
        for (const auto& algo : exec.algoList) {
            if (ALGO_TYPES.find(algo.algoType) == ALGO_TYPES.end()) {
                algoValid = false;
                break;
            }
        }
        if (!algoValid)
            continue;

        std::vector<std::string> unprocessedOps;
        if (exec.opType.empty()) {
            for (const auto& pair : OP_TYPES) {
                if (matchedOpTypes.find(pair.first) == matchedOpTypes.end())
                    unprocessedOps.push_back(pair.first);
            }
        } else if (
            OP_TYPES.find(exec.opType) != OP_TYPES.end() && matchedOpTypes.find(exec.opType) == matchedOpTypes.end()) {
            unprocessedOps.push_back(exec.opType);
        }
        if (unprocessedOps.empty())
            continue;

        bool isExecNegated = !exec.enable;
        bool hasNegatedAlgo
            = !isExecNegated && std::any_of(exec.algoList.begin(), exec.algoList.end(), [](const HcclAlgo& a) {
                  return !a.enable;
              });

        AlgoMatchCtx ctx{model, engineTypes, exec, keyToIdx, matchedOpTypes};
        for (const auto& opType : unprocessedOps) {
            MatchResult result;
            CollectMatchedNamesForOpType(ctx, opType, isExecNegated, hasNegatedAlgo, result);
            ProcessMatchedResults(ctx, opType, isExecNegated, result);
            bool allMatched = true;
            for (const auto& pair : OP_TYPES) {
                if (matchedOpTypes.find(pair.first) == matchedOpTypes.end()) {
                    allMatched = false;
                    break;
                }
            }
            if (allMatched) {
                return HCCL_SUCCESS;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult FilterCmByHcclAlgo(HcclComm comm, CostModel& cm, const std::vector<std::string>& candidateEngineNames)
{
    // 获取配置：通信域 hcclAlgo 优先，其次环境变量 HCCL_ALGO
    std::string algoConfig;
    HcclResult ret = HcclGetHcclAlgo(comm, algoConfig);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[FilterCmByHcclAlgo] HcclGetHcclAlgo failed, ret[%d], try env variable.", ret);
        algoConfig.clear();
    }

    if (algoConfig.empty()) {
        algoConfig = GetEnv("HCCL_ALGO");
        if (algoConfig == "EmptyString") {
            HCCL_WARNING("[FilterCmByHcclAlgo] both hcclAlgo and HCCL_ALGO env are empty, skip filtering.");
            return HCCL_SUCCESS;
        }
    }
    HCCL_INFO("[FilterCmByHcclAlgo] use algo config: [%s]", algoConfig.c_str());

    // 解析算法配置
    HcclAlgoParser algoParser;
    ret = algoParser.Parser(algoConfig);
    if (ret != HCCL_SUCCESS) {
        HcclDevType deviceType;
        CHK_RET(HcclGetDeviceType(deviceType));
        if (deviceType != HcclDevType::DEV_TYPE_910_93) {
            ret = SetHcclAlgoConfig(algoConfig);
            HCCL_WARNING("[FilterCmByHcclAlgo] parse algo failed, try Parse with old rules: ret[%d] .", ret);
        }
        return ret;
    }

    // 刷新 CostModel,使用 selector 传入的候选引擎前缀
    ret = UpdateCostModelWithAlgo(algoParser, cm, candidateEngineNames);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[FilterCmByHcclAlgo] UpdateCostModelWithAlgo failed, ret[%d].", ret);
        return ret;
    }

    std::string content;
    for (int i = 0; i < cm.count; i++) {
        if (!content.empty()) {
            content += ", ";
        }
        content += "[" + std::to_string(i) + "]"
                   + std::string(cm.costAlgoParams[i].algName ? cm.costAlgoParams[i].algName : "null") + ":"
                   + std::to_string(cm.costAlgoParams[i].count);
    }
    HCCL_DEBUG("[FilterCmByHcclAlgo] final costModel: %s", content.c_str());

    HCCL_INFO("[FilterCmByHcclAlgo] filter costModel success.");
    return HCCL_SUCCESS;
}

// ---------------------------------------------------------------------------
// 派生数组接口：从 map 构建 AlgoDimEntry 数组，供 selector 遍历
// ---------------------------------------------------------------------------
static std::vector<AlgoDimEntry> BuildDimEntries(const std::map<std::string, std::string>& m)
{
    std::vector<AlgoDimEntry> v;
    v.reserve(m.size());
    for (const auto& p : m) {
        v.push_back({p.first.c_str(), p.second.c_str()});
    }
    return v;
}

const AlgoDimEntry* GetAlgoEngines(int& count)
{
    static auto entries = BuildDimEntries(ENGINE_TYPES);
    count = static_cast<int>(entries.size());
    return entries.data();
}

const AlgoDimEntry* GetAlgoExecutors(int& count)
{
    static auto entries = BuildDimEntries(EXECUTOR_TYPES);
    count = static_cast<int>(entries.size());
    return entries.data();
}

const AlgoDimEntry* GetAlgoTemplates(int& count)
{
    static auto entries = BuildDimEntries(ALGO_TYPES);
    count = static_cast<int>(entries.size());
    return entries.data();
}

const EnginePrefixEntry* GetEnginePrefixEntries(int& count)
{
    static const std::vector<EnginePrefixEntry> entries = {
        {"CcuSched", OpExecuteConfig::CCU_SCHED}, {"CcuMS", OpExecuteConfig::CCU_MS},
        {"Aicpu", OpExecuteConfig::AICPU_TS},     {"Aiv", OpExecuteConfig::AIV},
        {"Dpu", OpExecuteConfig::HOSTCPU},
    };
    count = static_cast<int>(entries.size());
    return entries.data();
}

const OpTypePatternEntry* GetOpTypePatternEntries(int& count)
{
    static const std::vector<OpTypePatternEntry> entries = {
        {"ReduceScatterV", HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V},
        {"ReduceScatter", HcclCMDType::HCCL_CMD_REDUCE_SCATTER},
        {"AllGatherV", HcclCMDType::HCCL_CMD_ALLGATHER_V},
        {"AllGather", HcclCMDType::HCCL_CMD_ALLGATHER},
        {"AllReduce", HcclCMDType::HCCL_CMD_ALLREDUCE},
        {"AllToAllVC", HcclCMDType::HCCL_CMD_ALLTOALLVC},
        {"AllToAllV", HcclCMDType::HCCL_CMD_ALLTOALLV},
        {"AllToAll", HcclCMDType::HCCL_CMD_ALLTOALL},
        {"Broadcast", HcclCMDType::HCCL_CMD_BROADCAST},
        {"Reduce", HcclCMDType::HCCL_CMD_REDUCE},
        {"Scatter", HcclCMDType::HCCL_CMD_SCATTER},
    };
    count = static_cast<int>(entries.size());
    return entries.data();
}

const std::map<AlgoType, std::string>& GetAlgoTypeToNameMap()
{
    static const std::map<AlgoType, std::string> map = {
        {AlgoType::MESH, "Mesh"},
        {AlgoType::MESH_2DIE, "Mesh2Die"},
        {AlgoType::MESH_ONESHOT, "MeshOneShot"},
        {AlgoType::MESH_TWOSHOT, "MeshTwoShot"},
        {AlgoType::MESH_CONCUR, "MeshConcur"},
        {AlgoType::MESH_MULTILINK, "MeshMultiLink"},
        {AlgoType::MESH_CHUNK, "MeshChunk"},
        {AlgoType::MESH_CHUNK_TWOSHOT, "MeshChunkTwoShot"},
        {AlgoType::NHR, "NHR"},
        {AlgoType::NHR_MULTILINK, "NHRMultiLink"},
        {AlgoType::NHR_AICPU_REDUCE, "NHRAicpuReduce"},
        {AlgoType::MESH_SINGLE_CHANNEL, "MeshSingleChannel"},
        {AlgoType::MESH_CONCURRENT, "MeshConcurrent"},
    };
    return map;
}

const std::map<std::string, AlgoType>& GetAlgoNameToTypeMap()
{
    static const std::map<std::string, AlgoType> map = {
        {"Mesh", AlgoType::MESH},
        {"Mesh2Die", AlgoType::MESH_2DIE},
        {"MeshOneShot", AlgoType::MESH_ONESHOT},
        {"MeshTwoShot", AlgoType::MESH_TWOSHOT},
        {"MeshConcur", AlgoType::MESH_CONCUR},
        {"MeshMultiLink", AlgoType::MESH_MULTILINK},
        {"MeshChunk", AlgoType::MESH_CHUNK},
        {"MeshChunkTwoShot", AlgoType::MESH_CHUNK_TWOSHOT},
        {"NHR", AlgoType::NHR},
        {"NHRMultiLink", AlgoType::NHR_MULTILINK},
        {"NHRAicpuReduce", AlgoType::NHR_AICPU_REDUCE},
        {"MeshSingleChannel", AlgoType::MESH_SINGLE_CHANNEL},
        {"MeshConcurrent", AlgoType::MESH_CONCURRENT},
    };
    return map;
}

std::string AlgoTypeToString(AlgoType t)
{
    const auto& map = GetAlgoTypeToNameMap();
    auto it = map.find(t);
    if (it != map.end()) {
        return it->second;
    }
    return "Unknown";
}

std::string HcclCMDTypeToString(HcclCMDType opType)
{
    switch (opType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            return "AllReduce";
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            return "AllGather";
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
            return "AllGatherV";
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            return "ReduceScatter";
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
            return "ReduceScatterV";
        case HcclCMDType::HCCL_CMD_BROADCAST:
            return "Broadcast";
        case HcclCMDType::HCCL_CMD_REDUCE:
            return "Reduce";
        case HcclCMDType::HCCL_CMD_SCATTER:
            return "Scatter";
        case HcclCMDType::HCCL_CMD_ALLTOALL:
            return "AllToAll";
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
            return "AllToAllV";
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            return "AllToAllVC";
        case HcclCMDType::HCCL_CMD_BARRIER:
            return "Barrier";
        case HcclCMDType::HCCL_CMD_SEND:
            return "Send";
        case HcclCMDType::HCCL_CMD_RECEIVE:
            return "Recv";
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
            return "BatchSendRecv";
        default:
            return "Unknown";
    }
}

std::string OpExecuteConfigToString(OpExecuteConfig engine)
{
    switch (engine) {
        case OpExecuteConfig::AICPU_TS:
            return "Aicpu";
        case OpExecuteConfig::AIV:
            return "Aiv";
        case OpExecuteConfig::AIV_ONLY:
            return "AivOnly";
        case OpExecuteConfig::HOSTCPU:
            return "Dpu";
        case OpExecuteConfig::CCU_MS:
            return "CcuMS";
        case OpExecuteConfig::CCU_SCHED:
            return "CcuSched";
        default:
            return "Unknown";
    }
}

} // namespace ops_hccl
