/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* HCCL Tuner Plugin 示例实现。
 *
 * 功能：读取 JSON 配置文件，按 match 条件匹配集合通信调用，修改 cost table
 * （hcclTunerAlgoEntry_t[] 命名条目数组）以影响 Selector 算法选择。
 *
 * 编译：make（链接为 hccl_tuner_example.so）
 * 使用：export HCCL_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
 *       export HCCL_TUNER_CONFIG_FILE=/path/to/hccl_tuner_config.json
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <nlohmann/json.hpp>

#include "hccl_tuner_plugin.h"
#include "securec.h"

/* ===== 常量 ===== */
#define MAX_OP_TYPES 8
#define MAX_TOTAL_RULES 5000
#define MAX_RULES_PER_OP 2000
#define MAX_STR_LEN 64
#define MAX_FILE_SIZE (4 * 1024 * 1024)

/* ===== 维度校验（与 alg_parse.cc 的 ENGINE_TYPES/EXECUTOR_TYPES/ALGO_TYPES key 保持一致）===== */
static const char* g_validEngines[] = {"aicpu", "ccums", "ccusched", "aiv", "dpu"};
static const char* g_validExecutors[] = {"sole", "sequence", "parallel", "pipeline", "concur"};

static int is_valid_name(const char* s, const char** list, int count)
{
    if (s == nullptr) {
        return 0;
    }
    for (int i = 0; i < count; i++) {
        if (strcmp(s, list[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static int is_valid_engine(const char* s)
{
    return is_valid_name(s, g_validEngines, (int)(sizeof(g_validEngines) / sizeof(g_validEngines[0])));
}

static int is_valid_executor(const char* s)
{
    return is_valid_name(s, g_validExecutors, (int)(sizeof(g_validExecutors) / sizeof(g_validExecutors[0])));
}

static HcclCMDType op_type_from_name(const char* name)
{
    if (name == nullptr) {
        return HCCL_CMD_INVALID;
    }
    if (strcmp(name, "allreduce") == 0)
        return HCCL_CMD_ALLREDUCE;
    if (strcmp(name, "allgather") == 0)
        return HCCL_CMD_ALLGATHER;
    if (strcmp(name, "broadcast") == 0)
        return HCCL_CMD_BROADCAST;
    if (strcmp(name, "reduce") == 0)
        return HCCL_CMD_REDUCE;
    if (strcmp(name, "reduce_scatter") == 0)
        return HCCL_CMD_REDUCE_SCATTER;
    if (strcmp(name, "scatter") == 0)
        return HCCL_CMD_SCATTER;
    if (strcmp(name, "alltoall") == 0)
        return HCCL_CMD_ALLTOALL;
    if (strcmp(name, "alltoallv") == 0)
        return HCCL_CMD_ALLTOALLV;
    return HCCL_CMD_INVALID;
}
#define COMM_NAME_BUF_LEN 128

/* ===== 数据结构 ===== */
typedef struct {
    int hasMinRanks;
    int hasMaxRanks;
    int hasMinBytes;
    int hasMaxBytes;
    int hasDataType;
    int hasCommName;
    int hasMinNpusPerServer;
    int hasMaxNpusPerServer;
    int hasMinServers;
    int hasMaxServers;
    int hasMinPods;
    int hasMaxPods;
    int hasMinSuperPods;
    int hasMaxSuperPods;
    int hasBufferSize;
    uint32_t minRanks;
    uint32_t maxRanks;
    size_t minBytes;
    size_t maxBytes;
    char dataType[MAX_STR_LEN];
    HcclDataType dataTypeEnum;
    char commName[MAX_STR_LEN];
    uint32_t minNpusPerServer;
    uint32_t maxNpusPerServer;
    uint32_t minServers;
    uint32_t maxServers;
    uint32_t minPods;
    uint32_t maxPods;
    uint32_t minSuperPods;
    uint32_t maxSuperPods;
    uint64_t bufferSize;
} MatchCond;

typedef struct {
    MatchCond match;
    char engine[16];        /* "aicpu" 或 "" (通配) */
    char executor[16];      /* "sole" 或 "" */
    char templateName[128]; /* "meshoneshot" 单级 或 "meshconcurnhrnhr" 多级拼接串 */
    float cost;
    int hasCost;
} Rule;

typedef struct {
    HcclCMDType opType;
    int ruleCount;
    int ruleOffset; /* 在扁平 Rule[] 中的起始下标 */
} OpSetDesc;

/* per-comm 存储上下文（经 hostFuncs->ctxCreate 持久化到通信域 host 内存）。
 * Rule[] 紧随 StoredHeader 之后，变长，ctxCreate 时按 totalRuleCount 精确分配。 */
typedef struct {
    OpSetDesc opSets[MAX_OP_TYPES];
    int opSetCount;
    int totalRuleCount;
    hcclTunerCommInfo_t commInfo;
    char commNameBuf[COMM_NAME_BUF_LEN];
    int configValid; /* Schema 校验结果：1=有效可干预，0=无效不干预 */
} StoredHeader;

/* 访问 StoredHeader 后面的变长 Rule[] */
static inline Rule* GetRules(StoredHeader* ctx)
{
    return reinterpret_cast<Rule*>(reinterpret_cast<char*>(ctx) + sizeof(StoredHeader));
}

/* ===== 全局 hostFuncs（函数表，全进程相同）===== */
static hcclTunerHostFunctions_t g_hostFuncs;
static int g_hostFuncsReady = 0;

/* Schema 校验状态（per-init 局部变量，避免并发 comm init 竞争） */
typedef struct {
    int errors;
    int warnings;
    int foundVersion;
    int foundOpTypes;
} SchemaState;

#ifdef HCCL_TUNER_TESTING
static SchemaState g_lastSchema = {}; /* 供测试验证 schema 校验结果 */
#endif

static void SchemaWarn(SchemaState* s, const char* scope, const char* field)
{
    s->warnings++;
    if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
        g_hostFuncs.logFunction(
            HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "Schema: unknown field '%s' in %s, skipping", field, scope);
    }
}

static void SchemaError(SchemaState* s, const char* fmt, ...)
{
    s->errors++;
    if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
        va_list args;
        va_start(args, fmt);
        char buf[256] = {0};
        if (vsnprintf_s(buf, sizeof(buf), sizeof(buf) - 1, fmt, args) < 0) {
            buf[0] = '\0';
        }
        va_end(args);
        g_hostFuncs.logFunction(HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "Schema: %s", buf);
    }
}

/* ===== 数据类型字符串映射 ===== */
static const struct {
    const char* name;
    HcclDataType type;
} g_dataTypeMap[] = {
    {"int8", HCCL_DATA_TYPE_INT8},      {"int16", HCCL_DATA_TYPE_INT16},   {"int32", HCCL_DATA_TYPE_INT32},
    {"int64", HCCL_DATA_TYPE_INT64},    {"uint8", HCCL_DATA_TYPE_UINT8},   {"uint16", HCCL_DATA_TYPE_UINT16},
    {"uint32", HCCL_DATA_TYPE_UINT32},  {"uint64", HCCL_DATA_TYPE_UINT64}, {"fp16", HCCL_DATA_TYPE_FP16},
    {"float16", HCCL_DATA_TYPE_FP16},   {"fp32", HCCL_DATA_TYPE_FP32},     {"float32", HCCL_DATA_TYPE_FP32},
    {"fp64", HCCL_DATA_TYPE_FP64},      {"float64", HCCL_DATA_TYPE_FP64},  {"bfp16", HCCL_DATA_TYPE_BFP16},
    {"bfloat16", HCCL_DATA_TYPE_BFP16},
};

static HcclDataType ParseDataType(const char* name)
{
    for (size_t i = 0; i < sizeof(g_dataTypeMap) / sizeof(g_dataTypeMap[0]); i++) {
        if (strcmp(name, g_dataTypeMap[i].name) == 0) {
            return g_dataTypeMap[i].type;
        }
    }
    return HCCL_DATA_TYPE_RESERVED;
}

/* 维度值校验由插件自维护的 is_valid_engine/is_valid_executor 完成（template 不校验，靠运行时 warning 兜底）。 */

/* ===== JSON 解析（nlohmann/json）===== */

/* 第一遍：遍历 JSON 树，统计总规则数。返回总规则数，opSetCount 写入出参。 */
static int CountRules(const nlohmann::json& root, int* opSetCount)
{
    *opSetCount = 0;
    if (!root.is_object() || !root.contains("op_types") || !root["op_types"].is_object()) {
        return 0;
    }
    int total = 0;
    for (auto it = root["op_types"].begin(); it != root["op_types"].end(); ++it) {
        if (*opSetCount >= MAX_OP_TYPES) {
            break;
        }
        HcclCMDType opType = op_type_from_name(it.key().c_str());
        if (opType == HCCL_CMD_INVALID) {
            continue;
        }
        if (it.value().is_object() && it.value().contains("rules") && it.value()["rules"].is_array()) {
            total += (int)it.value()["rules"].size();
        }
        (*opSetCount)++;
    }
    return total;
}

static void ParseMatchField(const nlohmann::json& matchObj, Rule* r, SchemaState* s)
{
    static const char* knownKeys[]
        = {"min_ranks",           "max_ranks",           "min_bytes",   "max_bytes",   "data_type", "comm_name",
           "min_npus_per_server", "max_npus_per_server", "min_servers", "max_servers", "min_pods",  "max_pods",
           "min_super_pods",      "max_super_pods",      "buffer_size"};
    static const int knownKeyCount = (int)(sizeof(knownKeys) / sizeof(knownKeys[0]));
    for (auto it = matchObj.begin(); it != matchObj.end(); ++it) {
        bool known = false;
        for (int k = 0; k < knownKeyCount; k++) {
            if (it.key() == knownKeys[k]) {
                known = true;
                break;
            }
        }
        if (!known) {
            SchemaWarn(s, "match", it.key().c_str());
        }
    }
    if (matchObj.contains("min_ranks")) {
        r->match.hasMinRanks = 1;
        r->match.minRanks = matchObj["min_ranks"].get<uint32_t>();
    }
    if (matchObj.contains("max_ranks")) {
        r->match.hasMaxRanks = 1;
        r->match.maxRanks = matchObj["max_ranks"].get<uint32_t>();
    }
    if (matchObj.contains("min_bytes")) {
        r->match.hasMinBytes = 1;
        r->match.minBytes = matchObj["min_bytes"].get<size_t>();
    }
    if (matchObj.contains("max_bytes")) {
        r->match.hasMaxBytes = 1;
        r->match.maxBytes = matchObj["max_bytes"].get<size_t>();
    }
    if (matchObj.contains("data_type")) {
        if (snprintf_s(
                r->match.dataType, sizeof(r->match.dataType), sizeof(r->match.dataType) - 1, "%s",
                matchObj["data_type"].get<std::string>().c_str())
            < 0) {
            r->match.dataType[0] = '\0';
        }
        r->match.dataTypeEnum = ParseDataType(r->match.dataType);
        r->match.hasDataType = 1;
    }
    if (matchObj.contains("comm_name")) {
        if (snprintf_s(
                r->match.commName, sizeof(r->match.commName), sizeof(r->match.commName) - 1, "%s",
                matchObj["comm_name"].get<std::string>().c_str())
            < 0) {
            r->match.commName[0] = '\0';
        }
        r->match.hasCommName = 1;
    }
    if (matchObj.contains("min_npus_per_server")) {
        r->match.hasMinNpusPerServer = 1;
        r->match.minNpusPerServer = matchObj["min_npus_per_server"].get<uint32_t>();
    }
    if (matchObj.contains("max_npus_per_server")) {
        r->match.hasMaxNpusPerServer = 1;
        r->match.maxNpusPerServer = matchObj["max_npus_per_server"].get<uint32_t>();
    }
    if (matchObj.contains("min_servers")) {
        r->match.hasMinServers = 1;
        r->match.minServers = matchObj["min_servers"].get<uint32_t>();
    }
    if (matchObj.contains("max_servers")) {
        r->match.hasMaxServers = 1;
        r->match.maxServers = matchObj["max_servers"].get<uint32_t>();
    }
    if (matchObj.contains("min_pods")) {
        r->match.hasMinPods = 1;
        r->match.minPods = matchObj["min_pods"].get<uint32_t>();
    }
    if (matchObj.contains("max_pods")) {
        r->match.hasMaxPods = 1;
        r->match.maxPods = matchObj["max_pods"].get<uint32_t>();
    }
    if (matchObj.contains("min_super_pods")) {
        r->match.hasMinSuperPods = 1;
        r->match.minSuperPods = matchObj["min_super_pods"].get<uint32_t>();
    }
    if (matchObj.contains("max_super_pods")) {
        r->match.hasMaxSuperPods = 1;
        r->match.maxSuperPods = matchObj["max_super_pods"].get<uint32_t>();
    }
    if (matchObj.contains("buffer_size")) {
        r->match.hasBufferSize = 1;
        r->match.bufferSize = matchObj["buffer_size"].get<uint64_t>();
    }
}

static void ParseRule(const nlohmann::json& ruleObj, Rule* r, SchemaState* s)
{
    if (memset_s(r, sizeof(*r), 0, sizeof(*r)) != EOK) {
        SchemaError(s, "memset_s failed for rule");
        return;
    }

    if (!ruleObj.is_object()) {
        SchemaError(s, "rule is not an object");
        return;
    }

    // 检查 rule 层级未知字段（拼写检测）
    static const char* knownRuleKeys[] = {"match", "engine", "executor", "template", "cost", "args"};
    for (auto it = ruleObj.begin(); it != ruleObj.end(); ++it) {
        bool known = false;
        for (size_t k = 0; k < sizeof(knownRuleKeys) / sizeof(knownRuleKeys[0]); k++) {
            if (it.key() == knownRuleKeys[k]) {
                known = true;
                break;
            }
        }
        if (!known) {
            SchemaWarn(s, "rule", it.key().c_str());
        }
    }

    if (ruleObj.contains("match") && ruleObj["match"].is_object()) {
        ParseMatchField(ruleObj["match"], r, s);
    } else {
        SchemaError(s, "rule missing required field 'match'");
    }

    if (ruleObj.contains("engine")) {
        std::string buf = ruleObj["engine"].get<std::string>();
        if (!is_valid_engine(buf.c_str())) {
            SchemaError(s, "invalid engine '%s'", buf.c_str());
        } else {
            if (snprintf_s(r->engine, sizeof(r->engine), sizeof(r->engine) - 1, "%s", buf.c_str()) < 0) {
                r->engine[0] = '\0';
            }
        }
    }
    if (ruleObj.contains("executor")) {
        std::string buf = ruleObj["executor"].get<std::string>();
        if (!is_valid_executor(buf.c_str())) {
            SchemaError(s, "invalid executor '%s'", buf.c_str());
        } else {
            if (snprintf_s(r->executor, sizeof(r->executor), sizeof(r->executor) - 1, "%s", buf.c_str()) < 0) {
                r->executor[0] = '\0';
            }
        }
    }
    if (ruleObj.contains("template")) {
        std::string buf = ruleObj["template"].get<std::string>();
        /* template 不做枚举校验：单级是单 template 名，多级是拼接串（不可枚举）。
         * 拼写错误靠 ApplyRule 运行时 "no entry modified" warning 兜底。 */
        if (buf.empty()) {
            SchemaError(s, "invalid template '%s' (empty)", buf.c_str());
        } else {
            if (snprintf_s(r->templateName, sizeof(r->templateName), sizeof(r->templateName) - 1, "%s", buf.c_str())
                < 0) {
                r->templateName[0] = '\0';
            }
        }
    }
    if (ruleObj.contains("cost")) {
        r->cost = ruleObj["cost"].get<float>();
        r->hasCost = 1;
    }

    /* 必填字段校验 */
    if (!r->hasCost) {
        SchemaError(s, "rule missing required field 'cost'");
    }
    if (!r->match.hasMinRanks) {
        SchemaError(s, "rule missing required field 'min_ranks'");
    }
    if (!r->match.hasMaxRanks) {
        SchemaError(s, "rule missing required field 'max_ranks'");
    }
    if (!r->match.hasMinBytes) {
        SchemaError(s, "rule missing required field 'min_bytes'");
    }
    if (!r->match.hasMaxBytes) {
        SchemaError(s, "rule missing required field 'max_bytes'");
    }
    if (r->engine[0] == '\0') {
        SchemaError(s, "rule missing required field 'engine'");
    }
    if (r->executor[0] == '\0') {
        SchemaError(s, "rule missing required field 'executor'");
    }
    if (r->templateName[0] == '\0') {
        SchemaError(s, "rule missing required field 'template'");
    }
    if (r->match.hasMinRanks && r->match.hasMaxRanks && r->match.minRanks > r->match.maxRanks) {
        SchemaError(s, "min_ranks(%u) > max_ranks(%u)", r->match.minRanks, r->match.maxRanks);
    }
    if (r->match.hasMinBytes && r->match.hasMaxBytes && r->match.minBytes > r->match.maxBytes) {
        SchemaError(s, "min_bytes(%zu) > max_bytes(%zu)", r->match.minBytes, r->match.maxBytes);
    }
}

static void ParseOpRules(
    const nlohmann::json& opObj, StoredHeader* ctx, Rule* rules, int* curOffset, const char* opName, SchemaState* s)
{
    HcclCMDType opType = op_type_from_name(opName);
    if (opType == HCCL_CMD_INVALID) {
        SchemaError(s, "unknown op_type '%s'", opName);
        return;
    }
    if (ctx->opSetCount >= MAX_OP_TYPES) {
        SchemaError(s, "op_type count exceeds MAX_OP_TYPES(%d), '%s' skipped", MAX_OP_TYPES, opName);
        return;
    }
    OpSetDesc* desc = &ctx->opSets[ctx->opSetCount];
    desc->opType = opType;
    desc->ruleOffset = *curOffset;
    desc->ruleCount = 0;

    if (!opObj.is_object() || !opObj.contains("rules") || !opObj["rules"].is_array()) {
        SchemaError(s, "op_type '%s' missing 'rules' array", opName);
        ctx->opSetCount++;
        return;
    }

    for (const auto& ruleItem : opObj["rules"]) {
        if (desc->ruleCount < MAX_RULES_PER_OP) {
            ParseRule(ruleItem, &rules[*curOffset], s);
            (*curOffset)++;
            desc->ruleCount++;
        } else {
            SchemaError(
                s, "op_type '%s' rule count exceeds MAX_RULES_PER_OP(%d), extra rules skipped", opName,
                MAX_RULES_PER_OP);
            break;
        }
    }

    if (desc->ruleCount == 0) {
        SchemaError(s, "op_type '%s' has no rules (minItems=1)", opName);
    }
    ctx->opSetCount++;
}

static void ParseConfig(const nlohmann::json& root, StoredHeader* ctx, Rule* rules, int* curOffset, SchemaState* s)
{
    if (!root.is_object()) {
        SchemaError(s, "config root is not an object");
        return;
    }

    if (root.contains("version")) {
        s->foundVersion = 1;
        int ver = root["version"].get<int>();
        if (ver != 1) {
            SchemaError(s, "version must be 1, got %d", ver);
        }
    } else {
        SchemaError(s, "missing required field 'version'");
    }

    if (root.contains("op_types") && root["op_types"].is_object()) {
        s->foundOpTypes = 1;
        for (auto it = root["op_types"].begin(); it != root["op_types"].end(); ++it) {
            ParseOpRules(it.value(), ctx, rules, curOffset, it.key().c_str(), s);
        }
    } else {
        SchemaError(s, "missing required field 'op_types'");
    }
}

/* ===== 配置文件加载（多级 fallback）===== */
static char* ReadFile(const char* path, size_t* outLen)
{
    FILE* fp = fopen(path, "r");
    if (fp == NULL) {
        return NULL;
    }
    (void)fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    (void)fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz > MAX_FILE_SIZE) {
        fclose(fp);
        return NULL;
    }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (buf == NULL) {
        fclose(fp);
        return NULL;
    }
    size_t rd = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[rd] = '\0';
    *outLen = rd;
    return buf;
}

/* ===== 规则匹配 ===== */
static int MatchRule(const Rule* r, const hcclTunerCollInfo_t* collInfo, const hcclTunerCommInfo_t* commInfo)
{
    if (r->match.hasMinRanks && commInfo->nRanks < r->match.minRanks) {
        return 0;
    }
    if (r->match.hasMaxRanks && commInfo->nRanks > r->match.maxRanks) {
        return 0;
    }
    if (r->match.hasMinBytes && collInfo->nBytes < r->match.minBytes) {
        return 0;
    }
    if (r->match.hasMaxBytes && collInfo->nBytes > r->match.maxBytes) {
        return 0;
    }
    if (r->match.hasDataType) {
        if (r->match.dataTypeEnum != collInfo->dataType) {
            return 0;
        }
    }
    if (r->match.hasCommName) {
        if (commInfo->commName == NULL || strstr(commInfo->commName, r->match.commName) == NULL) {
            return 0;
        }
    }
    if (r->match.hasMinNpusPerServer && commInfo->nNpusPerServer < r->match.minNpusPerServer) {
        return 0;
    }
    if (r->match.hasMaxNpusPerServer && commInfo->nNpusPerServer > r->match.maxNpusPerServer) {
        return 0;
    }
    if (r->match.hasMinServers && commInfo->nServers < r->match.minServers) {
        return 0;
    }
    if (r->match.hasMaxServers && commInfo->nServers > r->match.maxServers) {
        return 0;
    }
    if (r->match.hasMinPods && commInfo->nPods < r->match.minPods) {
        return 0;
    }
    if (r->match.hasMaxPods && commInfo->nPods > r->match.maxPods) {
        return 0;
    }
    if (r->match.hasMinSuperPods && commInfo->nSuperPods < r->match.minSuperPods) {
        return 0;
    }
    if (r->match.hasMaxSuperPods && commInfo->nSuperPods > r->match.maxSuperPods) {
        return 0;
    }
    if (r->match.hasBufferSize && commInfo->bufferSize != r->match.bufferSize) {
        return 0;
    }
    return 1;
}

static void ApplyRule(const Rule* r, hcclTunerAlgoEntry_t* entries, int count)
{
    int modified = 0;
    for (int i = 0; i < count; i++) {
        if (r->engine[0] && (entries[i].engineName == NULL || strcmp(r->engine, entries[i].engineName) != 0))
            continue;
        if (r->executor[0] && (entries[i].executorName == NULL || strcmp(r->executor, entries[i].executorName) != 0))
            continue;
        if (r->templateName[0]
            && (entries[i].templateName == NULL || strcmp(r->templateName, entries[i].templateName) != 0))
            continue;
        if (entries[i].cost < 0.0f) {
            if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
                g_hostFuncs.logFunction(
                    HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "[TunerDFX] skip disabled: algName=%s cost=%.6f",
                    entries[i].algName ? entries[i].algName : "?", entries[i].cost);
            }
            continue;
        }
        float oldCost = entries[i].cost;
        entries[i].cost = r->hasCost ? r->cost : 0.0f;
        if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(
                HCCL_TUNER_LOG_INFO, __FILE__, __LINE__,
                "[TunerDFX] modify: algName=%s engine=%s executor=%s template=%s cost %.6f -> %.6f",
                entries[i].algName ? entries[i].algName : "?", entries[i].engineName ? entries[i].engineName : "?",
                entries[i].executorName ? entries[i].executorName : "?",
                entries[i].templateName ? entries[i].templateName : "?", oldCost, entries[i].cost);
        }
        modified++;
    }
    if (modified == 0 && g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
        g_hostFuncs.logFunction(
            HCCL_TUNER_LOG_WARN, __FILE__, __LINE__,
            "[TunerDFX] rule matched but no entry modified: engine=%s executor=%s template=%s cost=%.6f",
            r->engine[0] ? r->engine : "*", r->executor[0] ? r->executor : "*",
            r->templateName[0] ? r->templateName : "*", r->hasCost ? r->cost : 0.0f);
    }
}

/* ===== 插件接口实现 ===== */
static HcclResult MyInit(HcclComm comm, const hcclTunerCommInfo_t* commInfo, const hcclTunerHostFunctions_t* hostFuncs)
{
    if (commInfo == NULL || hostFuncs == NULL) {
        return HCCL_E_PTR;
    }
    /* hostFuncs 是函数表，全进程相同，存全局 */
    g_hostFuncs = *hostFuncs;
    g_hostFuncsReady = 1;

    /* 1. 读文件 + 解析 JSON（临时内存树，不持久化） */
    const char* envPath = getenv("HCCL_TUNER_CONFIG_FILE");
    const char* paths[] = {envPath, "./hccl_tuner_config.json", "/etc/hccl/hccl_tuner_config.json"};
    char* content = NULL;
    const char* loadedPath = NULL;
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        if (paths[i] == NULL || paths[i][0] == '\0') {
            continue;
        }
        size_t len = 0;
        content = ReadFile(paths[i], &len);
        if (content != NULL) {
            loadedPath = paths[i];
            break;
        }
        if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(
                HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "tuner config '%s' not found%s", paths[i],
                (i == 0) ? ", falling back to default paths" : "");
        }
    }
    if (content == NULL) {
        if (g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "no tuner config loaded, plugin inactive");
        }
        return HCCL_SUCCESS;
    }

    nlohmann::json root = nlohmann::json::parse(content, nullptr, false);
    if (root.is_discarded()) {
        if (g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "JSON parse failed for %s", loadedPath);
        }
        free(content);
        return HCCL_SUCCESS;
    }
    free(content);

    /* 2. 第一遍：遍历 JSON 树计数规则数 */
    int opSetCount = 0;
    int totalRules = CountRules(root, &opSetCount);
    if (totalRules > MAX_TOTAL_RULES) {
        if (g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(
                HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "total rules %d exceeds MAX_TOTAL_RULES(%d), plugin inactive",
                totalRules, MAX_TOTAL_RULES);
        }
        return HCCL_SUCCESS;
    }

    /* 3. ctxCreate 精确大小：Header + totalRules × Rule */
    uint64_t ctxSize = sizeof(StoredHeader) + (uint64_t)totalRules * sizeof(Rule);
    void* storedCtx = NULL;
    if (hostFuncs->ctxCreate(comm, "main", ctxSize, &storedCtx) != HCCL_SUCCESS || storedCtx == NULL) {
        if (g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "ctxCreate failed, plugin inactive");
        }
        return HCCL_SUCCESS;
    }
    if (memset_s(storedCtx, (size_t)ctxSize, 0, (size_t)ctxSize) != EOK) {
        if (g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "memset_s failed for storedCtx");
        }
        return HCCL_SUCCESS;
    }
    StoredHeader* ctx = (StoredHeader*)storedCtx;
    Rule* rules = GetRules(ctx);

    /* 4. 第二遍：填充 Header + Rule[] */
    SchemaState schema = {};
    int curOffset = 0;
    try {
        ParseConfig(root, ctx, rules, &curOffset, &schema);
    } catch (const nlohmann::json::exception& e) {
        SchemaError(&schema, "JSON field access error: %s", e.what());
    }
    ctx->totalRuleCount = curOffset;
#ifdef HCCL_TUNER_TESTING
    g_lastSchema = schema;
#endif
    if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
        g_hostFuncs.logFunction(
            HCCL_TUNER_LOG_INFO, __FILE__, __LINE__,
            "tuner config loaded from %s, opSetCount=%d, totalRules=%d, schemaErrors=%d, schemaWarnings=%d", loadedPath,
            ctx->opSetCount, ctx->totalRuleCount, schema.errors, schema.warnings);
    }
    if (schema.errors > 0) {
        if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(
                HCCL_TUNER_LOG_WARN, __FILE__, __LINE__,
                "Schema validation failed (%d errors), plugin will not intervene", schema.errors);
        }
        ctx->configValid = 0;
    } else {
        ctx->configValid = 1;
    }

    /* 5. 拷贝 commInfo（commName 需拷贝到持久缓冲，init 返回后 commInfo 指针失效） */
    ctx->commInfo = *commInfo;
    if (commInfo->commName != NULL) {
        if (snprintf_s(
                ctx->commNameBuf, sizeof(ctx->commNameBuf), sizeof(ctx->commNameBuf) - 1, "%s", commInfo->commName)
            < 0) {
            ctx->commNameBuf[0] = '\0';
        }
        ctx->commInfo.commName = ctx->commNameBuf;
    }
    if (g_hostFuncs.logFunction != NULL) {
        g_hostFuncs.logFunction(
            HCCL_TUNER_LOG_INFO, __FILE__, __LINE__, "tuner init done, comm[%p] nRanks[%u] opSetCount[%d]", comm,
            commInfo->nRanks, ctx->opSetCount);
    }
    return HCCL_SUCCESS;
}

static HcclResult MyGetCollInfo(
    HcclComm comm, const hcclTunerCollInfo_t* collInfo, hcclTunerAlgoEntry_t* entries, int count, int* matched)
{
    if (collInfo == NULL || entries == NULL || count <= 0 || matched == NULL) {
        return HCCL_E_PTR;
    }
    /* *matched 已被 HCCL 核心初始化为 0（未命中） */
    if (!g_hostFuncsReady || g_hostFuncs.ctxGet == NULL) {
        return HCCL_SUCCESS;
    }
    void* ctxPtr = NULL;
    uint64_t ctxSize = 0;
    if (g_hostFuncs.ctxGet(comm, "main", &ctxPtr, &ctxSize) != HCCL_SUCCESS || ctxPtr == NULL) {
        return HCCL_SUCCESS;
    }
    /* 校验 ctxGet 返回的 size，防止 engine bug 或版本不一致导致越界读 */
    if (ctxSize < sizeof(StoredHeader)) {
        if (g_hostFuncs.logFunction != NULL) {
            g_hostFuncs.logFunction(
                HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "ctxGet size %llu < expected %zu, skip intervention",
                (unsigned long long)ctxSize, sizeof(StoredHeader));
        }
        return HCCL_SUCCESS;
    }
    StoredHeader* ctx = (StoredHeader*)ctxPtr;

    /* Schema 校验失败时，不干预算法选择 */
    if (!ctx->configValid) {
        return HCCL_SUCCESS;
    }

    Rule* rules = GetRules(ctx);
    for (int i = 0; i < ctx->opSetCount; i++) {
        OpSetDesc* desc = &ctx->opSets[i];
        if (desc->opType != collInfo->collType) {
            continue;
        }
        for (int j = 0; j < desc->ruleCount; j++) { /* 首条命中即返回 */
            Rule* r = &rules[desc->ruleOffset + j];
            if (MatchRule(r, collInfo, &ctx->commInfo)) {
                if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
                    g_hostFuncs.logFunction(
                        HCCL_TUNER_LOG_INFO, __FILE__, __LINE__,
                        "[TunerDFX] rule hit: opType=%d nBytes=%zu dataType=%d ruleIdx=%d/%d "
                        "engine=%s executor=%s template=%s cost=%.6f",
                        (int)collInfo->collType, collInfo->nBytes, (int)collInfo->dataType, j, desc->ruleCount,
                        r->engine[0] ? r->engine : "*", r->executor[0] ? r->executor : "*",
                        r->templateName[0] ? r->templateName : "*", r->hasCost ? r->cost : 0.0f);
                }
                ApplyRule(r, entries, count);
                *matched = 1; /* 命中，设标志 */
                return HCCL_SUCCESS;
            }
        }
    }
    if (g_hostFuncsReady && g_hostFuncs.logFunction != NULL) {
        g_hostFuncs.logFunction(
            HCCL_TUNER_LOG_WARN, __FILE__, __LINE__, "[TunerDFX] no rule matched: opType=%d nBytes=%zu dataType=%d",
            (int)collInfo->collType, collInfo->nBytes, (int)collInfo->dataType);
    }
    return HCCL_SUCCESS;
}

/* ===== 插件函数表（V1，直接导出全局变量，核心通过 dlsym 获取）===== */
hcclTunerFuncs_v1_t hcclTunerPlugin_v1 = {MyInit, MyGetCollInfo, sizeof(hcclTunerFuncs_v1_t)};

/* 供 test_plugin.c 通过 #include "../plugin.c" 方式单元测试 */
#ifdef HCCL_TUNER_TESTING
StoredHeader* TunerGetStoredCtx(HcclComm comm)
{
    if (!g_hostFuncsReady || g_hostFuncs.ctxGet == NULL) {
        return NULL;
    }
    void* ctxPtr = NULL;
    uint64_t ctxSize = 0;
    if (g_hostFuncs.ctxGet(comm, "main", &ctxPtr, &ctxSize) != HCCL_SUCCESS) {
        return NULL;
    }
    return (StoredHeader*)ctxPtr;
}
#endif
