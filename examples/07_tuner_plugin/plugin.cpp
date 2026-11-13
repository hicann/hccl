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
 * （HcclTunerAlgoCostEntry[] 命名条目数组）以影响 Selector 算法选择。
 *
 * 入口：导出 HcclTunerPluginOpsGet（返回 const HcclTunerPluginOps*，纯静态初始化）。
 * 生命周期：Create 用标准 malloc 构造 tunerCtx 经出参返回，宿主拷贝进通信域 ctx 后
 *           free 临时缓冲；UpdateAlgoCostTable 直接收 tunerCtx。
 * 日志：自包含 log.h（printf），不再依赖宿主 logFunction 回调。
 *
 * 编译：make（链接为 hccl_tuner_example.so）
 * 使用：export HCCL_ALG_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
 *       export HCCL_ALG_CONFIG_FILE=/path/to/hccl_tuner_config.json
 */

#include <atomic>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <nlohmann/json.hpp>

#include "hccl_tuner_plugin.h"
#include "log.h"
#include "securec.h"

/* ===== 常量 ===== */
#define MAX_OP_TYPES 16 /* HCCL_CMDType 通信算子 12 种（含 v2 变体）+ 余量 */
#define MAX_TOTAL_RULES 5000
#define MAX_RULES_PER_OP 2000
#define MAX_STR_LEN 64
#define MAX_FILE_SIZE (4 * 1024 * 1024)
#define TUNER_CTX_READY_MAGIC 0x54554E43u /* "TUNC"：宿主 Copy 完成哨兵（StoredHeader.ctxReady） */

/* ===== 维度校验（engine 与 alg_parse.cc 的 ENGINE_TYPES key 保持一致；executor/template 不枚举校验）===== */
static const char* g_validEngines[] = {"aicpu", "ccums", "ccusched", "aiv", "dpu"};

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
    if (strcmp(name, "alltoall") == 0)
        return HCCL_CMD_ALLTOALL;
    if (strcmp(name, "alltoallv") == 0)
        return HCCL_CMD_ALLTOALLV;
    if (strcmp(name, "alltoallvc") == 0)
        return HCCL_CMD_ALLTOALLVC;
    if (strcmp(name, "reduce_scatterv") == 0)
        return HCCL_CMD_REDUCE_SCATTER_V;
    if (strcmp(name, "all_gatherv") == 0)
        return HCCL_CMD_ALLGATHER_V;
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
    int hasReduceOp;
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
    HcclReduceOp reduceOpEnum;
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
    int valid; /* 逐规则校验结果：1=有效，0=schema 剔除（不参与匹配） */
} Rule;

typedef struct {
    HcclCMDType opType;
    int ruleCount;
    int ruleOffset; /* 在扁平 CompactRule[] 中的起始下标 */
} OpSetDesc;

/* CompactRule：解析期编译产物，整条 64B 装进一条 cache line。
 * 15 个 has 标志消失——"是否配置"编码进哨兵值，比较指令天然完成通配：
 *   minBytes=0 / maxBytes=UINT64_MAX / dataTypeEnum=-1 时谓词恒不成立 = 通配。
 * 拓扑谓词（ranks/npus/servers/pods/superPods/commName/bufferSize）在 Create 编译期
 * 对照 commInfo/topoInfo 快照消费（旧 MatchRule 运行时用的就是该快照，编译期消费语义等价），
 * 失配规则整体剔除不落盘；engine/executor/templateName 以 uint32 偏移引用 ctx 尾部字符串区。
 * reduceOp 谓词（与 dataType 同款哨兵）：未配置→-1 通配；HcclReduceOp
 * 枚举全非负（RESERVED=255 仅作 parse 失败标记，非法规则已在 schema 阶段剔除）。 */
#define STR_OFF_WILDCARD 0xFFFFFFFFu /* 规则该维偏移哨兵（通配）：字符串区不存空串，有效偏移恒 < strAreaSize */
typedef struct {
    uint64_t minBytes;  /* 未配置→0：nBytes < 0 恒假 = 通配 */
    uint64_t maxBytes;  /* 未配置→UINT64_MAX：nBytes > MAX 恒假 = 通配 */
    uint32_t engineOff; /* ctx 字符串区偏移；STR_OFF_WILDCARD = 通配 */
    uint32_t executorOff;
    uint32_t templateOff;
    int32_t dataTypeEnum; /* 未配置→-1 通配；HcclDataType 枚举值全非负 */
    int32_t reduceOpEnum; /* 未配置→-1 通配；HcclReduceOp 枚举值全非负（RESERVED=255 不落盘） */
    float cost;           /* cost 为必填字段，有效规则恒有意义 */
    /* 终局语义（firsthit）下掩码/序号已退役：Update 只对首条谓词命中的规则做一次
     * ApplyRule strcmp。顶格 64B——单 cache line 预算不变。 */
    uint32_t reserved[6]; /* 预算余量 24B：再增字段将超单 cache line，需拆热/冷结构 */
} CompactRule;
static_assert(sizeof(CompactRule) == 64, "CompactRule must fit in one cache line");

/* per-comm 存储上下文（Create 时 malloc 构造，宿主 HcclEngineCtxCopy 进通信域后释放临时缓冲）。
 * 布局：StoredHeader + CompactRule[totalRuleCount] + 字符串区——
 * 3D 名驻留 ctx 尾部、规则以 uint32 偏移引用：裸 memcpy 后偏移天然有效（无需指针修正），
 * 通信域销毁时宿主回收 ctx、字符串随之释放——生命周期与通信域精确对齐，
 * .so 进程级池与发布/订阅并发面整体退役。 */
typedef struct {
    OpSetDesc opSets[MAX_OP_TYPES];
    int opSetCount;
    int totalRuleCount;      /* 编译后存活规则数（schema 有效且拓扑预筛通过） */
    int topoPrunedCount;     /* 拓扑预剔除数（失配规则运行时也永失配，等价剔除） */
    int schemaRejectedCount; /* schema 剔除数（坏规则只废自己） */
    HcclTunerCommInfo
        commInfo; /* bufferSize 快照（供诊断/测试观察）；commName 指针不持久化（防悬垂），内容在 commNameBuf */
    HcclTunerTopoInfo topoInfo; /* nRanks/nServers/... 快照 */
    char commNameBuf[COMM_NAME_BUF_LEN];
    uint32_t strAreaOff;  /* 字符串区在 ctx 内的字节偏移（紧随 CompactRule[] 之后） */
    uint32_t strAreaSize; /* 字符串区字节数（\0 分隔变长串，dedup 只增，Create 后不可变） */
    int configValid;      /* 配置级校验结果：1=有效可干预，0=fatal 整份拒绝 */
    uint32_t ctxReady; /* 落盘完成哨兵：宿主 EngineCtxCreate 后先清零、EngineCtxCopy 落盘后才带魔数
                        * ——Create 与 Copy 之间读到的未就绪 ctx 由 Update 首查兜底为 no-op */
} StoredHeader;

/* 访问 StoredHeader 后面的变长 CompactRule[] */
static inline CompactRule* GetRules(StoredHeader* ctx)
{
    return reinterpret_cast<CompactRule*>(reinterpret_cast<char*>(ctx) + sizeof(StoredHeader));
}

/* 访问 ctx 字符串区内的一条串（偏移来自 CompactRule 的 *Off 字段；通配返回 ""） */
static inline const char* CtxStr(const StoredHeader* ctx, uint32_t off)
{
    return (off == STR_OFF_WILDCARD) ? "" : (const char*)ctx + ctx->strAreaOff + off;
}

/* 字符串区：3D 名 dedup 驻留 ctx 尾部，规则以 uint32 偏移引用。
 * 生命周期随通信域（宿主回收 ctx 即回收），跨 comm 不共享、Create 后不可变——
 * 单写者（编译循环，宿主 g_tunerInitMutex 串行），无并发面。
 * 预算 MAX_STR_AREA=16KB：典型唯一 3D 名 <1KB（算法注册表名字集合），
 * 超限返回失败由调用方剔除该规则——报错优于静默通配。 */
#define MAX_STR_AREA (16u * 1024u)
typedef struct {
    char buf[MAX_STR_AREA];
    uint32_t used;
} StrAreaBuilder;

/* 落区（编译期，单线程）：dedup 追加，仅产出偏移。空串→WILDCARD 哨兵不占区；
 * 区满→返回 false（调用方剔除该规则，不静默通配）。 */
static bool CtxIntern(StrAreaBuilder* sa, const char* s, uint32_t* offOut)
{
    if (s == NULL || s[0] == '\0') {
        *offOut = STR_OFF_WILDCARD;
        return true;
    }
    for (uint32_t off = 0; off < sa->used; off += (uint32_t)strlen(sa->buf + off) + 1u) {
        if (strcmp(sa->buf + off, s) == 0) {
            *offOut = off;
            return true;
        }
    }
    const size_t len = strlen(s) + 1u;
    if (sa->used + (uint32_t)len > MAX_STR_AREA) {
        return false;
    }
    if (memcpy_s(sa->buf + sa->used, len, s, len) != EOK) {
        return false;
    }
    *offOut = sa->used;
    sa->used += (uint32_t)len;
    return true;
}

/* Schema 校验状态（per-Create 局部变量，避免并发 comm init 竞争） */
typedef struct {
    int errors;
    int warnings;
    int fatal; /* 配置级致命错误（root/version/op_types）：格式不可信，整份拒绝 */
    int foundVersion;
    int foundOpTypes;
} SchemaState;

#ifdef HCCL_TUNER_TESTING
static SchemaState g_lastSchema = {}; /* 供测试验证 schema 校验结果 */
static Rule* g_lastParseRules = NULL; /* 供测试观察 parse 级 valid/MatchCond（生产路径无此开销） */
static int g_lastParseRuleCount = 0;
static int g_applyRuleCalls = 0; /* 供测试断言谓词命中规则确实进入 ApplyRule（终局制下每 op 恰一次） */
#endif

static void SchemaWarn(SchemaState* s, const char* scope, const char* field)
{
    s->warnings++;
    HCCL_WARNING("Schema: unknown field '%s' in %s, skipping", field, scope);
}

static void SchemaError(SchemaState* s, const char* fmt, ...)
{
    s->errors++;
    va_list args;
    va_start(args, fmt);
    char buf[256] = {0};
    if (vsnprintf_s(buf, sizeof(buf), sizeof(buf) - 1, fmt, args) < 0) {
        buf[0] = '\0';
    }
    va_end(args);
    HCCL_WARNING("Schema: %s", buf);
}

/* 配置级致命错误：格式不可信，整份拒绝（区别于规则级 error 的逐条剔除）。 */
static void SchemaFatal(SchemaState* s, const char* fmt, ...)
{
    s->errors++;
    s->fatal++;
    va_list args;
    va_start(args, fmt);
    char buf[256] = {0};
    if (vsnprintf_s(buf, sizeof(buf), sizeof(buf) - 1, fmt, args) < 0) {
        buf[0] = '\0';
    }
    va_end(args);
    HCCL_WARNING("Schema: %s", buf);
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

/* ===== reduce 算子字符串映射 ===== */
static const struct {
    const char* name;
    HcclReduceOp op;
} g_reduceOpMap[] = {
    {"sum", HCCL_REDUCE_SUM},
    {"prod", HCCL_REDUCE_PROD},
    {"max", HCCL_REDUCE_MAX},
    {"min", HCCL_REDUCE_MIN},
};

static HcclReduceOp ParseReduceOp(const char* name)
{
    for (size_t i = 0; i < sizeof(g_reduceOpMap) / sizeof(g_reduceOpMap[0]); i++) {
        if (strcmp(name, g_reduceOpMap[i].name) == 0) {
            return g_reduceOpMap[i].op;
        }
    }
    return HCCL_REDUCE_RESERVED;
}

/* 维度值校验：engine 由插件自维护的 is_valid_engine 枚举校验；executor/template 不枚举校验
 * （合法集合随 HCCL 版本演进，固定枚举会拒绝新增值，如 executor 的 strictordered），拼错靠运行时 warning 兜底。 */

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

/* 整数匹配字段值域校验（Issue #738）：nlohmann 的 get<T>() 对窄化只做 static_cast（模运算截断），
 * 溢出/负数/超 u64 均静默通过。转换前统一三查：无符号整数类型、非负、不超目标类型上界；
 * 失败计入 SchemaError（走逐规则剔除），out 保持不变。 */
static int GetU32Checked(const nlohmann::json& obj, const char* field, uint32_t& out, SchemaState* s)
{
    const auto& v = obj[field];
    if (!v.is_number_unsigned() || v.get<uint64_t>() > 0xFFFFFFFFULL) {
        SchemaError(s, "field '%s' must be a non-negative integer within uint32 range", field);
        return -1;
    }
    out = v.get<uint32_t>();
    return 0;
}

static int GetU64Checked(const nlohmann::json& obj, const char* field, uint64_t& out, SchemaState* s)
{
    const auto& v = obj[field];
    if (!v.is_number_unsigned()) {
        /* number_unsigned 已保证非负且 <= UINT64_MAX：更大值被解析为 double，负数为 number_integer */
        SchemaError(s, "field '%s' must be a non-negative integer within uint64 range", field);
        return -1;
    }
    out = v.get<uint64_t>();
    return 0;
}

static void ParseMatchField(const nlohmann::json& matchObj, Rule* r, SchemaState* s)
{
    static const char* knownKeys[]
        = {"min_ranks",           "max_ranks",           "min_bytes",   "max_bytes",   "data_type", "comm_name",
           "min_npus_per_server", "max_npus_per_server", "min_servers", "max_servers", "min_pods",  "max_pods",
           "min_super_pods",      "max_super_pods",      "buffer_size", "reduce_op"};
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
        (void)GetU32Checked(matchObj, "min_ranks", r->match.minRanks, s);
    }
    if (matchObj.contains("max_ranks")) {
        r->match.hasMaxRanks = 1;
        (void)GetU32Checked(matchObj, "max_ranks", r->match.maxRanks, s);
    }
    if (matchObj.contains("min_bytes")) {
        r->match.hasMinBytes = 1;
        (void)GetU64Checked(matchObj, "min_bytes", r->match.minBytes, s);
    }
    if (matchObj.contains("max_bytes")) {
        r->match.hasMaxBytes = 1;
        (void)GetU64Checked(matchObj, "max_bytes", r->match.maxBytes, s);
    }
    if (matchObj.contains("data_type")) {
        if (!matchObj["data_type"].is_string()) {
            /* 类型守卫：非字符串直接 schema 剔除该规则——裸 get 抛 type_error 会中止整份解析，
             * 违反"逐规则剔除、坏规则只废自己"契约 */
            SchemaError(s, "data_type must be a string");
        } else {
            if (snprintf_s(
                    r->match.dataType, sizeof(r->match.dataType), sizeof(r->match.dataType) - 1, "%s",
                    matchObj["data_type"].get<std::string>().c_str())
                < 0) {
                r->match.dataType[0] = '\0';
            }
            r->match.dataTypeEnum = ParseDataType(r->match.dataType);
            if (r->match.dataTypeEnum == HCCL_DATA_TYPE_RESERVED) {
                SchemaError(s, "invalid data_type '%s'", r->match.dataType);
            }
            r->match.hasDataType = 1;
        }
    }
    if (matchObj.contains("reduce_op")) {
        if (!matchObj["reduce_op"].is_string()) {
            SchemaError(s, "reduce_op must be a string");
        } else {
            std::string buf = matchObj["reduce_op"].get<std::string>();
            r->match.reduceOpEnum = ParseReduceOp(buf.c_str());
            if (r->match.reduceOpEnum == HCCL_REDUCE_RESERVED) {
                SchemaError(s, "invalid reduce_op '%s'", buf.c_str());
            }
            r->match.hasReduceOp = 1;
        }
    }
    if (matchObj.contains("comm_name")) {
        if (!matchObj["comm_name"].is_string()) {
            SchemaError(s, "comm_name must be a string");
        } else if (
            snprintf_s(
                r->match.commName, sizeof(r->match.commName), sizeof(r->match.commName) - 1, "%s",
                matchObj["comm_name"].get<std::string>().c_str())
            < 0) {
            /* 截断若静默置空 → hasCommName=1 + 空串 strstr 恒命中 = 限定条件反转为通配
             * ——改报 schema error 剔除该规则，对齐 template 超长的防御风格 */
            SchemaError(s, "comm_name too long (max %d chars)", MAX_STR_LEN - 1);
        } else {
            r->match.hasCommName = 1;
        }
    }
    if (matchObj.contains("min_npus_per_server")) {
        r->match.hasMinNpusPerServer = 1;
        (void)GetU32Checked(matchObj, "min_npus_per_server", r->match.minNpusPerServer, s);
    }
    if (matchObj.contains("max_npus_per_server")) {
        r->match.hasMaxNpusPerServer = 1;
        (void)GetU32Checked(matchObj, "max_npus_per_server", r->match.maxNpusPerServer, s);
    }
    if (matchObj.contains("min_servers")) {
        r->match.hasMinServers = 1;
        (void)GetU32Checked(matchObj, "min_servers", r->match.minServers, s);
    }
    if (matchObj.contains("max_servers")) {
        r->match.hasMaxServers = 1;
        (void)GetU32Checked(matchObj, "max_servers", r->match.maxServers, s);
    }
    if (matchObj.contains("min_pods")) {
        r->match.hasMinPods = 1;
        (void)GetU32Checked(matchObj, "min_pods", r->match.minPods, s);
    }
    if (matchObj.contains("max_pods")) {
        r->match.hasMaxPods = 1;
        (void)GetU32Checked(matchObj, "max_pods", r->match.maxPods, s);
    }
    if (matchObj.contains("min_super_pods")) {
        r->match.hasMinSuperPods = 1;
        (void)GetU32Checked(matchObj, "min_super_pods", r->match.minSuperPods, s);
    }
    if (matchObj.contains("max_super_pods")) {
        r->match.hasMaxSuperPods = 1;
        (void)GetU32Checked(matchObj, "max_super_pods", r->match.maxSuperPods, s);
    }
    if (matchObj.contains("buffer_size")) {
        r->match.hasBufferSize = 1;
        (void)GetU64Checked(matchObj, "buffer_size", r->match.bufferSize, s);
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
        if (!ruleObj["engine"].is_string()) {
            SchemaError(s, "engine must be a string");
        } else {
            std::string buf = ruleObj["engine"].get<std::string>();
            if (!is_valid_engine(buf.c_str())) {
                SchemaError(s, "invalid engine '%s'", buf.c_str());
            } else {
                if (snprintf_s(r->engine, sizeof(r->engine), sizeof(r->engine) - 1, "%s", buf.c_str()) < 0) {
                    r->engine[0] = '\0';
                }
            }
        }
    }
    if (ruleObj.contains("executor")) {
        if (!ruleObj["executor"].is_string()) {
            SchemaError(s, "executor must be a string");
        } else {
            std::string buf = ruleObj["executor"].get<std::string>();
            /* executor 不做枚举校验：合法集合随 HCCL 版本演进（strictordered 为后增），
             * 固定枚举会拒绝新增值；拼错靠 ApplyRule 运行时 "no entry modified" warning 兜底。 */
            if (buf.empty()) {
                SchemaError(s, "invalid executor '%s' (empty)", buf.c_str());
            } else {
                if (snprintf_s(r->executor, sizeof(r->executor), sizeof(r->executor) - 1, "%s", buf.c_str()) < 0) {
                    r->executor[0] = '\0';
                }
            }
        }
    }
    if (ruleObj.contains("template")) {
        if (!ruleObj["template"].is_string()) {
            SchemaError(s, "template must be a string");
        } else {
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
    }
    if (ruleObj.contains("cost")) {
        if (!ruleObj["cost"].is_number()) {
            /* 同字符串字段：裸 get 抛 type_error 会中止整份解析，坏值只废这条规则 */
            SchemaError(s, "cost must be a number");
        } else {
            r->cost = ruleObj["cost"].get<float>();
            r->hasCost = 1;
        }
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
            int errBefore = s->errors;
            ParseRule(ruleItem, &rules[*curOffset], s);
            /* 逐规则剔除：本条解析无新增 error 才有效，坏规则只废自己 */
            rules[*curOffset].valid = (s->errors == errBefore);
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
        SchemaFatal(s, "config root is not an object");
        return;
    }

    if (root.contains("version")) {
        s->foundVersion = 1;
        int ver = root["version"].get<int>();
        if (ver != 1) {
            SchemaFatal(s, "version must be 1, got %d", ver);
        }
    } else {
        SchemaFatal(s, "missing required field 'version'");
    }

    if (root.contains("op_types") && root["op_types"].is_object()) {
        s->foundOpTypes = 1;
        for (auto it = root["op_types"].begin(); it != root["op_types"].end(); ++it) {
            ParseOpRules(it.value(), ctx, rules, curOffset, it.key().c_str(), s);
        }
    } else {
        SchemaFatal(s, "missing required field 'op_types'");
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

/* ===== 编译期拓扑消费 =====
 * 旧 MatchRule 的拓扑分支搬进 Create：对照 commInfo/topoInfo 快照一次性消费，
 * 失配规则整体剔除不落 ctx（运行时也永失配，等价）。
 * bytes/dataType 谓词编译进 CompactRule 哨兵值，运行时内联比较。 */
static int MatchTopology(const MatchCond* m, const HcclTunerCommInfo* commInfo, const HcclTunerTopoInfo* topoInfo)
{
    if (m->hasMinRanks && topoInfo->nRanks < m->minRanks) {
        return 0;
    }
    if (m->hasMaxRanks && topoInfo->nRanks > m->maxRanks) {
        return 0;
    }
    if (m->hasCommName) {
        if (commInfo->commName == NULL || strstr(commInfo->commName, m->commName) == NULL) {
            return 0;
        }
    }
    if (m->hasMinNpusPerServer && topoInfo->nNpusPerServer < m->minNpusPerServer) {
        return 0;
    }
    if (m->hasMaxNpusPerServer && topoInfo->nNpusPerServer > m->maxNpusPerServer) {
        return 0;
    }
    if (m->hasMinServers && topoInfo->nServers < m->minServers) {
        return 0;
    }
    if (m->hasMaxServers && topoInfo->nServers > m->maxServers) {
        return 0;
    }
    if (m->hasMinPods && topoInfo->nPods < m->minPods) {
        return 0;
    }
    if (m->hasMaxPods && topoInfo->nPods > m->maxPods) {
        return 0;
    }
    if (m->hasMinSuperPods && topoInfo->nSuperPods < m->minSuperPods) {
        return 0;
    }
    if (m->hasMaxSuperPods && topoInfo->nSuperPods > m->maxSuperPods) {
        return 0;
    }
    if (m->hasBufferSize && commInfo->bufferSize != m->bufferSize) {
        return 0;
    }
    return 1;
}

/* 返回: 实际修改 cost 的条目数（禁用条目跳过不计）。返回 0 时调用方终局（后续规则不再查看）。
 * 遍历按 entries[0].header.size 步长（ABI 约定，编译期 sizeof 下标版本不安全）。
 * bytes/dataType 已由哨兵值在扫描循环内联比较，此处只做 3D 名通配匹配。 */
static int ApplyRule(const StoredHeader* ctx, const CompactRule* r, HcclTunerAlgoCostEntry* entries, uint32_t count)
{
#ifdef HCCL_TUNER_TESTING
    g_applyRuleCalls++; /* 测试观察点：谓词命中的规则到达此处 */
#endif
    int modified = 0;
    if (entries == NULL || count == 0 || entries[0].header.size < sizeof(HcclTunerAlgoCostEntry)) {
        /* 步长不可信（宿主未填 header 或条目小于本插件预期布局），放弃修改；调用方终局 */
        return 0;
    }
    const uint32_t stride = entries[0].header.size;
    const char* rEngine = CtxStr(ctx, r->engineOff);
    const char* rExecutor = CtxStr(ctx, r->executorOff);
    const char* rTemplate = CtxStr(ctx, r->templateOff);
    for (uint32_t i = 0; i < count; i++) {
        HcclTunerAlgoCostEntry* e = (HcclTunerAlgoCostEntry*)((const char*)entries + (size_t)i * stride);
        if (r->engineOff != STR_OFF_WILDCARD && (e->engineName == NULL || strcmp(rEngine, e->engineName) != 0))
            continue;
        if (r->executorOff != STR_OFF_WILDCARD && (e->executorName == NULL || strcmp(rExecutor, e->executorName) != 0))
            continue;
        if (r->templateOff != STR_OFF_WILDCARD && (e->templateName == NULL || strcmp(rTemplate, e->templateName) != 0))
            continue;
        if (e->cost < 0.0f) {
            HCCL_WARNING("[TunerDFX] skip disabled: algName=%s cost=%.6f", e->algName ? e->algName : "?", e->cost);
            continue;
        }
        float oldCost = e->cost;
        e->cost = r->cost;
        HCCL_INFO(
            "[TunerDFX] modify: algName=%s engine=%s executor=%s template=%s cost %.6f -> %.6f",
            e->algName ? e->algName : "?", e->engineName ? e->engineName : "?", e->executorName ? e->executorName : "?",
            e->templateName ? e->templateName : "?", oldCost, e->cost);
        modified++;
    }
    if (modified == 0) {
        /* 命中但 0 改动（终局点，DEBUG 留痕；终局告警由调用方 0-modified 分支 WARN 打印） */
        HCCL_DEBUG(
            "[TunerDFX] rule matched but no entry modified: engine=%s executor=%s template=%s cost=%.6f",
            (r->engineOff != STR_OFF_WILDCARD) ? rEngine : "*", (r->executorOff != STR_OFF_WILDCARD) ? rExecutor : "*",
            (r->templateOff != STR_OFF_WILDCARD) ? rTemplate : "*", r->cost);
    }
    return modified;
}

/* ===== 插件接口实现 ===== */
static HcclResult MyCreate(
    HcclComm comm, const HcclTunerCommInfo* commInfo, const HcclTunerTopoInfo* topoInfo, void** tunerCtx,
    uint32_t* tunerCtxSize)
{
    if (commInfo == NULL || topoInfo == NULL || tunerCtx == NULL || tunerCtxSize == NULL) {
        return HCCL_E_PTR;
    }
    *tunerCtx = NULL;
    *tunerCtxSize = 0;

    /* 1. 读文件 + 解析 JSON（临时内存树，不持久化） */
    const char* envPath = getenv("HCCL_ALG_CONFIG_FILE");
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
        HCCL_WARNING("tuner config '%s' not found%s", paths[i], (i == 0) ? ", falling back to default paths" : "");
    }
    if (content == NULL) {
        HCCL_WARNING("no tuner config loaded, plugin inactive");
        return HCCL_SUCCESS; /* inactive：空 ctx，宿主跳过 acquire/copy */
    }

    nlohmann::json root = nlohmann::json::parse(content, nullptr, false);
    if (root.is_discarded()) {
        HCCL_WARNING("JSON parse failed for %s", loadedPath);
        free(content);
        return HCCL_SUCCESS; /* inactive */
    }
    free(content);

    /* 2. 第一遍：遍历 JSON 树计数规则数 */
    int opSetCount = 0;
    int totalRules = CountRules(root, &opSetCount);
    if (totalRules > MAX_TOTAL_RULES) {
        HCCL_WARNING("total rules %d exceeds MAX_TOTAL_RULES(%d), plugin inactive", totalRules, MAX_TOTAL_RULES);
        return HCCL_SUCCESS; /* inactive */
    }

    /* 3. malloc 解析期临时缓冲（Rule[] 宽格式，编译后整体丢弃）；
     *    ParseOpRules 以 StoredHeader* 为落点, 用栈上临时 header 承接 opSets 计数 */
    Rule* tempRules = (Rule*)calloc((size_t)totalRules, sizeof(Rule));
    if (tempRules == NULL) {
        HCCL_WARNING("malloc failed for parse rules, count=%d", totalRules);
        return HCCL_SUCCESS; /* inactive */
    }
    StoredHeader parseHdr = {}; /* 仅 opSets/opSetCount/totalRuleCount 三个字段被 Parse 层使用 */

    /* 4. 第二遍：填充 Rule[]（宽格式，含 15 个 has 标志与拓扑谓词） */
    SchemaState schema = {};
    int curOffset = 0;
    try {
        ParseConfig(root, &parseHdr, tempRules, &curOffset, &schema);
    } catch (const nlohmann::json::exception& e) {
        SchemaError(&schema, "JSON field access error: %s", e.what());
    }
    const int parsedRuleCount = curOffset;
    const int tempOpSetCount = parseHdr.opSetCount;
#ifdef HCCL_TUNER_TESTING
    g_lastSchema = schema;
    if (g_lastParseRules != NULL) {
        free(g_lastParseRules);
    }
    g_lastParseRules = tempRules;
    g_lastParseRuleCount = parsedRuleCount;
#endif
    int validRules = 0;
    for (int i = 0; i < parsedRuleCount; i++) {
        if (tempRules[i].valid) {
            validRules++;
        }
    }
    HCCL_INFO(
        "tuner config loaded from %s, opSetCount=%d, totalRules=%d, validRules=%d, schemaErrors=%d, "
        "schemaWarnings=%d",
        loadedPath, tempOpSetCount, parsedRuleCount, validRules, schema.errors, schema.warnings);

    /* 5. 编译：valid 且拓扑匹配的规则保序压缩进 CompactRule[]，
     *    3D 名 dedup 入 .so 进程级池；拓扑谓词在此消费，失配规则不落 ctx */
    CompactRule* tempCompact = (CompactRule*)calloc((size_t)validRules, sizeof(CompactRule));
    OpSetDesc newOpSets[MAX_OP_TYPES] = {};
    if (tempCompact == NULL) {
        HCCL_WARNING("compile alloc failed, plugin inactive");
#ifndef HCCL_TUNER_TESTING
        free(tempRules);
#endif
        return HCCL_SUCCESS;
    }
    int newOpSetCount = 0;
    int active = 0;
    int topoPruned = 0;
    int strRejected = 0;
    StrAreaBuilder strArea;
    strArea.used = 0;
    for (int i = 0; i < tempOpSetCount; i++) {
        const OpSetDesc* src = &parseHdr.opSets[i];
        newOpSets[i].opType = src->opType;
        newOpSets[i].ruleOffset = active;
        for (int j = 0; j < src->ruleCount; j++) {
            const Rule* tr = &tempRules[src->ruleOffset + j];
            if (!tr->valid) {
                continue; /* schema 剔除的规则不落盘 */
            }
            if (!MatchTopology(&tr->match, commInfo, topoInfo)) {
                topoPruned++;
                continue; /* 拓扑失配规则不落盘（运行时也永失配，等价） */
            }
            CompactRule* cr = &tempCompact[active];
            cr->minBytes = tr->match.hasMinBytes ? (uint64_t)tr->match.minBytes : 0ULL;
            cr->maxBytes = tr->match.hasMaxBytes ? (uint64_t)tr->match.maxBytes : UINT64_MAX;
            cr->dataTypeEnum = tr->match.hasDataType ? (int32_t)tr->match.dataTypeEnum : -1;
            cr->reduceOpEnum = tr->match.hasReduceOp ? (int32_t)tr->match.reduceOpEnum : -1;
            /* 3D 名落 ctx 字符串区：偏移供 ApplyRule strcmp；
             * "" 通配 → 哨兵不占区。区满 → 剔除该规则（报错优于静默通配）。 */
            if (!CtxIntern(&strArea, tr->engine, &cr->engineOff) || !CtxIntern(&strArea, tr->executor, &cr->executorOff)
                || !CtxIntern(&strArea, tr->templateName, &cr->templateOff)) {
                strRejected++;
                HCCL_WARNING("str area exhausted (max %uB), rule rejected", (unsigned)MAX_STR_AREA);
                continue;
            }
            cr->cost = tr->hasCost ? tr->cost : 0.0f;
            active++;
        }
        newOpSets[i].ruleCount = active - newOpSets[i].ruleOffset; /* 保序：幸存规则相对顺序 = JSON 顺序 */
    }
    newOpSetCount = tempOpSetCount;

    /* 6. malloc 最终 ctx（精确大小：Header + active × CompactRule + 字符串区——
     *    同源计算：尺寸与写入出自同一 strArea.used，无两处维护可漂移） */
    const uint32_t strAreaOff = (uint32_t)(sizeof(StoredHeader) + (uint64_t)active * sizeof(CompactRule));
    uint64_t ctxSize = (uint64_t)strAreaOff + strArea.used;
    void* tmp = malloc((size_t)ctxSize);
    if (tmp == NULL) {
        HCCL_WARNING("malloc failed for tuner ctx, size=%llu", (unsigned long long)ctxSize);
        free(tempCompact);
#ifndef HCCL_TUNER_TESTING
        free(tempRules);
#endif
        return HCCL_SUCCESS; /* inactive */
    }
    if (memset_s(tmp, (size_t)ctxSize, 0, (size_t)ctxSize) != EOK) {
        HCCL_WARNING("memset_s failed for tuner ctx");
        free(tmp);
        free(tempCompact);
#ifndef HCCL_TUNER_TESTING
        free(tempRules);
#endif
        return HCCL_SUCCESS; /* inactive */
    }
    StoredHeader* ctx = (StoredHeader*)tmp;
    ctx->opSetCount = newOpSetCount;
    for (int i = 0; i < newOpSetCount; i++) {
        ctx->opSets[i] = newOpSets[i];
    }
    ctx->totalRuleCount = active;
    ctx->topoPrunedCount = topoPruned;
    ctx->schemaRejectedCount = parsedRuleCount - validRules;
    ctx->strAreaOff = strAreaOff;
    ctx->strAreaSize = strArea.used;
    CompactRule* dstRules = GetRules(ctx);
    for (int i = 0; i < active; i++) {
        dstRules[i] = tempCompact[i]; /* 3D 名为字符串区偏移：随 ctx 裸 memcpy 天然有效 */
    }
    if (strArea.used > 0 && memcpy_s((char*)ctx + strAreaOff, strArea.used, strArea.buf, strArea.used) != EOK) {
        HCCL_WARNING("memcpy_s failed for str area, plugin inactive");
        free(tmp);
        free(tempCompact);
#ifndef HCCL_TUNER_TESTING
        free(tempRules);
#endif
        return HCCL_SUCCESS; /* inactive */
    }
    if (schema.fatal > 0) {
        /* 仅配置级致命错误整体拒绝；规则级错误已逐条剔除 */
        HCCL_WARNING("Schema fatal error (%d), plugin will not intervene", schema.fatal);
        ctx->configValid = 0;
    } else {
        ctx->configValid = 1;
    }
    HCCL_INFO(
        "compiled: %d active, %d topo-pruned, %d schema-rejected, %d str-rejected, strArea=%uB, "
        "sizeof(Rule)=%zu -> sizeof(CompactRule)=%zu, ctx=%lluB (legacy layout %lluB)",
        active, topoPruned, parsedRuleCount - validRules, strRejected, strArea.used, sizeof(Rule), sizeof(CompactRule),
        (unsigned long long)ctxSize,
        (unsigned long long)(sizeof(StoredHeader) + (uint64_t)parsedRuleCount * sizeof(Rule)));
    free(tempCompact);
#ifndef HCCL_TUNER_TESTING
    /* 解析期宽格式缓冲已编译完毕，成功路径同样必须释放——仅失败路径成对释放会漏 free，
     * 生产构建每次 Create 泄漏 totalRules×~350B（上限 ~1.7MB） */
    free(tempRules);
#endif

    /* 5. 拷贝 commInfo + topoInfo。commName 内容自持于内联缓冲 commNameBuf（随 ctx 裸
     * memcpy 天然有效，消费者直接读 commNameBuf）；commInfo.commName 指针不持久化——
     * 绝对地址经宿主 memcpy 进通信域后悬垂，置 NULL 防误用。 */
    ctx->commInfo = *commInfo;
    ctx->commInfo.commName = NULL;
    ctx->topoInfo = *topoInfo;
    if (commInfo->commName != NULL) {
        if (snprintf_s(
                ctx->commNameBuf, sizeof(ctx->commNameBuf), sizeof(ctx->commNameBuf) - 1, "%s", commInfo->commName)
            < 0) {
            ctx->commNameBuf[0] = '\0';
        }
    }

    /* 6. 出参返回，宿主 HcclEngineCtxCopy 进通信域 ctx 后 free(tmp) */
    ctx->ctxReady = TUNER_CTX_READY_MAGIC; /* 收尾最后置位：宿主 Copy 落盘完成的哨兵随数据一起拷入 */
    *tunerCtx = tmp;
    *tunerCtxSize = (uint32_t)ctxSize;
    HCCL_INFO(
        "tuner Create done, comm[%p] nRanks[%u] opSetCount[%d] ctxSize[%u]", comm, topoInfo->nRanks, ctx->opSetCount,
        *tunerCtxSize);
    return HCCL_SUCCESS;
}

static HcclResult MyUpdateAlgoCostTable(
    void* tunerCtx, const HcclTunerCollOpInfo* collOpInfo, HcclTunerAlgoCostEntry* entries, uint32_t algoCount,
    bool* matched)
{
    if (collOpInfo == NULL || entries == NULL || algoCount == 0 || matched == NULL) {
        return HCCL_E_PTR;
    }
    /* *matched 已被 HCCL 核心初始化为 false（未命中） */
    if (tunerCtx == NULL) {
        return HCCL_SUCCESS; /* 插件 inactive，不干预 */
    }
    StoredHeader* ctx = (StoredHeader*)tunerCtx; /* 宿主传入通信域 ctx，无需 ctxGet */

    /* 就绪哨兵：宿主 EngineCtxCreate 与 EngineCtxCopy 之间存在对 op 路径可见的窗口（同 comm
     * 并发首 op），窗口内 ctx 已清零、魔数未落盘——确定性 no-op，不改价、不按脏数据解析 */
    if (ctx->ctxReady != TUNER_CTX_READY_MAGIC) {
        return HCCL_SUCCESS;
    }

    /* 头部一致性 sanity（新 ABI 不传 ctxSize，旧版尺寸防线无处安放）：半拷贝/越界头部在此
     * 现形为 no-op，不改价、不越界；与宿主 Create 后清零 + Copy 落盘哨兵构成纵深防御。
     * strAreaOff 为构造不变量（Create 侧同公式，见 :971），垃圾/半拷贝数据极难同时碰撞
     * 全部交叉校验 */
    if (ctx->opSetCount < 0 || ctx->opSetCount > MAX_OP_TYPES || ctx->totalRuleCount < 0
        || ctx->totalRuleCount > MAX_TOTAL_RULES || ctx->strAreaSize > MAX_STR_AREA
        || ctx->strAreaOff != (uint32_t)(sizeof(StoredHeader) + (uint64_t)ctx->totalRuleCount * sizeof(CompactRule))) {
        HCCL_WARNING(
            "[TunerDFX] ctx header inconsistent (opSetCount=%d totalRuleCount=%d strAreaSize=%u strAreaOff=%u), "
            "skip.",
            ctx->opSetCount, ctx->totalRuleCount, ctx->strAreaSize, ctx->strAreaOff);
        return HCCL_SUCCESS;
    }

    /* 配置级致命错误（configValid=0）时不干预；规则级错误已在 Create 时逐条剔除 */
    if (!ctx->configValid) {
        return HCCL_SUCCESS;
    }

    CompactRule* rules = GetRules(ctx);

    /* 先确认本 opType 配置了规则组：无组 = tuner 未调该算子，正常态——
     * 静默早退，未配置算子无需逐规则扫描。
     * 诊断可见性走 DEBUG（诊断版 .so 重编 LOG_LEVEL_DEBUG 可见）。 */
    bool opSetPresent = false;
    for (int i = 0; i < ctx->opSetCount; i++) {
        if (ctx->opSets[i].opType == collOpInfo->collType) {
            opSetPresent = true;
            break;
        }
    }
    if (!opSetPresent) {
        HCCL_DEBUG("[TunerDFX] no rule group for opType=%d, skip silently", (int)collOpInfo->collType);
        return HCCL_SUCCESS;
    }

    for (int i = 0; i < ctx->opSetCount; i++) {
        OpSetDesc* desc = &ctx->opSets[i];
        if (desc->opType != collOpInfo->collType) {
            continue;
        }
        /* 内层边界（与头部 sanity 构成完整自证链）：ruleOffset/ruleCount 越界即跳过该组，
         * rules[] 下标全部有界 */
        if (desc->ruleOffset < 0
            || (uint64_t)desc->ruleOffset + (uint64_t)desc->ruleCount > (uint64_t)ctx->totalRuleCount) {
            HCCL_WARNING(
                "[TunerDFX] rule set inconsistent (opType=%d ruleOffset=%d ruleCount=%d), skip this set.",
                (int)desc->opType, desc->ruleOffset, desc->ruleCount);
            continue;
        }
        for (int j = 0; j < desc->ruleCount; j++) { /* 找到首条谓词命中的规则即终局 */
            const CompactRule* r = &rules[desc->ruleOffset + j];
            /* 哨兵通配：未配置的谓词编码成恒不成立的比较，编译期已消费拓扑谓词 */
            if (collOpInfo->nBytes < r->minBytes || collOpInfo->nBytes > r->maxBytes) {
                continue;
            }
            if (r->dataTypeEnum >= 0 && r->dataTypeEnum != (int32_t)collOpInfo->dataType) {
                continue;
            }
            if (r->reduceOpEnum >= 0 && r->reduceOpEnum != (int32_t)collOpInfo->reduceOp) {
                continue;
            }
            /* 终局语义（firsthit）：首条谓词命中的规则独占本次 op——
             * 名字匹配对这一条规则尽力而为：命中则改价；目标缺席/禁用则什么都不改。
             * 无论哪种结果，后续规则不再查看（多引擎让路模式已退役，需让路的配置请按引擎拆分规则）。 */
            int modified = ApplyRule(ctx, r, entries, algoCount);
            if (modified > 0) {
                HCCL_INFO(
                    "[TunerDFX] rule hit: opType=%d nBytes=%llu dataType=%d ruleIdx=%d/%d "
                    "engine=%s executor=%s template=%s cost=%.6f",
                    (int)collOpInfo->collType, (unsigned long long)collOpInfo->nBytes, (int)collOpInfo->dataType, j,
                    desc->ruleCount, (r->engineOff != STR_OFF_WILDCARD) ? CtxStr(ctx, r->engineOff) : "*",
                    (r->executorOff != STR_OFF_WILDCARD) ? CtxStr(ctx, r->executorOff) : "*",
                    (r->templateOff != STR_OFF_WILDCARD) ? CtxStr(ctx, r->templateOff) : "*", r->cost);
                *matched = true; /* 真正修改了 cost 才算命中，对齐头文件契约 */
                return HCCL_SUCCESS;
            }
            /* 本条规则谓词命中但 0 改动（目标缺席或全禁用）：终局，后续规则不再查看 */
            HCCL_WARNING(
                "[TunerDFX] first applicable rule 0 modified (target absent or disabled): opType=%d nBytes=%llu "
                "dataType=%d ruleIdx=%d/%d engine=%s executor=%s template=%s (terminate, later rules not examined)",
                (int)collOpInfo->collType, (unsigned long long)collOpInfo->nBytes, (int)collOpInfo->dataType, j,
                desc->ruleCount, (r->engineOff != STR_OFF_WILDCARD) ? CtxStr(ctx, r->engineOff) : "*",
                (r->executorOff != STR_OFF_WILDCARD) ? CtxStr(ctx, r->executorOff) : "*",
                (r->templateOff != STR_OFF_WILDCARD) ? CtxStr(ctx, r->templateOff) : "*");
            return HCCL_SUCCESS;
        }
    }
    /* 谓词无命中（区间外/dataType/reduceOp/空组，正常形态）→ DEBUG；
     * 目标缺席/禁用的终局告警已由上方 0-modified 分支的 WARN 打印。 */
    HCCL_DEBUG(
        "[TunerDFX] no applicable rule (predicates unmatched): opType=%d nBytes=%llu dataType=%d",
        (int)collOpInfo->collType, (unsigned long long)collOpInfo->nBytes, (int)collOpInfo->dataType);
    return HCCL_SUCCESS;
}

/* ===== 插件入口（返回 ops 指针，纯静态初始化，幂等无副作用）===== */
__attribute__((visibility("default"))) const HcclTunerPluginOps* HcclTunerPluginOpsGet(void)
{
    static const HcclTunerPluginOps ops = {
        {HCCL_TUNER_PLUGIN_OPS_VERSION, HCCL_TUNER_PLUGIN_OPS_MAGIC_WORD, sizeof(HcclTunerPluginOps), 0u},
        MyCreate,
        MyUpdateAlgoCostTable,
    };
    return &ops;
}
