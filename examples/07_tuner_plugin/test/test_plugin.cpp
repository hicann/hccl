/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* 单元测试：通过 #include "../plugin.cpp" 方式直接测试插件内部逻辑。
 * 测试适配层：不 mock hostFuncs——Create 直接真实 malloc，测试持有 tunerCtx，
 * UpdateAlgoCostTable 直传；mock 算法条目（HcclTunerAlgoCostEntry[]，含 ABI header）。 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* securec 函数 stub：测试不链接 securec 库，提供与 securec.h 声明签名一致的 C 实现。 */
extern "C" {
int memset_s(void* dest, size_t destMax, int c, size_t count)
{
    if (dest == NULL || destMax < count) {
        return -1;
    }
    memset(dest, c, count);
    return 0;
}
int memcpy_s(void* dest, size_t destMax, const void* src, size_t count)
{
    if (dest == NULL || src == NULL || destMax < count) {
        return -1;
    }
    memcpy(dest, src, count);
    return 0;
}
int snprintf_s(char* strDest, size_t destMax, size_t count, const char* format, ...)
{
    /* stub 保真：原实现忽略 count 转发 glibc vsnprintf——截断时写入截断内容并
     * 返回正值，与生产 securec（格式化结果超 count → 返回 -1）语义相反，导致超长防御路径
     * （engine/executor/template/comm_name 截断报错）从未被真实测到。对齐：
     * 按 count 限长写入 + 期望长度超 count 时返回 -1。 */
    if (strDest == NULL || destMax == 0 || count >= destMax) {
        return -1;
    }
    va_list args;
    va_start(args, format);
    int len = vsnprintf(strDest, count + 1, format, args);
    va_end(args);
    if (len < 0) {
        return -1;
    }
    return ((size_t)len + 1u > count) ? -1 : len; /* securec 语义：count 含结尾 '\0'，len==count 即截断 */
}
int vsnprintf_s(char* strDest, size_t destMax, size_t count, const char* format, va_list args)
{
    if (strDest == NULL || destMax == 0 || count >= destMax) {
        return -1;
    }
    va_list copy;
    va_copy(copy, args); /* vsnprintf 消费 va_list，先拷贝再传 */
    int len = vsnprintf(strDest, count + 1, format, copy);
    va_end(copy);
    if (len < 0) {
        return -1;
    }
    return ((size_t)len + 1u > count) ? -1 : len; /* securec 语义：count 含结尾 '\0'，len==count 即截断 */
}
}

#define HCCL_TUNER_TESTING
#include "../plugin.cpp"

#include <unistd.h>
#include <pthread.h>

/* ===== 测试适配层：模拟宿主行为（调 Create 并持有 ctx，Update 直传）===== */
static void* g_testCtx = NULL;
static uint32_t g_testCtxSize = 0;

/* 返回当前存储 ctx（测试直接持有） */
static StoredHeader* TunerGetStoredCtx(HcclComm comm)
{
    (void)comm;
    return (g_testCtx != NULL) ? (StoredHeader*)g_testCtx : NULL;
}

/* ===== 测试框架 ===== */
static int g_testsRun = 0;
static int g_testsPass = 0;

#define ASSERT(cond, msg)                                  \
    do {                                                   \
        g_testsRun++;                                      \
        if (cond) {                                        \
            g_testsPass++;                                 \
        } else {                                           \
            printf("FAIL: %s (line %d)\n", msg, __LINE__); \
        }                                                  \
    } while (0)

static void ResetPluginState(void)
{
    free(g_testCtx);
    g_testCtx = NULL;
    g_testCtxSize = 0;
    if (memset_s(&g_lastSchema, sizeof(g_lastSchema), 0, sizeof(g_lastSchema)) != EOK) {
    }
}

/* ===== 测试适配层（ASSERT 之后定义）：包装层断言 Create/Update 返回值——
 * 丢弃返回值的用例在失败时会让后续"未修改"断言空转通过，此处单点收口全部调用点 ===== */
static HcclResult TunerTestCreate(HcclComm comm, HcclTunerCommInfo* commInfo, HcclTunerTopoInfo* topoInfo)
{
    HcclResult ret = HcclTunerPluginOpsGet()->Create(comm, commInfo, topoInfo, &g_testCtx, &g_testCtxSize);
    ASSERT(ret == HCCL_SUCCESS, "TunerTestCreate returned error");
    return ret;
}

static HcclResult
TunerTestUpdate(HcclTunerCollOpInfo* collOpInfo, HcclTunerAlgoCostEntry* entries, uint32_t count, bool* matched)
{
    HcclResult ret = HcclTunerPluginOpsGet()->UpdateAlgoCostTable(g_testCtx, collOpInfo, entries, count, matched);
    ASSERT(ret == HCCL_SUCCESS, "TunerTestUpdate returned error");
    return ret;
}

static const char* TEST_CONFIG
    = "{"
      "  \"version\": 1,"
      "  \"op_types\": {"
      "    \"allreduce\": {"
      "      \"rules\": ["
      "        {"
      "          \"match\": {\"min_ranks\": 8, \"max_ranks\": 8, \"min_bytes\": 0, \"max_bytes\": 65536, "
      "\"data_type\": \"fp16\"},"
      "          \"engine\": \"aicpu\", \"executor\": \"sole\", \"template\": \"meshoneshot\", \"cost\": 0.0"
      "        },"
      "        {"
      "          \"match\": {\"min_ranks\": 8, \"max_ranks\": 4294967295, \"min_bytes\": 65536, "
      "\"max_bytes\": 4294967295},"
      "          \"engine\": \"dpu\", \"executor\": \"parallel\", \"template\": \"meshoneshot\", \"cost\": 1.5"
      "        }"
      "      ]"
      "    },"
      "    \"allgather\": {"
      "      \"rules\": ["
      "        {"
      "          \"match\": {\"min_ranks\": 0, \"max_ranks\": 4294967295, \"min_bytes\": 0, "
      "\"max_bytes\": 1048576, \"comm_name\": \"world\"},"
      "          \"engine\": \"aicpu\", \"executor\": \"sole\", \"template\": \"mesh\", \"cost\": 0.0"
      "        }"
      "      ]"
      "    }"
      "  }"
      "}";

static int WriteTestConfig(const char* path)
{
    FILE* fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs(TEST_CONFIG, fp);
    fclose(fp);
    return 0;
}

/* mock 算法条目（模拟 CostTableGen + Enrich 后的结果：3D 名 + ABI header 已填） */
#define MOCK_ENTRY_COUNT 4
static void MakeMockEntries(HcclTunerAlgoCostEntry* out)
{
    HcclTunerAlgoCostEntry tmpl = {};
    (void)HcclTunerAlgoCostEntryInit(&tmpl);
    /* entry 0: 匹配 allreduce 规则 1 (aicpu/sole/meshoneshot) */
    out[0] = tmpl;
    out[0].algName = "AicpuAllReduceSoleMeshOneShot";
    out[0].engineName = "aicpu";
    out[0].executorName = "sole";
    out[0].templateName = "meshoneshot";
    out[0].cost = 2.0f;
    /* entry 1: 匹配 allreduce 规则 2 (dpu/parallel/meshoneshot) */
    out[1] = tmpl;
    out[1].algName = "DpuAllReduceParallelMeshOneShot";
    out[1].engineName = "dpu";
    out[1].executorName = "parallel";
    out[1].templateName = "meshoneshot";
    out[1].cost = 3.0f;
    /* entry 2: 匹配 allgather 规则 1 (aicpu/sole/mesh) */
    out[2] = tmpl;
    out[2].algName = "AicpuAllGatherSoleMesh";
    out[2].engineName = "aicpu";
    out[2].executorName = "sole";
    out[2].templateName = "mesh";
    out[2].cost = 4.0f;
    /* entry 3: 不匹配任何规则 (ccums/concur/NHR) */
    out[3] = tmpl;
    out[3].algName = "CcuMsAllReduceConcurNHR";
    out[3].engineName = "ccums";
    out[3].executorName = "concur";
    out[3].templateName = "nhr";
    out[3].cost = 5.0f;
}

/* ===== 测试用例 ===== */

/* 1. 入口符号：ops 指针 + header 三连 + 幂等 */
static void TestDescriptorAndFuncs(void)
{
    const HcclTunerPluginOps* ops = HcclTunerPluginOpsGet();
    ASSERT(ops != NULL, "ops pointer returned");
    ASSERT(ops->Create != NULL, "Create set");
    ASSERT(ops->UpdateAlgoCostTable != NULL, "UpdateAlgoCostTable set");
    ASSERT(ops->header.magicWord == HCCL_TUNER_PLUGIN_OPS_MAGIC_WORD, "ops magicWord valid");
    ASSERT(ops->header.version == HCCL_TUNER_PLUGIN_OPS_VERSION, "ops version valid");
    ASSERT(ops->header.size == sizeof(HcclTunerPluginOps), "ops size valid");
    ASSERT(HcclTunerPluginOpsGet() == ops, "Get idempotent, same pointer");
}

/* 2. Create 成功 + JSON 解析正确 */
static void TestInitAndParse(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    commInfo.commName = "world_group";
    topoInfo.nRanks = 8;
    topoInfo.nServers = 1;
    topoInfo.nNpusPerServer = 8;

    HcclResult ret = TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);
    ASSERT(ret == HCCL_SUCCESS, "create success");

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context created and held");
    ASSERT(g_testCtxSize >= sizeof(StoredHeader), "ctx size covers header");
    ASSERT(ctx->opSetCount == 2, "2 op types parsed");
    // nlohmann::json 按 key 字母序遍历: allgather < allreduce
    ASSERT(ctx->opSets[0].opType == HCCL_CMD_ALLGATHER, "first op is allgather");
    ASSERT(ctx->opSets[0].ruleCount == 1, "allgather has 1 rule");
    ASSERT(ctx->opSets[1].opType == HCCL_CMD_ALLREDUCE, "second op is allreduce");
    ASSERT(ctx->opSets[1].ruleCount == 2, "allreduce has 2 rules");
    ASSERT(ctx->topoInfo.nRanks == 8, "topoInfo nRanks stored");
    ASSERT(ctx->commNameBuf[0] != '\0', "commName persisted in inline buffer");
    ASSERT(strstr(ctx->commNameBuf, "world") != NULL, "commName content correct");
    ASSERT(ctx->commInfo.commName == NULL, "commName pointer not persisted (memcpy-safe)");
}

/* 3. 规则匹配命中 */
static void TestRuleMatchHit(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    commInfo.commName = "world_group";
    topoInfo.nRanks = 8;
    topoInfo.nServers = 1;
    topoInfo.nNpusPerServer = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;

    HcclResult ret = TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(ret == HCCL_SUCCESS, "update success");
    ASSERT(matched, "rule matched");
    /* 第 1 条规则命中：aicpu/sole/meshoneshot, cost=0.0 */
    ASSERT(entries[0].cost == 0.0f, "matched rule applied cost=0.0");
    /* 其他位置未被修改 */
    ASSERT(entries[2].cost == 100.0f, "unmatched position unchanged");
}

/* 4. 规则未命中不修改 */
static void TestRuleNoMatch(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 4; /* 不匹配 min_ranks=8 */
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 50.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;

    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    int changed = 0;
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        if (entries[i].cost != 50.0f) {
            changed = 1;
            break;
        }
    }
    ASSERT(changed == 0, "no modification when rule does not match");
}

/* 5. 首条命中即返回 */
static void TestFirstMatchWins(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    /* nBytes=100000 > 65536 → 第 1 条不匹配（max_bytes=65536），第 2 条匹配（min_bytes=65536） */
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 100000;
    collInfo.dataType = HCCL_DATA_TYPE_FP32;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    /* 第 2 条规则：dpu/parallel/meshoneshot, cost=1.5 */
    ASSERT(entries[1].cost == 1.5f, "second rule matched (first didn't)");
    /* 第 1 条规则位置未被修改 */
    ASSERT(entries[0].cost == 100.0f, "first rule position unchanged (didn't match)");
}

/* 6. data_type 匹配 */
static void TestDataTypeMatch(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    /* fp32 不匹配第 1 条规则（要求 fp16），但 nBytes=4096 < 65536 所以第 2 条也不匹配 */
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP32;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[0].cost == 100.0f, "fp32 does not match fp16 rule");

    /* fp16 匹配 */
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    matched = false;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[0].cost == 0.0f, "fp16 matches fp16 rule");
}

/* 7. comm_name 匹配 */
static void TestCommNameMatch(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    commInfo.commName = "world_group";
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLGATHER;
    collInfo.nBytes = 1024;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    /* allgather 规则要求 comm_name 包含 "world" + max_bytes=1048576 → 命中 */
    ASSERT(entries[2].cost == 0.0f, "comm_name 'world' matched");

    /* comm_name 不匹配 */
    ResetPluginState();
    commInfo.commName = "other_group";
    TunerTestCreate((HcclComm)0x2, &commInfo, &topoInfo);
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    matched = false;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[2].cost == 100.0f, "comm_name 'other' did not match");
}

/* 8. 多 opType 隔离 */
static void TestOpTypeIsolation(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    commInfo.commName = "world_group";
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    /* allgather 规则命中位置 entry2；allreduce 规则位置 entry0/1 不应被影响 */
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLGATHER;
    collInfo.nBytes = 1024;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[2].cost == 0.0f, "allgather rule applied");
    ASSERT(entries[0].cost == 100.0f, "allreduce position not affected by allgather op");
}

/* 9. Schema 校验：拼写错误检测 */
static void TestSchemaTypoDetection(void)
{
    ResetPluginState();
    /* "mtach" 是 "match" 的拼写错误 */
    FILE* fp = fopen("/tmp/hccl_tuner_test_typo.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"mtach\":{\"min_ranks\":8},\"engine\":\"aicpu\","
        "\"executor\":\"sole\",\"template\":\"mesh\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_typo.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;

    HcclResult ret = TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);
    ASSERT(ret == HCCL_SUCCESS, "create success despite typo");
    ASSERT(g_lastSchema.warnings > 0, "typo 'mtach' detected as warning");

    /* 拼写错误导致缺 match → 规则级 error → 该规则被剔除，整体 configValid=1 */
    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    if (ctx == NULL) {
        ASSERT(false, "context stored");
        return;
    }
    ASSERT(ctx->configValid == 1, "configValid=1 with rule-level error only");
    ASSERT(g_lastParseRules[0].valid == 0, "typo rule rejected");
}

/* 10. Schema 校验：缺必填字段不干预 */
static void TestSchemaMissingRequired(void)
{
    ResetPluginState();
    /* 缺少必填字段 cost */
    FILE* fp = fopen("/tmp/hccl_tuner_test_typo.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8},\"engine\":\"aicpu\","
        "\"executor\":\"sole\",\"template\":\"mesh\"}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_typo.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 1, "configValid=1 (rule-level error only rejects the rule)");
    ASSERT(g_lastParseRules[0].valid == 0, "rule missing 'cost' rejected");

    /* 被剔除的规则不参与匹配 → 不干预 */
    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[2].cost == 100.0f, "rejected rule does not intervene");
}

/* 11. rule 缺 engine → 该规则剔除（不支持 defaults 继承） */
static void TestEngineMissingInRule(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_cfg.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_"
        "bytes\":0,\"max_bytes\":65536},"
        "\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 1, "configValid=1 when rule missing engine");
    ASSERT(g_lastSchema.errors > 0, "schema errors detected for missing engine");
    ASSERT(g_lastParseRules[0].valid == 0, "rule missing engine rejected");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 12. min_ranks=0 不阻断匹配（不限 rank 数下限） */
static void TestMatchMinRanksZero(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_cfg.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":0,\"max_ranks\":4294967295,"
        "\"min_bytes\":0,"
        "\"max_bytes\":65536},\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}"
        "}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    /* min_ranks=0 不阻断匹配，规则仍命中 */
    ASSERT(entries[0].cost == 0.0f, "min_ranks=0 does not block match");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 13. engine 越界值报 Schema error */
static void TestEngineOutOfRange(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_cfg.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_"
        "bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"invalid_engine\",\"executor\":\"sole\",\"template\":\"mesh\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    /* engine="invalid_engine" 不在合法枚举中 → 规则级 error → 该规则剔除 */
    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 1, "configValid=1 when engine out of range");
    ASSERT(g_lastParseRules[0].valid == 0, "invalid engine rule rejected");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 14. 必填字段缺失校验 */
static void TestRequiredFieldMissing(void)
{
    ResetPluginState();
    /* 缺 max_ranks */
    FILE* fp = fopen("/tmp/hccl_tuner_test_typo.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8,\"min_bytes\":0,\"max_"
        "bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"mesh\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_typo.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 1, "configValid=1 when missing max_ranks");
    ASSERT(g_lastSchema.errors > 0, "schema errors detected for missing max_ranks");
    ASSERT(g_lastParseRules[0].valid == 0, "rule missing max_ranks rejected");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 15. 范围校验：min > max */
static void TestRangeValidation(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_typo.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":16,\"max_ranks\":8,"
        "\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"mesh\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_typo.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 1, "configValid=1 when min_ranks > max_ranks");
    ASSERT(g_lastSchema.errors > 0, "schema errors detected for min_ranks > max_ranks");
    ASSERT(g_lastParseRules[0].valid == 0, "min>max rule rejected");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 16. cost < 0 的 entry 不被插件修改 */
static void TestCostNegativeSkip(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    /* entry 0 设为 -1（禁用），其他设为 100 */
    entries[0].cost = -1.0f;
    entries[1].cost = 100.0f;
    entries[2].cost = 100.0f;
    entries[3].cost = 100.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);

    /* entry 0 匹配规则1(aicpu/sole/meshoneshot) 但 cost=-1，禁用条目不改；规则1 即终局，
     * 规则2（min_bytes=65536）不再被查看 → 无任何修改，matched=false */
    ASSERT(entries[0].cost == -1.0f, "disabled entry (cost<0) not modified by tuner");
    ASSERT(!matched, "disabled target terminates, no modification -> matched=false");
}

/* 17. 多级算法 template 拼接串匹配 */
static void TestMultiLevelTemplateMatch(void)
{
    ResetPluginState();
    /* 多级算法规则：executor=sequence, template=meshconcurnhrnhr（拼接串，不枚举校验） */
    FILE* fp = fopen("/tmp/hccl_tuner_test_multi.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8,\"max_ranks\":8,"
        "\"min_bytes\":0,\"max_bytes\":1048576},"
        "\"engine\":\"aicpu\",\"executor\":\"sequence\","
        "\"template\":\"meshconcurnhrnhr\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_multi.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    /* template 拼接串不枚举校验 → configValid=1 */
    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 1, "configValid=1 (template concat string not enum-validated)");

    /* 模拟 Enrich 后的算法条目 */
    HcclTunerAlgoCostEntry entries[2] = {};
    (void)HcclTunerAlgoCostEntryInit(&entries[0]);
    /* entry 0: 多级算法，executor=sequence, template=拼接串 */
    entries[0].algName = "AicpuAllReduceSequenceMeshConcurNHRNHR";
    entries[0].engineName = "aicpu";
    entries[0].executorName = "sequence";
    entries[0].templateName = "meshconcurnhrnhr";
    entries[0].cost = 5.0f;
    /* entry 1: 单级算法，不匹配规则（executor=sole） */
    entries[1] = entries[0];
    entries[1].algName = "AicpuAllReduceSoleMeshOneShot";
    entries[1].executorName = "sole";
    entries[1].templateName = "meshoneshot";
    entries[1].cost = 10.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    TunerTestUpdate(&collInfo, entries, 2, &matched);

    /* 多级算法 entry[0] 匹配规则，cost 改为 0.0 */
    ASSERT(entries[0].cost == 0.0f, "multi-level template matched, cost=0.0");
    /* 单级算法 entry[1] 不匹配（executor=sole != sequence），cost 不变 */
    ASSERT(entries[1].cost == 10.0f, "single-level entry not modified");
    ASSERT(matched, "rule matched");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 18. 逐规则剔除：1 条坏规则（data_type 非法）只废自己，好规则照常生效 */
static void TestPerRuleRejection(void)
{
    ResetPluginState();
    /* 第 1 条 data_type=fp33（坏，兼测 data_type 校验），第 2 条合法（dpu/parallel/meshoneshot, cost=1.5） */
    FILE* fp = fopen("/tmp/hccl_tuner_test_mix.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536,\"data_type\":\"fp33\"},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":4294967295,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":1.5}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_mix.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(g_lastSchema.errors > 0, "invalid data_type reported as schema error");
    ASSERT(ctx->configValid == 1, "configValid=1 (bad rule only rejects itself)");
    ASSERT(g_lastParseRules[0].valid == 0, "bad rule (data_type fp33) rejected");
    ASSERT(g_lastParseRules[1].valid == 1, "good rule still valid");

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(matched, "good rule still matches");
    ASSERT(entries[1].cost == 1.5f, "good rule applied (dpu entry cost=1.5)");
    ASSERT(entries[0].cost == 100.0f, "position not targeted by good rule unchanged");
}

/* 20. 配置级致命错误（version≠1）→ 整体不干预 */
static void TestFatalVersionRejected(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_fatal.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":2,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_fatal.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(g_lastSchema.fatal > 0, "version mismatch marked fatal");
    ASSERT(ctx->configValid == 0, "configValid=0 on fatal error");

    /* 整体不干预 */
    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[0].cost == 100.0f, "no intervention on fatal schema error");
}

/* 21. executor 不做枚举校验：未来新增值（如 strictordered）正常加载不剔除 */
static void TestExecutorForwardCompat(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_exec.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"strictordered\",\"template\":\"mesh\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_exec.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(g_lastSchema.errors == 0, "future executor value no schema error");
    ASSERT(g_lastParseRules[0].valid == 1, "rule with future executor stays valid");
    ASSERT(ctx->configValid == 1, "configValid=1");
}

/* 22. 缺席终局：规则1 目标不在表，终局后规则2 不可达（Issue #743 核心场景） */
static void TestAbsentTargetTerminates(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_fall.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":1.5}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_fall.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    /* 表里没有 aicpu 条目（模拟引擎被过滤），只有 dpu 与 ccums */
    HcclTunerAlgoCostEntry entries[2] = {};
    (void)HcclTunerAlgoCostEntryInit(&entries[0]);
    entries[0].algName = "DpuAllReduceParallelMeshOneShot";
    entries[0].engineName = "dpu";
    entries[0].executorName = "parallel";
    entries[0].templateName = "meshoneshot";
    entries[0].cost = 3.0f;
    entries[1] = entries[0];
    entries[1].algName = "CcuMsAllReduceConcurNHR";
    entries[1].engineName = "ccums";
    entries[1].executorName = "concur";
    entries[1].templateName = "nhr";
    entries[1].cost = 5.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    g_applyRuleCalls = 0;
    TunerTestUpdate(&collInfo, entries, 2, &matched);
    ASSERT(!matched, "firsthit: first applicable rule (absent target) terminates, no match");
    ASSERT(entries[0].cost == 3.0f, "firsthit: dpu entry unchanged (rule 2 unreachable)");
    ASSERT(entries[1].cost == 5.0f, "firsthit: ccums entry unchanged");
    ASSERT(g_applyRuleCalls == 1, "firsthit: terminating rule reaches ApplyRule once");
}

/* 23. 禁用终局：规则1 目标在表但被禁用（cost<0），终局后规则2 不可达；禁用条目不被触碰 */
static void TestDisabledTargetTerminates(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_dis.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":1.5}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_dis.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    /* aicpu 条目存在但被禁用，dpu 条目正常 */
    HcclTunerAlgoCostEntry entries[2] = {};
    (void)HcclTunerAlgoCostEntryInit(&entries[0]);
    entries[0].algName = "AicpuAllReduceSoleMeshOneShot";
    entries[0].engineName = "aicpu";
    entries[0].executorName = "sole";
    entries[0].templateName = "meshoneshot";
    entries[0].cost = -1.0f;
    entries[1] = entries[0];
    entries[1].algName = "DpuAllReduceParallelMeshOneShot";
    entries[1].engineName = "dpu";
    entries[1].executorName = "parallel";
    entries[1].cost = 3.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    g_applyRuleCalls = 0;
    TunerTestUpdate(&collInfo, entries, 2, &matched);
    ASSERT(entries[0].cost == -1.0f, "disabled entry untouched");
    ASSERT(entries[1].cost == 3.0f, "firsthit: dpu entry unchanged (rule 2 unreachable)");
    ASSERT(!matched, "firsthit: target-disabled rule terminates, no match");
    ASSERT(g_applyRuleCalls == 1, "firsthit: ApplyRule ran once on the terminating rule");
}

/* 25. 优先级保持：两条规则目标都在表且可改 → 首条终结，第二条不评估 */
static void TestFirstEffectiveRuleWins(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_prio.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":1.5}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_prio.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    /* 两个目标都在表且可改：首条生效，第二条不评估 */
    HcclTunerAlgoCostEntry entries[2] = {};
    (void)HcclTunerAlgoCostEntryInit(&entries[0]);
    entries[0].algName = "AicpuAllReduceSoleMeshOneShot";
    entries[0].engineName = "aicpu";
    entries[0].executorName = "sole";
    entries[0].templateName = "meshoneshot";
    entries[0].cost = 2.0f;
    entries[1] = entries[0];
    entries[1].algName = "DpuAllReduceParallelMeshOneShot";
    entries[1].engineName = "dpu";
    entries[1].executorName = "parallel";
    entries[1].cost = 3.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    TunerTestUpdate(&collInfo, entries, 2, &matched);
    ASSERT(entries[0].cost == 0.0f, "first rule applied (aicpu cost=0.0)");
    ASSERT(entries[1].cost == 3.0f, "second rule not evaluated");
    ASSERT(matched, "first effective rule wins");
}

/* 26. u32 溢出被拒：min_ranks=4294967304（2^32+8，原截断为 8）→ SchemaError，规则剔除 */
static void TestU32OverflowRejected(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_ovf.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":4294967304,\"max_ranks\":4294967304,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_ovf.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(g_lastSchema.errors > 0, "u32 overflow reported as schema error");
    ASSERT(g_lastParseRules[0].valid == 0, "overflow rule rejected");
    ASSERT(ctx->configValid == 1, "configValid=1 (rule-level only)");
    ASSERT(g_lastParseRules[0].match.minRanks == 0, "out param untouched on failure (no truncation to 8)");
}

/* 27. 负数被拒：min_servers=-1 → SchemaError，规则剔除 */
static void TestNegativeRejected(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_neg.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":4294967295,\"min_bytes\":0,\"max_bytes\":65536,\"min_servers\":-1},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_neg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(g_lastSchema.errors > 0, "negative value reported as schema error");
    ASSERT(g_lastParseRules[0].valid == 0, "negative-value rule rejected");
}

/* 28. 超 u64 被拒：min_bytes=2^64（JSON 解析为 double）→ SchemaError，规则剔除 */
static void TestAboveU64Rejected(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_big.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":4294967295,\"min_bytes\":18446744073709551616,"
        "\"max_bytes\":18446744073709551615},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_big.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(g_lastSchema.errors > 0, "above-u64 value reported as schema error");
    ASSERT(g_lastParseRules[0].valid == 0, "above-u64 rule rejected");
}

/* 29. 边界值放行：0 与 u32/u64 上界正常解析、值正确存储 */
static void TestBoundaryValuesAccepted(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_bnd.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":0,\"max_ranks\":4294967295,\"min_bytes\":0,"
        "\"max_bytes\":18446744073709551615},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_bnd.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(g_lastSchema.errors == 0, "boundary values pass validation");
    ASSERT(g_lastParseRules[0].valid == 1, "boundary rule stays valid");
    ASSERT(g_lastParseRules[0].match.maxRanks == 4294967295u, "u32 upper bound stored");
    ASSERT(g_lastParseRules[0].match.maxBytes == (size_t)18446744073709551615ULL, "u64 upper bound stored");
}

/* 30. Create inactive：无配置文件 → 空 ctx，Update 直接收 NULL ctx 不干预 */
static void TestCreateInactive(void)
{
    ResetPluginState();
    unlink("/tmp/hccl_tuner_test_not_exist.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_not_exist.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;

    HcclResult ret = TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);
    ASSERT(ret == HCCL_SUCCESS, "create success when inactive");
    ASSERT(g_testCtx == NULL, "inactive: empty tuner ctx");
    ASSERT(g_testCtxSize == 0, "inactive: ctx size 0");

    /* Update 直接收 NULL ctx：不干预 */
    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    bool matched = false;
    ret = TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(ret == HCCL_SUCCESS, "update success with null ctx");
    ASSERT(!matched, "null ctx: no intervention");
    ASSERT(entries[0].cost == 100.0f, "null ctx: costs unchanged");
}

/* 31. entry header 未填（size=0）→ 步长防护：规则终局、不干预 */
static void TestEntryHeaderNotFilled(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    /* 模拟宿主未填 header 的条目（header.size=0，正常链路 Enrich 必填，此处验证防护） */
    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
        entries[i].header.size = 0;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    HcclResult ret = TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(ret == HCCL_SUCCESS, "update success");
    ASSERT(!matched, "stride guard: no modification when header.size invalid");
    ASSERT(entries[0].cost == 100.0f, "entries untouched when stride invalid");
}

/* 32. reduce_op 匹配：可选条件，sum 命中、max 不命中 */
static void TestReduceOpMatch(void)
{
    ResetPluginState();
    /* 规则带 reduce_op=sum */
    FILE* fp = fopen("/tmp/hccl_tuner_test_redop.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536,\"reduce_op\":\"sum\"},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_redop.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    /* schema 校验通过（合法 reduce_op） */
    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(g_lastSchema.errors == 0, "valid reduce_op no schema error");
    ASSERT(g_lastParseRules[0].valid == 1, "reduce_op rule stays valid");

    /* sum 命中 */
    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    collInfo.reduceOp = HCCL_REDUCE_SUM;
    bool matched = false;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(matched, "sum matches reduce_op=sum rule");
    ASSERT(entries[0].cost == 0.0f, "sum rule applied");

    /* max 不命中（reduce_op=sum ≠ max） */
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    collInfo.reduceOp = HCCL_REDUCE_MAX;
    matched = false;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(!matched, "max does not match reduce_op=sum rule");
    ASSERT(entries[0].cost == 100.0f, "max position unchanged");
}

/* 33. reduce_op 非法值：SchemaError + 规则剔除 */
static void TestReduceOpInvalid(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_redop2.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536,\"reduce_op\":\"xor\"},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_redop2.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(g_lastSchema.errors > 0, "invalid reduce_op reported as schema error");
    ASSERT(g_lastParseRules[0].valid == 0, "invalid reduce_op rule rejected");
}

/* ===== CompactRule 移植用例（V2 调用形态）===== */

/* 34. 拓扑预剔除：nRanks=8 时 min_ranks=100 的规则在 Create 编译期被剔，不落 ctx */
static void TestTopologyPrune(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_topo.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":100,\"max_ranks\":4294967295,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":9.9},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":4294967295,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":1.5}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_topo.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->opSets[0].ruleCount == 1, "topo-mismatched rule pruned at Create");
    ASSERT(ctx->topoPrunedCount == 1, "topoPrunedCount tracked");
    ASSERT(ctx->schemaRejectedCount == 0, "no schema rejection");
    ASSERT(strcmp(CtxStr(ctx, GetRules(ctx)[0].engineOff), "dpu") == 0, "surviving rule is the topo-matching one");
}

/* 35. 保序剔除：被剔规则让位后，幸存规则保持 JSON 相对顺序（顺序 = 优先级，不可重排） */
static void TestOrderPreservedAfterPrune(void)
{
    ResetPluginState();
    /* R1 拓扑失配被剔；R2/R3 可命中。若剔除导致重排，R3 会先于 R2 生效 → cost 断言失败 */
    FILE* fp = fopen("/tmp/hccl_tuner_test_ord.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":100,\"max_ranks\":4294967295,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":9.9},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":4294967295,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.5},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":4294967295,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":1.5}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_ord.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT]; /* 修复：MakeMockEntries 固定写 4 条，原 entries[2] 越界写（#3896
                                                         既有 bug，ASAN 定位） */
    MakeMockEntries(entries);
    entries[0].algName = "AicpuAllReduceSoleMeshOneShot";
    entries[0].engineName = "aicpu";
    entries[0].executorName = "sole";
    entries[0].templateName = "meshoneshot";
    entries[0].cost = 2.0f;
    entries[1] = entries[0];
    entries[1].algName = "DpuAllReduceParallelMeshOneShot";
    entries[1].engineName = "dpu";
    entries[1].executorName = "parallel";
    entries[1].cost = 3.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    bool matched = false;
    TunerTestUpdate(&collInfo, entries, 2, &matched);
    ASSERT(entries[0].cost == 0.5f, "first surviving rule (R2) wins with its priority position");
    ASSERT(entries[1].cost == 3.0f, "later rule (R3) not evaluated (first effective terminates)");
    ASSERT(matched, "matched");
}

/* 36. data_type 通配（哨兵 -1）：未配置 data_type 的规则对任意 dataType 生效 */
static void TestDataTypeWildcard(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_dtw.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":4294967295,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_dtw.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(GetRules(ctx)[0].dataTypeEnum == -1, "no data_type compiled to sentinel -1");

    HcclTunerAlgoCostEntry entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_INT32; /* 任意类型均可命中通配规则 */
    bool matched = false;
    TunerTestUpdate(&collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[0].cost == 0.0f, "wildcard data_type rule matches int32 coll");
}

/* 37. 编译汇总字段：标准配置全量存活，计数口径正确（active + topoPruned + schemaRejected = total） */
static void TestCompiledCounters(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    commInfo.commName = "world_group";
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->totalRuleCount == 3, "all 3 rules active");
    ASSERT(ctx->topoPrunedCount == 0, "no topo pruning on matching comm");
    ASSERT(ctx->schemaRejectedCount == 0, "no schema rejection");
}

/* 25. 目标缺席终局：首条命中规则进入 ApplyRule（g_applyRuleCalls==1 证明），目标缺席即终局，兜底规则不可达 */
static void TestFirsthitAbsentTargetTerminates(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_ff1.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"ccums\",\"executor\":\"concur\",\"template\":\"nhr\",\"cost\":0.5},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aiv\",\"executor\":\"singlering\",\"template\":\"ring\",\"cost\":0.7},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":1.5}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_ff1.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    g_applyRuleCalls = 0;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    /* 候选只有 dpu/parallel/meshoneshot——前 3 条规则目标全缺席 */
    HcclTunerAlgoCostEntry entries[1] = {};
    (void)HcclTunerAlgoCostEntryInit(&entries[0]);
    entries[0].algName = "DpuAllReduceParallelMeshOneShot";
    entries[0].engineName = "dpu";
    entries[0].executorName = "parallel";
    entries[0].templateName = "meshoneshot";
    entries[0].cost = 3.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    TunerTestUpdate(&collInfo, entries, 1, &matched);
    ASSERT(g_applyRuleCalls == 1, "firsthit: terminating rule reaches ApplyRule once");
    ASSERT(!matched, "firsthit: absent first rule terminates with no modification");
    ASSERT(entries[0].cost == 3.0f, "firsthit: fallback rule unreachable, entry unchanged");
}

/* 26. 终局名字精确性：首条命中规则（R1）三维与所有候选均不同体，ApplyRule 精确匹配
 *     0 改动 → 终局，后续规则（R2/R3）不再被查看 */
static void TestFirsthitMismatchTerminates(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_ff2.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.1},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"nhr\",\"cost\":0.2},"
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":0.3}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_ff2.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    g_applyRuleCalls = 0;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    /* 候选：dpu/parallel/meshoneshot + ccums/concur/nhr。
     * R1（aicpu/sole/meshoneshot）三维与候选无一同体 → ApplyRule 精确匹配 0 改动 → 终局；
     * R2/R3 不再被查看，两候选条目均保持原值 */
    HcclTunerAlgoCostEntry entries[2] = {};
    (void)HcclTunerAlgoCostEntryInit(&entries[0]);
    entries[0].algName = "DpuAllReduceParallelMeshOneShot";
    entries[0].engineName = "dpu";
    entries[0].executorName = "parallel";
    entries[0].templateName = "meshoneshot";
    entries[0].cost = 3.0f;
    entries[1] = entries[0];
    entries[1].algName = "CcumsAllReduceConcurNHR";
    entries[1].engineName = "ccums";
    entries[1].executorName = "concur";
    entries[1].templateName = "nhr";
    entries[1].cost = 5.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    TunerTestUpdate(&collInfo, entries, 2, &matched);
    ASSERT(g_applyRuleCalls == 1, "firsthit: terminating rule reaches ApplyRule once");
    ASSERT(!matched, "firsthit: R2/R3 unreachable after R1 terminates");
    ASSERT(entries[0].cost == 3.0f, "firsthit: entries unchanged");
    ASSERT(entries[1].cost == 5.0f, "firsthit: entries unchanged (second)");
}

/* snprintf 累加安全检查：截断时返回值是"期望长度"而非写入长度，会把 off 推过 cap——
 * buf+off 随即越界且 (size_t)(cap-off) 无符号下溢成巨大长度，继续拼接即栈/静态区越界写 */
static int OffOk(int off, size_t cap) { return (off >= 0) && ((size_t)off < cap); }

/* 带边界检查的追加拼接：拼接前判 off 落界，vsnprintf 截断/出错返回 0 且 off 不推进，
 * 调用方 ASSERT+return——大配置构建全程每步受控，封死"截断把 off 推过 cap"的越界链 */
static int AppendChecked(char* buf, size_t cap, int* off, const char* fmt, ...)
{
    if (!OffOk(*off, cap)) {
        return 0;
    }
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *off, cap - (size_t)*off, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= cap - (size_t)*off) {
        return 0;
    }
    *off += n;
    return 1;
}

/* 27. 大配置终局正确性：大量唯一名字编译不丢规则，终局匹配正确——正确性与名字数无关 */
static void TestFirsthitManyStringsCorrect(void)
{
    ResetPluginState();
    char buf[32768];
    int off = 0;
    if (!AppendChecked(buf, sizeof(buf), &off, "{\"version\":1,\"op_types\":{\"allgather\":{\"rules\":[")) {
        ASSERT(false, "27: config exceeds buf[32768]");
        return;
    }
    for (int i = 0; i < 69; i++) {
        if (!AppendChecked(
                buf, sizeof(buf), &off,
                "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
                "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"ff%02d\",\"cost\":0.1}%s",
                i, (i == 68) ? "" : ",")) {
            ASSERT(false, "27: config exceeds buf[32768]");
            return;
        }
    }
    if (!AppendChecked(
            buf, sizeof(buf), &off,
            ",{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
            "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":1.5}]}}}")) {
        ASSERT(false, "27: config exceeds buf[32768]");
        return;
    }
    FILE* fp = fopen("/tmp/hccl_tuner_test_ff3.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(buf, fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_ff3.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    g_applyRuleCalls = 0;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* octx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(octx != NULL && octx->totalRuleCount == 70, "overflow: 70 rules compiled");

    HcclTunerAlgoCostEntry entries[1] = {};
    (void)HcclTunerAlgoCostEntryInit(&entries[0]);
    entries[0].algName = "DpuAllGatherParallelMeshOneShot";
    entries[0].engineName = "dpu";
    entries[0].executorName = "parallel";
    entries[0].templateName = "meshoneshot";
    entries[0].cost = 3.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLGATHER;
    collInfo.nBytes = 4096;
    TunerTestUpdate(&collInfo, entries, 1, &matched);
    ASSERT(!matched, "firsthit: first absent rule terminates with no modification");
    ASSERT(entries[0].cost == 3.0f, "firsthit: entry unchanged");
    ASSERT(g_applyRuleCalls == 1, "firsthit: terminating rule reaches ApplyRule once");
}

/* 27b. 超大配置终局正确性：263 条规则（261×aicpu/sole/ffNNN + dpu/parallel/ff260
 *     + dpu/parallel/meshoneshot）——首条规则谓词命中但三维与候选不同体 → 0 改动终局，
 *     后续 262 条规则不再查看，条目保持原值 */
static void TestFirsthitHugeConfigCorrect(void)
{
    ResetPluginState();
    static char buf[65536];
    int off = 0;
    if (!AppendChecked(buf, sizeof(buf), &off, "{\"version\":1,\"op_types\":{\"allgather\":{\"rules\":[")) {
        ASSERT(false, "27b: config exceeds buf[65536]");
        return;
    }
    for (int i = 0; i < 261; i++) {
        if (!AppendChecked(
                buf, sizeof(buf), &off,
                "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
                "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"ff%03d\",\"cost\":0.1}%s",
                i, (i == 260) ? "" : ",")) {
            ASSERT(false, "27b: config exceeds buf[65536]");
            return;
        }
    }
    if (!AppendChecked(
            buf, sizeof(buf), &off,
            ",{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
            "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"ff260\",\"cost\":9.9}")) {
        ASSERT(false, "27b: config exceeds buf[65536]");
        return;
    }
    if (!AppendChecked(
            buf, sizeof(buf), &off,
            ",{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
            "\"engine\":\"dpu\",\"executor\":\"parallel\",\"template\":\"meshoneshot\",\"cost\":1.5}]}}}")) {
        ASSERT(false, "27b: config exceeds buf[65536]");
        return;
    }
    FILE* fp = fopen("/tmp/hccl_tuner_test_ffwide.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(buf, fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_ffwide.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    g_applyRuleCalls = 0;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* wctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(wctx != NULL && wctx->totalRuleCount == 263, "wide: 263 rules compiled");

    HcclTunerAlgoCostEntry entries[1] = {};
    (void)HcclTunerAlgoCostEntryInit(&entries[0]);
    entries[0].algName = "DpuAllGatherParallelMeshOneShot";
    entries[0].engineName = "dpu";
    entries[0].executorName = "parallel";
    entries[0].templateName = "meshoneshot";
    entries[0].cost = 3.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    bool matched = false;
    collInfo.collType = HCCL_CMD_ALLGATHER;
    collInfo.nBytes = 4096;
    TunerTestUpdate(&collInfo, entries, 1, &matched);
    ASSERT(!matched, "firsthit: first absent rule terminates with no modification");
    ASSERT(entries[0].cost == 3.0f, "firsthit: entry unchanged");
    ASSERT(g_applyRuleCalls == 1, "firsthit: terminating rule reaches ApplyRule once");
}

/* 28. CtxIntern 偏移/dedup/溢出单测（终局制下序号/翻译退役，仅产出偏移） */
static void TestCtxIntern(void)
{
    /* CtxIntern 语义：字符串区 per-Create 局部、无并发；终局制下仅产出偏移 */
    StrAreaBuilder sa;
    sa.used = 0;
    uint32_t off = 0;
    ASSERT(CtxIntern(&sa, NULL, &off) && off == STR_OFF_WILDCARD, "intern: NULL -> WILDCARD offset");
    ASSERT(CtxIntern(&sa, "", &off) && off == STR_OFF_WILDCARD, "intern: empty -> WILDCARD offset");

    /* dedup：同名一份，重复落区命中已有偏移 */
    static const char* probe = "intern-probe";
    uint32_t probeOff = 0;
    ASSERT(CtxIntern(&sa, probe, &probeOff) && probeOff == 0, "intern: first string at offset 0");
    ASSERT(CtxIntern(&sa, probe, &off) && off == probeOff, "intern: dedup returns same offset");
    ASSERT(strcmp(sa.buf + probeOff, probe) == 0, "intern: offset points at the interned copy");
    ASSERT(sa.used == strlen(probe) + 1u, "intern: used advanced by len+1");

    /* 不同名追加，偏移连续 */
    ASSERT(
        CtxIntern(&sa, "second", &off) && off == sa.used - strlen("second") - 1u,
        "intern: second string appended contiguously");

    /* 区满 → 返回 false（调用方剔除规则），不静默通配 */
    StrAreaBuilder big;
    big.used = 0;
    char ovfName[24];
    bool overflowed = false;
    for (int i = 0; i < 20000; i++) { /* 16KB / 最长 23B 串，700+ 轮内必满 */
        (void)snprintf(ovfName, sizeof(ovfName), "ffovf%06d", i);
        if (!CtxIntern(&big, ovfName, &off)) {
            overflowed = true;
            break;
        }
    }
    ASSERT(overflowed, "intern: str area overflow rejected (no silent wildcard)");
}

/* 30. v2 变体算子支持（S1）：reduce_scatterv / all_gatherv / alltoallvc 三组规则各自命中 */
static void TestV2OpTypes(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_v2op.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{"
        "\"reduce_scatterv\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"nhr\",\"cost\":0.25}]},"
        "\"all_gatherv\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"mesh\",\"cost\":0.5}]},"
        "\"alltoallvc\":{\"rules\":["
        "{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"ring\",\"cost\":0.75}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_v2op.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "v2 op types: context stored");
    ASSERT(ctx->configValid == 1, "v2 op types: config valid (no unknown op_type rejection)");

    HcclTunerAlgoCostEntry entries[1] = {};
    (void)HcclTunerAlgoCostEntryInit(&entries[0]);
    entries[0].engineName = "aicpu";
    entries[0].executorName = "sole";
    entries[0].cost = 9.0f;

    HcclTunerCollOpInfo collInfo = {};
    (void)HcclTunerCollOpInfoInit(&collInfo);
    collInfo.nBytes = 4096;

    /* reduce_scatterv */
    entries[0].algName = "AicpuReduceScatterV SoleNHR";
    entries[0].templateName = "nhr";
    entries[0].cost = 9.0f;
    bool matched = false;
    collInfo.collType = HCCL_CMD_REDUCE_SCATTER_V;
    TunerTestUpdate(&collInfo, entries, 1, &matched);
    ASSERT(matched, "v2 op: reduce_scatterv rule matches");
    ASSERT(entries[0].cost == 0.25f, "v2 op: reduce_scatterv cost modified");

    /* all_gatherv */
    entries[0].algName = "AicpuAllGatherV SoleMesh";
    entries[0].templateName = "mesh";
    entries[0].cost = 9.0f;
    matched = false;
    collInfo.collType = HCCL_CMD_ALLGATHER_V;
    TunerTestUpdate(&collInfo, entries, 1, &matched);
    ASSERT(matched, "v2 op: all_gatherv rule matches");
    ASSERT(entries[0].cost == 0.5f, "v2 op: all_gatherv cost modified");

    /* alltoallvc */
    entries[0].algName = "AicpuAllToAllVC SoleRing";
    entries[0].templateName = "ring";
    entries[0].cost = 9.0f;
    matched = false;
    collInfo.collType = HCCL_CMD_ALLTOALLVC;
    TunerTestUpdate(&collInfo, entries, 1, &matched);
    ASSERT(matched, "v2 op: alltoallvc rule matches");
    ASSERT(entries[0].cost == 0.75f, "v2 op: alltoallvc cost modified");

    /* 未配置组的算子（allreduce）静默不干预（S1 连带：无组不再刷 no rule matched WARN） */
    matched = false;
    entries[0].cost = 9.0f;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    TunerTestUpdate(&collInfo, entries, 1, &matched);
    ASSERT(!matched, "v2 op: unconfigured opType stays untouched");
    ASSERT(entries[0].cost == 9.0f, "v2 op: unconfigured opType cost unchanged");
}

/* 31. comm_name 超长报错：>63 字符 schema error 剔除，不再静默通配化 */
static void TestCommNameTooLong(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_cnl.json", "w");
    if (fp == NULL) { /* 判空防 /tmp 不可写时 SIGSEGV 掩盖失败原因 */
        ASSERT(false, "fopen failed for test config");
        return;
    }
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8,\"max_ranks\":8,"
        "\"min_bytes\":0,\"max_bytes\":65536,"
        "\"comm_name\":\"this_is_an_over_sixty_three_character_long_communication_domain_name_xxxxx\"},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_ALG_CONFIG_FILE", "/tmp/hccl_tuner_test_cnl.json", 1);

    HcclTunerCommInfo commInfo = {};
    HcclTunerTopoInfo topoInfo = {};
    (void)HcclTunerCommInfoInit(&commInfo);
    (void)HcclTunerTopoInfoInit(&topoInfo);
    topoInfo.nRanks = 8;
    TunerTestCreate((HcclComm)0x1, &commInfo, &topoInfo);

    ASSERT(g_lastSchema.errors > 0, "comm_name too long: schema error reported");
    ASSERT(g_lastParseRules != NULL && g_lastParseRules[0].valid == 0, "comm_name too long: rule rejected");
    ASSERT(
        g_lastParseRules != NULL && g_lastParseRules[0].match.hasCommName == 0,
        "comm_name too long: hasCommName not set (no silent wildcard)");
}

int main(void)
{
    TestDescriptorAndFuncs();
    TestInitAndParse();
    TestRuleMatchHit();
    TestRuleNoMatch();
    TestFirstMatchWins();
    TestDataTypeMatch();
    TestCommNameMatch();
    TestOpTypeIsolation();
    TestSchemaTypoDetection();
    TestSchemaMissingRequired();
    TestEngineMissingInRule();
    TestMatchMinRanksZero();
    TestEngineOutOfRange();
    TestRequiredFieldMissing();
    TestRangeValidation();
    TestCostNegativeSkip();
    TestMultiLevelTemplateMatch();
    TestPerRuleRejection();
    TestFatalVersionRejected();
    TestExecutorForwardCompat();
    TestAbsentTargetTerminates();
    TestDisabledTargetTerminates();
    TestFirstEffectiveRuleWins();
    TestU32OverflowRejected();
    TestNegativeRejected();
    TestAboveU64Rejected();
    TestBoundaryValuesAccepted();
    TestCreateInactive();
    TestEntryHeaderNotFilled();
    TestReduceOpMatch();
    TestReduceOpInvalid();
    TestTopologyPrune();
    TestOrderPreservedAfterPrune();
    TestDataTypeWildcard();
    TestCompiledCounters();
    TestFirsthitAbsentTargetTerminates();
    TestFirsthitMismatchTerminates();
    TestFirsthitManyStringsCorrect();
    TestFirsthitHugeConfigCorrect();
    TestCtxIntern();
    TestV2OpTypes();
    TestCommNameTooLong();

    printf("\n=== %d/%d tests passed ===\n", g_testsPass, g_testsRun);
    unlink("/tmp/hccl_tuner_test_cfg.json");
    unlink("/tmp/hccl_tuner_test_typo.json");
    unlink("/tmp/hccl_tuner_test_multi.json");
    unlink("/tmp/hccl_tuner_test_mix.json");
    unlink("/tmp/hccl_tuner_test_fatal.json");
    unlink("/tmp/hccl_tuner_test_exec.json");
    unlink("/tmp/hccl_tuner_test_fall.json");
    unlink("/tmp/hccl_tuner_test_dis.json");
    unlink("/tmp/hccl_tuner_test_prio.json");
    unlink("/tmp/hccl_tuner_test_ovf.json");
    unlink("/tmp/hccl_tuner_test_neg.json");
    unlink("/tmp/hccl_tuner_test_big.json");
    unlink("/tmp/hccl_tuner_test_bnd.json");
    unlink("/tmp/hccl_tuner_test_topo.json");
    unlink("/tmp/hccl_tuner_test_ord.json");
    unlink("/tmp/hccl_tuner_test_dtw.json");
    unlink("/tmp/hccl_tuner_test_ff1.json");
    unlink("/tmp/hccl_tuner_test_ff2.json");
    unlink("/tmp/hccl_tuner_test_ff3.json");
    unlink("/tmp/hccl_tuner_test_v2op.json");
    unlink("/tmp/hccl_tuner_test_cnl.json");
    unlink("/tmp/hccl_tuner_test_redop.json");
    unlink("/tmp/hccl_tuner_test_redop2.json");
    unlink("/tmp/hccl_tuner_test_ffwide.json");
    return (g_testsPass == g_testsRun) ? 0 : 1;
}
