/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* 单元测试：通过 #include "../plugin.c" 方式直接测试插件内部逻辑。
 * mock hostFuncs（ctxCreate/Get/Destroy + log）+ mock 算法条目（hcclTunerAlgoEntry_t[]）。 */

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
    (void)count;
    va_list args;
    va_start(args, format);
    int ret = vsnprintf(strDest, destMax, format, args);
    va_end(args);
    return ret;
}
int vsnprintf_s(char* strDest, size_t destMax, size_t count, const char* format, va_list args)
{
    (void)count;
    return vsnprintf(strDest, destMax, format, args);
}
}

#define HCCL_TUNER_TESTING
#include "../plugin.cpp"

#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

/* ===== mock hostFuncs ===== */
static void* g_mockCtx = NULL;
static uint64_t g_mockCtxSize = 0;

static HcclResult MockCtxCreate(HcclComm comm, const char* tag, uint64_t size, void** ctx)
{
    (void)comm;
    (void)tag;
    g_mockCtx = calloc(1, (size_t)size);
    g_mockCtxSize = size;
    *ctx = g_mockCtx;
    return HCCL_SUCCESS;
}

static HcclResult MockCtxGet(HcclComm comm, const char* tag, void** ctx, uint64_t* size)
{
    (void)comm;
    (void)tag;
    *ctx = g_mockCtx;
    if (size != NULL) {
        *size = g_mockCtxSize;
    }
    return (g_mockCtx != NULL) ? HCCL_SUCCESS : HCCL_E_INTERNAL;
}

static HcclResult MockCtxDestroy(HcclComm comm, const char* tag)
{
    (void)comm;
    (void)tag;
    free(g_mockCtx);
    g_mockCtx = NULL;
    g_mockCtxSize = 0;
    return HCCL_SUCCESS;
}

static void MockLog(int level, const char* file, int line, const char* fmt, ...)
{
    (void)level;
    (void)file;
    (void)line;
    va_list args;
    va_start(args, fmt);
    (void)vprintf(fmt, args);
    (void)putchar('\n');
    va_end(args);
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
    MockCtxDestroy(NULL, NULL);
    g_hostFuncsReady = 0;
    if (memset_s(&g_hostFuncs, sizeof(g_hostFuncs), 0, sizeof(g_hostFuncs)) != EOK) {
    }
    if (memset_s(&g_lastSchema, sizeof(g_lastSchema), 0, sizeof(g_lastSchema)) != EOK) {
    }
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

static hcclTunerHostFunctions_t MakeMockHostFuncs(void)
{
    hcclTunerHostFunctions_t hf = {};
    hf.ctxCreate = MockCtxCreate;
    hf.ctxGet = MockCtxGet;
    hf.ctxDestroy = MockCtxDestroy;
    hf.logFunction = MockLog;
    hf.structSize = sizeof(hcclTunerHostFunctions_t);
    return hf;
}

/* mock 算法条目（模拟 CostTableGen + Enrich 后的结果） */
#define MOCK_ENTRY_COUNT 4
static hcclTunerAlgoEntry_t MakeMockEntries(hcclTunerAlgoEntry_t* out)
{
    hcclTunerAlgoEntry_t tmpl = {};
    tmpl.structSize = sizeof(hcclTunerAlgoEntry_t);
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
    return tmpl;
}

/* ===== 测试用例 ===== */

/* 1. 函数表 */
static void TestDescriptorAndFuncs(void)
{
    ASSERT(hcclTunerPlugin_v1.init != NULL, "hcclTunerPlugin_v1 init set");
    ASSERT(hcclTunerPlugin_v1.getCollInfo != NULL, "getCollInfo set");
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;
    ASSERT(funcs.init != NULL && funcs.getCollInfo != NULL, "funcs populated");
}

/* 2. init 成功 + JSON 解析正确 */
static void TestInitAndParse(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.nServers = 1;
    commInfo.nNpusPerServer = 8;
    commInfo.commName = "world_group";
    commInfo.structSize = sizeof(commInfo);

    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    HcclResult ret = funcs.init((HcclComm)0x1, &commInfo, &hf);
    ASSERT(ret == HCCL_SUCCESS, "init success");

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->opSetCount == 2, "2 op types parsed");
    // nlohmann::json 按 key 字母序遍历: allgather < allreduce
    ASSERT(ctx->opSets[0].opType == HCCL_CMD_ALLGATHER, "first op is allgather");
    ASSERT(ctx->opSets[0].ruleCount == 1, "allgather has 1 rule");
    ASSERT(ctx->opSets[1].opType == HCCL_CMD_ALLREDUCE, "second op is allreduce");
    ASSERT(ctx->opSets[1].ruleCount == 2, "allreduce has 2 rules");
    ASSERT(ctx->commInfo.nRanks == 8, "commInfo nRanks stored");
    ASSERT(ctx->commInfo.commName != NULL, "commName persisted");
    ASSERT(strstr(ctx->commInfo.commName, "world") != NULL, "commName content correct");
}

/* 3. 规则匹配命中 */
static void TestRuleMatchHit(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.nServers = 1;
    commInfo.nNpusPerServer = 8;
    commInfo.commName = "world_group";
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    hcclTunerAlgoEntry_t entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    hcclTunerCollInfo_t collInfo = {};
    int matched = 0;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    collInfo.structSize = sizeof(collInfo);

    HcclResult ret = funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(ret == HCCL_SUCCESS, "getCollInfo success");
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
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 4; /* 不匹配 min_ranks=8 */
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    hcclTunerAlgoEntry_t entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 50.0f;
    }
    hcclTunerCollInfo_t collInfo = {};
    int matched = 0;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    collInfo.structSize = sizeof(collInfo);

    funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    int changed = 0;
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        if (entries[i].cost != 50.0f) {
            changed = 1;
            break;
        }
    }
    ASSERT(changed == 0, "no modification when rule does not match");
}

/* 5. first-match-wins */
static void TestFirstMatchWins(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    hcclTunerAlgoEntry_t entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    /* nBytes=4096, fp16 → 第 1 条规则命中（max_bytes=65536）。
     * 第 2 条规则也匹配（min_bytes=65536 不满足，因为 4096 < 65536，所以第 2 条不匹配）。
     * 改用 nBytes=100000，两条都匹配（第 1 条 max_bytes=65536 不满足）。
     * 实际上 100000 > 65536，第 1 条不匹配，第 2 条匹配。 */
    hcclTunerCollInfo_t collInfo = {};
    int matched = 0;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 100000;
    collInfo.dataType = HCCL_DATA_TYPE_FP32;
    collInfo.structSize = sizeof(collInfo);
    funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
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
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    hcclTunerAlgoEntry_t entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    /* fp32 不匹配第 1 条规则（要求 fp16），但 nBytes=4096 < 65536 所以第 2 条也不匹配 */
    hcclTunerCollInfo_t collInfo = {};
    int matched = 0;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP32;
    collInfo.structSize = sizeof(collInfo);
    funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[0].cost == 100.0f, "fp32 does not match fp16 rule");

    /* fp16 匹配 */
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    matched = 0;
    funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[0].cost == 0.0f, "fp16 matches fp16 rule");
}

/* 7. comm_name 匹配 */
static void TestCommNameMatch(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.commName = "world_group";
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    hcclTunerAlgoEntry_t entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    hcclTunerCollInfo_t collInfo = {};
    int matched = 0;
    collInfo.collType = HCCL_CMD_ALLGATHER;
    collInfo.nBytes = 1024;
    collInfo.structSize = sizeof(collInfo);
    funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    /* allgather 规则要求 comm_name 包含 "world" + max_bytes=1048576 → 命中 */
    ASSERT(entries[2].cost == 0.0f, "comm_name 'world' matched");

    /* comm_name 不匹配 */
    ResetPluginState();
    commInfo.commName = "other_group";
    funcs.init((HcclComm)0x2, &commInfo, &hf);
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    matched = 0;
    funcs.getCollInfo((HcclComm)0x2, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[2].cost == 100.0f, "comm_name 'other' did not match");
}

/* 8. 多 opType 隔离 */
static void TestOpTypeIsolation(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.commName = "world_group";
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    hcclTunerAlgoEntry_t entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    /* allgather 规则命中位置 (0,0,0)；allreduce 规则位置 (2,1,3) 不应被影响 */
    hcclTunerCollInfo_t collInfo = {};
    int matched = 0;
    collInfo.collType = HCCL_CMD_ALLGATHER;
    collInfo.nBytes = 1024;
    collInfo.structSize = sizeof(collInfo);
    funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[2].cost == 0.0f, "allgather rule applied");
    ASSERT(entries[0].cost == 100.0f, "allreduce position not affected by allgather op");
}

/* 9. Schema 校验：拼写错误检测 */
static void TestSchemaTypoDetection(void)
{
    ResetPluginState();
    /* "mtach" 是 "match" 的拼写错误 */
    FILE* fp = fopen("/tmp/hccl_tuner_test_typo.json", "w");
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"mtach\":{\"min_ranks\":8},\"engine\":\"aicpu\","
        "\"executor\":\"sole\",\"template\":\"mesh\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_typo.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    HcclResult ret = funcs.init((HcclComm)0x1, &commInfo, &hf);
    ASSERT(ret == HCCL_SUCCESS, "init success despite typo");
    ASSERT(g_lastSchema.warnings > 0, "typo 'mtach' detected as warning");

    /* 拼写错误导致缺 match → schema error → configValid=0 */
    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    if (ctx == NULL) {
        ASSERT(false, "context stored");
        return;
    }
    ASSERT(ctx->configValid == 0, "configValid=0 when schema has errors");
}

/* 10. Schema 校验：缺必填字段不干预 */
static void TestSchemaMissingRequired(void)
{
    ResetPluginState();
    /* 缺少必填字段 cost */
    FILE* fp = fopen("/tmp/hccl_tuner_test_typo.json", "w");
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8},\"engine\":\"aicpu\","
        "\"executor\":\"sole\",\"template\":\"mesh\"}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_typo.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 0, "configValid=0 when missing required field 'cost'");

    /* configValid=0 → getCollInfo 不干预 */
    hcclTunerAlgoEntry_t entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    hcclTunerCollInfo_t collInfo = {};
    int matched = 0;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    collInfo.structSize = sizeof(collInfo);
    funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    ASSERT(entries[2].cost == 100.0f, "does not intervene when config invalid");
}

/* 11. rule 缺 engine → configValid=0（不支持 defaults 继承） */
static void TestEngineMissingInRule(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_cfg.json", "w");
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_"
        "bytes\":0,\"max_bytes\":65536},"
        "\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 0, "configValid=0 when rule missing engine");
    ASSERT(g_lastSchema.errors > 0, "schema errors detected for missing engine");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 12. min_ranks=0 不阻断匹配（不限 rank 数下限） */
static void TestMatchMinRanksZero(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_cfg.json", "w");
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":0,\"max_ranks\":4294967295,"
        "\"min_bytes\":0,"
        "\"max_bytes\":65536},\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"meshoneshot\",\"cost\":0.0}]}}"
        "}",
        fp);
    fclose(fp);
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    hcclTunerAlgoEntry_t entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    for (int i = 0; i < MOCK_ENTRY_COUNT; i++) {
        entries[i].cost = 100.0f;
    }
    hcclTunerCollInfo_t collInfo = {};
    int matched = 0;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.structSize = sizeof(collInfo);
    funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);
    /* min_ranks=-1 应跳过检查，规则仍命中 */
    ASSERT(entries[0].cost == 0.0f, "min_ranks=0 does not block match");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 13. engine 越界值报 Schema error */
static void TestEngineOutOfRange(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_cfg.json", "w");
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8,\"max_ranks\":8,\"min_"
        "bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"invalid_engine\",\"executor\":\"sole\",\"template\":\"mesh\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    /* engine="invalid_engine" 不在合法枚举中 → SchemaError → configValid=0 */
    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 0, "configValid=0 when engine out of range");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 14. 必填字段缺失校验 */
static void TestRequiredFieldMissing(void)
{
    ResetPluginState();
    /* 缺 max_ranks */
    FILE* fp = fopen("/tmp/hccl_tuner_test_typo.json", "w");
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":8,\"min_bytes\":0,\"max_"
        "bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"mesh\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_typo.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 0, "configValid=0 when missing max_ranks");
    ASSERT(g_lastSchema.errors > 0, "schema errors detected for missing max_ranks");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 15. 范围校验：min > max */
static void TestRangeValidation(void)
{
    ResetPluginState();
    FILE* fp = fopen("/tmp/hccl_tuner_test_typo.json", "w");
    fputs(
        "{\"version\":1,\"op_types\":{\"allreduce\":{\"rules\":[{\"match\":{\"min_ranks\":16,\"max_ranks\":8,"
        "\"min_bytes\":0,\"max_bytes\":65536},"
        "\"engine\":\"aicpu\",\"executor\":\"sole\",\"template\":\"mesh\",\"cost\":0.0}]}}}",
        fp);
    fclose(fp);
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_typo.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    StoredHeader* ctx = TunerGetStoredCtx((HcclComm)0x1);
    ASSERT(ctx != NULL, "context stored");
    ASSERT(ctx->configValid == 0, "configValid=0 when min_ranks > max_ranks");
    ASSERT(g_lastSchema.errors > 0, "schema errors detected for min_ranks > max_ranks");

    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
}

/* 16. cost < 0 的 entry 不被插件修改 */
static void TestCostNegativeSkip(void)
{
    ResetPluginState();
    WriteTestConfig("/tmp/hccl_tuner_test_cfg.json");
    setenv("HCCL_TUNER_CONFIG_FILE", "/tmp/hccl_tuner_test_cfg.json", 1);

    hcclTunerCommInfo_t commInfo = {};
    commInfo.nRanks = 8;
    commInfo.structSize = sizeof(commInfo);
    hcclTunerHostFunctions_t hf = MakeMockHostFuncs();
    hcclTunerFuncs_v1_t funcs = hcclTunerPlugin_v1;

    funcs.init((HcclComm)0x1, &commInfo, &hf);

    hcclTunerAlgoEntry_t entries[MOCK_ENTRY_COUNT];
    MakeMockEntries(entries);
    /* entry 0 设为 -1（禁用），其他设为 100 */
    entries[0].cost = -1.0f;
    entries[1].cost = 100.0f;
    entries[2].cost = 100.0f;
    entries[3].cost = 100.0f;

    hcclTunerCollInfo_t collInfo = {};
    int matched = 0;
    collInfo.collType = HCCL_CMD_ALLREDUCE;
    collInfo.nBytes = 4096;
    collInfo.dataType = HCCL_DATA_TYPE_FP16;
    collInfo.structSize = sizeof(collInfo);
    funcs.getCollInfo((HcclComm)0x1, &collInfo, entries, MOCK_ENTRY_COUNT, &matched);

    /* entry 0 匹配规则1(aicpu/sole/meshoneshot) 但 cost=-1，应跳过不修改 */
    ASSERT(entries[0].cost == -1.0f, "disabled entry (cost<0) not modified by tuner");
    ASSERT(matched == 1, "rule still matched (matched=1)");
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

    printf("\n=== %d/%d tests passed ===\n", g_testsPass, g_testsRun);
    unlink("/tmp/hccl_tuner_test_cfg.json");
    unlink("/tmp/hccl_tuner_test_typo.json");
    return (g_testsPass == g_testsRun) ? 0 : 1;
}
