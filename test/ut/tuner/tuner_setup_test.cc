/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* tuner_setup.cc 宿主侧 UT——只保留必要用例（每例对应一条独立守卫或对外契约）：
 * 加载失败回退/header 三连校验/Create 失败仅影响本 comm/inactive 跳过/ctx 超限回退/
 * Copy 失败回滚/幂等守卫/Update 双向 matched 透传/ctx 未命中跳过/插件失败吞错/collType 守卫/未配置 no-op。
 * 通过 mock_tuner_stubs.cc 提供可配置的 dlopen/dlsym/HcclEngineCtx* mock，
 * 生产代码 tuner_setup.cc 由 CMakeLists 直接编入（非桩）。
 * MockReset() 在 SetUp 中调用，每个用例独立配置 mock 行为。 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>

#include "tuner_setup.h"

/* mock 状态在 stub 文件中定义，此处 extern 声明以在测试中访问 */
extern struct MockState {
    bool dlopenFail;
    bool dlsymFail;
    HcclTunerPluginOps ops;
    HcclResult createRet;
    void* createCtxOut;
    uint32_t createCtxSizeOut;
    HcclResult updateRet;
    bool updateMatched;
    uint64_t updateDelayUs;
    HcclResult ctxCreateRet;
    HcclResult ctxCopyRet;
    HcclResult ctxGetRet;
    void* ctxGetPtr;
    uint64_t ctxGetSize;
    int createCalls;
    int updateCalls;
    int ctxCreateCalls;
    int ctxCopyCalls;
    int ctxGetCalls;
    int ctxDestroyCalls;
} g_mock;

extern void MockReset();

/* g_loadStatus 非 static（int32_t, 外部链接）：测试重置回 LOAD_READY 允许重载。
 * LOAD_READY=0 / LOAD_SUCCESS=1 / LOAD_FAILED=-1（tuner_setup.cc constexpr）。 */
extern int32_t g_loadStatus;

class TunerSetupTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        MockReset();
        g_loadStatus = 0; /* LOAD_READY——重置允许每个用例独立重新加载插件 */
        /* 默认设 HCCL_ALG_TUNER_PLUGIN 让 dlopen 路径可达 */
        setenv("HCCL_ALG_TUNER_PLUGIN", "/mock/plugin.so", 1);
    }
    void TearDown() override { unsetenv("HCCL_ALG_TUNER_PLUGIN"); }

    /* 走通加载 + Create + ctx 拷贝成功路径（createCtxOut 必须真实 malloc：宿主成功路径会 free） */
    void InitLoadedPlugin()
    {
        g_mock.createCtxOut = malloc(1024);
        ASSERT_NE(g_mock.createCtxOut, nullptr);
        g_mock.createCtxSizeOut = 1024;
        ASSERT_EQ(HcclTunerInit((HcclComm)0x1, nullptr), HCCL_SUCCESS);
    }

    /* 让 EngineCtxGet 命中（mock 不解引用 ctx，静态缓冲即可） */
    void EnableUpdateCtx()
    {
        static uint8_t memCtx[1024];
        g_mock.ctxGetRet = HCCL_SUCCESS;
        g_mock.ctxGetPtr = memCtx;
        g_mock.ctxGetSize = 1024;
    }
};

/* ===== header 校验拒绝路径 ===== */

TEST_F(TunerSetupTest, DlopenFailFallsBack)
{
    /* 动态加载失败（dlopen/dlsym 同类）→ LOAD_FAILED 回退 CostModel，no-op */
    g_mock.dlopenFail = true;
    HcclTunerInit((HcclComm)0x1, nullptr);
    EXPECT_FALSE(HcclTunerIsLoaded());
    EXPECT_EQ(g_mock.createCalls, 0);
}

TEST_F(TunerSetupTest, WrongMagicFallsBack)
{
    g_mock.ops.header.magicWord = 0xDEADBEEF;
    HcclTunerInit((HcclComm)0x1, nullptr);
    EXPECT_FALSE(HcclTunerIsLoaded());
    EXPECT_EQ(g_mock.createCalls, 0);
}

TEST_F(TunerSetupTest, WrongVersionFallsBack)
{
    g_mock.ops.header.version = HCCL_TUNER_PLUGIN_OPS_VERSION + 1;
    HcclTunerInit((HcclComm)0x1, nullptr);
    EXPECT_FALSE(HcclTunerIsLoaded());
    EXPECT_EQ(g_mock.createCalls, 0);
}

TEST_F(TunerSetupTest, NullCreateFnFallsBack)
{
    g_mock.ops.Create = nullptr;
    HcclTunerInit((HcclComm)0x1, nullptr);
    EXPECT_FALSE(HcclTunerIsLoaded());
}

/* ===== Create/ctx 路径 ===== */

TEST_F(TunerSetupTest, CreateFailFallsBack)
{
    g_mock.createRet = HCCL_E_INTERNAL;
    HcclTunerInit((HcclComm)0x1, nullptr);
    /* Create 失败只影响当前 comm（本 comm 回退 CostModel），进程级插件加载状态不变：
     * 下一个 comm 仍会重试 Create */
    EXPECT_TRUE(HcclTunerIsLoaded());
    EXPECT_EQ(g_mock.createCalls, 1);
}

TEST_F(TunerSetupTest, CreateInactiveSkipsCtxCopy)
{
    /* Create 成功但 ctx 为 null = inactive → 不走 EngineCtxCreate/Copy */
    g_mock.createCtxOut = nullptr;
    g_mock.createCtxSizeOut = 0;
    HcclTunerInit((HcclComm)0x1, nullptr);
    EXPECT_TRUE(HcclTunerIsLoaded());
    EXPECT_EQ(g_mock.createCalls, 1);
    EXPECT_EQ(g_mock.ctxCreateCalls, 0);
}

TEST_F(TunerSetupTest, CtxCopyFailRollsBack)
{
    /* createCtxOut 必须是真实 malloc 指针：宿主在 Copy 失败回滚路径会 free（插件契约） */
    g_mock.createCtxOut = malloc(1024);
    ASSERT_NE(g_mock.createCtxOut, nullptr);
    g_mock.createCtxSizeOut = 1024;
    g_mock.ctxCopyRet = HCCL_E_INTERNAL;
    HcclTunerInit((HcclComm)0x1, nullptr);
    /* Copy 失败 → Destroy 回滚 */
    EXPECT_EQ(g_mock.ctxDestroyCalls, 1);
    EXPECT_TRUE(HcclTunerIsLoaded());
}

TEST_F(TunerSetupTest, InitIdempotent)
{
    InitLoadedPlugin();
    EXPECT_EQ(g_mock.createCalls, 1);

    /* 二次 init → 幂等守卫跳过 Create */
    EnableUpdateCtx();
    HcclTunerInit((HcclComm)0x1, nullptr);
    EXPECT_EQ(g_mock.createCalls, 1); /* Create 未被重复调用 */
}

TEST_F(TunerSetupTest, CtxSizeExceedsLimitFallsBack)
{
    /* ctx 申报尺寸超 4MB 上限 → 回退 CostModel，不建通信域 ctx */
    g_mock.createCtxOut = malloc(5u * 1024u * 1024u);
    ASSERT_NE(g_mock.createCtxOut, nullptr);
    g_mock.createCtxSizeOut = 5u * 1024u * 1024u;
    HcclTunerInit((HcclComm)0x1, nullptr);
    EXPECT_EQ(g_mock.ctxCreateCalls, 0);
    EXPECT_TRUE(HcclTunerIsLoaded());
}

/* ===== Update 路径 ===== */

TEST_F(TunerSetupTest, UpdateMatchedPropagates)
{
    InitLoadedPlugin();
    EnableUpdateCtx();

    /* collType 必填有效值（HCCL_CMD_INVALID=0 会被跳过） */
    HcclTunerCollOpInfo info = {};
    info.collType = HCCL_CMD_ALLREDUCE;
    HcclTunerAlgoCostEntry entries[2] = {};

    /* matched 双向输出契约：命中 → true 透传；未命中 → false 写回 */
    bool modified = false;
    g_mock.updateMatched = true;
    HcclTunerCallGetCollInfo((HcclComm)0x1, &info, entries, 2, &modified);
    EXPECT_EQ(g_mock.updateCalls, 1);
    EXPECT_TRUE(modified);

    g_mock.updateMatched = false;
    modified = true; /* 预置非默认值，断言宿主确实写回 false */
    HcclTunerCallGetCollInfo((HcclComm)0x1, &info, entries, 2, &modified);
    EXPECT_EQ(g_mock.updateCalls, 2);
    EXPECT_FALSE(modified);
}

TEST_F(TunerSetupTest, UpdateNoCtxSkips)
{
    /* 插件已加载但 ctx 不存在（ctxGet 未命中）→ Update 不被调用
     * （collType 有效，排除被 collType 守卫跳过的混淆） */
    InitLoadedPlugin();
    g_mock.ctxGetRet = HCCL_E_INTERNAL;
    HcclTunerCollOpInfo info = {};
    info.collType = HCCL_CMD_ALLREDUCE;
    HcclTunerAlgoCostEntry entries[2] = {};
    bool modified = false;
    HcclTunerCallGetCollInfo((HcclComm)0x1, &info, entries, 2, &modified);
    EXPECT_EQ(g_mock.updateCalls, 0);
}

TEST_F(TunerSetupTest, UpdateRetFailIgnored)
{
    InitLoadedPlugin();
    EnableUpdateCtx();
    g_mock.updateRet = HCCL_E_INTERNAL;

    /* 插件 Update 返回失败 → 宿主吞掉错误返回 HCCL_SUCCESS，modified=false */
    HcclTunerCollOpInfo info = {};
    info.collType = HCCL_CMD_ALLREDUCE;
    HcclTunerAlgoCostEntry entries[2] = {};
    bool modified = true;
    EXPECT_EQ(HcclTunerCallGetCollInfo((HcclComm)0x1, &info, entries, 2, &modified), HCCL_SUCCESS);
    EXPECT_EQ(g_mock.updateCalls, 1);
    EXPECT_FALSE(modified);
}

TEST_F(TunerSetupTest, CollTypeInvalidSkipsUpdate)
{
    InitLoadedPlugin();
    EnableUpdateCtx();

    /* 不支持的算子类型（HCCL_CMD_INVALID=0）→ Update 不被调用 */
    HcclTunerCollOpInfo info = {};
    info.collType = HCCL_CMD_INVALID;
    HcclTunerAlgoCostEntry entries[2] = {};
    bool modified = false;
    HcclTunerCallGetCollInfo((HcclComm)0x1, &info, entries, 2, &modified);
    EXPECT_EQ(g_mock.updateCalls, 0);
}

TEST_F(TunerSetupTest, NoEnvVarNoLoad)
{
    /* 未配置 HCCL_ALG_TUNER_PLUGIN → 加载失败置 LOAD_FAILED，no-op */
    unsetenv("HCCL_ALG_TUNER_PLUGIN");
    HcclTunerInit((HcclComm)0x1, nullptr);
    EXPECT_FALSE(HcclTunerIsLoaded());
    EXPECT_EQ(g_mock.createCalls, 0);
}
