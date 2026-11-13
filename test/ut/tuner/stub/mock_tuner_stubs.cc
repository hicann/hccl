/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* tuner_setup.cc 宿主侧 UT 桩——可配置 mock（extern "C" 保证链接）。
 * dlopen/dlsym + HcclEngineCtx* + 日志，通过全局 MockState 控制行为。 */

#include "hccl_tuner_plugin.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

/* ===== 可配置 mock 状态 ===== */
struct MockState {
    bool dlopenFail = false;
    bool dlsymFail = false;
    HcclTunerPluginOps ops = {};
    HcclResult createRet = HCCL_SUCCESS;
    void* createCtxOut = nullptr;
    uint32_t createCtxSizeOut = 0;
    HcclResult updateRet = HCCL_SUCCESS;
    bool updateMatched = false;
    uint64_t updateDelayUs = 0;
    HcclResult ctxCreateRet = HCCL_SUCCESS;
    HcclResult ctxCopyRet = HCCL_SUCCESS;
    HcclResult ctxGetRet = HCCL_E_INTERNAL;
    void* ctxGetPtr = nullptr;
    uint64_t ctxGetSize = 0;
    int createCalls = 0;
    int updateCalls = 0;
    int ctxCreateCalls = 0;
    int ctxCopyCalls = 0;
    int ctxGetCalls = 0;
    int ctxDestroyCalls = 0;
};
MockState g_mock; /* 非 static：测试 extern 访问 */

/* ===== Mock 插件函数 ===== */
static HcclResult
MockCreate(HcclComm, const HcclTunerCommInfo*, const HcclTunerTopoInfo*, void** tunerCtx, uint32_t* tunerCtxSize)
{
    g_mock.createCalls++;
    *tunerCtx = g_mock.createCtxOut;
    *tunerCtxSize = g_mock.createCtxSizeOut;
    return g_mock.createRet;
}

static HcclResult MockUpdate(void*, const HcclTunerCollOpInfo*, HcclTunerAlgoCostEntry*, uint32_t, bool* matched)
{
    g_mock.updateCalls++;
    *matched = g_mock.updateMatched;
    if (g_mock.updateDelayUs > 0) {
        usleep(static_cast<useconds_t>(g_mock.updateDelayUs));
    }
    return g_mock.updateRet;
}

static const HcclTunerPluginOps* MockPluginOpsGet() { return &g_mock.ops; }

/* ===== Mock dlopen/dlsym/dlerror/dlclose（extern "C"）===== */
extern "C" {
void* dlopen(const char*, int) { return g_mock.dlopenFail ? nullptr : (void*)0xD1CE; }
void* dlsym(void*, const char* name)
{
    if (g_mock.dlsymFail || strcmp(name, "HcclTunerPluginOpsGet") != 0) {
        return nullptr;
    }
    return (void*)&MockPluginOpsGet;
}
char* dlerror() { return nullptr; }
int dlclose(void*) { return 0; }

/* ===== Mock HcclEngineCtx* + HcclGetHcclBuffer（extern "C"）===== */
HcclResult HcclEngineCtxCreate(HcclComm, const char*, int, uint64_t, void**)
{
    g_mock.ctxCreateCalls++;
    return g_mock.ctxCreateRet;
}

HcclResult HcclEngineCtxCopy(HcclComm, int, const char*, void*, uint64_t, int)
{
    g_mock.ctxCopyCalls++;
    return g_mock.ctxCopyRet;
}

HcclResult HcclEngineCtxGet(HcclComm, const char*, int, void** ctx, uint64_t* size)
{
    g_mock.ctxGetCalls++;
    *ctx = g_mock.ctxGetPtr;
    *size = g_mock.ctxGetSize;
    return g_mock.ctxGetRet;
}

HcclResult HcclEngineCtxDestroy(HcclComm, const char*, int)
{
    g_mock.ctxDestroyCalls++;
    return HCCL_SUCCESS;
}

/* BuildCommInfo 调用的 hcomm 函数（签名对齐 hccl_comm.h/hccl_res.h 弱符号声明，
 * 测试二进制内强定义覆盖 CANN 库内弱定义） */
HcclResult HcclGetCommName(HcclComm, char* buf)
{
    /* 宿主侧调用方缓冲为 TUNER_COMM_NAME_MAX_LENGTH(128) 栈缓冲 */
    if (buf != nullptr) {
        (void)snprintf(buf, 128, "test_comm");
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetHcclBuffer(HcclComm, void** addr, uint64_t* size)
{
    if (addr != nullptr) {
        *addr = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return HCCL_SUCCESS;
}
} /* extern "C" */

/* ===== 日志桩 ===== */
bool HcclCheckLogLevel(int, int) { return false; }
bool IsErrorToWarn() { return false; }

/* ===== 测试辅助（测试文件 extern 访问）===== */
void MockReset()
{
    g_mock = MockState();
    g_mock.ops.header.magicWord = HCCL_TUNER_PLUGIN_OPS_MAGIC_WORD;
    g_mock.ops.header.version = HCCL_TUNER_PLUGIN_OPS_VERSION;
    g_mock.ops.header.size = sizeof(HcclTunerPluginOps);
    g_mock.ops.Create = MockCreate;
    g_mock.ops.UpdateAlgoCostTable = MockUpdate;
}
