/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * hostDPU inline reduce UT 桩实现：
 * - comm_negotiation（IsNegotiationComm C++本体 / NegotiateCommC）由共享桩打桩
 *   （test/ut/common/comm_negotiation_stub/neg_c_stub.cc），本UT不编译公共协商模块，仅验证策略层逻辑；
 * - HcclDlsym 按符号集合返回假句柄，驱动 HcommPrimitivesDlInit 的 INIT_SUPPORT_FLAG 真实流程；
 * - HcommEndpointCheckFeature 强符号定义，覆盖 hcomm_primitives_dl.cc 的弱默认实现；
 * - HcclEngineCtx* 用 (comm, tag) 内存 KV 保真实现，语义与 HCOMM 一致。
 */
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include "ut_ir_stub.h"
#include "hccl_dl.h"
#include "hcomm_primitives_dl.h"

namespace ut {
IrStubConfig g_irStub;

namespace {
    // EngineCtx KV：key = (comm, ctxTag)，value = calloc 分配的内存
    std::mutex g_kvMutex;
    std::map<std::pair<HcclComm, std::string>, std::pair<void*, uint64_t>> g_engineCtxKv;
} // namespace

void IrStubReset()
{
    std::lock_guard<std::mutex> lock(g_kvMutex);
    for (auto& item : g_engineCtxKv) {
        free(item.second.first);
    }
    g_engineCtxKv.clear();
    g_irStub = IrStubConfig();
    NegCStubReset(); // 公共协商 C 接口共享桩同步复位
}

bool IrStubHasCtx(HcclComm comm, const std::string& tag)
{
    std::lock_guard<std::mutex> lock(g_kvMutex);
    return g_engineCtxKv.find({comm, tag}) != g_engineCtxKv.end();
}

int8_t IrStubGetCapValue(HcclComm comm, const std::string& tag)
{
    std::lock_guard<std::mutex> lock(g_kvMutex);
    auto it = g_engineCtxKv.find({comm, tag});
    if (it == g_engineCtxKv.end() || it->second.second < sizeof(int8_t)) {
        return static_cast<int8_t>(-1);
    }
    return *(static_cast<int8_t*>(it->second.first));
}

HcclResult IrStubPresetCap(HcclComm comm, int8_t capVal, uint64_t size)
{
    void* ctx = nullptr;
    HcclResult ret = HcclEngineCtxCreate(comm, HOSTDPU_INLINE_CAP_TAG, CommEngine::COMM_ENGINE_CPU_TS, size, &ctx);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    *(static_cast<int8_t*>(ctx)) = capVal;
    return HCCL_SUCCESS;
}
} // namespace ut

using namespace ut;

/* ---------------- op_common 依赖桩 ---------------- */
namespace ops_hccl {
// IsStreamInCaptureMode 桩（真实定义在 op_common.cc，本 UT 不编译该文件；声明由被测源码前向声明）
bool IsStreamInCaptureMode(aclrtStream stream)
{
    (void)stream;
    return g_irStub.streamInCapture;
}

} // namespace ops_hccl

extern "C" {
/* ---------------- dlsym 桩：按符号集合返回假句柄（驱动 INIT_SUPPORT_FLAG 真实流程） ---------------- */
void* HcclDlsym(void* handle, const char* funcName)
{
    (void)handle;
    if (funcName == nullptr || g_irStub.dlsymSymbols.count(std::string(funcName)) == 0) {
        return nullptr;
    }
    return reinterpret_cast<void*>(0x1);
}

/* HcommEndpointCheckFeature 强符号桩：覆盖 hcomm_primitives_dl.cc 的弱默认实现，
 * 记录入参供用例断言（featureType 取值/端点透传）。9.1.0+ 才有该接口，桩同步门控 */
#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
HcommResult
HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc* endpointDesc, bool* value)
{
    g_irStub.checkFeatureCnt++;
    g_irStub.lastFeatureType = static_cast<int32_t>(featureType);
    g_irStub.lastEndpointDesc = endpointDesc;
    if (value == nullptr) {
        return HCCL_E_PARA;
    }
    if (g_irStub.checkFeatureRet != HCCL_SUCCESS) {
        *value = false;
        return g_irStub.checkFeatureRet;
    }
    *value = g_irStub.checkFeatureValue;
    return HCCL_SUCCESS;
}
#endif

/* ---------------- HcclEngineCtx KV 桩（语义与 HCOMM 一致） ---------------- */
HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    (void)engine;
    if (ctxTag == nullptr || ctx == nullptr) {
        return HCCL_E_PARA;
    }
    if (size == 0) {
        return HCCL_E_PARA; // 对齐HCOMM engine_ctx_c_adpt.cc：size为0拒绝创建
    }
    if (g_irStub.engineCtxCreateFail) {
        return HCCL_E_INTERNAL;
    }
    if (g_irStub.engineCtxCreateExist) {
        return HCCL_E_PARA; // 模拟并发首算子竞争：另一线程已创建同名条目（HCOMM已存在语义）
    }
    std::lock_guard<std::mutex> lock(g_kvMutex);
    auto key = std::make_pair(comm, std::string(ctxTag));
    // 与真实HCOMM语义一致：同名tag已存在即拒绝创建(HCCL_E_PARA)，不覆盖旧块
    if (g_engineCtxKv.find(key) != g_engineCtxKv.end()) {
        return HCCL_E_PARA;
    }
    void* ptr = calloc(1, size);
    if (ptr == nullptr) {
        return HCCL_E_INTERNAL;
    }
    g_engineCtxKv[key] = {ptr, size};
    *ctx = ptr;
    g_irStub.engineCtxCreateCnt++;
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size)
{
    (void)engine;
    if (ctxTag == nullptr || ctx == nullptr || size == nullptr) {
        return HCCL_E_PARA;
    }
    g_irStub.engineCtxGetCnt++;
    std::lock_guard<std::mutex> lock(g_kvMutex);
    // 并发时序注入：SUT写前二次Get（第2次）时模拟协商期间并发算子已写入终态条目
    if (g_irStub.injectCapOnSecondGet && g_irStub.engineCtxGetCnt == 2) {
        auto key = std::make_pair(comm, std::string(ctxTag));
        if (g_engineCtxKv.find(key) == g_engineCtxKv.end()) {
            void* injectPtr = calloc(1, sizeof(int8_t));
            if (injectPtr != nullptr) {
                *(static_cast<int8_t*>(injectPtr)) = g_irStub.injectCapValue;
                g_engineCtxKv[key] = {injectPtr, sizeof(int8_t)};
            }
        }
    }
    auto it = g_engineCtxKv.find({comm, std::string(ctxTag)});
    if (it == g_engineCtxKv.end()) {
        return HCCL_E_NOT_FOUND;
    }
    *ctx = it->second.first;
    *size = it->second.second;
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxDestroy(HcclComm comm, const char* ctxTag, CommEngine engine)
{
    (void)engine;
    std::lock_guard<std::mutex> lock(g_kvMutex);
    auto it = g_engineCtxKv.find({comm, std::string(ctxTag)});
    if (it == g_engineCtxKv.end()) {
        return HCCL_E_NOT_FOUND;
    }
    free(it->second.first);
    g_engineCtxKv.erase(it);
    return HCCL_SUCCESS;
}
} // extern "C"
