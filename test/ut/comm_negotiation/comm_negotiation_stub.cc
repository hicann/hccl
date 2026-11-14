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
 * comm_negotiation UT 桩实现：
 * - HcclEngineCtx* 用 (comm, tag) 内存 KV 保真实现，语义与 HCOMM 一致（含 HCCL_E_NOT_FOUND）；
 * - haclrtMemcpy/haclrtMemset 真实搬数（device 内存为桩 malloc 的普通内存）；
 * - HcclAllReduce(MIN)/HcclAllGather 提供"虚拟远端 rank"聚合语义；
 * - aclrt* 为最小可用的假资源分配。
 */
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <utility>
#include "ut_neg_stub.h"
#include "adapter_acl.h"
#include "adapter_error_manager_pub.h"
#include "securec.h"

namespace ut {
NegStubConfig g_negStub;

namespace {
    // EngineCtx KV：key = (comm, ctxTag)，value = calloc 分配的内存
    std::mutex g_kvMutex;
    std::map<std::pair<HcclComm, std::string>, std::pair<void*, uint64_t>> g_engineCtxKv;
    // 桩创建的子通信域句柄（HcclCommDestroy 只释放桩自建的句柄）
    std::set<HcclComm> g_fakeSubComms;
    // 子通信域句柄 -> 名称（HcclGetCommName 查询用）
    std::map<HcclComm, std::string> g_subCommNames;
    // SUT 注册的通信域状态回调
    HcclCommStateCallback g_stateCb = nullptr;
    void* g_stateCbArgs = nullptr;
    // 被注入驱逐的EngineCtx块（模拟并发销毁的条目），Reset统一回收避免模拟用例内存泄漏
    std::vector<void*> g_orphanCtxBlocks;
} // namespace

void NegStubReset()
{
    std::lock_guard<std::mutex> lock(g_kvMutex);
    for (auto& item : g_engineCtxKv) {
        free(item.second.first);
    }
    g_engineCtxKv.clear();
    for (void* ptr : g_orphanCtxBlocks) {
        free(ptr);
    }
    g_orphanCtxBlocks.clear();
    for (auto comm : g_fakeSubComms) {
        free(comm);
    }
    g_fakeSubComms.clear();
    g_subCommNames.clear();
    g_stateCb = nullptr;
    g_stateCbArgs = nullptr;
    g_negStub = NegStubConfig();
}

bool NegStubHasCtx(HcclComm comm, const std::string& tag)
{
    std::lock_guard<std::mutex> lock(g_kvMutex);
    return g_engineCtxKv.find({comm, tag}) != g_engineCtxKv.end();
}

HcclResult UtInvokeCommStateCb(HcclComm comm, HcclCommStatePhase phase)
{
    if (g_stateCb == nullptr) {
        return HCCL_SUCCESS;
    }
    return g_stateCb(comm, phase, g_stateCbArgs);
}
} // namespace ut

using namespace ut;

/* ---------------- ACL 运行时桩（extern "C"，最小假资源） ---------------- */
extern "C" {
aclError aclrtCreateStream(aclrtStream* stream)
{
    if (stream != nullptr) {
        *stream = calloc(1, sizeof(uint64_t)); // 假 stream，仅作非空句柄
    }
    return ACL_SUCCESS;
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    free(stream);
    return ACL_SUCCESS;
}

aclError aclrtMallocHost(void** hostPtr, size_t size)
{
    g_negStub.mallocHostCnt++; // 观测缓冲分配：懒建/扩容重建均计数
    if (hostPtr != nullptr) {
        *hostPtr = calloc(1, size);
    }
    return ACL_SUCCESS;
}

aclError aclrtFreeHost(void* hostPtr)
{
    g_negStub.freeHostCnt++; // 观测缓冲释放：扩容路径先释放旧缓冲
    free(hostPtr);
    return ACL_SUCCESS;
}

aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    (void)policy;
    if (devPtr != nullptr) {
        *devPtr = calloc(1, size); // 假 device 内存（普通内存），haclrtMemcpy 真实搬数
    }
    return ACL_SUCCESS;
}

aclError aclrtFree(void* devPtr)
{
    free(devPtr);
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtGetCurrentContext(aclrtContext* ctx)
{
    if (ctx != nullptr) {
        *ctx = nullptr; // UT 无真实 ctx，走 SUT 的跳过恢复分支
    }
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext ctx)
{
    (void)ctx;
    return ACL_SUCCESS;
}

aclError aclrtGetDevice(int32_t* deviceId)
{
    if (deviceId != nullptr) {
        *deviceId = 0;
    }
    return ACL_SUCCESS;
}

/* ---------------- HCCL/HCOMM 通信域与 EngineCtx 桩 ---------------- */
HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
{
    (void)comm;
    if (rankSize != nullptr) {
        *rankSize = g_negStub.rankSize;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t* rankId)
{
    (void)comm;
    if (rankId != nullptr) {
        *rankId = g_negStub.myRank;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommName(HcclComm comm, char* commName)
{
    if (g_negStub.getCommNameRet != HCCL_SUCCESS) {
        return g_negStub.getCommNameRet;
    }
    if (commName == nullptr) {
        return HCCL_E_PARA;
    }
    std::string name = g_negStub.commName;
    {
        std::lock_guard<std::mutex> lock(g_kvMutex);
        auto it = g_subCommNames.find(comm);
        if (it != g_subCommNames.end()) {
            name = it->second;
        }
    }
    errno_t ret = strcpy_s(commName, COMM_NAME_MAX_LENGTH, name.c_str());
    return (ret == EOK) ? HCCL_SUCCESS : HCCL_E_INTERNAL;
}

// 弱符号缺失target（UT_NEG_NO_SUBCOMM_SYMBOL）不定义本函数，模拟老版本HCOMM未提供该接口
#ifndef UT_NEG_NO_SUBCOMM_SYMBOL
HcclResult HcclCreateSubCommConfig(
    HcclComm* comm, uint32_t rankSize, uint32_t* rankIds, uint64_t rootRank, uint32_t myRank, HcclCommConfig* config,
    HcclComm* commOut)
{
    (void)rankSize;
    (void)rankIds;
    (void)rootRank;
    (void)myRank;
    if (comm == nullptr || config == nullptr || commOut == nullptr) {
        return HCCL_E_PARA;
    }
    *commOut = calloc(1, sizeof(uint64_t)); // 桩子通信域句柄
    {
        std::lock_guard<std::mutex> lock(g_kvMutex);
        g_fakeSubComms.insert(*commOut);
        g_subCommNames[*commOut] = std::string(config->hcclCommName);
    }
    g_negStub.createdSubComm = *commOut;
    g_negStub.createdSubCommName = std::string(config->hcclCommName);
    g_negStub.createSubCommCnt++;
    return HCCL_SUCCESS;
}
#endif

HcclResult HcclCommDestroy(HcclComm comm)
{
    if (comm == nullptr) {
        return HCCL_SUCCESS;
    }
    {
        std::lock_guard<std::mutex> lock(g_kvMutex);
        if (g_fakeSubComms.count(comm) > 0) {
            g_fakeSubComms.erase(comm);
            g_subCommNames.erase(comm);
            free(comm);
        }
    }
    g_negStub.commDestroyCnt++;
    return HCCL_SUCCESS;
}

HcclResult HcclCommResume(HcclComm comm)
{
    (void)comm;
    g_negStub.resumeCnt++;
    return g_negStub.resumeRet; // 注入返回值：覆盖RESUME_POST续用/失败降级两分支
}

bool HcommIsSupportHcclCommResume() { return true; }

bool HcommIsSupportHcclCommRegCommStateCallback() { return true; }

HcclResult HcclCommRegCommStateCallback(const char* regName, HcclCommStateCallback cb, void* args)
{
    (void)regName;
    g_stateCb = cb;
    g_stateCbArgs = args;
    g_negStub.regCbCnt++;
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    (void)engine;
    if (ctxTag == nullptr || ctx == nullptr) {
        return HCCL_E_PARA;
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
    return HCCL_SUCCESS;
}

HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size)
{
    (void)engine;
    ++g_negStub.ctxGetCnt;
    if (ctxTag == nullptr || ctx == nullptr || size == nullptr) {
        return HCCL_E_PARA;
    }
    std::lock_guard<std::mutex> lock(g_kvMutex);
    auto it = g_engineCtxKv.find({comm, std::string(ctxTag)});
    if (it == g_engineCtxKv.end()) {
        return HCCL_E_NOT_FOUND;
    }
    // 注入并发销毁：指定tag命中即摘除条目并按成功返回（出参保持命中语义，Probe阶段可正常入列），
    // 使Release阶段二次查询得NOT_FOUND；摘除的块由桩接管，Reset统一回收
    if (!g_negStub.ctxEvictTagOnGet.empty() && g_negStub.ctxEvictTagOnGet == ctxTag) {
        g_orphanCtxBlocks.push_back(it->second.first);
        *ctx = it->second.first;
        *size = it->second.second;
        g_engineCtxKv.erase(it);
        return HCCL_SUCCESS;
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

/* ---------------- 协商原语桩（虚拟多 rank 聚合语义） ---------------- */
HcclResult HcclAllReduce(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    (void)count;
    (void)dataType;
    (void)op;
    (void)comm;
    (void)stream;
    if (sendBuf == nullptr || recvBuf == nullptr) {
        return HCCL_E_PARA;
    }
    // MIN 聚合 = 全域 AND：本端值与注入的远端能力位取最小
    int32_t result = *static_cast<int32_t*>(sendBuf);
    for (int32_t remoteVal : g_negStub.remoteAllReduceVals) {
        result = (remoteVal < result) ? remoteVal : result;
    }
    *static_cast<int32_t*>(recvBuf) = result;
    g_negStub.allReduceCnt++;
    return HCCL_SUCCESS;
}

HcclResult HcclAllGather(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    (void)dataType;
    (void)comm;
    (void)stream;
    if (sendBuf == nullptr || recvBuf == nullptr) {
        return HCCL_E_PARA;
    }
    // rank0 = 本端载荷，rank1..N-1 = 注入的远端载荷
    errno_t ret = memcpy_s(recvBuf, sendCount, sendBuf, sendCount);
    if (ret != EOK) {
        return HCCL_E_INTERNAL;
    }
    uint8_t* base = static_cast<uint8_t*>(recvBuf);
    for (size_t i = 0; i < g_negStub.remoteGatherPayloads.size(); i++) {
        uint64_t offset = static_cast<uint64_t>(i + 1) * sendCount;
        size_t cpyLen = (g_negStub.remoteGatherPayloads[i].size() < sendCount) ?
                            g_negStub.remoteGatherPayloads[i].size() :
                            sendCount;
        ret = memcpy_s(base + offset, sendCount, g_negStub.remoteGatherPayloads[i].data(), cpyLen);
        if (ret != EOK) {
            return HCCL_E_INTERNAL;
        }
    }
    g_negStub.allGatherCnt++;
    return HCCL_SUCCESS;
}
} // extern "C"

/* ---------------- ErrorMgr 上下文桩（C++ 链接，与 adapter_error_manager_pub.h 声明一致） ---------------- */
ErrContext haclrtGetErrMgrContext(void) { return ErrContext{}; }

void haclrtSetErrMgrContext(ErrContext errorContext) { (void)errorContext; }

/* ---------------- 适配层内存接口（ops_hccl 命名空间，真实搬数） ---------------- */
namespace ops_hccl {
HcclResult haclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    (void)destMax;
    (void)kind;
    if (dst == nullptr || src == nullptr) {
        return HCCL_E_PARA;
    }
    return (memcpy(dst, src, count) == dst) ? HCCL_SUCCESS : HCCL_E_INTERNAL;
}

HcclResult haclrtMemset(void* dst, size_t destMax, int32_t value, size_t count)
{
    (void)destMax;
    if (dst == nullptr) {
        return HCCL_E_PARA;
    }
    memset(dst, value, count);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
