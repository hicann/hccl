/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include "comm_negotiation.h"
#include "comm_worker_mgr.h"
#include "log.h"
#include "adapter_acl.h"
#include "acl/acl_rt.h"
#include "dlhcomm_function.h"

namespace ops_hccl {
namespace {
    constexpr uint32_t NEGOTIATION_DATA_COUNT = 1;
    constexpr uint32_t NEGOTIATION_CCL_BUFFER_SIZE = 1; // 单位MB
    constexpr int32_t NEGOTIATION_SUCCESS_VAL = 0x5a5a5a5a;
    constexpr int32_t NEGOTIATION_FAIL_VAL = 0x00000000;
    // 通信域状态回调统一注册名：同名覆盖幂等（cb与args恒定），回调按comm动态查询协商tag判定归属
    constexpr char NEGOTIATION_RESOURCE_CALLBACK[] = "comm_negotiation_resource_callback";
    // 协商子通信域静态标记tag（防嵌套主防线）：创建协商子通信域时打入EngineCtx，随comm生命周期销毁
    constexpr char NEGOTIATION_COMM_MARK_TAG[] = "negotiation_comm_mark";
    // HcclCommConfig.hcclOpExpansionMode取值: 0:默认 1:host 2:aicpu 3:aiv，与HcclOpExpansionMode枚举值空间不同
    static constexpr uint32_t COMM_CONFIG_OP_EXPANSION_AICPU = 2;

    // 协商资源上下文：按 (comm, typeTag) 懒创建，缓冲区按需扩容复用（只扩不缩）
    struct NegotiationResCtx {
        aclrtStream stream = nullptr;
        void* hostSendBuf = nullptr;
        void* hostRecvBuf = nullptr;
        void* deviceSendBuf = nullptr;
        void* deviceRecvBuf = nullptr;
        HcclComm subComm = nullptr;
        uint64_t sendBufSize = 0; // 已分配的发送缓冲字节数
        uint64_t recvBufSize = 0; // 已分配的接收缓冲字节数
    };

    // typeTag注册表：cleanup cb遍历已注册typeTag集合处理对应协商资源（tag不存在即跳过，天然支持多tag）
    std::mutex g_negTypeTagMutex;
    std::set<std::string> g_negTypeTags;

    void RegisterNegotiationTypeTag(const std::string& typeTag)
    {
        std::lock_guard<std::mutex> lock(g_negTypeTagMutex);
        g_negTypeTags.insert(typeTag);
    }

    std::set<std::string> GetNegotiationTypeTags()
    {
        std::lock_guard<std::mutex> lock(g_negTypeTagMutex);
        return g_negTypeTags;
    }

    // 协商资源懒创建互斥：避免并发首次协商重复建链/重复分配（CCU现状容忍并发，此处收敛为串行懒创建）。
    // 锁内含HcclCreateSubCommConfig调用，该接口按契约无需socket建链与rank信息交换（纯本地派生子通信域，
    // 见docs/zh/user_guide/api_comm_impl.md），故锁内无跨rank阻塞等待；真正的跨rank会合点（协商AllReduce）
    // 在锁外由CommWorker执行
    std::mutex g_negCtxMutex;

    void CleanupNegotiationRes(NegotiationResCtx* ctx)
    {
        if (ctx->deviceSendBuf != nullptr) {
            (void)aclrtFree(ctx->deviceSendBuf);
            ctx->deviceSendBuf = nullptr;
        }
        if (ctx->deviceRecvBuf != nullptr) {
            (void)aclrtFree(ctx->deviceRecvBuf);
            ctx->deviceRecvBuf = nullptr;
        }
        if (ctx->hostSendBuf != nullptr) {
            (void)aclrtFreeHost(ctx->hostSendBuf);
            ctx->hostSendBuf = nullptr;
        }
        if (ctx->hostRecvBuf != nullptr) {
            (void)aclrtFreeHost(ctx->hostRecvBuf);
            ctx->hostRecvBuf = nullptr;
        }
        ctx->sendBufSize = 0;
        ctx->recvBufSize = 0;
        if (ctx->stream != nullptr) {
            (void)aclrtDestroyStream(ctx->stream);
            ctx->stream = nullptr;
        }
        if (ctx->subComm != nullptr) {
            (void)HcclCommDestroy(ctx->subComm);
            ctx->subComm = nullptr;
        }
    }

    void FreeNegotiationBuf(NegotiationResCtx* negCtx)
    {
        if (negCtx->deviceSendBuf != nullptr) {
            (void)aclrtFree(negCtx->deviceSendBuf);
            negCtx->deviceSendBuf = nullptr;
            negCtx->sendBufSize = 0;
        }
        if (negCtx->deviceRecvBuf != nullptr) {
            (void)aclrtFree(negCtx->deviceRecvBuf);
            negCtx->deviceRecvBuf = nullptr;
            negCtx->recvBufSize = 0;
        }
        if (negCtx->hostSendBuf != nullptr) {
            (void)aclrtFreeHost(negCtx->hostSendBuf);
            negCtx->hostSendBuf = nullptr;
        }
        if (negCtx->hostRecvBuf != nullptr) {
            (void)aclrtFreeHost(negCtx->hostRecvBuf);
            negCtx->hostRecvBuf = nullptr;
        }
    }

    // 保证协商stream与收发缓冲就绪：容量不足时释放重建（只扩不缩），足够时复用
    HcclResult EnsureNegotiationStreamAndBuf(NegotiationResCtx* negCtx, uint64_t sendSize, uint64_t recvSize)
    {
        if (negCtx->stream == nullptr) {
            aclError aclRet = aclrtCreateStream(&negCtx->stream);
            CHK_PRT_RET(
                aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtCreateStream failed, ret[%d].", __func__, aclRet),
                HCCL_E_RUNTIME);
        }
        if (negCtx->sendBufSize >= sendSize && negCtx->recvBufSize >= recvSize && negCtx->hostSendBuf != nullptr
            && negCtx->hostRecvBuf != nullptr && negCtx->deviceSendBuf != nullptr && negCtx->deviceRecvBuf != nullptr) {
            return HCCL_SUCCESS;
        }
        FreeNegotiationBuf(negCtx);

        aclError aclRet = aclrtMallocHost(&negCtx->hostSendBuf, sendSize);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtMallocHost for sendBuf failed, ret[%d].", __func__, aclRet),
            HCCL_E_RUNTIME);
        aclRet = aclrtMallocHost(&negCtx->hostRecvBuf, recvSize);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtMallocHost for recvBuf failed, ret[%d].", __func__, aclRet),
            HCCL_E_RUNTIME);
        aclRet = aclrtMalloc(&negCtx->deviceSendBuf, sendSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtMalloc for deviceSendBuf failed, ret[%d].", __func__, aclRet),
            HCCL_E_RUNTIME);
        aclRet = aclrtMalloc(&negCtx->deviceRecvBuf, recvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtMalloc for deviceRecvBuf failed, ret[%d].", __func__, aclRet),
            HCCL_E_RUNTIME);
        negCtx->sendBufSize = sendSize;
        negCtx->recvBufSize = recvSize;
        return HCCL_SUCCESS;
    }

    // 协商缓冲清零：防残留值干扰当次协商结果
    HcclResult ClearNegotiationBuf(NegotiationResCtx* negCtx, uint64_t sendSize, uint64_t recvSize)
    {
        errno_t memRet = memset_s(negCtx->hostSendBuf, negCtx->sendBufSize, 0, sendSize);
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[%s] memset_s for hostSendBuf failed, ret[%d].", __func__, memRet),
            HCCL_E_MEMORY);
        memRet = memset_s(negCtx->hostRecvBuf, negCtx->recvBufSize, 0, recvSize);
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[%s] memset_s for hostRecvBuf failed, ret[%d].", __func__, memRet),
            HCCL_E_MEMORY);
        HcclResult setRet = haclrtMemset(negCtx->deviceSendBuf, negCtx->sendBufSize, 0, sendSize);
        CHK_PRT_RET(
            setRet != HCCL_SUCCESS,
            HCCL_ERROR("[%s] haclrtMemset for deviceSendBuf failed, ret[%d].", __func__, setRet), HCCL_E_RUNTIME);
        setRet = haclrtMemset(negCtx->deviceRecvBuf, negCtx->recvBufSize, 0, recvSize);
        CHK_PRT_RET(
            setRet != HCCL_SUCCESS,
            HCCL_ERROR("[%s] haclrtMemset for deviceRecvBuf failed, ret[%d].", __func__, setRet), HCCL_E_RUNTIME);
        return HCCL_SUCCESS;
    }

    // 协商子通信域打静态标记（随comm生命周期销毁，供观测与后续扩展）：防嵌套防线判定由
    // IsNegotiationComm名称后缀承担，打标失败不影响判定
    HcclResult MarkNegotiationComm(HcclComm subComm)
    {
        void* markPtr = nullptr;
        HcclResult ret = HcclEngineCtxCreate(
            subComm, NEGOTIATION_COMM_MARK_TAG, CommEngine::COMM_ENGINE_CPU_TS, sizeof(uint8_t), &markPtr);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] mark negotiation comm failed, ret[%d], fallback to name-suffix check.", __func__, ret);
            return HCCL_SUCCESS; // 标记失败不阻断协商：名称后缀快速路径仍可判定
        }
        *(static_cast<uint8_t*>(markPtr)) = 1;
        return HCCL_SUCCESS;
    }

    // 第一遍仅探测：存在任一协商tag才回收常驻worker（worker是comm级资源，仅回收一次）
    HcclResult ProbeNegotiationTags(HcclComm comm, const char* commName, std::vector<std::string>& foundTags)
    {
        std::set<std::string> typeTags = GetNegotiationTypeTags();
        for (auto& typeTag : typeTags) {
            std::string negTag = std::string(commName) + "_" + typeTag + NEGOTIATION_COMM_SUFFIX;
            void* ctxPtr = nullptr;
            uint64_t ctxSize = 0;
            HcclResult getRet
                = HcclEngineCtxGet(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
            if (getRet == HCCL_E_NOT_FOUND || (getRet == HCCL_SUCCESS && ctxPtr == nullptr)) {
                continue;
            }
            CHK_PRT_RET(
                getRet != HCCL_SUCCESS,
                HCCL_ERROR("[%s] HcclEngineCtxGet failed, negTag[%s], ret[%d].", __func__, negTag.c_str(), getRet),
                getRet);
            foundTags.push_back(negTag);
        }
        return HCCL_SUCCESS;
    }

    // 对探测到的协商tag逐个恢复（快恢）或清理（销毁）协商资源
    HcclResult ReleaseNegotiationCtx(
        HcclComm comm, HcclCommStatePhase state, const char* commName, const std::vector<std::string>& foundTags)
    {
        // 先回收常驻worker（等待在途任务收尾并join），再清理其正在使用的stream/subComm；
        // 所有ctx销毁路径均同步回收worker，维持tag不存在即worker已回收的不变式
        CommWorkerMgr::GetInstance().Remove(comm);

        for (auto& negTag : foundTags) {
            void* ctxPtr = nullptr;
            uint64_t ctxSize = 0;
            HcclResult getRet
                = HcclEngineCtxGet(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
            if (getRet != HCCL_SUCCESS || ctxPtr == nullptr) {
                // NOT_FOUND说明条目已被并发路径清理（非泄漏）；其余错误亦不中断，保证剩余tag全部回收
                HCCL_ERROR(
                    "[%s] HcclEngineCtxGet failed after worker removed, negTag[%s], ret[%d].", __func__, negTag.c_str(),
                    getRet);
                continue;
            }
            auto* ctx = static_cast<NegotiationResCtx*>(ctxPtr);
            // 快恢场景：task abort时子通信域已被Suspend+Clean，owner恢复完成后级联恢复协商子通信域，
            // 复用stream/buffer/worker等进程级资源；失败或不支持时降级走下方清理路径，下次协商懒重建
            if (state == HcclCommStatePhase::HCCL_COMM_STATE_PHASE_RESUME_POST && ctx->subComm != nullptr
                && HcommIsSupportHcclCommResume()) {
                HcclResult resumeRet = HcclCommResume(ctx->subComm);
                if (resumeRet == HCCL_SUCCESS) {
                    HCCL_INFO(
                        "[%s] negotiation subComm resumed, negTag[%s], subComm[%p].", __func__, negTag.c_str(),
                        ctx->subComm);
                    continue;
                }
                HCCL_ERROR(
                    "[%s] resume subComm failed, negTag[%s], ret[%d], fallback to rebuild.", __func__, negTag.c_str(),
                    resumeRet);
            }
            CleanupNegotiationRes(ctx);
            (void)HcclEngineCtxDestroy(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
        }
        HCCL_INFO(
            "[%s] negotiation resources released, commName[%s], tagNum[%zu].", __func__, commName, foundTags.size());
        return HCCL_SUCCESS;
    }

    // 通信域状态回调：销毁前(DESTROY_PRE)清理协商资源，快恢后(RESUME_POST)级联恢复协商子通信域；
    // 遍历已注册typeTag集合，按comm动态查询协商tag判定归属，tag不存在即跳过（天然支持多tag）
    HcclResult NegotiationCleanupCb(HcclComm comm, HcclCommStatePhase state, void* args)
    {
        (void)args; // 注册时不携带ownerComm，回调按comm动态查询协商tag判定归属
        if (state != HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_PRE
            && state != HcclCommStatePhase::HCCL_COMM_STATE_PHASE_RESUME_POST) {
            return HCCL_SUCCESS;
        }
        char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
        CHK_RET(HcclGetCommName(comm, commName));

        std::vector<std::string> foundTags;
        CHK_RET(ProbeNegotiationTags(comm, commName, foundTags));
        if (foundTags.empty()) {
            return HCCL_SUCCESS;
        }
        return ReleaseNegotiationCtx(comm, state, commName, foundTags);
    }

    // 构造协商子通信域配置：协商专用最小buffer + AICPU模式 + 截断至hcclCommName上限的子通信域名
    HcclResult BuildSubCommConfig(const std::string& negTag, HcclCommConfig& subCommConfig)
    {
        HcclCommConfigInit(&subCommConfig);
        subCommConfig.hcclBufferSize = NEGOTIATION_CCL_BUFFER_SIZE;
        subCommConfig.hcclOpExpansionMode = COMM_CONFIG_OP_EXPANSION_AICPU;
        // negTag超长时按尾部截断至hcclCommName上限，保留"_negotiation"后缀区分父子通信域名
        std::string subCommName = negTag;
        const uint32_t subCommNameMaxLen = sizeof(subCommConfig.hcclCommName) - 1; // 127字节
        if (subCommName.size() > subCommNameMaxLen) {
            subCommName = subCommName.substr(subCommName.size() - subCommNameMaxLen);
            HCCL_WARNING(
                "[%s] negTag exceeds max length, truncated, negTag[%s], subCommName[%s].", __func__, negTag.c_str(),
                subCommName.c_str());
        }
        auto nameRet
            = sprintf_s(subCommConfig.hcclCommName, sizeof(subCommConfig.hcclCommName), "%s", subCommName.c_str());
        CHK_PRT_RET(nameRet <= 0, HCCL_ERROR("[%s] sprintf_s for hcclCommName failed.", __func__), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    // 创建协商子通信域（含弱符号校验与调用方acl上下文保护），创建结果经出参subComm返回
    HcclResult CreateSubCommWithCtxGuard(
        HcclComm comm, u32 rankSize, std::vector<uint32_t>& rankIds, u32 myRank, HcclCommConfig& subCommConfig,
        HcclComm& subComm)
    {
        // HcclCreateSubCommConfig为HCOMM侧弱符号：老版本HCOMM（或ST桩环境）未提供时地址为空，
        // 按协商失败降级返回，禁止空调用段错误
        if (HcclCreateSubCommConfig == nullptr) {
            HCCL_WARNING("[%s] HcclCreateSubCommConfig not supported, negotiation degrade.", __func__);
            return HCCL_E_UNAVAIL;
        }

        // HcclCreateSubCommConfig内部调用aclrtSetDevice会切换为默认ctx，保存调用前ctx并在创建后恢复，
        // 保证协商stream/buffer（属调用方ctx）与执行ctx一致，且不污染调用线程
        aclrtContext curCtx = nullptr;
        aclError aclRet = aclrtGetCurrentContext(&curCtx);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtGetCurrentContext failed, ret[%d].", __func__, aclRet),
            HCCL_E_RUNTIME);

        HcclResult subRet
            = HcclCreateSubCommConfig(&comm, rankSize, rankIds.data(), 0, myRank, &subCommConfig, &subComm);
        // 无论创建成败，内部ctx切换均已发生，须先恢复；ctx为null时跳过（ST仿真无ctx，走降级路径）
        if (curCtx != nullptr) {
            aclRet = aclrtSetCurrentContext(curCtx);
            CHK_PRT_RET(
                aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtSetCurrentContext failed, ret[%d].", __func__, aclRet),
                HCCL_E_RUNTIME);
        }
        CHK_PRT_RET(
            subRet != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclCreateSubCommConfig failed, ret[%d].", __func__, subRet),
            HCCL_E_UNAVAIL);
        return HCCL_SUCCESS;
    }

    // 一次注册覆盖通信域全部阶段：销毁前(DESTROY_PRE)清理协商资源，快恢后(RESUME_POST)级联恢复协商子通信域；
    // 统一注册名+同名覆盖：各comm创建协商资源时重复注册均覆盖同一注册项（cb与args恒定，幂等），
    // 注册项不随comm数量累积；回调按comm动态查询协商tag判定归属，tag不存在即跳过
    HcclResult RegisterStateCallback()
    {
        if (!HcommIsSupportHcclCommRegCommStateCallback()) {
            return HCCL_SUCCESS;
        }
        HcclResult regRet = HcclCommRegCommStateCallback(NEGOTIATION_RESOURCE_CALLBACK, NegotiationCleanupCb, nullptr);
        CHK_PRT_RET(
            regRet != HCCL_SUCCESS, HCCL_ERROR("[%s] HcclCommRegCommStateCallback failed, ret[%d].", __func__, regRet),
            regRet);
        return HCCL_SUCCESS;
    }

    HcclResult
    CreateNegotiationSubCommAndRegCb(HcclComm comm, const std::string& negTag, u32 rankSize, NegotiationResCtx* negCtx)
    {
        std::vector<uint32_t> rankIds(rankSize);
        for (u32 i = 0; i < rankSize; i++) {
            rankIds[i] = i;
        }
        // rankIds按序包含父通信域全部rank（恒等映射），本rank在子通信域中的rank id即为父通信域rank id
        u32 myRank = 0;
        CHK_RET(HcclGetRankId(comm, &myRank));

        HcclCommConfig subCommConfig;
        CHK_RET(BuildSubCommConfig(negTag, subCommConfig));
        CHK_RET(CreateSubCommWithCtxGuard(comm, rankSize, rankIds, myRank, subCommConfig, negCtx->subComm));

        // 打协商子通信域静态标记（随comm生命周期销毁，供观测与后续扩展）：防嵌套防线由
        // IsNegotiationComm名称后缀判定，打标失败降级不阻断
        CHK_RET(MarkNegotiationComm(negCtx->subComm));
        return RegisterStateCallback();
    }

    // 按 (comm, typeTag) 懒创建/复用协商资源上下文；缓冲容量不足时扩容重建
    HcclResult GetNegotiationCtx(
        HcclComm comm, const char* commName, const std::string& typeTag, u32 rankSize, uint64_t sendSize,
        uint64_t recvSize, NegotiationResCtx*& negCtx)
    {
        std::string negTag = std::string(commName) + "_" + typeTag + NEGOTIATION_COMM_SUFFIX;
        std::lock_guard<std::mutex> lock(g_negCtxMutex);
        void* ctxPtr = nullptr;
        uint64_t ctxSize = 0;
        bool needCreate = false;
        HcclResult getRet = HcclEngineCtxGet(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &ctxPtr, &ctxSize);
        if (getRet != HCCL_SUCCESS || ctxPtr == nullptr || ctxSize < sizeof(NegotiationResCtx)) {
            needCreate = true;
            // Get成功说明存在残留/异常条目（指针空或尺寸不足）：HCOMM语义为同名tag已存在即拒绝创建
            // (HCCL_E_PARA)，不覆盖旧块，须先销毁再建，保证Create返回全新且足尺寸的块
            if (getRet == HCCL_SUCCESS) {
                (void)HcclEngineCtxDestroy(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
            }
            ctxSize = sizeof(NegotiationResCtx);
            CHK_RET(HcclEngineCtxCreate(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, ctxSize, &ctxPtr));
            negCtx = static_cast<NegotiationResCtx*>(ctxPtr);
            errno_t memsetRet = memset_s(negCtx, sizeof(NegotiationResCtx), 0, sizeof(NegotiationResCtx));
            CHK_PRT_RET(
                memsetRet != EOK, HCCL_ERROR("[%s] memset_s failed, ret[%d].", __func__, memsetRet), HCCL_E_MEMORY);
        } else {
            negCtx = static_cast<NegotiationResCtx*>(ctxPtr);
        }

        HcclResult ret = EnsureNegotiationStreamAndBuf(negCtx, sendSize, recvSize);
        if (ret != HCCL_SUCCESS) {
            // ctx销毁与worker回收同步，维持tag不存在即worker已回收的不变式（防御懒重建失败时worker存活）
            CommWorkerMgr::GetInstance().Remove(comm);
            CleanupNegotiationRes(negCtx);
            (void)HcclEngineCtxDestroy(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
            return ret;
        }
        if (needCreate || negCtx->subComm == nullptr) {
            RegisterNegotiationTypeTag(typeTag);
            ret = CreateNegotiationSubCommAndRegCb(comm, negTag, rankSize, negCtx);
            if (ret != HCCL_SUCCESS) {
                CommWorkerMgr::GetInstance().Remove(comm);
                CleanupNegotiationRes(negCtx);
                (void)HcclEngineCtxDestroy(comm, negTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS);
                return ret;
            }
            HCCL_INFO(
                "[%s] negotiation resources created, typeTag[%s], subComm[%p], stream[%p].", __func__, typeTag.c_str(),
                negCtx->subComm, negCtx->stream);
        }
        CHK_RET(ClearNegotiationBuf(negCtx, sendSize, recvSize));
        return HCCL_SUCCESS;
    }

    HcclResult RunInCommWorker(HcclComm comm, const std::function<HcclResult()>& task)
    {
        return CommWorkerMgr::GetInstance().Submit(comm, task);
    }

    // AllReduce(INT32, MIN) = 全域 AND 原语
    HcclResult ExecuteNegotiationAllReduceMin(
        HcclComm comm, NegotiationResCtx* negCtx, bool localCapability, bool& allRanksCapable)
    {
        int32_t localVal = localCapability ? NEGOTIATION_SUCCESS_VAL : NEGOTIATION_FAIL_VAL;
        int32_t result = 0;
        HcclResult threadRet = RunInCommWorker(comm, [&]() -> HcclResult {
            errno_t memRet = memcpy_s(negCtx->hostSendBuf, negCtx->sendBufSize, &localVal, sizeof(int32_t));
            CHK_PRT_RET(
                memRet != EOK,
                HCCL_ERROR("[ExecuteNegotiationAllReduceMin] memcpy_s for hostSendBuf failed, ret[%d].", memRet),
                HCCL_E_MEMORY);

            HcclResult cpyRet = haclrtMemcpy(
                negCtx->deviceSendBuf, negCtx->sendBufSize, negCtx->hostSendBuf, sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
            CHK_PRT_RET(
                cpyRet != HCCL_SUCCESS,
                HCCL_ERROR("[ExecuteNegotiationAllReduceMin] haclrtMemcpy H2D failed, ret[%d].", cpyRet),
                HCCL_E_RUNTIME);

            HcclResult arRet = HcclAllReduce(
                negCtx->deviceSendBuf, negCtx->deviceRecvBuf, NEGOTIATION_DATA_COUNT, HCCL_DATA_TYPE_INT32,
                HCCL_REDUCE_MIN, negCtx->subComm, negCtx->stream);
            CHK_PRT_RET(
                arRet != HCCL_SUCCESS,
                HCCL_ERROR("[ExecuteNegotiationAllReduceMin] HcclAllReduce failed, ret[%d].", arRet), arRet);

            aclError aclRet = aclrtSynchronizeStream(negCtx->stream);
            CHK_PRT_RET(
                aclRet != ACL_SUCCESS,
                HCCL_ERROR("[ExecuteNegotiationAllReduceMin] aclrtSynchronizeStream failed, ret[%d].", aclRet),
                HCCL_E_RUNTIME);

            cpyRet = haclrtMemcpy(
                negCtx->hostRecvBuf, negCtx->recvBufSize, negCtx->deviceRecvBuf, sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
            CHK_PRT_RET(
                cpyRet != HCCL_SUCCESS,
                HCCL_ERROR("[ExecuteNegotiationAllReduceMin] haclrtMemcpy D2H failed, ret[%d].", cpyRet),
                HCCL_E_RUNTIME);

            memRet = memcpy_s(&result, sizeof(int32_t), negCtx->hostRecvBuf, sizeof(int32_t));
            CHK_PRT_RET(
                memRet != EOK,
                HCCL_ERROR("[ExecuteNegotiationAllReduceMin] memcpy_s for result failed, ret[%d].", memRet),
                HCCL_E_MEMORY);
            return HCCL_SUCCESS;
        });
        CHK_RET(threadRet);

        allRanksCapable = (result == NEGOTIATION_SUCCESS_VAL);
        HCCL_INFO(
            "[ExecuteNegotiationAllReduceMin] localVal[0x%x], result[0x%x], allRanksCapable[%d].", localVal, result,
            static_cast<int>(allRanksCapable));
        return HCCL_SUCCESS;
    }

    // AllGather(UINT8) = 全域检视原语，收集各rank载荷到调用方recvBuf
    HcclResult ExecuteNegotiationAllGather(
        HcclComm comm, NegotiationResCtx* negCtx, const void* localPayload, uint64_t payloadSize, void* recvBuf,
        uint64_t recvBufSize, u32 rankSize)
    {
        uint64_t totalSize = static_cast<uint64_t>(rankSize) * payloadSize;
        HcclResult threadRet = RunInCommWorker(comm, [&]() -> HcclResult {
            errno_t memRet = memcpy_s(negCtx->hostSendBuf, negCtx->sendBufSize, localPayload, payloadSize);
            CHK_PRT_RET(
                memRet != EOK,
                HCCL_ERROR("[ExecuteNegotiationAllGather] memcpy_s for hostSendBuf failed, ret[%d].", memRet),
                HCCL_E_MEMORY);

            HcclResult cpyRet = haclrtMemcpy(
                negCtx->deviceSendBuf, negCtx->sendBufSize, negCtx->hostSendBuf, payloadSize,
                ACL_MEMCPY_HOST_TO_DEVICE);
            CHK_PRT_RET(
                cpyRet != HCCL_SUCCESS,
                HCCL_ERROR("[ExecuteNegotiationAllGather] haclrtMemcpy H2D failed, ret[%d].", cpyRet), HCCL_E_RUNTIME);

            HcclResult agRet = HcclAllGather(
                negCtx->deviceSendBuf, negCtx->deviceRecvBuf, payloadSize, HCCL_DATA_TYPE_UINT8, negCtx->subComm,
                negCtx->stream);
            CHK_PRT_RET(
                agRet != HCCL_SUCCESS,
                HCCL_ERROR("[ExecuteNegotiationAllGather] HcclAllGather failed, ret[%d].", agRet), agRet);

            aclError aclRet = aclrtSynchronizeStream(negCtx->stream);
            CHK_PRT_RET(
                aclRet != ACL_SUCCESS,
                HCCL_ERROR("[ExecuteNegotiationAllGather] aclrtSynchronizeStream failed, ret[%d].", aclRet),
                HCCL_E_RUNTIME);

            cpyRet = haclrtMemcpy(
                negCtx->hostRecvBuf, negCtx->recvBufSize, negCtx->deviceRecvBuf, totalSize, ACL_MEMCPY_DEVICE_TO_HOST);
            CHK_PRT_RET(
                cpyRet != HCCL_SUCCESS,
                HCCL_ERROR("[ExecuteNegotiationAllGather] haclrtMemcpy D2H failed, ret[%d].", cpyRet), HCCL_E_RUNTIME);
            return HCCL_SUCCESS;
        });
        CHK_RET(threadRet);

        errno_t memRet = memcpy_s(recvBuf, recvBufSize, negCtx->hostRecvBuf, totalSize);
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[ExecuteNegotiationAllGather] memcpy_s for recvBuf failed, ret[%d].", memRet),
            HCCL_E_MEMORY);
        return HCCL_SUCCESS;
    }
} // namespace

bool IsNegotiationComm(HcclComm comm)
{
    if (comm == nullptr) {
        return false;
    }
    // 名称后缀快速路径先行：普通通信域不带后缀直接false，零EngineCtx探测开销（本函数被hostDPU
    // 门控逐算子调用，而HcclEngineCtxGet未命中在HCOMM侧打WARNING，算子粒度探测会日志刷屏）。
    // 命中即true：mark静态标记与名称恒同时存在（negTag构造保证后缀保留），标记查询对返回值无
    // 增量贡献；mark打标失败的降级场景同样依赖名称兜底，防嵌套防线不受影响。用户通信域撞名后缀
    // 时误判为true只会保守跳过协商（fail-safe，不使能inline），无正确性风险
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    if (HcclGetCommName(comm, commName) != HCCL_SUCCESS) {
        return false;
    }
    std::string name(commName);
    return name.size() > NEGOTIATION_COMM_SUFFIX.size()
           && name.compare(
                  name.size() - NEGOTIATION_COMM_SUFFIX.size(), NEGOTIATION_COMM_SUFFIX.size(), NEGOTIATION_COMM_SUFFIX)
                  == 0;
}

HcclResult NegotiateAllReduceMin(HcclComm comm, const std::string& typeTag, bool localCapability, bool& allRanksCapable)
{
    CHK_PTR_NULL(comm);
    HCCL_INFO(
        "[%s] start, comm[%p], typeTag[%s], localCapability[%d].", __func__, comm, typeTag.c_str(),
        static_cast<int>(localCapability));

    u32 rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    if (rankSize <= 1) {
        HCCL_INFO("[%s] single rank, skip negotiation.", __func__);
        allRanksCapable = localCapability;
        return HCCL_SUCCESS;
    }

    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));

    NegotiationResCtx* negCtx = nullptr;
    CHK_RET(GetNegotiationCtx(comm, commName, typeTag, rankSize, sizeof(int32_t), sizeof(int32_t), negCtx));
    return ExecuteNegotiationAllReduceMin(comm, negCtx, localCapability, allRanksCapable);
}

HcclResult NegotiateAllGather(
    HcclComm comm, const std::string& typeTag, const void* localPayload, uint64_t payloadSize, void* recvBuf,
    uint64_t recvBufSize)
{
    CHK_PTR_NULL(comm);
    if (localPayload == nullptr || recvBuf == nullptr || payloadSize == 0) {
        HCCL_ERROR("[%s] invalid param, typeTag[%s], payloadSize[%llu].", __func__, typeTag.c_str(), payloadSize);
        return HCCL_E_PARA;
    }

    u32 rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    uint64_t totalSize = static_cast<uint64_t>(rankSize) * payloadSize;
    // 容量校验前置于rank分支：单rank（totalSize即payloadSize）与多rank统一按入参错误返回HCCL_E_PARA
    if (recvBufSize < totalSize) {
        HCCL_ERROR("[%s] recvBufSize[%llu] < rankSize * payloadSize[%llu].", __func__, recvBufSize, totalSize);
        return HCCL_E_PARA;
    }
    if (rankSize <= 1) {
        HCCL_INFO("[%s] single rank, copy local payload only.", __func__);
        errno_t memRet = memcpy_s(recvBuf, recvBufSize, localPayload, payloadSize);
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[%s] memcpy_s for recvBuf failed, ret[%d].", __func__, memRet), HCCL_E_MEMORY);
        return HCCL_SUCCESS;
    }

    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));

    NegotiationResCtx* negCtx = nullptr;
    CHK_RET(GetNegotiationCtx(comm, commName, typeTag, rankSize, payloadSize, totalSize, negCtx));
    return ExecuteNegotiationAllGather(comm, negCtx, localPayload, payloadSize, recvBuf, recvBufSize, rankSize);
}
} // namespace ops_hccl
