/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <sstream>
#include <memory>
#include <cstring>
#include <sched.h>
#include <hccl/hccl_comm.h>
#include "alg_param.h"
#include "executor_base.h"
#include "coll_alg_exec_registry.h"
#include "coll_alg_v2_exec_registry.h"
#include "hcomm_primitives.h"
#include "hcomm_primitives_dl.h"
#include "dfx/task_exception_fun.h"
#include "kernel_launch.h"
#include "hcomm_diag_dl.h"
#include "hcomm_device_profiling_dl.h"
#include <unordered_map>
#include <mutex>
#include <atomic>
#include "hccl_device_comm_dl.h"
#include "exec_timeout_manager.h"
#include "alg_data_trans_wrapper.h"
#include "aicpu_task_cache_key.h"
#include "aicpu_task_cache_comm_manager.h"
#include "aicpu_task_cache_utils.h"
#include "aicpu_task_cache_policy.h"
#include "ins_send_executor.h"
#include "ins_recv_executor.h"

using namespace ops_hccl;
namespace {
// 统计缓存信息
struct CacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};

    double hitRate() const
    {
        uint64_t total = hits + misses;
        return total > 0 ? static_cast<double>(hits) / total : 0.0;
    }

    void Reset()
    {
        hits = 0;
        misses = 0;
    }
};

// 通信域缓存
class CommDomainCache {
public:
    explicit CommDomainCache(const std::string& commName) : commName_(commName) {}

    const std::string& GetCommName() const { return commName_; }

    // 获得缓存项，返回共享所有权保证使用期间对象稳定存活
    std::shared_ptr<const AlgResourceCtxSerializable> Get(const std::string& algTag)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(algTag);
        return it != cache_.end() ? it->second : nullptr;
    }

    // 缓存算法
    void Put(const std::string& algTag, const AlgResourceCtxSerializable& value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[algTag] = std::make_shared<AlgResourceCtxSerializable>(value);
    }

    // 移除特定算法
    bool Remove(const std::string& algTag)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.erase(algTag) > 0;
    }

    // 清空所有缓存项
    void Clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }

    CacheStats& GetStats() { return stats_; }
    const CacheStats& GetStats() const { return stats_; }

    size_t GetCacheSize() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

private:
    std::string commName_;
    std::map<std::string, std::shared_ptr<const AlgResourceCtxSerializable>> cache_;
    CacheStats stats_;
    mutable std::mutex mutex_;
};

// 通信域缓存管理器
class CommDomainCacheManager {
public:
    // 获取算法缓存
    std::shared_ptr<const AlgResourceCtxSerializable> Get(const std::string& algTag, const std::string& paramCommName)
    {
        std::string commName = ExtractCommName(algTag);
        // 提取失败时使用参数中的commName
        if (commName.empty())
            commName = paramCommName;

        auto commCache = GetOrCreateComm(commName);
        if (commCache) {
            auto& stats = commCache->GetStats();
            auto result = commCache->Get(algTag);
            if (result) {
                stats.hits++;
                return result;
            }
            stats.misses++;
        }
        return nullptr;
    }

    // 缓存算法结果
    void Put(const std::string& algTag, const AlgResourceCtxSerializable& value, const std::string& paramCommName)
    {
        std::string commName = ExtractCommName(algTag);
        if (commName.empty())
            commName = paramCommName;

        auto commCache = GetOrCreateComm(commName);
        if (commCache) {
            commCache->Put(algTag, value);
        }
    }

    // 释放通信域缓存
    bool ReleaseComm(const std::string& commName)
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        return commCaches_.erase(commName) > 0;
    }

    // 获得通信域统计信息
    bool GetCommStats(const std::string& commName, CacheStats& outStats, size_t& outCacheSize) const
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        auto it = commCaches_.find(commName);
        if (it != commCaches_.end()) {
            outStats.hits = it->second->GetStats().hits.load();
            outStats.misses = it->second->GetStats().misses.load();
            outCacheSize = it->second->GetCacheSize();
            return true;
        }
        return false;
    }

    // 获得全局统计信息
    void GetGlobalStats(
        size_t& totalCommDomains, size_t& totalcacheEntries, uint64_t& totalHits, uint64_t& totalMisses) const
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        totalCommDomains = commCaches_.size();
        totalcacheEntries = 0;
        totalHits = 0;
        totalMisses = 0;
        for (const auto& pair : commCaches_) {
            const auto& commName = pair.first;
            const auto& commCache = pair.second;
            totalcacheEntries += commCache->GetCacheSize();
            totalHits += commCache->GetStats().hits.load();
            totalMisses += commCache->GetStats().misses.load();
        }
    }

    // 清空所有缓存
    void ClearAll()
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        commCaches_.clear();
    }

    // 从algTag中提取通信域名称
    std::string ExtractCommName(const std::string& algTag)
    {
        size_t firstUnderscore = algTag.find('_');
        if (firstUnderscore == std::string::npos)
            return "";

        size_t secondUnderscore = algTag.find('_', firstUnderscore + 1);
        if (secondUnderscore == std::string::npos)
            return "";

        return algTag.substr(firstUnderscore + 1, secondUnderscore - firstUnderscore - 1);
    }

private:
    // 获取或创建通信域缓存
    std::shared_ptr<CommDomainCache> GetOrCreateComm(const std::string& commName)
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        auto it = commCaches_.find(commName);
        if (it != commCaches_.end()) {
            return it->second;
        }
        auto result = commCaches_.emplace(commName, std::make_shared<CommDomainCache>(commName));
        return result.first->second;
    }

    mutable std::mutex mapMutex_;
    std::map<std::string, std::shared_ptr<CommDomainCache>> commCaches_;
};

// 全局缓存管理器实例
static CommDomainCacheManager g_cacheManager;

std::unique_ptr<AlgResourceCtxSerializable> DeserializeResCtx(const OpParam* param)
{
    std::unique_ptr<AlgResourceCtxSerializable> resCtx = std::make_unique<AlgResourceCtxSerializable>();
    char* ctx = static_cast<char*>(param->resCtx);
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    resCtx->DeSerialize(seq);
    return resCtx;
}
} // namespace

namespace ops_hccl {
// 选择走新（CollAlgExecRegistryV2）/老（CollAlgExecRegistry）算子流程
// A5芯片或者template名称前缀为"opv2_"（当前A2的HostNic Send/Recv使用）走新流程，其他芯片走老流程
bool IsOpsV2(const char* algName, HcclDevType deviceType)
{
    // 检查algName前缀是否为"opv2_"
    if (algName != nullptr) {
        const char* prefix = "opv2_";
        if (strncmp(algName, prefix, strlen(prefix)) == 0) {
            return true;
        }
    }

    // 根据deviceType判断
    if (shouldGoOutPlace(deviceType)) {
        return true;
    }

    return false;
}
} // namespace ops_hccl

inline HcclResult EnforceLaunchTask(const char* algTag)
{
    if (HcommBatchModeEnd(algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", algTag);
        return HCCL_E_INTERNAL;
    }
    if (HcommBatchModeStart(algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set batch mode, tag is %s.", algTag);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

inline HcclResult
OpOrchestrate(OpParam* param, const AlgResourceCtxSerializable* resCtxPtr, ThreadHandle thread, std::string& algName)
{
    // RTSQ等待时间: 与算子展开无关, 但resCtx固定该设置不会再变更
    if (HcommIsSupportHcommThreadResAcquireTimeOut()) {
        CHK_RET(HcclThreadResAcquireTimeOut(resCtxPtr->fullTimeout));
    }

    // NotifyWait等待时间: 只在算子展开过程中使用
    if (HcommIsSupportHcommSetNotifyWaitTimeOut()) {
        CHK_RET(HcclSetNotifyWaitTimeOut(resCtxPtr->waitTimeout));
    }

    // 主thread等待Host stream的通知
    u32 maxNotifyNum = resCtxPtr->notifyNumOnMainThread;
    if (!resCtxPtr->isHcclThreadAcquireWithConfigSupported) {
        for (u32 i = 0; i < resCtxPtr->notifyNumPerThread.size(); i++) {
            if (resCtxPtr->notifyNumPerThread[i] > maxNotifyNum) {
                maxNotifyNum = resCtxPtr->notifyNumPerThread[i];
            }
        }
    }
    HCCL_DEBUG(
        "[%s]Notify wait on thread[%llu], maxNotifyNum[%u], timeout[%u] s", __func__, thread, maxNotifyNum,
        resCtxPtr->waitTimeout);
    CHK_RET(HcclThreadNotifyWaitOnThreadDefault(thread, maxNotifyNum, resCtxPtr->waitTimeout));

    // 设置执行超时时间: 用于NotifyWait, 只在算子展开过程中使用
    ExecTimeoutManager::Instance().SetExecTimeout(param->opConfig.execTimeout);

    // 设置BatchTransfer是否可行: 只在算子展开过程中使用
    CHK_RET(InitHcommBatchTransferOnThreadSupported(resCtxPtr->isHcommBatchTransferOnThreadSupported));

    // 根据算法名字获取executor: 只用于算子展开
    std::shared_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param->opType, algName);
    if (executor.get() == nullptr) {
        HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
        return HCCL_E_INTERNAL;
    }

    // 执行算法编排: 只用于算子展开
    if (executor->Orchestrate(*param, *resCtxPtr) != HCCL_SUCCESS) {
        HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

static HcclResult HcclOrderLaunchNotifyRecord(const OpParam* param)
{
    ThreadHandle exportHostOrderThread = param->exportHostOrderThread;
    ThreadHandle deviceOrderThread = param->deviceOrderThread;
    HCCL_INFO(
        "[%s]. Before Notify1 Record, commName[%s], exportHostOrderThread is [0x%llx], deviceOrderThread is [0x%llx]",
        __func__, param->commName, exportHostOrderThread, deviceOrderThread);

    if (exportHostOrderThread != 0 && deviceOrderThread != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(deviceOrderThread, exportHostOrderThread, HOST_ORDER_THREAD_NOTIFY_IDX)));
        HCCL_INFO(
            "[%s]. After Notify1 Record deviceOrderThread is [0x%llx], exportHostOrderThread is [0x%llx]", __func__,
            deviceOrderThread, exportHostOrderThread);
    }

    return HCCL_SUCCESS;
}

extern "C" unsigned int HcclLaunchAicpuKernel(OpParam* param)
{
    // 修改当前进程的调度策略和优先级
    struct sched_param schedParam;
    schedParam.sched_priority = 0; // 设置优先级为0
    if (sched_setscheduler(0, SCHED_OTHER, &schedParam) == -1) {
        HCCL_ERROR("%s sched_setscheduler to SCHED_OTHER failed", __func__);
        return 1;
    }
    if (param == nullptr) {
        HCCL_ERROR("%s param is nullptr", __func__);
        return 1;
    }
    HCCL_INFO("Entry-%s, commName[%s], tag[%s], algTag[%s]", __func__, param->commName, param->tag, param->algTag);
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }

    // AICPU 按序下发
    CHK_RET(HcclOrderLaunchNotifyRecord(param));

    std::string algName = std::string(param->algName);
    if (!ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        ScatterOpInfo opInfo;
        if (CreateScatter(param, &opInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("%s CreateScatter fail", __func__);
            return 1;
        }

        if (HcommIsSupportHcommRegOpInfo()
            && HcommRegOpInfo(param->commName, reinterpret_cast<void*>(&opInfo), sizeof(ScatterOpInfo))
                   != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s HcommRegOpInfo fail, commName[%s], algTag[%s], size[%u]", __func__, param->commName, opInfo.algTag,
                sizeof(ScatterOpInfo));
            return 1;
        }

        if (HcommIsSupportHcommRegOpTaskException()
            && HcommRegOpTaskException(param->commName, ops_hccl::GetScatterOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s HcommRegOpTaskException fail, commName[%s], algTag[%s]", __func__, param->commName, param->algTag);
            return 1;
        }
    }

    if (ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        // 判断通信域状态
        HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
        if (HcommIsSupportHcclCommGetStatus()) {
            auto statusRet = HcclCommGetStatus(param->commName, &commStatus);
            if (statusRet != HCCL_SUCCESS) {
                HCCL_ERROR("%s HcclCommGetStatus fail, commName[%s], ret = %d", __func__, param->commName, statusRet);
                return 1;
            }
            if (commStatus == HCCL_COMM_STATUS_SUSPENDING) {
                if (HcommReleaseComm(param->commName) == HCCL_SUCCESS) {
                    HCCL_WARNING("%s commStatus is suspending, release commName[%s]", __func__, param->commName);
                } else {
                    HCCL_ERROR(
                        "%s commStatus is suspending, HcommReleaseComm fail, commName[%s]", __func__, param->commName);
                }
                return 301U; /* 301U: AICPUSUSPENDING_ERROR */
            }
            if (commStatus != HCCL_COMM_STATUS_READY) {
                HCCL_ERROR("%s commStatus is not ready!, commStatus = %d", __func__, static_cast<int>(commStatus));
                return 1;
            }
        }

        std::shared_ptr<const AlgResourceCtxSerializable> cachedResCtxHolder;
        std::unique_ptr<AlgResourceCtxSerializable> resCtx;
        const AlgResourceCtxSerializable* resCtxPtr{nullptr};
        u32 hitRateNum = 100;
        if (param->opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            // 通过缓存实现反序列化优化
            cachedResCtxHolder = g_cacheManager.Get(param->algTag, param->commName);
            if (cachedResCtxHolder != nullptr && IsResCtxCacheReusable(*cachedResCtxHolder, *param)) {
                HCCL_INFO("[%s] Cache HIT for algTag[%s]", __func__, param->algTag);
                std::string commName = g_cacheManager.ExtractCommName(param->algTag);
                if (commName.empty())
                    commName = param->commName;

                CacheStats stats;
                size_t cacheSize;
                if (g_cacheManager.GetCommStats(commName, stats, cacheSize)) {
                    HCCL_DEBUG(
                        "[%s] comm[%s] hitRate=%.2f%%, cacheSize=%zu", __func__, commName.c_str(),
                        stats.hitRate() * hitRateNum, cacheSize);
                }
                resCtxPtr = cachedResCtxHolder.get();
            } else {
                bool isStaleCache = (cachedResCtxHolder != nullptr);
                // 未命中或者通信域恢复后缓存失效，进行反序列化并存入缓存
                resCtx = DeserializeResCtx(param);
                g_cacheManager.Put(param->algTag, *resCtx, param->commName);
                resCtxPtr = resCtx.get();
                if (isStaleCache) {
                    HCCL_INFO(
                        "[%s] Cache STALE and refreshed for algTag[%s], cachedComm[%p], currentComm[%p]", __func__,
                        param->algTag, cachedResCtxHolder->commInfoPtr, param->hcclComm);
                } else {
                    HCCL_INFO("[%s] Cache MISS and stored for algTag[%s]", __func__, param->algTag);
                }
            }
        } else {
            resCtx = DeserializeResCtx(param);
            resCtxPtr = resCtx.get();
        }

        // 还原变长指针
        HcclResult ret = HCCL_SUCCESS;
        if (param->opType == HCCL_CMD_BATCH_SEND_RECV) {
            ret = ops_hccl::RestoreVarDataBatchSendRecv(*param);
        } else if (
            param->opType == HCCL_CMD_ALLTOALLV || param->opType == HCCL_CMD_ALLTOALLVC
            || param->opType == HCCL_CMD_ALLTOALL) {
            ret = ops_hccl::RestoreVarDataAlltoAllV(*param, *resCtxPtr);
        } else if (param->opType == HCCL_CMD_REDUCE_SCATTER_V) {
            ret = ops_hccl::RestoreVarDataReduceScatterV(*param, *resCtxPtr);
        } else if (param->opType == HCCL_CMD_ALLGATHER_V) {
            ret = ops_hccl::RestoreVarDataAllGatherV(*param, *resCtxPtr);
        }
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("failed to restore optype [%d] data and counts.", param->opType);
            return 1;
        }
        // 获取Device测主thread
        ThreadHandle thread = resCtxPtr->threads[0];
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        // 要在下第一个task之前上报
        HcclDfxOpInfoCompat dfxOpInfo{};
        if (ConvertToHcclDfxOpInfo(param, &dfxOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("ConvertToHcclDfxOpInfo fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }
        if (HcclDfxRegOpInfoByCommId(param->commName, reinterpret_cast<void*>(&dfxOpInfo)) != HCCL_SUCCESS) {
            HCCL_ERROR("HcclDfxRegOpInfoByCommId fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }

        // 上报上报mainstream数据,第一个任务
        if (HcommProfilingReportKernelStartTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%sfailed to report MainStream And FirstTask, thread %lu, param->commName %s.", __func__, thread,
                param->commName);
            return 1;
        }

        ThreadHandle exportedAicpuTsThread = param->opThread;

        // 检查aicpu task cache使能约束
        bool enableCache = false;
        if (HcommIsSupportHcommAicpuTsTaskCacheLookup()) {
            CHK_RET(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(*param, *resCtxPtr, enableCache));
        }

        // 打印算子信息用于调试
        static std::atomic<uint64_t> opUnfoldIdx{0};
        HCCL_INFO(
            "[HcclLaunchAicpuKernel] opUnfoldIdx[%llu] commName[%s] opType[%u] inputPtr[0x%016llx] inputSize[%llu] "
            "outputPtr[0x%016llx] outputSize[%llu] opMode[%u] algName[%s] isZeroCopy[%d] opExpanMode[%u] "
            "enableCache[%d]",
            opUnfoldIdx.fetch_add(1, std::memory_order_relaxed), param->commName, static_cast<uint32_t>(param->opType),
            param->inputPtr, param->inputSize, param->outputPtr, param->outputSize,
            static_cast<uint32_t>(param->opMode), param->algName, param->isZeroCopy,
            static_cast<uint32_t>(param->commOpExpansionMode), enableCache);

        if (enableCache) { // 使能aicpu task cache
            // 注意: OpOrchestrate尚未调用, 首个NotifyWait与算子展开相关的task尚未生成, AicpuTsThread中一定无SQE
            // 因此, 无需通过强制下发SQE, 来避免cache miss下缓存算法无关的task 或 cache hit下task下发乱序

            // 准备地址信息 (当前rank的userIn和userOut)
            constexpr uint64_t ADDRS_COUNT = 2;
            void* addrs[ADDRS_COUNT] = {param->inputPtr, param->outputPtr};
            uint64_t inputSize = 0;
            uint64_t outputSize = 0;
            CHK_RET(static_cast<HcclResult>(AicpuTaskCacheUtils::GetInputOutputInfoForCache(
                *param, resCtxPtr->topoInfo.userRankSize, inputSize, outputSize)));
            uint64_t sizes[ADDRS_COUNT] = {inputSize, outputSize};

            std::string cacheTag;
            bool isCacheHit = false;
            // 组装aicpu task cache tag
            CHK_RET(AicpuTaskCacheKey::GetAicpuTaskCacheTag(*param, inputSize, cacheTag));

            // 查询aicpu task cache
            if (HcommIsSupportHcommAicpuTsTaskCacheLookup()) {
                CHK_RET(static_cast<HcclResult>(HcommAicpuTsTaskCacheLookup(cacheTag.c_str(), &isCacheHit)));
            }
            HCCL_INFO("[HcclLaunchAicpuKernel] isCacheHit[%d] for cacheTag[%s]", isCacheHit, cacheTag.c_str());

            if (!isCacheHit) { // cache miss
                HcclResult cacheRet = HCCL_SUCCESS;
                do {
                    // 算子展开前, 通知aicpu task cache开始缓存task
                    if (HcommIsSupportHcommAicpuTsTaskCacheStart()) {
                        cacheRet = static_cast<HcclResult>(
                            HcommAicpuTsTaskCacheStart(cacheTag.c_str(), addrs, sizes, ADDRS_COUNT));
                        CHK_PRT_BREAK(
                            cacheRet != HCCL_SUCCESS,
                            HCCL_ERROR("[%s] HcommAicpuTsTaskCacheStart error, ret[%d]", __func__, cacheRet), (void)0);
                    }

                    // 设置算子展开相关的配置, 下发首个NotifyWait, 构造executor并执行算子展开
                    cacheRet = OpOrchestrate(param, resCtxPtr, thread, algName);
                    CHK_PRT_BREAK(
                        cacheRet != HCCL_SUCCESS, HCCL_ERROR("[%s] OpOrchestrate error, ret[%d]", __func__, cacheRet),
                        (void)0);

                    // 使用aicpu task cache后确保算子展开相关的SQE通过LaunchTask被缓存, 用于cache
                    // miss下避免缓存算法无关的task 注意: cache hit时, task刷新后直接下发, 这里无需强制下发 注意:
                    // hccl无法识别cache容量是否已满; 理论上如果cache容量满了, cache不使能, 无需强制下发
                    // (仅首次执行触发, 开销有限)
                    cacheRet = EnforceLaunchTask(param->algTag);
                    CHK_PRT_BREAK(
                        cacheRet != HCCL_SUCCESS,
                        HCCL_ERROR("[%s] EnforceLaunchTask error, ret[%d]", __func__, cacheRet), (void)0);

                    // 算子展开后, 通知aicpu task cache停止缓存task
                    if (HcommIsSupportHcommAicpuTsTaskCacheEnd()) {
                        cacheRet = static_cast<HcclResult>(HcommAicpuTsTaskCacheEnd(cacheTag.c_str()));
                        CHK_PRT_BREAK(
                            cacheRet != HCCL_SUCCESS,
                            HCCL_ERROR("[%s] HcommAicpuTsTaskCacheEnd error, ret[%d]", __func__, cacheRet), (void)0);
                    }
                } while (0);

                if (UNLIKELY(cacheRet != HCCL_SUCCESS)) {
                    if (HcommIsSupportHcommAicpuTsTaskCacheClear()) {
                        HCCL_ERROR("[%s] cache submit error, clear tag[%s]", __func__, cacheTag.c_str());
                        (void)HcommAicpuTsTaskCacheClear(cacheTag.c_str());
                    }
                    return cacheRet;
                }

                // 首次缓存记录通信域与tag关系
                AicpuTaskCacheCommManager::Instance().AddCommTagMap(param->hcclComm, cacheTag);
            } else { // cache hit
                // 刷新并下发task
                if (HcommIsSupportHcommAicpuTsTaskCacheExecute()) {
                    CHK_RET(static_cast<HcclResult>(
                        HcommAicpuTsTaskCacheExecute(cacheTag.c_str(), addrs, sizes, ADDRS_COUNT)));
                }
            }
        } else { // 不使能aicpu task cache
            // 设置算子展开相关的配置, 下发首个NotifyWait, 构造executor并执行算子展开
            CHK_RET(OpOrchestrate(param, resCtxPtr, thread, algName));
        }

        constexpr u32 DEFAULT_NOTIFY_IDX = 0;
        HCCL_DEBUG(
            "[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
            exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));

        // 上报mainstream数据,最后一个任务
        if (HcommProfilingReportKernelEndTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s failed to report MainStream And LastTask, thread %lu, param->commName %s.", __func__, thread,
                param->commName);
            return 1;
        }

        if (HcommProfilingReportDeviceOp(param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommProfilingReportDeviceOp fail, commName[%s]", __func__, param->commName);
            return 1;
        }

        if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
            return 1;
        }
    } else {
        std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
        if (executor.get() == nullptr) {
            HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
            return 1;
        }
        AlgResourceCtx* resCtx = reinterpret_cast<AlgResourceCtx*>(param->resCtx);
        // 获取Device测主thread
        ThreadHandle* threadHandlePtr
            = reinterpret_cast<ThreadHandle*>(reinterpret_cast<u8*>(resCtx) + sizeof(AlgResourceCtx));
        ThreadHandle thread = threadHandlePtr[0];
        ThreadHandle exportedAicpuTsThread = resCtx->opThread;
        u32 notifyNumOnMainThread = resCtx->notifyNumOnMainThread;
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        if (exportedAicpuTsThread != 0) {
            if (HcommProfilingInit(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to init Profiling");
                return 1;
            }

            // 上报主流和第一个task  wait之前
            if (HcommProfilingReportMainStreamAndFirstTask(thread) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to report MainStream And FirstTask");
                return 1;
            }

            // 主thread等待Host stream的通知
            HCCL_DEBUG(
                "[%s]Notify wait on thread[%llu], notifyNumOnMainThread[%u], timeout[%u] s", __func__, thread,
                notifyNumOnMainThread, CUSTOM_TIMEOUT);
            CHK_RET(
                static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, notifyNumOnMainThread, CUSTOM_TIMEOUT)));
        } else {
            if (HcommAclrtNotifyWaitOnThread(thread, resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to wait notify[%d] from host main stream", resCtx->notifyIds[0]);
                return 1;
            }
        }

        // 执行算法编排
        if (executor->Orchestrate(*param, resCtx) != HCCL_SUCCESS) {
            HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
            return 1;
        }

        if (exportedAicpuTsThread != 0) {
            // 上报device侧的op 附加信息
            HcomProInfoTmp profInfo;
            std::string algTypeStr(param->algTypeStr);
            if (strcpy_s(profInfo.algType, sizeof(profInfo.algType), algTypeStr.c_str()) != EOK) {
                HCCL_ERROR("[%s] strcpy_s profInfo.algType failed.", __func__);
                return 1;
            }
            if (strcpy_s(profInfo.commName, sizeof(profInfo.commName), param->commName) != EOK) {
                HCCL_ERROR("[%s] strcpy_s profInfo.commName failed.", __func__);
                return 1;
            }
            profInfo.commNameLen = strlen(param->commName);
            profInfo.dataCount = param->DataDes.count;
            profInfo.dataType = static_cast<uint8_t>(param->DataDes.dataType);
            profInfo.rankSize = resCtx->topoInfo.userRankSize;
            HcommProfilingReportDeviceHcclOpInfo(profInfo);

            // 主thread通知Host stream
            constexpr u32 DEFAULT_NOTIFY_IDX = 0;
            HCCL_DEBUG(
                "[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
                exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));

            // 上报主流和最后一个task 在notify之后
            if (HcommProfilingReportMainStreamAndLastTask(thread) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to report MainStream And LastTask");
                return 1;
            }

            if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
                HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
                return 1;
            }

            if (HcommProfilingEnd(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to End Profiling");
                return 1;
            }
        } else {
            if (HcommAclrtNotifyRecordOnThread(thread, resCtx->notifyIds[1]) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to record host main stream");
                return 1;
            }

            if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
                HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
                return 1;
            }
        }
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param->tag, param->algTag, param->commName);
    return 0;
}

extern "C" unsigned int HcclLaunchP2pAicpuKernel(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("%s args is nullptr", __func__);
        return 1;
    }
    HcclP2pKernelParam* params = static_cast<HcclP2pKernelParam*>(args);
    ThreadHandle sendRecvThread = params->sendRecvThread;
    void* paramPtr = static_cast<void*>(&params->opParams[0]);
    OpParam* param = static_cast<OpParam*>(paramPtr);

    if (param == nullptr) {
        HCCL_ERROR("%s param is nullptr", __func__);
        return 1;
    }

    if (param->opType != HcclCMDType::HCCL_CMD_SEND && param->opType != HcclCMDType::HCCL_CMD_RECEIVE) {
        HCCL_ERROR("%s only support SEND/RECV, opType[%d]", __func__, static_cast<int>(param->opType));
        return 1;
    }

    HCCL_INFO(
        "Entry-%s, commName[%s], tag[%s], algTag[%s], opType[%d]", __func__, param->commName, param->tag, param->algTag,
        static_cast<int>(param->opType));
    // 保留通信域管理 - 保证生命周期安全
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    std::string algName = std::string(param->algName);
    // 根据算法名字获取executor
    if (ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        // 判断通信域状态
        HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
        if (HcommIsSupportHcclCommGetStatus()) {
            auto statusRet = HcclCommGetStatus(param->commName, &commStatus);
            if (statusRet != HCCL_SUCCESS) {
                HCCL_ERROR("%s HcclCommGetStatus fail, commName[%s], ret = %d", __func__, param->commName, statusRet);
                return 1;
            }
            if (commStatus == HCCL_COMM_STATUS_SUSPENDING) {
                if (HcommReleaseComm(param->commName) == HCCL_SUCCESS) {
                    HCCL_WARNING("%s commStatus is suspending, release commName[%s]", __func__, param->commName);
                } else {
                    HCCL_ERROR(
                        "%s commStatus is suspending, HcommReleaseComm fail, commName[%s]", __func__, param->commName);
                }
                return 301U; /* 301U: AICPUSUSPENDING_ERROR */
            }
            if (commStatus != HCCL_COMM_STATUS_READY) {
                HCCL_ERROR("%s commStatus is not ready!, commStatus = %d", __func__, static_cast<int>(commStatus));
                return 1;
            }
        }

        std::shared_ptr<const AlgResourceCtxSerializable> cachedResCtxHolder;
        std::unique_ptr<AlgResourceCtxSerializable> resCtx;
        const AlgResourceCtxSerializable* resCtxPtr{nullptr};
        u32 hitRateNum = 100;

        // 通过缓存实现反序列化优化
        cachedResCtxHolder = g_cacheManager.Get(param->algTag, param->commName);
        if (cachedResCtxHolder != nullptr && IsResCtxCacheReusable(*cachedResCtxHolder, *param)) {
            HCCL_INFO("[%s] Cache HIT for algTag[%s]", __func__, param->algTag);
            std::string commName = g_cacheManager.ExtractCommName(param->algTag);
            if (commName.empty())
                commName = param->commName;

            CacheStats stats;
            size_t cacheSize;
            if (g_cacheManager.GetCommStats(commName, stats, cacheSize)) {
                HCCL_DEBUG(
                    "[%s] comm[%s] hitRate=%.2f%%, cacheSize=%zu", __func__, commName.c_str(),
                    stats.hitRate() * hitRateNum, cacheSize);
            }
            resCtxPtr = cachedResCtxHolder.get();
        } else {
            bool isStaleCache = (cachedResCtxHolder != nullptr);
            // 未命中或者通信域恢复后缓存失效，进行反序列化并存入缓存
            resCtx = DeserializeResCtx(param);
            g_cacheManager.Put(param->algTag, *resCtx, param->commName);
            resCtxPtr = resCtx.get();
            if (isStaleCache) {
                HCCL_INFO(
                    "[%s] Cache STALE and refreshed for algTag[%s], cachedComm[%p], currentComm[%p]", __func__,
                    param->algTag, cachedResCtxHolder->commInfoPtr, param->hcclComm);
            } else {
                HCCL_INFO("[%s] Cache MISS and stored for algTag[%s]", __func__, param->algTag);
            }
        }

        // 获取Device测主thread
        ThreadHandle thread = resCtxPtr->threads[0];
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        // 要在下第一个task之前上报
        HcclDfxOpInfoCompat dfxOpInfo{};
        if (ConvertToHcclDfxOpInfo(param, &dfxOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("ConvertToHcclDfxOpInfo fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }
        if (HcclDfxRegOpInfoByCommId(param->commName, (&dfxOpInfo)) != HCCL_SUCCESS) {
            HCCL_ERROR("HcclDfxRegOpInfoByCommId fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }

        // 上报上报mainstream数据,第一个任务
        if (HcommProfilingReportKernelStartTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%sfailed to report MainStream And FirstTask, thread %lu, param->commName %s.", __func__, thread,
                param->commName);
            return 1;
        }

        std::shared_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param->opType, algName);
        if (executor.get() == nullptr) {
            HCCL_ERROR(
                "Fail to find executor for algName[%s], opType[%d]", algName.c_str(), static_cast<int>(param->opType));
            HcommReleaseComm(param->commName);
            return 1;
        }

        ExecTimeoutManager::Instance().SetExecTimeout(param->opConfig.execTimeout);
        HcclResult ret = HCCL_SUCCESS;
        ret = executor->OrchestrateWithThread(*param, *resCtxPtr, sendRecvThread);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("orchestrate failed for alg:%s, opType[%d]", param->algName, static_cast<int>(param->opType));
            HcommReleaseComm(param->commName);
            return 1;
        }
        // 上报mainstream数据,最后一个任务
        if (HcommProfilingReportKernelEndTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s failed to report MainStream And LastTask, thread %lu, param->commName %s.", __func__, thread,
                param->commName);
            return 1;
        }
        if (HcommProfilingReportDeviceOp(param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommProfilingReportDeviceOp fail, commName[%s]", __func__, param->commName);
            return 1;
        }
        if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
            return 1;
        }
    } else {
        HCCL_ERROR(
            "%s P2P only support OpsV2, algName[%s], deviceType[%d]", __func__, param->algName,
            static_cast<int>(param->deviceType));
        HcommReleaseComm(param->commName);
        return 1;
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param->tag, param->algTag, param->commName);
    return 0;
}

HcclResult ops_hccl::RestoreVarDataBatchSendRecv(OpParam& param)
{
    u64 sendRecvItemSize = static_cast<u64>(sizeof(HcclSendRecvItem));
    u64 itemNum = static_cast<u64>(param.batchSendRecvDataDes.itemNum);
    if (param.varMemSize != itemNum * sendRecvItemSize) {
        HCCL_ERROR(
            "param.varMemSize[%lu] is not equal to itemNum[%lu] multiply [HcclSendRecvItem] size[%lu]."
            "Failed to restore end recv info for BatchSendRecv!",
            param.varMemSize, itemNum, sendRecvItemSize);
        return HCCL_E_PARA;
    }
    param.batchSendRecvDataDes.sendRecvItemsPtr = reinterpret_cast<HcclSendRecvItem*>(param.varData);
    return HCCL_SUCCESS;
}

HcclResult ops_hccl::RestoreVarDataAlltoAllV(OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    CHK_PRT_RET(
        param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR(
            "[RestoreVarDataAlltoAllV] param.varMemSize [%llu] is invalid,"
            " ALL_TO_ALL_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize, ALL_TO_ALL_V_VECTOR_NUM, rankSize, sizeof(u64)),
        HCCL_E_PARA);

    constexpr u32 ALL_TO_ALL_V_OFFSET_SCOUNTS = 0;
    constexpr u32 ALL_TO_ALL_V_OFFSET_RECV_COUNTS = 1;
    constexpr u32 ALL_TO_ALL_V_OFFSET_SDISPLS = 2;
    constexpr u32 ALL_TO_ALL_V_OFFSET_RDISPLS = 3;

    u64* data = reinterpret_cast<u64*>(param.varData);
    param.all2AllVDataDes.sendCounts = data;
    param.all2AllVDataDes.recvCounts = data + ALL_TO_ALL_V_OFFSET_RECV_COUNTS * rankSize;
    param.all2AllVDataDes.sdispls = data + ALL_TO_ALL_V_OFFSET_SDISPLS * rankSize;
    param.all2AllVDataDes.rdispls = data + ALL_TO_ALL_V_OFFSET_RDISPLS * rankSize;

    return HCCL_SUCCESS;
}

HcclResult ops_hccl::RestoreVarDataReduceScatterV(OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%u", rankSize);
    CHK_PRT_RET(
        param.varMemSize != REDUCE_SCATTER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR(
            "[RestoreVarDataReduceScatterV] param.varMemSize [%llu] is invalid,"
            "REDUCE_SCATTER_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize, REDUCE_SCATTER_V_VECTOR_NUM, rankSize, sizeof(u64)),
        HCCL_E_PARA);

    u64* data = reinterpret_cast<u64*>(param.varData);
    param.vDataDes.counts = data;
    param.vDataDes.displs = data + rankSize;
    return HCCL_SUCCESS;
}

HcclResult ops_hccl::RestoreVarDataAllGatherV(OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%u", rankSize);
    CHK_PRT_RET(
        param.varMemSize != ALL_GATHER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR(
            "[RestoreVarDataAllGatherV] param.varMemSize [%llu] is invalid,"
            "ALL_GATHER_V_VECTOR_NUM is [%u], rankSize is [%u], sizeof(u64) is [%u],",
            param.varMemSize, ALL_GATHER_V_VECTOR_NUM, rankSize, sizeof(u64)),
        HCCL_E_PARA);

    u64* data = reinterpret_cast<u64*>(param.varData);
    param.vDataDes.counts = data;
    for (u64 i = 0; i < rankSize; i++) {
        HCCL_INFO("param.vDataDes.counts[%u]:%u", i, reinterpret_cast<u64*>(param.vDataDes.counts)[i]);
    }
    param.vDataDes.displs = data + rankSize;
    for (u64 i = 0; i < rankSize; i++) {
        HCCL_INFO("param.vDataDes.displs[%u]:%u", i, reinterpret_cast<u64*>(param.vDataDes.displs)[i]);
    }
    return HCCL_SUCCESS;
}

extern "C" unsigned int HcclLaunchAicpuKernelA3(OpParam* param)
{
    if (param == nullptr) {
        HCCL_ERROR("%s param is nullptr", __func__);
        return 1;
    }
    HCCL_INFO("Entry-%s, commName[%s], tag[%s], algTag[%s]", __func__, param->commName, param->tag, param->algTag);
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }

    std::string algName = std::string(param->algName);
    if (!ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        ScatterOpInfo opInfo;
        if (CreateScatter(param, &opInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("%s CreateScatter fail", __func__);
            return 1;
        }

        if (HcommIsSupportHcommRegOpInfo()
            && HcommRegOpInfo(param->commName, reinterpret_cast<void*>(&opInfo), sizeof(ScatterOpInfo))
                   != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s HcommRegOpInfo fail, commName[%s], algTag[%s], size[%u]", __func__, param->commName, opInfo.algTag,
                sizeof(ScatterOpInfo));
            return 1;
        }

        if (HcommIsSupportHcommRegOpTaskException()
            && HcommRegOpTaskException(param->commName, ops_hccl::GetScatterOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s HcommRegOpTaskException fail, commName[%s], algTag[%s]", __func__, param->commName, param->algTag);
            return 1;
        }
    }

    // 根据算法名字获取executor
    if (ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        // 判断通信域状态
        HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
        if (HcommIsSupportHcclCommGetStatus()) {
            auto statusRet = HcclCommGetStatus(param->commName, &commStatus);
            if (statusRet != HCCL_SUCCESS) {
                HCCL_ERROR("%s HcclCommGetStatus fail, commName[%s], ret = %d", __func__, param->commName, statusRet);
                return 1;
            }
            if (commStatus == HCCL_COMM_STATUS_SUSPENDING) {
                if (HcommReleaseComm(param->commName) == HCCL_SUCCESS) {
                    HCCL_WARNING("%s commStatus is suspending, release commName[%s]", __func__, param->commName);
                } else {
                    HCCL_ERROR(
                        "%s commStatus is suspending, HcommReleaseComm fail, commName[%s]", __func__, param->commName);
                }
                return 301U; /* 301U: AICPUSUSPENDING_ERROR */
            }
            if (commStatus != HCCL_COMM_STATUS_READY) {
                HCCL_ERROR("%s commStatus is not ready!, commStatus = %d", __func__, static_cast<int>(commStatus));
                return 1;
            }
        }

        std::shared_ptr<const AlgResourceCtxSerializable> cachedResCtxHolder;
        std::unique_ptr<AlgResourceCtxSerializable> resCtx;
        const AlgResourceCtxSerializable* resCtxPtr{nullptr};
        if (param->opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            // 通过缓存实现反序列化优化
            cachedResCtxHolder = g_cacheManager.Get(param->algTag, param->commName);
            if (cachedResCtxHolder != nullptr && IsResCtxCacheReusable(*cachedResCtxHolder, *param)) {
                HCCL_INFO("[%s] Cache HIT for algTag[%s]", __func__, param->algTag);
                std::string commName = g_cacheManager.ExtractCommName(param->algTag);
                if (commName.empty())
                    commName = param->commName;

                CacheStats stats;
                size_t cacheSize;
                if (g_cacheManager.GetCommStats(commName, stats, cacheSize)) {
                    HCCL_DEBUG(
                        "[%s] comm[%s] hitRate=%.2f%%, cacheSize=%zu", __func__, commName.c_str(),
                        stats.hitRate() * 100, cacheSize);
                }
                resCtxPtr = cachedResCtxHolder.get();
            } else {
                bool isStaleCache = (cachedResCtxHolder != nullptr);
                // 未命中或者通信域恢复后缓存失效，进行反序列化并存入缓存
                resCtx = DeserializeResCtx(param);
                g_cacheManager.Put(param->algTag, *resCtx, param->commName);
                resCtxPtr = resCtx.get();
                if (isStaleCache) {
                    HCCL_INFO(
                        "[%s] Cache STALE and refreshed for algTag[%s], cachedComm[%p], currentComm[%p]", __func__,
                        param->algTag, cachedResCtxHolder->commInfoPtr, param->hcclComm);
                } else {
                    HCCL_INFO("[%s] Cache MISS and stored for algTag[%s]", __func__, param->algTag);
                }
            }
        } else {
            resCtx = DeserializeResCtx(param);
            resCtxPtr = resCtx.get();
        }

        // 还原变长指针
        HcclResult ret = HCCL_SUCCESS;
        if (param->opType == HCCL_CMD_BATCH_SEND_RECV) {
            ret = ops_hccl::RestoreVarDataBatchSendRecv(*param);
        } else if (
            param->opType == HCCL_CMD_ALLTOALLV || param->opType == HCCL_CMD_ALLTOALLVC
            || param->opType == HCCL_CMD_ALLTOALL) {
            ret = ops_hccl::RestoreVarDataAlltoAllV(*param, *resCtxPtr);
        } else if (param->opType == HCCL_CMD_REDUCE_SCATTER_V) {
            ret = ops_hccl::RestoreVarDataReduceScatterV(*param, *resCtxPtr);
        } else if (param->opType == HCCL_CMD_ALLGATHER_V) {
            ret = ops_hccl::RestoreVarDataAllGatherV(*param, *resCtxPtr);
        }
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("failed to restore optype [%d] data and counts.", param->opType);
            return 1;
        }
        // 获取Device测主thread
        ThreadHandle thread = resCtxPtr->threads[0];
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        // 要在下第一个task之前上报
        HcclDfxOpInfoCompat dfxOpInfo{};
        if (ConvertToHcclDfxOpInfo(param, &dfxOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("ConvertToHcclDfxOpInfo fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }
        if (HcclDfxRegOpInfoByCommId(param->commName, reinterpret_cast<void*>(&dfxOpInfo)) != HCCL_SUCCESS) {
            HCCL_ERROR("HcclDfxRegOpInfoByCommId fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }

        // 上报上报mainstream数据,第一个任务
        if (HcommProfilingReportKernelStartTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%sfailed to report MainStream And FirstTask, thread %lu, param->commName %s.", __func__, thread,
                param->commName);
            return 1;
        }

        // 主thread等待Host stream的通知
        ThreadHandle exportedAicpuTsThread = param->opThread;
        u32 maxNotifyNum = resCtxPtr->notifyNumOnMainThread;
        for (u32 i = 0; i < resCtxPtr->notifyNumPerThread.size(); i++) {
            if (resCtxPtr->notifyNumPerThread[i] > maxNotifyNum) {
                maxNotifyNum = resCtxPtr->notifyNumPerThread[i];
            }
        }
        HCCL_DEBUG(
            "[%s]Notify wait on thread[%llu], maxNotifyNum[%u], timeout[%u] s", __func__, thread, maxNotifyNum,
            CUSTOM_TIMEOUT);
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, maxNotifyNum, CUSTOM_TIMEOUT)));

        std::shared_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param->opType, algName);
        if (executor.get() == nullptr) {
            HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
            return 1;
        }

        // 设置执行超时时间
        ExecTimeoutManager::Instance().SetExecTimeout(param->opConfig.execTimeout);
        // 设置BatchTransfer是否可行
        CHK_RET(InitHcommBatchTransferOnThreadSupported(resCtxPtr->isHcommBatchTransferOnThreadSupported));
        // 执行算法编排
        if (executor->Orchestrate(*param, *resCtxPtr) != HCCL_SUCCESS) {
            HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
            return 1;
        }

        // 上报mainstream数据,最后一个任务
        if (HcommProfilingReportKernelEndTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s failed to report MainStream And LastTask, thread %lu, param->commName %s.", __func__, thread,
                param->commName);
            return 1;
        }

        constexpr u32 DEFAULT_NOTIFY_IDX = 0;
        HCCL_DEBUG(
            "[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
            exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));

        if (HcommProfilingReportDeviceOp(param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommProfilingReportDeviceOp fail, commName[%s]", __func__, param->commName);
            return 1;
        }

        if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
            return 1;
        }
    } else {
        std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
        if (executor.get() == nullptr) {
            HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
            return 1;
        }
        AlgResourceCtx* resCtx = reinterpret_cast<AlgResourceCtx*>(param->resCtx);
        // 获取Device测主thread
        ThreadHandle* threadHandlePtr
            = reinterpret_cast<ThreadHandle*>(reinterpret_cast<u8*>(resCtx) + sizeof(AlgResourceCtx));
        ThreadHandle thread = threadHandlePtr[0];
        ThreadHandle exportedAicpuTsThread = resCtx->opThread;
        u32 notifyNumOnMainThread = resCtx->notifyNumOnMainThread;
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        if (exportedAicpuTsThread != 0) {
            if (HcommProfilingInit(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to init Profiling");
                return 1;
            }

            // 上报主流和第一个task  wait之前
            if (HcommProfilingReportMainStreamAndFirstTask(thread) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to report MainStream And FirstTask");
                return 1;
            }

            // 主thread等待Host stream的通知
            HCCL_DEBUG(
                "[%s]Notify wait on thread[%llu], notifyNumOnMainThread[%u], timeout[%u] s", __func__, thread,
                notifyNumOnMainThread, CUSTOM_TIMEOUT);
            CHK_RET(
                static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, notifyNumOnMainThread, CUSTOM_TIMEOUT)));
        } else {
            if (HcommAclrtNotifyWaitOnThread(thread, resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to wait notify[%d] from host main stream", resCtx->notifyIds[0]);
                return 1;
            }
        }

        // 执行算法编排
        if (executor->Orchestrate(*param, resCtx) != HCCL_SUCCESS) {
            HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
            return 1;
        }

        if (exportedAicpuTsThread != 0) {
            // 上报device侧的op 附加信息
            HcomProInfoTmp profInfo;
            std::string algTypeStr(param->algTypeStr);
            if (strcpy_s(profInfo.algType, sizeof(profInfo.algType), algTypeStr.c_str()) != EOK) {
                HCCL_ERROR("[%s] strcpy_s profInfo.algType failed.", __func__);
                return 1;
            }
            if (strcpy_s(profInfo.commName, sizeof(profInfo.commName), param->commName) != EOK) {
                HCCL_ERROR("[%s] strcpy_s profInfo.commName failed.", __func__);
                return 1;
            }
            profInfo.commNameLen = strlen(param->commName);
            profInfo.dataCount = param->DataDes.count;
            profInfo.dataType = static_cast<uint8_t>(param->DataDes.dataType);
            profInfo.rankSize = resCtx->topoInfo.userRankSize;
            HcommProfilingReportDeviceHcclOpInfo(profInfo);

            // 主thread通知Host stream
            constexpr u32 DEFAULT_NOTIFY_IDX = 0;
            HCCL_DEBUG(
                "[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
                exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));

            // 上报主流和最后一个task 在notify之后
            if (HcommProfilingReportMainStreamAndLastTask(thread) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to report MainStream And LastTask");
                return 1;
            }

            if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
                HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
                return 1;
            }

            if (HcommProfilingEnd(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to End Profiling");
                return 1;
            }
        } else {
            if (HcommAclrtNotifyRecordOnThread(thread, resCtx->notifyIds[1]) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to record host main stream");
                return 1;
            }

            if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
                HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
                return 1;
            }
        }
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param->tag, param->algTag, param->commName);
    return 0;
}

extern "C" unsigned int HcclLaunchAicpuCacheEvictKernel(HcclComm* comm)
{
    if (comm == nullptr) {
        HCCL_ERROR("%s comm is nullptr", __func__);
        return 1;
    }
    HCCL_INFO("Entry-%s, comm[%p]", __func__, *comm);
    if (*comm != nullptr) {
        if (AicpuTaskCacheCommManager::Instance().EvictTaskCache(*comm) != HCCL_SUCCESS) {
            HCCL_ERROR("%s EvictTaskCache fail, comm[%p]", __func__, *comm);
            return 1;
        }
    }

    return 0;
}
