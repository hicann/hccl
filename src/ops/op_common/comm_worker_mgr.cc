/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_worker_mgr.h"
#include <vector>
#include "log.h"

namespace ops_hccl {

CommWorkerMgr& CommWorkerMgr::GetInstance()
{
    static CommWorkerMgr instance;
    return instance;
}

CommWorkerMgr::~CommWorkerMgr()
{
    // 进程退出兜底，防joinable线程terminate。锁内仅swap搬空map，join全部放锁外，
    // 避免worker嵌套Submit申请读锁时与写锁互等成环
    std::unordered_map<int32_t, std::unordered_map<HcclComm, std::shared_ptr<CommWorker>>> deleteWorkers;
    {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        deleteWorkers.swap(workers_); // 空箱换满箱：锁内零join、零分配、零节点销毁
    }
    for (auto& deviceItem : deleteWorkers) {
        for (auto& commItem : deviceItem.second) {
            HCCL_WARNING(
                "[%s] worker still alive at process exit, fallback shutdown, comm[%p], deviceId[%d].", __func__,
                commItem.first, deviceItem.first);
            commItem.second->Shutdown(); // 锁外join：在途任务先收尾，worker再退出
        }
    }
} // deleteWorkers析构 → ~CommWorker(幂等Shutdown)，仍在锁外

HcclResult CommWorkerMgr::Submit(HcclComm comm, const std::function<HcclResult()>& task)
{
    aclrtContext curCtx = nullptr;
    aclError aclRet = aclrtGetCurrentContext(&curCtx);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtGetCurrentContext failed, ret[%d].", __func__, aclRet),
        HCCL_E_RUNTIME);

    // 通信域归属device取发起线程当前绑定device（发起协商的线程必已绑定comm对应device）
    int32_t deviceId = -1;
    aclError devRet = aclrtGetDevice(&deviceId);
    CHK_PRT_RET(
        devRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclrtGetDevice failed, ret[%d].", __func__, devRet), HCCL_E_RUNTIME);

    // 快路径：读锁查找，首次创建后每次协商都走这里
    std::shared_ptr<CommWorker> worker;
    {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        auto devIter = workers_.find(deviceId);
        if (devIter != workers_.end()) {
            auto iter = devIter->second.find(comm);
            if (iter != devIter->second.end()) {
                worker = iter->second;
            }
        }
    }
    // 慢路径：写锁创建，二次检查防止并发重复创建
    if (worker == nullptr) {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        auto& deviceWorkers = workers_[deviceId];
        auto iter = deviceWorkers.find(comm);
        if (iter == deviceWorkers.end()) {
            std::shared_ptr<CommWorker> newWorker(new (std::nothrow) CommWorker());
            CHK_PRT_RET(newWorker == nullptr, HCCL_ERROR("[%s] new CommWorker failed.", __func__), HCCL_E_MEMORY);
            HcclResult startRet = newWorker->Start();
            CHK_RET(startRet);
            worker = newWorker;
            deviceWorkers.emplace(comm, std::move(newWorker));
            HCCL_INFO("[%s] worker created for comm[%p] deviceId[%d].", __func__, comm, deviceId);
        } else {
            worker = iter->second;
        }
    }
    // 持有shared_ptr调用：与Remove并发时对象存活到本次Submit返回
    return worker->Submit(curCtx, task);
}

void CommWorkerMgr::Remove(HcclComm comm)
{
    std::vector<std::shared_ptr<CommWorker>> removedWorkers;
    {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        for (auto& deviceItem : workers_) {
            auto iter = deviceItem.second.find(comm);
            if (iter != deviceItem.second.end()) {
                removedWorkers.push_back(iter->second);
                (void)deviceItem.second.erase(iter);
            }
        }
    }
    for (auto& worker : removedWorkers) {
        worker->Shutdown();
        HCCL_INFO("[%s] worker removed for comm[%p].", __func__, comm);
    }
}

} // namespace ops_hccl
