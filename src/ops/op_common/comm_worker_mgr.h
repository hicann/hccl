/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_WORKER_MGR_H
#define HCCL_COMM_WORKER_MGR_H

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include "comm_worker.h"
#include "hccl.h"

namespace ops_hccl {

// 进程级worker管理：deviceId -> comm -> worker（线程绑comm、comm绑device）；
// 读多写少：读锁查找复用，写锁创建/销毁；销毁回调无deviceId，Remove遍历各device桶摘除
class CommWorkerMgr {
public:
    static CommWorkerMgr& GetInstance();

    HcclResult Submit(HcclComm comm, const std::function<HcclResult()>& task);

    // comm销毁前调用：遍历各device桶摘除该comm全部worker（正常一个comm仅一个worker，
    // 极端时序下可能在多个桶各有残留），锁外等待在途任务收尾并join，避免长时间持写锁
    void Remove(HcclComm comm);

private:
    CommWorkerMgr() = default;
    ~CommWorkerMgr();
    CommWorkerMgr(const CommWorkerMgr&) = delete;
    CommWorkerMgr& operator=(const CommWorkerMgr&) = delete;

    std::shared_mutex mtx_{};
    std::unordered_map<int32_t, std::unordered_map<HcclComm, std::shared_ptr<CommWorker>>> workers_{};
};

} // namespace ops_hccl

#endif // HCCL_COMM_WORKER_MGR_H
