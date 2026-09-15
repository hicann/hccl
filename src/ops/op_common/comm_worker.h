/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_WORKER_H
#define HCCL_COMM_WORKER_H

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include "acl/acl_rt.h"
#include "hccl.h"

namespace ops_hccl {

// comm级常驻worker：替代"每次协商临时建线程+join"，首个任务到达时创建，后续复用
class CommWorker {
public:
    CommWorker() = default;
    ~CommWorker();
    CommWorker(const CommWorker&) = delete;
    CommWorker& operator=(const CommWorker&) = delete;

    HcclResult Start();

    // 投递任务并阻塞等待结果，语义与"临时线程+join"一致；单任务槽，同一worker同时最多一个在途任务
    HcclResult Submit(aclrtContext ctx, const std::function<HcclResult()>& task);

    // 幂等：在途任务先执行完并回传结果，worker再退出，不丢结果
    void Shutdown();

private:
    void WorkerLoop();

    std::thread thread_{};
    std::mutex mtx_{};
    std::condition_variable cv_{};
    std::function<HcclResult()> task_{};
    aclrtContext ctx_{nullptr};
    HcclResult result_{HCCL_E_INTERNAL};
    bool hasTask_{false};
    bool stop_{false};
};

} // namespace ops_hccl

#endif // HCCL_COMM_WORKER_H
