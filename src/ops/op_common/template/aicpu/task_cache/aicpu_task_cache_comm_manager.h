/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_TASK_CACHE_COMM_MANAGER_H
#define HCCL_AICPU_TASK_CACHE_COMM_MANAGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <shared_mutex>

#include "hccl_comm.h"

namespace ops_hccl {

class AicpuTaskCacheCommManager {
public:
    static AicpuTaskCacheCommManager &Instance();

    // 记录通信域与tag的关系
    void AddCommTagMap(HcclComm comm, const std::string &tagName);

    // 清除特定通信域的缓存
    HcclResult EvictTaskCache(HcclComm comm);

private:
    AicpuTaskCacheCommManager() = default;
    ~AicpuTaskCacheCommManager() = default;

    // 禁用拷贝和移动操作
    AicpuTaskCacheCommManager(const AicpuTaskCacheCommManager &) = delete;
    AicpuTaskCacheCommManager &operator=(const AicpuTaskCacheCommManager &) = delete;
    AicpuTaskCacheCommManager(AicpuTaskCacheCommManager &&) = delete;
    AicpuTaskCacheCommManager &operator=(AicpuTaskCacheCommManager &&) = delete;

    std::unordered_map<HcclComm, std::vector<std::string>> commToTagMap_;
    mutable std::shared_timed_mutex mutex_;
};
} // namespace ops_hccl
#endif
