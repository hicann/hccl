/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exec_timeout_manager.h"
#include "log.h"
#include "alg_param.h"

namespace ops_hccl {

ExecTimeoutManager::ExecTimeoutManager() : execTimeout_(CUSTOM_TIMEOUT), timeoutSet_(false)
{
    HCCL_INFO("[ExecTimeoutManager] Initialized with default timeout: %u seconds", CUSTOM_TIMEOUT);
}

ExecTimeoutManager::~ExecTimeoutManager() {}

ExecTimeoutManager& ExecTimeoutManager::Instance()
{
    static ExecTimeoutManager instance;
    return instance;
}

void ExecTimeoutManager::SetExecTimeout(u32 execTimeout)
{
    u32 timeoutValue = execTimeout;
    execTimeout_.store(timeoutValue, std::memory_order_relaxed);
    timeoutSet_.store(true, std::memory_order_relaxed);
    HCCL_INFO("[ExecTimeoutManager] Setting exec timeout to: %u seconds", timeoutValue);
}

u32 ExecTimeoutManager::GetExecTimeout()
{
    bool isSet = timeoutSet_.load(std::memory_order_relaxed);
    u32 timeout = isSet ? execTimeout_.load(std::memory_order_relaxed) : CUSTOM_TIMEOUT;
    HCCL_DEBUG("[ExecTimeoutManager] Getting exec timeout: %u seconds.", timeout);
    return timeout;
}

} // namespace ops_hccl
