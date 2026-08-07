/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_AICPU_TIMEOUT_H
#define OPS_HCCL_AICPU_TIMEOUT_H

#include <cstdint>
#include <limits>

namespace ops_hccl {

constexpr uint32_t AICPU_FULL_TIMEOUT_OFFSET = 20;
constexpr uint32_t AICPU_KERNEL_TIMEOUT_OFFSET = 30;
constexpr uint32_t AICPU_HOST_NOTIFY_TIMEOUT_OFFSET = 50;

struct AicpuTimeout {
    uint32_t waitTimeout = 0;
    uint32_t fullTimeout = 0;
    uint32_t hostNotifyTimeout = 0;
    uint16_t kernelLaunchTimeout = 0;
};

inline uint32_t AddAicpuTimeoutOffset(uint32_t timeout, uint32_t offset)
{
    if (timeout == 0) {
        return 0;
    }
    if (timeout > std::numeric_limits<uint32_t>::max() - offset) {
        return std::numeric_limits<uint32_t>::max();
    }
    return timeout + offset;
}

inline uint16_t ToKernelLaunchTimeout(uint32_t timeout)
{
    if (timeout > std::numeric_limits<uint16_t>::max()) {
        return std::numeric_limits<uint16_t>::max();
    }
    return static_cast<uint16_t>(timeout);
}

inline AicpuTimeout DeriveAicpuTimeout(uint32_t execTimeout)
{
    AicpuTimeout timeout;
    timeout.waitTimeout = execTimeout;
    timeout.fullTimeout = AddAicpuTimeoutOffset(execTimeout, AICPU_FULL_TIMEOUT_OFFSET);
    timeout.hostNotifyTimeout = AddAicpuTimeoutOffset(execTimeout, AICPU_HOST_NOTIFY_TIMEOUT_OFFSET);
    timeout.kernelLaunchTimeout
        = ToKernelLaunchTimeout(AddAicpuTimeoutOffset(execTimeout, AICPU_KERNEL_TIMEOUT_OFFSET));
    return timeout;
}

} // namespace ops_hccl

#endif // OPS_HCCL_AICPU_TIMEOUT_H
