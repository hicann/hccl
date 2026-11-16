/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_STUCK_DETECT_ALG_PARAM_H
#define UT_STUCK_DETECT_ALG_PARAM_H

#include <cstdint>

namespace ops_hccl {

// 仅保留 CCU 卡死计数需要的字段（真实定义见 op_common/inc/alg_param.h）
struct CcuKernelArgBase {
    bool stuckCounterEnabled{false};
    uint64_t stuckHeadAddr{0};
    uint64_t stuckTailAddr{0};
    uint64_t stuckAddOneAddr{0};
    uint64_t stuckCounterToken{0};
    void* stuckOrigKernelFunc{nullptr};
};

// 仅保留 CCU 注册期选择 trampoline 需要的字段
struct CcuKernelInfo {
    char kernelFuncName[64]{};
    void* kernelFunc{nullptr};
    void* kernelArg{nullptr};
};

// 仅保留 AICPU 卡死计数需要的字段（真实定义见 op_common/inc/alg_param.h）
struct OpParam {
    uint64_t counterHeadAddr{0};
    uint64_t counterTailAddr{0};
    uint64_t counterAddOneAddr{0};
    uint32_t isEnableCounter{0};
};

} // namespace ops_hccl

#endif // UT_STUCK_DETECT_ALG_PARAM_H
