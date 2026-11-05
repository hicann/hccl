/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TEMPLATE_SELECTOR_TEST_AIV_UTILS_H
#define AIV_TEMPLATE_SELECTOR_TEST_AIV_UTILS_H

#include "alg_param.h"

namespace ops_hccl {
enum class KernelArgsType { ARGS_TYPE_SERVER, ARGS_TYPE_TWO_SHOT };

struct AivOpArgs {
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_ALLTOALL;
    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;
    bool isWrite = false;
    u64 input = 0;
    u64 output = 0;
    u32 rank = 0;
    u32 rankSize = 0;
    u64 count = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    u32 op = 0;
    u32 root = 0;
    u32 sliceId = 0;
    void* buffersIn = nullptr;
    void* stream = nullptr;
    HcclComm hcclComm = nullptr;
    bool isOpBase = false;
    u32 numBlocks = 0;
    u64 inputSliceStride = 0;
    u64 outputSliceStride = 0;
    u64 repeatNum = 0;
    u64 inputRepeatStride = 0;
    u64 outputRepeatStride = 0;
};

namespace aiv_selector_test {
    inline AivOpArgs launchedArgs;
    inline u32 kernelLaunches = 0;
} // namespace aiv_selector_test

inline HcclResult ExecuteKernelLaunch(const AivOpArgs& args)
{
    aiv_selector_test::launchedArgs = args;
    ++aiv_selector_test::kernelLaunches;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

#endif // AIV_TEMPLATE_SELECTOR_TEST_AIV_UTILS_H
