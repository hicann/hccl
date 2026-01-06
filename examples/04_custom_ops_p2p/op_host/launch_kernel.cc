/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common.h"
#include "load_kernel.h"
#include "launch_kernel.h"

namespace ops_hccl_p2p {

thread_local aclrtNotify g_notifies[AICPU_CONTROL_NOTIFY_NUM];

HcclResult LaunchKernel(OpParam &param, aclrtStream stream)
{
    // Host stream通知Device主thread
    ACLCHECK(aclrtRecordNotify(g_notifies[0], stream));

    // 执行device侧的算法编排
    std::string kernelName = "HcclLaunchP2PAicpuKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    ACLCHECK(aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle));
    ACLCHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));
    aclrtParamHandle paraHandle;
    ACLCHECK(aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle));
    ACLCHECK(aclrtKernelArgsFinalize(argsHandle));

    uint16_t NOTIFY_DEFAULT_WAIT_TIME = 27 * 68;   // notifywait默认1836等待时长
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr uint32_t blockDim = 1;

    ACLCHECK(aclrtLaunchKernelWithConfig(funcHandle, blockDim, stream, &cfg, argsHandle, nullptr));

    // Host stream等待Device的通知
    ACLCHECK(aclrtWaitAndResetNotify(g_notifies[1], stream, CUSTOM_TIMEOUT));

    return HCCL_SUCCESS;
}
}
