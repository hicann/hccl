/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "common.h"
#include "load_kernel.h"
#include "launch_kernel.h"

namespace ops_hccl_p2p {

thread_local aclrtNotify g_notifies[AICPU_CONTROL_NOTIFY_NUM];

HcclResult LaunchKernelWithAsc(OpParam &param, aclrtStream stream)
{
    // Host stream通知Device主thread
    ACLCHECK(aclrtRecordNotify(g_notifies[0], stream));

    // 调用使用 ASC 编译的函数，支持 <<<>>> 语法
    HcclResult ret = LaunchKernelAsc(param, stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[LaunchKernelWithAsc] Binary mode failed, ret[%d]", ret);
        return ret;
    }

    // Host stream等待Device的通知
    ACLCHECK(aclrtWaitAndResetNotify(g_notifies[1], stream, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchKernelWithAclrt(OpParam &param, aclrtStream stream)
{
    // 加载 AICPU Kernel，获取 AICPU 侧链接库的句柄
    CHK_RET(LoadAICPUKernel());

    // Host stream通知Device主thread
    ACLCHECK(aclrtRecordNotify(g_notifies[0], stream));

    // 获取 Kernel 函数句柄
    std::string kernelName = "HcclLaunchP2PAicpuKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    ACLCHECK(aclrtBinaryGetFunction(g_binKernelHandle, kernelName.c_str(), &funcHandle));

    // 构造 Kernel 函数入参
    ACLCHECK(aclrtKernelArgsInit(funcHandle, &argsHandle));
    aclrtParamHandle paraHandle;
    ACLCHECK(aclrtKernelArgsAppend(argsHandle, &param, sizeof(OpParam), &paraHandle));
    ACLCHECK(aclrtKernelArgsFinalize(argsHandle));

    // 下发 Kernel
    uint16_t NOTIFY_DEFAULT_WAIT_TIME = 27 * 68; // notifywait默认1836等待时长
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr uint32_t numBlocks = 1;
    ACLCHECK(aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stream, &cfg, argsHandle, nullptr));

    // Host stream等待Device的通知
    ACLCHECK(aclrtWaitAndResetNotify(g_notifies[1], stream, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(OpParam &param, aclrtStream stream)
{
    // 通过环境变量判断 Kernel 下发方式，默认使用 aclrt 接口方式
    char *kernelMode = getenv("HCCL_CUSTOM_KERNEL_LAUNCH_ASC");
    HCCL_INFO("[LaunchKernel] HCCL_CUSTOM_KERNEL_LAUNCH_ASC: %s", kernelMode);
    KernelLaunchMode mode = (kernelMode != nullptr && strcmp(kernelMode, "1") == 0)
            ? KERNEL_LAUNCH_ASC : KERNEL_LAUNCH_ACLRT;
    if (mode == KERNEL_LAUNCH_ASC) {
        // <<<>>> 尖括号调用方式
        HCCL_INFO("[LaunchKernel] Launching kernel with ascendc");
        return LaunchKernelWithAsc(param, stream);
    } else if (mode == KERNEL_LAUNCH_ACLRT) {
        // 传统 ACL API 方式
        HCCL_INFO("[LaunchKernel] Launching kernel with aclrt");
        return LaunchKernelWithAclrt(param, stream);
    } else {
        HCCL_ERROR("[LaunchKernel] Invalid launch mode: %d", static_cast<int>(mode));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
}
