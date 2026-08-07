/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_comm.h"
#include "hccl_host_comm_dl.h"
#include "hcomm_dlsym.h"
#include "load_kernel.h"
#include "log.h"

using namespace ops_hccl;

constexpr u16 NOTIFY_DEFAULT_WAIT_TIME = 27 * 68; // 单位秒，notifywait默认1836等待时长

static inline HcclResult
LaunchKernelAndSyncStream_(aclrtFuncHandle funcHandle, aclrtArgsHandle argsHandle, aclrtStream stream)
{
    // 下发kernel
    aclrtLaunchKernelAttr attr{};
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    aclrtLaunchKernelCfg cfg{};
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;
    aclError ret = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stream, &cfg, argsHandle, nullptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[%s][aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed", __func__, ret),
        HCCL_E_RUNTIME);

    constexpr u32 streamTimeout = NOTIFY_DEFAULT_WAIT_TIME * 1000; // 单位毫秒
    ret = aclrtSynchronizeStreamWithTimeout(stream, streamTimeout);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] sync stream failed, errNo[0x%016llx]", __func__, ret), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult AicpuCacheEvictKernelLaunch(HcclComm comm)
{
    const char kernelName[] = "HcclLaunchAicpuCacheEvictKernel";
    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;

    // 共用libscatter_aicpu_kernel.so, 如果没有加载过，当前没有aicpu算子，直接返回即可。
    if (g_binKernelHandle == nullptr) {
        HCCL_INFO("[%s] aicpu file not loaded, ignore", __func__);
        return HCCL_SUCCESS;
    }
    // 获取function handle
    aclError ret = aclrtBinaryGetFunction(g_binKernelHandle, kernelName, &funcHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[aclrtBinaryGetFunction]errNo[0x%016llx] kernelName:%s", ret, kernelName),
        HCCL_E_RUNTIME);

    // 初始化和准备参数
    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] kernelName:%s", ret, kernelName),
        HCCL_E_RUNTIME);
    aclrtParamHandle paraHandle;
    ret = aclrtKernelArgsAppend(argsHandle, &comm, sizeof(comm), &paraHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsAppend]errNo[0x%016llx] kernelName:%s", ret, kernelName),
        HCCL_E_RUNTIME);
    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[aclrtKernelArgsFinalize]errNo[0x%016llx] kernelName:%s", ret, kernelName),
        HCCL_E_RUNTIME);

    // 创建流
    aclrtStream stream;
    ret = aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_FAST_SYNC);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] create stream failed, errNo[0x%016llx]", __func__, ret), HCCL_E_RUNTIME);

    HCCL_INFO("[%s]launch kernel[%s] comm[%p]", __func__, kernelName, comm);
    HcclResult result = LaunchKernelAndSyncStream_(funcHandle, argsHandle, stream);

    // 销毁流
    ret = aclrtDestroyStream(stream);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] destroy stream failed, errNo[0x%016llx]", __func__, ret);
        result = (result == HCCL_SUCCESS) ? HCCL_E_RUNTIME : result;
    }
    return result;
}

HcclResult AicpuTaskCacheCommStateCallback(HcclComm comm, HcclCommStatePhase state, void* args)
{
    (void)args;
    HCCL_INFO("[%s] comm[%p] state[%d]", __func__, comm, state);
    if (state == HCCL_COMM_STATE_PHASE_DESTROY_POST || state == HCCL_COMM_STATE_PHASE_RESUME_POST) {
        // 通信域销毁或者N秒快恢时，调用device接口，清理通信域相关的task缓存
        CHK_RET(AicpuCacheEvictKernelLaunch(comm));
    }

    return HCCL_SUCCESS;
}

__attribute__((constructor)) void RegisterAicpuTaskCacheCallback()
{
    // 确保dlsym符号已解析, HcommDlInit内部幂等，HcommIsSupportHcclCommRegCommStateCallback依赖
    HcommDlInit();
    const char REG_NAME[] = "aicpu_task_cache_callback";
    HCCL_INFO("[%s] start register comm state callback", __func__);
    if (HcommIsSupportHcclCommRegCommStateCallback()) {
        CHK_PRT(HcclCommRegCommStateCallback(REG_NAME, AicpuTaskCacheCommStateCallback, nullptr));
    }
}
