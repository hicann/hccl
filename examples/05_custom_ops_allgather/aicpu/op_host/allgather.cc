/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <iostream>

#include "log.h"
#include "utils.h"
#include "common.h"
#include "hccl_custom_allgather.h"
#include "load_kernel.h"
#include "launch_kernel.h"

using namespace ops_hccl_allgather;

namespace {
// aicpu主线程handle及其notify数量，缓存于COMM_ENGINE_CPU_TS(通信域host内存)context，供复用
struct AicpuMainThreadCache {
    ThreadHandle thread;
    uint32_t notifyNum;
};

HcclResult InitAicpuResource(HcclComm comm, OpParam& param)
{
    CommEngine engine = CommEngine::COMM_ENGINE_AICPU_TS;

    void* ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        // device资源已经存在，复用
        HCCL_INFO("[HcclAllGatherCustom] Engine context already exists");
        param.resCtxDevice = ctx;
        param.ctxSize = size;
        // aicpu主thread为comm级资源(非stream绑定)可复用；从通信域host内存context取回其句柄，重新导出到cpu引擎
        void* hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(AicpuMainThreadCache);
        CHK_RET(HcclEngineCtxGet(comm, param.tag, COMM_ENGINE_CPU_TS, &hostCtx, &hostCtxSize));
        auto* mainThreadCache = static_cast<AicpuMainThreadCache*>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &mainThreadCache->thread, COMM_ENGINE_CPU_TS, &param.aicpuThreadOnCpu));
        param.aicpuRecordCpuIdx = mainThreadCache->notifyNum;
    } else {
        // 不存在，新创建Context
        HCCL_INFO("[HcclAllGatherCustom] Creating engine context");
        AlgResourceCtx resCtxHost;

        // 申请资源Thread和Channel (comm级资源，可随context复用)
        CHK_RET(HcclAllocAlgResourceAICPU(comm, param, resCtxHost));

        // 使用threads[0]作为主AICPU thread，同时负责算法执行和host/device同步
        // 将aicpu主thread导出到cpu引擎，供host侧通知device使用
        CHK_RET(
            HcclThreadExportToCommEngine(comm, 1, &resCtxHost.threads[0], COMM_ENGINE_CPU_TS, &param.aicpuThreadOnCpu));
        param.aicpuRecordCpuIdx = resCtxHost.notifyNumOnMainThread;

        // 序列化并拷贝到device context
        CHK_RET(HcclMemcpyCtxHostToDevice(comm, param, resCtxHost, &param.resCtxDevice, &param.ctxSize));
        // 将aicpu主线程句柄缓存到通信域host内存context(COMM_ENGINE_CPU_TS)，供后续复用
        void* hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(AicpuMainThreadCache);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, COMM_ENGINE_CPU_TS, hostCtxSize, &hostCtx));
        auto* mainThreadCache = static_cast<AicpuMainThreadCache*>(hostCtx);
        mainThreadCache->thread = resCtxHost.threads[0];
        mainThreadCache->notifyNum = resCtxHost.notifyNumOnMainThread;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllGatherCustom(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather");
    if (ret <= 0) {
        HCCL_ERROR("[HcclAllGatherCustom] Failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(GetDeviceType(&param.devType));
    if (param.devType != DEVICE_TYPE_A5) {
        HCCL_ERROR("[HcclAllGatherCustom] Not Support Device Type [%u]", param.devType);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(HcclGetCommName(comm, param.commName));
    HCCL_INFO("[HcclAllGatherCustom] commName: %s", param.commName);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    // ==============================================
    // STEP 2: 申请host同步thread (从stream转换)
    // 该thread与stream绑定，每次调用都必须重新申请，不能随context复用
    // ==============================================
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &param.cpuThread));
    // 将host cpu thread导出到aicpu引擎，供device侧通知host使用(每次调用刷新)
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, COMM_ENGINE_AICPU_TS, &param.cpuThreadOnAicpu));

    // ==============================================
    // STEP 3: 创建/复用资源
    // ==============================================
    CHK_RET(InitAicpuResource(comm, param));
    // ==============================================
    // STEP 4: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(LaunchKernel(param, stream));

    return HCCL_SUCCESS;
}
