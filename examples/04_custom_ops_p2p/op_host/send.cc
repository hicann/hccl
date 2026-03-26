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
#include "utils.h"
#include "common.h"
#include "hccl_custom_p2p.h"
#include "load_kernel.h"
#include "launch_kernel.h"

using namespace ops_hccl_p2p;

HcclResult HcclSendCustom(
    void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", "hccl_custom_p2p");
    if (ret <= 0) {
        HCCL_ERROR("[HcclSendCustom] Failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(HcclGetCommName(comm, param.commName));

    param.inputPtr = sendBuf;
    param.count = count;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SEND;

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    uint32_t rank, rankSize;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    CHK_RET(GetDeviceType(&param.devType));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine engine = CommEngine::COMM_ENGINE_AICPU;

    void * ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtx);
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        // device资源已经存在
        HCCL_INFO("[HcclSendCustom] Engine context already exists");
        param.resCtx = static_cast<AlgResourceCtx *>(ctx);
    } else {
        // 不存在，新创建Context
        HCCL_INFO("[HcclSendCustom] Creating engine context");
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, size, &ctx));
        param.resCtx = static_cast<AlgResourceCtx *>(ctx);
        AlgResourceCtx resCtxHost;

        // ==============================================
        // STEP 2.1: 申请thread
        // ==============================================
        // 将传入的stream转换为thread，并申请1个notify；同时导出为AICPU上可用的thread
        CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &param.cpuThread));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, COMM_ENGINE_AICPU_TS, &resCtxHost.cpuThreadOnAicpu));

        // 创建一个AICPU_TS类型的thread，并申请1个notify；同时导出为CPU上可用的thread
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 1, &resCtxHost.aicpuThread));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, COMM_ENGINE_CPU_TS, &param.aicpuThreadOnCpu));

        // ==============================================
        // STEP 2.2: 建立通信链路Channel，两个 rank 之间建立 1 个 channel
        // ==============================================
        CHK_RET(AcquireChannel(comm, engine, param.devType, rank, destRank, &(resCtxHost.channelHandle)));

        // ==============================================
        // STEP 2.3: 获取本端和远端的中转内存
        // ==============================================
        CHK_RET(HcclGetHcclBuffer(comm, &(resCtxHost.localBuffer.addr), &(resCtxHost.localBuffer.size)));
        CHK_RET(HcclChannelGetHcclBuffer(comm, resCtxHost.channelHandle, &(resCtxHost.remoteBuffer.addr),
                                         &(resCtxHost.remoteBuffer.size)));

        ACLCHECK(aclrtMemcpy(param.resCtx, size, &resCtxHost, size, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(LaunchKernel(param, stream));

    return HCCL_SUCCESS;
}
