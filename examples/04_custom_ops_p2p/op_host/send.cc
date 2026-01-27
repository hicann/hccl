/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl/hccl_types.h"
#include "hccl/hcomm_primitives.h"
#include "log.h"
#include "common.h"
#include "hccl_custom_p2p.h"
#include "load_kernel.h"
#include "launch_kernel.h"

using namespace std;
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

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine engine = CommEngine::COMM_ENGINE_AICPU;
    CHK_RET(LoadAICPUKernel());

    void * ctx = nullptr;
    uint64_t size = sizeof(AlgResourceCtx);
    if (HcclEngineCtxGet(comm, param.tag, engine, &ctx, &size) == HCCL_SUCCESS) {
        // device资源已经存在
        param.resCtx = static_cast<AlgResourceCtx *>(ctx);
    } else {
        // 不存在，新创建Context
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, engine, size, &ctx));
        param.resCtx = static_cast<AlgResourceCtx *>(ctx);

        // ==============================================
        // STEP 2.1: 申请thread
        // ==============================================
        ACLCHECK(aclrtCreateNotify(&(g_notifies[0]), ACL_NOTIFY_DEFAULT));
        ACLCHECK(aclrtCreateNotify(&(g_notifies[1]), ACL_NOTIFY_DEFAULT));
        AlgResourceCtx resCtxHost;
        for (uint32_t idx = 0; idx < AICPU_CONTROL_NOTIFY_NUM; idx++) {
            ACLCHECK(aclrtGetNotifyId(g_notifies[idx], &(resCtxHost.notifyIds[idx])));
        }
        CHK_RET(HcclThreadAcquire(comm, engine, 1, 0, &(resCtxHost.threadHandle)));

        // ==============================================
        // STEP 2.2: 建立通信链路Channel，两个 rank 之间建立 1 个 channel
        // ==============================================
        HcclChannelDesc channelDesc;
        HcclChannelDescInit(&channelDesc, 1);
        channelDesc.remoteRank = destRank;
        channelDesc.channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
        channelDesc.notifyNum = 2;
        CHK_RET(HcclChannelAcquire(comm, engine, &channelDesc, 1, &(resCtxHost.channelHandle)));

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
