/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <memory>
#include "hccl/hcomm_primitives.h"
#include "common.h"
#include "exec_op.h"

using namespace ops_hccl_p2p;

#ifdef __cplusplus
extern "C" {
#endif
HcclResult __attribute__((weak)) HcommRegOpInfo(const char* commId, void* opInfo, size_t size);
#ifdef __cplusplus
}
#endif

extern "C" unsigned int HcclLaunchP2PAicpuKernel(OpParam *param)
{
    HCCL_INFO("Entry-%s, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) { 
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }

    if (HcommRegOpInfo != nullptr &&
        HcommRegOpInfo(param->commName, reinterpret_cast<void *>(param), sizeof(OpParam)) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommRegOpInfo fail, commName[%s], algTag[%s], param[%p], size[%u]",
            __func__, param->commName, param->algTag, param, sizeof(OpParam));
        return 1;
    }

    // 获取Device侧主thread
    ThreadHandle thread = param->resCtx->threadHandle;
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed start batch mode");
        return 1;
    }

    // 主thread等待Host stream的通知
    if (HcommAclrtNotifyWaitOnThread(thread, param->resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to wait notify[%d] from host main stream", param->resCtx->notifyIds[0]);
        return 1;
    }

    // 执行算法编排
    if (ExecOp(*param, param->resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("orchestrate failed for op:%d", param->opType);
        return 1;
    }

    // 主thread通知Host stream
    if (HcommAclrtNotifyRecordOnThread(thread, param->resCtx->notifyIds[1]) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to record host main stream");
        return 1;
    }

    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("failed end batch mode");
        return 1;
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, commName[%s], tag[%s]", __func__, param->commName, param->tag);
    return 0;
}
