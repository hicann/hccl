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
#include "alg_param.h"
#include "executor_base.h"
#include "coll_alg_exec_registry.h"
#include "hcomm_primitives.h"

using namespace ops_hccl;

extern "C" unsigned int HcclLaunchAicpuKernel(OpParam *param)
{
    // 根据算法名字获取executor
    std::string algName = std::string(param->algName);
    std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
    if (executor.get() == nullptr) {
        HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
        return 1;
    }

    // 获取Device测主thread
    ThreadHandle* threadHandlePtr = reinterpret_cast<ThreadHandle *>(reinterpret_cast<u8 *>(param->resCtx) +
        sizeof(AlgResourceCtx));
    ThreadHandle thread = threadHandlePtr[0];
    if (HcommSetLaunchMode(param->algTag, LAUNCH_MODE_BATCH) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
        return 1;
    }

    // 主thread等待Host stream的通知
    if (CommLocalBareNotifyWait(thread, param->resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to wait notify[%d] from host main stream", param->resCtx->notifyIds[0]);
        return 1;
    }

    // 执行算法编排
    if (executor->Orchestrate(*param, param->resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
        return 1;
    }

    // 主thread通知Host stream
    if (CommLocalBareNotifyRecord(thread, param->resCtx->notifyIds[1]) != HCCL_SUCCESS) {
        HCCL_ERROR("failed to record host main stream");
        return 1;
    }

    if (HcommSetLaunchMode(param->algTag, LAUNCH_MODE_EAGER) != HCCL_SUCCESS) {
        HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
        return 1;
    }

    return 0;
}
