/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_single_executor.h"

namespace ops_hccl {
ScatterSingleExecutor::ScatterSingleExecutor() : ScatterExecutorBase()
{
    desc_.level1SupportedAlgos = {
        AlgTypeLevel1::ALG_LEVEL1_NHR,
        AlgTypeLevel1::ALG_LEVEL1_NB,
        AlgTypeLevel1::ALG_LEVEL1_RING
    };
    desc_.level2SupportedAlgos = {
        AlgTypeLevel2::ALG_LEVEL2_NHR,
        AlgTypeLevel2::ALG_LEVEL2_NB,
        AlgTypeLevel2::ALG_LEVEL2_RING
    };
}

HcclResult ScatterSingleExecutor::Orchestrate(const OpParam &param, AlgResourceCtx* resCtx)
{
    HcclUs startut = TIME_NOW();

    topoInfo_ = &(resCtx->topoInfo);
    algResource_ = resCtx;
    tag_ = param.tag;
    algType_ = resCtx->algType;

    // 做参数的还原
    ThreadHandle* threadHandlePtr = reinterpret_cast<ThreadHandle *>(reinterpret_cast<u8 *>(algResource_) + sizeof(AlgResourceCtx));
    thread_ = threadHandlePtr[0];

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;

    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ScatterSingleExecutor][Orchestrate]errNo[0x%016llx]all reduce executor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    HCCL_INFO("tag[%s] Scatter executor orchestrate success, take time [%lld]us.",
        param.tag, DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult ScatterSingleExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[ScatterSingleExecutor][KernelRun] starts.");
    u64 totalSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, execMem.outputPtr, execMem.inputPtr, totalSize)));

    return HCCL_SUCCESS;
}
REGISTER_EXEC("ScatterSingleExecutor", ScatterSingleRank, ScatterSingleExecutor);
}