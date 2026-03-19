/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_mc2.h"
#include "log.h"
#include "sal.h"
#include "alg_env_config.h"
#include "hccl_inner_dl.h"
#include "param_check.h"

using namespace ops_hccl;

constexpr uint32_t ALG_CONFIG_SIZE = 128;
struct HcclOpArgs {
    HcclDataType srcDataType;
    HcclDataType dstDataType;
    HcclReduceOp reduceType;
    uint64_t count;
    char algConfig[ALG_CONFIG_SIZE];
    CommEngine commEngine;
    uint64_t reverse;

    void Init() {
        srcDataType = HCCL_DATA_TYPE_FP16;
        dstDataType = HCCL_DATA_TYPE_FP16;
        reduceType = HCCL_REDUCE_SUM;
        count = 0;
    }
};

HcclResult HcclKfcAllocOpArgs(void **opArgs)
{
    CHK_PTR_NULL(opArgs);

    HcclOpArgs *opArgsMem = (HcclOpArgs *)malloc(sizeof(HcclOpArgs));
    if (opArgsMem == nullptr) {
        HCCL_ERROR("[HcclKfcAllocOpArgs] malloc HcclOpArgs mem failed, please check.");
        return HCCL_E_INTERNAL;
    }
    opArgsMem->Init();
    *opArgs = opArgsMem;
    HCCL_RUN_INFO("[HcclKfcAllocOpArgs] malloc HcclOpArgs success, please fill mem[%p->%p] in it.", opArgs, *opArgs);

    return HCCL_SUCCESS;
}

HcclResult HcclKfcFreeOpArgs(void *opArgs)
{
    CHK_PTR_NULL(opArgs);

    free(opArgs);
    opArgs = nullptr;

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetSrcDataType(void *opArgs, uint8_t srcDataType)
{
    CHK_PTR_NULL(opArgs);
    CHK_RET(HcomCheckDataType(static_cast<HcclDataType>(srcDataType)));

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->srcDataType = static_cast<HcclDataType>(srcDataType);

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetDstDataType(void *opArgs, uint8_t dstDataType)
{
    CHK_PTR_NULL(opArgs);
    CHK_RET(HcomCheckDataType(static_cast<HcclDataType>(dstDataType)));

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->dstDataType = static_cast<HcclDataType>(dstDataType);

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetReduceType(void *opArgs, uint32_t reduceType)
{
    CHK_PTR_NULL(opArgs);
    CHK_RET(HcomCheckReductionOp(static_cast<HcclReduceOp>(reduceType)));

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->reduceType = static_cast<HcclReduceOp>(reduceType);

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetCount(void *opArgs, uint64_t count)
{
    CHK_PTR_NULL(opArgs);
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR("[%s] count[%llu] is invalid (bigger than MAX count[%lu])", __func__, count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->count = count;

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetAlgConfig(void *opArgs, char *algConfig)
{
    CHK_PTR_NULL(opArgs);
    CHK_PTR_NULL(algConfig);

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    s32 ret = strcpy_s(opArgsPtr->algConfig, ALG_CONFIG_SIZE, algConfig);
    if (ret != EOK) {
        HCCL_ERROR("[%s] strcpy_s algConfig failed, ret[%d]", __func__, ret);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclKfcOpArgsSetCommEngine(void *opArgs, uint8_t commEngine)
{
    CHK_PTR_NULL(opArgs);
    // A3只支持AICPU和AIV场景
    if (commEngine != COMM_ENGINE_AICPU && commEngine != COMM_ENGINE_AIV) {
        HCCL_ERROR("[%s] commEngine[%u] not supported", __func__, commEngine);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    opArgsPtr->commEngine = static_cast<CommEngine>(commEngine);

    return HCCL_SUCCESS;
}

HcclResult HcclCreateOpResCtx(HcclComm comm, uint8_t opType, void *opArgs, void **opResCtx)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(opArgs);
    CHK_PTR_NULL(opResCtx);
    if (opType >= static_cast<uint8_t>(HcclCMDType::HCCL_CMD_MAX)) {
        HCCL_ERROR("[%s] invalid opType[%u]", __func__, opType);
        return HCCL_E_PARA;
    }

    CHK_RET(InitEnvConfig());

    HcclOpArgs *opArgsPtr = static_cast<HcclOpArgs *>(opArgs);
    if (GetExternalInputHcclEnableEntryLog()) {
        HCCL_RUN_INFO("Entry-HcclKfcCreateOpResCtx, opType[%u], opArgs[%p], srcDataType[%u], dstDataType[%u], reduceType[%u], "
            "count[%llu], algConfig[%s], commEngine[%u], opResCtx[%p]",
            opType, opArgs, opArgsPtr->srcDataType, opArgsPtr->dstDataType, opArgsPtr->reduceType,
            opArgsPtr->count, opArgsPtr->algConfig, opArgsPtr->commEngine, opResCtx);
    }

    CHK_RET(HcclCreateOpResCtxInner(comm, opType, opArgsPtr->srcDataType, opArgsPtr->dstDataType,
        opArgsPtr->reduceType, opArgsPtr->count, opArgsPtr->algConfig, opArgsPtr->commEngine, opResCtx));

    return HCCL_SUCCESS;
}