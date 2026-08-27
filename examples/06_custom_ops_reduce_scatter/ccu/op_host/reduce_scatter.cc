/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <memory>
#include <hccl/hccl_res_expt.h>

#include "log.h"
#include "common.h"
#include "hccl_custom_reduce_scatter.h"
#include "utils.h"
#include "exec_op.h"

using namespace std;
using namespace ops_hccl_rs;

static HcclResult
InitAlgResourceCtx(HcclComm comm, OpParam& param, std::unique_ptr<AlgResourceCtxSerializable>& resCtxHost)
{
    void* ctx = nullptr;
    uint64_t size = 0;

    if (HcclEngineCtxGet(comm, param.tag, param.engine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("[HcclReduceScatterCustom] Engine context already exists, reuse it");
        param.ctxSize = size;
        char* resCtxSequence = static_cast<char*>(ctx);
        std::vector<char> ctxData(resCtxSequence, resCtxSequence + param.ctxSize);
        resCtxHost->DeSerialize(ctxData);
    } else {
        HCCL_INFO("[HcclReduceScatterCustom] Creating engine context");
        HcclResult ret = AllocAlgResource(comm, param, *resCtxHost);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("failed to alloc alg resource.");
            return ret;
        }
        std::vector<char> seq = resCtxHost->Serialize();
        uint64_t ctxSize = seq.size();

        void* newCtx = nullptr;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, param.engine, ctxSize, &newCtx));
        if (memcpy_s(newCtx, ctxSize, seq.data(), ctxSize) != EOK) {
            HCCL_ERROR("[HcclReduceScatterCustom] memcpy_s failed");
            return HCCL_E_INTERNAL;
        }
        param.ctxSize = ctxSize;
        HCCL_INFO("Execute GetAlgResCCU success.");
    }

    return HCCL_SUCCESS;
}

HcclResult HcclReduceScatterCustom(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    HCCL_INFO("Start to execute HcclReduceScatterCustom");

    // 1.校验参数是否为空
    CHK_PTR_NULL(stream);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);

    // 2.获取算子参数信息
    OpParam param;

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    if (dataType >= HCCL_DATA_TYPE_RESERVED || SIZE_TABLE[dataType] == 0) {
        HCCL_ERROR("[HcclReduceScatterCustom] invalid dataType [%u]", dataType);
        return HCCL_E_PARA;
    }
    uint32_t perDataSize = SIZE_TABLE[dataType];
    uint64_t outputSize = recvCount * perDataSize;
    uint64_t inputSize = outputSize * param.rankSize;
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s", "hccl_custom_reduce_scatter");
    if (ret <= 0) {
        HCCL_ERROR("[HcclReduceScatterCustom] Failed to fill param.tag");
        return HCCL_E_INTERNAL;
    }
    CHK_RET(GetDeviceType(&param.devType));
    if (param.devType != DEVICE_TYPE_A5) {
        HCCL_ERROR("[HcclReduceScatterCustom] Not Support Device Type [%u]", param.devType);
        return HCCL_E_INTERNAL;
    }

    param.stream = stream;
    CHK_RET(HcclGetCommName(comm, param.commName));
    HCCL_INFO("[HcclReduceScatterCustom] commName: %s", param.commName);

    param.opMode = OpMode::OPBASE;
    param.engine = CommEngine::COMM_ENGINE_CCU;

    param.inputPtr = sendBuf;
    param.inputSize = inputSize;
    param.outputPtr = recvBuf;
    param.outputSize = outputSize;
    param.count = recvCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    param.algType = AlgType::ALG_TYPE_MESH_1D;
    param.reduceOp = op;

    // 3. 创建资源
    std::unique_ptr<AlgResourceCtxSerializable> resCtxHost = std::make_unique<AlgResourceCtxSerializable>();
    CHK_RET(InitAlgResourceCtx(comm, param, resCtxHost));

    // 4.下发 CCU 任务
    CHK_RET(ExecOp(param, *resCtxHost));

    HCCL_INFO("HcclReduceScatterCustom executed successfully");
    return HCCL_SUCCESS;
}
