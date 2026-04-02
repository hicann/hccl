/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfx/task_exception_fun.h"

#include <string>
#include <sstream>
#include <memory>
#include "log.h"
#include "alg_param.h"

namespace ops_hccl {
HcclResult CreateScatter(OpParam *param, ScatterOpInfo *opInfo)
{
    CHK_PTR_NULL(param);
    CHK_PTR_NULL(opInfo);
    s32 sRet = strncpy_s(opInfo->algTag, ALG_TAG_LENGTH, param->algTag, ALG_TAG_LENGTH);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, return %d.", __func__, sRet), HCCL_E_MEMORY);
    sRet = strncpy_s(opInfo->commName, COMM_INDENTIFIER_MAX_LENGTH, param->commName, COMM_INDENTIFIER_MAX_LENGTH);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, return %d.", __func__, sRet), HCCL_E_MEMORY);
    opInfo->count = param->DataDes.count;
    opInfo->dataType = param->DataDes.dataType;
    opInfo->opType = param->opType;
    opInfo->root = param->root;
    opInfo->inputPtr = param->inputPtr;
    opInfo->outputPtr = param->outputPtr;
    return HCCL_SUCCESS;
}

void GetScatterOpInfo(const void *opInfo, char *outPut, size_t size)
{
    const ScatterOpInfo *info = reinterpret_cast<const ScatterOpInfo *>(opInfo);
    std::stringstream ss;
    ss << "tag:" << info->algTag << ", ";
    ss << "group:" << info->commName << ", ";
    ss << "count:" << info->count << ", ";
    ss << "dataType:" << info->dataType << ", ";
    ss << "opType:" << info->opType << ", ";
    ss << "rootId:" << info->root << ", ";
    ss << "dstAddr:0x" << std::hex << info->inputPtr << ", ";
    ss << "srcAddr:0x" << std::hex << info->outputPtr << ".";

    std::string strTmp = ss.str();
    s32 sRet = strncpy_s(outPut, size, strTmp.c_str(), std::min(size, strTmp.size()));
    if (strTmp.size() >= size || sRet != EOK) {
        HCCL_ERROR("%s strncpy_s fail, src size[%u], dst size[%u], sRet[%d]", strTmp.size(), size, sRet);
    }
}

uint32_t GetHcclDfxOpInfoDataType(const OpParam &param) {
    uint32_t dataType = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V
        || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        dataType = static_cast<u32>(param.vDataDes.dataType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        dataType = static_cast<u32>(param.all2AllDataDes.sendType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        dataType = static_cast<u32>(param.all2AllVDataDes.sendType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dataType = static_cast<u32>(param.all2AllVCDataDes.sendType);
    } else {
        dataType = static_cast<u32>(param.DataDes.dataType);
    }
    return dataType;
}

HcclResult ConvertToHcclDfxOpInfo(OpParam *param, HcclDfxOpInfo *hcclDfxOpInfo)
{
    CHK_PTR_NULL(param);
    CHK_PTR_NULL(hcclDfxOpInfo);
    hcclDfxOpInfo->opMode = static_cast<u32>(param->opMode);
    hcclDfxOpInfo->opType = static_cast<u32>(param->opType);
    hcclDfxOpInfo->reduceOp = static_cast<u32>(param->reduceType);
    hcclDfxOpInfo->dataType = GetHcclDfxOpInfoDataType(*param);
    hcclDfxOpInfo->dataCount = param->dataCount;
    hcclDfxOpInfo->root = param->root;
    hcclDfxOpInfo->engine = param->engine;
    hcclDfxOpInfo->cpuTsThread = param->opThread;

    s32 sRet = strncpy_s(hcclDfxOpInfo->algTag, ALG_TAG_LENGTH, param->algTag, ALG_TAG_LENGTH);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, param.algTag %s,  return %d.", __func__, param->algTag, sRet), HCCL_E_MEMORY);

    hcclDfxOpInfo->cpuWaitAicpuNotifyIdx = param->aicpuRecordCpuIdx;
    return HCCL_SUCCESS;
}
}