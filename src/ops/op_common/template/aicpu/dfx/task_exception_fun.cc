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
#include "dlsym_common.h"
#include "comm_engine_utils.h"

namespace ops_hccl {
HcclResult CreateScatter(OpParam* param, ScatterOpInfo* opInfo)
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

void GetScatterOpInfo(const void* opInfo, char* outPut, size_t size)
{
    const ScatterOpInfo* info = reinterpret_cast<const ScatterOpInfo*>(opInfo);
    std::stringstream ss;
    ss << "tag:" << info->algTag << ", ";
    ss << "group:" << info->commName << ", ";
    ss << "count:" << info->count << ", ";
    ss << "dataType:" << info->dataType << ", ";
    ss << "opType:" << info->opType << ", ";
    ss << "rootId:" << info->root << ", ";
    ss << "srcAddr:0x" << std::hex << info->inputPtr << ", ";
    ss << "dstAddr:0x" << std::hex << info->outputPtr << ".";

    std::string strTmp = ss.str();
    s32 sRet = strncpy_s(outPut, size, strTmp.c_str(), std::min(size, strTmp.size()));
    if (strTmp.size() >= size || sRet != EOK) {
        HCCL_ERROR(
            "[%s] strncpy_s failed, src size[%zu], dst size[%zu], sRet[%d]", __func__, strTmp.size(), size, sRet);
    }
}

HcclResult GetHcclDfxOpInfoDataType(const OpParam& param, uint32_t& dataType)
{
    dataType = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        dataType = static_cast<u32>(param.vDataDes.dataType);
    } else if (
        param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dataType = static_cast<u32>(param.all2AllVDataDes.sendType);
    } else if (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        CHK_PRT_RET(
            param.batchSendRecvDataDes.itemNum == 0, HCCL_INFO("[%s]tag[%s] itemNum is 0, skip", __func__, param.tag),
            HCCL_SUCCESS);
        CHK_PRT_RET(
            param.batchSendRecvDataDes.sendRecvItemsPtr == nullptr,
            HCCL_ERROR("[%s]fail, tag[%s] sendRecvItemsPtr is nullptr", __func__, param.tag), HCCL_E_PTR);
        dataType
            = static_cast<u32>(param.batchSendRecvDataDes.sendRecvItemsPtr->dataType); // dfx功能只能上报一个数据类型
    } else {
        dataType = static_cast<u32>(param.DataDes.dataType);
    }
    HCCL_INFO("[%s]tag[%s], dataType[%u], opType[%u]", __func__, param.tag, dataType, param.opType);
    return HCCL_SUCCESS;
}

HcclResult ConvertToHcclDfxOpInfo(OpParam* param, HcclDfxOpInfoCompat* hcclDfxOpInfo)
{
    CHK_PTR_NULL(param);
    CHK_PTR_NULL(hcclDfxOpInfo);
    hcclDfxOpInfo->opMode = static_cast<u32>(param->opMode);
    hcclDfxOpInfo->opType = static_cast<u32>(param->opType);
    hcclDfxOpInfo->reduceOp = static_cast<u32>(param->reduceType);
    CHK_RET(GetHcclDfxOpInfoDataType(*param, hcclDfxOpInfo->dataType));
    hcclDfxOpInfo->dataCount = param->dataCount;
    hcclDfxOpInfo->root = param->root;
    hcclDfxOpInfo->engine = param->engine;
    hcclDfxOpInfo->cpuTsThread = param->opThread;
    s32 sRet = strncpy_s(hcclDfxOpInfo->algTag, ALG_TAG_LENGTH, param->algTag, ALG_TAG_LENGTH);
    CHK_PRT_RET(
        sRet != EOK, HCCL_ERROR("%s call strncpy_s failed, param.algTag %s, return %d.", __func__, param->algTag, sRet),
        HCCL_E_MEMORY);
    hcclDfxOpInfo->cpuWaitAicpuNotifyIdx = param->aicpuRecordCpuIdx;
    hcclDfxOpInfo->inputMemAddr = reinterpret_cast<uint64_t>(param->inputPtr);
    hcclDfxOpInfo->inputMemSize = param->inputSize;
    hcclDfxOpInfo->outputMemAddr = reinterpret_cast<uint64_t>(param->outputPtr);
    hcclDfxOpInfo->outputMemSize = param->outputSize;
    HCCL_INFO(
        "[%s]HcclDfxOpInfo param: algTag[%s], opMode[%u], opType[%u], reduceOp[%u], dataType[%u], dataCount[%llu],"
        "root[%u], engine[%s], cpuTsThread[%u], cpuWaitAicpuNotifyIdx[%u], "
        "inputMemAddr[0x%llx], inputMemSize[%llu], outputMemAddr[0x%llx], outputMemSize[%llu]",
        __func__, hcclDfxOpInfo->algTag, hcclDfxOpInfo->opMode, hcclDfxOpInfo->opType, hcclDfxOpInfo->reduceOp,
        hcclDfxOpInfo->dataType, hcclDfxOpInfo->dataCount, hcclDfxOpInfo->root,
        GetEnumToString(GetCommEngineStatusStrMap(), hcclDfxOpInfo->engine).c_str(), hcclDfxOpInfo->cpuTsThread,
        hcclDfxOpInfo->cpuWaitAicpuNotifyIdx, hcclDfxOpInfo->inputMemAddr, hcclDfxOpInfo->inputMemSize,
        hcclDfxOpInfo->outputMemAddr, hcclDfxOpInfo->outputMemSize);
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
