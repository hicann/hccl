/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_task_cache_utils.h"
namespace ops_hccl {

bool AicpuTaskCacheUtils::IsNonVariableOpType(HcclCMDType opType)
{
    // 注意: 当前hccl不支持HcclGather
    if (opType == HcclCMDType::HCCL_CMD_BROADCAST || opType == HcclCMDType::HCCL_CMD_ALLREDUCE
        || opType == HcclCMDType::HCCL_CMD_REDUCE || opType == HcclCMDType::HCCL_CMD_ALLGATHER
        || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opType == HcclCMDType::HCCL_CMD_ALLTOALL
        || opType == HcclCMDType::HCCL_CMD_SCATTER) {
        return true;
    }
    return false;
}

HcclResult AicpuTaskCacheUtils::GetInputOutputInfoForCache(
    const OpParam& param, const uint32_t rankSize, uint64_t& inputSize, uint64_t& outputSize)
{
    const HcclCMDType opType = param.opType;

    // 准备data type和count
    // NOTE: 非V类算子 (DataRes), V类算子 (VDataDes), All2All类算子 (All2AllDataDes), batch类算子
    // (BatchSendRecvDataDes/BatchWriteDataDes)
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) { // alltoall算子
        // 注意: sendType和recvType一定相同
        HcclDataType sendType = param.all2AllDataDes.sendType;

        // 注意: 对于alltoall算子, inputSize和outputSize一定相同 (但不能直接使用param.input/outputSize,
        // alltoall算子不会设置这两个字段)
        CHK_PRT_RET(
            param.all2AllVDataDes.sendCounts == nullptr,
            HCCL_ERROR("[%s] all2AllVDataDes.sendCounts is nullptr, opType[%u]", __func__, opType), HCCL_E_PARA);
        const uint64_t sendCount = *(reinterpret_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts));
        inputSize = sendCount * rankSize * DATATYPE_SIZE_TABLE[sendType];

        // 注意: 不能使用param.All2AllDataDes.recvCount * rankSize * SIZE_TABLE[recvType],
        // 因为alltoall使用sendCount来表示send/recvCount, 而recvCount本身为0
        outputSize = inputSize;

        HCCL_DEBUG(
            "[AicpuTaskCacheUtils][%s] opType[%u] rankSize[%u] sendType[%u] "
            "inputSize[%llu] outputSize[%llu] sendCount[%llu] dataTypeSize[%u]",
            __func__, opType, rankSize, sendType, inputSize, outputSize, sendCount, DATATYPE_SIZE_TABLE[sendType]);
    } else {
        inputSize = param.inputSize;
        outputSize = param.outputSize;

        HCCL_DEBUG(
            "[AicpuTaskCacheUtils][%s] opType[%u] rankSize[%u] inputSize[%llu] outputSize[%llu]", __func__, opType,
            rankSize, inputSize, outputSize);
    }

    return HCCL_SUCCESS;
}

} // namespace ops_hccl
