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

bool AicpuTaskCacheUtils::IsAlltoallvType(HcclCMDType opType)
{
    return opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC;
}

HcclResult AicpuTaskCacheUtils::AppendVOpInfoForCache(
    const OpParam& param, uint32_t rankSize, std::vector<void*>& addrs, std::vector<uint64_t>& sizes)
{
    CHK_PRT_RET(
        rankSize == 0 || addrs.size() != 2 || sizes.size() != 2,
        HCCL_ERROR("[%s] invalid rankSize or memory arrays", __func__), HCCL_E_PARA);
    const uint64_t arrayCount = 2 + uint64_t{2} * rankSize;
    CHK_PRT_RET(
        arrayCount > UINT32_MAX, HCCL_ERROR("[%s] V array count exceeds protocol limit", __func__), HCCL_E_PARA);
    addrs.reserve(arrayCount);
    sizes.reserve(arrayCount);
    const uint64_t* counts[]
        = {static_cast<const uint64_t*>(param.all2AllVDataDes.sendCounts),
           static_cast<const uint64_t*>(param.all2AllVDataDes.recvCounts)};
    const uint64_t* displs[]
        = {static_cast<const uint64_t*>(param.all2AllVDataDes.sdispls),
           static_cast<const uint64_t*>(param.all2AllVDataDes.rdispls)};
    const HcclDataType types[] = {param.all2AllVDataDes.sendType, param.all2AllVDataDes.recvType};
    // HCOMM不知道算子data type。传输协议中的counts和displs必须分别为字节数和字节偏移，不能传元素数。
    // 布局：[input, output, send rank 0..N-1, recv rank 0..N-1]；前两项仍为完整user buffer。
    for (uint32_t side = 0; side < 2; ++side) {
        CHK_PTR_NULL(counts[side]);
        CHK_PTR_NULL(displs[side]);
        const uint32_t type = static_cast<uint32_t>(types[side]);
        CHK_PRT_RET(
            type >= static_cast<uint32_t>(HCCL_DATA_TYPE_RESERVED), HCCL_ERROR("[%s] invalid data type", __func__),
            HCCL_E_PARA);
        const uint64_t typeSize = DATATYPE_SIZE_TABLE[type];
        CHK_PRT_RET(typeSize == 0, HCCL_ERROR("[%s] zero type size", __func__), HCCL_E_PARA);
        const uint64_t base = reinterpret_cast<uint64_t>(addrs[side]);
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            CHK_PRT_RET(
                counts[side][rank] > UINT64_MAX / typeSize || displs[side][rank] > UINT64_MAX / typeSize,
                HCCL_ERROR("[%s] byte conversion overflows", __func__), HCCL_E_PARA);
            const uint64_t bytes = counts[side][rank] * typeSize;
            const uint64_t offset = displs[side][rank] * typeSize;
            CHK_PRT_RET(
                offset > UINT64_MAX - base || bytes > UINT64_MAX - (base + offset)
                    || (bytes != 0 && (offset > sizes[side] || bytes > sizes[side] - offset)),
                HCCL_ERROR("[%s] V memory range is invalid", __func__), HCCL_E_PARA);
            // 零count的displ仍保留，但不要求对应实际可访问的内存。
            addrs.push_back(reinterpret_cast<void*>(base + offset));
            sizes.push_back(bytes);
        }
    }
    return HCCL_SUCCESS;
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
        const uint32_t type = static_cast<uint32_t>(sendType);
        CHK_PRT_RET(
            type >= static_cast<uint32_t>(HCCL_DATA_TYPE_RESERVED), HCCL_ERROR("[%s] invalid data type", __func__),
            HCCL_E_PARA);
        const uint64_t typeSize = DATATYPE_SIZE_TABLE[type];
        CHK_PRT_RET(typeSize == 0, HCCL_ERROR("[%s] zero type size", __func__), HCCL_E_PARA);
        CHK_PRT_RET(sendCount > UINT64_MAX / typeSize, HCCL_ERROR("[%s] send bytes overflow", __func__), HCCL_E_PARA);
        const uint64_t sendBytes = sendCount * typeSize;
        CHK_PRT_RET(
            rankSize == 0 || sendBytes > UINT64_MAX / rankSize, HCCL_ERROR("[%s] input size overflow", __func__),
            HCCL_E_PARA);
        inputSize = sendBytes * rankSize;

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
