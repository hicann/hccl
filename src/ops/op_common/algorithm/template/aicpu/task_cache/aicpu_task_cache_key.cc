/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <string>

#include "aicpu_task_cache_key.h"
#include "aicpu_task_cache_utils.h"

namespace ops_hccl {

namespace {
    // 将无符号整数转为十进制字符串, 写入ptr
    // 返回写入后的指针; uint64_t最大20位数字, 调用方保证缓冲区有足够空间
    // 兼容uint32_t和unsigned long long (uint32_t隐式提升为uint64_t)
    inline char* UIntToChars(char* ptr, uint64_t val)
    {
        // uint64_t最大20位数字 (18446744073709551615)
        char tmp[20];
        char* tp = tmp;

        do {
            *tp++ = static_cast<char>('0' + val % 10U);
            val /= 10U;
        } while (val > 0);

        // 反转写入目标缓冲区
        while (tp > tmp) {
            *ptr++ = *--tp;
        }
        return ptr;
    }
} // namespace

HcclResult AicpuTaskCacheKey::GetAicpuTaskCacheTag(const OpParam& param, uint64_t inputSize, std::string& cacheTag)
{
    // 暂时不考虑v类算子 (应该被cache使能约束拦截, 不应该进入本函数), dataType一定不是reserved
    const HcclCMDType opType = param.opType;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) { // alltoall算子
        dataType = param.all2AllDataDes.sendType;
    } else if (AicpuTaskCacheUtils::IsNonVariableOpType(opType)) { // 非alltoall的非v类算子
        dataType = param.DataDes.dataType;
    } else {
        HCCL_ERROR("[AicpuTaskCacheKey][AicpuTaskCacheKey] invalid opType[%d] for aicpu task cache", opType);
        return HCCL_E_PARA;
    }

    // 获取rootRank
    // 注意: broadcast/reduce/scatter的task编排受rootRank影响
    uint32_t rootRank = 0;
    if (opType == HcclCMDType::HCCL_CMD_BROADCAST || opType == HcclCMDType::HCCL_CMD_SCATTER
        || opType == HcclCMDType::HCCL_CMD_REDUCE) {
        rootRank = param.root;
    }

    // 获取其他字段
    const HcclReduceOp reduceType = param.reduceType;
    const bool isZeroCopy = param.isZeroCopy;
    const OpMode opMode = param.opMode;

    // 使用'-'作为间隔符, 拼接cacheTag
    // 注意: 把input size放在前面, 如果需要解析, 可以减少解析开销
    // 注意: commId放在最后, 如果需要解析, 无需考虑commId中含有delimiter的情况
    // 注意: enum class不能转为uint8_t, 否则会作为char输出; 需显式转为uint32_t后再用to_chars, 否则编译失败
    const char* commId = param.commName;  // 最大长度COMM_INDENTIFIER_MAX_LENGTH (128)
    constexpr size_t RESERVED_SIZE = 256; // commId+7个整数, 最多128+80+7个字符, 预留256足够
    char buf[RESERVED_SIZE];
    char* ptr = buf;
    const char delimiter = '-';

    // inputSize (uint64最多20个字符)
    ptr = UIntToChars(ptr, static_cast<unsigned long long>(inputSize));
    *ptr++ = delimiter;

    // opType (uint32最多10个字符)
    ptr = UIntToChars(ptr, static_cast<uint32_t>(opType));
    *ptr++ = delimiter;

    // dataType (uint32最多10个字符)
    ptr = UIntToChars(ptr, static_cast<uint32_t>(dataType));
    *ptr++ = delimiter;

    // reduceType (uint32最多10个字符)
    ptr = UIntToChars(ptr, static_cast<uint32_t>(reduceType));
    *ptr++ = delimiter;

    // isZeroCopy (uint32最多10个字符)
    ptr = UIntToChars(ptr, static_cast<uint32_t>(isZeroCopy));
    *ptr++ = delimiter;

    // opMode (uint32最多10个字符)
    ptr = UIntToChars(ptr, static_cast<uint32_t>(opMode));
    *ptr++ = delimiter;

    // rootRank (uint32最多10个字符)
    ptr = UIntToChars(ptr, rootRank);
    *ptr++ = delimiter;

    // 拼接commId (最后一段, 不加delimiter后缀; 最多128个字符)
    size_t commLen = std::strlen(commId);
    // commId最大长度为COMM_INDENTIFIER_MAX_LENGTH, 不会失败
    (void)memcpy_s(ptr, COMM_INDENTIFIER_MAX_LENGTH, commId, commLen);
    ptr += commLen;

    // 重新更新cacheTag长度
    cacheTag.assign(buf, static_cast<size_t>(ptr - buf));

    HCCL_INFO(
        "[AicpuTaskCacheKey][GetAicpuTaskCacheTag] cacheTag[%s] from commId[%s] opType[%d] dataType[%d] "
        "reduceType[%d] isZeroCopy[%d] inputSize[%llu] opMode[%d] rootRank[%d]",
        cacheTag.c_str(), commId, opType, dataType, reduceType, isZeroCopy, inputSize, opMode, rootRank);

    return HCCL_SUCCESS;
}

} // namespace ops_hccl
