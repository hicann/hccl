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

    // 将有符号整数转为十进制字符串(支持负数), 写入ptr; 其余语义同 UIntToChars
    inline char* Int64ToChars(char* ptr, int64_t val)
    {
        uint64_t absVal = 0;
        if (val < 0) {
            *ptr++ = '-';
            absVal = (val == INT64_MIN) ? (static_cast<uint64_t>(INT64_MAX) + 1U) : static_cast<uint64_t>(-val);
        } else {
            absVal = static_cast<uint64_t>(val);
        }
        return UIntToChars(ptr, absVal);
    }
} // namespace

HcclResult AicpuTaskCacheKey::GetAicpuTaskCacheTag(
    const OpParam& param, uint64_t inputSize, int64_t diff, uint64_t overlapBytes, std::string& cacheTag)
{
    const HcclCMDType opType = param.opType;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) { // alltoall算子
        dataType = param.all2AllDataDes.sendType;
    } else if (AicpuTaskCacheUtils::IsAlltoallvType(opType)) {
        // V类回放按真实类型换算后的字节长度/偏移刷新任务，key使用固定类型标记。
        dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
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
    const bool supportSymmetricMemory = param.supportSymmetricMemory;
    const bool isCapture = param.isCapture;

    // 使用'-'作为间隔符, 拼接cacheTag
    // tag作为不透明key整体比较; diff负号和commId都可能包含'-', 不得按间隔符拆分解析
    // 注意: enum class不能转为uint8_t, 否则会作为char输出; 需显式转为uint32_t后再用to_chars, 否则编译失败
    const char* commId = param.commName;  // 最大长度COMM_INDENTIFIER_MAX_LENGTH (128)
    constexpr size_t RESERVED_SIZE = 384; // commId+11个整数, 最多128+11*(20+1)+1(负号)个字符, 预留384足够
    char buf[RESERVED_SIZE];
    char* ptr = buf;
    const char delimiter = '-';

    // V类算子已在cache准入时限制为小数据量，tag无需区分实际inputSize
    const uint64_t cacheInputSize = AicpuTaskCacheUtils::IsAlltoallvType(opType) ? 0 : inputSize;
    // inputSize (uint64最多20个字符)
    ptr = UIntToChars(ptr, cacheInputSize);
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

    // supportSymmetricMemory (uint32最多10个字符)
    ptr = UIntToChars(ptr, static_cast<uint32_t>(supportSymmetricMemory));
    *ptr++ = delimiter;

    // 捕获态与非捕获态的任务下发模式不同, 不得共享缓存; isCapture独立于opMode
    ptr = UIntToChars(ptr, static_cast<uint32_t>(isCapture));
    *ptr++ = delimiter;

    // inplace 重叠特征(两段): 重叠位置 + 重叠宽度
    //   非 inplace: "0-0"; 完全相同: "0-<inputSize>"; 部分重叠/嵌套: "±diff-<overlap>"
    // 注意: diff 保证重放正确性 -- tag 相同 ⟹ 两次调用的位置差一致 ⟹ 现有"先命中先得"的地址
    // 匹配在重叠区刷出的结果一致(含嵌套场景, 不同嵌位 diff 不同, 天然不命中而走 miss 重录);
    // overlap 兼任形态判别(宽度 0 与完全相同的 inputSize 天然区分, 无需额外标志位)
    ptr = Int64ToChars(ptr, diff);
    *ptr++ = delimiter;

    // overlapBytes (uint64最多20个字符)
    ptr = UIntToChars(ptr, static_cast<unsigned long long>(overlapBytes));
    *ptr++ = delimiter;

    // rootRank (uint32最多10个字符)
    ptr = UIntToChars(ptr, rootRank);
    *ptr++ = delimiter;

    // 拼接commId (最后一段, 不加delimiter后缀; 最多128个字符)
    size_t commLen = std::strlen(commId);
    CHK_SAFETY_FUNC_RET(memcpy_s(ptr, COMM_INDENTIFIER_MAX_LENGTH, commId, commLen));
    ptr += commLen;

    // 重新更新cacheTag长度
    cacheTag.assign(buf, static_cast<size_t>(ptr - buf));

    HCCL_INFO(
        "[AicpuTaskCacheKey][GetAicpuTaskCacheTag] cacheTag[%s] from commId[%s] opType[%d] dataType[%d] "
        "reduceType[%d] isZeroCopy[%d] inputSize[%llu] opMode[%d] supportSymmetricMemory[%d] isCapture[%d] diff[%lld] "
        "overlapBytes[%llu] rootRank[%d]",
        cacheTag.c_str(), commId, opType, dataType, reduceType, isZeroCopy, cacheInputSize, opMode,
        supportSymmetricMemory, isCapture, diff, overlapBytes, rootRank);

    return HCCL_SUCCESS;
}

} // namespace ops_hccl
