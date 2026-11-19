/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_REDUCE_SEQUENCE_RHD_TAIL_H
#define HCCLV2_REDUCE_SEQUENCE_RHD_TAIL_H

#include <algorithm>
#include "alg_param.h"
#include "alg_data_trans_wrapper.h"
#include "alg_template_base.h"

namespace ops_hccl {

/* DPU 卸载三级 RHD 的非整除数据尾 staging（executor 侧，模板零改动）：RhdDpuInter 的均匀
 * σ-unit 契约只覆盖 totalRankAlign 整除数据；非整除余量直接入链会在 L2 每 pod-slice 丢 8/4
 * 字节尾（409：72 源仅合 24）。每轮 loop 只跑整除对齐前缀；余量单独一轮 staged 进 CCL 末端
 * 隔离 tile，补位车道用有效输入复制初始化（逐元素归约，不影响有效前缀），仅回拷有效字节。 */
struct ReduceRhdTail {
    HcclMem storage{};
    u64 rankAlign{0};
    u64 dataTypeSize{0};
    u64 inputOffset{0};
    u64 outputOffset{0};
};

inline HcclResult
ReduceRhdPrepareTailWorkspace(u64 dataCount, u64 rankAlign, u64 dataTypeSize, HcclMem& workspace, ReduceRhdTail& tail)
{
    CHK_PRT_RET(
        rankAlign == 0 || dataTypeSize == 0,
        HCCL_ERROR("[ReduceRhdPrepareTailWorkspace] invalid rank alignment or data type size"), HCCL_E_PARA);
    if (dataCount % rankAlign == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        rankAlign > workspace.size / dataTypeSize / 2,
        HCCL_ERROR("[ReduceRhdPrepareTailWorkspace] CCL buffer too small for tail tiles"), HCCL_E_INTERNAL);
    const u64 tileBytes = rankAlign * dataTypeSize;
    const u64 slotBytes = (tileBytes / HCCL_MIN_SLICE_ALIGN + static_cast<u64>(tileBytes % HCCL_MIN_SLICE_ALIGN != 0))
                          * HCCL_MIN_SLICE_ALIGN;
    CHK_PRT_RET(
        workspace.addr == nullptr || slotBytes >= workspace.size / 2,
        HCCL_ERROR("[ReduceRhdPrepareTailWorkspace] no CCL workspace left after tail reservation"), HCCL_E_INTERNAL);
    tail.storage = workspace;
    tail.rankAlign = rankAlign;
    tail.dataTypeSize = dataTypeSize;
    tail.inputOffset = workspace.size - 2 * slotBytes;
    tail.outputOffset = workspace.size - slotBytes;
    workspace.size = tail.inputOffset;
    return HCCL_SUCCESS;
}

// 先跑整除对齐前缀；仅最后的 count < rankAlign 余量需要 staging
inline u64 ReduceRhdSelectLoopCount(u64 remaining, u64 maxCountPerLoop, u64 rankAlign)
{
    const u64 count = std::min(remaining, maxCountPerLoop);
    return count >= rankAlign ? count / rankAlign * rankAlign : count;
}

inline HcclResult ReduceRhdStageTail(
    const OpParam& param, const ReduceRhdTail& tail, u64 processedCount, u64 validCount, ThreadHandle thread)
{
    CHK_PRT_RET(
        validCount == 0 || validCount >= tail.rankAlign || processedCount > param.DataDes.count
            || validCount > param.DataDes.count - processedCount,
        HCCL_ERROR("[ReduceRhdStageTail] invalid input tail range"), HCCL_E_PARA);
    const u64 validBytes = validCount * tail.dataTypeSize;
    CHK_RET(LocalCopy(
        thread, DataSlice(param.inputPtr, processedCount * tail.dataTypeSize, validBytes, validCount),
        DataSlice(tail.storage.addr, tail.inputOffset, validBytes, validCount)));
    // 补位车道用已初始化的有效输入复制填充：归约逐元素，这些车道不影响有效前缀，
    // 无需按 reduce 算子/数据类型构造单位元
    u64 filledCount = validCount;
    while (filledCount < tail.rankAlign) {
        const u64 copyCount = std::min(filledCount, tail.rankAlign - filledCount);
        const u64 copyBytes = copyCount * tail.dataTypeSize;
        CHK_RET(LocalCopy(
            thread, DataSlice(tail.storage.addr, tail.inputOffset, copyBytes, copyCount),
            DataSlice(tail.storage.addr, tail.inputOffset + filledCount * tail.dataTypeSize, copyBytes, copyCount)));
        filledCount += copyCount;
    }
    HCCL_INFO(
        "[ReduceRhdStageTail] processedCount[%llu] validCount[%llu] paddedCount[%llu]", processedCount, validCount,
        tail.rankAlign);
    return HCCL_SUCCESS;
}

inline HcclResult ReduceRhdCopyTailOutput(
    const OpParam& param, const ReduceRhdTail& tail, u64 processedCount, u64 validCount, ThreadHandle thread)
{
    const u64 validBytes = validCount * tail.dataTypeSize;
    return LocalCopy(
        thread, DataSlice(tail.storage.addr, tail.outputOffset, validBytes, validCount),
        DataSlice(param.outputPtr, processedCount * tail.dataTypeSize, validBytes, validCount));
}

} // namespace ops_hccl

#endif // HCCLV2_REDUCE_SEQUENCE_RHD_TAIL_H
