/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_RHD_TAIL_H
#define HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_RHD_TAIL_H

#include <algorithm>
#include <type_traits>
#include "alg_parse.h"
#include "alg_param.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"

namespace ops_hccl {

// 二级/三级 AllReduce RHD 共用的模板识别、相位桥接与尾块处理。
// 检测 RHD 模板；未声明 props 的模板按非 RHD 处理。
template <typename T, typename = void>
struct IsRhdTemplate : std::false_type {};

template <typename T>
struct IsRhdTemplate<T, std::enable_if_t<(T::props.algoType == AlgoType::RHD)>> : std::true_type {};

// RS 跳过 POST、AG 跳过 PRE 时，代表需将被合并方份额从 scratch 复制到 AG 的 owner 槽。
// 仅用于均匀分片：源槽为 unitStart+1，目标槽为 pairedAlgRank；被合并方由 AG POST 回传结果。
inline HcclResult ArSeqRhdBridgeMergedShare(
    u32 levelRankSize, u32 myLevelIdx, u64 subSliceSize, u64 subSliceCount, void* scratchAddr, u64 scratchBaseOff,
    void* outGridAddr, u64 outGridBaseOff, ThreadHandle thread)
{
    RhdPart1Info part1{};
    CHK_RET(CalcRhdPart1Info(levelRankSize, myLevelIdx, INVALID_VALUE_RANKID, part1));
    if (!part1.inPart1 || !part1.isRepresentative) {
        return HCCL_SUCCESS; // 仅配对代表需要桥接
    }
    const u64 srcOff = scratchBaseOff + (part1.unitStart + 1) * subSliceSize; // 被合并方终态槽 = 2t+1
    const u64 dstOff = outGridBaseOff + part1.pairedAlgRank * subSliceSize;   // AG 网格 owner 槽 = t+r
    DataSlice src(scratchAddr, srcOff, subSliceSize, subSliceCount);
    DataSlice dst(outGridAddr, dstOff, subSliceSize, subSliceCount);
    CHK_RET(LocalCopy(thread, src, dst));
    HCCL_INFO(
        "[ArSeqRhdBridgeMergedShare] myLevelIdx[%u] paired[%u] unitStart[%u] srcOff[%llu] dstOff[%llu] "
        "size[%llu]",
        myLevelIdx, part1.pairedAlgRank, part1.unitStart, srcOff, dstOff, subSliceSize);
    return HCCL_SUCCESS;
}

// Executor-owned tail staging. The common RS/AG templates keep their uniform-slot contract.
struct ArSeqRhdTail {
    HcclMem storage{};
    u64 rankAlign{0};
    u64 dataTypeSize{0};
    u64 inputOffset{0};
    u64 outputOffset{0};
};

inline HcclResult
ArSeqRhdPrepareTailWorkspace(u64 dataCount, u64 rankAlign, u64 dataTypeSize, HcclMem& workspace, ArSeqRhdTail& tail)
{
    CHK_PRT_RET(
        rankAlign == 0 || dataTypeSize == 0,
        HCCL_ERROR("[ArSeqRhdPrepareTailWorkspace] invalid rank alignment or data type size"), HCCL_E_PARA);
    if (dataCount % rankAlign == 0) {
        return HCCL_SUCCESS;
    }
    // Bound multiplication and rounding before reserving two aligned rank-sized tiles.
    CHK_PRT_RET(
        rankAlign > workspace.size / dataTypeSize / 2,
        HCCL_ERROR("[ArSeqRhdPrepareTailWorkspace] CCL buffer too small for tail tiles"), HCCL_E_INTERNAL);
    const u64 tileBytes = rankAlign * dataTypeSize;
    const u64 slotBytes = (tileBytes / HCCL_MIN_SLICE_ALIGN + static_cast<u64>(tileBytes % HCCL_MIN_SLICE_ALIGN != 0))
                          * HCCL_MIN_SLICE_ALIGN;
    CHK_PRT_RET(
        workspace.addr == nullptr || slotBytes >= workspace.size / 2,
        HCCL_ERROR("[ArSeqRhdPrepareTailWorkspace] no CCL workspace left after tail reservation"), HCCL_E_INTERNAL);
    tail.storage = workspace;
    tail.rankAlign = rankAlign;
    tail.dataTypeSize = dataTypeSize;
    tail.inputOffset = workspace.size - 2 * slotBytes;
    tail.outputOffset = workspace.size - slotBytes;
    workspace.size = tail.inputOffset;
    return HCCL_SUCCESS;
}

// Process the aligned prefix first; only the final count < rankAlign needs staging.
inline u64 ArSeqRhdSelectLoopCount(u64 remaining, u64 maxCountPerLoop, u64 rankAlign)
{
    const u64 count = std::min(remaining, maxCountPerLoop);
    return count >= rankAlign ? count / rankAlign * rankAlign : count;
}

// 单轮规划：有效元素数、进链的对齐槽元素数、以及是否需要尾块 staging。
// 供 executor 循环与 UT 共用，避免"整除前缀 + 余量"计数逻辑两处漂移。
struct ArSeqRhdLoopPlan {
    u64 validCount{0};
    u64 currDataCount{0};
    bool paddedTail{false};
};

inline ArSeqRhdLoopPlan ArSeqRhdPlanLoop(u64 remaining, u64 maxCountPerLoop, u64 rankAlign)
{
    ArSeqRhdLoopPlan plan{};
    plan.validCount = ArSeqRhdSelectLoopCount(remaining, maxCountPerLoop, rankAlign);
    plan.currDataCount = std::max(plan.validCount, rankAlign);
    plan.paddedTail = (plan.validCount != plan.currDataCount);
    return plan;
}

inline HcclResult ArSeqRhdStageTail(
    const OpParam& param, const ArSeqRhdTail& tail, u64 processedCount, u64 validCount, ThreadHandle thread)
{
    CHK_PRT_RET(
        validCount == 0 || validCount >= tail.rankAlign || processedCount > param.DataDes.count
            || validCount > param.DataDes.count - processedCount,
        HCCL_ERROR("[ArSeqRhdStageTail] invalid input tail range"), HCCL_E_PARA);
    const u64 validBytes = validCount * tail.dataTypeSize;
    CHK_RET(LocalCopy(
        thread, DataSlice(param.inputPtr, processedCount * tail.dataTypeSize, validBytes, validCount),
        DataSlice(tail.storage.addr, tail.inputOffset, validBytes, validCount)));
    // Duplicate initialized input into discarded lanes. Reduction is elementwise, so these
    // lanes never affect the valid prefix and need no reduce-op/data-type-specific identity.
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
        "[ArSeqRhdStageTail] processedCount[%llu] validCount[%llu] paddedCount[%llu]", processedCount, validCount,
        tail.rankAlign);
    return HCCL_SUCCESS;
}

inline HcclResult ArSeqRhdCopyTailOutput(
    const OpParam& param, const ArSeqRhdTail& tail, u64 processedCount, u64 validCount, ThreadHandle thread)
{
    const u64 validBytes = validCount * tail.dataTypeSize;
    return LocalCopy(
        thread, DataSlice(tail.storage.addr, tail.outputOffset, validBytes, validCount),
        DataSlice(param.outputPtr, processedCount * tail.dataTypeSize, validBytes, validCount));
}

} // namespace ops_hccl

#endif // HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_RHD_TAIL_H
