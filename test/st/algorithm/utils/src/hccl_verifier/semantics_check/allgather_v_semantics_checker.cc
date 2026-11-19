/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_v_semantics_checker.h"

#include <algorithm>
#include <map>
#include <vector>
#include "log.h"

namespace HcclSim {
HcclResult
TaskCheckAllGatherVSemantics(std::map<RankId, RankMemorySemantics>& allRankMemSemantics, VDataDesTag& vDataDes)
{
    u32 rankSize = allRankMemSemantics.size();
    if (rankSize == 0) {
        return HCCL_SUCCESS;
    }
    if (vDataDes.counts.size() < rankSize) {
        HCCL_ERROR("[AllGatherV] counts size[%zu] is less than rankSize[%u]", vDataDes.counts.size(), rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    // displs 允许空洞；缺省时按 counts 连续排布兼容既有用例。
    std::vector<u64> displs = vDataDes.displs;
    if (displs.size() < rankSize) {
        displs.assign(rankSize, 0);
        u64 offset = 0;
        for (u32 i = 0; i < rankSize; i++) {
            displs[i] = offset;
            offset += vDataDes.counts[i];
        }
    }

    const u64 dataTypeSize = HCCL_SIZE_TABLE[vDataDes.dataType];
    u64 totalDataSize = 0;
    for (u32 i = 0; i < rankSize; i++) {
        totalDataSize += vDataDes.counts[i] * dataTypeSize;
    }

    // 按起始地址排序非零区间（兼容非单调 displs，SE 决策允许乱序布局）。
    // 单调 displs 下排序结果与 rank 索引序一致，既有用例行为不变。
    struct RangeEntry {
        u64 start;
        u64 size;
        u32 rankIdx;
    };
    std::vector<RangeEntry> sortedRanges;
    for (u32 i = 0; i < rankSize; i++) {
        if (vDataDes.counts[i] > 0) {
            sortedRanges.push_back({displs[i] * dataTypeSize, vDataDes.counts[i] * dataTypeSize, i});
        }
    }
    std::sort(sortedRanges.begin(), sortedRanges.end(), [](const RangeEntry& a, const RangeEntry& b) {
        return a.start < b.start;
    });

    for (RankId rankId = 0; rankId < rankSize; rankId++) {
        auto rankIter = allRankMemSemantics.find(rankId);
        if (rankIter == allRankMemSemantics.end()) {
            HCCL_ERROR("Missing rank %u mem semantics", rankId);
            return HcclResult::HCCL_E_PARA;
        }

        const auto& outputSemantics = rankIter->second[BufferType::OUTPUT];
        u64 checkedSize = 0;
        size_t curRange = 0;
        u64 curDataSize = 0;

        for (const auto& ele : outputSemantics) {
            // 定位 ele 所属的源 rank 区间（按地址排序后推进），拒绝落在空洞里的语义块。
            while (curRange < sortedRanges.size()) {
                const u64 rangeStart = sortedRanges[curRange].start;
                const u64 rangeSize = sortedRanges[curRange].size;
                if (ele.startAddr < rangeStart) {
                    HCCL_ERROR(
                        "[rankId:%u]buffer semantic startAddr %llu falls into gap before rank %u range start %llu",
                        rankId, ele.startAddr, sortedRanges[curRange].rankIdx, rangeStart);
                    return HcclResult::HCCL_E_PARA;
                }
                if (ele.startAddr >= rangeStart + rangeSize) {
                    curRange++;
                    curDataSize = 0;
                    continue;
                }
                break;
            }
            if (curRange >= sortedRanges.size()) {
                HCCL_ERROR(
                    "[rankId:%u]buffer semantic startAddr %llu exceeds all displs ranges", rankId, ele.startAddr);
                return HcclResult::HCCL_E_PARA;
            }

            const u64 rangeStart = sortedRanges[curRange].start;
            const u64 rangeSize = sortedRanges[curRange].size;
            const u64 expectedStart = rangeStart + curDataSize;
            if (ele.startAddr != expectedStart) {
                HCCL_ERROR(
                    "[rankId:%u]Missing buffer semantic: expected startAddr is %llu, while cur startAddr is %llu, "
                    "cur buffer semantic is %s",
                    rankId, expectedStart, ele.startAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            if (ele.size == 0 || ele.size > rangeSize - curDataSize) {
                HCCL_ERROR(
                    "[rankId:%u]buffer semantic size %llu exceeds rank %u remaining range %llu", rankId, ele.size,
                    sortedRanges[curRange].rankIdx, rangeSize - curDataSize);
                return HcclResult::HCCL_E_PARA;
            }
            if (ele.srcBufs.size() != 1) {
                HCCL_ERROR(
                    "[rankId:%u]Cur buffer semantic should not be reduce, which mean srcBufs size should be 1, "
                    "while cur buffer semantic is %s",
                    rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();
            if (srcBuf.rankId != sortedRanges[curRange].rankIdx) {
                HCCL_ERROR(
                    "[rankId:%u]Cur buffer semantic should come from rank %u, while it come from rank %u, "
                    "cur buffer semantic is %s",
                    rankId, sortedRanges[curRange].rankIdx, srcBuf.rankId, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_ERROR(
                    "[rankId:%u]Cur buffer semantic srcBufs bufType is not INPUT, cur buffer semantic is %s", rankId,
                    ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }
            if (srcBuf.srcAddr != curDataSize) {
                HCCL_ERROR(
                    "[rankId:%u]Cur buffer semantic srcBufs srcAddr should be %llu, while it is %llu, cur buffer "
                    "semantic is %s",
                    rankId, curDataSize, srcBuf.srcAddr, ele.Describe().c_str());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;
            if (curDataSize == rangeSize) {
                curRange++;
                curDataSize = 0;
            }
            checkedSize += ele.size;
        }

        if (checkedSize != totalDataSize) {
            HCCL_ERROR(
                "[rankId:%u]Missing buffer semantics in tail: already checked total size is %llu, while expected "
                "total size is %llu",
                rankId, checkedSize, totalDataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

} // namespace HcclSim
