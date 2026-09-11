/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DATA_OPS_H
#define DATA_OPS_H

#include <vector>
#include <algorithm>
#include "alg_param.h"
#include "data_types.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

inline HcclResult CheckInputDataRanks(const DataParams& tempAlgParams, const char* tag)
{
    CHK_PRT_RET(
        tempAlgParams.ranksForInputData.empty(), HCCL_ERROR("[%s] ranksForInputData is empty.", tag), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

// channel 维度的数据切分信息：split/offset 成组出现，tail 向量为空表示无尾块。
struct ChannelSplitInfo {
    std::vector<u64> dataSplit{};
    std::vector<u64> dataOffset{};
    std::vector<u64> dataSplitTail{};
    std::vector<u64> dataOffsetTail{};

    bool ByChannel() const { return !dataSplit.empty(); }
    bool HasTail() const { return !dataSplitTail.empty(); }
    u64 Split(u32 channelIdx, bool isTailRank) const
    {
        return isTailRank ? dataSplitTail[channelIdx] : dataSplit[channelIdx];
    }
    u64 Offset(u32 channelIdx, bool isTailRank) const
    {
        return isTailRank ? dataOffsetTail[channelIdx] : dataOffset[channelIdx];
    }
};

HcclResult PreCopyData(
    const DataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u32>& ranksForInputData,
    u32 channelIdx = 0, const ChannelSplitInfo& channelSplit = {});

HcclResult CalcRanksForOutput(
    const std::vector<u32>& ranksForInputData, const std::vector<u32>& subCommRanks, u32 myRank,
    std::vector<u32>& ranksForOutputData);

HcclResult PostCopyData(
    const DataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u32>& ranksForOutputData,
    u32 channelIdx = 0, const ChannelSplitInfo& channelSplit = {}, const std::vector<u32>& skipRanks = {});

// 合并直拷：inputBufferPtr → outputBufferPtr，跳过 cclBuffer 中转
HcclResult DirectCopyData(
    const DataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u32>& ranksForData,
    u32 channelIdx = 0, const ChannelSplitInfo& channelSplit = {});

struct DataSizeInfo {
    u32 dataTypeSize{0};
    u64 sliceSize{0};
    u64 tailSize{0};
};

DataSizeInfo CalcDataSizeInfo(const DataParams& tempAlgParams);

inline u64 CalcRankDataSize(const DataSizeInfo& sizeInfo, u32 rank, u32 globalTailRankId)
{
    return (sizeInfo.tailSize > 0 && rank == globalTailRankId) ? sizeInfo.tailSize : sizeInfo.sliceSize;
}

HcclResult CalcDataSplitByPortGroup(
    u64 totalDataSize, u32 dataTypeSize, const std::vector<ChannelInfo>& channels, std::vector<u64>& dataSplit,
    std::vector<u64>& dataOffset);

} // namespace ops_hccl

#endif // DATA_OPS_H
