/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_ops.h"

#include "log.h"

#include <limits>

namespace ops_hccl {

DataSizeInfo CalcDataSizeInfo(const DataParams& tempAlgParams)
{
    DataSizeInfo info;
    info.dataTypeSize = DATATYPE_SIZE_TABLE[tempAlgParams.dataType];
    info.sliceSize = tempAlgParams.sliceCount * info.dataTypeSize;
    info.tailSize = (tempAlgParams.tailCount == 0) ? info.sliceSize :
                                                     (info.sliceSize + tempAlgParams.tailCount * info.dataTypeSize);
    return info;
}

HcclResult CalcRanksForOutput(
    const std::vector<u32>& ranksForInputData, const std::vector<u32>& subCommRanks, u32 myRank,
    std::vector<u32>& ranksForOutputData)
{
    ranksForOutputData.resize(ranksForInputData.size() * subCommRanks.size());
    size_t outputIdx = 0;
    for (u32 subCommRank : subCommRanks) {
        const long long rankOffset = static_cast<long long>(subCommRank) - static_cast<long long>(myRank);
        for (u32 inputRank : ranksForInputData) {
            const long long outputRank = static_cast<long long>(inputRank) + rankOffset;
            CHK_PRT_RET(
                outputRank < 0 || outputRank > static_cast<long long>(std::numeric_limits<u32>::max()),
                HCCL_ERROR(
                    "[CalcRanksForOutput] outputRank is invalid, inputRank=%u, "
                    "subCommRank=%u, myRank=%u.",
                    inputRank, subCommRank, myRank),
                HCCL_E_PARA);
            ranksForOutputData[outputIdx++] = static_cast<u32>(outputRank);
        }
    }
    std::sort(ranksForOutputData.begin(), ranksForOutputData.end());
    return HCCL_SUCCESS;
}

HcclResult PreCopyData(
    const DataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u32>& ranksForInputData,
    u32 channelIdx, const ChannelSplitInfo& channelSplit)
{
    HCCL_DEBUG("[PreCopyData] start, channelIdx[%u].", channelIdx);

    if (ranksForInputData.empty()) {
        HCCL_ERROR("[PreCopyData] ranksForInputData is empty.");
        return HCCL_E_INTERNAL;
    }

    if (tempAlgParams.inputBufferType == BufferType::HCCL_BUFFER
        || tempAlgParams.inputBufferPtr == tempAlgParams.cclBufferPtr) {
        return HCCL_SUCCESS;
    }

    const DataSizeInfo sizeInfo = CalcDataSizeInfo(tempAlgParams);
    const bool byChannel = channelSplit.ByChannel();
    const bool hasTail = channelSplit.HasTail();

    for (size_t idx = 0; idx < ranksForInputData.size(); ++idx) {
        u32 rank = ranksForInputData[idx];
        const bool isTailRank = (hasTail && rank == tempAlgParams.globalTailRankId);
        const u64 curSplit = byChannel ? channelSplit.Split(channelIdx, isTailRank) :
                                         CalcRankDataSize(sizeInfo, rank, tempAlgParams.globalTailRankId);
        if (curSplit == 0) {
            continue;
        }
        const u64 sliceCount = curSplit / sizeInfo.dataTypeSize;
        const u64 curOffset = byChannel ? channelSplit.Offset(channelIdx, isTailRank) : 0;

        const u64 inOff
            = tempAlgParams.dataOffset + tempAlgParams.sliceOffset + idx * tempAlgParams.dataStride + curOffset;
        const u64 cclOff = tempAlgParams.sliceOffset + rank * tempAlgParams.scratchStride + curOffset;
        DataSlice srcSlice(tempAlgParams.inputBufferPtr, inOff, curSplit, sliceCount);
        DataSlice dstSlice(tempAlgParams.cclBufferPtr, cclOff, curSplit, sliceCount);
        CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
    }

    HCCL_DEBUG("[PreCopyData] end.");
    return HCCL_SUCCESS;
}

HcclResult PostCopyData(
    const DataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u32>& ranksForOutputData,
    u32 channelIdx, const ChannelSplitInfo& channelSplit, const std::vector<u32>& skipRanks)
{
    HCCL_DEBUG("[PostCopyData] start, channelIdx[%u].", channelIdx);

    if (tempAlgParams.outputBufferType == BufferType::HCCL_BUFFER || tempAlgParams.enableRemoteMemAccess) {
        return HCCL_SUCCESS;
    }

    const DataSizeInfo sizeInfo = CalcDataSizeInfo(tempAlgParams);
    const bool byChannel = channelSplit.ByChannel();
    const bool hasTail = channelSplit.HasTail();

    for (size_t idx = 0; idx < ranksForOutputData.size(); ++idx) {
        u32 rank = ranksForOutputData[idx];
        if (std::find(skipRanks.begin(), skipRanks.end(), rank) != skipRanks.end()) {
            continue;
        }
        const bool isTailRank = (hasTail && rank == tempAlgParams.globalTailRankId);
        const u64 curSplit = byChannel ? channelSplit.Split(channelIdx, isTailRank) :
                                         CalcRankDataSize(sizeInfo, rank, tempAlgParams.globalTailRankId);
        if (curSplit == 0) {
            continue;
        }
        const u64 sliceCount = curSplit / sizeInfo.dataTypeSize;
        const u64 curOffset = byChannel ? channelSplit.Offset(channelIdx, isTailRank) : 0;
        const u64 cclOff = tempAlgParams.sliceOffset + rank * tempAlgParams.scratchStride + curOffset;
        const u64 outOff
            = tempAlgParams.dataOffset + tempAlgParams.sliceOffset + idx * tempAlgParams.dataStride + curOffset;
        DataSlice srcSlice(tempAlgParams.cclBufferPtr, cclOff, curSplit, sliceCount);
        DataSlice dstSlice(tempAlgParams.outputBufferPtr, outOff, curSplit, sliceCount);
        CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
    }

    HCCL_DEBUG("[PostCopyData] end.");
    return HCCL_SUCCESS;
}

HcclResult DirectCopyData(
    const DataParams& tempAlgParams, const ThreadHandle& thread, const std::vector<u32>& ranksForData, u32 channelIdx,
    const ChannelSplitInfo& channelSplit)
{
    HCCL_DEBUG("[DirectCopyData] start, channelIdx[%u].", channelIdx);

    if (ranksForData.empty()) {
        HCCL_ERROR("[DirectCopyData] ranksForData is empty.");
        return HCCL_E_INTERNAL;
    }

    if (tempAlgParams.inputBufferType == BufferType::HCCL_BUFFER
        || tempAlgParams.inputBufferPtr == tempAlgParams.outputBufferPtr) {
        return HCCL_SUCCESS;
    }

    const DataSizeInfo sizeInfo = CalcDataSizeInfo(tempAlgParams);
    const bool byChannel = channelSplit.ByChannel();
    const bool hasTail = channelSplit.HasTail();

    for (size_t idx = 0; idx < ranksForData.size(); ++idx) {
        u32 rank = ranksForData[idx];
        const bool isTailRank = (hasTail && rank == tempAlgParams.globalTailRankId);
        const u64 curSplit = byChannel ? channelSplit.Split(channelIdx, isTailRank) :
                                         CalcRankDataSize(sizeInfo, rank, tempAlgParams.globalTailRankId);
        if (curSplit == 0) {
            continue;
        }
        const u64 sliceCount = curSplit / sizeInfo.dataTypeSize;
        const u64 curOffset = byChannel ? channelSplit.Offset(channelIdx, isTailRank) : 0;
        const u64 off
            = tempAlgParams.dataOffset + tempAlgParams.sliceOffset + idx * tempAlgParams.dataStride + curOffset;
        DataSlice srcSlice(tempAlgParams.inputBufferPtr, off, curSplit, sliceCount);
        DataSlice dstSlice(tempAlgParams.outputBufferPtr, off, curSplit, sliceCount);
        CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
    }

    HCCL_DEBUG("[DirectCopyData] end.");
    return HCCL_SUCCESS;
}

HcclResult CalcDataSplitByPortGroup(
    u64 totalDataSize, u32 dataTypeSize, const std::vector<ChannelInfo>& channels, std::vector<u64>& dataSplit,
    std::vector<u64>& dataOffset)
{
    u32 channelNum = static_cast<u32>(channels.size());
    if (channelNum == 0) {
        HCCL_ERROR("[CalcDataSplitByPortGroup] channels is empty.");
        return HCCL_E_INTERNAL;
    }
    u32 totalPorts = 0;
    for (const auto& ch : channels) {
        totalPorts += ch.portGroupSize;
    }
    if (totalPorts == 0) {
        totalPorts = channelNum;
    }

    dataSplit.assign(channelNum, 0);
    dataOffset.assign(channelNum, 0);
    u64 allocated = 0;
    for (u32 i = 0; i < channelNum; ++i) {
        if (i == channelNum - 1) {
            dataSplit[i] = totalDataSize - allocated;
        } else {
            u64 splitSize = totalDataSize * channels[i].portGroupSize / totalPorts;
            splitSize = splitSize / dataTypeSize * dataTypeSize;
            dataSplit[i] = splitSize;
            allocated += splitSize;
        }
    }
    u64 offset = 0;
    for (u32 i = 0; i < channelNum; ++i) {
        dataOffset[i] = offset;
        offset += dataSplit[i];
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
