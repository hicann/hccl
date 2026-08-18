/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "omnipipe_template_utils.h"
#include <algorithm>
#include <vector>

namespace ops_hccl {
HcclResult FillOmniPipeTemplateAlgParams(
    TemplateDataParams& tempAlgParams, const StepSliceInfo& stepSliceInfo, bool supportSymmetricMemory,
    u64 processedDataCount, u64 dataTypeSize)
{
    tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.hcclBuffBaseOff = stepSliceInfo.buffInfo.hcclBuffBaseOff;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    if (supportSymmetricMemory) {
        // user input 保持完整分片布局，只有输入基址随已处理数据量推进；ccl scratch 仍从 0 复用。
        tempAlgParams.buffInfo.inBuffBaseOff += processedDataCount * dataTypeSize;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult PrepareOmniPipeDataSplitForMultiChannel(
    CommonAlgTemplateBase* algTemplate, const TemplateDataParams& tempAlgParams, HcclDataType dataType,
    TemplateResource& templateResource, std::vector<std::vector<std::vector<u64>>>& dataSplitVec,
    std::vector<std::vector<std::vector<u64>>>& dataOffsetVec)
{
    dataSplitVec.clear();
    dataOffsetVec.clear();
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    for (uint32_t idx = 0; idx < tempAlgParams.stepSliceInfo.stepSliceSize.size(); idx++) {
        std::vector<std::vector<u64>> dataSplitVecByRepeat;
        std::vector<std::vector<u64>> dataOffsetVecByRepeat;
        for (uint32_t rpt = 0; rpt < tempAlgParams.stepSliceInfo.stepSliceSize[0].size(); rpt++) {
            u64 totalDataCount = tempAlgParams.stepSliceInfo.stepSliceSize[idx][rpt] / dataTypeSize;
            std::vector<u64> dataSplit;
            std::vector<u64> dataOffset;
            std::vector<u64> curElemCountOut;
            algTemplate->CalcDataSplitByPortGroup(
                totalDataCount, dataTypeSize, templateResource.channels.begin()->second, curElemCountOut, dataSplit,
                dataOffset);
            dataSplitVecByRepeat.push_back(dataSplit);
            dataOffsetVecByRepeat.push_back(dataOffset);
        }
        dataSplitVec.push_back(dataSplitVecByRepeat);
        dataOffsetVec.push_back(dataOffsetVecByRepeat);
    }
    return HcclResult::HCCL_SUCCESS;
}

namespace {
    bool TryClassifyOmniPipeChannel(
        u32 localRank, const ChannelInfo& channel,
        const std::vector<const std::vector<std::vector<u32>>*>& subCommsByLevel,
        const std::vector<uint64_t>& rankSizesByLevel,
        std::vector<std::map<u32, std::vector<ChannelInfo>>>& channelsByLevel)
    {
        for (u32 level = 0; level < subCommsByLevel.size(); ++level) {
            if (subCommsByLevel[level] == nullptr || rankSizesByLevel[level] <= 1) {
                continue;
            }

            const auto& subComms = *subCommsByLevel[level];
            const auto subCommIter = std::find_if(
                subComms.begin(), subComms.end(), [localRank, &channel](const std::vector<u32>& subComm) {
                    const bool containsLocalRank
                        = std::find(subComm.begin(), subComm.end(), localRank) != subComm.end();
                    const bool containsRemoteRank
                        = std::find(subComm.begin(), subComm.end(), channel.remoteRank) != subComm.end();
                    return containsLocalRank && containsRemoteRank;
                });
            if (subCommIter == subComms.end()) {
                continue;
            }

            channelsByLevel[level][channel.remoteRank].push_back(channel);
            return true;
        }
        return false;
    }
} // namespace

HcclResult ClassifyOmniPipeChannelsByLevel(
    u32 localRank, const std::vector<std::vector<ChannelInfo>>& channels,
    const std::vector<const std::vector<std::vector<u32>>*>& subCommsByLevel,
    const std::vector<uint64_t>& rankSizesByLevel,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& channelsByLevel)
{
    if (subCommsByLevel.size() != rankSizesByLevel.size()) {
        HCCL_ERROR(
            "[ClassifyOmniPipeChannelsByLevel] level count mismatch, subCommLevelCount[%zu], "
            "rankSizeLevelCount[%zu].",
            subCommsByLevel.size(), rankSizesByLevel.size());
        return HCCL_E_PARA;
    }

    channelsByLevel.assign(subCommsByLevel.size(), {});
    for (const auto& channelGroup : channels) {
        for (const auto& channel : channelGroup) {
            if (!TryClassifyOmniPipeChannel(localRank, channel, subCommsByLevel, rankSizesByLevel, channelsByLevel)) {
                HCCL_WARNING(
                    "[ClassifyOmniPipeChannelsByLevel] discard unclassified channel, "
                    "remoteRank[%u] is absent from every active sub-communicator.",
                    channel.remoteRank);
            }
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
