/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_V2_BROADCAST_SEQUENCE_EXECUTOR_UTILS_H
#define HCCL_V2_BROADCAST_SEQUENCE_EXECUTOR_UTILS_H

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <vector>
#include "alg_param.h"
#include "channel.h"
#include "template_utils.h"

namespace ops_hccl {
namespace broadcast_sequence_executor {

    // 同一批线程顺序执行 Scatter/AllGather，各模板的 notify 需求按线程下标取并集。
    inline void MergeNotifyNumPerThread(std::vector<u32>& dst, const std::vector<u32>& src)
    {
        dst.resize(std::max(dst.size(), src.size()), 0);
        for (size_t i = 0; i < src.size(); ++i) {
            dst[i] = std::max(dst[i], src[i]);
        }
    }

    // 退化层没有 peer channel 时保持模板默认 channelsPerRank_=1。
    template <typename TemplatePtr, typename ChannelMap>
    inline HcclResult SetChannelsPerRankIfNotEmpty(const TemplatePtr& algTemplate, const ChannelMap& channels)
    {
        if (channels.empty()) {
            return HCCL_SUCCESS;
        }
        return algTemplate->SetchannelsPerRank(channels);
    }

    // Scatter/AllGather 共用同一份通道资源前，两侧申请的协议和端点类型必须一致。
    // 例如 Scatter 侧保留了 RoCE、AllGatherRHD 侧已过滤 RoCE 时，直接合并会让
    // AllGatherRHD 在运行期重新拿到 CalcRes 阶段已排除的链路。这里只校验全局签名集合，
    // 不要求 peer 集合和每 peer 通道数一致，保留后续按 peer 求并集的能力。
    inline HcclResult ValidateChannelRequestCompatible(
        const std::vector<HcclChannelDesc>& scatterChannels, const std::vector<HcclChannelDesc>& allGatherChannels)
    {
        using ChannelSignature = std::tuple<CommProtocol, EndpointLocType, EndpointLocType>;
        using ChannelSignatureMap = std::map<u32, std::set<ChannelSignature>>;
        const auto collectSignatures = [](const std::vector<HcclChannelDesc>& channels) {
            ChannelSignatureMap signaturesByRank;
            for (const auto& channel : channels) {
                signaturesByRank[channel.remoteRank].emplace(
                    channel.channelProtocol, channel.localEndpoint.loc.locType, channel.remoteEndpoint.loc.locType);
            }
            return signaturesByRank;
        };

        const ChannelSignatureMap scatterSignaturesByRank = collectSignatures(scatterChannels);
        const ChannelSignatureMap allGatherSignaturesByRank = collectSignatures(allGatherChannels);
        std::set<ChannelSignature> scatterSignatures;
        std::set<ChannelSignature> allGatherSignatures;
        for (const auto& item : scatterSignaturesByRank) {
            scatterSignatures.insert(item.second.begin(), item.second.end());
        }
        for (const auto& item : allGatherSignaturesByRank) {
            allGatherSignatures.insert(item.second.begin(), item.second.end());
        }
        if (scatterSignatures == allGatherSignatures) {
            for (const auto& item : scatterSignaturesByRank) {
                const auto allGatherIt = allGatherSignaturesByRank.find(item.first);
                if (allGatherIt != allGatherSignaturesByRank.end() && allGatherIt->second != item.second) {
                    return HCCL_E_PARA;
                }
            }
            return HCCL_SUCCESS;
        }
        return HCCL_E_PARA;
    }

    // Scatter/AllGather 复用同一层通道请求：先按 remoteRank 求 peer 并集，再为每个 peer
    // 保留 channel 数更多的请求。不能按整个 vector 长度二选一，否则 UB 多通道可能让较短
    // 但 peer 更完整的集合被丢弃；同一 peer 的协议集合必须一致，避免模板拿到错误链路。
    inline HcclResult MergeChannelRequest(
        const std::vector<HcclChannelDesc>& scatterChannels, const std::vector<HcclChannelDesc>& allGatherChannels,
        std::vector<HcclChannelDesc>& mergedChannels)
    {
        std::map<u32, std::vector<HcclChannelDesc>> scatterByRank;
        std::map<u32, std::vector<HcclChannelDesc>> allGatherByRank;
        for (const auto& channel : scatterChannels) {
            scatterByRank[channel.remoteRank].push_back(channel);
        }
        for (const auto& channel : allGatherChannels) {
            allGatherByRank[channel.remoteRank].push_back(channel);
        }

        std::set<u32> peerRanks;
        for (const auto& item : scatterByRank) {
            peerRanks.insert(item.first);
        }
        for (const auto& item : allGatherByRank) {
            peerRanks.insert(item.first);
        }

        mergedChannels.clear();
        for (u32 remoteRank : peerRanks) {
            auto scatterIt = scatterByRank.find(remoteRank);
            auto allGatherIt = allGatherByRank.find(remoteRank);
            const std::vector<HcclChannelDesc>* selected = nullptr;
            if (scatterIt == scatterByRank.end()) {
                selected = &allGatherIt->second;
            } else if (allGatherIt == allGatherByRank.end()) {
                selected = &scatterIt->second;
            } else {
                std::set<CommProtocol> scatterProtocols;
                std::set<CommProtocol> allGatherProtocols;
                for (const auto& channel : scatterIt->second) {
                    scatterProtocols.insert(channel.channelProtocol);
                }
                for (const auto& channel : allGatherIt->second) {
                    allGatherProtocols.insert(channel.channelProtocol);
                }
                if (scatterProtocols != allGatherProtocols) {
                    return HCCL_E_PARA;
                }
                selected = scatterIt->second.size() >= allGatherIt->second.size() ? &scatterIt->second :
                                                                                    &allGatherIt->second;
            }
            mergedChannels.insert(mergedChannels.end(), selected->begin(), selected->end());
        }
        return HCCL_SUCCESS;
    }

    // floor/tail 分片：前 rankSize-1 个 rank 均分，最后一个 rank 承接全部余数。
    inline void SplitFloorTail(u64 totalCount, u32 rankSize, u64& sliceCount, u64& tailCount)
    {
        if (rankSize == 0) {
            sliceCount = 0;
            tailCount = 0;
            return;
        }
        sliceCount = totalCount / rankSize;
        tailCount = totalCount - sliceCount * (rankSize - 1);
    }

    // 生成所有 rank 的 floor/tail 切片，并同步写入模板使用的 sliceSize/tailSize。
    inline void FillFloorTailSlices(u64 dataCount, u64 dataTypeSize, u32 sliceNum, TemplateDataParams& params)
    {
        params.allRankSliceSize.clear();
        params.allRankDispls.clear();
        params.allRankProcessedDataCount.clear();
        params.allRankSliceSize.reserve(sliceNum);
        params.allRankDispls.reserve(sliceNum);
        params.allRankProcessedDataCount.reserve(sliceNum);

        if (sliceNum == 0) {
            params.sliceSize = 0;
            params.tailSize = 0;
            return;
        }

        u64 sliceCount = 0;
        u64 tailCount = 0;
        SplitFloorTail(dataCount, sliceNum, sliceCount, tailCount);
        params.allRankSliceSize.assign(sliceNum, sliceCount * dataTypeSize);
        params.allRankProcessedDataCount.assign(sliceNum, sliceCount);
        params.allRankSliceSize.back() = tailCount * dataTypeSize;
        params.allRankProcessedDataCount.back() = tailCount;

        params.allRankDispls.resize(sliceNum, 0);
        for (u32 rankIdx = 1; rankIdx < sliceNum; ++rankIdx) {
            params.allRankDispls[rankIdx] = params.allRankDispls[rankIdx - 1] + params.allRankSliceSize[rankIdx - 1];
        }
        params.sliceSize = params.allRankSliceSize.front();
        params.tailSize = params.allRankSliceSize.back();
    }

} // namespace broadcast_sequence_executor
} // namespace ops_hccl

#endif // HCCL_V2_BROADCAST_SEQUENCE_EXECUTOR_UTILS_H
