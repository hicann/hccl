/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "barrier_rhd_utils.h"
#include <algorithm>
#include <limits>
#include <set>
#include "channel.h"

namespace ops_hccl {
HcclResult
BuildBarrierRhdSteps(u32 myRank, const std::vector<std::vector<u32>>& subCommRanks, std::vector<BarrierRhdStep>& steps)
{
    steps.clear();
    CHK_PRT_RET(
        subCommRanks.size() != 1 || subCommRanks[0].empty() || subCommRanks[0].size() > std::numeric_limits<u32>::max(),
        HCCL_ERROR("[BarrierRHD] invalid subcomm for rank[%u].", myRank), HCCL_E_PARA);
    const auto& ranks = subCommRanks[0];
    std::set<u32> uniqueRanks(ranks.begin(), ranks.end());
    CHK_PRT_RET(uniqueRanks.size() != ranks.size(), HCCL_ERROR("[BarrierRHD] duplicate subcomm ranks."), HCCL_E_PARA);
    u32 algRank = 0;
    CHK_RET(GetAlgRank(myRank, ranks, algRank));
    const u32 rankSize = static_cast<u32>(ranks.size());
    RhdPart1Info info;
    CHK_RET(CalcRhdPart1Info(rankSize, algRank, INVALID_VALUE_RANKID, info));
    RhdStepInfo pre;
    RhdStepInfo post;
    CHK_RET(GenRhdPrePostSteps(info, pre, post, RhdSegMode::SHARE, RhdSegMode::FULL));
    if (info.inPart1) {
        CHK_PRT_RET(
            pre.peer >= rankSize || post.peer >= rankSize,
            HCCL_ERROR("[BarrierRHD] invalid part1 peer for rank[%u].", myRank), HCCL_E_INTERNAL);
        steps.push_back({RhdPhase::PRE, pre.step, ranks[pre.peer]});
    }
    // 调用方过滤被折叠的 rank，仅为主体参与者生成 AG 步骤。
    if (!info.inPart1 || info.isRepresentative) {
        std::vector<RhdStepInfo> mainSteps;
        CHK_RET(GenRhdAllGatherSteps(info, mainSteps));
        for (const auto& step : mainSteps) {
            const u32 peer = RhdAlgRankOfVirtual(rankSize, INVALID_VALUE_RANKID, step.peer);
            CHK_PRT_RET(
                peer >= rankSize || peer == algRank,
                HCCL_ERROR("[BarrierRHD] invalid MAIN peer[%u] for rank[%u].", peer, myRank), HCCL_E_INTERNAL);
            steps.push_back({RhdPhase::MAIN_AG, step.step, ranks[peer], step.peer, peer});
        }
    }
    if (info.inPart1) {
        steps.push_back({RhdPhase::POST, post.step, ranks[post.peer]});
    }
    return HCCL_SUCCESS;
}

HcclResult CalcBarrierRhdResources(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subCommRanks, bool hostDpu, AlgResourceRequest& resourceRequest)
{
#ifndef AICPU_COMPILE
    CHK_PTR_NULL(topoInfo);
    std::vector<BarrierRhdStep> steps;
    CHK_RET(BuildBarrierRhdSteps(topoInfo->userRank, subCommRanks, steps));
    std::set<u32> peers;
    for (const auto& step : steps) {
        peers.insert(step.peerRank);
    }
    std::vector<HcclChannelDesc> channels;
    if (!peers.empty() && hostDpu) {
        std::vector<u32> partnerSubcomm{topoInfo->userRank};
        partnerSubcomm.insert(partnerSubcomm.end(), peers.begin(), peers.end());
        CHK_RET(CalcChannelRequestMesh1DHighestHostRoce(comm, param, topoInfo, {partnerSubcomm}, channels));
    } else {
        for (u32 peer : peers) {
            std::vector<HcclChannelDesc> peerChannels;
            CHK_RET(CreateChannelRequestByRankId(comm, param, topoInfo->userRank, peer, peerChannels));
            // HostDPU 执行器的内层 RHD 也由 AICPU 执行，因此需要 DEVICE 端点。
            for (const auto& channel : peerChannels) {
                CHK_PRT_RET(
                    channel.localEndpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE
                        || channel.remoteEndpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE,
                    HCCL_ERROR(
                        "[BarrierRHD] AICPU requires DEVICE endpoints: rank[%u] peer[%u] localLoc[%u] remoteLoc[%u].",
                        topoInfo->userRank, peer, static_cast<u32>(channel.localEndpoint.loc.locType),
                        static_cast<u32>(channel.remoteEndpoint.loc.locType)),
                    HCCL_E_NOT_SUPPORT);
            }
            channels.insert(channels.end(), peerChannels.begin(), peerChannels.end());
        }
    }
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.channels = {channels};
    return HCCL_SUCCESS;
#else
    // 资源计算仅在 Host 侧执行。
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subCommRanks;
    (void)hostDpu;
    (void)resourceRequest;
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult CheckBarrierRhdChannels(
    const std::vector<BarrierRhdStep>& steps, const std::map<u32, std::vector<ChannelInfo>>& channels,
    EndpointLocType expectedLocation)
{
    for (const auto& step : steps) {
        auto it = channels.find(step.peerRank);
        CHK_PRT_RET(
            it == channels.end() || it->second.empty(),
            HCCL_ERROR(
                "[BarrierRHD] missing channel: phase[%u] step[%u] peer[%u].", static_cast<u32>(step.phase), step.step,
                step.peerRank),
            HCCL_E_INTERNAL);
        const auto& channel = it->second[0];
        CHK_PRT_RET(
            channel.locationType != expectedLocation,
            HCCL_ERROR(
                "[BarrierRHD] incompatible channel: phase[%u] step[%u] peer[%u] location[%u] expected[%u].",
                static_cast<u32>(step.phase), step.step, step.peerRank, static_cast<u32>(channel.locationType),
                static_cast<u32>(expectedLocation)),
            HCCL_E_NOT_SUPPORT);
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
