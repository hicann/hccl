/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include <vector>
#include <set>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "alg_type.h"
#include "channel_request.h"
#include "topo.h"
#include "alg_env_config.h"

namespace ops_hccl {
HcclResult CalcLevel0ChannelRequest(const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgType& algType, std::vector<ChannelDesc> &channels)
{
    channels.clear();
    SubCommInfo &subCommInfo = algHierarchyInfo.infos[COMM_LEVEL0];
    std::set<u32> connectRanks; // 非通信域rank

    switch (algType.algoLevel0) {
        case AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING:
        case AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING:
            CHK_RET(CalcRingChannelConnect(subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_MESH:
        default:
            CHK_RET(CalcMeshChannelConnect(subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
    }

    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    for (u32 rank: connectRanks) {
        ChannelDesc channelDesc;
        CHK_RET(GetUserRankBySubCommRank(rank, COMM_LEVEL0, algHierarchyInfo, channelDesc.remoteRank));
        channelDesc.protocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult CalcLevel1ChannelRequest(const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgType& algType, std::vector<ChannelDesc> &channels)
{
    channels.clear();
    SubCommInfo &subCommInfo = algHierarchyInfo.infos[COMM_LEVEL1];
    std::set<u32> connectRanks; // 非通信域rank

    switch (algType.algoLevel1) {
        case AlgTypeLevel1::ALG_LEVEL1_NB:
            CHK_RET(CalcNBChannelConnect(subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        case AlgTypeLevel1::ALG_LEVEL1_NHR:
            CHK_RET(CalcNHRChannelConnect(subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        default:
            CHK_RET(CalcRingChannelConnect(subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
    }

    // level1走rdma的几种条件：A2单机A+X开启switch；A2多机；A3开启disableHccs；A3跨超卡数不一致
    bool isA2UsedRdma = topoInfo->deviceType == DevType::DEV_TYPE_910B && (topoInfo->serverNum > 1 ||
        (topoInfo->serverNum == 1 && topoInfo->isDiffDeviceModule && GetExternalInputIntraRoceSwitch()));
    bool isA3UsedRdma = topoInfo->deviceType == DevType::DEV_TYPE_910_93 &&
        ((topoInfo->superPodNum > 1 && topoInfo->multiSuperPodDiffServerNumMode) ||
        (topoInfo->superPodNum == 1 && topoInfo->serverNum > 1 && GetExternalInputInterHccsDisable()));
    bool isUsedRdma = isA2UsedRdma || isA3UsedRdma;

    CommProtocol protocol = isUsedRdma ? CommProtocol::COMM_PROTOCOL_ROCE : CommProtocol::COMM_PROTOCOL_HCCS;
    for (u32 rank: connectRanks) {
        ChannelDesc channelDesc;
        CHK_RET(GetUserRankBySubCommRank(rank, COMM_LEVEL1, algHierarchyInfo, channelDesc.remoteRank));
        channelDesc.protocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult CalcLevel2ChannelRequest(const OpParam& param, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    AlgType& algType, std::vector<ChannelDesc> &channels)
{
    channels.clear();
    SubCommInfo &subCommInfo = algHierarchyInfo.infos[COMM_LEVEL2];
    std::set<u32> connectRanks; // 非通信域rank

    switch (algType.algoLevel2) {
        case AlgTypeLevel2::ALG_LEVEL2_NB:
            CHK_RET(CalcNBChannelConnect(subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        case AlgTypeLevel2::ALG_LEVEL2_NHR:
            CHK_RET(CalcNHRChannelConnect(subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
        default:
            CHK_RET(CalcRingChannelConnect(subCommInfo.localRank, subCommInfo.localRankSize, INVALID_VALUE_RANKID, connectRanks));
            break;
    }

    // level2当前一定走rdma
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_ROCE;

    for (u32 rank: connectRanks) {
        ChannelDesc channelDesc;
        CHK_RET(GetUserRankBySubCommRank(rank, COMM_LEVEL2, algHierarchyInfo, channelDesc.remoteRank));
        channelDesc.protocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}
}