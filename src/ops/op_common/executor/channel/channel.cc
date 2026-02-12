/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "topo_host.h"
#include "alg_env_config.h"

namespace ops_hccl {
HcclResult CalcLevel0ChannelRequest(const OpParam& param, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    const AlgType& algType, std::vector<HcclChannelDesc> &channels)
{
    (void) param;
    (void) topoInfo;
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
        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        CHK_RET(GetUserRankBySubCommRank(rank, COMM_LEVEL0, algHierarchyInfo, channelDesc.remoteRank));
        channelDesc.channelProtocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult CalcLevel1ChannelRequest(const OpParam& param, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    const AlgType& algType, std::vector<HcclChannelDesc> &channels)
{
    (void) param;
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
        ((topoInfo->superPodNum > 1 && (topoInfo->multiSuperPodDiffServerNumMode || topoInfo->multiModuleDiffDeviceNumMode)) ||
        (topoInfo->superPodNum == 1 && topoInfo->serverNum > 1 && GetExternalInputInterHccsDisable()));
    bool isUsedRdma = isA2UsedRdma || isA3UsedRdma;

    CommProtocol protocol = isUsedRdma ? CommProtocol::COMM_PROTOCOL_ROCE : CommProtocol::COMM_PROTOCOL_HCCS;
    for (u32 rank: connectRanks) {
        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        CHK_RET(GetUserRankBySubCommRank(rank, COMM_LEVEL1, algHierarchyInfo, channelDesc.remoteRank));
        channelDesc.channelProtocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult CalcLevel2ChannelRequest(const OpParam& param, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo,
    const AlgType& algType, std::vector<HcclChannelDesc> &channels)
{
    (void) param;
    (void) topoInfo;
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
        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        CHK_RET(GetUserRankBySubCommRank(rank, COMM_LEVEL2, algHierarchyInfo, channelDesc.remoteRank));
        channelDesc.channelProtocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1D(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
    std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
#ifndef AICPU_COMPILE
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank); 
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
                HcclResult::HCCL_E_PARA);
    
    // 获取本rank的全局rankId
    u32 myRank = topoInfo->userRank;
    for (u32 rank: subcommInfo[COMM_LEVEL0]) {
        if (rank == topoInfo->userRank) {
            continue;
        }
        uint32_t *netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector = std::vector<uint32_t>(netLayers, netLayers + netLayerNum);
        for (auto netLayer : netLayersVector) {
            CommLink *linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));
            for (u32 idx = 0; idx < listSize; idx++) {
                HcclChannelDesc channelDesc;
                HcclChannelDescInit(&channelDesc, 1);
                channelDesc.remoteRank = rank;
                CommLink link = linkList[idx];
                channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
                channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                HCCL_DEBUG("[CalcChannelRequestMesh1D] local device phyId: %u, remote device phyId: %u.",
                            channelDesc.localEndpoint.loc.device.devPhyId,
                            channelDesc.remoteEndpoint.loc.device.devPhyId);
                HCCL_INFO("[CalcChannelRequestMesh1D] Add channel request between %zu and %zu, netLayerIdx %u, "
                          "linkListIdx %u, protocol %zu",
                          myRank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);
                channelDesc.channelProtocol = link.linkAttr.linkProtocol;
                channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
                channels.push_back(channelDesc);
            }
            if (listSize > 0) {
                break;
            }
        }
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh2D(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
    std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
#ifndef AICPU_COMPILE
    channels.clear();
    u32 myRank = topoInfo->userRank; // 全局rankId

    std::set<u32> connectRanks;
    if (subcommInfo.size() == 2) {
        CHK_RET(CalcMesh2DChannelConnect(myRank, subcommInfo, connectRanks));
    }
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    }

    for (u32 rank: connectRanks) {
        HcclChannelDesc channelDesc;
        HcclChannelDescInit(&channelDesc, 1);
        channelDesc.remoteRank = rank;
        CommLink *linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, 0, myRank, channelDesc.remoteRank, &linkList, &listSize));
        bool protocolExists = false;
        for (u32 idx = 0; idx < listSize; idx++) {
            CommLink link = linkList[idx];
            if (link.linkAttr.linkProtocol == protocol) {
                channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
                channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                protocolExists = true;
                HCCL_INFO("[%s]Add channel request between %zu and %zu with protocol %zu type %u", __func__,
                    myRank, channelDesc.remoteRank, link.dstEndpointDesc.protocol, link.srcEndpointDesc.commAddr.type);
                break;
            }
        }
        CHK_PRT_RET(!protocolExists,
            HCCL_ERROR("[%s] protocol[%u] not exists between %zu and %zu", __func__, protocol, myRank, channelDesc.remoteRank),
                HCCL_E_NOT_FOUND);
        channelDesc.channelProtocol = protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNhr(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
    std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
#ifndef AICPU_COMPILE
    channels.clear();
    std::set<u32> connectRanks;
    u32 myRank = topoInfo->userRank; // 全局rankId
    auto it = std::find(subcommInfo[0].begin(), subcommInfo[0].end(), myRank); 
    CHK_PRT_RET((it == subcommInfo[0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", myRank),
                HcclResult::HCCL_E_PARA);

    // 根据SubCommInfo查找LocalRank和localRankSize
    u32 localRank = std::distance(subcommInfo[0].begin(), it);;
    u32 localRankSize = subcommInfo[0].size();
    CHK_RET(CalcNHRChannelConnect(localRank, localRankSize, INVALID_VALUE_RANKID, connectRanks));
    for (u32 rankIdx: connectRanks) {
        uint32_t *netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector = std::vector<uint32_t>(netLayers, netLayers + netLayerNum);
        for (auto netLayer : netLayersVector) {
            CommLink *linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, subcommInfo[0][rankIdx], &linkList, &listSize));
            for (u32 idx = 0; idx < listSize; idx++) {
                HcclChannelDesc channelDesc;
                HcclChannelDescInit(&channelDesc, 1);
                channelDesc.remoteRank = subcommInfo[0][rankIdx];
                CommLink link = linkList[idx];
                channelDesc.localEndpoint = link.srcEndpointDesc;
                channelDesc.remoteEndpoint = link.dstEndpointDesc;
                HCCL_DEBUG("[CalcLevel1ChannelRequestNhr] local device phyId: %u, remote device phyId: %u.",
                            channelDesc.localEndpoint.loc.device.devPhyId,
                            channelDesc.remoteEndpoint.loc.device.devPhyId);
                HCCL_INFO("[CalcLevel1ChannelRequestNhr] Add channel request between %zu and %zu, netLayerIdx %u, "
                          "linkListIdx %u, protocol %zu",
                          myRank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);
                channelDesc.channelProtocol = link.linkAttr.linkProtocol;
                channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
                channels.push_back(channelDesc);
            }
            if (listSize > 0) {
                break;
            }
        }
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CreateChannelRequestByRankId(HcclComm comm, u32 myRank, u32 remoteRank,
    std::vector<HcclChannelDesc> &channels)
{
#ifndef AICPU_COMPILE
    channels.clear();
    
    uint32_t *netLayers;
    uint32_t netLayerNum;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector = std::vector<uint32_t>(netLayers, netLayers + netLayerNum);
    bool findFlag = false;
    for (auto netLayer : netLayersVector) {
        CommLink *linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize));
        for (u32 idx = 0; idx < listSize; idx++) {
            HcclChannelDesc channelDesc;
            HcclChannelDescInit(&channelDesc, 1);
            channelDesc.remoteRank = remoteRank;
            CommLink link = linkList[idx];
            channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
            channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            HCCL_INFO("[CreateChannelRequestByRankId] Add channel request between %zu and %zu with protocol %zu", \
                myRank, channelDesc.remoteRank, channelDesc.remoteEndpoint.protocol);
            channelDesc.channelProtocol = link.linkAttr.linkProtocol;
            channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
            channels.push_back(channelDesc);
        }
        if (listSize > 0) {
            findFlag = true;
            break;
        }
    }
    if (!findFlag) {
        HCCL_ERROR("[CreateChannelRequestByRankId] My rank %zu has no link with remote rank %zu", \
            myRank, remoteRank);
        return HCCL_E_INTERNAL;
    }

#endif
    return HCCL_SUCCESS;
}
}