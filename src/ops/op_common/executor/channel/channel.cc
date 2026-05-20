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
#include "ccu_alg_template_base.h"

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

HcclResult ProcessMeshInfo(HcclComm comm,const std::vector<std::vector<u32>>& subcommInfo,
                        std::map<u32, u32>& rank2ChannelIdx, u32 myRank,
                        std::vector<std::vector<HcclChannelDesc>>& channelsPerDie,
                        u32 enableDieNum, u32 enableDieId,
                        std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
#ifndef AICPU_COMPILE
    constexpr u32 DIE_NUM_1 = 1;
    constexpr u32 DIE_NUM_2 = 2;
    constexpr u32 DIE_0 = 0;
    constexpr u32 DIE_1 = 1;
    for(u32 rank: subcommInfo[COMM_LEVEL0]){
        HCCL_INFO("rank = %lld",rank);
        if (rank == myRank) {
            continue;
        }
        if (enableDieNum == DIE_NUM_1) {
            CHK_RET(CcuAlgTemplateBase::SelectChannelToVec(comm, myRank, rank, rankIdToChannelDesc, enableDieId,
                rank2ChannelIdx, channelsPerDie[DIE_0]));
            HCCL_INFO("enableDieNum = %lld",enableDieNum);
        } else if (enableDieNum == DIE_NUM_2) {
            // 加入fromRank 2个die的链路
            CHK_RET(CcuAlgTemplateBase::SelectChannelToVec(comm, myRank, rank, rankIdToChannelDesc, DIE_0,
                rank2ChannelIdx, channelsPerDie[DIE_0]));
            CHK_RET(CcuAlgTemplateBase::SelectChannelToVec(comm, myRank, rank, rankIdToChannelDesc, DIE_1,
                rank2ChannelIdx, channelsPerDie[DIE_1]));
            HCCL_INFO("enableDieNum = %lld",enableDieNum);
        }
    }
    return HcclResult::HCCL_SUCCESS;
#endif
}

HcclResult ProcessFlattenLink(HcclComm comm, u32 myRank, const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
#ifndef AICPU_COMPILE
    std::map<u32, std::vector<HcclChannelDesc>> rankIdToChannelDesc;
    CHK_RET(CcuAlgTemplateBase::RestoreChannelMap(channels, rankIdToChannelDesc));
    uint32_t enableDieNum = 0;
    uint32_t enableDieId = 0;
    CHK_RET(CcuAlgTemplateBase::GetDieInfoFromChannelDescs(comm, rankIdToChannelDesc, myRank, enableDieNum, enableDieId));
    HCCL_INFO("enableDieNum = %llu, enableDieId = %llu",enableDieNum, enableDieId);
    if (enableDieNum < 1 || enableDieNum > CCU_DIE_NUM_MAX_2) { // 目前只支持1个或2个die
        HCCL_ERROR("[ProcessFlattenLink] get channelDescs fail");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::vector<std::vector<HcclChannelDesc>> channelsPerDie;
    channelsPerDie.resize(enableDieNum);
    std::map<u32, u32> rank2ChannelIdx;
    CHK_RET(ProcessMeshInfo(comm, subcommInfo, rank2ChannelIdx, myRank, channelsPerDie, enableDieNum, enableDieId, rankIdToChannelDesc));
    if (enableDieNum > 1) { // 通过端口数划分channel，适配跨框die0连die1的场景，避免建链失败
        CHK_RET(CcuAlgTemplateBase::ReverseChannelPerDieIfNeed(comm, myRank, channelsPerDie));// 通过端口数划分channel，适配跨框die0连die1的场景，避免建链失败
    }
    channels = channelsPerDie[0];
    return HcclResult::HCCL_SUCCESS;
#endif
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
        (topoInfo->serverNum == 1 && topoInfo->isDiffDeviceModule && GetExternalInputIntraRoceSwitch() > 0));
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

HcclResult GetProtocolByEngine(const OpParam& param, std::vector<CommProtocol> &protocols)
{
    protocols.clear();
#if CANN_VERSION_NUM >= 90000000
    switch (param.engine) {
        case CommEngine::COMM_ENGINE_AICPU:
        case CommEngine::COMM_ENGINE_AICPU_TS:
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_CTP);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_TP);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_PCIE);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBOE);
            break;
        case CommEngine::COMM_ENGINE_CCU:
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_CTP);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_TP);
            break;
        case CommEngine::COMM_ENGINE_AIV:
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UB_MEM);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_PCIE);
            break;
        case CommEngine::COMM_ENGINE_CPU:
            // level 1到level n-1使用UB协议，server内建联，最外层使用网卡建联
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_CTP);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_UBC_TP);
            protocols.push_back(CommProtocol::COMM_PROTOCOL_ROCE);
            break;
        case CommEngine::COMM_ENGINE_CPU_TS:
            protocols.push_back(CommProtocol::COMM_PROTOCOL_ROCE);
            break;
        default:
            HCCL_WARNING("[GetProtocolByEngine] Unknown engine[%d], set protocol to RESERVED",
                         static_cast<int>(param.engine));
            break;
    }
#else
    // 8.5.0 CANN 无 UBC_CTP/UB_MEM 等枚举值；此函数所在的 CalcChannelRequestXxx/CreateChannelRequestByRankId 通路
    // 仅 9.0.0 新路径使用，运行时已由算子入口 GetHcommVersion() < 90000000 分流到 HcclXxxInner，
    // 8.5.0 下不会真正走到。这里保留空桩让 libhccl.so 外部链接（hccl_test 等）能解析符号。
    (void)param;
#endif
    return HCCL_SUCCESS;
}

HcclResult CreateChannelFromLink(HcclComm comm, u32 myRank, u32 rank, uint32_t netLayer, u32 idx,
    const CommLink& link, const std::string& funcName, std::vector<HcclChannelDesc>& channels)
{
    (void) comm;
    HcclChannelDesc channelDesc;
    HcclChannelDescInit(&channelDesc, 1);
    channelDesc.remoteRank = rank;
    channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
    channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    HCCL_DEBUG("%s local device phyId: %u, remote device phyId: %u.",
                funcName.c_str(), channelDesc.localEndpoint.loc.device.devPhyId,
                channelDesc.remoteEndpoint.loc.device.devPhyId);
    HCCL_INFO("%s Add channel request between %zu and %zu, netLayerIdx %u, "
              "linkListIdx %u, protocol %zu",
              funcName.c_str(), myRank, channelDesc.remoteRank, netLayer, idx, channelDesc.remoteEndpoint.protocol);
    channelDesc.channelProtocol = link.linkAttr.linkProtocol;
    channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
    channels.push_back(channelDesc);
    return HCCL_SUCCESS;
}

HcclResult ProcessLinkForProtocol(HcclComm comm, const std::vector<CommProtocol>& expectedProtocols,
    const std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
    std::vector<HcclChannelDesc>& channels, bool& protocolFound, const std::string& funcName)
{
    protocolFound = false;
    for (auto expectedProtocol : expectedProtocols) {
        for (u32 idx = 0; idx < linkList.size(); idx++) {
            if (linkList[idx].linkAttr.linkProtocol == expectedProtocol) {
                CHK_RET(CreateChannelFromLink(comm, myRank, remoteRank, netLayer, idx, linkList[idx],
                    funcName, channels));
                protocolFound = true;
            }
        }
        if (protocolFound) {
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult GetRankFullMeshLayers(HcclComm comm, const std::vector<std::vector<u32>>& subcommInfo, std::vector<uint32_t> netLayersVector, u32 myRank, u32 &curNetLayer)
{
#ifndef AICPU_COMPILE
    for (auto netLayer : netLayersVector) {
        bool isStainPath =  true;
        HCCL_INFO("netlayer=%d",netLayer);
        CommLink *linkList = nullptr;
        u32 listSize = 0;
        for(u32 rank: subcommInfo[COMM_LEVEL0]){
            if (rank == myRank) {
                continue;
            }
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));
            HCCL_INFO("dstrank = %d,listsize = %d",rank,listSize);
            if (listSize == 0){
                isStainPath = false;
                break;
            }
        }
        if(isStainPath) {
            curNetLayer = netLayer;
            HCCL_INFO("curNetLayer=%d",curNetLayer);
            break;
        }
        CHK_PRT_RET((curNetLayer == 0)&& (netLayer != 0),
            HCCL_ERROR("[GetRankFullMeshLayers] Failed to get cur netlayer myRank=%u .", myRank), HcclResult::HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
#endif
}

HcclResult CalcChannelRequestMesh1D(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,	 
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)	 
{	 
 #ifndef AICPU_COMPILE	 
     (void) param;	 
     channels.clear();	 
     auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);	 
     CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),	 
                 HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),	 
                 HcclResult::HCCL_E_PARA);	 
    u32 myRank = topoInfo->userRank;	 
    std::vector<CommProtocol> expectedProtocols;	 
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));	 
    for (u32 rank: subcommInfo[COMM_LEVEL0]) {	 
        if (rank == topoInfo->userRank) {	 
            continue;	 
        }	 
        size_t channelCountBefore = channels.size();	 
        uint32_t *netLayers;	 
        uint32_t netLayerNum;	 
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));	 
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);	 
        for (auto netLayer : netLayersVector) { 
            CommLink *linkList = nullptr; 
            u32 listSize; 
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize)); 
            if (listSize == 0) { 
                continue; 
            } 
            std::vector<CommLink> links(linkList, linkList + listSize); 
            bool protocolFound = false; 
            CHK_RET(ProcessLinkForProtocol(comm, expectedProtocols, links, myRank, rank, netLayer, channels, protocolFound, 
                std::string("[CalcChannelRequestMesh1D]"))); 
            if (channels.size() > channelCountBefore) { 
                break; 
            } 
        } 
        CHK_PRT_RET(channels.size() == channelCountBefore, 
            HCCL_ERROR("[CalcChannelRequestMesh1D] Failed to create channel between myRank=%u and rank=%u, there is no link.", 
                myRank, rank), HcclResult::HCCL_E_INTERNAL); 
    }	 
#endif	
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DFullMesh(HcclComm comm, const OpParam& param, 
    const TopoInfoWithNetLayerDetails* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
    std::vector<HcclChannelDesc> &channels)
{
#ifndef AICPU_COMPILE
    channels.clear();
    (void) param;
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
                HcclResult::HCCL_E_PARA);
    std::vector<CommProtocol> expectedProtocols;
    u32 myRank = topoInfo->userRank;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    uint32_t *netLayers, netLayerNum;
    uint32_t curNetLayer = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
    CHK_RET(GetRankFullMeshLayers(comm, subcommInfo, netLayersVector, myRank, curNetLayer));

    for (u32 rank: subcommInfo[COMM_LEVEL0]) {
        if (rank == topoInfo->userRank) {
            continue;
        }
        CommLink *linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, curNetLayer, myRank, rank, &linkList, &listSize));
        CHK_PRT_RET((listSize == 0),
            HCCL_ERROR("[CalcChannelRequestMesh1D] These is no link between myRank=%u, dstRank=%u.", myRank,rank), HcclResult::HCCL_E_INTERNAL);
        std::vector<CommLink> links(linkList, linkList + listSize);
        bool protocolFound = false;
        CHK_RET(ProcessLinkForProtocol(comm, expectedProtocols, links, myRank, rank, curNetLayer, channels, protocolFound,
            std::string("[CalcChannelRequestMesh1D]")));

    }
    if (curNetLayer != 0) { // 通过端口数划分channel，适配跨框die0连die1的场景，避免建链失败
        HCCL_INFO("curNetLayer = %lld",curNetLayer);
        CHK_RET(ProcessFlattenLink(comm, myRank, subcommInfo, channels));
    }
    return HCCL_SUCCESS;
#endif
}

static HcclResult CheckNetLayerExists(HcclComm comm, u32 netLayer, const std::string &tag, bool linkRequired)
{
#ifndef AICPU_COMPILE
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
    bool netLayerValid = false;
    for (auto layer : netLayersVector) {
        if (layer == netLayer) {
            netLayerValid = true;
            break;
        }
    }
    if (!netLayerValid) {
        CHK_PRT_RET(linkRequired,
            HCCL_ERROR("[%s] netLayer[%u] does not exist in rankGraph.", tag.c_str(), netLayer),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_WARNING("[%s] netLayer[%u] does not exist in rankGraph, skip.", tag.c_str(), netLayer);
        return HCCL_SUCCESS;
    }
#else
    (void)comm;
    (void)netLayer;
    (void)tag;
    (void)linkRequired;
#endif
    return HCCL_SUCCESS;
}

static HcclResult CalcChannelRequestMesh1DByLevel(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels,
    u32 netLayer, const std::string &tag, bool linkRequired)
{
#ifndef AICPU_COMPILE
    channels.clear();
    
    CHK_RET(CheckNetLayerExists(comm, netLayer, tag, linkRequired));

    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()), HCCL_ERROR("[%s] Rank [%u] is not in commInfo.",
        tag.c_str(), topoInfo->userRank), HcclResult::HCCL_E_PARA);

    u32 myRank = topoInfo->userRank;
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    for (u32 rank: subcommInfo[COMM_LEVEL0]) {
        if (rank == topoInfo->userRank) {
            continue;
        }
        size_t channelCountBefore = channels.size();

        CommLink *linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));

        if (listSize == 0) {
            if (linkRequired) {
                HCCL_ERROR("[%s] No intra-frame link between myRank=%u and rank=%u.", tag.c_str(), myRank, rank);
                return HcclResult::HCCL_E_INTERNAL;
            }
            HCCL_WARNING("[%s] No inter-frame link between myRank=%u and rank=%u.", tag.c_str(), myRank, rank);
            continue;
        }

        std::vector<CommLink> links(linkList, linkList + listSize);
        bool protocolFound = false;
        CHK_RET(ProcessLinkForProtocol(comm, expectedProtocols, links, myRank, rank, netLayer, channels, protocolFound, tag));

        if (channels.size() == channelCountBefore) {
            if (linkRequired) {
                HCCL_ERROR("[%s] No matching protocol intra-frame link between myRank=%u and rank=%u.", tag.c_str(), myRank, rank);
                return HcclResult::HCCL_E_INTERNAL;
            }
            HCCL_WARNING("[%s] No matching protocol inter-frame link between myRank=%u and rank=%u.", tag.c_str(), myRank, rank);
        }
    }
#else
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    (void)netLayer;
    (void)tag;
    (void)linkRequired;
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DLevel0(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
    return CalcChannelRequestMesh1DByLevel(comm, param, topoInfo, subcommInfo, channels,
        0, "CalcChannelRequestMesh1DLevel0", true);
}

HcclResult CalcChannelRequestMesh1DLevel1(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
    return CalcChannelRequestMesh1DByLevel(comm, param, topoInfo, subcommInfo, channels,
        1, "CalcChannelRequestMesh1DLevel1", false);
}

HcclResult CalcChannelRequestMesh2D(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
#ifndef AICPU_COMPILE
    channels.clear();
    u32 myRank = topoInfo->userRank; // 全局rankId

    std::set<u32> connectRanks;
    if (subcommInfo.size() == 2) { // 2D Mesh
        CHK_RET(CalcMesh2DChannelConnect(myRank, subcommInfo, connectRanks));
    }
#if CANN_VERSION_NUM >= 90000000
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    if (param.engine == CommEngine::COMM_ENGINE_AIV) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
    }
#else
    // 8.5.0 CANN 无 UBC_CTP/UB_MEM 枚举值；CalcChannelRequestMesh2D 整体属 9.0.0 新特性，
    // 主源已由算子入口 GetHcommVersion() 守护避免运行时调用；8.5.0 下用 HCCS 协议占位仅为可编
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    (void)param;
#endif

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

HcclResult ProcessLinkForProtocolNhr(HcclComm comm, const std::vector<CommProtocol>& expectedProtocols,
    const std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
    std::vector<HcclChannelDesc>& channels, bool& protocolFound)
{
    return ProcessLinkForProtocol(comm, expectedProtocols, linkList, myRank, remoteRank,
        netLayer, channels, protocolFound, std::string("[CalcLevel1ChannelRequestNhr]"));
}

HcclResult CalcChannelRequestNhr(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels)
{
#ifndef AICPU_COMPILE
    (void) param;
    channels.clear();
    std::set<u32> connectRanks;
    u32 myRank = topoInfo->userRank;
    auto it = std::find(subcommInfo[0].begin(), subcommInfo[0].end(), myRank);
    CHK_PRT_RET((it == subcommInfo[0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", myRank),
                HcclResult::HCCL_E_PARA);

    u32 localRank = std::distance(subcommInfo[0].begin(), it);
    u32 localRankSize = subcommInfo[0].size();
    CHK_RET(CalcNHRChannelConnect(localRank, localRankSize, INVALID_VALUE_RANKID, connectRanks));

    // 根据engine获取期望的协议类型列表
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    for (u32 rankIdx: connectRanks) {
        size_t channelCountBefore = channels.size();
        uint32_t *netLayers;
        uint32_t netLayerNum;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);

        for (auto netLayer : netLayersVector) {
            if (netLayerNum > 1 && netLayer == 0) {
                continue; // 跨框场景，nhr算法只取layer1的的链路
            }
            CommLink *linkList = nullptr;
            u32 listSize;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, subcommInfo[0][rankIdx], &linkList, &listSize));

            if (listSize == 0) {
                continue;
            }

            std::vector<CommLink> links(linkList, linkList + listSize);
            bool protocolFound = false;
            CHK_RET(ProcessLinkForProtocolNhr(comm, expectedProtocols, links, myRank, subcommInfo[0][rankIdx], netLayer, channels, protocolFound));

            if (channels.size() > channelCountBefore) {
                break;
            }
        }

        CHK_PRT_RET(channels.size() == channelCountBefore,
            HCCL_ERROR("[CalcChannelRequestNhr] Failed to create channel between myRank=%u and rank=%u, there is no link.",
                myRank, subcommInfo[0][rankIdx]), HcclResult::HCCL_E_INTERNAL);
    }
#endif
    return HCCL_SUCCESS;
}

#if CANN_VERSION_NUM >= 90000000
static bool IsEndPointEqual(EndpointDesc &endPoint0, EndpointDesc &endPoint1)
{
    HCCL_INFO("endPoint0:phyId[%u], protocol[%u], addr.type[%u], addr.id[%u]",
            endPoint0.loc.device.devPhyId,
            endPoint0.protocol,
            endPoint0.commAddr.type,
            endPoint0.commAddr.id);
    HCCL_INFO("endPoint1:phyId[%u], protocol[%u], addr.type[%u], addr.id[%u]",
            endPoint1.loc.device.devPhyId,
            endPoint1.protocol,
            endPoint1.commAddr.type,
            endPoint1.commAddr.id);
    if (endPoint0.protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
        return (endPoint0.protocol == endPoint1.protocol) &&
            (endPoint0.commAddr.type == endPoint1.commAddr.type);
    } else {
        return (endPoint0.protocol == endPoint1.protocol) &&
            (endPoint0.commAddr.type == endPoint1.commAddr.type) &&
            (memcmp(endPoint0.commAddr.eid, endPoint1.commAddr.eid, sizeof(endPoint0.commAddr.eid)) == 0);
    }

}
#endif /* CANN_VERSION_NUM >= 90000000 */

HcclResult GetTopoTypeByLink(HcclComm comm, uint32_t netLayer, CommLink &link, CommTopo &topoType)
{
#if defined(AICPU_COMPILE) || CANN_VERSION_NUM < 90000000
    // 8.5.0 CANN 无 HcclRankGraphGetEndpointNum / GetEndpointDesc / GetTopoType 等 9.0.0-only API，
    // 且 CommAddr.eid 字段也不存在；整函数在 8.5.0 下不提供真实实现（上游在 9.0.0 新路径里调用，
    // 入口版本号守护后 8.5.0 永远走不到这里）。
    (void)comm; (void)netLayer; (void)link; (void)topoType;
    return HCCL_SUCCESS;
#else
    uint32_t* topoInstList = nullptr;
    uint32_t listSize;
    CHK_RET(HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInstList, &listSize));         // 获取当前rank的所有TopoInst
    HCCL_INFO("[%s][%u] listSize = %u", __func__, __LINE__, listSize);

    for (uint32_t topoInstIdx = 0; topoInstIdx < listSize; topoInstIdx++) { // 遍历topoInst
        uint32_t topoInstId = topoInstList[topoInstIdx];
        uint32_t endPointNum;
        CHK_RET(HcclRankGraphGetEndpointNum(comm, netLayer, topoInstId, &endPointNum));
        EndpointDesc *endPointDescs = (EndpointDesc*)malloc(endPointNum * sizeof(EndpointDesc));
        CHK_RET(HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &endPointNum, endPointDescs));
        CHK_RET(HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &topoType));
        HCCL_DEBUG("[%s]topoInstId=%u, endPointNum=%u, topoType=%u", __func__, topoInstId, endPointNum, topoType);
        for (uint32_t endPointIdx = 0; endPointIdx < endPointNum; endPointIdx++) {
            EndpointDesc endPoint = endPointDescs[endPointIdx];
            if (IsEndPointEqual(link.srcEndpointDesc, endPoint) == true) {  // 当前TopoInst和link的endPoint相同，说明link属于当前TopoInst
                return HCCL_SUCCESS;
            }
        }
        HCCL_WARNING("[%s]No Endpoint matches on TopoInst[%u].", __func__, topoInstId);
    }
    HCCL_ERROR("[%s]Cannot get TopoType by Link.", __func__);
    return HCCL_E_INTERNAL;
#endif
}

/*
*   获取link对应的channel。对于2个rank之间，存在多条link的场景，会优先获取指定TopoType的1条channel。
*   如果多条link都没有指定的TopoType，则返回第一条link对应的channel。
*/
HcclResult ProcessLinksForChannel(HcclComm comm, u32 myRank, u32 rank, std::vector<HcclChannelDesc> &channels, CommTopo priorityTopo)
{
#ifndef AICPU_COMPILE
    uint32_t *netLayers;
    uint32_t netLayerNum;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector(netLayers, netLayers + netLayerNum);
    for (auto netLayer : netLayersVector) {
        CommLink *linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, rank, &linkList, &listSize));
        HCCL_INFO("[CalcChannelRequestWithPriorTopo] netLayer=%u, linkListSize=%u", netLayer, listSize);

        if (listSize == 0) {
            HCCL_WARNING("[CalcChannelRequestWithPriorTopo]There is no link between rank[%u] and rank[%u].", myRank,
                         rank);
            break;
        }

        uint32_t priorityLink = 0;
        CommTopo topoType;
        for (u32 idx = 0; idx < listSize; idx++) {
            CHK_RET(GetTopoTypeByLink(comm, netLayer, linkList[idx], topoType));
            if (topoType == priorityTopo) {
                priorityLink = idx;
                HCCL_INFO("[CalcChannelRequestWithPriorTopo] Found link[%u] with priority topotype[%u].", idx,
                          topoType);
                break;
            }
        }
        HcclChannelDesc channelDesc;
        HcclChannelDescInit(&channelDesc, 1);
        channelDesc.remoteRank = rank;
        CommLink link = linkList[priorityLink];
        channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
        channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        CHK_RET(GetTopoTypeByLink(comm, netLayer, linkList[priorityLink], topoType));
        HCCL_INFO("[CalcChannelRequestWithPriorTopo]Add channel request between %u and %u with protocol %u "
                  "and topoType %u. And Priority topoType is %u.",
                  myRank, channelDesc.remoteRank, channelDesc.remoteEndpoint.protocol, topoType, priorityTopo);
        channelDesc.channelProtocol = link.srcEndpointDesc.protocol;
        channelDesc.notifyNum = NORMAL_NOTIFY_NUM;
        channels.push_back(channelDesc);
        if (listSize > 0) {
            break;
        }
    }

#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DWithPriorityTopo(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels, CommTopo priorityTopo)
{
#ifndef AICPU_COMPILE
    (void) param;
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
                HcclResult::HCCL_E_PARA);

    u32 myRank = topoInfo->userRank;
    for (u32 rank : subcommInfo[COMM_LEVEL0]) {
        if (rank != myRank) {
            CHK_RET(ProcessLinksForChannel(comm, myRank, rank, channels, priorityTopo));
        }
    }
    HCCL_INFO("[%s] success.", __func__);
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNHRWithPriorityTopo(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc> &channels, CommTopo priorityTopo)
{
#ifndef AICPU_COMPILE
    (void) param;
    channels.clear();
    auto it = std::find(subcommInfo[COMM_LEVEL0].begin(), subcommInfo[COMM_LEVEL0].end(), topoInfo->userRank);
    CHK_PRT_RET((it == subcommInfo[COMM_LEVEL0].end()),
                HCCL_ERROR("[CollAlgFactory] [channel] Rank [%d] is not in commInfo.", topoInfo->userRank),
                HcclResult::HCCL_E_PARA);

    std::set<u32> connectRanks;
    u32 localRank = std::distance(subcommInfo[0].begin(), it);
    u32 localRankSize = subcommInfo[0].size();
    u32 myRank = topoInfo->userRank;
    CHK_RET(CalcNHRChannelConnect(localRank, localRankSize, INVALID_VALUE_RANKID, connectRanks));

    for (u32 rank : connectRanks) {
        if (rank != localRank) {
            CHK_RET(ProcessLinksForChannel(comm, myRank, subcommInfo[0][rank], channels, priorityTopo));
        }
    }
    HCCL_INFO("[%s] success.", __func__);
#endif
    return HCCL_SUCCESS;
}

HcclResult CreateChannelRequestByRankId(HcclComm comm, const OpParam& param, u32 myRank, u32 remoteRank,
    std::vector<HcclChannelDesc> &channels, u32 channelRepeatNum)
{
#ifndef AICPU_COMPILE
    channels.clear();
    std::vector<CommProtocol> expectedProtocols;
    CHK_RET(GetProtocolByEngine(param, expectedProtocols));

    uint32_t *netLayers;
    uint32_t netLayerNum;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<uint32_t> netLayersVector = std::vector<uint32_t>(netLayers, netLayers + netLayerNum);

    for (auto netLayer : netLayersVector) {
        CommLink *linkList = nullptr;
        u32 listSize;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize));
        if (listSize == 0) {
            continue;
        }
        std::vector<CommLink> links(linkList, linkList + listSize);
        bool protocolFound = false;
        for (u32 i = 0; i < channelRepeatNum; i++) {
            CHK_RET(ProcessLinkForProtocol(comm, expectedProtocols, links, myRank, remoteRank, netLayer, channels, protocolFound,
                std::string("[CreateChannelRequestByRankId]")));
        }

        if (channels.size() > 0) {
            break;
        }
    }
    CHK_PRT_RET(channels.size() == 0,
        HCCL_ERROR("[CreateChannelRequestByRankId] Failed to create channel between myRank=%u and rank=%u, there is no link.",
            myRank, remoteRank), HcclResult::HCCL_E_INTERNAL);

#endif
    return HCCL_SUCCESS;
}
}