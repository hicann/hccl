/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_CHANNEL
#define OPS_HCCL_SRC_OPS_CHANNEL

#include "alg_param.h"

namespace ops_hccl {

enum CommPlane { COMM_LEVEL0 = 0, COMM_LEVEL1, COMM_LEVEL2, COMM_LEVEL_RESERVED };
constexpr u32 NORMAL_NOTIFY_NUM = 3;

HcclResult CalcLevel0ChannelRequest(
    const OpParam& param, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo, const AlgType& algType,
    std::vector<HcclChannelDesc>& channels);
HcclResult CalcLevel1ChannelRequest(
    const OpParam& param, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo, const AlgType& algType,
    std::vector<HcclChannelDesc>& channels);
HcclResult CalcLevel2ChannelRequest(
    const OpParam& param, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo, const AlgType& algType,
    std::vector<HcclChannelDesc>& channels);
HcclResult CalcChannelRequestMesh1D(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels);
HcclResult CalcChannelRequestMesh1DFullMesh(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels);
HcclResult CalcChannelRequestMesh1DLevel0(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels);
HcclResult CalcChannelRequestMesh1DLevel1(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels);
HcclResult CalcChannelRequestNhr(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels);
HcclResult CalcChannelRequestMesh2D(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels);
HcclResult CreateChannelRequestByRankId(
    HcclComm comm, const OpParam& param, u32 myRank, u32 remoteRank, std::vector<HcclChannelDesc>& channels,
    u32 channelRepeatNum = 1);
HcclResult CalcChannelRequestMesh1DWithPriorityTopo(
    HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
    std::vector<HcclChannelDesc>& channels, CommTopo priorityTopo);
HcclResult CalcChannelRequestNHRWithPriorityTopo(
    HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
    std::vector<HcclChannelDesc>& channels, CommTopo priorityTopo);
HcclResult GetTopoTypeByLink(HcclComm comm, uint32_t netLayer, CommLink& link, CommTopo& topoType);
HcclResult ProcessLinksForChannel(
    HcclComm comm, u32 myRank, u32 rank, std::vector<HcclChannelDesc>& channels, CommTopo priorityTopo,
    u32 topoLevelNums);
HcclResult ProcessLinksForChannelMutiJetty(
    HcclComm comm, CommProtocol& expectedProtocol, std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank,
    uint32_t netLayer, std::vector<HcclChannelDesc>& channels, bool execptMesh, bool isIsolation = false);
HcclResult GetProtocolByEngine(const OpParam& param, std::vector<CommProtocol>& protocols);
HcclResult ProcessMeshInfo(
    const HcclComm comm, const std::vector<std::vector<u32>>& subcommInfo, std::map<u32, u32>& rank2ChannelIdx,
    u32 myRank, std::vector<std::vector<HcclChannelDesc>>& channelsPerDie, u32 enableDieNum, u32 enableDieId,
    std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc);
HcclResult ProcessFlattenLink(
    HcclComm comm, u32 myRank, const std::vector<std::vector<u32>>& subcommInfo,
    std::vector<HcclChannelDesc>& channels);
HcclResult GetRankFullMeshLayers(
    HcclComm comm, const std::vector<std::vector<u32>>& subcommInfo, std::vector<uint32_t> netLayersVector, u32 myRank,
    u32& curNetLayer);
HcclResult CalcChannelRequestNhrMultiJetty(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels, bool isIsolation = false);
HcclResult CalcChannelRequestMeshClosMultiJetty(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels, bool isIsolation = false,
    bool execptMesh = true);
} // namespace ops_hccl

#endif
