/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo.h"
#include "hccl_rank_graph.h"
#include "hccl_rankgraph.h"
#include "hcomm_primitives.h"
#include "hccl_res.h"
#include "hcomm_primitives.h"
#include "hccl.h"
#include "adapter_acl.h"
#include "channel.h"
#include "hccl_common.h"

constexpr u32 FACTOR_NUM_TWO = 2;
constexpr s32 DEVICE_PER_MODULE = 8;

namespace ops_hccl {

HcclResult InitRankInfo(HcclComm comm, TopoInfo* topoInfo)
{
#ifndef AICPU_COMPILE
    void* temp = nullptr;
    struct GraphRankInfo* graph;
    u32 len = 0;
    // 获取rankGraph
    CHK_RET(HcclGetRankGraph(comm, GraphType::RANK_GRAPH_910_93, &temp, &len));
    graph = static_cast<struct GraphRankInfo*>(temp);
    // 将rankGraph转换为rankList
    std::vector<struct GraphRankInfo> rankList;
    u32 eleCnt = len / sizeof(struct GraphRankInfo);
    for (u32 index = 0; index < eleCnt; index++) {
        rankList.push_back(*(graph + index));
    }
    // 提取本rank的信息
    CHK_RET(CalcMyRankInfo(comm, rankList, topoInfo));
    // 提取服务器层级的信息，比如服务器个数、每服务器卡数、服务器层拓扑是否对称
    CHK_RET(SetServerModuleInfo(rankList, topoInfo));
    topoInfo->multiSuperPodDiffServerNumMode = false;
    if (topoInfo->deviceType == DevType::DEV_TYPE_910_93) {
        // 提取超节点层级的信息，比如超节点个数、每个超节点的服务器个数、
        // 超节点层拓扑是否对称
        CHK_RET(SetSuperPodInfo(rankList, topoInfo));
        // 获取本服务器内的链路信息
        CHK_RET(CalcLinkInfo(comm, rankList, topoInfo));
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult CalcMyRankInfo(HcclComm comm, const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo)
{
    CHK_RET(HcclGetRankSize(comm, &(topoInfo->userRankSize)));
    CHK_RET(HcclGetRankId(comm, &(topoInfo->userRank)));
    CHK_RET(hrtGetDeviceType(topoInfo->deviceType));

    for (u32 i = 0; i < rankList.size(); i++) {
        if (rankList[i].rankId == topoInfo->userRank) {
            topoInfo->serverIdx = rankList[i].serverIdx;
            topoInfo->superPodIdx = rankList[i].superPodIdx;
            topoInfo->devicePhyId = rankList[i].deviceInfo.devicePhyId;
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult SetServerModuleInfo(const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo)
{
    topoInfo->isDiffDeviceModule = IsDiffDeviceModule(rankList, topoInfo);
    HCCL_DEBUG("[SetServerModuleInfo]isDiffDeviceModule[%u]", topoInfo->isDiffDeviceModule);

    std::map<u32, std::vector<struct GraphRankInfo>> moduleMap;
    std::set<u32> serverIdxs;
    for (struct GraphRankInfo rankInfo : rankList) {
        if (static_cast<s32>(rankInfo.deviceInfo.devicePhyId) == HOST_DEVICE_ID) {
            continue;
        }
        serverIdxs.insert(rankInfo.serverIdx);

        u32 moduleIdx;
        CHK_RET(GetModuleIdx(rankInfo, topoInfo, moduleIdx));

        if (rankInfo.rankId == topoInfo->userRank) {
            topoInfo->moduleIdx = moduleIdx;
        }
        auto iter = moduleMap.find(moduleIdx);
        if (iter == moduleMap.end()) {
            std::vector<struct GraphRankInfo> rankInfoList;
            rankInfoList.push_back(rankInfo);
            moduleMap.insert(std::make_pair(moduleIdx, rankInfoList));
        } else {
            iter->second.push_back(rankInfo);
        }
    }
    if (moduleMap.size() == 0) {
        return HCCL_SUCCESS;
    }

    topoInfo->multiModuleDiffDeviceNumMode = false;
    topoInfo->serverNum = serverIdxs.size();
    topoInfo->moduleNum = moduleMap.size();
    u32 preDeviceNum = moduleMap.begin()->second.size();
    topoInfo->deviceNumPerModule = preDeviceNum;
    for (auto &moduleInfo : moduleMap) {
        u32 curDeviceNum = moduleInfo.second.size();
        if (curDeviceNum != preDeviceNum) {
            topoInfo->multiModuleDiffDeviceNumMode = true;
        }

        HCCL_INFO("module[%d] contains [%d]devices", moduleInfo.first, moduleInfo.second.size());
        for (auto &rankInfo : moduleInfo.second) {
            HCCL_INFO("moduleIdx[%d] Info: rankId[%d], serverId[%s], serverIdx[%d], devicePhyId[%d]",
                moduleInfo.first, rankInfo.rankId, rankInfo.serverId.c_str(), rankInfo.serverIdx,
                rankInfo.deviceInfo.devicePhyId);
        }
    }

    HCCL_RUN_INFO("different module contains different numbers of cards:[%d]",
        topoInfo->multiModuleDiffDeviceNumMode);
    return HCCL_SUCCESS;
}

u32 CalGCD(u32 a, u32 b)
{
    if (a == 0 || b == 0) {
        return 1;
    }

    u32 gcd = b;
    while (a % b != 0) {
        gcd = a % b;
        a = b;
        b = gcd;
    }
    HCCL_DEBUG("[CalGCD]a[%u] b[%u], gcd[%u]", a, b, gcd);
    return gcd;
}

u32 CalGCD(std::vector<u32> &nums)
{
    if (nums.size() == 0) {
        return 1;
    }
    std::sort(nums.begin(), nums.end(), [](const u32 &num1, const u32 &num2) {
        return num1 > num2;
    });

    u32 curGcd = nums[0];
    for (u32 i = 1; i < nums.size(); i++) {
        curGcd = CalGCD(curGcd, nums[i]);
    }
    HCCL_DEBUG("[CalGCD]size[%u], gcd[%u]", nums.size(), curGcd);
    return curGcd;
}

/* 超节点数目以及超节点间server数解析 */
HcclResult SetSuperPodInfo(const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo)
{
    topoInfo->superPodNum = 0;
    topoInfo->multiSuperPodDiffServerNumMode = false;

    std::map<std::string, std::set<u32>> superPodToServerNum; // 记录每个超节点中的server数目
    for (struct GraphRankInfo rankInfo : rankList) {
        // superPodId为空时, 返回超节点数量为0, 按照非超节点模式处理
        CHK_PRT_RET(rankInfo.superPodId.empty(),
                    HCCL_DEBUG("ranks[%u] superPodId[%s] is empty, set superPodNum to zero", rankInfo.rankId,
                                rankInfo.superPodId.c_str()),
                    HCCL_SUCCESS);

        superPodToServerNum[rankInfo.superPodId].insert(rankInfo.serverIdx);
    }
    topoInfo->superPodNum = superPodToServerNum.size();
    u32 preServerNum = superPodToServerNum.begin()->second.size();
    topoInfo->serverNumPerSuperPod = preServerNum;
    std::vector<u32> superPodServerNumVec;
    for (auto superPodItem : superPodToServerNum) {
        u32 curServerNum = superPodItem.second.size();
        if (curServerNum != preServerNum) {
            topoInfo->multiSuperPodDiffServerNumMode = true;
        }
        superPodServerNumVec.push_back(curServerNum);
        HCCL_INFO("[Set][SuperPodInfo]SuperPod[%s] contains [%d]servers", superPodItem.first.c_str(), superPodItem.second.size());
    }
    HCCL_RUN_INFO("[Set][SuperPodInfo]different surperPod contains different numbers of servers:[%d]",
                    topoInfo->multiSuperPodDiffServerNumMode);

    // 跨超Server数非对称场景走NHR-HCF算法，该不存在server数不一致场景
    if (!topoInfo->multiModuleDiffDeviceNumMode && topoInfo->multiSuperPodDiffServerNumMode) {
        topoInfo->serverNumPerSuperPod = CalGCD(superPodServerNumVec);
        topoInfo->multiSuperPodDiffServerNumMode = false;
        topoInfo->superPodNum = topoInfo->serverNum / topoInfo->serverNumPerSuperPod;
        HCCL_RUN_INFO("[CalcGeneralTopoInfoForA3] gcdServerNumPerSuperPod[%u] original superPodNum[%u] "
            "converted superPodNum[%u]", topoInfo->serverNumPerSuperPod, superPodToServerNum.size(), topoInfo->superPodNum);
    }

    return HCCL_SUCCESS;
}

/* 用于标识集群中是否存在A2 A+X形态 */
bool IsDiffDeviceModule(const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo)
{
    bool minDevice = false;
    bool maxDevice = false;
    bool isDiffMeshAggregation = false;
    if (topoInfo->deviceType != DevType::DEV_TYPE_910B || rankList.size() == 0) {
        HCCL_INFO("[IsDiffDeviceModule] deviceType[%d], rankList.size[%u]", topoInfo->deviceType, rankList.size());
        return false;
    }
    for (const struct GraphRankInfo &rankInfo : rankList) {
        if (rankInfo.deviceInfo.devicePhyId < DEVICE_PER_MODULE) {
            minDevice = true;
        } else {
            maxDevice = true;
        }
    }
    if (minDevice && maxDevice) {
        isDiffMeshAggregation = true;
    }
    return isDiffMeshAggregation;
}

HcclResult GetModuleIdx(const struct GraphRankInfo &rankInfo, TopoInfo* topoInfo, u32& moduleIdx)
{
    u32 serverIdx = rankInfo.serverIdx;
    if (topoInfo->deviceType == DevType::DEV_TYPE_910B && topoInfo->isDiffDeviceModule) {
        moduleIdx = serverIdx * FACTOR_NUM_TWO + rankInfo.deviceInfo.devicePhyId / DEVICE_PER_MODULE_A2;
    } else {
        moduleIdx = serverIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult CalcLinkInfo(HcclComm comm, const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo)
{
    (void) comm;
    std::vector<s32> devIdList;
    for (auto rankInfo : rankList) {
        if (rankInfo.serverIdx == topoInfo->serverIdx) {
            devIdList.push_back(rankInfo.deviceInfo.devicePhyId);
        }
    }

    std::unordered_map<u32, u32> pairLinkCounter;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::PXI_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] = 0;
    for (auto &localDevId : devIdList) {
        for (auto &destDevId : devIdList) {
            if (localDevId == destDevId || localDevId == HOST_DEVICE_ID || destDevId == HOST_DEVICE_ID) {
                continue;
            }
            LinkTypeInServer linkType;
            CHK_RET(haclrtGetPairDeviceLinkType(localDevId, destDevId, linkType));
            pairLinkCounter[static_cast<u32>(linkType)]++;
        }
    }
    for (auto it : pairLinkCounter) {
        HCCL_DEBUG("[CalcLinkInfo] pair link counter information linkType[%u], size[%u]", it.first, it.second);
    }

    // 解析得到各类算法需要的信息
    u32 hccsSWNum = pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)];
    u32 sioNum = pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)];
    HCCL_DEBUG("[CalcLinkInfo] hccsSWNum[%u], sioNum[%u], deviceNumPerModule[%u]", hccsSWNum, sioNum,
        topoInfo->deviceNumPerModule);
    if (hccsSWNum == 0 || sioNum == 0) {
        topoInfo->isHCCSSWNumEqualToTwiceSIONum = false;
    } else {
        // The following 2 means that the device has no HCCS_SW link with itself and its companion linked by same SIO link.
        topoInfo->isHCCSSWNumEqualToTwiceSIONum =
            (hccsSWNum == (topoInfo->deviceNumPerModule - 2) * topoInfo->deviceNumPerModule) &&
           (sioNum == topoInfo->deviceNumPerModule);
    }
    return HCCL_SUCCESS;
}

/* 针对A2对称拓扑通用的拓扑信息获取方式，支持A+X */
HcclResult CalcGeneralTopoInfoForA2(HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo)
{
    (void) comm;
    algHierarchyInfo.levels = 2;
    algHierarchyInfo.infos[COMM_LEVEL0].localRank = topoInfo->userRank % topoInfo->deviceNumPerModule;
    algHierarchyInfo.infos[COMM_LEVEL0].localRankSize = topoInfo->deviceNumPerModule;
    algHierarchyInfo.infos[COMM_LEVEL1].localRank = topoInfo->moduleIdx;
    algHierarchyInfo.infos[COMM_LEVEL1].localRankSize = topoInfo->moduleNum;
    HCCL_INFO("[CalcGeneralTopoInfoForA2] userRank[%u] serverIdx[%u] l0Rank[%u] l1Rank[%u]",
        topoInfo->userRank, topoInfo->serverIdx, algHierarchyInfo.infos[COMM_LEVEL0].localRank,
        algHierarchyInfo.infos[COMM_LEVEL1].localRank);
    return HCCL_SUCCESS;
}

/* 针对A3对称拓扑通用的拓扑信息获取方式 */
HcclResult CalcGeneralTopoInfoForA3(HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo)
{
    (void) comm;
    algHierarchyInfo.levels = 3;
    algHierarchyInfo.infos[COMM_LEVEL0].localRank = topoInfo->userRank % topoInfo->deviceNumPerModule;
    algHierarchyInfo.infos[COMM_LEVEL0].localRankSize = topoInfo->deviceNumPerModule;
    algHierarchyInfo.infos[COMM_LEVEL1].localRank = topoInfo->serverIdx % topoInfo->serverNumPerSuperPod;
    algHierarchyInfo.infos[COMM_LEVEL1].localRankSize = topoInfo->serverNumPerSuperPod;
    algHierarchyInfo.infos[COMM_LEVEL2].localRank = topoInfo->serverIdx / topoInfo->serverNumPerSuperPod;
    algHierarchyInfo.infos[COMM_LEVEL2].localRankSize = topoInfo->superPodNum;
    HCCL_INFO("[CalcGeneralTopoInfoForA3] userRank[%u] serverIdx[%u] superPodIdx[%u] l0Rank[%u] l1Rank[%u] l2Rank[%u]",
        topoInfo->userRank, topoInfo->serverIdx, topoInfo->superPodIdx,
        algHierarchyInfo.infos[COMM_LEVEL0].localRank, algHierarchyInfo.infos[COMM_LEVEL1].localRank,
        algHierarchyInfo.infos[COMM_LEVEL2].localRank);

    return HCCL_SUCCESS;
}

/* 针对非对称场景打平拓扑通用的拓扑信息获取方式 */
HcclResult CalcGeneralTopoInfoForComm(HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo)
{
    (void) comm;
    algHierarchyInfo.levels = 2;
    algHierarchyInfo.infos[COMM_LEVEL0].localRank = 0;
    algHierarchyInfo.infos[COMM_LEVEL0].localRankSize = 1;
    algHierarchyInfo.infos[COMM_LEVEL1].localRank = topoInfo->userRank;
    algHierarchyInfo.infos[COMM_LEVEL1].localRankSize = topoInfo->userRankSize;
    HCCL_INFO("[CalcGeneralTopoInfoForComm] userRank[%u] serverIdx[%u] l1Rank[%u]",
        topoInfo->userRank, topoInfo->serverIdx, algHierarchyInfo.infos[COMM_LEVEL1].localRank);
    return HCCL_SUCCESS;
}

/* 计算每个level内其他rank的全局rank号 */
HcclResult GetUserRankBySubCommRank(u32 subCommRank, u32 curLevel, AlgHierarchyInfo& algHierarchyInfo, u32 &userRank)
{
    userRank = 0;
    u32 preLevelsRankSize = 1;
    for (u32 level = 0; level < algHierarchyInfo.levels; level++) {
        if (level == curLevel) {
            userRank += subCommRank * preLevelsRankSize;
        } else {
            userRank += algHierarchyInfo.infos[level].localRank * preLevelsRankSize;
        }
        preLevelsRankSize *= algHierarchyInfo.infos[level].localRankSize;
    }
    HCCL_INFO("[GetUserRankBySubCommRank]subCommRank[%u] level[%u] -> userRank[%u]", subCommRank, curLevel, userRank);
    return HCCL_SUCCESS;
}

/* 根据全局rank号计算对应在某个level内的rank号 */
HcclResult GetSubCommRankByUserRank(u32 userRank, u32 curLevel, AlgHierarchyInfo& algHierarchyInfo, u32 &subCommRank)
{
    u32 preLevelsRankSize = 1;
    for (u32 level = 0; level < algHierarchyInfo.levels; level++) {
        if (level == curLevel) {
            subCommRank = userRank / preLevelsRankSize % algHierarchyInfo.infos[level].localRankSize;
            HCCL_INFO("[GetSubCommRankByUserRank]userRank[%u] level[%u] -> subCommRank[%u]", userRank, curLevel, subCommRank);
            return HCCL_SUCCESS;
        }
        preLevelsRankSize *= algHierarchyInfo.infos[level].localRankSize;
    }
    return HCCL_SUCCESS;
}
}