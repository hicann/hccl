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
#include "hcomm_primitives.h"
#include "hccl_res.h"
#include "hcomm_primitives.h"
#include "hccl.h"
#include "adapter_acl.h"
#include "channel.h"
#include "hccl_common.h"

constexpr u32 FACTOR_NUM_TWO = 2;
constexpr s32 DEVICE_PER_MODULE = 8;
constexpr uint32_t NET_LAYER_NUM_TWO = 2;
constexpr uint32_t NET_LAYER_NUM_THREE = 3;

namespace ops_hccl {

HcclResult InitRankInfo(HcclComm comm, TopoInfo* topoInfo)
{
#ifndef AICPU_COMPILE
    // 提取本rank的信息
    CHK_RET(CalcMyRankInfo(comm, topoInfo));
    // 提取服务器层级的信息，比如服务器个数、每服务器卡数、服务器层拓扑是否对称
    std::unordered_map<u32, u32> pairLinkCounter;
    CHK_RET(GetPairLinkCounter(comm, topoInfo, pairLinkCounter));
    CHK_RET(SetServerModuleInfo(comm, topoInfo, pairLinkCounter));
    topoInfo->multiSuperPodDiffServerNumMode = false;
    if (topoInfo->deviceType == DevType::DEV_TYPE_910_93) {
        // 提取超节点层级的信息，比如超节点个数、每个超节点的服务器个数、
        // 超节点层拓扑是否对称
        CHK_RET(SetSuperPodInfo(comm, topoInfo));
        // 获取本服务器内的链路信息
        CHK_RET(CalcLinkInfo(topoInfo, pairLinkCounter));
    }
#endif
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
HcclResult CalcMyRankInfo(HcclComm comm, TopoInfo* topoInfo)
{
    CHK_RET(HcclGetRankSize(comm, &(topoInfo->userRankSize)));
    CHK_RET(HcclGetRankId(comm, &(topoInfo->userRank)));
    CHK_RET(hrtGetDeviceType(topoInfo->deviceType));
    uint32_t *netlayers = nullptr;
    uint32_t netLayersNum = 0;
    CHK_RET(HcclGetNetLayers(comm, &netlayers, &netLayersNum));

    // 获取moduleIdx
    CHK_RET(CalcGroupIdx(comm, topoInfo, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0)));
    // 获取superPodIdx
    if (netLayersNum >= NET_LAYER_NUM_TWO) {
        CHK_RET(CalcGroupIdx(comm, topoInfo, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L1)));
    } else {
        topoInfo->superPodIdx = 0;
    }
    HCCL_DEBUG("[CalcMyRankInfo]userRank[%u], userRankSize[%u], deviceType[%d], netLayersNum[%u], moduleIdx[%u] and superPodIdx[%u]",
        topoInfo->userRank, topoInfo->userRankSize, topoInfo->deviceType, netLayersNum, topoInfo->moduleIdx, topoInfo->superPodIdx);
    return HCCL_SUCCESS;
}
#endif
#ifndef AICPU_COMPILE
HcclResult SetServerModuleInfo(HcclComm comm, TopoInfo* topoInfo, const std::unordered_map<u32, u32> &pairLinkCounter)
{
    topoInfo->isDiffDeviceModule = IsDiffDeviceModule(topoInfo, pairLinkCounter);
    CHK_RET(GetModuleIdx(comm, topoInfo));
    HCCL_DEBUG("[SetServerModuleInfo]isDiffDeviceModule[%u], moduleIdx[%u]", topoInfo->isDiffDeviceModule, topoInfo->moduleIdx);
    // 910B A+X场景下RankGraph在初始化时已经通过GetModuleIdx刷新moduleIdx与serverIdx关系, 新接口应当不感知
    std::map<u32, std::vector<u32>> moduleMap;
    CHK_RET(GetModuleMap(comm, topoInfo, moduleMap));
    CHK_RET(GetDeviceNumPerModule(comm, topoInfo, moduleMap));
    topoInfo->moduleNum = moduleMap.size();

    topoInfo->multiModuleDiffDeviceNumMode = false;
    for (u32 i = 0; i < moduleMap.size(); ++i) {
        if (moduleMap[i].size() != topoInfo->deviceNumPerModule) {
            topoInfo->multiModuleDiffDeviceNumMode = true;
        }
        HCCL_INFO("module[%u] contains [%d]devices", i, moduleMap[i].size());
    }

    HCCL_RUN_INFO("different module contains different numbers of cards:[%d]",
        topoInfo->multiModuleDiffDeviceNumMode);
    return HCCL_SUCCESS;
}
#endif

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

#ifndef AICPU_COMPILE
/* 超节点数目以及超节点间server数解析 */
HcclResult SetSuperPodInfo(HcclComm comm, TopoInfo* topoInfo)
{
    topoInfo->multiSuperPodDiffServerNumMode = false;

    uint32_t level0RankListNum = 0;
    uint32_t level1RankListNum = 0;
    uint32_t *level0SizeList = nullptr;
    uint32_t *level1SizeList = nullptr; // 每个超节点里的rankSize {8, 8}
    std::vector<uint32_t> superPodToServerNum;
    uint32_t *netlayers = nullptr;
    uint32_t netLayersNum = 0;
    CHK_RET(HcclGetNetLayers(comm, &netlayers, &netLayersNum));
    if (netLayersNum == NET_LAYER_NUM_THREE) {
        CHK_RET(HcclGetInstSizeListByNetLayer(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0),
            &level0SizeList, &level0RankListNum));
        for (uint32_t i = 0; i < level0RankListNum; i++) {
            HCCL_DEBUG("[SetSuperPodInfo]netLayer[%u] level0RankListNum[%u] level0SizeList[%u]=[%u]",
                static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), level0RankListNum, i, level0SizeList[i]);
        }
        CHK_RET(HcclGetInstSizeListByNetLayer(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L1),
            &level1SizeList, &level1RankListNum));
        for (uint32_t i = 0; i < level1RankListNum; i++) {
            HCCL_DEBUG("[SetSuperPodInfo]netLayer[%u] level1RankListNum[%u] level1SizeList[%u]=[%u]",
                static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L1), level1RankListNum, i, level1SizeList[i]);
        }
        topoInfo->superPodNum = level1RankListNum;
        // 根据level0SizeList, level2SizeList, 重新构造superPodToServerNum
        std::vector<uint32_t> level0SizeListVector(level0SizeList, level0SizeList + level0RankListNum);
        std::vector<uint32_t> level1SizeListVector(level1SizeList, level1SizeList + level1RankListNum);
        CHK_RET(CalculateServersPerSuperPod(level0SizeListVector, level1SizeListVector, superPodToServerNum));
        for (uint32_t i = 0; i < superPodToServerNum.size(); i++) {
            HCCL_DEBUG("[SetSuperPodInfo]superpod[%u]: severNum[%u]", i, superPodToServerNum[i]);
        }
        topoInfo->serverNumPerSuperPod = superPodToServerNum[topoInfo->superPodIdx];
        HCCL_DEBUG("level0RankListNum[%u], level1RankListNum[%u], set superPodNum[%u], serverNumPerSuperPod[%u]",
            level0RankListNum, level1RankListNum, topoInfo->superPodNum, topoInfo->serverNumPerSuperPod);
    } else if (netLayersNum == NET_LAYER_NUM_TWO) {
        CHK_RET(HcclGetInstSizeListByNetLayer(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0),
            &level0SizeList, &level0RankListNum));
        for (uint32_t i = 0; i < level0RankListNum; i++) {
            HCCL_DEBUG("[SetSuperPodInfo]netLayer[%u] level0RankListNum[%u] level0SizeList[%u]=[%u]",
                static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), level0RankListNum, i, level0SizeList[i]);
        }
        topoInfo->superPodNum = 1;
        topoInfo->serverNumPerSuperPod = level0RankListNum;
        HCCL_DEBUG("[SetSuperPodInfo]level0RankListNum[%u], set superPodNum[%u], serverNumPerSuperPod[%u]",
            level0RankListNum, topoInfo->superPodNum, topoInfo->serverNumPerSuperPod);
        return HCCL_SUCCESS;
    } else {
        topoInfo->superPodNum = 1;
        topoInfo->serverNumPerSuperPod = 1;
        HCCL_DEBUG("[SetSuperPodInfo]level0RankListNum[%u], set superPodNum[%u], serverNumPerSuperPod[%u]",
            level0RankListNum, topoInfo->superPodNum, topoInfo->serverNumPerSuperPod);
        return HCCL_SUCCESS;
    }
    // 根据superPodToServerNum判断多个超节点内的sever数是否一致
    for (size_t i = 1; i < superPodToServerNum.size(); ++i) {
        if (superPodToServerNum[i] != superPodToServerNum[0]) {
            topoInfo->multiSuperPodDiffServerNumMode = true;
        }
    }
    HCCL_RUN_INFO("[Set][SuperPodInfo]different surperPod contains different numbers of servers:[%d]",
                    topoInfo->multiSuperPodDiffServerNumMode);

    // 跨超Server数非对称场景走NHR-HCF算法，该不存在server数不一致场景
    if (!topoInfo->multiModuleDiffDeviceNumMode && topoInfo->multiSuperPodDiffServerNumMode) {
        topoInfo->serverNumPerSuperPod = CalGCD(superPodToServerNum);
        topoInfo->multiSuperPodDiffServerNumMode = false;
        topoInfo->superPodNum = topoInfo->serverNum / topoInfo->serverNumPerSuperPod;
        HCCL_RUN_INFO("[CalcGeneralTopoInfoForA3] gcdServerNumPerSuperPod[%u] original superPodNum[%u] "
            "converted superPodNum[%u]", topoInfo->serverNumPerSuperPod, level1RankListNum, topoInfo->superPodNum);
    }

    return HCCL_SUCCESS;
}
#endif
#ifndef AICPU_COMPILE
/* 用于标识集群中是否存在A2 A+X形态 */
bool IsDiffDeviceModule(TopoInfo* topoInfo, const std::unordered_map<u32, u32> &pairLinkCounter)
{
    bool isDiffMeshAggregation = false;
    if (topoInfo->deviceType != DevType::DEV_TYPE_910B || topoInfo->userRankSize == 0) {
        HCCL_INFO("[IsDiffDeviceModule] deviceType[%d], topoInfo->userRankSize[%u]", topoInfo->deviceType, topoInfo->userRankSize);
        return false;
    }
    // 统计除HCCS外的所有非HCCS通信协议链路的总数
    // 如果总计数大于零，则表示拓扑中存在不同的设备模块（将isDiffMeshAggregation标志设置为true）
    u32 count = 0;
    u32 excludedKey = static_cast<u32>(CommProtocol::COMM_PROTOCOL_HCCS);
    for (const auto& pair : pairLinkCounter) {
        if (pair.first != excludedKey) {
            count += pair.second;
            HCCL_INFO("[IsDiffDeviceModule] Found key[%u]-value[%u] pair", pair.first, pair.second);
        }
    }
    if (count != 0) {
        isDiffMeshAggregation = true;
    }
    return isDiffMeshAggregation;
}
#endif
#ifndef AICPU_COMPILE
HcclResult CalcLinkInfo(TopoInfo* topoInfo, const std::unordered_map<u32, u32> &pairLinkCounter)
{
    // 解析得到各类算法需要的信息
    u32 hccsSWNum = 0;
    auto it = pairLinkCounter.find(static_cast<u32>(CommProtocol::COMM_PROTOCOL_HCCS));
    if (it != pairLinkCounter.end()) {
        hccsSWNum = it->second;
    }

    u32 sioNum = 0;
    it = pairLinkCounter.find(static_cast<u32>(CommProtocol::COMM_PROTOCOL_SIO));
    if (it != pairLinkCounter.end()) {
        sioNum = it->second;
    }
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
#endif

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
    HCCL_INFO("[CalcGeneralTopoInfoForA3] userRank[%u] serverIdx[%u] superPodIdx[%u] l0Rank[%u] l1Rank[%u] l2Rank[%u] "
        "deviceNumPerModule[%u] serverNumPerSuperPod[%u] superPodNum[%u]"
        "l0RankSize[%u] l1RankSize[%u] l2RankSize[%u]",
        topoInfo->userRank, topoInfo->serverIdx, topoInfo->superPodIdx,
        algHierarchyInfo.infos[COMM_LEVEL0].localRank, algHierarchyInfo.infos[COMM_LEVEL1].localRank,
        algHierarchyInfo.infos[COMM_LEVEL2].localRank,
        topoInfo->deviceNumPerModule, topoInfo->serverNumPerSuperPod, topoInfo->superPodNum,
        algHierarchyInfo.infos[COMM_LEVEL0].localRankSize, algHierarchyInfo.infos[COMM_LEVEL1].localRankSize,
        algHierarchyInfo.infos[COMM_LEVEL2].localRankSize);

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

#ifndef AICPU_COMPILE
HcclResult CalcGroupIdx(HcclComm comm, TopoInfo* topoInfo, uint32_t netLayer)
{
    uint32_t rankListNum;
    uint32_t *rankSizeList;
    CHK_RET(HcclGetInstSizeListByNetLayer(comm, netLayer, &rankSizeList, &rankListNum));
    for (uint32_t i = 0; i < rankListNum; i++) {
        HCCL_DEBUG("[CalcGroupIdx]netLayer[%u] rankListNum[%u] rankSizeList[%u]=[%u]",
            netLayer, rankListNum, i, rankSizeList[i]);
    }
    uint32_t currentGroup = 0;
    uint32_t cumulativeRank = 0;

    for (uint32_t i = 0; i < rankListNum; ++i) {
        cumulativeRank += rankSizeList[i];
        if (topoInfo->userRank < cumulativeRank) {
            currentGroup = i;
            break;
        }
    }
    HCCL_DEBUG("[CalcGroupIdx]currentGroup[%u]", currentGroup);
    if (netLayer == static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0)) {
        topoInfo->serverIdx = currentGroup;
        topoInfo->serverNum = rankListNum;
        HCCL_DEBUG("[CalcGroupIdx]netLayer[%u] currentGroup[%u] serverIdx[%u] serverNum[%u]",
            netLayer, currentGroup, topoInfo->serverIdx, topoInfo->serverNum);
    } else if (netLayer == static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L1)) {
        topoInfo->superPodIdx = currentGroup;
        HCCL_DEBUG("[CalcGroupIdx]netLayer[%u] currentGroup[%u] superPodIdx[%u]", netLayer, currentGroup, topoInfo->superPodIdx);
    } else {
        HCCL_ERROR("[CalcGroupIdx]netLayer[%u] is not supported", netLayer);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
#endif
#ifndef AICPU_COMPILE
HcclResult GetPairLinkCounter(HcclComm comm, TopoInfo* topoInfo, std::unordered_map<u32, u32> &pairLinkCounter)
{
    // 需要当前sever里的pairLinkCounter
    pairLinkCounter[static_cast<u32>(CommProtocol::COMM_PROTOCOL_HCCS)] = 0;
    pairLinkCounter[static_cast<u32>(CommProtocol::COMM_PROTOCOL_TCP)] = 0;
    pairLinkCounter[static_cast<u32>(CommProtocol::COMM_PROTOCOL_ROCE)] = 0;
    pairLinkCounter[static_cast<u32>(CommProtocol::COMM_PROTOCOL_UB_CTP)] = 0;
    pairLinkCounter[static_cast<u32>(CommProtocol::COMM_PROTOCOL_UB_TP)] = 0;
    pairLinkCounter[static_cast<u32>(CommProtocol::COMM_PROTOCOL_PCIE)] = 0;
    pairLinkCounter[static_cast<u32>(CommProtocol::COMM_PROTOCOL_SIO)] = 0;

    // 首先获取当前rank所在服务器的信息，确定服务器的起始和结束rank
    uint32_t currentServerStartRank = GetCurrentServerStartRank(comm, topoInfo);
    uint32_t currentServerEndRank = GetCurrentServerEndRank(comm, topoInfo);

    for (u32 srcRank = currentServerStartRank; srcRank < currentServerEndRank; ++srcRank) {
        for (u32 dstRank = currentServerStartRank; dstRank < currentServerEndRank; ++dstRank) {
            if (srcRank == dstRank) {
                continue;
            }
            CommLink *linkList = nullptr; // 必须初始化为nullptr
            uint32_t listSize = 0;
            HCCL_DEBUG("[GetPairLinkCounter]Getting links between srcRank[%u] and dstRank[%u]", srcRank, dstRank);
            CHK_RET(HcclGetLinks(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0),
                srcRank, dstRank, &linkList, &listSize));
            // 如果listSize为0，表示这两个rank之间没有直接link，直接进入下一轮循环
            if (listSize == 0) {
                HCCL_DEBUG("[GetPairLinkCounter]No links found between srcRank[%u] and dstRank[%u]", srcRank, dstRank);
                continue;
            }
            // =======================================================
            // 关键部分：遍历获取到的 linkList
            // =======================================================
            for (uint32_t i = 0; i < listSize; ++i) {
                CommLink& currentLink = linkList[i]; // 获取当前循环到的链路对象

                // --- 在这里处理 currentLink ---
                HCCL_DEBUG("  Link[%u] found between srcRank[%u] and dstRank[%u]:", i, srcRank, dstRank);
                HCCL_DEBUG("    LinkType: %u", currentLink.protocol);     // 假设有 linkType 成员
                HCCL_DEBUG("    srcEndPoint: %u", currentLink.srcEndPoint); // 假设有此成员
                HCCL_DEBUG("    dstEndPoint: %u", currentLink.dstEndPoint); // 假设有此成员

                // 可以将链路类型统计起来
                // 原始代码中的 pairLinkCounter 应该在这里使用
                pairLinkCounter[static_cast<u32>(currentLink.protocol)]++;
            }
        }
    }
    for (auto it : pairLinkCounter) {
        HCCL_DEBUG("[GetPairLinkCounter] pair link counter information linkType[%u], size[%u]", it.first, it.second);
    }
    return HCCL_SUCCESS;
}
#endif
#ifndef AICPU_COMPILE
// 获取当前服务器的startRank
uint32_t GetCurrentServerStartRank(HcclComm comm, TopoInfo* topoInfo)
{
    uint32_t rankListNum = 0;
    uint32_t *rankSizeList = nullptr;
    
    // 获取L0层级（服务器级别）的实例大小列表
    CHK_RET(HcclGetInstSizeListByNetLayer(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), &rankSizeList, &rankListNum));
    
    // 确定当前rank属于哪个服务器
    uint32_t currentServerStartRank = 0;
    for (u32 i = 0; i < topoInfo->serverIdx; ++i) {
        currentServerStartRank += rankSizeList[i];
    }
    return currentServerStartRank;
}
#endif
#ifndef AICPU_COMPILE
// 获取当前服务器的EndRank
uint32_t GetCurrentServerEndRank(HcclComm comm, TopoInfo* topoInfo)
{
    uint32_t rankListNum = 0;
    uint32_t *rankSizeList = nullptr;
    
    // 获取L0层级（服务器级别）的实例大小列表
    CHK_RET(HcclGetInstSizeListByNetLayer(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), &rankSizeList, &rankListNum));
    
    // 确定当前rank属于哪个服务器
    uint32_t currentServerStartRank = 0;
    for (u32 i = 0; i < topoInfo->serverIdx; ++i) {
        currentServerStartRank += rankSizeList[i];
    }
    uint32_t currentServerCount = rankSizeList[topoInfo->serverIdx];
    uint32_t currentServerEndRank = currentServerStartRank + currentServerCount;
    return currentServerEndRank;
}
#endif
#ifndef AICPU_COMPILE
HcclResult GetDeviceNumPerModule(HcclComm comm, TopoInfo* topoInfo, std::map<u32, std::vector<u32>> &moduleMap)
{
    if (topoInfo->deviceType == DevType::DEV_TYPE_910B && topoInfo->isDiffDeviceModule) {
        // 根据生成好的moduleMap计算当前rank所在module的设备数
        uint32_t srcRank = topoInfo->userRank;
        uint32_t moduleIdx = 0;
        HcclResult ret = GetModuleIdxByRank(comm, srcRank, topoInfo, moduleIdx);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[GetDeviceNumPerModule]Failed to get moduleIdx for rank[%u]", srcRank);
            return ret;
        }

        // 从moduleMap中获取当前rank所在module的设备数量
        auto it = moduleMap.find(moduleIdx);
        if (it != moduleMap.end()) {
            topoInfo->deviceNumPerModule = static_cast<u32>(it->second.size());
        } else {
            HCCL_ERROR("[GetDeviceNumPerModule]ModuleIdx[%u] not found in moduleMap", moduleIdx);
            return HCCL_E_PARA;
        }
    } else {
        uint32_t rankNum = 0;
        CHK_RET(HcclGetInstSizeByNetLayer(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), &rankNum));
        topoInfo->deviceNumPerModule = rankNum;
    }
    return HCCL_SUCCESS;
}
#endif
#ifndef AICPU_COMPILE
HcclResult GetModuleMap(HcclComm comm, TopoInfo* topoInfo, std::map<u32, std::vector<u32>> &moduleMap)
{
    // 遍历每一个rank使用GetModuleIdxByRank获取每一个rank的moduleIdx
    for (u32 rank = 0; rank < topoInfo->userRankSize; ++rank) {
        u32 moduleIdx = 0;
        HcclResult ret = GetModuleIdxByRank(comm, rank, topoInfo, moduleIdx);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[GetModuleMap]Failed to get moduleIdx for rank[%u]", rank);
            return ret;
        }
        moduleMap[moduleIdx].push_back(rank);
    }

    // 打印moduleMap
    HCCL_DEBUG("[GetModuleMap] ModuleMap:");
    for (const auto& pair : moduleMap) {
        std::string ranksStr = "{";
        for (size_t i = 0; i < pair.second.size(); ++i) {
            if (i > 0) {
                ranksStr += ", ";
            }
            ranksStr += std::to_string(pair.second[i]);
        }
        ranksStr += "}";
        HCCL_DEBUG("[GetModuleMap]  ModuleIdx[%u]: %s", pair.first, ranksStr.c_str());
    }

    return HCCL_SUCCESS;
}
#endif
#ifndef AICPU_COMPILE
HcclResult GetModuleIdx(HcclComm comm, TopoInfo* topoInfo)
{
    uint32_t moduleIdx = 0;
    HcclResult ret = GetModuleIdxByRank(comm, topoInfo->userRank, topoInfo, moduleIdx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[GetModuleIdx]Failed to get moduleIdx for rank[%u]", topoInfo->userRank);
        return ret;
    }
    topoInfo->moduleIdx = moduleIdx;
    return HCCL_SUCCESS;
}
#endif
#ifndef AICPU_COMPILE
HcclResult GetModuleIdxByRank(HcclComm comm, uint32_t rank, TopoInfo* topoInfo, uint32_t &moduleIdx)
{
    uint32_t rankServerIdx = 0;
    uint32_t accumulatedRanks = 0;
    uint32_t rankListNum = 0;
    uint32_t *rankSizeList = nullptr;

    CHK_RET(HcclGetInstSizeListByNetLayer(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0), &rankSizeList, &rankListNum));

    for (u32 i = 0; i < rankListNum; ++i) {
        if (rank < accumulatedRanks + rankSizeList[i]) {
            rankServerIdx = i;
            break;
        }
        accumulatedRanks += rankSizeList[i];
    }
    if (topoInfo->deviceType == DevType::DEV_TYPE_910B && topoInfo->isDiffDeviceModule) {
        // 计算给定rank所在server的起始rank
        // 这里需要根据给定的rank确定其对应的server索引

        uint32_t dstRank = accumulatedRanks; // 目标server的起始rank
        uint32_t srcRank = rank; // 源rank
        CommLink *linkList = nullptr; // 必须初始化为nullptr
        uint32_t listSize = 0;
        uint32_t rankModuleIdx = 1;
        if (srcRank != dstRank) {
            HCCL_DEBUG("[GetModuleIdxByRank]Getting links between srcRank[%u] and dstRank[%u]", srcRank, dstRank);
            CHK_RET(HcclGetLinks(comm, static_cast<uint32_t>(HcclNetLayer::HCCL_NetLayer_L0),
                srcRank, dstRank, &linkList, &listSize));
            for (uint32_t i = 0; i < listSize; ++i) {
                CommLink& currentLink = linkList[i]; // 获取当前循环到的链路对象

                HCCL_DEBUG("  Link[%u] found between srcRank[%u] and dstRank[%u]:", i, srcRank, dstRank);
                HCCL_DEBUG("    LinkType: %u", currentLink.protocol);
                HCCL_DEBUG("    srcEndPoint: %u", currentLink.srcEndPoint);
                HCCL_DEBUG("    dstEndPoint: %u", currentLink.dstEndPoint);

                if (currentLink.protocol == CommProtocol::COMM_PROTOCOL_HCCS) {
                    rankModuleIdx = 0;
                    break;
                }
            }
        } else {
            rankModuleIdx = 0;
        }
        moduleIdx = rankServerIdx * FACTOR_NUM_TWO + rankModuleIdx;
    } else {
        // 对于非 910B 或者非不同设备模块的情况，moduleIdx 等于 serverIdx
        // 需要确定给定rank对应的serverIdx
        moduleIdx = rankServerIdx;
    }
    return HCCL_SUCCESS;
}
#endif
#ifndef AICPU_COMPILE
HcclResult CalculateServersPerSuperPod(const std::vector<uint32_t> &l0Sizes,
                                       const std::vector<uint32_t> &l1Sizes,
                                       std::vector<uint32_t> &serversPerSuperPod)
{
    if (l0Sizes.empty() || l1Sizes.empty()) {
        HCCL_ERROR("[CalculateServersPerSuperPod]l0Sizes.size[%u], l1Sizes.size[%u]", l0Sizes.size(), l1Sizes.size());
        return HCCL_E_PARA;
    }

    // L0层(服务器)的总rank数应该等于L1层(超节点)的总rank数
    uint32_t totalL0Ranks = 0;
    for (uint32_t size : l0Sizes) {
        totalL0Ranks += size;
    }

    uint32_t totalL1Ranks = 0;
    for (uint32_t size : l1Sizes) {
        totalL1Ranks += size;
    }

    if (totalL0Ranks != totalL1Ranks) {
        // 这表明拓扑数据不一致
        HCCL_ERROR("[CalculateServersPerSuperPod]totalL0Ranks[%u] != totalL1Ranks[%u]", totalL0Ranks, totalL1Ranks);
        return HCCL_E_PARA;
    }

    // 通过将L0组映射到L1组来计算每个超节点的服务器数，使用累计和方法
    uint32_t l0Index = 0;
    uint32_t cumulativeL0Ranks = 0; // 累计已处理的L0 ranks数量

    for (uint32_t i = 0; i < l1Sizes.size(); ++i) {
        uint32_t targetCumulative = cumulativeL0Ranks + l1Sizes[i]; // 当前L1组所需的累计rank数
        uint32_t serversInCurrentSuperPod = 0;

        // 遍历L0组，直到累计的ranks数达到目标
        while (cumulativeL0Ranks < targetCumulative && l0Index < l0Sizes.size()) {
            // 如果当前L0组的ranks数不超过目标，直接添加整个L0组
            if (cumulativeL0Ranks + l0Sizes[l0Index] <= targetCumulative) {
                cumulativeL0Ranks += l0Sizes[l0Index];
                serversInCurrentSuperPod++; // 使用了一个完整的L0组（一个服务器）
                l0Index++;
            } else {
                HCCL_WARNING("[CalculateServersPerSuperPod]cumulativeL0Ranks:[%u] + l0Sizes[%u]:[%u] > targetCumulative:[%u], "
                    "which is equal to cumulativeL0Ranks:[%u] + l1Sizes[%u]:[%u]",
                    cumulativeL0Ranks, l0Index, l0Sizes[l0Index], targetCumulative, cumulativeL0Ranks, i, l1Sizes[i]);
                // 当前L0组的部分ranks被用于当前L1组，仍计为使用了一个服务器
                // 这种情况下我们只使用了当前L0组的一部分来达到目标
                cumulativeL0Ranks = targetCumulative;
                serversInCurrentSuperPod++; // 计数增加
                l0Index++; // 移动到下一个L0组
                break; // 已达到目标
            }
        }
        serversPerSuperPod.push_back(serversInCurrentSuperPod);
    }
    return HCCL_SUCCESS;
}
#endif
}