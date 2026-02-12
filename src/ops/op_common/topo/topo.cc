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
#include "config_log.h"

constexpr u32 FACTOR_NUM_TWO = 2;
constexpr s32 DEVICE_PER_MODULE = 8;
constexpr uint32_t NET_LAYER_NUM_TWO = 2;
constexpr uint32_t NET_LAYER_NUM_THREE = 3;

namespace ops_hccl {

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

}