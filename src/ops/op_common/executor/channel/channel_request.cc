/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include <vector>
#include <algorithm>
#include <hccl/hccl_types.h>
#include "channel_request.h"
#include "hccl/base.h"
#include "log.h"

namespace ops_hccl {

HcclResult CalcRingChannelConnect(u32 rank, u32 rankSize, u32 root, std::set<u32> &connectRanks)
{
    (void)root;
    connectRanks.clear();
    if (rankSize == HCCL_RANK_SIZE_EQ_ONE) { // 只有一张卡时不需要建链
        HCCL_INFO("[CalcRingChannelConnect] no need to create links, rankSize[%u].", rankSize);
        return HCCL_SUCCESS;
    }

    const u32 targetRankPos = static_cast<u32>(rank + 1) % rankSize;
    const u32 targetRankNeg = static_cast<u32>(rank + rankSize - 1) % rankSize;
    connectRanks.insert(targetRankPos);
    connectRanks.insert(targetRankNeg);
    HCCL_INFO("[CalcRingChannelConnect]localRank[%u], rankPos[%u], rankNeg[%u]", rank, targetRankPos, targetRankNeg);
    return HCCL_SUCCESS;
}

HcclResult CalcMeshChannelConnect(u32 rank, u32 rankSize, u32 root, std::set<u32> &connectRanks)
{
    (void)root;
    connectRanks.clear();
    if (rankSize == HCCL_RANK_SIZE_EQ_ONE) { // 只有一张卡时不需要建链
        HCCL_INFO("[CalcMeshChannelConnect] no need to create links, rankSize[%u].", rankSize);
        return HCCL_SUCCESS;
    }

    for (u32 dstRank = 0; dstRank < rankSize; dstRank++) {
        if (dstRank == rank) {
            continue;
        }
        connectRanks.insert(dstRank);
        HCCL_INFO("[CalcMeshChannelConnect]localRank[%u], rankDst[%u]", rank, dstRank);
    }
    return HCCL_SUCCESS;
}

HcclResult CalcNHRChannelConnect(u32 rank, u32 rankSize, u32 root, std::set<u32> &connectRanks)
{
    (void)root;
    connectRanks.clear();
    if (rankSize == HCCL_RANK_SIZE_EQ_ONE) { // 只有一张卡时不需要建链
        HCCL_INFO("[CalcNHRChannelConnect] no need to create links, rankSize[%u].", rankSize);
        return HCCL_SUCCESS;
    }

    for (u32 delta = 1; delta < rankSize; delta <<= 1) {
        const u32 targetRankPos = static_cast<u32>(rank + delta) % rankSize;
        const u32 targetRankNeg = static_cast<u32>(rank + rankSize - delta) % rankSize;
        connectRanks.insert(targetRankPos);
        connectRanks.insert(targetRankNeg);
        HCCL_INFO("[CalcNHRChannelConnect]localRank[%u], rankPos[%u], rankNeg[%u]", rank, targetRankPos, targetRankNeg);
    }
    return HCCL_SUCCESS;
}

HcclResult CalcNBChannelConnect(u32 rank, u32 rankSize, u32 root, std::set<u32> &connectRanks)
{
    (void)root;
    connectRanks.clear();
    if (rankSize == HCCL_RANK_SIZE_EQ_ONE) { // 只有一张卡时不需要建链
        HCCL_INFO("[CalcNBChannelConnect] no need to create links, rankSize[%u].", rankSize);
        return HCCL_SUCCESS;
    }

    for (u32 delta = 1; delta < rankSize; delta <<= 1) {
        const u32 targetRankPos = static_cast<u32>(rank + delta) % rankSize;
        const u32 targetRankNeg = static_cast<u32>(rank + rankSize - delta) % rankSize;
        connectRanks.insert(targetRankPos);
        connectRanks.insert(targetRankNeg);
        HCCL_INFO("[CalcNBChannelConnect]localRank[%u], rankPos[%u], rankNeg[%u]", rank, targetRankPos, targetRankNeg);
    }
    return HCCL_SUCCESS;
}

HcclResult GetAlgRankChannel(u32 virtRank, const std::vector<u32> &tempVTopo, u32 &algRank)
{
    std::vector<u32>::const_iterator topoVecIter = std::find(tempVTopo.begin(), tempVTopo.end(), virtRank);
    CHK_PRT_RET(topoVecIter == tempVTopo.end(), HCCL_ERROR("[GetAlgRankChannel] Invalid virtual Rank!"), HCCL_E_PARA);
    algRank = distance(tempVTopo.begin(), topoVecIter);
    return HCCL_SUCCESS;
}

HcclResult CalcMesh2DChannelConnect(u32 myRank, const std::vector<std::vector<u32>> &tempVTopo, std::set<u32> &connectRanks)
                            // const u32 linkNumBtwPeers, AlgTempResReq &tempResReq)
{
    connectRanks.clear();
    u32 myAlgRank;
    for (u32 dim = 0; dim < tempVTopo.size(); dim++) {
        CHK_RET(GetAlgRankChannel(myRank, tempVTopo[dim], myAlgRank));
        for (u32 queIdx = 0; queIdx < tempVTopo[dim].size() - 1; queIdx++) {
            u32 neighborAlgRank = (myAlgRank + 1 + queIdx) % (tempVTopo[dim].size());
            CHK_PRT_RET(neighborAlgRank > (tempVTopo[dim].size() - 1),
                HCCL_ERROR("[CalcResLinksMesh2D] neighborAlgRank[%u] is invalid, the Max rank[%u].",
                    neighborAlgRank, tempVTopo[dim].size() - 1), HCCL_E_INTERNAL);
            u32 neighborRank = tempVTopo[dim][neighborAlgRank];
            connectRanks.insert(neighborRank);
            HCCL_INFO("[CalcMesh2DChannelConnect]localRank[%u], rankDst[%u]", myRank, neighborRank);
            // tempResReq.links[neighborRank] = linkNumBtwPeers;
        }
    }

    return HCCL_SUCCESS;
}

}