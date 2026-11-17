/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_stub.h"

#include <algorithm>

thread_local MockRankGraph g_rankGraph;

// GetLayers返回的列表需在调用期间保持有效，用thread_local静态存储承接
static thread_local std::vector<uint32_t> g_layers;
// GetTopoInstsByLayer返回的topoInst列表同理
static thread_local std::vector<uint32_t> g_topoInsts = {0};
// GetEndpointDesc返回的endpoint：全零与link默认构造的srcEndpointDesc相等，
// 使GetTopoTypeByLink内IsEndPointEqual命中、真实分支正常返回
static thread_local EndpointDesc g_endpoints[1] = {};

extern "C" {
HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum)
{
    (void)comm;
    g_layers = g_rankGraph.netLayers;
    *netLayers = g_layers.data();
    *netLayerNum = static_cast<uint32_t>(g_layers.size());
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetLinks(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)
{
    (void)comm;
    (void)srcRank;
    g_rankGraph.getLinksCalls++;
    auto it = g_rankGraph.links.find({netLayer, dstRank});
    if (it == g_rankGraph.links.end()) {
        *links = nullptr;
        *linkNum = 0;
        return HCCL_SUCCESS;
    }
    *links = it->second.data();
    *linkNum = static_cast<uint32_t>(it->second.size());
    return HCCL_SUCCESS;
}

HcclResult
HcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    (void)comm;
    (void)netLayer;
    g_topoInsts = g_rankGraph.topoInsts;
    *topoInsts = g_topoInsts.data();
    *topoInstNum = static_cast<uint32_t>(g_topoInsts.size());
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num)
{
    (void)comm;
    (void)layer;
    auto it = g_rankGraph.endpointNums.find(topoInstId);
    *num = it == g_rankGraph.endpointNums.end() ? 1 : it->second;
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetEndpointDesc(
    HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    (void)comm;
    (void)layer;
    auto it = g_rankGraph.endpointActualNums.find(topoInstId);
    const uint32_t actualNum = it == g_rankGraph.endpointActualNums.end() ? *descNum : it->second;
    // actualNum可超过请求容量以模拟HCOMM异常回写；但桩不得越界写目标buffer
    const uint32_t writableNum = std::min(actualNum, *descNum);
    for (uint32_t idx = 0; idx < writableNum; idx++) {
        endpointDesc[idx] = g_endpoints[0];
    }
    *descNum = actualNum;
    return HCCL_SUCCESS;
}

HcclResult HcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType)
{
    (void)comm;
    (void)netLayer;
    (void)topoInstId;
    *topoType = g_rankGraph.topoType;
    return HCCL_SUCCESS;
}
}
