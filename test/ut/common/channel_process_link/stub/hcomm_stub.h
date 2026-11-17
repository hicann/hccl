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

// ProcessLinksForChannel依赖的HCOMM rank graph接口打桩：可执行文件内同名强符号优先于动态库解析，
// 桩按预设的拓扑表返回可控数据，不访问真实rank graph。

#ifndef CHANNEL_PROCESS_LINK_HCOMM_STUB_H
#define CHANNEL_PROCESS_LINK_HCOMM_STUB_H

#include <map>
#include <utility>
#include <vector>

#include <hccl/hccl_types.h>
#include <hccl/hccl_rank_graph.h>

// 桩状态：描述一个可配置的"本rank视角"拓扑
struct MockRankGraph {
    // 返回给HcclRankGraphGetLayers的netLayer列表，如{0}或{0,1}
    std::vector<uint32_t> netLayers = {0};
    // 返回给HcclRankGraphGetTopoInstsByLayer的topoInst列表；GetTopoTypeByLink可直接配置多inst跳过行为
    std::vector<uint32_t> topoInsts = {0};
    // key为(netLayer, dstRank)，值为该层上myRank到dstRank的link列表；无条目即无link
    std::map<std::pair<uint32_t, uint32_t>, std::vector<CommLink>> links;
    // key为topoInstId；未配置时GetEndpointNum返回1，保持既有正常路径默认行为
    std::map<uint32_t, uint32_t> endpointNums;
    // key为topoInstId；未配置时GetEndpointDesc回写值等于请求值
    std::map<uint32_t, uint32_t> endpointActualNums;
    // GetTopoTypeByLink链路上的topoType返回值
    CommTopo topoType = CommTopo::COMM_TOPO_1DMESH;
    uint32_t getLinksCalls = 0;
};

extern thread_local MockRankGraph g_rankGraph;

#endif // CHANNEL_PROCESS_LINK_HCOMM_STUB_H
