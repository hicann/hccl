/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_model.h"
#include "rank_table.h"

namespace HcclSim {

HcclResult GenGraphRankInfos(TopoMeta topoMate, std::vector<GraphRankInfo> &rankGraphs)
{
    u32 superPodId = 0;
    u32 serverId = 0;
    u32 rankId = 0;
    u32 boxIpStart = 168430090;   // Server起始IP
    u32 devIpStart = 3232238090;  // 设备起始IP
    for (auto pod = 0; pod < topoMate.size(); pod++) {
        for (auto server = 0; server < topoMate[pod].size(); server++) {
            CommAddr hostIp;
            hostIp.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
            hostIp.id = htonl(boxIpStart++);
            for (auto device = 0; device < topoMate[pod][server].size(); device++) {
                auto phyDevices = topoMate[pod][server];
                GraphRankInfo rankGraph;
                rankGraph.rankId = rankId;
                rankGraph.serverIdx = serverId;
                rankGraph.serverId = std::to_string(serverId);
                rankGraph.superDeviceId = superPodId;
                rankGraph.superPodId = std::to_string(superPodId);
                rankGraph.hostIp = hostIp;

                CommAddr devIp;
                devIp.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
                devIp.id = htonl(devIpStart++);
                rankGraph.deviceInfo.deviceIp = devIp;
                rankGraph.deviceInfo.devicePhyId = phyDevices[device];
                rankGraphs.push_back(rankGraph);
            }
            serverId++;
        }
        superPodId++;
    }
    return HCCL_SUCCESS;
}

};