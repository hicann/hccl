/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RANKGRAPH_H
#define HCCL_RANKGRAPH_H

#include "hccl_res.h"


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct GraphDeviceInfo {
    s32 devicePhyId;                     // 服务器内device唯一标识
    DevType deviceType { DevType::DEV_TYPE_NOSOC };                  // 服务器内device类型
    CommAddr deviceIp; // device 对应的网卡ip
    CommAddr backupDeviceIp; // 同一卡另一个device的网卡ip，应用于重执行借轨场景
    u32 port { HCCL_INVALID_PORT };
    u32 vnicPort { HCCL_INVALID_PORT };
    u32 backupPort { HCCL_INVALID_PORT };
};

struct GraphRankInfo {
    u32 rankId = 0xFFFFFFFF;            // rank 标识，userRank,cloud时hcom计算填入
    u32 localRank = 0xFFFFFFFF;         // 本server内rank号
    std::string serverId;               // 集群内服务器唯一标识
    u32 serverIdx = INVALID_UINT;       // Server在ranktable中的自然顺序（用户指定）
    u32 superDeviceId = INVALID_UINT;   // 超节点device id，超节点内唯一
    std::string superPodId;             // 超节点标识
    u32 superPodIdx = INVALID_UINT;     // SuperPod在ranktable中的自然顺序（用户指定）
    CommAddr hostIp;                    // 本server的host ip，用于host rdma通信
    u32 hostPort = HCCL_INVALID_PORT;   // 本rank进行host socket通信使用的端口
    u32 nodeId = INVALID_UINT;          // 离线编译逻辑ranktable 和NumaConfig中的node id相同
    s32 itemId = INVALID_UINT;          // 离线编译逻辑ranktable 和NumaConfig中的item id相同
    GraphDeviceInfo deviceInfo;         // 设备信息
    s32 bindDeviceId = INVALID_INT;     // 绑定的device id
    std::string originalSuperPodId;     // 划分逻辑超节点前的原超节点ID，来源为用户配置
};

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif
