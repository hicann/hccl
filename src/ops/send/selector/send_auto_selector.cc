/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_auto_selector.h"
#include "selector_registry.h"
#include "topo_host.h"

namespace ops_hccl {
    SelectorStatus SendAutoSelector::SelectAicpuAlgo(
        const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
        std::string &selectAlgName) const
    {
        (void) topoInfo;
        HCCL_INFO("[SendAutoSelector][SelectAicpuAlgo] opType:%d", opParam.opType);

        selectAlgName = "InsSend";
        return SelectorStatus::MATCH;
    }

    SelectorStatus SendAutoSelector::SelectAivAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
    {
        (void) topoInfo;
        if (topoInfo->topoLevelNums == TOPO_LEVEL_NUM_3) {
            HCCL_AIV_NOT_MATCH_LOG(opParam, HCCL_DEBUG, "[SendAutoSelector][%s] aiv is not supported with level2Uboe, reset to default.",
                __func__);
            return SelectorStatus::NOT_MATCH;
        }

        HCCL_INFO("[SendAutoSelector][SelectAivAlgo] opType:%d", opParam.opType);
        selectAlgName = "AivSend";
        return SelectorStatus::MATCH;
    }

    SelectorStatus SendAutoSelector::SelectDPUAlgo(const TopoInfoWithNetLayerDetails *topoInfo, const OpParam &opParam,
        const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap, std::string &selectAlgName) const
    {
        HCCL_INFO("[SendAutoSelector][SelectDPUAlgo] opType:%d", opParam.opType);

        // 通过 topoInfo 中的 netLayers 获取链路信息，判断本端和对端的 locationType
        // host nic -- device nic 场景走 opv2_insSendHostDpu，其他走 InsSendDPU
        u32 myRank = topoInfo->userRank;
        u32 remoteRank = opParam.sendRecvRemoteRank;
        HcclComm comm = opParam.hcclComm;

        // 获取最高层（最后一个 netLayer）
        const std::vector<u32> &netLayers = topoInfo->netLayerDetails.netLayers;
        if (netLayers.empty()) {
            HCCL_WARNING("[SendAutoSelector][SelectDPUAlgo] netLayers is empty, use default InsSendDPU");
            selectAlgName = "InsSendDPU";
            return SelectorStatus::MATCH;
        }

        // 从最高层开始查找可用链路
        for (auto it = netLayers.rbegin(); it != netLayers.rend(); ++it) {
            u32 netLayer = *it;
            CommLink *linkList = nullptr;
            u32 listSize = 0;
            HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("[SendAutoSelector][SelectDPUAlgo] HcclRankGraphGetLinks failed, netLayer:%u", netLayer);
                continue;
            }
            if (listSize == 0 || linkList == nullptr) {
                continue;
            }

            // 获取第一条链路的 endpoint 信息
            EndpointDesc &srcEndpoint = linkList[0].srcEndpointDesc;
            EndpointDesc &dstEndpoint = linkList[0].dstEndpointDesc;
            EndpointLocType srcLocType = srcEndpoint.loc.locType;
            EndpointLocType dstLocType = dstEndpoint.loc.locType;

            HCCL_INFO("[SendAutoSelector][SelectDPUAlgo] myRank:%u, remoteRank:%u, netLayer:%u, "
                "srcLocType:%d, dstLocType:%d",
                myRank, remoteRank, netLayer, srcLocType, dstLocType);

            // host nic -- device nic: 本端是 HOST，对端是 DEVICE
            if (srcLocType == ENDPOINT_LOC_TYPE_HOST && dstLocType == ENDPOINT_LOC_TYPE_DEVICE) {
                selectAlgName = "opv2_insSendHostDpu";
            } else {
                selectAlgName = "InsSendDPU";
            }
            return SelectorStatus::MATCH;
        }

        // 没有找到链路，使用默认
        HCCL_WARNING("[SendAutoSelector][SelectDPUAlgo]no link found for rank:%u and rank:%u, use default InsSendDPU",
            myRank, remoteRank);
        selectAlgName = "InsSendDPU";
        return SelectorStatus::MATCH;
    }

    REGISTER_SELECTOR_BY_OPTYPE(HcclCMDType::HCCL_CMD_SEND, 18, SendAutoSelector);
} // namespace ops_hccl
