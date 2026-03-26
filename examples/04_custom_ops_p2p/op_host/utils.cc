/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include "log.h"
#include "utils.h"
#include "common.h"

namespace ops_hccl_p2p {
HcclResult GetDeviceType(DeviceType *deviceType) {
    const char *socNamePtr = aclrtGetSocName();
    if (socNamePtr == nullptr) {
        HCCL_ERROR("[GetDeviceType] Failed to get soc name");
        return HCCL_E_RUNTIME;
    }

    std::string socName(socNamePtr);
    if (socName.find("Ascend910B") != std::string::npos) {
        *deviceType = DEVICE_TYPE_A2;
        return HCCL_SUCCESS;
    }
    if (socName.find("Ascend910_93") != std::string::npos) {
        *deviceType = DEVICE_TYPE_A3;
        return HCCL_SUCCESS;
    }
    if (socName.find("Ascend950") != std::string::npos) {
        *deviceType = DEVICE_TYPE_A5;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[GetDeviceType] Unsupported soc name: %s", socName.c_str());
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AcquireChannel(HcclComm comm, CommEngine engine, DeviceType devType,
                          uint32_t srcRank, uint32_t dstRank, ChannelHandle *channel)
{
  if (devType == DEVICE_TYPE_A2 || devType == DEVICE_TYPE_A3) {
    // Atlas A2/A3 创建 Channel
    HcclChannelDesc desc;
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = dstRank;
    desc.channelProtocol = CommProtocol::COMM_PROTOCOL_HCCS;
    desc.notifyNum = 2;
    CHK_RET(HcclChannelAcquire(comm, engine, &desc, 1, channel));
    return HCCL_SUCCESS;
  }
  if (devType != DEVICE_TYPE_A5) {
    HCCL_ERROR("[AcquireChannel] Unsupported device type %d", devType);
    return HCCL_E_NOT_SUPPORT;
  }

  // Ascend 950 创建 Channel
  uint32_t netLayer = 0, listSize = 0;
  CommLink *linkList = nullptr;
  CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList,
                                &listSize));

  HcclChannelDesc desc;
  CHK_RET(HcclChannelDescInit(&desc, 1));
  CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
  bool protocolExists = false;
  for (uint32_t idx = 0; idx < listSize; idx++) {
    CommLink link = linkList[idx];
    if (link.linkAttr.linkProtocol == protocol) {
      desc.remoteRank = dstRank;
      desc.notifyNum = 2;
      desc.channelProtocol = link.linkAttr.linkProtocol;
      desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
      desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
      desc.localEndpoint.loc = link.srcEndpointDesc.loc;
      desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
      desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
      desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
      protocolExists = true;
      break;
    }
  }
  if (!protocolExists) {
    HCCL_ERROR(
        "[AcquireChannel] Protocol %d not found between rank %u and rank %u",
        protocol, srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
  }
  CHK_RET(HcclChannelAcquire(comm, engine, &desc, 1, channel));
  return HCCL_SUCCESS;
}
}
