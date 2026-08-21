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
#include <iostream>
#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include "log.h"
#include "utils.h"
#include "common.h"

namespace ops_hccl_allgather {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

HcclResult GetDeviceType(DeviceType* deviceType)
{
    const char* socNamePtr = aclrtGetSocName();
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

HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc& desc)
{
    uint32_t netLayer = 0;
    uint32_t listSize = 0;
    CommLink* linkList = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));

    CHK_RET(HcclChannelDescInit(&desc, 1));
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
    bool protocolExists = false;
    for (uint32_t idx = 0; idx < listSize; idx++) {
        CommLink link = linkList[idx];
        if (link.linkAttr.linkProtocol == protocol) {
            desc.remoteRank = dstRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
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
        HCCL_ERROR("[FillChannelDesc] Protocol %d not found between rank %u and rank %u", protocol, srcRank, dstRank);
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclMemcpyCtxHostToDevice(
    HcclComm comm, const OpParam& param, AlgResourceCtx& resCtxHost, void** resCtxSequence, uint64_t* ctxSize)
{
    // 序列化
    std::vector<char> seq = resCtxHost.Serialize();
    uint64_t size = seq.size();

    void* ctx = nullptr;

    // 创建Context, aicpu和host dpu申请device内存
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, COMM_ENGINE_AICPU_TS, size, &ctx));
    // 从Host内存拷贝到Device Context内存上
    CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_AICPU_TS, param.tag, seq.data(), size, 0));
    // 将内存强转为AlgResourceCtx结构体
    *resCtxSequence = ctx;
    *ctxSize = size;
    HCCL_INFO("Memcpy hostCtx to device success.");
    return HCCL_SUCCESS;
}

HcclResult HcclGetThreadAICPU(HcclComm comm, const OpParam& param, AlgResourceCtx& resCtxHost)
{
    // Mesh算法所需资源
    uint32_t slaveThreadNum = resCtxHost.slaveThreadNum;
    uint32_t notifyNumOnMainThread = resCtxHost.notifyNumOnMainThread;
    uint32_t threadNum = slaveThreadNum + 1;
    uint32_t maxNotifyNum = notifyNumOnMainThread;
    std::vector<ThreadHandle> threads(threadNum);
    // maxNotifyNum需要再增加一个用于host-device同步
    CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, threadNum, maxNotifyNum + 1, threads.data()));
    HCCL_DEBUG("threads ptr is %p\n", threads.data());
    for (uint32_t i = 0; i < threadNum; i++) {
        resCtxHost.threads.push_back(threads[i]);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetChannelAICPU(HcclComm comm, const OpParam& param, AlgResourceCtx& resCtxHost)
{
    uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> descs(channelNum);
    std::vector<ChannelHandle> channels(channelNum);
    std::vector<uint32_t> remoteRanks(channelNum);

    uint32_t idx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, descs[idx]));
        remoteRanks[idx] = remoteRank;
        idx++;
    }

    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, descs.data(), channelNum, channels.data()));

    for (uint32_t i = 0; i < channelNum; i++) {
        ChannelInfo channel;
        channel.remoteRank = remoteRanks[i];
        channel.handle = channels[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        void* cclBuf;
        uint64_t cclBufSize;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channels[i], &cclBuf, &cclBufSize));
        channel.remoteCclMem = CommBuffer{cclBuf, cclBufSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclAllocAlgResourceAICPU(HcclComm comm, const OpParam& param, AlgResourceCtx& resCtxHost)
{
    void* cclBufferAddr;
    uint64_t cclBufferSize;
    // 从通信域获取CCL buffer
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtxHost.cclMem = CommBuffer{cclBufferAddr, cclBufferSize};
    uint32_t threadNum = param.rankSize > 1 ? param.rankSize - 1 : 1;
    resCtxHost.slaveThreadNum = threadNum - 1;
    resCtxHost.notifyNumOnMainThread = resCtxHost.slaveThreadNum;
    resCtxHost.notifyNumPerThread = std::vector<uint32_t>(resCtxHost.slaveThreadNum, 1);
    CHK_RET(HcclGetThreadAICPU(comm, param, resCtxHost));
    CHK_RET(HcclGetChannelAICPU(comm, param, resCtxHost));
    return HCCL_SUCCESS;
}
} // namespace ops_hccl_allgather
