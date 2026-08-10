/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_custom_allgather.h"
#include "launch_kernel.h"
#include "common.h"
#include <vector>
#include <cstring>
#include <string>
#include <map>
#include <hccl_rank_graph.h>

using namespace ops_hccl_allgather;

constexpr uint32_t AIV_TAG_ADDR_OFFSET = 16 * 1024;

static uint32_t GetDataTypeSize(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
            return 1;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_BFP16:
            return 2;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_FP32:
        case HCCL_DATA_TYPE_UINT32:
            return 4;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            return 8;
        case HCCL_DATA_TYPE_INT128:
            return 16;
        default:
            return 0;
    }
}

static HcclResult FillOpParam(OpParam& param, uint64_t dataSize, HcclDataType dataType)
{
    uint32_t dataTypeSize = GetDataTypeSize(dataType);
    CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("[FillOpParam] unsupported dataType[%d]", dataType), HCCL_E_PARA);
    param.len = dataSize / dataTypeSize;
    param.dataType = dataType;
    param.reduceOp = 0;
    param.root = 0;
    param.tagId = 1;
    param.inputSliceStride = dataSize;
    param.outputSliceStride = dataSize;
    param.repeatNum = 1;
    param.inputRepeatStride = 0;
    param.outputRepeatStride = 0;
    param.isOpBase = true;
    param.headCountMem = 0;
    param.tailCountMem = 0;
    param.addOneMem = 0;
    param.counterMemSize = 0;
    param.isEnableCounter = false;
    return HCCL_SUCCESS;
}

static std::map<std::string, HcclMemHandle> g_memHandleCache;

static HcclResult InitAivBuffer(HcclComm comm, const char* aivTag, void*& aivCommInfoPtr, HcclMemHandle& memHandle)
{
    uint64_t aivCommInfoSize = AIV_TAG_BUFF_LEN;
    auto hcclRet = HcclEngineCtxGet(comm, aivTag, CommEngine::COMM_ENGINE_AIV, &aivCommInfoPtr, &aivCommInfoSize);

    if (hcclRet == HCCL_SUCCESS && aivCommInfoPtr != nullptr) {
        if (g_memHandleCache.find(aivTag) == g_memHandleCache.end()) {
            HCCL_ERROR("[%s] aiv memHandle not found in cache", __func__);
            return HCCL_E_INTERNAL;
        }
        memHandle = g_memHandleCache[aivTag];
        return HCCL_SUCCESS;
    }

    hcclRet = HcclEngineCtxCreate(comm, aivTag, CommEngine::COMM_ENGINE_AIV, AIV_TAG_BUFF_LEN, &aivCommInfoPtr);
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to create AIV buffer. ret=%d", __func__, hcclRet);
        return hcclRet;
    }

    ACLCHECK(aclrtMemset(aivCommInfoPtr, AIV_TAG_BUFF_LEN, 0, AIV_TAG_BUFF_LEN));
    CommMem regMem{COMM_MEM_TYPE_DEVICE, aivCommInfoPtr, AIV_TAG_BUFF_LEN};
    hcclRet = HcclCommMemReg(comm, aivTag, &regMem, &memHandle);
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] Failed to register memory. ret=%d", __func__, hcclRet);
        return hcclRet;
    }
    g_memHandleCache[aivTag] = memHandle;

    return HCCL_SUCCESS;
}

static bool IsAivP2pProtocol(uint32_t protocol)
{
    HCCL_INFO("[%s] protocol[%u]", __func__, protocol);
    return protocol == COMM_PROTOCOL_UB_MEM || protocol == COMM_PROTOCOL_HCCS || protocol == COMM_PROTOCOL_HCCS_ONLY
           || protocol == COMM_PROTOCOL_SIO;
}

static HcclResult
BuildChannelRequests(HcclComm comm, uint32_t rank, uint32_t rankSize, std::vector<HcclChannelDesc>& channelRequests)
{
    for (size_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
        if (remoteRank == rank)
            continue;
        uint32_t* netLayers = nullptr;
        uint32_t netLayerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        bool foundChannel = false;
        for (uint32_t layerIdx = 0; layerIdx < netLayerNum && !foundChannel; layerIdx++) {
            uint32_t listSize = 0;
            CommLink* linkList = nullptr;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], rank, remoteRank, &linkList, &listSize));
            for (uint32_t idx = 0; idx < listSize && !foundChannel; idx++) {
                if (!IsAivP2pProtocol(linkList[idx].linkAttr.linkProtocol)) {
                    continue;
                }
                HcclChannelDesc desc;
                HcclChannelDescInit(&desc, 1);
                desc.remoteRank = remoteRank;
                desc.localEndpoint.protocol = linkList[idx].srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = linkList[idx].srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = linkList[idx].srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = linkList[idx].dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = linkList[idx].dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = linkList[idx].dstEndpointDesc.loc;
                desc.channelProtocol = linkList[idx].linkAttr.linkProtocol;
                desc.notifyNum = 3;
                channelRequests.push_back(desc);
                foundChannel = true;
            }
        }
    }

    HCCL_INFO("[%s] channelRequests size[%zu] rankSize[%u]", __func__, channelRequests.size(), rankSize);
    return HCCL_SUCCESS;
}

static HcclResult AcquireChannelsAndBuffers(
    HcclComm comm, HcclMemHandle& memHandle, std::vector<HcclChannelDesc>& channelRequests, uint64_t* buffersInOut)
{
    for (auto& channelDesc : channelRequests) {
        channelDesc.memHandles = &memHandle;
        channelDesc.memHandleNum = 1;
    }

    uint32_t validNum = channelRequests.size();
    std::vector<ChannelHandle> levelNChannels(validNum);
    if (validNum > 0) {
        CHK_RET(HcclChannelAcquire(
            comm, CommEngine::COMM_ENGINE_AIV, channelRequests.data(), validNum, levelNChannels.data()));
    }

    for (uint32_t idx = 0; idx < validNum; idx++) {
        uint32_t currentRank = channelRequests[idx].remoteRank;
        void* remoteBufferAddr = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
        buffersInOut[2 * currentRank] = (uint64_t)remoteBufferAddr;

        uint32_t memNum = 0;
        CommMem* remoteMems = nullptr;
        char** memTags = nullptr;
        CHK_RET(HcclChannelGetRemoteMems(comm, levelNChannels[idx], &memNum, &remoteMems, &memTags));
        CHK_PRT_RET(memNum == 0, HCCL_ERROR("[%s] HcclChannelGetRemoteMems memNum is 0", __func__), HCCL_E_PARA);
        buffersInOut[2 * currentRank + 1] = (uint64_t)remoteMems[memNum - 1].addr;
    }
    return HCCL_SUCCESS;
}

static HcclResult
SetupRemoteChannels(HcclComm comm, uint32_t rank, uint32_t rankSize, HcclMemHandle& memHandle, uint64_t* buffersInOut)
{
    std::vector<HcclChannelDesc> channelRequests;
    CHK_RET(BuildChannelRequests(comm, rank, rankSize, channelRequests));
    CHK_RET(AcquireChannelsAndBuffers(comm, memHandle, channelRequests, buffersInOut));
    return HCCL_SUCCESS;
}

HcclResult PrepareResources(HcclComm comm, OpParam& param, aclrtStream stream)
{
    void* aivCommInfoPtr = nullptr;
    HcclMemHandle memHandle;
    CHK_RET(InitAivBuffer(comm, param.tag, aivCommInfoPtr, memHandle));

    uint32_t rank = 0, rankSize = 0;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));

    void* cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));

    uint64_t buffersInOut[MAX_RANK_SIZE_A3 * 2] = {};
    buffersInOut[2 * rank] = (uint64_t)cclBufferAddr;
    buffersInOut[2 * rank + 1] = (uint64_t)aivCommInfoPtr;
    CHK_RET(SetupRemoteChannels(comm, rank, rankSize, memHandle, buffersInOut));

    const uint64_t buffersInOutBytes = static_cast<uint64_t>(rankSize) * 2 * sizeof(uint64_t);
    ACLCHECK(
        aclrtMemcpy(aivCommInfoPtr, buffersInOutBytes, buffersInOut, buffersInOutBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    param.buffIn = (uint64_t)aivCommInfoPtr;

    HCCL_INFO("[%s] Alloc res success.", __func__);
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclAllGatherCustom(
    void* sendBuf, void* recvBuf, uint64_t dataSize, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HCCL_INFO(
        "[HcclAllGatherCustom] Entry. dataSize=%lu sendBuf=%p recvBuf=%p dataType=%d", dataSize, sendBuf, recvBuf,
        dataType);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;

    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "%s_opbase", commName);
    if (ret <= 0)
        return HCCL_E_INTERNAL;

    CHK_RET(PrepareResources(comm, param, stream));

    uint32_t rank = 0;
    uint32_t rankSize = 0;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));

    param.buffIn = (uint64_t)param.buffIn;
    param.input = (uint64_t)sendBuf;
    param.output = (uint64_t)recvBuf;
    param.rank = rank;
    param.rankSize = rankSize;
    param.xRankSize = rankSize;
    param.yRankSize = 0;
    param.zRankSize = 0;
    CHK_RET(FillOpParam(param, dataSize, dataType));

    HCCL_INFO(
        "[HcclAllGatherCustom] bufferin=%llu commName=%s tag=%s", static_cast<uint64_t>(param.buffIn), commName,
        param.tag);

    HCCL_INFO("[HcclAllGatherCustom] Launching kernel... rank=%u rankSize=%u", rank, rankSize);
    CHK_RET(LaunchKernel(param, stream));
    HCCL_INFO("[HcclAllGatherCustom] Launch success");

    return HCCL_SUCCESS;
}
