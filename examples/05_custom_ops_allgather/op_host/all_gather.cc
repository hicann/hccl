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
#include <hccl_rank_graph.h>

using namespace ops_hccl_allgather;

constexpr uint32_t AIV_TAG_ADDR_OFFSET = 16 * 1024;

static HcclResult InitAivBuffer(HcclComm comm, const char* aivTag, void*& aivCommInfoPtr, HcclMemHandle& memHandle) {
    uint64_t aivCommInfoSize = AIV_TAG_BUFF_LEN;
    auto hcclRet = HcclEngineCtxGet(comm, aivTag, CommEngine::COMM_ENGINE_AIV, &aivCommInfoPtr, &aivCommInfoSize);
    
    if (hcclRet == HCCL_SUCCESS && aivCommInfoPtr != nullptr) {
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
    
    return HCCL_SUCCESS;
}

static HcclResult SetupRemoteChannels(HcclComm comm, uint32_t rank, uint32_t rankSize, HcclMemHandle& memHandle, void** buffersIn, void** buffersOut) {
    std::vector<HcclChannelDesc> channelRequests;
    for (size_t remoteRank = 0; remoteRank < rankSize; remoteRank++) {
        if (remoteRank == rank) continue;
        uint32_t netLayer = 0, listSize = 0;
        CommLink *linkList = nullptr;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, rank, remoteRank, &linkList, &listSize));
        for (uint32_t idx = 0; idx < listSize; idx++) {
            HcclChannelDesc desc;
            HcclChannelDescInit(&desc, 1);
            desc.memHandles = &memHandle;
            desc.memHandleNum = 1;
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
        }
    }

    uint32_t validNum = channelRequests.size();
    std::vector<ChannelHandle> levelNChannels(validNum);
    if (validNum > 0) {
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, channelRequests.data(), validNum, levelNChannels.data()));
    }

    for (uint32_t idx = 0; idx < validNum; idx++) {
        uint32_t currentRank = channelRequests[idx].remoteRank;
        void* remoteBufferAddr = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, levelNChannels[idx], &remoteBufferAddr, &remoteBufferSize));
        buffersIn[currentRank] = remoteBufferAddr;

        uint32_t memNum = 0;
        CommMem* remoteMems = nullptr;
        char** memTags = nullptr;
        CHK_RET(HcclChannelGetRemoteMems(comm, levelNChannels[idx], &memNum, &remoteMems, &memTags));
        CHK_PRT_RET(memNum != 1, HCCL_ERROR("[%s] HcclChannelGetRemoteMems memNum not 1", __func__), HCCL_E_PARA);
        buffersOut[currentRank] = remoteMems[0].addr;
    }
    return HCCL_SUCCESS;
}

HcclResult PrepareResources(HcclComm comm, OpParam& param, aclrtStream stream) {
    std::string aivTagStr = std::string(param.tag) + "_AIV";
    
    void* aivCommInfoPtr = nullptr;
    HcclMemHandle memHandle;
    CHK_RET(InitAivBuffer(comm, aivTagStr.c_str(), aivCommInfoPtr, memHandle));

    uint32_t rank = 0, rankSize = 0;
    CHK_RET(HcclGetRankId(comm, &rank));
    CHK_RET(HcclGetRankSize(comm, &rankSize));

    void* cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));

    void* buffersIn[MAX_RANK_SIZE] = {};
    void* buffersOut[MAX_RANK_SIZE] = {};
    buffersIn[rank] = cclBufferAddr;
    buffersOut[rank] = aivCommInfoPtr;

    CHK_RET(SetupRemoteChannels(comm, rank, rankSize, memHandle, buffersIn, buffersOut));

    param.buffIn = (uint64_t)aivCommInfoPtr;
    ACLCHECK(aclrtMemcpy(aivCommInfoPtr, MAX_RANK_SIZE * sizeof(void*), buffersIn, 
                         MAX_RANK_SIZE * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtMemcpy(static_cast<uint8_t*>(aivCommInfoPtr) + AIV_TAG_ADDR_OFFSET, MAX_RANK_SIZE * sizeof(void*),
                         buffersOut, MAX_RANK_SIZE * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE));

    HCCL_INFO("[%s] Alloc res success.", __func__);
    return HCCL_SUCCESS;
}

extern "C" HcclResult HcclAllGatherCustom(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream) {
    HCCL_INFO("[HcclAllGatherCustom] Entry. sendCount=%lu sendBuf=%p recvBuf=%p", sendCount, sendBuf, recvBuf);
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    int ret = sprintf_s(param.tag, sizeof(param.tag), "AllGather_%s_Custom", commName);
    if (ret <= 0) return HCCL_E_INTERNAL;
    
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
    param.len = sendCount; 
    param.dataType = dataType;
    param.reduceOp = 0; 
    param.root = 0; 
    param.tagId = 1; 
    
    param.inputSliceStride = sendCount;
    param.outputSliceStride = sendCount;
    
    param.repeatNum = 1;
    param.inputRepeatStride = 0;
    param.outputRepeatStride = 0;
    param.isOpBase = true;
    
    param.headCountMem = 0;
    param.tailCountMem = 0;
    param.addOneMem = 0;
    param.counterMemSize = 0;
    param.isEnableCounter = false;

    HCCL_INFO("[HcclAllGatherCustom] Launching kernel... rank=%u rankSize=%u", rank, rankSize);
    CHK_RET(LaunchKernel(param, stream));
    HCCL_INFO("[HcclAllGatherCustom] Launch success");
    
    return HCCL_SUCCESS;
}