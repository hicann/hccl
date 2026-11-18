/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "alg_data_trans_wrapper.h"
#include "ins_v2_batch_send_recv_nda_executor.h"
#include "alg_attrs_registry.h"
#include "ins_temp_batch_send_recv_nda.h"
#include "op_common.h"
#include "hcomm_primitives_dl.h"
#include "hccl_common.h"

namespace ops_hccl {

InsV2BatchSendRecvNdaExecutor::InsV2BatchSendRecvNdaExecutor() {}

HcclResult InsV2BatchSendRecvNdaExecutor::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    (void)algAttrs;
    CHK_PTR_NULL(topoInfo);
    CHK_PRT_RET(
        topoInfo->userRankSize == 0,
        HCCL_ERROR("[InsV2BatchSendRecvNdaExecutor][CalcAlgHierarchyInfoV2] rankSize is 0"), HCCL_E_PARA);
    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].resize(1);
    algHierarchyInfo.infos[0][0].clear();
    for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
        algHierarchyInfo.infos[0][0].push_back(rankId);
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);
    myRank_ = topoInfo->userRank;
    itemNum_ = param.batchSendRecvDataDes.itemNum;
    itemPtr_ = param.batchSendRecvDataDes.sendRecvItemsPtr;

    // thread和notify资源
    // Keep the resource shape fixed for the communicator, including sparse/self-only calls.
    resourceRequest.slaveThreadNum = SEND_RECV_THREAD_NUM - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = resourceRequest.slaveThreadNum;

    // 将本次任务需要通信的remoteRank加入集合
    CHK_PTR_NULL(itemPtr_);
    commTargetUserRankSet_.clear();
    for (u32 i = 0; i < itemNum_; i++) {
        commTargetUserRankSet_.insert((itemPtr_ + i)->remoteRank);
        HCCL_DEBUG(
            "[InsV2BatchSendRecvNdaExecutor][CalcRes] insert remoteUserRank[%u] to Set ", (itemPtr_ + i)->remoteRank);
    }
    HCCL_DEBUG(
        "[InsV2BatchSendRecvNdaExecutor][CalcRes]commTargetUserRankSet_ size[%zu]", commTargetUserRankSet_.size());

    // 遍历remoteRank集合，如果在algHierarchyInfo中，则建链，否则报错
    // BSR has one logical level; HCOMM handles DEVICE and HOST endpoints in the same batch.
    std::vector<HcclChannelDesc> channelLevel0;
    for (const u32& remoteRank : commTargetUserRankSet_) {
        if (remoteRank == static_cast<uint32_t>(myRank_)) {
            continue;
        }
        auto it = std::find(algHierarchyInfo.infos[0][0].begin(), algHierarchyInfo.infos[0][0].end(), remoteRank);
        if (it == algHierarchyInfo.infos[0][0].end()) {
            HCCL_ERROR(
                "[InsV2BatchSendRecvNdaExecutor][CalcRes] task remoteRank[%u] has no direct link with "
                "myRank[%u]",
                remoteRank, myRank_);
            return HCCL_E_PARA;
        } else {
            std::vector<HcclChannelDesc> channelByRank;
            // 和同一个远端之间的send和recv在不同流上，所以需要申请2条channel
            CHK_RET(CalcChannelRequest(comm, param, remoteRank, channelByRank));
            channelLevel0.insert(channelLevel0.end(), channelByRank.begin(), channelByRank.end());
        }
    }
    resourceRequest.channels = {channelLevel0};

    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::CalcChannelRequest(
    HcclComm comm, const OpParam& param, u32 remoteRank, std::vector<HcclChannelDesc>& channels) const
{
#ifndef AICPU_COMPILE
    // Detect a supported device path without swallowing RankGraph query errors.
    std::vector<CommProtocol> protocols;
    CHK_RET(GetProtocolByEngine(param, protocols));
    u32* layers = nullptr;
    u32 layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    CHK_PTR_NULL(layers);
    bool hasDeviceLink = false;
    for (u32 i = 0; i < layerNum; ++i) {
        CommLink* links = nullptr;
        u32 linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layers[i], myRank_, remoteRank, &links, &linkNum));
        if (linkNum == 0) {
            continue;
        }
        CHK_PTR_NULL(links);
        for (u32 j = 0; j < linkNum; ++j) {
            hasDeviceLink = hasDeviceLink
                            || (links[j].srcEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE
                                && links[j].dstEndpointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE
                                && std::find(protocols.begin(), protocols.end(), links[j].linkAttr.linkProtocol)
                                       != protocols.end());
        }
    }
    if (hasDeviceLink) {
        CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, remoteRank, channels));
    } else {
        TopoInfoWithNetLayerDetails peerTopo;
        peerTopo.userRank = myRank_;
        CHK_RET(CalcChannelRequestMesh1DHighestHostRoce(comm, param, &peerTopo, {{myRank_, remoteRank}}, channels));
    }
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[BatchSendRecvNda] no peer channel"), HCCL_E_NOT_SUPPORT);
    // One link per peer, duplicated for independent send/receive notification streams.
    const HcclChannelDesc selected = channels.front();
    if (selected.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
        CHK_PRT_RET(
            selected.channelProtocol != COMM_PROTOCOL_ROCE
                || selected.remoteEndpoint.loc.locType != ENDPOINT_LOC_TYPE_HOST,
            HCCL_ERROR("[BatchSendRecvNda] unsupported HOST peer"), HCCL_E_NOT_SUPPORT);
    } else {
        CHK_PRT_RET(
            selected.localEndpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE
                || selected.remoteEndpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE,
            HCCL_ERROR("[BatchSendRecvNda] mixed endpoint locations"), HCCL_E_NOT_SUPPORT);
    }
    channels.assign(CHANNEL_NUM_PER_RANK_PAIR, selected);
    return HCCL_SUCCESS;
#else
    (void)comm;
    (void)param;
    (void)remoteRank;
    (void)channels;
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult InsV2BatchSendRecvNdaExecutor::SetDataTypeSize(HcclDataType dataType)
{
    CHK_PRT_RET(
        !IsValidHcclDataType(dataType), HCCL_ERROR("[InsV2BatchSendRecvNdaExecutor] invalid dataType[%d]", dataType),
        HCCL_E_PARA);
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType];
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::ProcessSelfSendRecvTasks(ThreadHandle& thread)
{
    while (!sendToSelfDeque_.empty() && !recvFromSelfDeque_.empty()) {
        if (sendToSelfDeque_.front()->count == recvFromSelfDeque_.front()->count
            && sendToSelfDeque_.front()->dataType == recvFromSelfDeque_.front()->dataType) {
            CHK_RET(SetDataTypeSize(sendToSelfDeque_.front()->dataType));
            u64 dataSize = sendToSelfDeque_.front()->count * dataTypeSize_;

            void* inputDataPtr = sendToSelfDeque_.front()->buf;
            void* outputDataPtr = recvFromSelfDeque_.front()->buf;
            DataSlice srcSlice(inputDataPtr, 0, dataSize);
            DataSlice dstSlice(outputDataPtr, 0, dataSize);
            CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
            HCCL_DEBUG(
                "[InsV2BatchSendRecvNdaExecutor][ProcessSelfSendRecvTasks] inputData[%p], outputData[%p], "
                "dataSize[%llu]",
                inputDataPtr, outputDataPtr, dataSize);
            sendToSelfDeque_.pop_front();
            recvFromSelfDeque_.pop_front();
        } else {
            HCCL_ERROR("[InsV2BatchSendRecvNdaExecutor][ProcessSelfSendRecvTasks] Send task and recv task to self : "
                       "count or dataType is not equal, please check the task list.");
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

bool InsV2BatchSendRecvNdaExecutor::SortSendItems(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const
{
    u32 aFlag = (a->remoteRank <= static_cast<uint32_t>(myRank_)) ? (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank <= static_cast<uint32_t>(myRank_)) ? (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag != bFlag) {
        return aFlag > bFlag;
    } else if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;
}

bool InsV2BatchSendRecvNdaExecutor::SortRecvItems(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const
{
    u32 aFlag = (a->remoteRank < static_cast<uint32_t>(myRank_)) ? (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank < static_cast<uint32_t>(myRank_)) ? (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag != bFlag) {
        return aFlag < bFlag;
    } else if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;
}

bool InsV2BatchSendRecvNdaExecutor::SortSelfItems(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const
{
    if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;
}

HcclResult InsV2BatchSendRecvNdaExecutor::GetPairWiseList(const HcclSendRecvItem* sendRecvInfo, u32 itemNum)
{
    HCCL_INFO("[InsV2BatchSendRecvNdaExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_INFO(
            "[InsV2BatchSendRecvNdaExecutor][GetPairWiseList] index is %u, itemNum is %u,"
            "localRankID is %d, remoteRank is %u, dataCount is %llu, sendRecvType is %u, rankSize is %u.",
            i, itemNum, myRank_, sendRecvInfo->remoteRank, sendRecvInfo->count,
            static_cast<u32>(sendRecvInfo->sendRecvType), rankSize_);
        if (sendRecvInfo->count > 0) {
            CHK_PTR_NULL(sendRecvInfo->buf);
            if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
                sendDeque_.push_back(sendRecvInfo);
            } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
                recvDeque_.push_back(sendRecvInfo);
            } else {
                HCCL_ERROR(
                    "[InsV2BatchSendRecvNdaExecutor][GetPairWiseList] sendRecvType wrong sendRecvType is %d, "
                    "rankID is %d, remoteRank is %u.",
                    sendRecvInfo->sendRecvType, myRank_, sendRecvInfo->remoteRank);
                return HcclResult::HCCL_E_PARA;
            }
        }
        sendRecvInfo++;
    }

    /* 此处的排序逻辑(pair-wise算法):
        1.sendDeque元素顺序是:先放remoteRank号小于等于root rank的第一个任务，依次减小(循环索引)直至放完
        2.recvDeque元素顺序是:先放remoteRank号大于等于root rank的第一个任务，依次增大(循环索引)直至放完
        如果有rank间重复send/recv场景，按照收发数据count从大到小排序
        如果数据count也一致，则按照dataType枚举值从大到小排序
    */
    auto sendCompare = [this](const HcclSendRecvItem* a, const HcclSendRecvItem* b) {
        return this->SortSendItems(a, b);
    };

    auto recvCompare = [this](const HcclSendRecvItem* a, const HcclSendRecvItem* b) {
        return this->SortRecvItems(a, b);
    };

    std::stable_sort(sendDeque_.begin(), sendDeque_.end(), sendCompare);
    std::stable_sort(recvDeque_.begin(), recvDeque_.end(), recvCompare);

    // 筛选自收发任务
    while ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))
           && (!recvDeque_.empty() && recvDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))) {
        sendToSelfDeque_.push_back(sendDeque_.front());
        recvFromSelfDeque_.push_back(recvDeque_.front());
        sendDeque_.pop_front();
        recvDeque_.pop_front();
    }
    // 自收发任务按照收发长度大小排序
    auto selfDequeCompare = [this](const HcclSendRecvItem* a, const HcclSendRecvItem* b) {
        return this->SortSelfItems(a, b);
    };

    std::stable_sort(sendToSelfDeque_.begin(), sendToSelfDeque_.end(), selfDequeCompare);
    std::stable_sort(recvFromSelfDeque_.begin(), recvFromSelfDeque_.end(), selfDequeCompare);

    // 如果自发自收任务没有完全匹配
    if ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))
        || (!recvDeque_.empty() && recvDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))) {
        HCCL_ERROR("[InsV2BatchSendRecvNdaExecutor] SendTask and Recv Task to rank itself do not match,"
                   "please check the task list.");
        return HCCL_E_PARA;
    }
    HCCL_INFO("[InsV2BatchSendRecvNdaExecutor][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::CalcSendSlices(PeerQueues& slices)
{
    while (!sendDeque_.empty()) {
        const HcclSendRecvItem* sendItem = sendDeque_.front();
        HCCL_INFO(
            "[InsV2BatchSendRecvNdaExecutor][CalcSendSlices] remoteRank[%u], buf[%p], count[%llu],"
            "dataType[%u], sendRecvType[%d].",
            sendItem->remoteRank, sendItem->buf, sendItem->count, sendItem->dataType, sendItem->sendRecvType);
        // 计算每轮搬运的最大数据量
        CHK_RET(SetDataTypeSize(sendItem->dataType));
        u64 maxCountPerLoop = maxTmpMemSize_ / dataTypeSize_;
        CHK_PRT_RET(
            maxCountPerLoop == 0, HCCL_ERROR("[InsV2BatchSendRecvNdaExecutor] send CCL slot is too small"),
            HCCL_E_PARA);
        u8* curInputPtr = static_cast<u8*>(sendItem->buf);
        CHK_PTR_NULL(curInputPtr);

        CHK_PRT_RET(
            sendItem->remoteRank >= rankSize_ || sendItem->remoteRank == myRank_,
            HCCL_ERROR("[BatchSendRecvNda] invalid remote peer[%u]", sendItem->remoteRank), HCCL_E_INTERNAL);
        ChannelInfo channel;
        CHK_RET(GetSendChannel(sendItem->remoteRank, channel));
        CHK_PRT_RET(
            channel.locationType != ENDPOINT_LOC_TYPE_DEVICE && channel.locationType != ENDPOINT_LOC_TYPE_HOST,
            HCCL_ERROR("[BatchSendRecvNda] unsupported channel location"), HCCL_E_PARA);
        const u32 peerIndex = sendItem->remoteRank < myRank_ ? sendItem->remoteRank : sendItem->remoteRank - 1;
        auto& sendSlices = slices[peerIndex * DIRECTIONS_PER_PEER];
        u64 curOffset = 0;
        u64 resDataCount = sendItem->count;
        while (resDataCount > 0) {
            // 判断本轮需搬运的数据量
            u64 transferCount = resDataCount > maxCountPerLoop ? maxCountPerLoop : resDataCount;
            u64 transferSize = transferCount * dataTypeSize_;
            curInputPtr = static_cast<u8*>(sendItem->buf) + curOffset;
            sendSlices.emplace_back(static_cast<void*>(curInputPtr), transferSize, sendItem->remoteRank);
            HCCL_DEBUG(
                "[InsV2BatchSendRecvNdaExecutor][CalcSendSlices] slice curOffset[%llu], slice size[%llu] "
                "curInputPtr "
                "[%p].",
                curOffset, transferSize, curInputPtr);
            curOffset += transferSize;
            resDataCount -= transferCount;
        }
        sendDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::CalcRecvSlices(PeerQueues& slices)
{
    while (!recvDeque_.empty()) {
        const HcclSendRecvItem* recvItem = recvDeque_.front();
        HCCL_INFO(
            "[InsV2BatchSendRecvNdaExecutor][CalcRecvSlices] remoteRank[%u], buf[%p], count[%llu],"
            "dataType[%u], sendRecvType[%d].",
            recvItem->remoteRank, recvItem->buf, recvItem->count, recvItem->dataType, recvItem->sendRecvType);
        // 计算每轮搬运的最大数据量
        CHK_RET(SetDataTypeSize(recvItem->dataType));
        u64 maxCountPerLoop = maxTmpMemSize_ / dataTypeSize_;
        CHK_PRT_RET(
            maxCountPerLoop == 0, HCCL_ERROR("[InsV2BatchSendRecvNdaExecutor] recv CCL slot is too small"),
            HCCL_E_PARA);
        u8* curOutputPtr = static_cast<u8*>(recvItem->buf);
        CHK_PTR_NULL(curOutputPtr);

        CHK_PRT_RET(
            recvItem->remoteRank >= rankSize_ || recvItem->remoteRank == myRank_,
            HCCL_ERROR("[BatchSendRecvNda] invalid remote peer[%u]", recvItem->remoteRank), HCCL_E_INTERNAL);
        ChannelInfo channel;
        CHK_RET(GetRecvChannel(recvItem->remoteRank, channel));
        CHK_PRT_RET(
            channel.locationType != ENDPOINT_LOC_TYPE_DEVICE && channel.locationType != ENDPOINT_LOC_TYPE_HOST,
            HCCL_ERROR("[BatchSendRecvNda] unsupported channel location"), HCCL_E_PARA);
        const u32 peerIndex = recvItem->remoteRank < myRank_ ? recvItem->remoteRank : recvItem->remoteRank - 1;
        auto& recvSlices = slices[peerIndex * DIRECTIONS_PER_PEER + 1];
        u64 curOffset = 0;
        u64 resDataCount = recvItem->count;
        while (resDataCount > 0) {
            // 判断本轮需搬运的数据量
            u64 transferCount = resDataCount > maxCountPerLoop ? maxCountPerLoop : resDataCount;
            u64 transferSize = transferCount * dataTypeSize_;
            curOutputPtr = static_cast<u8*>(recvItem->buf) + curOffset;
            recvSlices.emplace_back(static_cast<void*>(curOutputPtr), transferSize, recvItem->remoteRank);
            HCCL_DEBUG(
                "[InsV2BatchSendRecvNdaExecutor][CalcRecvSlices] slice curOffset[%llu], slice size[%llu] "
                "curOutputPtr "
                "[%p].",
                curOffset, transferSize, curOutputPtr);
            curOffset += transferSize;
            resDataCount -= transferCount;
        }
        recvDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::GetSendChannel(u32 remoteRank, ChannelInfo& sendChannel) const
{
    auto it = remoteRankToChannelInfo_[0].find(remoteRank);
    if (it == remoteRankToChannelInfo_[0].end()) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvNdaExecutor][GetSendChannel] Cannot find channel for remoteRank[%u]", remoteRank);
        return HCCL_E_INTERNAL;
    }
    if (it->second.size() < CHANNEL_NUM_PER_RANK_PAIR) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvNdaExecutor][GetSendChannel] Channel number[%zu] is less than expected "
            "number[%u]",
            it->second.size(), CHANNEL_NUM_PER_RANK_PAIR);
        return HCCL_E_INTERNAL;
    }
    if (remoteRank < myRank_) {
        sendChannel = it->second[0];
    } else {
        sendChannel = it->second[1];
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::GetRecvChannel(u32 remoteRank, ChannelInfo& recvChannel) const
{
    auto it = remoteRankToChannelInfo_[0].find(remoteRank);
    if (it == remoteRankToChannelInfo_[0].end()) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvNdaExecutor][GetRecvChannel] Cannot find channel for remoteRank[%u]", remoteRank);
        return HCCL_E_INTERNAL;
    }
    if (it->second.size() < CHANNEL_NUM_PER_RANK_PAIR) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvNdaExecutor][GetRecvChannel] Channel number[%zu] is less than expected "
            "number[%u]",
            it->second.size(), CHANNEL_NUM_PER_RANK_PAIR);
        return HCCL_E_INTERNAL;
    }
    if (remoteRank > myRank_) {
        recvChannel = it->second[0];
    } else {
        recvChannel = it->second[1];
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::InitTemplateParams(
    const SendRecvSlice& slice, ThreadHandle thread, BatchSendRecvOpType opType, SliceContext& context)
{
    ChannelInfo channel;
    if (opType == BatchSendRecvOpType::SEND) {
        CHK_RET(GetSendChannel(slice.remoteRank_, channel));
    } else {
        CHK_RET(GetRecvChannel(slice.remoteRank_, channel));
    }
    CHK_PRT_RET(
        channel.remoteRank != slice.remoteRank_,
        HCCL_ERROR(
            "[InsV2BatchSendRecvNdaExecutor][InitTemplateParams] slice peer[%u] differs from channel peer[%u]",
            slice.remoteRank_, channel.remoteRank),
        HCCL_E_INTERNAL);
    OpParam& sliceParam = context.param;
    sliceParam.sendRecvRemoteRank = slice.remoteRank_;
    TemplateDataParams& tempAlgParams = context.data;
    tempAlgParams.buffInfo.inputPtr = slice.addr_;
    tempAlgParams.buffInfo.outputPtr = slice.addr_;
    tempAlgParams.buffInfo.hcclBuff = cclMem_;
    tempAlgParams.count = 0;
    tempAlgParams.buffInfo.inBuffBaseOff = 0;
    tempAlgParams.buffInfo.outBuffBaseOff = 0;
    tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
    tempAlgParams.inputSliceStride = cclSlotSize_;
    tempAlgParams.sliceSize = slice.size_;
    tempAlgParams.opType = opType;
    TemplateResource& templateResource = context.resource;
    templateResource.channels[slice.remoteRank_] = {channel};
    templateResource.threads = {thread};
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::InitTemplate(SliceContext& context)
{
    context.algorithm = std::make_unique<InsTempBatchSendRecvNda>(
        context.param, myRank_, std::vector<std::vector<u32>>{{myRank_, context.param.sendRecvRemoteRank}});
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::RunTemplate(SliceContext& context)
{
    return context.algorithm->KernelRun(context.param, context.data, context.resource);
}

HcclResult InsV2BatchSendRecvNdaExecutor::RunAllSlices(const OpParam& param, PeerQueues& slices)
{
    // Ascending peers are the local projection of the (low, high) pair order used by the serial version.
    // Submit low->high before high->low so finite queues can drain even for unequal slice counts.
    for (u32 peerIndex = 0; peerIndex < slices.size() / DIRECTIONS_PER_PEER; ++peerIndex) {
        const u32 peer = peerIndex < myRank_ ? peerIndex : peerIndex + 1;
        const u32 base = peerIndex * DIRECTIONS_PER_PEER;
        if (slices[base].empty() && slices[base + 1].empty()) {
            continue;
        }
        for (u32 order = 0; order < DIRECTIONS_PER_PEER; ++order) {
            const u32 direction = myRank_ < peer ? order : 1U - order;
            auto& queue = slices[base + direction];
            const auto opType = direction == 0 ? BatchSendRecvOpType::SEND : BatchSendRecvOpType::RECV;
            SliceContext context(param);
            while (!queue.empty()) {
                CHK_RET(InitTemplateParams(queue.front(), threads_[direction], opType, context));
                if (context.algorithm == nullptr) {
                    CHK_RET(InitTemplate(context));
                }
                CHK_RET(RunTemplate(context));
                queue.pop_front();
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::RunLoopSendRecv(const OpParam& param, PeerQueues& slices)
{
    const std::vector<ThreadHandle> subThreads{threads_[1]};
    const std::vector<u32> notifyIndices{0};
    CHK_RET(PreSyncInterThreads(threads_[0], subThreads, notifyIndices));
    CHK_RET(RunAllSlices(param, slices));
    // The main SEND thread waits for the RECV worker through a queued notification.
    CHK_RET(PostSyncInterThreads(threads_[0], subThreads, notifyIndices));
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvNdaExecutor::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BatchSendRecvNdaExecutor][Orchestrate] Orchestrate Start.");
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (threads_.size() < SEND_RECV_THREAD_NUM) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvNdaExecutor][Orchestrate] threads num[%zu] is less than [%u]!", threads_.size(),
            SEND_RECV_THREAD_NUM);
        return HCCL_E_PARA;
    }
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    CHK_PRT_RET(
        rankSize_ == 0 || myRank_ >= rankSize_,
        HCCL_ERROR("[InsV2BatchSendRecvNdaExecutor] invalid rank[%u]/size[%u]", myRank_, rankSize_), HCCL_E_PARA);
    cclMem_ = resCtx.cclMem;
    // SEND reuses TX after each peer handshake. RECV grants HOST writers access to RX one slice at a time.
    cclSlotSize_ = cclMem_.size / 2;
    maxTmpMemSize_ = std::min(static_cast<u64>(cclSlotSize_), static_cast<u64>(UB_MAX_DATA_SIZE));
    CHK_PRT_RET(
        (rankSize_ > 1 && maxTmpMemSize_ == 0),
        HCCL_ERROR("[InsV2BatchSendRecvNdaExecutor][Orchestrate] maxTmpMemSize equals to zero."), HCCL_E_PARA);
    // 任务信息
    itemNum_ = param.batchSendRecvDataDes.itemNum;
    itemPtr_ = param.batchSendRecvDataDes.sendRecvItemsPtr;

    // 任务排序
    CHK_RET(GetPairWiseList(itemPtr_, itemNum_));
    // 执行自收发任务
    CHK_RET(ProcessSelfSendRecvTasks(threads_[0]));
    // 任务切片
    PeerQueues slices((rankSize_ - 1) * DIRECTIONS_PER_PEER);
    CHK_RET(CalcSendSlices(slices));
    CHK_RET(CalcRecvSlices(slices));
    // 循环执行任务
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    HcclResult runRet = RunLoopSendRecv(param, slices);
    const HcclResult restoreRet = static_cast<HcclResult>(HcommBatchModeStart(param.algTag));
    CHK_RET(runRet);
    CHK_RET(restoreRet);

    return HCCL_SUCCESS;
}

std::vector<CostModelParam> InsV2BatchSendRecvNdaExecutor::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    return {{0.0f, 0.0f, 1.0f, 0.0f}};
}

AlgNetMeta InsV2BatchSendRecvNdaExecutor::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    (void)algName;
    return BuildDefaultMeshNetMeta(topoInfo);
}

REGISTER_EXECUTOR_IMPL(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, AicpuBatchSendRecvNdaMesh, InsV2BatchSendRecvNdaExecutor);
REGISTER_ALG_ATTRS(
    AicpuBatchSendRecvNdaMesh, a.engine = OpExecuteConfig::AICPU_TS; topo.supportLevel0Topos = LEVEL0_TOPO_ANY;
    topo.isSupportLevel0PcieMix = true; topo.isHostDpuOnly = true; topo.isSupportLevel1Nhr = true;
    op.opCustomCheck = [](const OpParam& param, const TopoInfoWithNetLayerDetails* info) {
        bool supported = false;
        return CheckSupportNda(param.hcclComm, info, supported) == HCCL_SUCCESS && supported;
    };);
} // namespace ops_hccl
