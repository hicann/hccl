/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "alg_data_trans_wrapper.h"
#include "ins_v2_batch_send_recv_sole_executor.h"
#include "ins_temp_batch_send_recv_dpu.h"
#include "hccl_rank_graph.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2BatchSendRecvSoleExecutor()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate> std::string InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Describe() const
{
    return "Instruction BatchSendRecvS Executor.";
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // AlgHierarchyInfoForAllLevel固定为一层
    CHK_PRT_RET((topoInfo->userRankSize == 0),
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][CalcAlgHierarchyInfo] Rank [%u], rankSize is 0.", myRank_),
        HcclResult::HCCL_E_PARA);

    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].resize(1);
    algHierarchyInfo.infos[0][0].clear();
    for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
        algHierarchyInfo.infos[0][0].push_back(rankId);
    }
    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][CalcAlgHierarchyInfo] [%u] Success.", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
    AlgResourceRequest &resourceRequest)
{
#ifndef AICPU_COMPILE
    // 变量检查
    CHK_PTR_NULL(topoInfo);
    myRank_ = topoInfo->userRank;
    itemNum_ = param.batchSendRecvDataDes.itemNum;
    itemPtr_ = param.batchSendRecvDataDes.sendRecvItemsPtr;

    // thread和notify资源
    resourceRequest.slaveThreadNum = 1;
    resourceRequest.notifyNumPerThread = {1};
    resourceRequest.notifyNumOnMainThread = 1;

    // 将本次任务需要通信的remoteRank加入集合
    CHK_PTR_NULL(itemPtr_);
    commTargetUserRankSet_.clear();
    for (u32 i = 0; i < itemNum_; i++) {
        commTargetUserRankSet_.insert((itemPtr_ + i)->remoteRank);
        HCCL_DEBUG(
            "[InsV2BatchSendRecvSoleExecutor][CalcRes] insert remoteUserRank[%u] to Set ", (itemPtr_ + i)->remoteRank);
    }
    HCCL_DEBUG(
        "[InsV2BatchSendRecvSoleExecutor][CalcRes]commTargetUserRankSet_ size[%zu]", commTargetUserRankSet_.size());

    // 遍历remoteRank集合，如果在algHierarchyInfo中，则建链，否则报错
    std::vector<HcclChannelDesc> channelLevel0;
    for (const u32 &remoteRank : commTargetUserRankSet_) {
        if (remoteRank == static_cast<uint32_t>(myRank_)) {
            continue;
        }
        CommLink *linkList = nullptr;
        u32 listSize = 0;
        u32 channelNum = 2;
        auto it = std::find(algHierarchyInfo.infos[0][0].begin(), algHierarchyInfo.infos[0][0].end(), remoteRank);
        if (it == algHierarchyInfo.infos[0][0].end()) {
            HCCL_ERROR(
                "[InsV2BatchSendRecvSoleExecutor][CalcRes] task remoteRank[%u] has no direct link with myRank[%u]",
                remoteRank, myRank_);
        } else {
            CHK_RET(HcclRankGraphGetLinks(comm, 0, myRank_, remoteRank, &linkList, &listSize));
            if (listSize == 0) {
                channelNum = 1;
            }
        }

        std::vector<HcclChannelDesc> channelByRank;
        // dpu用1条channel即可 aicpu 通信需要2条
        CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, remoteRank, channelByRank, channelNum));
        // 直接插入到channelLevel0末尾，同一个level的所有rank的所有channel都放在同一级vector中
        channelLevel0.insert(channelLevel0.end(), channelByRank.begin(), channelByRank.end());
    }
    resourceRequest.channels.push_back(channelLevel0);
#endif // !AICPU_COMPILE
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ProcessSelfSendRecvTasks(ThreadHandle &thread)
{
    while (!sendToSelfDeque_.empty() && !recvFromSelfDeque_.empty()) {
        if (sendToSelfDeque_.front()->count == recvFromSelfDeque_.front()->count
            && sendToSelfDeque_.front()->dataType == recvFromSelfDeque_.front()->dataType) {
            dataTypeSize_ = DATATYPE_SIZE_TABLE[sendToSelfDeque_.front()->dataType];
            u64 dataSize = sendToSelfDeque_.front()->count * dataTypeSize_;

            void *inputDataPtr = sendToSelfDeque_.front()->buf;
            void *outputDataPtr = recvFromSelfDeque_.front()->buf;
            DataSlice srcSlice(inputDataPtr, 0, dataSize);
            DataSlice dstSlice(outputDataPtr, 0, dataSize);
            CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
            HCCL_DEBUG("[InsV2BatchSendRecvSoleExecutor][ProcessSelfSendRecvTasks] inputData[%p], outputData[%p], "
                       "dataSize[%llu]",
                inputDataPtr, outputDataPtr, dataSize);
            sendToSelfDeque_.pop_front();
            recvFromSelfDeque_.pop_front();
        } else {
            HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][ProcessSelfSendRecvTasks] Send task and recv task to self : "
                       "count or dataType is not equal, please check the task list.");
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::SortSendItems(
    const HcclSendRecvItem *a, const HcclSendRecvItem *b) const
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

template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::SortRecvItems(
    const HcclSendRecvItem *a, const HcclSendRecvItem *b) const
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

template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::SortSelfItems(
    const HcclSendRecvItem *a, const HcclSendRecvItem *b) const
{
    if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetPairWiseList(
    const HcclSendRecvItem *sendRecvInfo, u32 itemNum)
{
    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][GetPairWiseList] index is %u, itemNum is %u,"
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
                HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][GetPairWiseList] sendRecvType wrong sendrecvType is %d, "
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
    auto sendCompare = [this](const HcclSendRecvItem *a, const HcclSendRecvItem *b) {
        return this->SortSendItems(a, b);
    };

    auto recvCompare = [this](const HcclSendRecvItem *a, const HcclSendRecvItem *b) {
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
    auto selfDequeCompare = [this](const HcclSendRecvItem *a, const HcclSendRecvItem *b) {
        return this->SortSelfItems(a, b);
    };

    std::stable_sort(sendToSelfDeque_.begin(), sendToSelfDeque_.end(), selfDequeCompare);
    std::stable_sort(recvFromSelfDeque_.begin(), recvFromSelfDeque_.end(), selfDequeCompare);

    // 如果自发自收任务没有完全匹配
    if ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))
        || (!recvDeque_.empty() && recvDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))) {
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor] SendTask and Recv Task to rank itself do not match,"
                   "please check the task list.");
        return HCCL_E_PARA;
    }
    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate> HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcSendSlices()
{
    while (!sendDeque_.empty()) {
        const HcclSendRecvItem *sendItem = sendDeque_.front();
        HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][CalcSendSlices] remoteRank[%u], buf[%p], count[%llu],"
                  "dataType[%u], sendRecvType[%d].",
            sendItem->remoteRank, sendItem->buf, sendItem->count, sendItem->dataType, sendItem->sendRecvType);
        // 计算每轮搬运的最大数据量
        dataTypeSize_ = DATATYPE_SIZE_TABLE[sendItem->dataType];
        u64 maxCountPerLoop = maxTmpMemSize_ / dataTypeSize_;
        u8 *curInputPtr = static_cast<u8 *>(sendItem->buf);
        CHK_PTR_NULL(curInputPtr);

        u64 curOffset = 0;
        u64 resDataCount = sendItem->count;
        while (resDataCount > 0) {
            // 判断本轮需搬运的数据量
            u64 transferCount = resDataCount > maxCountPerLoop ? maxCountPerLoop : resDataCount;
            u64 transferSize = transferCount * dataTypeSize_;
            curInputPtr = static_cast<u8 *>(sendItem->buf) + curOffset;
            sendDataSilces_.emplace_back(static_cast<void *>(curInputPtr), transferSize, sendItem->remoteRank);
            HCCL_DEBUG("[InsV2BatchSendRecvSoleExecutor][CalcSendSlices] slice curOffset[%llu], slice size[%llu] "
                       "curInputPtr [%p].",
                curOffset, transferSize, curInputPtr);
            curOffset += transferSize;
            resDataCount -= transferCount;
        }
        sendDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate> HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRecvSlices()
{
    while (!recvDeque_.empty()) {
        const HcclSendRecvItem *recvItem = recvDeque_.front();
        HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][CalcRecvSlices] remoteRank[%u], buf[%p], count[%llu],"
                  "dataType[%u], sendRecvType[%d].",
            recvItem->remoteRank, recvItem->buf, recvItem->count, recvItem->dataType, recvItem->sendRecvType);
        // 计算每轮搬运的最大数据量
        dataTypeSize_ = DATATYPE_SIZE_TABLE[recvItem->dataType];
        u64 maxCountPerLoop = maxTmpMemSize_ / dataTypeSize_;
        u8 *curOutputPtr = static_cast<u8 *>(recvItem->buf);
        CHK_PTR_NULL(curOutputPtr);

        u64 curOffset = 0;
        u64 resDataCount = recvItem->count;
        while (resDataCount > 0) {
            // 判断本轮需搬运的数据量
            u64 transferCount = resDataCount > maxCountPerLoop ? maxCountPerLoop : resDataCount;
            u64 transferSize = transferCount * dataTypeSize_;
            curOutputPtr = static_cast<u8 *>(recvItem->buf) + curOffset;
            recvDataSilces_.emplace_back(static_cast<void *>(curOutputPtr), transferSize, recvItem->remoteRank);
            HCCL_DEBUG("[InsV2BatchSendRecvSoleExecutor][CalcRecvSlices] slice curOffset[%llu], slice size[%llu] "
                       "curOutputPtr [%p].",
                curOffset, transferSize, curOutputPtr);
            curOffset += transferSize;
            resDataCount -= transferCount;
        }
        recvDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetSendChannel(
    u32 remoteRank, ChannelInfo &sendChannel) const
{
    auto it = remoteRankToChannelInfo_[0].find(remoteRank);
    if (it == remoteRankToChannelInfo_[0].end()) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvSoleExecutor][GetSendChannel] Cannot find channel for remoteRank[%u]", remoteRank);
        return HCCL_E_INTERNAL;
    }
    if (it->second.size() < channelNumPerRankPair_) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvSoleExecutor][GetSendChannel] Channel number[%u] is less than expected number[%u]",
            it->second.size(), channelNumPerRankPair_);
        return HCCL_E_INTERNAL;
    }

    if (it->second[0].locationType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        sendChannel = it->second[0];
        return HCCL_SUCCESS;
    }
    if (remoteRank < myRank_) {
        sendChannel = it->second[0];
    } else {
        sendChannel = it->second[1];
    }

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetRecvChannel(
    u32 remoteRank, ChannelInfo &recvChannel) const
{
    auto it = remoteRankToChannelInfo_[0].find(remoteRank);
    if (it == remoteRankToChannelInfo_[0].end()) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvSoleExecutor][GetRecvChannel] Cannot find channel for remoteRank[%u]", remoteRank);
        return HCCL_E_INTERNAL;
    }
    if (it->second.size() < channelNumPerRankPair_) {
        HCCL_ERROR(
            "[InsV2BatchSendRecvSoleExecutor][GetRecvChannel] Channel number[%u] is less than expected number[%u]",
            it->second.size(), channelNumPerRankPair_);
        return HCCL_E_INTERNAL;
    }

    if (it->second[0].locationType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        recvChannel = it->second[0];
        return HCCL_SUCCESS;
    }
    if (remoteRank > myRank_) {
        recvChannel = it->second[0];
    } else {
        recvChannel = it->second[1];
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ProcessSendDataSlice(
    const OpParam &param, SendRecvSlice &sendSlice, BatchSendRecvOpType opType) const
{
    ChannelInfo sendChannel;
    CHK_RET(GetSendChannel(sendSlice.remoteRank_, sendChannel));
    // 构造param
    OpParam sendRecvParam = param;
    sendRecvParam.sendRecvRemoteRank = sendSlice.remoteRank_;
    // 构造template
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(sendRecvParam, myRank_, algHierarchyInfo_.infos[0]);
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = sendSlice.addr_;
    tempAlgParams.buffInfo.outputPtr = sendChannel.remoteCclMem.addr;; // 无论跨框还是框内 都要发送到对方的ccl上
    tempAlgParams.buffInfo.hcclBuff = cclMem_;       // 本端的ccl
    tempAlgParams.sliceSize = sendSlice.size_;
    tempAlgParams.count = sendSlice.size_ / dataTypeSize_;
    tempAlgParams.opType = opType; // 传入实际操作
    // 这里用来放每张卡可以用的cclBuffer的大小
    tempAlgParams.inputSliceStride = maxTmpMemSize_;
    tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
    // template资源
    TemplateResource templateResource;
    templateResource.channels = remoteRankToChannelInfo_[0];
    templateResource.threads = threads_;
    templateResource.npu2DpuShmemPtr = npu2DpuShmemPtr_;
    templateResource.dpu2NpuShmemPtr = dpu2NpuShmemPtr_;
    CHK_RET(algTemplate->KernelRun(sendRecvParam, tempAlgParams, templateResource));
    HCCL_DEBUG("[InsV2BatchSendRecvSoleExecutor][ProcessSendDataSlice] Process a send task by dpu template, "
               "CCLBuffer[%p], remoteRank[%u].",
        cclMem_.addr, sendSlice.remoteRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ProcessRecvDataSlice(
    const OpParam &param, SendRecvSlice &recvSlice, BatchSendRecvOpType opType) const
{
    ChannelInfo recvChannel;
    CHK_RET(GetRecvChannel(recvSlice.remoteRank_, recvChannel));
    // 构造param
    OpParam sendRecvParam = param;
    sendRecvParam.sendRecvRemoteRank = recvSlice.remoteRank_;
    // 构造template
    if (algHierarchyInfo_.infos.empty())
    {
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][ProcessRecvDataSlice] algHierarchyInfo infos is empty!");
        return HCCL_E_PARA;
    }

    if (algHierarchyInfo_.infos[0].empty())
    {
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][ProcessRecvDataSlice] algHierarchyInfo infos[0] is empty!");
        return HCCL_E_PARA;
    }
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(sendRecvParam, myRank_, algHierarchyInfo_.infos[0]);
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr
        = recvChannel.remoteInput.addr; // 此处channel.remoteInput不是对端input buffer 这里地址实际上不会被使用
    tempAlgParams.buffInfo.outputPtr = recvSlice.addr_; // 最后读到本端ccl上
    tempAlgParams.buffInfo.hcclBuff = cclMem_;          // 本端的ccl
    tempAlgParams.sliceSize = recvSlice.size_;
    tempAlgParams.count = recvSlice.size_ / dataTypeSize_;
    tempAlgParams.opType = opType; // 传入实际操作
    // 这里用来放每张卡可以用的cclBuffer的大小
    tempAlgParams.inputSliceStride = maxTmpMemSize_;
    tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
    // template资源
    TemplateResource templateResource;
    templateResource.channels = remoteRankToChannelInfo_[0];
    templateResource.threads = threads_;
    templateResource.npu2DpuShmemPtr = npu2DpuShmemPtr_;
    templateResource.dpu2NpuShmemPtr = dpu2NpuShmemPtr_;
    CHK_RET(algTemplate->KernelRun(sendRecvParam, tempAlgParams, templateResource));
    HCCL_DEBUG("[InsV2BatchSendRecvSoleExecutor][ProcessRecvDataSlice] Process a recv task by read mode, "
               "outputBuffer[%p], remoteRank[%u].",
        recvSlice.addr_, recvSlice.remoteRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::RunLoopSendRecv(const OpParam &param)
{
    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][RunLoopSendRecv] Process start.");
    // 前同步
    std::vector<ThreadHandle> subThreads = {threads_[1]};
    std::vector<u32> notifyIdxMainToSub = {0};
    CHK_RET(PreSyncInterThreads(threads_[0], subThreads, notifyIdxMainToSub));

    for (auto &slice : recvDataSilces_) {
        // 发recv的record
        CHK_RET(ProcessRecvDataSlice(param, slice, BatchSendRecvOpType::RECORD));
    }

    for (auto &slice : sendDataSilces_) {
        // 发writeWithNotify
        CHK_RET(ProcessSendDataSlice(param, slice, BatchSendRecvOpType::SEND));
    }

    while (!recvDataSilces_.empty()) {
        // waitRecv,按recv接收数据
        CHK_RET(ProcessRecvDataSlice(param, recvDataSilces_.front(), BatchSendRecvOpType::RECV));
        recvDataSilces_.pop_front();
    }

    while (!sendDataSilces_.empty()) {
        // fence
        CHK_RET(ProcessSendDataSlice(param, sendDataSilces_.front(), BatchSendRecvOpType::FENCE));
        sendDataSilces_.pop_front();
    }
    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][RunLoopSendRecv] Process all tasks finish.");
    // 后同步
    std::vector<u32> notifyIdxSubToMain = {0};
    CHK_RET(PostSyncInterThreads(threads_[0], subThreads, notifyIdxSubToMain));
    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][RunLoopSendRecv] post sync success.");

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BatchSendRecvSoleExecutor][Orchestrate] Orchestrate Start.");
    // 给channels_和threads_赋值
    if (resCtx.threads.empty()) {
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][Orchestrate] threads is empty!");
        return HCCL_E_INTERNAL;
    }

    algHierarchyInfo_.infos.resize(1);
    algHierarchyInfo_.infos[0].resize(1);
    algHierarchyInfo_.infos[0][0].clear();
    for (uint32_t rankId = 0; rankId < resCtx.topoInfo.userRankSize; rankId++) {
        algHierarchyInfo_.infos[0][0].push_back(rankId);
    }

    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    if (remoteRankToChannelInfo_.empty() || remoteRankToChannelInfo_[0].empty()) {
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][Orchestrate] no channel found!");
        return HCCL_E_INTERNAL;
    }

    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    CHK_PRT_RET((rankSize_ == 0),
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][Orchestrate] rankSize equals to zero."), HCCL_E_PARA);
    cclMem_ = resCtx.cclMem;
    cclMem_.size = cclMem_.size / rankSize_;
    maxTmpMemSize_ = std::min(cclMem_.size, UB_MAX_DATA_SIZE);
    CHK_PRT_RET((maxTmpMemSize_ == 0),
        HCCL_ERROR("[InsV2BatchSendRecvSoleExecutor][Orchestrate] maxTmpMemSize equals to zero."), HCCL_E_PARA);

    // 任务信息
    itemNum_ = param.batchSendRecvDataDes.itemNum;
    itemPtr_ = param.batchSendRecvDataDes.sendRecvItemsPtr;
    npu2DpuShmemPtr_ = resCtx.npu2DpuShmemPtr;
    dpu2NpuShmemPtr_ = resCtx.dpu2NpuShmemPtr;

    // 任务排序
    CHK_RET(GetPairWiseList(itemPtr_, itemNum_));
    // 执行自收发任务
    CHK_RET(ProcessSelfSendRecvTasks(threads_[0]));
    // 任务切片
    CHK_RET(CalcSendSlices());
    CHK_RET(CalcRecvSlices());
    // 循环执行任务
    CHK_RET(RunLoopSendRecv(param));

    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, InsBatchSendRecvDPU, InsV2BatchSendRecvSoleExecutor, TopoMatch1D,
    InsTempBatchSendRecvDpu);
} // namespace ops_hccl