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
#include "ins_v2_batch_send_recv_executor.h"

namespace ops_hccl { 
InsV2BatchSendRecvExecutor::InsV2BatchSendRecvExecutor()
{
}

HcclResult InsV2BatchSendRecvExecutor::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    algHierarchyInfo.infos.resize(1);
    algHierarchyInfo.infos[0].resize(1);
    for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
        algHierarchyInfo.infos[0][0].push_back(rankId);
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::CalcRes(HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
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
        HCCL_DEBUG("[InsV2BatchSendRecvExecutor][CalcRes] insert remoteUserRank[%u] to Set ",
            (itemPtr_ + i)->remoteRank);
    }
    HCCL_DEBUG("[InsV2BatchSendRecvExecutor][CalcRes]commTargetUserRankSet_ size[%zu]",
        commTargetUserRankSet_.size());

    // 遍历remoteRank集合，如果在algHierarchyInfo中，则建链，否则报错
    std::vector<HcclChannelDesc> channelLevel0;
    for (const u32& remoteRank : commTargetUserRankSet_) {
        if (remoteRank == static_cast<uint32_t>(myRank_)) {
            continue;
        }
        auto it = std::find(algHierarchyInfo.infos[0][0].begin(), algHierarchyInfo.infos[0][0].end(), remoteRank);
        if (it == algHierarchyInfo.infos[0][0].end()) {
            HCCL_ERROR("[InsV2BatchSendRecvExecutor][CalcRes] task remoteRank[%u] has no direct link with myRank[%u]",
                remoteRank, myRank_);
            return HCCL_E_PARA;
        } else {
            std::vector<HcclChannelDesc> channelByRank;
            // 和同一个远端之间的send和recv在不同流上，所以需要申请2条channel
            CHK_RET(CreateChannelRequestByRankId(comm, myRank_, remoteRank, channelByRank, channelNumPerRankPair_));
            // 直接插入到channelLevel0末尾，同一个level的所有rank的所有channel都放在同一级vector中
            channelLevel0.insert(channelLevel0.end(), channelByRank.begin(), channelByRank.end());
        }
    }
    resourceRequest.channels.push_back(channelLevel0);

    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::ProcessSelfSendRecvTasks(ThreadHandle& thread)
{
    while (!sendToSelfDeque_.empty() && !recvFromSelfDeque_.empty()) {
        if (sendToSelfDeque_.front()->count == recvFromSelfDeque_.front()->count &&
            sendToSelfDeque_.front()->dataType == recvFromSelfDeque_.front()->dataType) {
            dataTypeSize_ =  DATATYPE_SIZE_TABLE[sendToSelfDeque_.front()->dataType];
            u64 dataSize = sendToSelfDeque_.front()->count * dataTypeSize_;

            void* inputDataPtr = sendToSelfDeque_.front()->buf;
            void* outputDataPtr = recvFromSelfDeque_.front()->buf;
            DataSlice srcSlice(inputDataPtr, 0, dataSize);
            DataSlice dstSlice(outputDataPtr, 0, dataSize);
            CHK_RET(LocalCopy(thread, srcSlice, dstSlice));
            HCCL_DEBUG("[InsV2BatchSendRecvExecutor][ProcessSelfSendRecvTasks] inpuData[%p], outputData[%p], dataSize[%llu]",
                inputDataPtr, outputDataPtr, dataSize);
            sendToSelfDeque_.pop_front();
            recvFromSelfDeque_.pop_front();
        } else {
            HCCL_ERROR("[InsV2BatchSendRecvExecutor][ProcessSelfSendRecvTasks] Send task and recv task to self : "\
                "count or dataType is not equal, please check the task list.");
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

bool InsV2BatchSendRecvExecutor::SortSendItems(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const{
    u32 aFlag = (a->remoteRank <= static_cast<uint32_t>(myRank_)) ?
        (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank <= static_cast<uint32_t>(myRank_)) ?
        (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag != bFlag) {
        return aFlag > bFlag;
    } else if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;    
}

bool InsV2BatchSendRecvExecutor::SortRecvItems(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const{
    u32 aFlag = (a->remoteRank < static_cast<uint32_t>(myRank_)) ?
        (a->remoteRank + rankSize_) : a->remoteRank;
    u32 bFlag = (b->remoteRank < static_cast<uint32_t>(myRank_)) ?
        (b->remoteRank + rankSize_) : b->remoteRank;
    if (aFlag != bFlag) {
        return aFlag < bFlag;
    } else if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;  
}

bool InsV2BatchSendRecvExecutor::SortSelfItems(const HcclSendRecvItem* a, const HcclSendRecvItem* b) const{
    if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;  
}

HcclResult InsV2BatchSendRecvExecutor::GetPairWiseList(const HcclSendRecvItem *sendRecvInfo, u32 itemNum)
{
    HCCL_INFO("[InsV2BatchSendRecvExecutor][GetPairWiseList] Start sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_INFO("[InsV2BatchSendRecvExecutor][GetPairWiseList] index is %u, itemNum is %u,"\
            "localRankID is %d, remoteRank is %u, sendRecvType is %u, rankSize is %u.",
            i, itemNum, myRank_, sendRecvInfo->remoteRank,
            static_cast<u32>(sendRecvInfo->sendRecvType), rankSize_);
        CHK_PTR_NULL(sendRecvInfo->buf);

        if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
            sendDeque_.push_back(sendRecvInfo);
        } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
            recvDeque_.push_back(sendRecvInfo);
        } else {
            HCCL_ERROR("[InsV2BatchSendRecvExecutor][GetPairWiseList] sendRecvType wrong sendrecvType is %d, "\
                "rankID is %d, remoteRank is %u.", sendRecvInfo->sendRecvType, myRank_, sendRecvInfo->remoteRank);
            return HcclResult::HCCL_E_PARA;
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
    while ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_)) &&
        (!recvDeque_.empty() && recvDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))) {
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
    if ((!sendDeque_.empty() && sendDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_)) ||
        (!recvDeque_.empty() && recvDeque_.front()->remoteRank == static_cast<uint32_t>(myRank_))) {
            HCCL_ERROR("[InsV2BatchSendRecvExecutor] SendTask and Recv Task to rank itself do not match,"\
            "please check the task list.");
        return HCCL_E_PARA;
    }
    HCCL_INFO("[InsV2BatchSendRecvExecutor][GetPairWiseList] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::CalcSendSlices()
{
    while (!sendDeque_.empty()) {
        const HcclSendRecvItem* sendItem = sendDeque_.front();
        HCCL_INFO("[InsV2BatchSendRecvExecutor][CalcSendSlices] remoteRank[%u], buf[%p], count[%llu],"\
            "dataType[%u], sendRecvType[%d].", sendItem->remoteRank, sendItem->buf,
            sendItem->count, sendItem->dataType, sendItem->sendRecvType);
        // 计算每轮搬运的最大数据量
        dataTypeSize_ = DATATYPE_SIZE_TABLE[sendItem->dataType];
        u64 maxCountPerLoop = maxTmpMemSize_ / dataTypeSize_;
        u8* curInputPtr = static_cast<u8*>(sendItem->buf);
        CHK_PTR_NULL(curInputPtr);

        u64 curOffset = 0;
        u64 resDataCount = sendItem->count;
        while(resDataCount > 0) {
            // 判断本轮需搬运的数据量
            u64 transferCount = resDataCount > maxCountPerLoop ? maxCountPerLoop : resDataCount;
            u64 transferSize = transferCount * dataTypeSize_;
            curInputPtr = static_cast<u8*>(sendItem->buf) + curOffset;
            sendDataSilces_.emplace_back(static_cast<void*>(curInputPtr), transferSize, sendItem->remoteRank);
            HCCL_DEBUG("[InsV2BatchSendRecvExecutor][CalcSendSlices] slice curOffset[%llu], slice size[%llu] curInputPtr [%p].",
                curOffset, transferSize, curInputPtr);
            curOffset += transferSize;
            resDataCount -= transferCount;
        }
        sendDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::CalcRecvSlices()
{
    while (!recvDeque_.empty()) {
        const HcclSendRecvItem* recvItem = recvDeque_.front();
        HCCL_INFO("[InsV2BatchSendRecvExecutor][CalcRecvSlices] remoteRank[%u], buf[%p], count[%llu],"\
            "dataType[%u], sendRecvType[%d].", recvItem ->remoteRank, recvItem ->buf,
            recvItem->count, recvItem->dataType, recvItem->sendRecvType);
        // 计算每轮搬运的最大数据量
        dataTypeSize_ = DATATYPE_SIZE_TABLE[recvItem->dataType];
        u64 maxCountPerLoop = maxTmpMemSize_ / dataTypeSize_;
        u8* curOutputPtr = static_cast<u8*>(recvItem->buf);
        CHK_PTR_NULL(curOutputPtr);

        u64 curOffset = 0;
        u64 resDataCount = recvItem->count;
        while(resDataCount > 0) {
            // 判断本轮需搬运的数据量
            u64 transferCount = resDataCount > maxCountPerLoop ? maxCountPerLoop : resDataCount;
            u64 transferSize = transferCount * dataTypeSize_;
            curOutputPtr = static_cast<u8*>(recvItem->buf) + curOffset;
            recvDataSilces_.emplace_back(static_cast<void*>(curOutputPtr), transferSize, recvItem->remoteRank);
            HCCL_DEBUG("[InsV2BatchSendRecvExecutor][CalcRecvSlices] slice curOffset[%llu], slice size[%llu] curOutputPtr [%p].",
                curOffset, transferSize, curOutputPtr);
            curOffset += transferSize;
            resDataCount -= transferCount;
        }
        recvDeque_.pop_front();
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::GetSendChannel(u32 remoteRank, ChannelInfo& sendChannel) const
{
    auto it = remoteRankToChannelInfo_[0].find(remoteRank);
    if (it == remoteRankToChannelInfo_[0].end()) {
        HCCL_ERROR("[InsV2BatchSendRecvExecutor][GetSendChannel] Cannot find channel for remoteRank[%u]",
            remoteRank);
        return HCCL_E_INTERNAL;
    }
    if (it->second.size() < channelNumPerRankPair_) {
        HCCL_ERROR("[InsV2BatchSendRecvExecutor][GetSendChannel] Channel number[%u] is less than expected number[2]",
            it->second.size());
        return HCCL_E_INTERNAL;
    }
    if (remoteRank < myRank_) {
        sendChannel = it->second[0];
    } else {
        sendChannel = it->second[1];
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::GetRecvChannel(u32 remoteRank, ChannelInfo& recvChannel) const
{
    auto it = remoteRankToChannelInfo_[0].find(remoteRank);
    if (it == remoteRankToChannelInfo_[0].end()) {
        HCCL_ERROR("[InsV2BatchSendRecvExecutor][GetRecvChannel] Cannot find channel for remoteRank[%u]",
            remoteRank);
        return HCCL_E_INTERNAL;
    }
    if (it->second.size() < channelNumPerRankPair_) {
        HCCL_ERROR("[InsV2BatchSendRecvExecutor][GetRecvChannel] Channel number[%u] is less than expected number[2]",
            it->second.size());
        return HCCL_E_INTERNAL;
    }
    if (remoteRank > myRank_) {
        recvChannel = it->second[0];
    } else {
        recvChannel = it->second[1];
    }
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::ProcessSendDataSlice(SendRecvSlice& sendSlice, ThreadHandle& thread) const
{
    // 数据从input mem copy到CCL buffer上
    DataSlice srcSlice(sendSlice.addr_, 0, sendSlice.size_);
    DataSlice dstSlice(cclMem_.addr, 0, sendSlice.size_);
    CHK_RET(LocalCopy(thread, srcSlice, dstSlice));

    // 发送数据
    ChannelInfo sendChannel;
    CHK_RET(GetSendChannel(sendSlice.remoteRank_, sendChannel));
    HCCL_DEBUG("[InsV2BatchSendRecvExecutor][ProcessSendDataSlice] myRank:[%u], remoteRank:[%u], "\
        "send channel handle[0x%llx].", myRank_, sendSlice.remoteRank_, sendChannel.handle);
    SlicesList slices({}, {}); // read模式下发送端不感知数据信息，直接给一个空的
    DataInfo sendDataInfo(sendChannel, slices);
    SendRead(sendDataInfo, thread);
    HCCL_DEBUG("[InsV2BatchSendRecvExecutor][ProcessSendDataSlice] Process a send task by read mode, CCLBuffer[%p], remoteRank[%u].",
        cclMem_.addr, sendSlice.remoteRank_);
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::ProcessRecvDataSlice(SendRecvSlice& recvSlice, ThreadHandle& thread) const
{
    // 接收数据
    ChannelInfo recvChannel;
    CHK_RET(GetRecvChannel(recvSlice.remoteRank_, recvChannel));
    HCCL_DEBUG("[InsV2BatchSendRecvExecutor][ProcessRecvDataSlice] myRank:[%u], remoteRank:[%u], "\
        "recv channel handle[0x%llx].", myRank_, recvSlice.remoteRank_, recvChannel.handle);

    void* remoteCclBuffAddr = recvChannel.remoteCclMem.addr;
    DataSlice recvSrcDataSlice(remoteCclBuffAddr, 0, recvSlice.size_);
    DataSlice recvDstDataSlice(recvSlice.addr_, 0, recvSlice.size_); // 采用DMA消减，直接接收到本端的outputBuffer上
    SlicesList slices({recvSrcDataSlice}, {recvDstDataSlice});
    DataInfo recvDataInfo(recvChannel, slices);
    RecvRead(recvDataInfo, thread);
    HCCL_DEBUG("[InsV2BatchSendRecvExecutor][ProcessRecvDataSlice] Process a recv task by read mode, outputBuffer[%p], remoteRank[%u].",
        recvSlice.addr_, recvSlice.remoteRank_);
    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::RunLoopSendRecv()
{
    // 前同步
    std::vector<ThreadHandle> subThreads = {threads_[1]};
    std::vector<u32> notifyIdxMainToSub = {0};
    CHK_RET(PreSyncInterThreads(threads_[0], subThreads, notifyIdxMainToSub));

    while(!sendDataSilces_.empty() || !recvDataSilces_.empty()) {
        if(!sendDataSilces_.empty()) {
            CHK_RET(ProcessSendDataSlice(sendDataSilces_.front(), threads_[0]));
            sendDataSilces_.pop_front();
        }
        if(!recvDataSilces_.empty()) {
            CHK_RET(ProcessRecvDataSlice(recvDataSilces_.front(), threads_[1]));
            recvDataSilces_.pop_front();
        }
    }
    HCCL_INFO("[InsV2BatchSendRecvExecutor][RunLoopSendRecv] Process all tasks finish.");
    // 后同步
    std::vector<u32> notifyIdxSubToMain = {0};
    CHK_RET(PostSyncInterThreads(threads_[0], subThreads, notifyIdxSubToMain));
    HCCL_INFO("[InsV2BatchSendRecvExecutor][RunLoopSendRecv] post sync success.");

    return HCCL_SUCCESS;
}

HcclResult InsV2BatchSendRecvExecutor::Orchestrate(const OpParam &param,
    const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2BatchSendRecvExecutor][Orchestrate] Orchestrate Start.");
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    cclMem_ = resCtx.cclMem;
    maxTmpMemSize_ = std::min(resCtx.cclMem.size, UB_MAX_DATA_SIZE);
    CHK_PRT_RET((maxTmpMemSize_ == 0),
        HCCL_ERROR("[InsV2BatchSendRecvExecutor][Orchestrate] maxTmpMemSize equals to zero."),
        HCCL_E_PARA);
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    // 任务信息
    itemNum_ = param.batchSendRecvDataDes.itemNum;
    itemPtr_ = param.batchSendRecvDataDes.sendRecvItemsPtr;

    // 任务排序
    CHK_RET(GetPairWiseList(itemPtr_, itemNum_));
    // 执行自收发任务
    CHK_RET(ProcessSelfSendRecvTasks(threads_[0]));
    // 任务切片
    CHK_RET(CalcSendSlices());
    CHK_RET(CalcRecvSlices());
    // 循环执行任务
    CHK_RET(RunLoopSendRecv());

    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_IMPL(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, InsBatchSendRecv, InsV2BatchSendRecvExecutor);
}