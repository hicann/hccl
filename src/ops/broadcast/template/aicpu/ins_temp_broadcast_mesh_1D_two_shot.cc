/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/ins_temp_broadcast_mesh_1D_two_shot.h"

namespace ops_hccl {
InsTempBroadcastMesh1DTwoShot::InsTempBroadcastMesh1DTwoShot(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempBroadcastMesh1DTwoShot::~InsTempBroadcastMesh1DTwoShot()
{
}


HcclResult InsTempBroadcastMesh1DTwoShot::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, 
                                                  AlgResourceRequest& resourceRequest)
{
    // mesh 算法只做level 0 层级的
    GetRes(resourceRequest);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}


HcclResult InsTempBroadcastMesh1DTwoShot::GetRes(AlgResourceRequest &resourceRequest)
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] GetRes. myRank[%d] notifyNumOnMainThread[%d] rankSize[%d] threadNum[%d]", 
               myRank_, resourceRequest.notifyNumOnMainThread, templateRankSize_, threadNum);

    return HCCL_SUCCESS;
}

u64 InsTempBroadcastMesh1DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 1;
    return scratchMultiple;
}

u64 InsTempBroadcastMesh1DTwoShot::GetThreadNum()
{
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    return threadNum;
}

// 按照mesh的方式计算SliceInfo，例如N张卡，就是N份slice
HcclResult InsTempBroadcastMesh1DTwoShot::CalcDataSliceInfo(const u64 dataSize, RankSliceInfo &sliceInfoVec)
{
    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] CalcDataSliceInfo entry.");
    // 一般情况下，mesh的temp是单级的
    sliceInfoVec.resize(templateRankSize_);
    u64 chunkSize = RoundUp(dataSize, (templateRankSize_ * dataTypeSize_)) * dataTypeSize_;

    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot] CalcDataSliceInfo. myRank[%d] dataSize[%d] rankSize[%d] dataTypeSize[%d] chunkSize[%d]", 
               myRank_, dataSize, templateRankSize_, dataTypeSize_, chunkSize);
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64       currChunkSize  = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice          = {accumOff, currChunkSize};
        HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot] CalcDataSliceInfo. myRank[%d] offset[%d] dataSize[%d]", 
                    myRank_, accumOff, currChunkSize);
        sliceInfoVec[rankIdx].push_back(slice);
        accumOff += currChunkSize;
    }

    CHK_PRT_RET((sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size != dataSize),
                HCCL_ERROR("[InsTempBroadcastMesh1DTwoShot] Rank [%d], SliceInfo calculation error!", myRank_),
                HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

// 计算scatter的通信rank集合
HcclResult InsTempBroadcastMesh1DTwoShot::CalcCommRankSetforScatter(const u32 groupRankSize,
                                                                    std::vector<u32> &commRanks) const
{
    (void)groupRankSize;
    commRanks.clear();

    if (u32(myRank_) != root_) {
        commRanks.emplace_back(root_);
        return HcclResult::HCCL_SUCCESS;
    }

    for (auto& rankIter : tempVirtRankMap_) {
        if (u32(myRank_) != u32(rankIter.first)) {
            commRanks.emplace_back(u32(rankIter.first));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

// 计算allgather的通信rank集合
HcclResult InsTempBroadcastMesh1DTwoShot::CalcCommRankSetforAllGather(const u32 groupRankSize,
                                                                      std::vector<u32> &commRanks) const
{
    (void)groupRankSize;
    commRanks.clear();

    for (auto& rankIter : tempVirtRankMap_) {
        if (u32(myRank_) != u32(rankIter.first) && root_ != u32(rankIter.first)) {
            commRanks.emplace_back(u32(rankIter.first));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::RootSendData(const u64 memOffset,
                                             const u32 remoteRank,
                                             const TemplateDataParams &tempAlgParams,
                                             const std::vector<ThreadHandle> &threads,
                                             const u32 id, 
                                             const std::map<u32, std::vector<ChannelInfo>> &channels,
                                             const RankSliceInfo &sliceInfoVec) const
{
    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot]: RootSendData entry[%d].", myRank_);

    u32 myRankIdx = tempVirtRankMap_.at(myRank_);
    u32 remoteRankIdx = tempVirtRankMap_.at(remoteRank);
    // root执行常规scatter发送，将remoteRank的数据分片发送至remoteRank的buf中
    u64 sendSrcOffset0 = sliceInfoVec[remoteRankIdx][0].offset + memOffset;
    u64 sendDstOffset0 = sliceInfoVec[remoteRankIdx][0].offset;

    sendDstOffset0 += tempAlgParams.buffInfo.hcclBuffBaseOff;

    const ChannelInfo &linkSend = channels.at(remoteRank)[0];
    void* DstPtr = linkSend.remoteCclMem.addr;
 


    DataSlice sendSrcSlice0 = DataSlice(tempAlgParams.buffInfo.inputPtr, sendSrcOffset0, sliceInfoVec[remoteRankIdx][0].size);
    DataSlice sendDstSlice0 = DataSlice(DstPtr, sendDstOffset0, sliceInfoVec[remoteRankIdx][0].size);

    std::vector<DataSlice> sendSrcSliceVec0 = {sendSrcSlice0};
    std::vector<DataSlice> sendDstSliceVec0 = {sendDstSlice0};
    SlicesList sendDataSlice0(sendSrcSliceVec0, sendDstSliceVec0);
    DataInfo sendDataInfo0(linkSend, sendDataSlice0);
    CHK_RET(SendWrite(sendDataInfo0, threads[id]));


    // root将自己数据分片发送至对端
    u64 sendSrcOffset1 = sliceInfoVec[myRankIdx][0].offset + memOffset;
    u64 sendDstOffset1 = sliceInfoVec[myRankIdx][0].offset;


    sendDstOffset1 += tempAlgParams.buffInfo.hcclBuffBaseOff;

    DataSlice sendSrcSlice1 = DataSlice(tempAlgParams.buffInfo.inputPtr, sendSrcOffset1, sliceInfoVec[myRankIdx][0].size);
    DataSlice sendDstSlice1 = DataSlice(DstPtr, sendDstOffset1, sendDstOffset1);

    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot] RootSendData: sendSrcSlice1.myRank[%d] addr[%p] offset[%d] Size[%d]",
              myRank_, tempAlgParams.buffInfo.inputPtr, sendSrcOffset1, sliceInfoVec[myRankIdx][0].size);
    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot] RootSendData: sendSrcSlice1.myRank[%d] addr[%p] offset[%d] Size[%d]",
              myRank_, DstPtr, sendDstOffset1, sliceInfoVec[myRankIdx][0].size);

    std::vector<DataSlice> sendSrcSliceVec1 = {sendSrcSlice1};
    std::vector<DataSlice> sendDstSliceVec1 = {sendDstSlice1};
    SlicesList sendDataSlice1(sendSrcSliceVec1, sendDstSliceVec1);
    DataInfo sendDataInfo1(linkSend, sendDataSlice1);
    CHK_RET(SendWrite(sendDataInfo1, threads[id]));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::RankRecvData(const u64 memOffset,
                                             const u32 remoteRank,
                                             const TemplateDataParams &tempAlgParams,
                                             const std::vector<ThreadHandle> &threads,
                                             const u32 id, 
                                             const std::map<u32, std::vector<ChannelInfo>> &channels,
                                             const RankSliceInfo &sliceInfoVec) const
{
    u32 myRankIdx = tempVirtRankMap_.at(myRank_);
    u32 rootIdx = tempVirtRankMap_.at(root_);
    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot][RankRecvData],myRank_[%u], root_[%u], remoteRank[%u] ", myRank_, root_, remoteRank);
    for(auto i: channels) {
        HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot][RankRecvData],myRank_[%u], channels[%u]= size[%u] ", myRank_, i.first, i.second.size());
    }
    // 非root执行常规scatter接收，从root接收本rank的数据分片
    u64 sendSrcOffset0 = sliceInfoVec[myRankIdx][0].offset + memOffset;
    u64 sendDstOffset0 = sliceInfoVec[myRankIdx][0].offset;
    sendDstOffset0 += tempAlgParams.buffInfo.hcclBuffBaseOff;

    void *DstPtr = tempAlgParams.buffInfo.hcclBuff.addr;
    const ChannelInfo &linkRecv = channels.at(remoteRank)[0];
    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot][RankRecvData],myRank_[%u] resource end", myRank_);

    DataSlice recvSrcSlice0 = DataSlice(tempAlgParams.buffInfo.inputPtr, sendSrcOffset0, sliceInfoVec[myRankIdx][0].size);
    DataSlice recvDstSlice0 = DataSlice(DstPtr, sendDstOffset0, sliceInfoVec[myRankIdx][0].size);

    std::vector<DataSlice> recvSrcSliceVec0 = {recvSrcSlice0};
    std::vector<DataSlice> recvDstSliceVec0 = {recvDstSlice0};
    SlicesList recvDataSlice0(recvSrcSliceVec0, recvDstSliceVec0);
    DataInfo recvDataInfo0(linkRecv, recvDataSlice0);
    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot][RankRecvData],myRank_[%u] threads size[%u] idx[%u]", myRank_, threads.size(), id);
    CHK_RET(RecvWrite(recvDataInfo0, threads[id]));
    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot][RankRecvData],myRank_[%u] RecvWrite1 end", myRank_);

    // 非root接收root的数据分片
    u64 sendSrcOffset1 = sliceInfoVec[rootIdx][0].offset + memOffset;
    u64 sendDstOffset1 = sliceInfoVec[rootIdx][0].offset;
    sendDstOffset1 += tempAlgParams.buffInfo.hcclBuffBaseOff;


    DataSlice recvSrcSlice1 = DataSlice(tempAlgParams.buffInfo.inputPtr, sendSrcOffset1, sliceInfoVec[rootIdx][0].size);
    DataSlice recvDstSlice1 = DataSlice(DstPtr, sendDstOffset1, sliceInfoVec[rootIdx][0].size);

    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot] RankRecvData: recvSrcSlice1.myRank[%d] addr[%p] offset[%d] Size[%d]",
              myRank_, tempAlgParams.buffInfo.inputPtr, sendSrcOffset1, sliceInfoVec[rootIdx][0].size);
    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot] RankRecvData: recvDstSlice1.myRank[%d] addr[%p] offset[%d] Size[%d]",
              myRank_, DstPtr, sendDstOffset1, sliceInfoVec[rootIdx][0].size);

    std::vector<DataSlice> recvSrcSliceVec1= {recvSrcSlice1};
    std::vector<DataSlice> recvDstSliceVec1 = {recvDstSlice1};
    SlicesList recvDataSlice1(recvSrcSliceVec1, recvDstSliceVec1);
    DataInfo recvDataInfo1(linkRecv, recvDataSlice1);
    CHK_RET(RecvWrite(recvDataInfo1, threads[id]));
    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot][RankRecvData],myRank_[%u] end", myRank_);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::RunScatter(const std::vector<u32> &commRanks,
                                             const TemplateDataParams &tempAlgParams,
                                             const std::map<u32, std::vector<ChannelInfo>> &channels,
                                             const std::vector<ThreadHandle> &threads,
                                             const RankSliceInfo &sliceInfoVec)
{
    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot: Scatter entry.");
    // 主从流同步
    if (commRanks.size() > 1) {
        std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
    }
    u64 memOffset = tempAlgParams.buffInfo.inBuffBaseOff;

    // DMA消减，直接从root的inputbuf传输数据至对端buf
    for(u32 i = 0 ; i < commRanks.size(); i++) {
        u32 remoteRank = static_cast<s32>(commRanks[i]);
        if (u32(myRank_) == root_) {
            // root只发不收
            CHK_RET(RootSendData(memOffset, remoteRank, tempAlgParams, threads, i, channels, sliceInfoVec));
        } else {
            // 非root只收不发
            CHK_RET(RankRecvData(memOffset, remoteRank, tempAlgParams, threads, i, channels, sliceInfoVec));
        }
    }

    // 主从流同步
    if (commRanks.size() > 1) {
        std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot: Scatter finish.");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::RunAllGather(const std::vector<u32> &commRanks,
                                             const TemplateDataParams &tempAlgParams,
                                             const std::map<u32, std::vector<ChannelInfo>> &channels,
                                             const std::vector<ThreadHandle> &threads,
                                             const RankSliceInfo &sliceInfoVec)
{
    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot: AllGather entry.");

    if (commRanks.size() > 1) {
        std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_));
    }

    for(u32 i = 0 ; i < commRanks.size(); i++) {
        s32 remoteRank = static_cast<s32>(commRanks[i]);


        u32 myRankIdx = tempVirtRankMap_.at(myRank_);
        u32 remoteRankIdx = tempVirtRankMap_.at(remoteRank);

        u64 sendSrcOffset = sliceInfoVec[myRankIdx][0].offset;
        u64 sendDstOffset = sliceInfoVec[myRankIdx][0].offset;
        u64 recvSrcOffset = sliceInfoVec[remoteRankIdx][0].offset;
        u64 recvDstOffset = sliceInfoVec[remoteRankIdx][0].offset;

        sendSrcOffset += tempAlgParams.buffInfo.hcclBuffBaseOff;
        recvSrcOffset += tempAlgParams.buffInfo.hcclBuffBaseOff;
        sendDstOffset += tempAlgParams.buffInfo.hcclBuffBaseOff;
        recvDstOffset += tempAlgParams.buffInfo.hcclBuffBaseOff;


        void *SrcPtr = tempAlgParams.buffInfo.hcclBuff.addr;
        void *DstPtr = tempAlgParams.buffInfo.hcclBuff.addr;

        const ChannelInfo &linkSendRecv = channels.at(remoteRank)[0];
        void* remoteDstPtr = linkSendRecv.remoteCclMem.addr;


        DataSlice sendSrcSlice = DataSlice(SrcPtr, sendSrcOffset, sliceInfoVec[myRankIdx][0].size);
        DataSlice sendDstSlice = DataSlice(remoteDstPtr, sendDstOffset, sliceInfoVec[myRankIdx][0].size);
        std::vector<DataSlice> sendSrcSliceVec = {sendSrcSlice};
        std::vector<DataSlice> sendDstSliceVec = {sendDstSlice};
        SlicesList sendDataSlice(sendSrcSliceVec, sendDstSliceVec);

        DataSlice recvSrcSlice = DataSlice(SrcPtr, recvSrcOffset, sliceInfoVec[remoteRankIdx][0].size);
        DataSlice recvDstSlice = DataSlice(DstPtr, recvDstOffset, sliceInfoVec[remoteRankIdx][0].size);
        std::vector<DataSlice> recvSrcSliceVec = {recvSrcSlice};
        std::vector<DataSlice> recvDstSliceVec = {recvDstSlice};
        SlicesList recvDataSlice(recvSrcSliceVec, recvDstSliceVec);

        TxRxSlicesList sendRecvSlice(sendDataSlice, recvDataSlice);
        TxRxChannels  sendRecvLinks(linkSendRecv, linkSendRecv);

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlice);
        CHK_RET(SendRecvWrite(sendRecvInfo, threads[i]));
    }

    if (commRanks.size() > 1) {
        std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_));
    }

    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot: AllGather finish.");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastMesh1DTwoShot::PostCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    u64 inOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;

    DataSlice usrInSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, inOffset, tempAlgParams.sliceSize, tempAlgParams.count);
    DataSlice usrOutSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.outBuffBaseOff,
                tempAlgParams.sliceSize, tempAlgParams.count);


    CHK_RET(LocalCopy(threads[0], usrInSlice, usrOutSlice));

    return HcclResult::HCCL_SUCCESS;
}

void InsTempBroadcastMesh1DTwoShot::SetRoot(u32 root)
{
    root_ = root;
    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root_);
}

HcclResult InsTempBroadcastMesh1DTwoShot::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
                                          const TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot entry.");
    HCCL_DEBUG("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot rank[%d] slicesize[%d] count[%d].",
               myRank_, tempAlgParams.sliceSize, tempAlgParams.count);
    dataType_ = param.DataDes.dataType;
    dataTypeSize_  = tempAlgParams.sliceSize/tempAlgParams.count;

    for (int i = 0; i < subCommRanks_[0].size(); i++) {
        tempVirtRankMap_.insert(std::make_pair(subCommRanks_[0][i], i));
    }

    RankSliceInfo sliceInfoVec{};
    CHK_RET(CalcDataSliceInfo(tempAlgParams.sliceSize, sliceInfoVec));

    threadNum_ = templateResource.threads.size();

    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot Run]RankID:[%d], root:[%u]", myRank_, root_);

    std::vector<u32> scatterCommRanks;
    CHK_RET(CalcCommRankSetforScatter(templateRankSize_, scatterCommRanks));  // 计算scatter步骤的通信对象
    CHK_RET(RunScatter(scatterCommRanks, tempAlgParams, templateResource.channels, templateResource.threads, sliceInfoVec)); // 运行scatter步骤

    if (u32(myRank_) != root_) {
        std::vector<u32> allgatherCommRanks;
        CHK_RET(CalcCommRankSetforAllGather(templateRankSize_, allgatherCommRanks)); // 计算allgather步骤的通信对象
        CHK_RET(RunAllGather(allgatherCommRanks, tempAlgParams, templateResource.channels, templateResource.threads, sliceInfoVec)); // 运行allgather步骤
    }

    // 单算子模式
    if (u32(myRank_) != root_){
        CHK_RET(PostCopy(tempAlgParams, templateResource.threads));
    }

    HCCL_INFO("[InsTempBroadcastMesh1DTwoShot] BroadcastMesh1DTwoShot finish.");

    return HcclResult::HCCL_SUCCESS;
}

void InsTempBroadcastMesh1DTwoShot::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempBroadcastMesh1DTwoShot::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace Hccl

