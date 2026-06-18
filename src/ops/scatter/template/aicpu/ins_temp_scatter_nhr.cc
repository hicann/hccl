/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_nhr.h"

namespace ops_hccl {
InsTempScatterNHR::InsTempScatterNHR(const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks)
                                        : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempScatterNHR::~InsTempScatterNHR()
{}

u64 InsTempScatterNHR::GetThreadNum() const
{
    u64 threadNum = channelsPerRank_;
    return threadNum;
}

void InsTempScatterNHR::SetRoot(u32 root)
{   
    HCCL_INFO("[InsTempScatterNHR][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root);
    root_ = root;
}

HcclResult InsTempScatterNHR::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                        AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);

    std::vector<HcclChannelDesc> level0Channels;
    std::vector<HcclChannelDesc> myChannelDescs;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        CHK_RET(CalcChannelRequestNHRWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_CLOS));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                level0Channels.push_back(channel);
            }
        }
    } else {
        CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    channelsPerRank_ = CalcChannelsPerRank(level0Channels);
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::GetRes(AlgResourceRequest &resourceRequest) const
{
    u32 threadNum = channelsPerRank_;
    resourceRequest.slaveThreadNum = threadNum - 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempScatterNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    return templateRankSize_;
}

HcclResult InsTempScatterNHR::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo)
{
    u32 rankSize = templateRankSize_;
    u32 myAlgRank;
    u32 rootAlgRank;
    HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] root_ before getAlgRank [%u] ", root_);
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] myRank_[%u], myAlgRank[%u]", myRank_, myAlgRank);
    GetAlgRank(root_, subCommRanks_[0], rootAlgRank);
    HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] root_[%u], rootAlgRank[%u]", root_, rootAlgRank);
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.step = step;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    stepInfo.myRank = myRank_;

    u32 deltaRoot = (rootAlgRank + rankSize - myAlgRank) % rankSize;
    u32 deltaRankPair = 1 << step;

    // ScatterNHR 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    // ScatterNHR 判断是否是2的幂
    u32 nRanks = 0;  // 本步需要进行收/发的rank数
    bool isPerfect = (rankSize & (rankSize - 1)) == 0;
    if (!isPerfect && step == nSteps - 1) {
        nRanks = rankSize - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) {  // ScatterNHR 需要发
        HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] Need to Send: deltaRoot[%u], nRanks[%d]", deltaRoot, nRanks);
        u32 sendTo = (myAlgRank + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
        HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] rankSize[%u], myAlgRank[%d], sendTo Idx[%u]", subCommRanks_[0].size(), myAlgRank, sendTo);
        stepInfo.toRank = subCommRanks_[0].at(sendTo);
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) {  // 需要收
        HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] Need to Recv: deltaRoot[%u], nRanks[%d], deltaRankPair[%d]", deltaRoot, nRanks, deltaRankPair);
        u32 recvFrom = (myAlgRank + deltaRankPair) % rankSize;
        u32 rxSliceIdx = myAlgRank;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
        HCCL_DEBUG("[InsTempScatterNHR][GetStepInfo] rankSize[%u], myAlgRank[%d], recvFrom Idx[%u]", subCommRanks_[0].size(), myAlgRank, recvFrom);
        stepInfo.fromRank = subCommRanks_[0].at(recvFrom);
        stepInfo.nSlices = nSlices;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::PreprareDataSplitForMultiChannel(const TemplateResource &templateResource, const TemplateDataParams &tempAlgParams) {
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u64 totalDataCount = tempAlgParams.sliceSize / dataTypeSize;
    std::vector<u64> elemCountOut;
    CHK_RET(CalcDataSplitByPortGroup(totalDataCount, dataTypeSize, templateResource.channels.begin()->second, elemCountOut, dataSplit_, dataOffset_));
    if (tempAlgParams.tailSize > 0) {
        u64 totalDataCountTail = tempAlgParams.tailSize / dataTypeSize;
        CHK_RET(CalcDataSplitByPortGroup(totalDataCountTail, dataTypeSize, templateResource.channels.begin()->second, elemCountOut, dataSplitTail_, dataOffsetTail_));
    }
    
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::KernelRun(const OpParam& param, const TemplateDataParams &tempAlgParams,
                     TemplateResource& templateResource)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    threadNum_ =  GetThreadNum();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataType_ = param.DataDes.dataType;

    bool isPcieProtocal = IsPcieProtocol(templateResource.channels);  // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocal;  // 是否使用Read模式
    HCCL_DEBUG("[InsTempScatterNHR] Use Dma Read[%d]", isDmaRead_);
    CHK_RET(PreprareDataSplitForMultiChannel(templateResource, tempAlgParams));
    
    HCCL_INFO("[InsTempScatterNHR] queNum_ =  [%d], threads size = [%d]", threadNum_, templateResource.threads.size());
    HCCL_INFO("[InsTempScatterNHR] Run Start");
    CHK_PRT_RET(templateResource.threads.empty(), 
 	            HCCL_ERROR("[InsTempScatterNHR][KernelRun] threads is empty"), 
 	            HCCL_E_INTERNAL);
    CHK_PRT_RET(threadNum_ > templateResource.threads.size(),
        HCCL_ERROR("[InsTempScatterNHR] Rank [%d], requiredThread Error.", myRank_),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PTR_NULL(tempAlgParams.buffInfo.hcclBuff.addr);
    CHK_PTR_NULL(tempAlgParams.buffInfo.inputPtr);
    CHK_PTR_NULL(tempAlgParams.buffInfo.outputPtr);
    CHK_RET(PreCopy(tempAlgParams, templateResource.threads));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunNHR(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.begin() + threadNum_);
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));
    HCCL_INFO("[InsTempScatterNHR] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::PreCopy(const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads) const
{
    if (u32(myRank_) != root_ || tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        HCCL_INFO("[InsTempScatterNHR][Precopy] skip precopy, myRank = %u, root = %u", myRank_, root_);
        return HCCL_SUCCESS;
    }

    u32 curSliceSize = 0;
    u32 curCount = 0;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        for (u32 algRank = 0; algRank < templateRankSize_; algRank++) {
            u64 srcOffset = r * tempAlgParams.inputRepeatStride + tempAlgParams.inputSliceStride * algRank +
                            tempAlgParams.buffInfo.inBuffBaseOff;
            u64 dstOffset = r * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff +
                            algRank * tempAlgParams.sliceSize;
            
            curSliceSize = tempAlgParams.tailSize !=0 && algRank == templateRankSize_ - 1? tempAlgParams.tailSize : processSize_;
            curCount = curSliceSize / dataTypeSize;
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, curSliceSize, curCount);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, curSliceSize, curCount);
            CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads) const
{
    u32 myAlgRank;
    GetAlgRank(myRank_, subCommRanks_[0], myAlgRank);
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u32 curSliceSize = tempAlgParams.tailSize !=0 && myAlgRank == templateRankSize_ - 1? tempAlgParams.tailSize : processSize_;
    u32 curCount = curSliceSize / dataTypeSize;
    for (u32 r = 0; r < tempAlgParams.repeatNum; r++) {
        u64 srcOffset = r * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff +
                        myAlgRank * tempAlgParams.sliceSize;
        u64 dstOffset = tempAlgParams.buffInfo.outBuffBaseOff + myAlgRank * tempAlgParams.outputSliceStride + r * tempAlgParams.sliceSize;
        DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, curSliceSize, curCount);
        DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, curSliceSize, curCount);
        if ((tempAlgParams.buffInfo.outBuffType == BufferType::HCCL_BUFFER && srcOffset == dstOffset) || enableRemoteMemAccess_) {
            continue;
        }
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads.at(0), srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::RunNHR(const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    // nhr主体部分
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    HCCL_DEBUG("[RunNHR] root_ at RunNHR [%u] ", root_);
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        for (u32 step = 0; step < nSteps; step++) {
            AicpuNHRStepInfo stepInfo;
            GetStepInfo(step, nSteps, stepInfo);
            // 只有Tx,使用send指令
            if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() == 0) {
                CHK_RET(BatchSend(stepInfo, channels, threads.at(channelIdx), tempAlgParams, channelIdx));
            }
            // 只有Rx，使用recv指令
            else if (stepInfo.txSliceIdxs.size() == 0 && stepInfo.rxSliceIdxs.size() > 0) {
                CHK_RET(BatchRecv(stepInfo, channels, threads.at(channelIdx), tempAlgParams, channelIdx));
            }
            // 既有Tx又有Rx，使用SendRecv指令
            else if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() > 0) {
                CHK_RET(BatchSR(stepInfo, channels, threads.at(channelIdx), tempAlgParams, channelIdx));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::BatchSend(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels,
    const ThreadHandle &thread, const TemplateDataParams &tempAlgParams, u32 channelIdx) const
{
    CHK_PRT_RET(channels.find(stepInfo.toRank) == channels.end() || channels.at(stepInfo.toRank).empty(), 
 	             HCCL_ERROR("[InsTempScatterNHR][BatchSend] remoteRank[%d] not found in channels", stepInfo.toRank), 
 	             HCCL_E_INTERNAL);
    const ChannelInfo &linkSend = channels.at(stepInfo.toRank)[channelIdx];
    CHK_PTR_NULL(linkSend.remoteCclMem.addr);
    HCCL_INFO("[InsTempScatterNHR][BatchSend] myRank[%d], toRank[%d]", myRank_, stepInfo.toRank);
    void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;
    u32 curSliceSize = 0;
    u32 curCount = 0;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 repeat = 0; repeat < tempAlgParams.repeatNum; repeat++) {
        for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
            u32 txId = stepInfo.txSliceIdxs.at(i);
            const u64 txPartialOffset = (txId == templateRankSize_ - 1 && tempAlgParams.tailSize != 0) ? dataOffsetTail_[channelIdx]: dataOffset_[channelIdx];
            u64 srcDstOffset = repeat * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff +
                            txId * tempAlgParams.sliceSize + txPartialOffset;
            curSliceSize = (tempAlgParams.tailSize !=0 && txId == templateRankSize_ - 1) ? dataSplitTail_[channelIdx]: dataSplit_[channelIdx];
            curCount = curSliceSize / dataTypeSize;
            DataSlice srcSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcDstOffset, curSliceSize, curCount);
            DataSlice dstSlice = DataSlice(remoteCclBuffAddr, srcDstOffset, curSliceSize, curCount);
            srcSlices.push_back(srcSlice);
            dstSlices.push_back(dstSlice);
        }
    }
    SlicesList txSlicesList({srcSlices}, {dstSlices});
    DataInfo sendData(linkSend, txSlicesList);
    if (isDmaRead_) {
        CHK_PRT_RET(static_cast<HcclResult>(SendRead(sendData, thread)),
            HCCL_ERROR("[InsTempScatterNHR] BatchSend failed"),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(static_cast<HcclResult>(SendWrite(sendData, thread)),
            HCCL_ERROR("[InsTempScatterNHR] BatchSend failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::BatchRecv(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels,
    const ThreadHandle &thread, const TemplateDataParams &tempAlgParams, u32 channelIdx) const
{
    CHK_PRT_RET(channels.find(stepInfo.fromRank) == channels.end() || channels.at(stepInfo.fromRank).empty(), 
                HCCL_ERROR("[InsTempScatterNHR][BatchRecv] remoteRank[%d] not found in channels", stepInfo.fromRank), 
                HCCL_E_INTERNAL);
    const ChannelInfo &linkRecv = channels.at(stepInfo.fromRank)[channelIdx];
    CHK_PTR_NULL(linkRecv.remoteCclMem.addr);
    void* remoteCclBuffAddr = linkRecv.remoteCclMem.addr;
    std::vector<DataSlice> srcSlices;
    std::vector<DataSlice> dstSlices;
    u32 curSliceSize = 0;
    u32 curCount = 0;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 repeat = 0; repeat < tempAlgParams.repeatNum; repeat++) {
        for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
            u32 rxId = stepInfo.rxSliceIdxs.at(i);
            const u64 rxPartialOffset = (rxId == templateRankSize_ - 1 && tempAlgParams.tailSize != 0) ? dataOffsetTail_[channelIdx]: dataOffset_[channelIdx];
            u64 srcDstOffset = repeat * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff +
                                rxId * tempAlgParams.sliceSize + rxPartialOffset;
            curSliceSize = (tempAlgParams.tailSize !=0 && rxId == templateRankSize_ - 1) ? dataSplitTail_[channelIdx]: dataSplit_[channelIdx];
            curCount = curSliceSize / dataTypeSize;
            DataSlice srcSlice = DataSlice(remoteCclBuffAddr, srcDstOffset, curSliceSize, curCount);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcDstOffset, curSliceSize, curCount);
            srcSlices.push_back(srcSlice);
            dstSlices.push_back(dstSlice);
        }
    }
    SlicesList rxSlicesList({srcSlices}, {dstSlices});
    DataInfo recvData(linkRecv, rxSlicesList);
    if (isDmaRead_) {
        CHK_PRT_RET(static_cast<HcclResult>(RecvRead(recvData, thread)),
            HCCL_ERROR("[InsTempScatterNHR] BatchRecv failed"),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(static_cast<HcclResult>(RecvWrite(recvData, thread)),
            HCCL_ERROR("[InsTempScatterNHR] BatchRecv failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterNHR::BatchSR(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels,
    const ThreadHandle &thread, const TemplateDataParams &tempAlgParams, u32 channelIdx) const
{
    CHK_PRT_RET(channels.find(stepInfo.fromRank) == channels.end() || channels.at(stepInfo.fromRank).empty(), 
 	            HCCL_ERROR("[InsTempScatterNHR][BatchSR] remoteRank[%d] not found in channels", stepInfo.fromRank), 
 	            HCCL_E_INTERNAL);
    CHK_PRT_RET(channels.find(stepInfo.toRank) == channels.end() || channels.at(stepInfo.toRank).empty(), 
                HCCL_ERROR("[InsTempScatterNHR][BatchSend] remoteRank[%d] not found in channels", stepInfo.toRank), 
                HCCL_E_INTERNAL);
    const ChannelInfo &linkSend = channels.at(stepInfo.toRank)[channelIdx];
    const ChannelInfo &linkRecv = channels.at(stepInfo.fromRank)[channelIdx];
    CHK_PTR_NULL(linkSend.remoteCclMem.addr);
    CHK_PTR_NULL(linkRecv.remoteCclMem.addr);
    void* sendRemoteCclBuffAddr = linkSend.remoteCclMem.addr;
    void* recvRemoteCclBuffAddr = linkRecv.remoteCclMem.addr;

    u32 curSliceSize = 0;
    u32 curCount = 0;
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    for (u32 repeat = 0; repeat < tempAlgParams.repeatNum; repeat++) {
        for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
            u32 txId = stepInfo.txSliceIdxs.at(i);
            const u64 txPartialOffset = (txId == templateRankSize_ - 1 && tempAlgParams.tailSize != 0) ? dataOffsetTail_[channelIdx]: dataOffset_[channelIdx];
            u64 srcDstOffset = repeat * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff +
                            txId * tempAlgParams.sliceSize + txPartialOffset;
            curSliceSize = (tempAlgParams.tailSize !=0 && txId == templateRankSize_ - 1) ? dataSplitTail_[channelIdx]: dataSplit_[channelIdx];
            curCount = curSliceSize / dataTypeSize;
            DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcDstOffset, curSliceSize, curCount);
            DataSlice dstSlice(sendRemoteCclBuffAddr, srcDstOffset, curSliceSize, curCount);
            txSrcSlices.push_back(srcSlice);
            txDstSlices.push_back(dstSlice);
        }
        for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
            u32 rxId = stepInfo.rxSliceIdxs.at(i);
            const u64 rxPartialOffset = (rxId == templateRankSize_ - 1 && tempAlgParams.tailSize != 0) ? dataOffsetTail_[channelIdx]: dataOffset_[channelIdx];
            u64 srcDstOffset = repeat * templateRankSize_ * tempAlgParams.sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff +
                            rxId * tempAlgParams.sliceSize + rxPartialOffset;
            curSliceSize = (tempAlgParams.tailSize !=0 && rxId == templateRankSize_ - 1) ? dataSplitTail_[channelIdx]: dataSplit_[channelIdx];
            curCount = curSliceSize / dataTypeSize;
            DataSlice srcSlice = DataSlice(recvRemoteCclBuffAddr, srcDstOffset, curSliceSize, curCount);
            DataSlice dstSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcDstOffset, curSliceSize, curCount);
            rxSrcSlices.push_back(srcSlice);
            rxDstSlices.push_back(dstSlice);
        }
    }
    SendRecvInfo sendRecvInfo{
        {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};

    if (isDmaRead_) {
        CHK_PRT_RET(SendRecvRead(sendRecvInfo, thread),
            HCCL_ERROR("[InsTempScatterNHR] RunNHR BatchSendRecv failed"),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread),
            HCCL_ERROR("[InsTempScatterNHR] RunNHR BatchSendRecv failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}
void InsTempScatterNHR::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = GetThreadNum();
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempScatterNHR::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = GetThreadNum();
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}
}  // namespace ops_hccl