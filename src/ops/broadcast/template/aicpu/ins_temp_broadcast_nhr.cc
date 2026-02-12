/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_broadcast_nhr.h"

namespace ops_hccl {
InsTempBroadcastNHR::InsTempBroadcastNHR(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

InsTempBroadcastNHR::~InsTempBroadcastNHR()
{
}


HcclResult InsTempBroadcastNHR::CalcRes(HcclComm comm, const OpParam& param, const TopoInfo* topoInfo,
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

HcclResult InsTempBroadcastNHR::GetRes(AlgResourceRequest &resourceRequest)
{
    u32 threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

u64 InsTempBroadcastNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 1;
    return scratchMultiple;
}

u64 InsTempBroadcastNHR::GetThreadNum()
{
    u32 threadNum = 1;
    return threadNum;
}

// SliceInfoVec for NHR
HcclResult InsTempBroadcastNHR::CalcDataSliceInfo(const u64 dataSize, RankSliceInfo &sliceInfoVec)
{

    sliceInfoVec.clear();
    sliceInfoVec.resize(templateRankSize_);
    u64 chunkSize = RoundUp(dataSize, (templateRankSize_ * dataTypeSize_)) * dataTypeSize_;

    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64       currChunkSize  = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice          = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx].push_back(slice);
        accumOff += currChunkSize;
    }

    CHK_PRT_RET((sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size != dataSize),
                HCCL_ERROR("[InsTempBroadcastNHR] Rank [%d], SliceInfo calculation error!", myRank_),
                HcclResult::HCCL_E_INTERNAL);

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::PostCopy(const TemplateDataParams &tempAlgParams,
                                            const std::vector<ThreadHandle> &threads) const
{
    if (u32(myRank_) != root_) {
        HCCL_INFO("[InsTempBroadcastNHR][PostCopy] Opbase && isBottom, copy from outBuff to userOut");
        u64 inOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;

        DataSlice usrInSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, inOffset, tempAlgParams.sliceSize, tempAlgParams.count);
        DataSlice usrOutSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.outBuffBaseOff,
                    tempAlgParams.sliceSize, tempAlgParams.count);


        CHK_RET(LocalCopy(threads[0], usrInSlice, usrOutSlice));
    } else {
        HCCL_INFO("[InsTempBroadcastNHR][PostCopy] Offload Model, skip postcopy");
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::PreCopy(const TemplateDataParams &tempAlgParams,
                                            const std::vector<ThreadHandle> &threads) const
{
    if (u32(myRank_) == root_){
            DataSlice usrInSlice = DataSlice(tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff,
                    tempAlgParams.sliceSize, tempAlgParams.count);
            DataSlice usrOutSlice = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff,
                    tempAlgParams.sliceSize, tempAlgParams.count);
            CHK_PRT_RET(LocalCopy(threads[0], usrInSlice, usrOutSlice),
                HCCL_ERROR("[InsTempBroadcastNHR] RunScatter userIn to cclIn copy failed"),

            HcclResult::HCCL_E_INTERNAL);
    } else {
        HCCL_INFO("[InsTempBroadcastNHR][PostCopy] Offload Model, skip postcopy");
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::GetAllGatherStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo)
{
    u32 rankIdx = tempVirtRankMap_[myRank_];
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    // 计算通信对象
    u32 deltaRank = 1 << (nSteps - 1 - step);
    u32 recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    u32 sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    u32 nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    u32 deltaSliceIndex = 1 << (nSteps - step);
    u32 txSliceIdx = rankIdx;
    u32 rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.toRank = sendTo;
    stepInfo.fromRank = recvFrom;

    HCCL_DEBUG("[InsTempBroadcastNHR][GetAllGatherStepInfo] myRank_[%u] toRank[%u] fromRank[%u] nSteps[%u], step[%u], rankIdx[%u]",
            myRank_, sendTo, recvFrom, nSteps, step, rankIdx);
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG("[InsTempBroadcastNHR][GetAllGatherStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]",
            i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

// NHR每步的算法描述原理函数
HcclResult InsTempBroadcastNHR::GetScatterStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo) const
{
    u32 rankIdx = tempVirtRankMap_.at(myRank_);
    u32 rootIdx = tempVirtRankMap_.at(root_);
    u32 rankSize = templateRankSize_;

    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    stepInfo.nSlices = 0;
    stepInfo.toRank = INVALID_U32;
    stepInfo.fromRank = INVALID_U32;
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;

    u32 deltaRoot = (rootIdx + rankSize - rankIdx) % rankSize;
    u32 deltaRankPair = 1 << step;

    // 数据份数和数据编号增量
    u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
    u32 deltaSliceIndex = 1 << (step + 1);

    // 判断是否是2的幂
    u32 nRanks = 0; // 本步需要进行收/发的rank数
    bool isPerfect = (rankSize & (rankSize - 1)) == 0;
    if (!isPerfect && step == nSteps - 1) {
        nRanks = rankSize - deltaRankPair;
    } else {
        nRanks = deltaRankPair;
    }

    if (deltaRoot < nRanks) { // 需要发
        u32 sendTo = (rankIdx + rankSize - deltaRankPair) % rankSize;
        u32 txSliceIdx = sendTo;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetTxSliceIdx = txSliceIdx;
            stepInfo.txSliceIdxs.push_back(targetTxSliceIdx);
            txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.toRank = sendTo;
        stepInfo.nSlices = nSlices;
    } else if (deltaRoot >= deltaRankPair && deltaRoot < nRanks + deltaRankPair) { // 需要收
        u32 recvFrom = (rankIdx + deltaRankPair) % rankSize;
        u32 rxSliceIdx = rankIdx;
        for (u32 i = 0; i < nSlices; i++) {
            u32 targetRxSliceIdx = rxSliceIdx;
            stepInfo.rxSliceIdxs.push_back(targetRxSliceIdx);
            rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }

        stepInfo.fromRank = recvFrom;
        stepInfo.nSlices = nSlices;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::RunScatter(const RankSliceInfo &sliceInfoVec,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    // nhr主体部分,从ScratchIn计算，结果放至ScratchOut上, 该部分均从inType搬运到outType
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetScatterStepInfo(step, nSteps, stepInfo));
        HCCL_INFO("[InsTempBroadcastNHR]RunScatter GetScatterStepInfo after:[%d], root:[%u]", myRank_, root_);
        CHK_PRT_RET(BatchTxRx(stepInfo, channels, threads, sliceInfoVec),
                HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx failed"),
                HcclResult::HCCL_E_INTERNAL);
    }

    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempBroadcastNHR::GetRankFromMap(const u32 rankIdx) const
{
    u32 rank = -1;
    for (auto &pair : tempVirtRankMap_) {
        if (pair.second == rankIdx) {
            rank = pair.first;
            break;
        }
    }
    return rank;
}

HcclResult InsTempBroadcastNHR::RunAllGather(const RankSliceInfo &sliceInfoVec,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads)
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);

    u64 memOffset = buffInfo_.hcclBuffBaseOff;

    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetAllGatherStepInfo(step, nSteps, stepInfo));

        const ChannelInfo &linkRecv = channels.at(GetRankFromMap(stepInfo.fromRank))[0];
        const ChannelInfo &linkSend = channels.at(GetRankFromMap(stepInfo.toRank))[0];

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        HCCL_DEBUG("[InsTempBroadcastNHR] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
            myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u64 txOffset   = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].offset + memOffset;
            u64 txSize     = sliceInfoVec[stepInfo.txSliceIdxs[i]][0].size;
            u64 rxOffset   = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].offset + memOffset;
            u64 rxSize     = sliceInfoVec[stepInfo.rxSliceIdxs[i]][0].size;
            void* remoteSendCclBuffAddr = linkSend.remoteCclMem.addr;
            void* remoteRecvCclBuffAddr = linkRecv.remoteCclMem.addr;
            DataSlice txSrcSlice = DataSlice(buffInfo_.hcclBuff.addr, txOffset, txSize);
            DataSlice txDstSlice = DataSlice(remoteSendCclBuffAddr, txOffset, txSize);
            DataSlice rxSrcSlice = DataSlice(remoteRecvCclBuffAddr, rxOffset, rxSize);
            DataSlice rxDstSlice = DataSlice(buffInfo_.hcclBuff.addr, rxOffset, rxSize);
            txSrcSlices.push_back(txSrcSlice);
            txDstSlices.push_back(txDstSlice);
            rxSrcSlices.push_back(rxSrcSlice);
            rxDstSlices.push_back(rxDstSlice);
        }

        TxRxChannels sendRecvLinks(linkSend, linkRecv);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList);
        CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[0]), HCCL_ERROR("[InsTempBroadcastNHR] RunAllGather send failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

// Send multiple DataSlices
HcclResult InsTempBroadcastNHR::BatchTxRx(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const RankSliceInfo &sliceInfoVec)
{
    HCCL_INFO("[InsTempBroadcastNHR]BatchTxRx entry:[%d], root:[%u]", myRank_, root_);
    u64 memOffset = buffInfo_.hcclBuffBaseOff;
    // 只有Tx,使用send指令
    if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() == 0) {
        CHK_RET(BatchSend(stepInfo, channels, threads, sliceInfoVec, memOffset));
    }
    // 只有Rx，使用recv指令
    else if (stepInfo.txSliceIdxs.size() == 0 && stepInfo.rxSliceIdxs.size() > 0) {
        CHK_RET(BatchRecv(stepInfo, channels, threads, sliceInfoVec, memOffset));
    }
    // 既有Tx又有Rx，使用SendRecv指令
    else if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() > 0) {
        CHK_RET(BatchSR(stepInfo, channels, threads, sliceInfoVec, memOffset));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::BatchSend(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const RankSliceInfo &sliceInfoVec, u64 memOffset) const
{
    const ChannelInfo &linkSend = channels.at(GetRankFromMap(stepInfo.toRank))[0];
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs[i];
        void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;
        DataSlice txSrcSlice = DataSlice(buffInfo_.hcclBuff.addr, memOffset + sliceInfoVec[txId][0].offset, sliceInfoVec[txId][0].size);
        DataSlice txDstSlice = DataSlice(remoteCclBuffAddr, memOffset + sliceInfoVec[txId][0].offset, sliceInfoVec[txId][0].size);
        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
    }
    SlicesList txSlicesList(txSrcSlices, txDstSlices);
    DataInfo sendData(linkSend, txSlicesList);
    CHK_PRT_RET(SendWrite(sendData, threads[0]), HCCL_ERROR("[InsTempBroadcastNHR] BatchSend failed"),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::BatchRecv(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const RankSliceInfo &sliceInfoVec, u64 memOffset) const
{
    HCCL_INFO("[InsTempBroadcastNHR]BatchRecv entry:[%d], root:[%u]", myRank_, root_);
    const ChannelInfo &linkRecv = channels.at(GetRankFromMap(stepInfo.fromRank))[0];
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs[i];
        void* remoteCclBuffAddr = linkRecv.remoteCclMem.addr;
        DataSlice rxSrcSlice = DataSlice(remoteCclBuffAddr, memOffset + sliceInfoVec[rxId][0].offset, sliceInfoVec[rxId][0].size);
        DataSlice rxDstSlice = DataSlice(buffInfo_.hcclBuff.addr, memOffset + sliceInfoVec[rxId][0].offset, sliceInfoVec[rxId][0].size);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);
    }
    SlicesList rxSlicesList(rxSrcSlices, rxDstSlices);
    DataInfo recvData(linkRecv, rxSlicesList);
    CHK_PRT_RET(RecvWrite(recvData, threads[0]), HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx Recv failed"),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::BatchSR(AicpuNHRStepInfo &stepInfo, const std::map<u32, std::vector<ChannelInfo>> &channels, const std::vector<ThreadHandle> &threads,
    const RankSliceInfo &sliceInfoVec, u64 memOffset)const
{

    const ChannelInfo &linkSend = channels.at(GetRankFromMap(stepInfo.toRank))[0];
    const ChannelInfo &linkRecv = channels.at(GetRankFromMap(stepInfo.fromRank))[0];
    TxRxChannels linkSendRecv = {linkSend, linkRecv};

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs[i];
        void* remoteSendCclBuffAddr = linkSend.remoteCclMem.addr;
        DataSlice txSrcSlice = DataSlice(buffInfo_.hcclBuff.addr, memOffset + sliceInfoVec[txId][0].offset, sliceInfoVec[txId][0].size);
        DataSlice txDstSlice = DataSlice(remoteSendCclBuffAddr, memOffset + sliceInfoVec[txId][0].offset, sliceInfoVec[txId][0].size);
        txSrcSlices.push_back(txSrcSlice);
        txSrcSlices.push_back(txDstSlice);
    }
    SlicesList txSlicesList(txSrcSlices, txSrcSlices);
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs[i];
        void* remoteRecvCclBuffAddr = linkRecv.remoteCclMem.addr;
        DataSlice rxSrcSlice = DataSlice(remoteRecvCclBuffAddr, memOffset + sliceInfoVec[rxId][0].offset, sliceInfoVec[rxId][0].size);
        DataSlice rxDstSlice = DataSlice(buffInfo_.hcclBuff.addr, memOffset + sliceInfoVec[rxId][0].offset, sliceInfoVec[rxId][0].size);
        rxSrcSlices.push_back(rxSrcSlice);              
        rxDstSlices.push_back(rxDstSlice);
    }
    SlicesList rxSlicesList(rxSrcSlices, rxDstSlices);
    TxRxSlicesList txRxSlicesList(txSlicesList, rxSlicesList);
    SendRecvInfo sendRecvInfo(linkSendRecv, txRxSlicesList);
    CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[0]), HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx SendRecv failed"),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

void InsTempBroadcastNHR::SetRoot(u32 root)
{
    root_ = root;
    HCCL_INFO("[InsTempBroadcastNHR][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root_);
}

HcclResult InsTempBroadcastNHR::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
                                          const TemplateResource& templateResource)
{
    buffInfo_     = tempAlgParams.buffInfo;
    dataTypeSize_ = tempAlgParams.sliceSize/tempAlgParams.count;
    HCCL_INFO("[InsTempBroadcastNHR] BroadcastNHR entry.");

    for (int i = 0; i < subCommRanks_[0].size(); i++) {
        tempVirtRankMap_.insert(std::make_pair(subCommRanks_[0][i], i));
        HCCL_DEBUG("[InsTempBroadcastNHR] KernelRun.subCommRanks_[0][i][%d],i[%d],myRank[%d],root_[%d]",subCommRanks_[0][i], i, myRank_, root_);
    }
    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcDataSliceInfo(tempAlgParams.sliceSize, sliceInfoVec));

    threadNum_ = templateResource.threads.size();


    HCCL_INFO("[InsTempBroadcastNHR Run]RankID:[%d], root:[%u]", myRank_, root_);

    CHK_RET(PreCopy(tempAlgParams, templateResource.threads));


    CHK_RET(RunScatter(sliceInfoVec, templateResource.channels, templateResource.threads));
    CHK_RET(RunAllGather(sliceInfoVec, templateResource.channels, templateResource.threads));
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));

    HCCL_INFO("[InsTempBroadcastNHR] BroadcastNHR finish.");

    return HcclResult::HCCL_SUCCESS;
}

void InsTempBroadcastNHR::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 slaveThreadNum = threadNum - 1;
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempBroadcastNHR::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ - 1 : 1;
    u32 notifyNum = threadNum - 1;
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

} // namespace Hccl
