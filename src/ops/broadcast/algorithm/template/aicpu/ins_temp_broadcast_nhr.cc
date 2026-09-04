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
#include <cstring>
#include "channel.h"

namespace ops_hccl {
std::vector<CostModelParam> InsTempBroadcastNHR::CalcCostCoeff(CalcCostCoeffParam param)
{
    // NHR递归halving-doubling算法（scatter+allgather两阶段），始终走CLOS网络
    CommTopo netType = CommTopo::COMM_TOPO_CLOS;
    bool isMultiLink = (param.algName != nullptr && strstr(param.algName, "TwoShotMultiLink") != nullptr);
    // TwoShotMultiLink走CLOS 8端口，普通NHR走6端口
    int portNum = isMultiLink ? 8 : 6;
    // TwoShotMultiLink多通道并行，数据拆分和同步开销略大，kernelNum增加2
    int kernelNum = isMultiLink ? 12 : 10;
    int taskNum = 8 * (param.rankSize - 1);
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    // NHR两阶段：scatter阶段每轮发D/R，allgather阶段每轮发D/R，共2D/R
    CostModelManager::Global()->CalcNHRParams(
        param.dataRatio * 2 / param.rankSize, netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        // 原selector: CalcLocalCopyParams(param.n) 即全量数据的本地拷贝（root拷入、非root拷出，平均1份全量）
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
    }
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    // nhr实测和理论估计相差较大，先用经验值（和all_reduce NHR一致）
    D = 1e-6 * taskNum;
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    return params;
}

InsTempBroadcastNHR::InsTempBroadcastNHR(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempBroadcastNHR::~InsTempBroadcastNHR() {}

HcclResult InsTempBroadcastNHR::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> level0Channels;
    // MESH_1D_CLOS 层0 CLOS/MESH 实例尺寸对称（GCD>1），Level1Nhr 不可达；本分支仅单级拓扑可达
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestNhrMultiJetty(comm, param, topoInfo, subCommRanks_, myChannelDescs));
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UB_CTP) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[InsTempBroadcastNHR::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    CHK_PRT_RET(
        level0Channels.empty(), HCCL_ERROR("[InsTempBroadcastNHR::CalcRes] no UB_CTP channel after filter"),
        HcclResult::HCCL_E_INTERNAL);
    resourceRequest.channels.push_back(level0Channels);
    channelsPerRank_ = CalcChannelsPerRank(level0Channels);
    if (channelsPerRank_ > MAX_JETTY_NUM) {
        HCCL_ERROR(
            " %s channelsPerRank_ %u is greater than MAX_JETTY_NUM %u", __func__, channelsPerRank_, MAX_JETTY_NUM);
    } else {
        HCCL_DEBUG(" %s channelsPerRank_ is %u ", __func__, channelsPerRank_);
    }
    GetRes(resourceRequest);
    return HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::GetRes(AlgResourceRequest& resourceRequest) const
{
    u32 threadNum = channelsPerRank_;
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
    u64 scratchMultiple = 0;
    if (!enableRemoteMemAccess_) {
        scratchMultiple = 1;
    }
    return scratchMultiple;
}

u64 InsTempBroadcastNHR::GetThreadNum() const { return channelsPerRank_; }

HcclResult
InsTempBroadcastNHR::PostCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    if ((!enableRemoteMemAccess_) && (u32(myRank_) != root_)) {
        HCCL_INFO("[InsTempBroadcastNHR][PostCopy] Opbase && isBottom, copy from outBuff to userOut");
        u64 inOffset = tempAlgParams.buffInfo.hcclBuffBaseOff;

        DataSlice usrInSlice
            = DataSlice(tempAlgParams.buffInfo.hcclBuff.addr, inOffset, tempAlgParams.sliceSize, tempAlgParams.count);
        DataSlice usrOutSlice = DataSlice(
            tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize,
            tempAlgParams.count);
        CHK_RET(LocalCopy(threads[0], usrInSlice, usrOutSlice));
    } else {
        HCCL_INFO("[InsTempBroadcastNHR][PostCopy] Offload Model, skip postcopy");
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempBroadcastNHR::PreCopy(const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads) const
{
    if ((!enableRemoteMemAccess_) && (u32(myRank_) == root_)) {
        DataSlice usrInSlice = DataSlice(
            tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.sliceSize,
            tempAlgParams.count);
        DataSlice usrOutSlice = DataSlice(
            tempAlgParams.buffInfo.hcclBuff.addr, tempAlgParams.buffInfo.hcclBuffBaseOff, tempAlgParams.sliceSize,
            tempAlgParams.count);
        CHK_PRT_RET(
            LocalCopy(threads[0], usrInSlice, usrOutSlice),
            HCCL_ERROR("[InsTempBroadcastNHR] RunScatter userIn to cclIn copy failed"), HcclResult::HCCL_E_INTERNAL);
    } else {
        HCCL_INFO("[InsTempBroadcastNHR][PostCopy] Offload Model, skip postcopy");
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::GetAllGatherStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo)
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

    HCCL_DEBUG(
        "[InsTempBroadcastNHR][GetAllGatherStepInfo] myRank_[%u] toRank[%u] fromRank[%u] nSteps[%u], step[%u], "
        "rankIdx[%u]",
        myRank_, sendTo, recvFrom, nSteps, step, rankIdx);
    for (u32 i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG(
            "[InsTempBroadcastNHR][GetAllGatherStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx,
            rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

// NHR每步的算法描述原理函数
HcclResult InsTempBroadcastNHR::GetScatterStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const
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

HcclResult InsTempBroadcastNHR::RunScatter(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    // nhr主体部分,从ScratchIn计算，结果放至ScratchOut上, 该部分均从inType搬运到outType
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetScatterStepInfo(step, nSteps, stepInfo));
        HCCL_INFO("[InsTempBroadcastNHR]RunScatter GetScatterStepInfo after:[%d], root:[%u]", myRank_, root_);
        CHK_PRT_RET(
            BatchTxRx(stepInfo, channels, threads, channelIdx), HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx failed"),
            HcclResult::HCCL_E_INTERNAL);
    }

    return HCCL_SUCCESS;
}

u32 InsTempBroadcastNHR::GetRankFromMap(const u32 rankIdx) const
{
    u32 rank = -1;
    for (auto& pair : tempVirtRankMap_) {
        if (pair.second == rankIdx) {
            rank = pair.first;
            break;
        }
    }
    return rank;
}

HcclResult InsTempBroadcastNHR::RunAllGather(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);

    u64 memOffset = (!enableRemoteMemAccess_) ? buffInfo_.hcclBuffBaseOff : buffInfo_.outBuffBaseOff;

    for (u32 step = 0; step < nSteps; step++) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetAllGatherStepInfo(step, nSteps, stepInfo));

        u32 fromRankKey = GetRankFromMap(stepInfo.fromRank);
        u32 toRankKey = GetRankFromMap(stepInfo.toRank);
        auto itRecv = channels.find(fromRankKey);
        auto itSend = channels.find(toRankKey);
        CHK_PRT_RET(
            itRecv == channels.end() || itSend == channels.end(),
            HCCL_ERROR(
                "[%s] rank[%u] channel not found, fromRankKey[%u] found[%d] toRankKey[%u] found[%d]", __func__, myRank_,
                fromRankKey, itRecv != channels.end(), toRankKey, itSend != channels.end()),
            HCCL_E_INTERNAL);
        const ChannelInfo& linkRecv = itRecv->second[channelIdx];
        const ChannelInfo& linkSend = itSend->second[channelIdx];
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        void* localBuff = (!enableRemoteMemAccess_) ? buffInfo_.hcclBuff.addr : buffInfo_.outputPtr;
        void* remoteSendAddr = (!enableRemoteMemAccess_) ? linkSend.remoteCclMem.addr : buffInfo_.outputPtr;
        void* remoteRecvAddr = (!enableRemoteMemAccess_) ? linkRecv.remoteCclMem.addr : buffInfo_.outputPtr;
        for (u32 i = 0; i < stepInfo.nSlices; i++) {
            u32 txIdx = stepInfo.txSliceIdxs[i];
            u32 rxIdx = stepInfo.rxSliceIdxs[i];
            u64 txOff = (txIdx == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
            u64 txSz = (txIdx == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
            u64 rxOff = (rxIdx == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
            u64 rxSz = (rxIdx == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
            txSrcSlices.emplace_back(localBuff, txIdx * sliceSize_ + txOff + memOffset, txSz);
            txDstSlices.emplace_back(remoteSendAddr, txIdx * sliceSize_ + txOff + memOffset, txSz);
            rxSrcSlices.emplace_back(remoteRecvAddr, rxIdx * sliceSize_ + rxOff + memOffset, rxSz);
            rxDstSlices.emplace_back(localBuff, rxIdx * sliceSize_ + rxOff + memOffset, rxSz);
        }

        TxRxChannels sendRecvLinks(linkSend, linkRecv);
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});

        SendRecvInfo sendRecvInfo(sendRecvLinks, sendRecvSlicesList, dataType_);
        if (isDmaRead_) {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, threads[channelIdx]),
                HCCL_ERROR("[InsTempBroadcastNHR] RunAllGather send failed"), HcclResult::HCCL_E_INTERNAL);
        } else {
            CHK_PRT_RET(
                SendRecvBatchWrite(sendRecvInfo, threads[channelIdx]),
                HCCL_ERROR("[InsTempBroadcastNHR] RunAllGather send failed"), HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

// Send multiple DataSlices
HcclResult InsTempBroadcastNHR::BatchTxRx(
    AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    HCCL_INFO("[InsTempBroadcastNHR]BatchTxRx entry:[%d], root:[%u]", myRank_, root_);
    u64 memOffset = (!enableRemoteMemAccess_) ? buffInfo_.hcclBuffBaseOff : buffInfo_.inBuffBaseOff;
    // 只有Tx,使用send指令
    if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() == 0) {
        CHK_RET(BatchSend(stepInfo, channels, threads, memOffset, channelIdx));
    }
    // 只有Rx，使用recv指令
    else if (stepInfo.txSliceIdxs.size() == 0 && stepInfo.rxSliceIdxs.size() > 0) {
        CHK_RET(BatchRecv(stepInfo, channels, threads, memOffset, channelIdx));
    }
    // 既有Tx又有Rx，使用SendRecv指令
    else if (stepInfo.txSliceIdxs.size() > 0 && stepInfo.rxSliceIdxs.size() > 0) {
        CHK_RET(BatchSR(stepInfo, channels, threads, memOffset, channelIdx));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::BatchSend(
    AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, u64 memOffset, u32 channelIdx) const
{
    u32 toRankKey = GetRankFromMap(stepInfo.toRank);
    auto itSend = channels.find(toRankKey);
    CHK_PRT_RET(
        itSend == channels.end(),
        HCCL_ERROR("[%s] rank[%u] toRankKey[%u] not found in channels", __func__, myRank_, toRankKey), HCCL_E_INTERNAL);
    const ChannelInfo& linkSend = itSend->second[channelIdx];
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs[i];
        u64 partialOffset = (txId == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
        u64 partialSize = (txId == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
        void* srcBuffAddr = (!enableRemoteMemAccess_) ? buffInfo_.hcclBuff.addr : buffInfo_.inputPtr;
        void* remoteBuffAddr
            = (!enableRemoteMemAccess_) ? linkSend.remoteCclMem.addr : linkSend.remoteOutputGraphMode.addr;
        DataSlice txSrcSlice = DataSlice(srcBuffAddr, memOffset + txId * sliceSize_ + partialOffset, partialSize);
        DataSlice txDstSlice = DataSlice(remoteBuffAddr, memOffset + txId * sliceSize_ + partialOffset, partialSize);
        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
    }
    SlicesList txSlicesList(txSrcSlices, txDstSlices);
    DataInfo sendData(linkSend, txSlicesList, dataType_);
    if (isDmaRead_) {
        CHK_PRT_RET(
            SendRead(sendData, threads[channelIdx]), HCCL_ERROR("[InsTempBroadcastNHR] BatchSend failed"),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            SendBatchWrite(sendData, threads[channelIdx]), HCCL_ERROR("[InsTempBroadcastNHR] BatchSend failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::BatchRecv(
    AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, u64 memOffset, u32 channelIdx) const
{
    HCCL_INFO("[InsTempBroadcastNHR]BatchRecv entry:[%d], root:[%u]", myRank_, root_);
    u32 fromRankKey = GetRankFromMap(stepInfo.fromRank);
    auto itRecv = channels.find(fromRankKey);
    CHK_PRT_RET(
        itRecv == channels.end(),
        HCCL_ERROR("[%s] rank[%u] fromRankKey[%u] not found in channels", __func__, myRank_, fromRankKey),
        HCCL_E_INTERNAL);
    const ChannelInfo& linkRecv = itRecv->second[channelIdx];
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs[i];
        u64 partialOffset = (rxId == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
        u64 partialSize = (rxId == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
        void* remoteBuffAddr
            = (!enableRemoteMemAccess_) ? linkRecv.remoteCclMem.addr : linkRecv.remoteOutputGraphMode.addr;
        void* BuffAddr = (!enableRemoteMemAccess_) ? buffInfo_.hcclBuff.addr : buffInfo_.inputPtr;
        DataSlice rxSrcSlice = DataSlice(remoteBuffAddr, memOffset + rxId * sliceSize_ + partialOffset, partialSize);
        DataSlice rxDstSlice = DataSlice(BuffAddr, memOffset + rxId * sliceSize_ + partialOffset, partialSize);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);
    }
    SlicesList rxSlicesList(rxSrcSlices, rxDstSlices);
    DataInfo recvData(linkRecv, rxSlicesList, dataType_);
    if (isDmaRead_) {
        CHK_PRT_RET(
            RecvRead(recvData, threads[channelIdx]), HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx Recv failed"),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            RecvWrite(recvData, threads[channelIdx]), HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx Recv failed"),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::BatchSR(
    AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const std::vector<ThreadHandle>& threads, u64 memOffset, u32 channelIdx) const
{
    u32 toRankKey = GetRankFromMap(stepInfo.toRank);
    u32 fromRankKey = GetRankFromMap(stepInfo.fromRank);
    auto itSend = channels.find(toRankKey);
    auto itRecv = channels.find(fromRankKey);
    CHK_PRT_RET(
        itSend == channels.end() || itRecv == channels.end(),
        HCCL_ERROR(
            "[%s] rank[%u] channel not found, toRankKey[%u] found[%d] fromRankKey[%u] found[%d]", __func__, myRank_,
            toRankKey, itSend != channels.end(), fromRankKey, itRecv != channels.end()),
        HCCL_E_INTERNAL);
    const ChannelInfo& linkSend = itSend->second[channelIdx];
    const ChannelInfo& linkRecv = itRecv->second[channelIdx];
    TxRxChannels linkSendRecv = {linkSend, linkRecv};

    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
        u32 txId = stepInfo.txSliceIdxs[i];
        u64 partialOffset = (txId == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
        u64 partialSize = (txId == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
        void* remoteSendBuffAddr
            = (!enableRemoteMemAccess_) ? linkSend.remoteCclMem.addr : linkSend.remoteOutputGraphMode.addr;
        void* BuffAddr = (!enableRemoteMemAccess_) ? buffInfo_.hcclBuff.addr : buffInfo_.inputPtr;
        DataSlice txSrcSlice = DataSlice(BuffAddr, memOffset + txId * sliceSize_ + partialOffset, partialSize);
        DataSlice txDstSlice
            = DataSlice(remoteSendBuffAddr, memOffset + txId * sliceSize_ + partialOffset, partialSize);
        txSrcSlices.push_back(txSrcSlice);
        txDstSlices.push_back(txDstSlice);
    }
    SlicesList txSlicesList(txSrcSlices, txSrcSlices);
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
        u32 rxId = stepInfo.rxSliceIdxs[i];
        u64 partialOffset = (rxId == templateRankSize_ - 1) ? dataOffsetTail_[channelIdx] : dataOffset_[channelIdx];
        u64 partialSize = (rxId == templateRankSize_ - 1) ? dataSplitTail_[channelIdx] : dataSplit_[channelIdx];
        void* remoteRecvBuffAddr
            = (!enableRemoteMemAccess_) ? linkRecv.remoteCclMem.addr : linkRecv.remoteOutputGraphMode.addr;
        void* BuffAddr = (!enableRemoteMemAccess_) ? buffInfo_.hcclBuff.addr : buffInfo_.inputPtr;
        DataSlice rxSrcSlice
            = DataSlice(remoteRecvBuffAddr, memOffset + rxId * sliceSize_ + partialOffset, partialSize);
        DataSlice rxDstSlice = DataSlice(BuffAddr, memOffset + rxId * sliceSize_ + partialOffset, partialSize);
        rxSrcSlices.push_back(rxSrcSlice);
        rxDstSlices.push_back(rxDstSlice);
    }
    SlicesList rxSlicesList(rxSrcSlices, rxDstSlices);
    TxRxSlicesList txRxSlicesList(txSlicesList, rxSlicesList);
    SendRecvInfo sendRecvInfo(linkSendRecv, txRxSlicesList);
    if (isDmaRead_) {
        CHK_PRT_RET(
            SendRecvRead(sendRecvInfo, threads[channelIdx]),
            HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx SendRecv failed"), HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            SendRecvWrite(sendRecvInfo, threads[channelIdx]),
            HCCL_ERROR("[InsTempBroadcastNHR] BatchTxRx SendRecv failed"), HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempBroadcastNHR::SetRoot(u32 root)
{
    root_ = root;
    HCCL_INFO("[InsTempBroadcastNHR][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root_);
}

HcclResult InsTempBroadcastNHR::PrepareDataSplitForMultiChannel(const TemplateResource& templateResource)
{
    CHK_PRT_RET(
        templateResource.channels.empty() || templateResource.channels.begin()->second.empty(),
        HCCL_ERROR("[InsTempBroadcastNHR][PrepareDataSplitForMultiChannel] channels is empty."), HCCL_E_INTERNAL);
    std::vector<u64> elemCountOut;
    u64 totalDataCount = sliceSize_ / dataTypeSize_;
    CHK_RET(CalcDataSplitByPortGroup(
        totalDataCount, dataTypeSize_, templateResource.channels.begin()->second, elemCountOut, dataSplit_,
        dataOffset_));
    if (tailSize_ > 0 && tailSize_ != sliceSize_) {
        std::vector<u64> elemCountOutTail;
        u64 totalDataCountTail = tailSize_ / dataTypeSize_;
        CHK_RET(CalcDataSplitByPortGroup(
            totalDataCountTail, dataTypeSize_, templateResource.channels.begin()->second, elemCountOutTail,
            dataSplitTail_, dataOffsetTail_));
    } else {
        dataOffsetTail_ = dataOffset_;
        dataSplitTail_ = dataSplit_;
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempBroadcastNHR::PreSyncSubThreads(const std::vector<ThreadHandle>& threads)
{
    if (threadNum_ <= 1) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.begin() + threadNum_);
    GetNotifyIdxMainToSub(notifyIdxMainToSub_);
    return PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_);
}

HcclResult InsTempBroadcastNHR::PostSyncSubThreads(const std::vector<ThreadHandle>& threads)
{
    if (threadNum_ <= 1) {
        return HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.begin() + threadNum_);
    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    return PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_);
}

HcclResult InsTempBroadcastNHR::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempBroadcastNHR] BroadcastNHR entry.");
    buffInfo_ = tempAlgParams.buffInfo;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    bool isPcieProtocal = IsPcieProtocol(templateResource.channels); // 判断是否存在pcie链路
    isDmaRead_ = isPcieProtocal;                                     // 是否使用Read模式
    HCCL_DEBUG("[InsTempBroadcastNHR] Use Dma Read[%d]", isDmaRead_);
    HCCL_INFO("[InsTempBroadcastNHR] BroadcastNHR entry.");

    for (int i = 0; i < subCommRanks_[0].size(); i++) {
        tempVirtRankMap_.insert(std::make_pair(subCommRanks_[0][i], i));
        HCCL_DEBUG(
            "[InsTempBroadcastNHR] KernelRun.subCommRanks_[0][i][%d],i[%d],myRank[%d],root_[%d]", subCommRanks_[0][i],
            i, myRank_, root_);
    }
    tempAlgParams_ = tempAlgParams;
    u64 dataSize = tempAlgParams.sliceSize;
    sliceSize_ = (dataSize / (templateRankSize_ * dataTypeSize_)) * dataTypeSize_;
    tailSize_ = dataSize - sliceSize_ * (templateRankSize_ - 1);
    CHK_RET(PrepareDataSplitForMultiChannel(templateResource));
    threadNum_ = GetThreadNum();
    CHK_PRT_RET(
        threadNum_ > templateResource.threads.size(),
        HCCL_ERROR(
            "[InsTempBroadcastNHR] Rank [%d], requiredQue [%u] more than templateQueNum [%zu].", myRank_, threadNum_,
            templateResource.threads.size()),
        HcclResult::HCCL_E_INTERNAL);
    HCCL_INFO(
        "[InsTempBroadcastNHR Run]RankID:[%d], root:[%u], channelsPerRank_:[%u]", myRank_, root_, channelsPerRank_);

    CHK_RET(PreCopy(tempAlgParams, templateResource.threads));
    CHK_RET(PreSyncSubThreads(templateResource.threads));
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(RunScatter(templateResource.channels, templateResource.threads, channelIdx));
        CHK_RET(RunAllGather(templateResource.channels, templateResource.threads, channelIdx));
    }
    CHK_RET(PostSyncSubThreads(templateResource.threads));
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));

    HCCL_INFO("[InsTempBroadcastNHR] BroadcastNHR finish.");

    return HcclResult::HCCL_SUCCESS;
}

void InsTempBroadcastNHR::GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMainToSub)
{
    notifyIdxMainToSub.clear();
    u32 slaveThreadNum = GetThreadNum() - 1;
    for (u32 i = 0; i < slaveThreadNum; i++) {
        notifyIdxMainToSub.push_back(0);
    }
}

void InsTempBroadcastNHR::GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain)
{
    notifyIdxSubToMain.clear();
    u32 notifyNum = GetThreadNum() - 1;
    for (u32 i = 0; i < notifyNum; i++) {
        notifyIdxSubToMain.push_back(i);
    }
}

} // namespace ops_hccl
