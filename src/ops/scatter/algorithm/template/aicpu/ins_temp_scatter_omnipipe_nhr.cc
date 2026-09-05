/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_omnipipe_nhr.h"
#include "omnipipe_template_utils.h"

namespace ops_hccl {
InsTempScatterOmniPipeNHR::InsTempScatterOmniPipeNHR(
    const OpParam& param, const u32 rankId, // 传通信域的u32，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempScatterNHR(param, rankId, subCommRanks)
{}

InsTempScatterOmniPipeNHR::~InsTempScatterOmniPipeNHR() {}

void InsTempScatterOmniPipeNHR::SetRoot(u32 root)
{
    HCCL_DEBUG("[InsTempScatterOmniPipeNHR][SetRoot] myRank_ [%u], set root_ [%u] ", myRank_, root);
    root_ = root;
}

// 语义改为返回当前template的类型，mesh返回1，nhr返回0
u64 InsTempScatterOmniPipeNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) { return 0; }

void InsTempScatterOmniPipeNHR::SetDoTask(bool doTask)
{
    HCCL_DEBUG("[InsTempScatterOmniPipeNHR][SetDoTask] myRank_ [%u], set doTask_ [%u] ", myRank_, doTask);
    doTask_.store(doTask, std::memory_order_relaxed);
}

HcclResult InsTempScatterOmniPipeNHR::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_DEBUG("[InsTempScatterOmniPipeNHR] KernelRun start");
    if (templateRankSize_ == 1) {
        HCCL_DEBUG("[InsTempScatterOmniPipeNHR] Rank [%d], template ranksize is 1.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    if (!doTask_.load(std::memory_order_relaxed)) {
        HCCL_DEBUG("[InsTempScatterOmniPipeNHR] Rank [%d], doTask_ is false, skip KernelRun.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;
    dataType_ = param.DataDes.dataType;
    enableRemoteMemAccess_ = tempAlgParams.enableRemoteMemAccess;
    threadNum_ = GetThreadNum();
    // 参考ccu版BuildSliceInfoVec：按rank组织stepSliceInfo（跳过root），并按channel拆分
    CHK_RET(PrepareScatterDataSplit(tempAlgParams_, templateResource));
    HCCL_DEBUG("MT channelsPerRank_ = %u, templateRankSize_ = %u", channelsPerRank_, templateRankSize_);
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    CHK_RET(RunNHR(templateResource.channels, templateResource.threads, tempAlgParams));
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempScatterOmniPipeNHR] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHR::DoLocalCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO("[InsTempScatterOmniPipeNHR][DoLocalCopy] DoLocalCopy myRank_ = [%u]", myRank_);
    if (tempAlgParams.sliceSize == 0) {
        HCCL_INFO("Rank [%d], get slicesize zero. skip localcopy", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    void* dstAddr = tempAlgParams.buffInfo.outputPtr;
    void* srcAddr = tempAlgParams.buffInfo.inputPtr;
    auto srcSlice
        = DataSlice(srcAddr, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.sliceSize, tempAlgParams.count);
    auto dstSlice
        = DataSlice(dstAddr, tempAlgParams.buffInfo.outBuffBaseOff, tempAlgParams.sliceSize, tempAlgParams.count);
    HCCL_INFO(
        "myRank[%u], srcSlice:%s, dstSlice:%s", myRank_, srcSlice.Describe().c_str(), dstSlice.Describe().c_str());
    CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHR::PrepareScatterDataSplit(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    // 参考ccu版BuildSliceInfoVec：stepSliceInfo第二维按rank分组（跳过root），每组repeatNum个piece
    // 预计算每个rank的stride，并按channel拆分sliceSize，供GetNHRDataSize用txSliceIdxs直接索引
    dataSplitVec_.clear();
    dataOffsetVec_.clear();
    inputOmniSliceStrideVec_.clear();
    outputOmniSliceStrideVec_.clear();
    repeatNum_ = 0;

    const auto& stepSliceInfo = tempAlgParams.stepSliceInfo;
    if (stepSliceInfo.stepSliceSize.empty() || stepSliceInfo.stepSliceSize[0].empty()) {
        HCCL_INFO("[InsTempScatterOmniPipeNHR][PrepareScatterDataSplit] stepSliceSize empty, skip.");
        return HcclResult::HCCL_SUCCESS;
    }

    // 第一维固定取0（参考ccu版 xRankSize_=1，myRank_ % xRankSize_ = 0）
    const u32 dim0Idx = myRank_ % stepSliceInfo.stepSliceSize.size();
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    u32 myAlgRank = 0;
    u32 rootAlgRank = 0;
    HCCL_DEBUG("[InsTempScatterOmniPipeNHR][PrepareScatterDataSplit] myRank_ = [%u], root_ = [%u]", myRank_, root_);
    LogSubCommRanks();
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    CHK_RET(GetAlgRank(root_, subCommRanks_[0], rootAlgRank));

    u64 totalPieceNum = stepSliceInfo.stepSliceSize[dim0Idx].size();
    u64 peerNum = templateRankSize_ - 1;
    CHK_PRT_RET(
        peerNum == 0 || totalPieceNum % peerNum != 0,
        HCCL_ERROR(
            "[PrepareScatterDataSplit] totalPieceNum[%llu] not divisible by peerNum[%llu]", totalPieceNum, peerNum),
        HcclResult::HCCL_E_INTERNAL);
    repeatNum_ = totalPieceNum / peerNum;

    // 预计算 [rank][rpt]
    inputOmniSliceStrideVec_.assign(templateRankSize_, std::vector<u64>(repeatNum_, 0));
    outputOmniSliceStrideVec_.assign(templateRankSize_, std::vector<u64>(repeatNum_, 0));
    dataSplitVec_.assign(templateRankSize_, std::vector<std::vector<u64>>(repeatNum_));
    dataOffsetVec_.assign(templateRankSize_, std::vector<std::vector<u64>>(repeatNum_));

    u32 originIndex = 0;
    for (u32 ridx = 0; ridx < templateRankSize_; ridx++) {
        CHK_RET(FillOneRankDataSplit(
            ridx, rootAlgRank, originIndex, dim0Idx, dataTypeSize, stepSliceInfo, templateResource));
    }
    return HcclResult::HCCL_SUCCESS;
}

void InsTempScatterOmniPipeNHR::LogSubCommRanks() const
{
    for (size_t i = 0; i < subCommRanks_.size(); ++i) {
        std::stringstream ss;
        for (size_t j = 0; j < subCommRanks_[i].size(); ++j) {
            ss << subCommRanks_[i][j] << " ";
        }
        HCCL_DEBUG("[%s] subCommRanks_[%zu] content: %s", __func__, i, ss.str().c_str());
    }
}

// 处理单个 rank 的数据拆分：root填0跳过，非root按channel拆分sliceSize，并预计算stride
HcclResult InsTempScatterOmniPipeNHR::FillOneRankDataSplit(
    u32 ridx, u32 rootAlgRank, u32& originIndex, u32 dim0Idx, u32 dataTypeSize, const StepSliceInfo& stepSliceInfo,
    const TemplateResource& templateResource)
{
    if (ridx == rootAlgRank) {
        // root跳过，填0
        for (u32 rpt = 0; rpt < repeatNum_; rpt++) {
            dataSplitVec_[ridx][rpt].assign(channelsPerRank_, 0);
            dataOffsetVec_[ridx][rpt].assign(channelsPerRank_, 0);
        }
        return HcclResult::HCCL_SUCCESS;
    }
    for (u32 rpt = 0; rpt < repeatNum_; rpt++) {
        u64 sliceStrideIndex = repeatNum_ * originIndex + rpt;
        u64 sliceSize = stepSliceInfo.stepSliceSize[dim0Idx][sliceStrideIndex];
        inputOmniSliceStrideVec_[ridx][rpt] = stepSliceInfo.inputOmniPipeSliceStride[dim0Idx][sliceStrideIndex];
        outputOmniSliceStrideVec_[ridx][rpt] = stepSliceInfo.outputOmniPipeSliceStride[dim0Idx][sliceStrideIndex];
        // 按channel拆分
        u64 totalDataCount = sliceSize / dataTypeSize;
        std::vector<u64> dataSplit;
        std::vector<u64> dataOffset;
        std::vector<u64> curElemCountOut;
        CHK_RET(CalcDataSplitByPortGroup(
            totalDataCount, dataTypeSize, templateResource.channels.begin()->second, curElemCountOut, dataSplit,
            dataOffset));
        dataSplitVec_[ridx][rpt] = dataSplit;
        dataOffsetVec_[ridx][rpt] = dataOffset;
    }
    originIndex++;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHR::GetNHRDataSize(
    const AicpuNHRStepInfo& st, const u32 channelIdx, void* sendCclBuffAddr, void* recvCclBuffAddr,
    const u32 dataTypeSize, const u64 rptNum, std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices,
    std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices)
{
    HCCL_DEBUG("GetNHRDataSize myRank_[%u], root_[%u] , st.nSlices[%u]", myRank_, root_, st.nSlices);
    for (u32 i = 0; i < st.nSlices; ++i) {
        HCCL_DEBUG("st.txSliceIdxs.size[%u] st.rxSliceIdxs.size[%u]", st.txSliceIdxs.size(), st.rxSliceIdxs.size());
        bool hasTx = i < st.txSliceIdxs.size();
        bool hasRx = i < st.rxSliceIdxs.size();
        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            if (hasTx
                && !AppendTxSlice(
                    st.txSliceIdxs[i], channelIdx, rpt, sendCclBuffAddr, dataTypeSize, txSrcSlices, txDstSlices)) {
                continue;
            }
            if (hasRx) {
                AppendRxSlice(
                    st.rxSliceIdxs[i], channelIdx, rpt, recvCclBuffAddr, dataTypeSize, rxSrcSlices, rxDstSlices);
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

bool InsTempScatterOmniPipeNHR::AppendTxSlice(
    u32 txIdx, u32 channelIdx, u64 rpt, void* sendCclBuffAddr, u32 dataTypeSize, std::vector<DataSlice>& txSrcSlices,
    std::vector<DataSlice>& txDstSlices)
{
    u64 size = dataSplitVec_[txIdx][rpt][channelIdx];
    u64 offset = dataOffsetVec_[txIdx][rpt][channelIdx];
    HCCL_DEBUG("size[%llu], offset[%llu]", size, offset);
    if (size == 0) {
        return false;
    }
    const auto& stepSliceInfo = tempAlgParams_.stepSliceInfo;
    u64 outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    u64 txSrcOff = stepSliceInfo.buffInfo.inBuffBaseOff + inputOmniSliceStrideVec_[txIdx][rpt] + offset;
    void* txSrcPtr = tempAlgParams_.buffInfo.inputPtr;
    if (myRank_ != root_) {
        txSrcPtr = tempAlgParams_.buffInfo.outputPtr;
        txSrcOff = outBuffBaseOff + outputOmniSliceStrideVec_[txIdx][rpt] + offset;
    }
    u64 txDstOff = outBuffBaseOff + outputOmniSliceStrideVec_[txIdx][rpt] + offset;
    HCCL_DEBUG("txSrcOff[%llu], txDstOff[%llu]", txSrcOff, txDstOff);
    txSrcSlices.emplace_back(txSrcPtr, txSrcOff, size, size / dataTypeSize);
    txDstSlices.emplace_back(sendCclBuffAddr, txDstOff, size, size / dataTypeSize);
    return true;
}

void InsTempScatterOmniPipeNHR::AppendRxSlice(
    u32 rxIdx, u32 channelIdx, u64 rpt, void* recvCclBuffAddr, u32 dataTypeSize, std::vector<DataSlice>& rxSrcSlices,
    std::vector<DataSlice>& rxDstSlices)
{
    u64 size = dataSplitVec_[rxIdx][rpt][channelIdx];
    u64 offset = dataOffsetVec_[rxIdx][rpt][channelIdx];
    HCCL_DEBUG("size[%llu], offset[%llu]", size, offset);
    if (size == 0) {
        return;
    }
    u64 rxOffset
        = tempAlgParams_.stepSliceInfo.buffInfo.outBuffBaseOff + outputOmniSliceStrideVec_[rxIdx][rpt] + offset;
    HCCL_DEBUG("rxOffset[%llu]", rxOffset);
    rxSrcSlices.emplace_back(recvCclBuffAddr, rxOffset, size, size / dataTypeSize);
    rxDstSlices.emplace_back(tempAlgParams_.buffInfo.outputPtr, rxOffset, size, size / dataTypeSize);
}

HcclResult InsTempScatterOmniPipeNHR::RunNHR(
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads,
    const TemplateDataParams& tempAlgParams)
{
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    // 片数取PrepareScatterDataSplit预计算的repeatNum_（参考ccu版RunScatterNHRDispatch）
    const u64 rptNum = repeatNum_;
    bool isPcieProtocal = IsPcieProtocol(channels);
    HCCL_DEBUG(
        "[Scatter-OmniPipe-NHR][RunNHR] root[%u], nSteps[%u], rptNum[%llu], isPcieProtocal[%d], channelsPerRank_[%u]",
        root_, nSteps, rptNum, isPcieProtocal, channelsPerRank_);
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        for (u32 step = 0; step < nSteps; step++) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, nSteps, stepInfo));
            // 本步既不收也不发，跳过
            if (stepInfo.txSliceIdxs.empty() && stepInfo.rxSliceIdxs.empty()) {
                continue;
            }
            HCCL_DEBUG(
                "step[%u], stepInfo.txSliceIdxs.size[%u] stepInfo.rxSliceIdxs.size[%u]", step,
                stepInfo.txSliceIdxs.size(), stepInfo.rxSliceIdxs.size());
            // 只有Tx，使用Send指令（root分发场景）
            if (!stepInfo.txSliceIdxs.empty() && stepInfo.rxSliceIdxs.empty()) {
                CHK_RET(ExecuteTxOnlyStep(
                    stepInfo, channelIdx, dataTypeSize, rptNum, isPcieProtocal, channels, threads, step));
            } else if (stepInfo.txSliceIdxs.empty() && !stepInfo.rxSliceIdxs.empty()) {
                // 只有Rx，使用Recv指令（非root接收场景）
                CHK_RET(ExecuteRxOnlyStep(
                    stepInfo, channelIdx, dataTypeSize, rptNum, isPcieProtocal, channels, threads, step));
            } else {
                // 既有Tx又有Rx，使用SendRecv指令
                CHK_RET(ExecuteTxRxStep(
                    stepInfo, channelIdx, dataTypeSize, rptNum, isPcieProtocal, channels, threads, step));
            }
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// Tx-only分支：只有发送，查toRank channel，构建tx切片，按isPcieProtocal选SendRead/SendBatchWrite
HcclResult InsTempScatterOmniPipeNHR::ExecuteTxOnlyStep(
    const AicpuNHRStepInfo& stepInfo, u32 channelIdx, u32 dataTypeSize, u64 rptNum, bool isPcieProtocal,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads, u32 step)
{
    CHK_PRT_RET(
        channels.find(stepInfo.toRank) == channels.end() || channels.at(stepInfo.toRank).empty(),
        HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] toRank[%u] not in channels", stepInfo.toRank), HCCL_E_INTERNAL);
    const ChannelInfo& linkSend = channels.at(stepInfo.toRank)[channelIdx];
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> dummyRx;
    std::vector<DataSlice> dummyRxDst;
    CHK_RET(GetNHRDataSize(
        stepInfo, channelIdx, linkSend.remoteCclMem.addr, nullptr, dataTypeSize, rptNum, txSrcSlices, txDstSlices,
        dummyRx, dummyRxDst));
    if (txSrcSlices.empty()) {
        HCCL_DEBUG("[Scatter-OmniPipe-NHR][RunNHR] txSrcSlices empty, skip.");
        return HcclResult::HCCL_SUCCESS;
    }
    SlicesList txSlicesList({txSrcSlices}, {txDstSlices});
    DataInfo sendData(linkSend, txSlicesList);
    if (isPcieProtocal) {
        CHK_PRT_RET(
            static_cast<HcclResult>(SendRead(sendData, threads.at(channelIdx))),
            HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] BatchSend failed (step=%u)", step), HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            static_cast<HcclResult>(SendBatchWrite(sendData, threads.at(channelIdx))),
            HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] BatchSend failed (step=%u)", step), HCCL_E_INTERNAL);
    }
    HCCL_DEBUG("[Scatter-OmniPipe-NHR][RunNHR] myRank[%u] send data toRank[%u] success", myRank_, stepInfo.toRank);
    return HcclResult::HCCL_SUCCESS;
}

// Rx-only分支：只有接收，查fromRank channel，构建rx切片，按isPcieProtocal选RecvRead/RecvWrite
HcclResult InsTempScatterOmniPipeNHR::ExecuteRxOnlyStep(
    const AicpuNHRStepInfo& stepInfo, u32 channelIdx, u32 dataTypeSize, u64 rptNum, bool isPcieProtocal,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads, u32 step)
{
    CHK_PRT_RET(
        channels.find(stepInfo.fromRank) == channels.end() || channels.at(stepInfo.fromRank).empty(),
        HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] fromRank[%u] not in channels", stepInfo.fromRank), HCCL_E_INTERNAL);
    const ChannelInfo& linkRecv = channels.at(stepInfo.fromRank)[channelIdx];
    std::vector<DataSlice> dummyTxSrc;
    std::vector<DataSlice> dummyTxDst;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    CHK_RET(GetNHRDataSize(
        stepInfo, channelIdx, nullptr, linkRecv.remoteCclMem.addr, dataTypeSize, rptNum, dummyTxSrc, dummyTxDst,
        rxSrcSlices, rxDstSlices));
    if (rxSrcSlices.empty()) {
        HCCL_DEBUG("[Scatter-OmniPipe-NHR][RunNHR] rxSrcSlices empty, skip.");
        return HcclResult::HCCL_SUCCESS;
    }
    SlicesList rxSlicesList({rxSrcSlices}, {rxDstSlices});
    DataInfo recvData(linkRecv, rxSlicesList);
    if (isPcieProtocal) {
        CHK_PRT_RET(
            static_cast<HcclResult>(RecvRead(recvData, threads.at(channelIdx))),
            HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] BatchRecv failed (step=%u)", step), HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            static_cast<HcclResult>(RecvWrite(recvData, threads.at(channelIdx))),
            HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] BatchRecv failed (step=%u)", step), HCCL_E_INTERNAL);
    }
    HCCL_DEBUG("[Scatter-OmniPipe-NHR][RunNHR] myRank[%u] recv data fromRank[%u] success", myRank_, stepInfo.fromRank);
    return HcclResult::HCCL_SUCCESS;
}

// Tx+Rx分支：同时收发，查toRank/fromRank channel，构建tx/rx切片，按isPcieProtocal选SendRecvRead/SendRecvWrite
HcclResult InsTempScatterOmniPipeNHR::ExecuteTxRxStep(
    const AicpuNHRStepInfo& stepInfo, u32 channelIdx, u32 dataTypeSize, u64 rptNum, bool isPcieProtocal,
    const std::map<u32, std::vector<ChannelInfo>>& channels, const std::vector<ThreadHandle>& threads, u32 step)
{
    CHK_PRT_RET(
        channels.find(stepInfo.toRank) == channels.end() || channels.at(stepInfo.toRank).empty(),
        HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] toRank[%u] not in channels", stepInfo.toRank), HCCL_E_INTERNAL);
    CHK_PRT_RET(
        channels.find(stepInfo.fromRank) == channels.end() || channels.at(stepInfo.fromRank).empty(),
        HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] fromRank[%u] not in channels", stepInfo.fromRank), HCCL_E_INTERNAL);
    const ChannelInfo& linkSend = channels.at(stepInfo.toRank)[channelIdx];
    const ChannelInfo& linkRecv = channels.at(stepInfo.fromRank)[channelIdx];
    std::vector<DataSlice> txSrcSlices;
    std::vector<DataSlice> txDstSlices;
    std::vector<DataSlice> rxSrcSlices;
    std::vector<DataSlice> rxDstSlices;
    CHK_RET(GetNHRDataSize(
        stepInfo, channelIdx, linkSend.remoteCclMem.addr, linkRecv.remoteCclMem.addr, dataTypeSize, rptNum, txSrcSlices,
        txDstSlices, rxSrcSlices, rxDstSlices));
    if (txSrcSlices.empty() && rxSrcSlices.empty()) {
        HCCL_DEBUG("[Scatter-OmniPipe-NHR][RunNHR] txSrcSlices and rxSrcSlices empty, skip.");
        return HcclResult::HCCL_SUCCESS;
    }
    SendRecvInfo info{{linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}};
    if (isPcieProtocal) {
        CHK_PRT_RET(
            static_cast<HcclResult>(SendRecvRead(info, threads.at(channelIdx))),
            HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] BatchSR failed (step=%u)", step), HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            static_cast<HcclResult>(SendRecvWrite(info, threads.at(channelIdx))),
            HCCL_ERROR("[Scatter-OmniPipe-NHR][RunNHR] BatchSR failed (step=%u)", step), HCCL_E_INTERNAL);
    }
    HCCL_DEBUG(
        "[Scatter-OmniPipe-NHR][RunNHR] myRank[%u] sendRecv data toRank[%u] fromRank[%u] success", myRank_,
        stepInfo.toRank, stepInfo.fromRank);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
