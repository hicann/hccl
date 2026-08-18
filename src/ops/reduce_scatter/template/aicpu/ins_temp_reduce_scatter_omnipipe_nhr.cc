/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_omnipipe_nhr.h"
#include "omnipipe_template_utils.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "hccl_sym_win.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
constexpr u32 SMALL_COUNT_512KB = 512 * 1024;
namespace ops_hccl {
InsTempReduceScatterOmniPipeNHR::InsTempReduceScatterOmniPipeNHR(
    const OpParam& param, const u32 rankId, // 传通信域的u32，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempReduceScatterNHR(param, rankId, subCommRanks)
{}

InsTempReduceScatterOmniPipeNHR::~InsTempReduceScatterOmniPipeNHR() {}

// 语义改为返回当前template的类型，mesh返回1，nhr返回0
u64 InsTempReduceScatterOmniPipeNHR::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 0;
}

HcclResult InsTempReduceScatterOmniPipeNHR::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO(
        "[InsTempReduceScatterOmniPipeNHR][KernelRun] start NHR reduce-scatter template, "
        "rank[%u], symmetric[%d].",
        myRank_, param.supportSymmetricMemory);
    if (templateRankSize_ == 1) {
        HCCL_INFO(
            "[InsTempReduceScatterOmniPipeNHR][KernelRun] skip communication for single-rank template, "
            "rank[%u].",
            myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    tempAlgParams_ = tempAlgParams;
    channels_ = templateResource.channels;
    dataType_ = param.DataDes.dataType;
    // 缓存 user input 对称窗口、窗口内偏移和开关，供 RunNHR 获取对端 input 地址。
    supportSymmetricMemory_ = param.supportSymmetricMemory;
    inputSymWindow_ = param.inputSymWindow;
    inputOffset_ = param.inputOffset;

    threadNum_ = GetThreadNum();

    // 模板内部只处理 NHR 主从线程同步；每个 loop 的前后本地拷贝由 executor 统一编排。
    CHK_RET(PrepareOmniPipeDataSplitForMultiChannel(
        static_cast<CommonAlgTemplateBase*>(this), tempAlgParams_, dataType_, templateResource, dataSplitVec_,
        dataOffsetVec_));
    HCCL_DEBUG(
        "[InsTempReduceScatterOmniPipeNHR][KernelRun] data split prepared for multi-channel NHR, "
        "channelsPerRank[%u], templateRankSize[%u].",
        channelsPerRank_, templateRankSize_);
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        // 主线程通知各子线程开始处理其对应通道。
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    HCCL_DEBUG(
        "[InsTempReduceScatterOmniPipeNHR][KernelRun] launch NHR communication tasks, "
        "channelCount[%u].",
        channelsPerRank_);
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(RunNHR(templateResource.threads, channelIdx));
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempReduceScatterOmniPipeNHR][KernelRun] NHR reduce-scatter template completed, rank[%u].", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::DoLocalCopy(
    const TemplateDataParams& tempAlgParams, const std::vector<ThreadHandle>& threads)
{
    HCCL_INFO(
        "[InsTempReduceScatterOmniPipeNHR][DoLocalCopy] start local copy, rank[%u], repeatNum[%llu].", myRank_,
        tempAlgParams.repeatNum);
    auto iter = std::find(subCommRanks_[0].begin(), subCommRanks_[0].end(), myRank_);
    if (iter == subCommRanks_[0].end()) {
        HCCL_ERROR(
            "[InsTempReduceScatterOmniPipeNHR][DoLocalCopy] local rank is absent from the sub-communicator, "
            "rank[%u].",
            myRank_);
        return HCCL_E_INTERNAL;
    }

    // 区分前后搬运
    void* srcAddr;
    void* dstAddr;
    if (tempAlgParams.buffInfo.inBuffType == BufferType::INPUT) {
        srcAddr = tempAlgParams.buffInfo.inputPtr;
        dstAddr = tempAlgParams.buffInfo.hcclBuff.addr;
    } else if (tempAlgParams.buffInfo.inBuffType == BufferType::HCCL_BUFFER) {
        srcAddr = tempAlgParams.buffInfo.hcclBuff.addr;
        dstAddr = tempAlgParams.buffInfo.outputPtr;
    } else {
        HCCL_ERROR(
            "[InsTempReduceScatterOmniPipeNHR][DoLocalCopy] unsupported input buffer type[%d].",
            static_cast<int>(tempAlgParams.buffInfo.inBuffType));
        return HCCL_E_PARA;
    }
    // 这里的循环precopy是ranksize-1，postcopy是1
    for (auto i = 0; i < tempAlgParams.repeatNum; ++i) {
        auto srcSlice = DataSlice(
            srcAddr, tempAlgParams.buffInfo.inBuffBaseOff + i * tempAlgParams.inputSliceStride, tempAlgParams.sliceSize,
            tempAlgParams.count);
        auto dstSlice = DataSlice(
            dstAddr, tempAlgParams.buffInfo.outBuffBaseOff + i * tempAlgParams.outputSliceStride,
            tempAlgParams.sliceSize, tempAlgParams.count);
        CHK_RET(static_cast<HcclResult>(LocalCopy(threads[0], srcSlice, dstSlice)));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::GetNHRDataSize(
    const AicpuNHRStepInfo& st, const u32 channelIdx, void* sendCclBuffAddr, void* recvCclBuffAddr,
    const u32 dataTypeSize, const u64 rptNum, std::vector<DataSlice>& txSrcSlices, std::vector<DataSlice>& txDstSlices,
    std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices)
{
    for (u32 i = 0; i < st.nSlices; ++i) {
        const u32 txIdx = st.txSliceIdxs[i]; // 算法序
        const u32 rxIdx = st.rxSliceIdxs[i];
        for (u64 rpt = 0; rpt < rptNum; ++rpt) {
            u64 scratchBaseTx = tempAlgParams_.buffInfo.inBuffBaseOff
                                + tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt];
            u64 scratchBaseRx = tempAlgParams_.buffInfo.inBuffBaseOff
                                + tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[rxIdx][rpt];
            scratchBaseTx += dataOffsetVec_[txIdx][rpt][channelIdx];
            scratchBaseRx += dataOffsetVec_[rxIdx][rpt][channelIdx];
            // 已对齐，这边都用input
            const u64 txScOff = scratchBaseTx + tempAlgParams_.stepSliceInfo.stepInputSliceStride[txIdx];
            const u64 rxScOff = scratchBaseRx + tempAlgParams_.stepSliceInfo.stepInputSliceStride[rxIdx];

            // 对称路径在 user input 上原位归约，对端地址由 RunNHR 传入。
            void* localAddr
                = supportSymmetricMemory_ ? tempAlgParams_.buffInfo.inputPtr : tempAlgParams_.buffInfo.hcclBuff.addr;
            DataSlice txSrcSlice = DataSlice(
                localAddr, txScOff, dataSplitVec_[txIdx][rpt][channelIdx],
                dataSplitVec_[txIdx][rpt][channelIdx] / dataTypeSize); // 发送源
            DataSlice txDstSlice = DataSlice(
                sendCclBuffAddr, txScOff, dataSplitVec_[txIdx][rpt][channelIdx],
                dataSplitVec_[txIdx][rpt][channelIdx] / dataTypeSize); // 发送目标（对称路径为对端 input）
            DataSlice rxSrcSlice = DataSlice(
                recvCclBuffAddr, rxScOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize); // 接收源（对称路径为对端 input）
            DataSlice rxDstSlice = DataSlice(
                localAddr, rxScOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize); // 接收目标（对称路径归约到本端 input）
            txSrcSlices.emplace_back(txSrcSlice);
            txDstSlices.emplace_back(txDstSlice);
            rxSrcSlices.emplace_back(rxSrcSlice);
            rxDstSlices.emplace_back(rxDstSlice);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::RunNHR(const std::vector<ThreadHandle>& threads, u32 channelIdx)
{
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    CHK_PRT_RET(
        threads.empty(),
        HCCL_ERROR("[InsTempReduceScatterOmniPipeNHR][RunNHR] no thread is available for NHR communication."),
        HcclResult::HCCL_E_INTERNAL);

    if (templateRankSize_ <= 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    const bool isPcieProtocol = IsPcieProtocol(channels_);
    // 步进参数，片数由inputOmniPipeSliceStride确定
    const u64 repeatNum = std::max<u64>(1, tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[0].size());

    // 预计算步骤列表（算法序）
    std::vector<AicpuNHRStepInfo> steps;
    CHK_RET(GetStepInfoList(steps));
    for (const AicpuNHRStepInfo& stepInfo : steps) {
        CHK_RET(RunNHRStep(threads, channelIdx, stepInfo, dataTypeSize, repeatNum, isPcieProtocol));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::RunNHRStep(
    const std::vector<ThreadHandle>& threads, u32 channelIdx, const AicpuNHRStepInfo& stepInfo, u32 dataTypeSize,
    u64 repeatNum, bool isPcieProtocol)
{
    u32 recvFromRank = 0;
    u32 sendToRank = 0;
    CHK_RET(ValidateNHRStepResources(stepInfo, recvFromRank, sendToRank));
    const ChannelInfo linkRecv = channels_.at(recvFromRank).at(channelIdx);
    const ChannelInfo linkSend = channels_.at(sendToRank).at(channelIdx);
    HCCL_DEBUG(
        "[InsTempReduceScatterOmniPipeNHR][RunNHR] selected channels for current step, "
        "step[%u], channelIdx[%u], recvFromRank[%u], sendToRank[%u], recvChannelRank[%u], "
        "sendChannelRank[%u].",
        stepInfo.step, channelIdx, recvFromRank, sendToRank, linkRecv.remoteRank, linkSend.remoteRank);

    void* recvRemoteAddr = nullptr;
    void* sendRemoteAddr = nullptr;
    CHK_RET(GetNHRRemoteAddrs(recvFromRank, sendToRank, linkRecv, linkSend, recvRemoteAddr, sendRemoteAddr));
    std::vector<DataSlice> txSrcSlices, txDstSlices, rxSrcSlices, rxDstSlices;
    CHK_RET(GetNHRDataSize(
        stepInfo, channelIdx, sendRemoteAddr, recvRemoteAddr, dataTypeSize, repeatNum, txSrcSlices, txDstSlices,
        rxSrcSlices, rxDstSlices));
    const SendRecvReduceInfo info{
        {linkSend, linkRecv}, {{txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices}}, dataType_, reduceOp_};
    return ExchangeNHRStep(threads, channelIdx, stepInfo, recvFromRank, sendToRank, info, isPcieProtocol);
}

HcclResult InsTempReduceScatterOmniPipeNHR::ValidateNHRStepResources(
    const AicpuNHRStepInfo& stepInfo, u32& recvFromRank, u32& sendToRank)
{
    recvFromRank = subCommRanks_[0].at(stepInfo.fromRank);
    sendToRank = subCommRanks_[0].at(stepInfo.toRank);
    CHK_PRT_RET(
        recvFromRank == static_cast<u32>(-1) || sendToRank == static_cast<u32>(-1),
        HCCL_ERROR(
            "[InsTempReduceScatterOmniPipeNHR][RunNHR] failed to map algorithm ranks to user ranks, "
            "fromAlgRank[%u], toAlgRank[%u], step[%u].",
            stepInfo.fromRank, stepInfo.toRank, stepInfo.step),
        HcclResult::HCCL_E_INTERNAL);
    CHK_PRT_RET(
        channels_.count(recvFromRank) == 0 || channels_.count(sendToRank) == 0 || channels_.at(recvFromRank).empty()
            || channels_.at(sendToRank).empty(),
        HCCL_ERROR(
            "[InsTempReduceScatterOmniPipeNHR][RunNHR] required communication channel is missing, "
            "recvFromRank[%u], sendToRank[%u], step[%u].",
            recvFromRank, sendToRank, stepInfo.step),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::GetNHRRemoteAddrs(
    u32 recvFromRank, u32 sendToRank, const ChannelInfo& linkRecv, const ChannelInfo& linkSend, void*& recvRemoteAddr,
    void*& sendRemoteAddr)
{
    if (!supportSymmetricMemory_) {
        sendRemoteAddr = linkSend.remoteCclMem.addr;
        recvRemoteAddr = linkRecv.remoteCclMem.addr;
        return HcclResult::HCCL_SUCCESS;
    }
    HcclResult ret = HcclSymWinGetPeerPointer(inputSymWindow_, inputOffset_, sendToRank, &sendRemoteAddr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || sendRemoteAddr == nullptr,
        HCCL_ERROR(
            "[InsTempReduceScatterOmniPipeNHR][RunNHR] failed to get peer input pointer for "
            "the send target, sendToRank[%u], ret[%d], ptr[%p].",
            sendToRank, ret, sendRemoteAddr),
        HcclResult::HCCL_E_INTERNAL);
    ret = HcclSymWinGetPeerPointer(inputSymWindow_, inputOffset_, recvFromRank, &recvRemoteAddr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || recvRemoteAddr == nullptr,
        HCCL_ERROR(
            "[InsTempReduceScatterOmniPipeNHR][RunNHR] failed to get peer input pointer for "
            "the receive source, recvFromRank[%u], ret[%d], ptr[%p].",
            recvFromRank, ret, recvRemoteAddr),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterOmniPipeNHR::ExchangeNHRStep(
    const std::vector<ThreadHandle>& threads, u32 channelIdx, const AicpuNHRStepInfo& stepInfo, u32 recvFromRank,
    u32 sendToRank, const SendRecvReduceInfo& info, bool isPcieProtocol)
{
    if (isPcieProtocol) {
        CHK_PRT_RET(
            SendRecvReadReduce(info, threads[channelIdx]),
            HCCL_ERROR(
                "[InsTempReduceScatterOmniPipeNHR][RunNHR] read-reduce communication failed, "
                "step[%u], channelIdx[%u], recvFromRank[%u], sendToRank[%u].",
                stepInfo.step, channelIdx, recvFromRank, sendToRank),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            SendRecvWriteReduce(info, threads[channelIdx]),
            HCCL_ERROR(
                "[InsTempReduceScatterOmniPipeNHR][RunNHR] write-reduce communication failed, "
                "step[%u], channelIdx[%u], recvFromRank[%u], sendToRank[%u].",
                stepInfo.step, channelIdx, recvFromRank, sendToRank),
            HcclResult::HCCL_E_INTERNAL);
    }
    return HcclResult::HCCL_SUCCESS;
}

//  计算每轮收发的对端以及slice编号
HcclResult InsTempReduceScatterOmniPipeNHR::GetStepInfoList(std::vector<AicpuNHRStepInfo>& stepInfoList)
{
    // 将本 rank 号转换成算法使用的索引号
    u32 u32x = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], u32x));
    stepInfoList.clear();
    u32 nSteps = GetNHRStepNum(templateRankSize_);
    stepInfoList.resize(nSteps);

    for (u32 step = 0; step < nSteps; step++) {
        // 计算通信对象
        u32 deltaRank = 1 << step;
        u32 sendTo = (u32x + templateRankSize_ - deltaRank) % templateRankSize_;
        u32 recvFrom = (u32x + deltaRank) % templateRankSize_;

        // 数据份数和数据编号增量
        u32 nSlices = (templateRankSize_ - 1 + (1 << step)) / (1 << (step + 1));
        u32 deltaSliceIndex = 1 << (step + 1);
        u32 rxSliceIdx = u32x;
        u32 txSliceIdx = sendTo;

        AicpuNHRStepInfo& currStepInfo = stepInfoList[step];
        currStepInfo.step = step;
        currStepInfo.toRank = sendTo;
        currStepInfo.myRank = u32x;
        currStepInfo.fromRank = recvFrom;
        currStepInfo.nSlices = nSlices;

        // 计算本rank在每轮收/发中的slice编号
        currStepInfo.txSliceIdxs.reserve(nSlices);
        currStepInfo.rxSliceIdxs.reserve(nSlices);
        for (u32 i = 0; i < nSlices; i++) {
            currStepInfo.txSliceIdxs.push_back(txSliceIdx);
            currStepInfo.rxSliceIdxs.push_back(rxSliceIdx);
            HCCL_DEBUG(
                "[InsTempReduceScatterOmniPipeNHR][GetStepInfoList] build slice mapping for NHR step, "
                "step[%u], sliceOrder[%u], txSliceIdx[%u], rxSliceIdx[%u].",
                step, i, txSliceIdx, rxSliceIdx);
            txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
            rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
