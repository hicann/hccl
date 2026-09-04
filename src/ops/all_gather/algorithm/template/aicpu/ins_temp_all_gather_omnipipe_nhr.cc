/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_omnipipe_nhr.h"
#include "alg_data_trans_wrapper.h"
#include "omnipipe_template_utils.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "hccl_sym_win.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */

namespace ops_hccl {
InsTempAllGatherOmniPipeNHR::InsTempAllGatherOmniPipeNHR(
    const OpParam& param,
    const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempAllGatherNHR(param, rankId, subCommRanks)
{}

InsTempAllGatherOmniPipeNHR::~InsTempAllGatherOmniPipeNHR() {}

HcclResult InsTempAllGatherOmniPipeNHR::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO(
        "[InsTempAllGatherOmniPipeNHR][KernelRun] start NHR all-gather template, rank[%u], symmetric[%d].", myRank_,
        tempAlgParams.enableRemoteMemAccess);
    if (templateRankSize_ == 1) {
        HCCL_INFO(
            "[InsTempAllGatherOmniPipeNHR][KernelRun] skip communication for single-rank template, rank[%u].", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    // NHR 按每个对端配置的通道数并行处理数据分片。
    HCCL_DEBUG(
        "[InsTempAllGatherOmniPipeNHR][KernelRun] prepare multi-channel NHR, channelsPerRank[%u].", channelsPerRank_);
    InitKernelParams(param, tempAlgParams);

    CHK_RET(PrepareOmniPipeDataSplitForMultiChannel(
        static_cast<CommonAlgTemplateBase*>(this), tempAlgParams_, dataType_, templateResource, dataSplitVec_,
        dataOffsetVec_));

    CHK_RET(SyncInterThreads(templateResource.threads, true));
    HCCL_DEBUG(
        "[InsTempAllGatherOmniPipeNHR][KernelRun] launch NHR channels, channelsPerRank[%u], "
        "templateRankSize[%u].",
        channelsPerRank_, templateRankSize_);
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(RunAllGatherNHR(templateResource.threads, templateResource.channels, channelIdx));
    }
    CHK_RET(SyncInterThreads(templateResource.threads, false));
    CHK_RET(SyncInterThreads(templateResource.threads, true));
    HCCL_DEBUG(
        "[InsTempAllGatherOmniPipeNHR][KernelRun] check last-step scratch data for output copy, "
        "channelsPerRank[%u], templateRankSize[%u], lastStepCopy[%d].",
        channelsPerRank_, templateRankSize_, lastStepNhrCopy_);
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        if (lastStepNhrCopy_) {
            DoLastStepCopyNhr(templateResource.threads, templateResource.channels, channelIdx);
        }
    }
    CHK_RET(SyncInterThreads(templateResource.threads, false));
    HCCL_INFO("[InsTempAllGatherOmniPipeNHR][KernelRun] finish NHR all-gather template, rank[%u].", myRank_);
    return HcclResult::HCCL_SUCCESS;
}

void InsTempAllGatherOmniPipeNHR::InitKernelParams(const OpParam& param, const TemplateDataParams& tempAlgParams)
{
    threadNum_ = GetThreadNum();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    tempAlgParams_.buffInfo.outputPtr = param.outputPtr;
    omniLastStepRead_ = tempAlgParams.omniLastStepRead_;
    lastStepNhrCopy_ = false;
    inputSymWindow_ = param.inputSymWindow;
    outputSymWindow_ = param.outputSymWindow;
    inputOffset_ = param.inputOffset;
    outputOffset_ = param.outputOffset;
    enableRemoteMemAccess_ = param.supportSymmetricMemory;
}

HcclResult InsTempAllGatherOmniPipeNHR::SyncInterThreads(const std::vector<ThreadHandle>& threads, bool mainToSub)
{
    if (threadNum_ <= 1) {
        return HcclResult::HCCL_SUCCESS;
    }
    std::vector<ThreadHandle> subThreads(threads.begin() + 1, threads.end());
    if (mainToSub) {
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        return PreSyncInterThreads(threads[0], subThreads, notifyIdxMainToSub_);
    }
    GetNotifyIdxSubToMain(notifyIdxSubToMain_);
    return PostSyncInterThreads(threads[0], subThreads, notifyIdxSubToMain_);
}

namespace {
    // NHR 单步通信中一对 tx/rx slice 在 ccl scratch 与 user output 布局上的读写偏移。
    struct NhrSliceOffsets {
        u64 txScratchBase = 0; // ccl scratch 上发送 slice 基址（仅供日志打印）
        u64 rxScratchBase = 0; // ccl scratch 上接收 slice 基址（仅供日志打印）
        u64 txScratchOff = 0;  // ccl scratch 上发送 slice 偏移
        u64 rxScratchOff = 0;  // ccl scratch 上接收 slice 偏移
        u64 txOutOff = 0;      // user output 上发送 slice 偏移（含已处理数据量推进）
        u64 rxOutOff = 0;      // user output 上接收 slice 偏移（含已处理数据量推进）
    };

    NhrSliceOffsets CalcNhrSliceOffsets(
        const TemplateDataParams& tempAlgParams, const std::vector<std::vector<std::vector<u64>>>& dataOffsetVec,
        u32 txIdx, u32 rxIdx, u32 rpt, u32 channelIdx, u32 dataTypeSize, bool needOutputOffset)
    {
        NhrSliceOffsets off;
        off.txScratchBase = tempAlgParams.buffInfo.inBuffBaseOff
                            + tempAlgParams.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt]
                            + dataOffsetVec[txIdx][rpt][channelIdx];
        off.rxScratchBase = tempAlgParams.buffInfo.outBuffBaseOff
                            + tempAlgParams.stepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt]
                            + dataOffsetVec[rxIdx][rpt][channelIdx];
        off.txScratchOff = off.txScratchBase + tempAlgParams.stepSliceInfo.stepInputSliceStride[txIdx];
        off.rxScratchOff = off.rxScratchBase + tempAlgParams.stepSliceInfo.stepInputSliceStride[rxIdx];
        if (!needOutputOffset) {
            return off;
        }

        const u64 txOutBase = tempAlgParams.buffInfo.inBuffBaseOff
                              + tempAlgParams.omniReadDstStepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt]
                              + dataOffsetVec[txIdx][rpt][channelIdx];
        const u64 rxOutBase = tempAlgParams.buffInfo.outBuffBaseOff
                              + tempAlgParams.omniReadDstStepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt]
                              + dataOffsetVec[rxIdx][rpt][channelIdx];
        off.txOutOff = txOutBase + tempAlgParams.omniReadDstStepSliceInfo.stepInputSliceStride[txIdx]
                       + tempAlgParams.processedDataCount * dataTypeSize;
        off.rxOutOff = rxOutBase + tempAlgParams.omniReadDstStepSliceInfo.stepInputSliceStride[rxIdx]
                       + tempAlgParams.processedDataCount * dataTypeSize;
        return off;
    }
} // namespace

HcclResult InsTempAllGatherOmniPipeNHR::DoLastStepCopyNhr(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const u32& channelIdx)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 nSteps = GetNHRStepNum(templateRankSize_); // NHR 通信步数， celi(log2(rankSize))
    bool isPcieProtocal = IsPcieProtocol(channels);      // 判断是否存在pcie链路
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 step = 0; step < nSteps - 1; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo)); // 计算当前step要通信的卡，数据
        for (u32 i = 0; i < stepInfo.nSlices; ++i) {
            for (u32 rpt = 0; rpt < tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size(); ++rpt) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];
                const NhrSliceOffsets off = CalcNhrSliceOffsets(
                    tempAlgParams_, dataOffsetVec_, txIdx, rxIdx, rpt, channelIdx, dataTypeSize, true);
                DataSlice rxSrcSlices = DataSlice(
                    tempAlgParams_.buffInfo.hcclBuff.addr, off.rxScratchOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                    dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize);
                DataSlice rxDstSlices = DataSlice(
                    tempAlgParams_.buffInfo.outputPtr, off.rxOutOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                    dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize);
                CHK_RET(LocalCopy(threads[channelsPerRank_ + channelIdx], rxSrcSlices, rxDstSlices));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOmniPipeNHR::RunAllGatherNHR(
    const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const u32& channelIdx)
{
    HCCL_INFO(
        "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] start NHR slice exchange, rank[%u], "
        "channelIdx[%u], symmetric[%d].",
        myRank_, channelIdx, tempAlgParams_.enableRemoteMemAccess);
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 nSteps = GetNHRStepNum(templateRankSize_); // NHR 通信步数， celi(log2(rankSize))
    bool isPcieProtocal = IsPcieProtocol(channels);      // 判断是否存在pcie链路
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 step = 0; step < nSteps; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo)); // 计算当前step要通信的卡，数据

        const ChannelInfo& channelRecv = channels.at(GetRankFromMap(stepInfo.fromRank))[channelIdx];
        const ChannelInfo& channelSend = channels.at(GetRankFromMap(stepInfo.toRank))[channelIdx];
        // 普通步骤在 ccl scratch 间传输，末步读可直接落到 user output；
        // 对称路径的所有步骤均直接在本端与对端的 user output 窗口间传输。

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        void* sendCclBuffAddr = channelSend.remoteCclMem.addr;
        void* recvCclBuffAddr = channelRecv.remoteCclMem.addr;

        // 对称路径分别获取发送端和接收端 rank 的远端 output 地址。
        u32 recvRank = GetRankFromMap(stepInfo.fromRank);
        u32 sendRank = GetRankFromMap(stepInfo.toRank);
        void* sendRemoteOut = nullptr;
        void* recvRemoteOut = nullptr;
        if (enableRemoteMemAccess_) {
            HcclResult ret = GetSymWinRemoteMem(outputSymWindow_, outputOffset_, sendRank, &sendRemoteOut);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS || sendRemoteOut == nullptr,
                HCCL_ERROR(
                    "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] failed to get peer output "
                    "pointer for send target, remoteRank[%u], ret[%d], peerOutput[%p].",
                    sendRank, ret, sendRemoteOut),
                HcclResult::HCCL_E_INTERNAL);

            ret = GetSymWinRemoteMem(outputSymWindow_, outputOffset_, recvRank, &recvRemoteOut);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS || recvRemoteOut == nullptr,
                HCCL_ERROR(
                    "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] failed to get peer output "
                    "pointer for receive source, remoteRank[%u], ret[%d], peerOutput[%p].",
                    recvRank, ret, recvRemoteOut),
                HcclResult::HCCL_E_INTERNAL);

            HCCL_INFO(
                "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] peer output pointers are ready for "
                "symmetric communication, sendRank[%u], sendPeerOutput[%p], recvRank[%u], recvPeerOutput[%p].",
                sendRank, sendRemoteOut, recvRank, recvRemoteOut);
        }

        if (omniLastStepRead_ && (step == nSteps - 1)) {
            lastStepNhrCopy_ = true;
        }
        bool isLastStepRead = omniLastStepRead_ && (step == nSteps - 1);

        HCCL_DEBUG(
            "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] build communication step, rank[%u], "
            "rankSize[%u], recvFromAlgRank[%u], sendToAlgRank[%u], step[%u], stepCount[%u], sliceCount[%u].",
            myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);
        const bool needOutputOffset = enableRemoteMemAccess_ || isLastStepRead;
        for (u32 i = 0; i < stepInfo.nSlices; ++i) {
            const u32 txIdx = stepInfo.txSliceIdxs[i];
            const u32 rxIdx = stepInfo.rxSliceIdxs[i];
            for (u32 rpt = 0; rpt < tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size(); ++rpt) {
                const NhrSliceOffsets off = CalcNhrSliceOffsets(
                    tempAlgParams_, dataOffsetVec_, txIdx, rxIdx, rpt, channelIdx, dataTypeSize, needOutputOffset);
                HCCL_DEBUG(
                    "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] calculate send scratch base, "
                    "step[%u], channelIdx[%u], sliceIdx[%u], repeatIdx[%u], sliceStride[%llu], "
                    "channelOffset[%llu], baseOffset[%llu].",
                    step, channelIdx, txIdx, rpt, tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt],
                    dataOffsetVec_[txIdx][rpt][channelIdx], off.txScratchBase);
                HCCL_DEBUG(
                    "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] calculate receive scratch base, "
                    "step[%u], channelIdx[%u], sliceIdx[%u], repeatIdx[%u], sliceStride[%llu], "
                    "channelOffset[%llu], baseOffset[%llu].",
                    step, channelIdx, rxIdx, rpt, tempAlgParams_.stepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt],
                    dataOffsetVec_[rxIdx][rpt][channelIdx], off.rxScratchBase);

                // 对称路径在本端 outputPtr 与相应对端的 output 窗口之间直接收发。
                if (enableRemoteMemAccess_) {
                    txSrcSlices.emplace_back(
                        tempAlgParams_.buffInfo.outputPtr, off.txOutOff, dataSplitVec_[txIdx][rpt][channelIdx],
                        dataSplitVec_[txIdx][rpt][channelIdx] / dataTypeSize);
                    txDstSlices.emplace_back(
                        sendRemoteOut, off.txOutOff, dataSplitVec_[txIdx][rpt][channelIdx],
                        dataSplitVec_[txIdx][rpt][channelIdx] / dataTypeSize);
                    rxSrcSlices.emplace_back(
                        recvRemoteOut, off.rxOutOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                        dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize);
                    rxDstSlices.emplace_back(
                        tempAlgParams_.buffInfo.outputPtr, off.rxOutOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                        dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize);
                } else if (isLastStepRead) {
                    txSrcSlices.emplace_back(
                        tempAlgParams_.buffInfo.outputPtr, off.txOutOff, dataSplitVec_[txIdx][rpt][channelIdx],
                        dataSplitVec_[txIdx][rpt][channelIdx] / dataTypeSize);
                    txDstSlices.emplace_back(
                        sendCclBuffAddr, off.txScratchOff, dataSplitVec_[txIdx][rpt][channelIdx],
                        dataSplitVec_[txIdx][rpt][channelIdx] / dataTypeSize);
                    rxSrcSlices.emplace_back(
                        recvCclBuffAddr, off.rxScratchOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                        dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize);
                    rxDstSlices.emplace_back(
                        tempAlgParams_.buffInfo.outputPtr, off.rxOutOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                        dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize);
                } else {
                    txSrcSlices.emplace_back(
                        tempAlgParams_.buffInfo.hcclBuff.addr, off.txScratchOff, dataSplitVec_[txIdx][rpt][channelIdx],
                        dataSplitVec_[txIdx][rpt][channelIdx] / dataTypeSize);
                    txDstSlices.emplace_back(
                        sendCclBuffAddr, off.txScratchOff, dataSplitVec_[txIdx][rpt][channelIdx],
                        dataSplitVec_[txIdx][rpt][channelIdx] / dataTypeSize);
                    rxSrcSlices.emplace_back(
                        recvCclBuffAddr, off.rxScratchOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                        dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize);
                    rxDstSlices.emplace_back(
                        tempAlgParams_.buffInfo.hcclBuff.addr, off.rxScratchOff, dataSplitVec_[rxIdx][rpt][channelIdx],
                        dataSplitVec_[rxIdx][rpt][channelIdx] / dataTypeSize);
                }
            }
        }
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        if (!isLastStepRead) {
            if (isPcieProtocal) {
                CHK_PRT_RET(
                    SendRecvRead(sendRecvInfo, threads[channelIdx]),
                    HCCL_ERROR(
                        "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] PCIe read exchange "
                        "failed, step[%u], channelIdx[%u], recvRank[%u], sendRank[%u].",
                        step, channelIdx, recvRank, sendRank),
                    HcclResult::HCCL_E_INTERNAL);
            } else {
                CHK_PRT_RET(
                    SendRecvWrite(sendRecvInfo, threads[channelIdx]),
                    HCCL_ERROR(
                        "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] write exchange failed, "
                        "step[%u], channelIdx[%u], recvRank[%u], sendRank[%u].",
                        step, channelIdx, recvRank, sendRank),
                    HcclResult::HCCL_E_INTERNAL);
            }
        } else {
            CHK_PRT_RET(
                SendRecvRead(sendRecvInfo, threads[channelIdx]),
                HCCL_ERROR(
                    "[InsTempAllGatherOmniPipeNHR][RunAllGatherNHR] last-step read exchange "
                    "failed, step[%u], channelIdx[%u], recvRank[%u], sendRank[%u].",
                    step, channelIdx, recvRank, sendRank),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
