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

namespace ops_hccl {
InsTempAllGatherOmniPipeNHR::InsTempAllGatherOmniPipeNHR(const OpParam& param,
                                                         const u32 rankId,  // 传通信域的rankId，userRank
                                                         const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempAllGatherNHR(param, rankId, subCommRanks)
{
}

InsTempAllGatherOmniPipeNHR::~InsTempAllGatherOmniPipeNHR()
{
}

HcclResult InsTempAllGatherOmniPipeNHR::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
                                                TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherOmniPipeNHR] Run start");
    if (templateRankSize_ == 1) {
        HCCL_INFO("[InsTempAllGatherOmniPipeNHR] Rank [%d], template ranksize is 1.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    // 强行增加一下
    HCCL_DEBUG("MT device channelsPerRank_= %u", channelsPerRank_);
    threadNum_ = GetThreadNum();
    tempAlgParams_ = tempAlgParams;
    dataType_ = param.DataDes.dataType;
    tempAlgParams_.buffInfo.outputPtr = param.outputPtr;
    omniLastStepRead_ = tempAlgParams.omniLastStepRead_;
    
    CHK_RET(PrepareOmniPipeDataSplitForMultiChannel(static_cast<CommonAlgTemplateBase*>(this), tempAlgParams_, dataType_, templateResource, 
        dataSplitVec_, dataOffsetVec_));

    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        // todo 这里的subThreads的size是0，就是说templateResource.threads给的值不对
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    HCCL_DEBUG("MT channelsPerRank_ = %llu, templateRankSize_ = %llu", channelsPerRank_, templateRankSize_);
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        CHK_RET(RunAllGatherNHR(templateResource.threads, templateResource.channels, channelIdx));
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        // todo 这里的subThreads的size是0，就是说templateResource.threads给的值不对
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }
    HCCL_DEBUG("MT channelsPerRank_ = %llu, templateRankSize_ = %llu", channelsPerRank_, templateRankSize_);
    for (u32 channelIdx = 0; channelIdx < channelsPerRank_; channelIdx++) {
        if (lastStepNhrCopy_){
            DoLastStepCopyNhr(templateResource.threads, templateResource.channels, channelIdx);
        }
    }
    if (threadNum_ > 1) {
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    HCCL_INFO("[InsTempAllGatherOmniPipeNHR] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOmniPipeNHR::DoLastStepCopyNhr(const std::vector<ThreadHandle>& threads,
                               const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 &channelIdx)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 nSteps = GetNHRStepNum(templateRankSize_);  // NHR 通信步数， celi(log2(rankSize))
    bool isPcieProtocal = IsPcieProtocol(channels);  // 判断是否存在pcie链路
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 step = 0; step < nSteps-1; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));  // 计算当前step要通信的卡，数据
        for (u32 i = 0; i < stepInfo.nSlices; ++i) {
            for (u32 rpt = 0; rpt < tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size(); ++rpt) {
                const u32 txIdx = stepInfo.txSliceIdxs[i];
                const u32 rxIdx = stepInfo.rxSliceIdxs[i];
                u64 txScratchBase = tempAlgParams_.buffInfo.inBuffBaseOff +
                                            tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt];
                txScratchBase += dataOffsetVec_[txIdx][rpt][channelIdx];
                u64 rxScratchBase = tempAlgParams_.buffInfo.outBuffBaseOff +
                                        tempAlgParams_.stepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt];
                rxScratchBase += dataOffsetVec_[rxIdx][rpt][channelIdx];
                const u64 txScratchOff = txScratchBase + tempAlgParams_.stepSliceInfo.stepInputSliceStride[txIdx];
                const u64 rxScratchOff = rxScratchBase + tempAlgParams_.stepSliceInfo.stepInputSliceStride[rxIdx];
                u64 txOutBase = tempAlgParams_.buffInfo.inBuffBaseOff +
                                        tempAlgParams_.omniReadDstStepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt];
                txOutBase += dataOffsetVec_[txIdx][rpt][channelIdx];
                u64 rxOutBase = tempAlgParams_.buffInfo.outBuffBaseOff +
                                        tempAlgParams_.omniReadDstStepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt];
                rxOutBase += dataOffsetVec_[rxIdx][rpt][channelIdx];
                const u64 txOutOff = txOutBase + tempAlgParams_.omniReadDstStepSliceInfo.stepInputSliceStride[txIdx] + tempAlgParams_.processedDataCount*dataTypeSize;
                const u64 rxOutOff = rxOutBase + tempAlgParams_.omniReadDstStepSliceInfo.stepInputSliceStride[rxIdx] + tempAlgParams_.processedDataCount*dataTypeSize;
                DataSlice rxSrcSlices = DataSlice(tempAlgParams_.buffInfo.hcclBuff.addr, rxScratchOff,
                                            dataSplitVec_[rxIdx][rpt][channelIdx],
                                            dataSplitVec_[rxIdx][rpt][channelIdx]/ dataTypeSize);
                DataSlice rxDstSlices = DataSlice(tempAlgParams_.buffInfo.outputPtr, rxOutOff,
                                        dataSplitVec_[rxIdx][rpt][channelIdx],
                                        dataSplitVec_[rxIdx][rpt][channelIdx]/ dataTypeSize);
                CHK_RET(LocalCopy(threads[channelsPerRank_ + channelIdx], rxSrcSlices, rxDstSlices));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOmniPipeNHR::RunAllGatherNHR(const std::vector<ThreadHandle>& threads,
                                                        const std::map<u32, std::vector<ChannelInfo>>& channels, const u32 &channelIdx)
{
    HCCL_INFO("RunAllGatherNHRInsTempAllGatherOmniPipeNHR");
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 nSteps = GetNHRStepNum(templateRankSize_);  // NHR 通信步数， celi(log2(rankSize))
    bool isPcieProtocal = IsPcieProtocol(channels);  // 判断是否存在pcie链路
    const u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType_];
    for (u32 step = 0; step < nSteps; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));  // 计算当前step要通信的卡，数据

        const ChannelInfo& channelRecv = channels.at(GetRankFromMap(stepInfo.fromRank))[channelIdx];
        const ChannelInfo& channelSend = channels.at(GetRankFromMap(stepInfo.toRank))[channelIdx];
        // 构造SendRecv， 都是Scratch到Scratch的传输，没有DMA消减

        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        void* sendCclBuffAddr = channelSend.remoteCclMem.addr;
        void* recvCclBuffAddr = channelRecv.remoteCclMem.addr;

        if (omniLastStepRead_ && (step == nSteps - 1)) {
            lastStepNhrCopy_ = true;
        }

        HCCL_DEBUG(
            "[InsTempAllGatherOmniPipeNHR] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
            myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        for (u32 i = 0; i < stepInfo.nSlices; ++i) {
            const u32 txIdx = stepInfo.txSliceIdxs[i];
            const u32 rxIdx = stepInfo.rxSliceIdxs[i];
            for (u32 rpt = 0; rpt < tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size(); ++rpt) {
                u64 txScratchBase = tempAlgParams_.buffInfo.inBuffBaseOff +
                                        tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt];
                HCCL_DEBUG("MT inputOmniPipeSliceStride[%u][%u] = %llu", txIdx, rpt, tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt]);
                HCCL_DEBUG("MT txScratchBase = %llu", txScratchBase);
                txScratchBase += dataOffsetVec_[txIdx][rpt][channelIdx];
                HCCL_DEBUG("MT dataOffset_[%u] = %llu", channelIdx, dataOffsetVec_[txIdx][rpt][channelIdx]);
                u64 rxScratchBase = tempAlgParams_.buffInfo.outBuffBaseOff +
                                        tempAlgParams_.stepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt];
                HCCL_DEBUG("MT outputOmniPipeSliceStride[%u][%u] = %llu", rxIdx, rpt, tempAlgParams_.stepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt]);
                HCCL_DEBUG("MT txScratchBase = %llu", rxScratchBase);
                rxScratchBase += dataOffsetVec_[rxIdx][rpt][channelIdx];
                HCCL_DEBUG("MT dataOffset_[%u] = %llu", channelIdx, dataOffsetVec_[rxIdx][rpt][channelIdx]);
                const u64 txScratchOff = txScratchBase + tempAlgParams_.stepSliceInfo.stepInputSliceStride[txIdx];
                const u64 rxScratchOff = rxScratchBase + tempAlgParams_.stepSliceInfo.stepInputSliceStride[rxIdx];

                if (!omniLastStepRead_ || step != nSteps - 1){
                    txSrcSlices.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, txScratchOff,
                                            dataSplitVec_[txIdx][rpt][channelIdx],
                                            dataSplitVec_[txIdx][rpt][channelIdx]/ dataTypeSize);
                    txDstSlices.emplace_back(sendCclBuffAddr, txScratchOff,
                                            dataSplitVec_[txIdx][rpt][channelIdx],
                                            dataSplitVec_[txIdx][rpt][channelIdx]/ dataTypeSize);
                    rxSrcSlices.emplace_back(recvCclBuffAddr, rxScratchOff,
                                            dataSplitVec_[rxIdx][rpt][channelIdx],
                                            dataSplitVec_[rxIdx][rpt][channelIdx]/ dataTypeSize);
                    rxDstSlices.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, rxScratchOff,
                                            dataSplitVec_[rxIdx][rpt][channelIdx],
                                            dataSplitVec_[rxIdx][rpt][channelIdx]/ dataTypeSize);
                }else {
                    u64 txOutBase = tempAlgParams_.buffInfo.inBuffBaseOff +
                                        tempAlgParams_.omniReadDstStepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt];
                    HCCL_DEBUG("MT inputOmniPipeSliceStride[%u][%u] = %llu", txIdx, rpt, 
                                    tempAlgParams_.omniReadDstStepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt]);
                    HCCL_DEBUG("MT txOutBase = %llu", txOutBase);
                    txOutBase += dataOffsetVec_[txIdx][rpt][channelIdx];
                    HCCL_DEBUG("MT dataOffset_[%u] = %llu", channelIdx, dataOffsetVec_[txIdx][rpt][channelIdx]);
                    u64 rxOutBase = tempAlgParams_.buffInfo.outBuffBaseOff +
                                    tempAlgParams_.omniReadDstStepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt];
                    HCCL_DEBUG("MT outputOmniPipeSliceStride[%u][%u] = %llu", rxIdx, rpt, 
                                    tempAlgParams_.omniReadDstStepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt]);
                    HCCL_DEBUG("MT txOutBase = %llu", rxOutBase);
                    rxOutBase += dataOffsetVec_[rxIdx][rpt][channelIdx];
                    HCCL_DEBUG("MT dataOffset_[%u] = %llu", channelIdx, dataOffsetVec_[rxIdx][rpt][channelIdx]);
                    const u64 txOutOff = txOutBase + tempAlgParams_.omniReadDstStepSliceInfo.stepInputSliceStride[txIdx]
                                                   + tempAlgParams_.processedDataCount*dataTypeSize;

                    const u64 rxOutOff = rxOutBase + tempAlgParams_.omniReadDstStepSliceInfo.stepInputSliceStride[rxIdx]
                                                   + tempAlgParams_.processedDataCount*dataTypeSize;

                    txSrcSlices.emplace_back(tempAlgParams_.buffInfo.outputPtr, txOutOff,
                                            dataSplitVec_[txIdx][rpt][channelIdx],
                                            dataSplitVec_[txIdx][rpt][channelIdx]/ dataTypeSize);
                    txDstSlices.emplace_back(sendCclBuffAddr, txScratchOff,
                                            dataSplitVec_[txIdx][rpt][channelIdx],
                                            dataSplitVec_[txIdx][rpt][channelIdx]/ dataTypeSize);
                    rxSrcSlices.emplace_back(recvCclBuffAddr, rxScratchOff,
                                            dataSplitVec_[rxIdx][rpt][channelIdx],
                                            dataSplitVec_[rxIdx][rpt][channelIdx]/ dataTypeSize);
                    rxDstSlices.emplace_back(tempAlgParams_.buffInfo.outputPtr, rxOutOff,
                                            dataSplitVec_[rxIdx][rpt][channelIdx],
                                            dataSplitVec_[rxIdx][rpt][channelIdx]/ dataTypeSize);
                }
            }
        }
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        if (!omniLastStepRead_ || step!=nSteps-1){
            if (isPcieProtocal) {
                CHK_PRT_RET(SendRecvBatchRead(sendRecvInfo, threads[channelIdx]),
                        HCCL_ERROR("[InsTempAllGatherOmniPipeNHR] sendrecv failed (step=%u)", step),
                        HcclResult::HCCL_E_INTERNAL);
            }else {
                CHK_PRT_RET(SendRecvBatchWrite(sendRecvInfo, threads[channelIdx]),
                        HCCL_ERROR("[InsTempAllGatherOmniPipeNHR] sendrecv failed (step=%u)", step),
                        HcclResult::HCCL_E_INTERNAL);
            }
        }else {
            CHK_PRT_RET(SendRecvBatchRead(sendRecvInfo, threads[channelIdx]),
                    HCCL_ERROR("[InsTempAllGatherOmniPipeNHR] sendrecv failed (step=%u)", step),
                    HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl