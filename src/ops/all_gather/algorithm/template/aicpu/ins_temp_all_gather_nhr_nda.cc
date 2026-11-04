/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_nhr_nda.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "channel.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {
InsTempAllGatherNHRNDA::InsTempAllGatherNHRNDA(
    const OpParam& param, const uint32_t rankId, const std::vector<std::vector<uint32_t>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempAllGatherNHRNDA::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO(
        "[InsTempAllGatherNHRNDA][CalcRes]slaveThreadNum[%u] notifyNumOnMainThread[%u] level1Channels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherNHRNDA::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    u64 scratchMultiple = templateRankSize_;
    HCCL_INFO("[InsTempAllGatherNHRNDA][CalcScratchMultiple] templateScratchMultiplier[%llu]", scratchMultiple);
    return scratchMultiple;
}

HcclResult InsTempAllGatherNHRNDA::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherNHRNDA] Run Start");

    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempAllGatherNHRNDA] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(LocalDataCopy(tempAlgParams, templateResource));

    // 转换成eager-mode，保障AICPU指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempAllGatherNHRNDA] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(RunNHR(tempAlgParams, templateResource.channels, templateResource.threads[0]));

    // 将执行模式转换回到batch
    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempAllGatherNHRNDA] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PostLocalCopy(tempAlgParams, templateResource));

    HCCL_INFO("[InsTempAllGatherNHRNDA] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRNDA::GetStepInfo(uint32_t step, uint32_t nSteps, AicpuNHRStepInfo& stepInfo) const
{
    uint32_t rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();

    // 计算通信对象
    uint32_t deltaRank = 1 << (nSteps - 1 - step);
    uint32_t recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    uint32_t sendTo = (rankIdx + deltaRank) % templateRankSize_;

    // 数据份数和数据编号增量
    uint32_t nSlices = (templateRankSize_ - 1 + (1 << (nSteps - 1 - step))) / (1 << (nSteps - step));
    uint32_t deltaSliceIndex = 1 << (nSteps - step);
    uint32_t txSliceIdx = rankIdx;
    uint32_t rxSliceIdx = (rankIdx - (1 << (nSteps - 1 - step)) + templateRankSize_) % templateRankSize_;

    stepInfo.nSlices = nSlices;
    stepInfo.fromRank = recvFrom;
    stepInfo.toRank = sendTo;

    for (uint32_t i = 0; i < nSlices; i++) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);

        HCCL_DEBUG(
            "[InsTempAllGatherNHRNDA][GetStepInfo] i[%u] txSliceIdx[%u] rxSliceIdx[%u]", i, txSliceIdx, rxSliceIdx);

        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIndex) % templateRankSize_;
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 InsTempAllGatherNHRNDA::GetRankFromMap(const uint32_t rankIdx) const { return subCommRanks_[0].at(rankIdx); }

HcclResult
InsTempAllGatherNHRNDA::LocalDataCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    uint32_t algRankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRankIdx));

    for (uint64_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 inBaseOff = tempAlgParams.buffInfo.inBuffBaseOff + rpt * tempAlgParams.inputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBaseoff = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        const u64 inOff = tempAlgParams.inputSliceStride * algRankIdx + inBaseOff;
        const u64 scOff = tempAlgParams.sliceSize * algRankIdx + scratchBaseoff;

        DataSlice srcSlices(tempAlgParams.buffInfo.inputPtr, inOff, tempAlgParams.sliceSize, tempAlgParams.count);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, scOff, tempAlgParams.sliceSize, tempAlgParams.count);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlices, dstSlice));
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRNDA::RunNHR(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const ThreadHandle& thread) const
{
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    for (uint32_t rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        for (uint32_t step = 0; step < nSteps; ++step) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, nSteps, stepInfo));

            HCCL_DEBUG(
                "[InsTempAllGatherNHRNDA] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] "
                "nSlices[%u]",
                myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

            stepInfo.toRank = GetRankFromMap(stepInfo.toRank);
            stepInfo.fromRank = GetRankFromMap(stepInfo.fromRank);
            HCCL_INFO(
                "[InsTempAllGatherNHRNDA][RunNHR] converted toRank=%u fromRank=%u channelsSize=%zu", stepInfo.toRank,
                stepInfo.fromRank, channels.size());

            bool hasTx = stepInfo.txSliceIdxs.size() > 0;
            bool hasRx = stepInfo.rxSliceIdxs.size() > 0;
            if (!hasTx && !hasRx) {
                continue;
            }

            auto txIt = hasTx ? channels.find(stepInfo.toRank) : channels.end();
            auto rxIt = hasRx ? channels.find(stepInfo.fromRank) : channels.end();
            if (hasTx && txIt == channels.end()) {
                HCCL_ERROR(
                    "[InsTempAllGatherNHRNDA] tx channel not found for rank=%u, channels size=%zu", stepInfo.toRank,
                    channels.size());
                return HCCL_E_PARA;
            }
            if (hasRx && rxIt == channels.end()) {
                HCCL_ERROR(
                    "[InsTempAllGatherNHRNDA] rx channel not found for rank=%u, channels size=%zu", stepInfo.fromRank,
                    channels.size());
                return HCCL_E_PARA;
            }

            const ChannelInfo& txCh = txIt->second[0];
            const ChannelInfo& rxCh = rxIt->second[0];
            void* sendCclBuffAddr = txCh.remoteCclMem.addr;
            void* recvCclBuffAddr = rxCh.remoteCclMem.addr;

            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;

            // 构造发送 slice
            for (u32 i = 0; i < stepInfo.txSliceIdxs.size(); i++) {
                u32 txId = stepInfo.txSliceIdxs.at(i);
                u64 sliceSize = tempAlgParams.allRankSliceSize.at(txId);
                u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(txId);
                u64 sliceOffset = tempAlgParams.allRankDispls.at(txId);
                u64 srcDstOffset
                    = rpt * templateRankSize_ * sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff + sliceOffset;

                if (sliceSize != 0) {
                    txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, srcDstOffset, sliceSize, sliceCount);
                    txDstSlices.emplace_back(sendCclBuffAddr, srcDstOffset, sliceSize, sliceCount);
                }
            }

            // 构造接收 slice
            for (u32 i = 0; i < stepInfo.rxSliceIdxs.size(); i++) {
                u32 rxId = stepInfo.rxSliceIdxs.at(i);
                u64 sliceSize = tempAlgParams.allRankSliceSize.at(rxId);
                u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(rxId);
                u64 sliceOffset = tempAlgParams.allRankDispls.at(rxId);
                u64 srcDstOffset
                    = rpt * templateRankSize_ * sliceSize + tempAlgParams.buffInfo.hcclBuffBaseOff + sliceOffset;

                if (sliceSize != 0) {
                    rxSrcSlices.emplace_back(recvCclBuffAddr, srcDstOffset, sliceSize, sliceCount);
                    rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, srcDstOffset, sliceSize, sliceCount);
                }
            }

            TxRxChannels sendRecvChannels(txCh, rxCh);
            TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

            CHK_PRT_RET(
                SendRecvWriteForNda(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempAllGatherNHRNDA] SendRecvWriteForNda failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
InsTempAllGatherNHRNDA::PostLocalCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    for (u32 rpt = 0; rpt < tempAlgParams.repeatNum; ++rpt) {
        const u64 outBaseOff = tempAlgParams.buffInfo.outBuffBaseOff + rpt * tempAlgParams.outputRepeatStride;
        const u64 scratchRepeatStride = tempAlgParams.sliceSize * templateRankSize_;
        const u64 scratchBase = tempAlgParams.buffInfo.hcclBuffBaseOff + rpt * scratchRepeatStride;

        for (auto rank : subCommRanks_[0]) {
            u32 algRank = 0;
            CHK_RET(GetAlgRank(rank, subCommRanks_[0], algRank));
            u64 scratchOffset = tempAlgParams.sliceSize * algRank + scratchBase;
            u64 outOffset = tempAlgParams.outputSliceStride * algRank + outBaseOff;
            DataSlice srcSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, tempAlgParams.sliceSize, tempAlgParams.count);
            DataSlice dstSlice(
                tempAlgParams.buffInfo.outputPtr, outOffset, tempAlgParams.sliceSize, tempAlgParams.count);
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempAllGatherNHRNDA", InsTempAllGatherNHRNDA);
} // namespace ops_hccl
