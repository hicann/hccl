/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_nhr_nda_inter.h"

#include "alg_data_trans_wrapper.h"
#include "alg_v2_template_register.h"
#include "channel.h"
#include "dpu_alg_data_trans_wrapper.h"

namespace ops_hccl {

InsTempAllGatherNhrNdaInter::InsTempAllGatherNhrNdaInter(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempAllGatherNhrNdaInter::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> interChannels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, interChannels));
    resourceRequest.channels.push_back(interChannels);
    HCCL_INFO(
        "[InsTempAllGatherNhrNdaInter][CalcRes] slaveThreadNum[%u] notifyNumOnMainThread[%u] "
        "interChannels[%u].",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, interChannels.size());
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherNhrNdaInter::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    return templateRankSize_;
}

HcclResult InsTempAllGatherNhrNdaInter::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    if (templateResource.threads.empty()) {
        HCCL_ERROR("[InsTempAllGatherNhrNdaInter] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(LocalDataCopy(tempAlgParams, templateResource));
    CHK_PRT_RET(
        HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS,
        HCCL_ERROR("[InsTempAllGatherNhrNdaInter] failed to end batch mode, tag[%s].", param.algTag), HCCL_E_INTERNAL);

    CHK_RET(RunNHR(tempAlgParams, templateResource.channels, templateResource.threads[0]));

    CHK_PRT_RET(
        HcommBatchModeStart(param.algTag) != HCCL_SUCCESS,
        HCCL_ERROR("[InsTempAllGatherNhrNdaInter] failed to start batch mode, tag[%s].", param.algTag),
        HCCL_E_INTERNAL);
    CHK_RET(PostLocalCopy(tempAlgParams, templateResource));
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNhrNdaInter::GetStepInfo(u32 step, u32 stepNum, AicpuNHRStepInfo& stepInfo) const
{
    u32 rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));
    stepInfo.step = step;
    stepInfo.myRank = rankIdx;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();

    const u32 deltaRank = 1U << (stepNum - 1 - step);
    stepInfo.fromRank = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    stepInfo.toRank = (rankIdx + deltaRank) % templateRankSize_;

    const u32 sliceNum = (templateRankSize_ - 1 + (1U << (stepNum - 1 - step))) / (1U << (stepNum - step));
    const u32 deltaSliceIdx = 1U << (stepNum - step);
    u32 txSliceIdx = rankIdx;
    u32 rxSliceIdx = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
    stepInfo.nSlices = sliceNum;

    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + templateRankSize_ - deltaSliceIdx) % templateRankSize_;
        rxSliceIdx = (rxSliceIdx + templateRankSize_ - deltaSliceIdx) % templateRankSize_;
    }
    return HCCL_SUCCESS;
}

u32 InsTempAllGatherNhrNdaInter::GetRankFromMap(u32 rankIdx) const { return subCommRanks_[0].at(rankIdx); }

HcclResult InsTempAllGatherNhrNdaInter::LocalDataCopy(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    u32 algRankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], algRankIdx));
    const u64 sliceSize = tempAlgParams.allRankSliceSize.at(algRankIdx);
    const u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(algRankIdx);
    const u64 sliceOffset = tempAlgParams.allRankDispls.at(algRankIdx);
    if (sliceSize == 0) {
        HCCL_INFO("[InsTempAllGatherNhrNdaInter][LocalDataCopy] Rank[%u] has no data.", myRank_);
        return HCCL_SUCCESS;
    }

    for (u64 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; ++repeatIdx) {
        const u64 inputOffset = tempAlgParams.buffInfo.inBuffBaseOff + repeatIdx * tempAlgParams.inputRepeatStride;
        const u64 scratchOffset
            = tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride + sliceOffset;
        DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr, inputOffset, sliceSize, sliceCount);
        DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, scratchOffset, sliceSize, sliceCount);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNhrNdaInter::RunNHR(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const ThreadHandle& thread) const
{
    const u32 stepNum = GetNHRStepNum(templateRankSize_);
    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; ++repeatIdx) {
        for (u32 step = 0; step < stepNum; ++step) {
            AicpuNHRStepInfo stepInfo;
            CHK_RET(GetStepInfo(step, stepNum, stepInfo));
            stepInfo.toRank = GetRankFromMap(stepInfo.toRank);
            stepInfo.fromRank = GetRankFromMap(stepInfo.fromRank);

            auto txIt = channels.find(stepInfo.toRank);
            auto rxIt = channels.find(stepInfo.fromRank);
            if (txIt == channels.end() || rxIt == channels.end() || txIt->second.empty() || rxIt->second.empty()) {
                HCCL_ERROR(
                    "[InsTempAllGatherNhrNdaInter][RunNHR] channel not found, toRank[%u], fromRank[%u].",
                    stepInfo.toRank, stepInfo.fromRank);
                return HCCL_E_PARA;
            }

            const ChannelInfo& txChannel = txIt->second[0];
            const ChannelInfo& rxChannel = rxIt->second[0];
            std::vector<DataSlice> txSrcSlices;
            std::vector<DataSlice> txDstSlices;
            std::vector<DataSlice> rxSrcSlices;
            std::vector<DataSlice> rxDstSlices;
            for (u32 txSliceIdx : stepInfo.txSliceIdxs) {
                const u64 sliceSize = tempAlgParams.allRankSliceSize.at(txSliceIdx);
                if (sliceSize == 0) {
                    continue;
                }
                const u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(txSliceIdx);
                const u64 sliceOffset = tempAlgParams.buffInfo.hcclBuffBaseOff
                                        + repeatIdx * tempAlgParams.outputRepeatStride
                                        + tempAlgParams.allRankDispls.at(txSliceIdx);
                txSrcSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, sliceOffset, sliceSize, sliceCount);
                txDstSlices.emplace_back(txChannel.remoteCclMem.addr, sliceOffset, sliceSize, sliceCount);
            }
            for (u32 rxSliceIdx : stepInfo.rxSliceIdxs) {
                const u64 sliceSize = tempAlgParams.allRankSliceSize.at(rxSliceIdx);
                if (sliceSize == 0) {
                    continue;
                }
                const u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(rxSliceIdx);
                const u64 sliceOffset = tempAlgParams.buffInfo.hcclBuffBaseOff
                                        + repeatIdx * tempAlgParams.outputRepeatStride
                                        + tempAlgParams.allRankDispls.at(rxSliceIdx);
                rxSrcSlices.emplace_back(rxChannel.remoteCclMem.addr, sliceOffset, sliceSize, sliceCount);
                rxDstSlices.emplace_back(tempAlgParams.buffInfo.hcclBuff.addr, sliceOffset, sliceSize, sliceCount);
            }

            TxRxChannels sendRecvChannels(txChannel, rxChannel);
            TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
            CHK_PRT_RET(
                SendRecvWriteForNda(sendRecvInfo, thread) != HCCL_SUCCESS,
                HCCL_ERROR("[InsTempAllGatherNhrNdaInter] SendRecvWriteForNda failed, step[%u].", step),
                HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNhrNdaInter::PostLocalCopy(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    if (tempAlgParams.buffInfo.outputPtr == nullptr) {
        return HCCL_SUCCESS;
    }

    for (u32 repeatIdx = 0; repeatIdx < tempAlgParams.repeatNum; ++repeatIdx) {
        const u64 outputBaseOffset
            = tempAlgParams.buffInfo.outBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride;
        const u64 scratchBaseOffset
            = tempAlgParams.buffInfo.hcclBuffBaseOff + repeatIdx * tempAlgParams.outputRepeatStride;
        for (u32 algRank = 0; algRank < subCommRanks_[0].size(); ++algRank) {
            const u64 sliceSize = tempAlgParams.allRankSliceSize.at(algRank);
            if (sliceSize == 0) {
                continue;
            }
            const u64 sliceCount = tempAlgParams.allRankProcessedDataCount.at(algRank);
            const u64 sliceOffset = tempAlgParams.allRankDispls.at(algRank);
            DataSlice srcSlice(
                tempAlgParams.buffInfo.hcclBuff.addr, scratchBaseOffset + sliceOffset, sliceSize, sliceCount);
            DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, outputBaseOffset + sliceOffset, sliceSize, sliceCount);
            CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempAllGatherNhrNdaInter", InsTempAllGatherNhrNdaInter);

} // namespace ops_hccl
