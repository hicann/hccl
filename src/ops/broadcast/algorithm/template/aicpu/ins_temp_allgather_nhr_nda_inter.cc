/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_allgather_nhr_nda_inter.h"
#include "alg_v2_template_register.h"
#include "channel.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "exec_timeout_manager.h"

namespace ops_hccl {
InsTempAllGatherNHRNDAInter::InsTempAllGatherNHRNDAInter(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempAllGatherNHRNDAInter::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_RET(GetRes(resourceRequest));
    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestNhrHighestHostRoce(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRNDAInter::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherNHRNDAInter::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult InsTempAllGatherNHRNDAInter::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty() || templateRankSize_ != subCommRanks_[0].size(),
        HCCL_ERROR("[InsTempAllGatherNHRNDAInter] invalid sub-communicator"), HCCL_E_PARA);
    CHK_PRT_RET(
        tempAlgParams.repeatNum != 1,
        HCCL_ERROR(
            "[InsTempAllGatherNHRNDAInter] expect repeatNum[1], got[%llu]",
            static_cast<unsigned long long>(tempAlgParams.repeatNum)),
        HCCL_E_PARA);
    CHK_PRT_RET(
        tempAlgParams.allRankSliceSize.size() != templateRankSize_
            || tempAlgParams.allRankDispls.size() != templateRankSize_
            || tempAlgParams.allRankProcessedDataCount.size() != templateRankSize_,
        HCCL_ERROR(
            "[InsTempAllGatherNHRNDAInter] invalid rank slice layout: rankSize[%u], sizes[%zu], displs[%zu], "
            "counts[%zu]",
            templateRankSize_, tempAlgParams.allRankSliceSize.size(), tempAlgParams.allRankDispls.size(),
            tempAlgParams.allRankProcessedDataCount.size()),
        HCCL_E_PARA);
    if (templateRankSize_ == 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        templateResource.threads.empty(), HCCL_ERROR("[InsTempAllGatherNHRNDAInter] no execution thread"), HCCL_E_PARA);
    // The preceding phase has already placed data at hcclBuffBaseOff + allRankDispls.
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    HcclResult ret = static_cast<HcclResult>(
        HcommThreadJoin(templateResource.threads[0], ExecTimeoutManager::Instance().GetExecTimeout()));
    if (ret == HCCL_SUCCESS) {
        ret = RunNHR(tempAlgParams, templateResource);
    }
    const HcclResult restoreRet = static_cast<HcclResult>(HcommBatchModeStart(param.algTag));
    return ret == HCCL_SUCCESS ? restoreRet : ret;
}

HcclResult InsTempAllGatherNHRNDAInter::GetStepInfo(u32 step, u32 stepNum, AicpuNHRStepInfo& stepInfo) const
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

HcclResult InsTempAllGatherNHRNDAInter::BuildSlices(
    const TemplateDataParams& tempAlgParams, const std::vector<u32>& sliceIdxs, void* buffer,
    std::vector<DataSlice>& slices) const
{
    for (u32 rank : sliceIdxs) {
        CHK_PRT_RET(
            rank >= templateRankSize_, HCCL_ERROR("[InsTempAllGatherNHRNDAInter] invalid slice rank[%u]", rank),
            HCCL_E_INTERNAL);
        const u64 size = tempAlgParams.allRankSliceSize[rank];
        if (size != 0) {
            const u64 offset = tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.allRankDispls[rank];
            slices.emplace_back(buffer, offset, size, tempAlgParams.allRankProcessedDataCount[rank]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempAllGatherNHRNDAInter::RunNHR(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource) const
{
    const u32 steps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < steps; ++step) {
        AicpuNHRStepInfo info{};
        CHK_RET(GetStepInfo(step, steps, info));
        const u32 toRank = subCommRanks_[0][info.toRank];
        const u32 fromRank = subCommRanks_[0][info.fromRank];
        const auto txIt = templateResource.channels.find(toRank);
        const auto rxIt = templateResource.channels.find(fromRank);
        CHK_PRT_RET(
            txIt == templateResource.channels.end() || rxIt == templateResource.channels.end() || txIt->second.empty()
                || rxIt->second.empty(),
            HCCL_ERROR("[InsTempAllGatherNHRNDAInter] missing channel for send[%u] recv[%u]", toRank, fromRank),
            HCCL_E_INTERNAL);
        const auto& txChannel = txIt->second[0];
        const auto& rxChannel = rxIt->second[0];
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;
        CHK_RET(BuildSlices(tempAlgParams, info.txSliceIdxs, tempAlgParams.buffInfo.hcclBuff.addr, txSrcSlices));
        CHK_RET(BuildSlices(tempAlgParams, info.txSliceIdxs, txChannel.remoteCclMem.addr, txDstSlices));
        CHK_PRT_RET(
            !txSrcSlices.empty()
                && (tempAlgParams.buffInfo.hcclBuff.addr == nullptr || txChannel.remoteCclMem.addr == nullptr),
            HCCL_ERROR("[InsTempAllGatherNHRNDAInter] null buffer for send[%u]", toRank), HCCL_E_PARA);
        CHK_RET(BuildSlices(tempAlgParams, info.rxSliceIdxs, rxChannel.remoteCclMem.addr, rxSrcSlices));
        CHK_RET(BuildSlices(tempAlgParams, info.rxSliceIdxs, tempAlgParams.buffInfo.hcclBuff.addr, rxDstSlices));
        TxRxChannels channels(txChannel, rxChannel);
        TxRxSlicesList slices({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        SendRecvInfo sendRecvInfo(channels, slices);
        // Preserve notification pairing for empty slices as well.
        CHK_RET(SendRecvWriteForNda(sendRecvInfo, templateResource.threads[0]));
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempAllGatherNHRNDAInter", InsTempAllGatherNHRNDAInter);
} // namespace ops_hccl
