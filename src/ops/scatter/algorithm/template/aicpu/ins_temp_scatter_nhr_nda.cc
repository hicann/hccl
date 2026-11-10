/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_nhr_nda.h"
#include "channel.h"
#include "alg_v2_template_register.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"

namespace ops_hccl {

InsTempScatterSequenceNHRNDA::InsTempScatterSequenceNHRNDA(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
    templateRankSize_ = subCommRanks.empty() ? 0 : subCommRanks[0].size();
}

HcclResult InsTempScatterSequenceNHRNDA::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_RET(GetRes(resourceRequest));
    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestNhrHighestHostRoce(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterSequenceNHRNDA::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempScatterSequenceNHRNDA::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult InsTempScatterSequenceNHRNDA::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty(), HCCL_ERROR("[ScatterSequenceNDA] empty rank group"),
        HCCL_E_PARA);
    CHK_PRT_RET(
        tempAlgParams.allRankSliceSize.size() != templateRankSize_
            || tempAlgParams.allRankDispls.size() != templateRankSize_
            || tempAlgParams.allRankProcessedDataCount.size() != templateRankSize_,
        HCCL_ERROR("[ScatterSequenceNDA] invalid rank slice layout"), HCCL_E_PARA);
    CHK_PRT_RET(templateResource.threads.empty(), HCCL_ERROR("[ScatterSequenceNDA] no execution thread"), HCCL_E_PARA);
    if (templateRankSize_ > 1) {
        // Eager launch keeps each RoCE notify wait executing promptly.
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        HcclResult ret = RunNHR(tempAlgParams, templateResource.channels, templateResource.threads[0]);
        // Restore batch mode on both the success and failure paths.
        const HcclResult restoreRet = static_cast<HcclResult>(HcommBatchModeStart(param.algTag));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS || restoreRet != HCCL_SUCCESS,
            HCCL_ERROR("[ScatterSequenceNDA] NHR transfer failed, runRet[%d], restoreRet[%d]", ret, restoreRet),
            ret != HCCL_SUCCESS ? ret : restoreRet);
    }

    // Broadcast's Scatter phase must leave each slice at its original CCL displacement.
    if (tempAlgParams.buffInfo.outBuffType == BufferType::OUTPUT && tempAlgParams.buffInfo.outputPtr != nullptr) {
        u32 myAlgRank = 0;
        CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
        CHK_PRT_RET(
            myAlgRank >= tempAlgParams.allRankSliceSize.size(), HCCL_ERROR("[ScatterSequenceNDA] missing own slice"),
            HCCL_E_PARA);
        const u64 size = tempAlgParams.allRankSliceSize[myAlgRank];
        const u64 count = tempAlgParams.allRankProcessedDataCount[myAlgRank];
        if (size != 0) {
            DataSlice src(
                tempAlgParams.buffInfo.hcclBuff.addr,
                tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.allRankDispls[myAlgRank], size, count);
            DataSlice dst(tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, size, count);
            CHK_RET(LocalCopy(templateResource.threads[0], src, dst));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
InsTempScatterSequenceNHRNDA::GetStepInfo(u32 step, u32 nSteps, u32 rootAlgRank, AicpuNHRStepInfo& stepInfo) const
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 deltaRoot = (rootAlgRank + templateRankSize_ - myAlgRank) % templateRankSize_;
    const u32 deltaRank = 1U << step;
    const u32 deltaSlice = deltaRank * 2;
    const u32 nSlices = (templateRankSize_ - 1 + deltaRank) / deltaSlice;
    const bool powerOfTwo = (templateRankSize_ & (templateRankSize_ - 1)) == 0;
    const u32 nRanks = !powerOfTwo && step == nSteps - 1 ? templateRankSize_ - deltaRank : deltaRank;
    stepInfo.txSliceIdxs.clear();
    stepInfo.rxSliceIdxs.clear();
    // Stop at the root boundary: non-power-of-two groups must not forward slices they never received.
    if (deltaRoot < nRanks) {
        u32 slice = (myAlgRank + templateRankSize_ - deltaRank) % templateRankSize_;
        stepInfo.toRank = slice;
        for (u32 i = 0; i < nSlices && deltaRoot + deltaRank + i * deltaSlice < templateRankSize_; ++i) {
            stepInfo.txSliceIdxs.push_back(slice);
            slice = (slice + templateRankSize_ - deltaSlice) % templateRankSize_;
        }
    } else if (deltaRoot >= deltaRank && deltaRoot < nRanks + deltaRank) {
        stepInfo.fromRank = (myAlgRank + deltaRank) % templateRankSize_;
        u32 slice = myAlgRank;
        for (u32 i = 0; i < nSlices && deltaRoot + i * deltaSlice < templateRankSize_; ++i) {
            stepInfo.rxSliceIdxs.push_back(slice);
            slice = (slice + templateRankSize_ - deltaSlice) % templateRankSize_;
        }
    }
    return HCCL_SUCCESS;
}

u32 InsTempScatterSequenceNHRNDA::GetRankFromMap(u32 rankIdx) const { return subCommRanks_[0].at(rankIdx); }

HcclResult InsTempScatterSequenceNHRNDA::RunNHR(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const ThreadHandle& thread) const
{
    CHK_PRT_RET(
        tempAlgParams.allRankSliceSize.size() != templateRankSize_
            || tempAlgParams.allRankDispls.size() != templateRankSize_
            || tempAlgParams.allRankProcessedDataCount.size() != templateRankSize_,
        HCCL_ERROR("[ScatterSequenceNDA] invalid rank slice layout"), HCCL_E_PARA);
    CHK_PRT_RET(
        tempAlgParams.repeatNum != 1, HCCL_ERROR("[ScatterSequenceNDA] expected one rank-indexed group"), HCCL_E_PARA);
    u32 myAlgRank = 0;
    u32 rootAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    CHK_RET(GetAlgRank(tempAlgParams.root, subCommRanks_[0], rootAlgRank));
    const u32 steps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < steps; ++step) {
        AicpuNHRStepInfo info{};
        CHK_RET(GetStepInfo(step, steps, rootAlgRank, info));
        if (info.txSliceIdxs.empty() && info.rxSliceIdxs.empty()) {
            continue;
        }
        // Scatter has one direction per rank at each NHR step.
        const bool send = !info.txSliceIdxs.empty();
        const u32 peerAlgIdx = send ? info.toRank : info.fromRank;
        CHK_PRT_RET(
            peerAlgIdx >= subCommRanks_[0].size(),
            HCCL_ERROR("[ScatterSequenceNDA] invalid peer index[%u]", peerAlgIdx), HCCL_E_PARA);
        const u32 peer = GetRankFromMap(peerAlgIdx);
        const auto channelIt = channels.find(peer);
        CHK_PRT_RET(
            channelIt == channels.end() || channelIt->second.empty(),
            HCCL_ERROR("[ScatterSequenceNDA] missing channel for peer[%u]", peer), HCCL_E_INTERNAL);
        const auto& channel = channelIt->second[0];
        if (!send) {
            CHK_RET(RecvNdaSlices(channel, thread));
            continue;
        }
        std::vector<DataSlice> localSlices;
        for (u32 rank : info.txSliceIdxs) {
            if (rank == rootAlgRank) {
                continue; // The root slice stays local and has no transfer slot.
            }
            CHK_PRT_RET(
                rank >= templateRankSize_,
                HCCL_ERROR("[ScatterSequenceNDA] invalid peer slice[%u], root[%u]", rank, rootAlgRank),
                HCCL_E_INTERNAL);
            const u64 size = tempAlgParams.allRankSliceSize[rank];
            if (size != 0) {
                localSlices.emplace_back(
                    tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.allRankDispls[rank], size,
                    tempAlgParams.allRankProcessedDataCount[rank]);
            }
        }
        // Keep ACK/DATA/FIN_ACK paired even when this step contains only zero-sized pieces.
        std::vector<DataSlice> remoteSlices;
        for (const auto& slice : localSlices) {
            remoteSlices.emplace_back(channel.remoteCclMem.addr, slice.offset_, slice.size_, slice.count_);
        }
        CHK_RET(SendNdaSlices(channel, localSlices, remoteSlices, thread));
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempScatterSequenceNHRNDA", InsTempScatterSequenceNHRNDA);
} // namespace ops_hccl
