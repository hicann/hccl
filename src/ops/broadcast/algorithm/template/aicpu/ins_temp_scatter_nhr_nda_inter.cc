/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_nhr_nda_inter.h"
#include "alg_v2_template_register.h"
#include "channel.h"
#include "dpu_alg_data_trans_wrapper.h"

namespace ops_hccl {
InsTempScatterNHRNDAInter::InsTempScatterNHRNDAInter(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempScatterNHRNDAInter::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumOnMainThread = 0;
    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestNhrHighestHostRoce(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);
    return HCCL_SUCCESS;
}

u64 InsTempScatterNHRNDAInter::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

HcclResult InsTempScatterNHRNDAInter::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty() || templateRankSize_ != subCommRanks_[0].size(),
        HCCL_ERROR("[InsTempScatterNHRNDAInter] invalid sub-communicator"), HCCL_E_PARA);
    CHK_PRT_RET(
        tempAlgParams.repeatNum != 1,
        HCCL_ERROR(
            "[InsTempScatterNHRNDAInter] expect repeatNum[1], got[%llu]",
            static_cast<unsigned long long>(tempAlgParams.repeatNum)),
        HCCL_E_PARA);
    CHK_PRT_RET(
        tempAlgParams.allRankSliceSize.size() != templateRankSize_
            || tempAlgParams.allRankDispls.size() != templateRankSize_
            || tempAlgParams.allRankProcessedDataCount.size() != templateRankSize_,
        HCCL_ERROR(
            "[InsTempScatterNHRNDAInter] invalid rank slice layout: rankSize[%u], sizes[%zu], displs[%zu], counts[%zu]",
            templateRankSize_, tempAlgParams.allRankSliceSize.size(), tempAlgParams.allRankDispls.size(),
            tempAlgParams.allRankProcessedDataCount.size()),
        HCCL_E_PARA);
    if (templateRankSize_ == 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        templateResource.threads.empty(), HCCL_ERROR("[InsTempScatterNHRNDAInter] no execution thread"), HCCL_E_PARA);
    root_ = tempAlgParams.root;
    // The preceding phase has already placed data at hcclBuffBaseOff + allRankDispls.
    // Intra-phase notifications and NDA transfers are ordered on the same main thread.
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    const HcclResult ret = RunNHR(tempAlgParams, templateResource);
    const HcclResult restoreRet = static_cast<HcclResult>(HcommBatchModeStart(param.algTag));
    return ret == HCCL_SUCCESS ? restoreRet : ret;
}

HcclResult InsTempScatterNHRNDAInter::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const
{
    u32 myAlgRank = 0;
    u32 rootAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    CHK_RET(GetAlgRank(root_, subCommRanks_[0], rootAlgRank));
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
        stepInfo.toRank = subCommRanks_[0][slice];
        for (u32 i = 0; i < nSlices && deltaRoot + deltaRank + i * deltaSlice < templateRankSize_; ++i) {
            stepInfo.txSliceIdxs.push_back(slice);
            slice = (slice + templateRankSize_ - deltaSlice) % templateRankSize_;
        }
    } else if (deltaRoot >= deltaRank && deltaRoot < nRanks + deltaRank) {
        stepInfo.fromRank = subCommRanks_[0][(myAlgRank + deltaRank) % templateRankSize_];
        u32 slice = myAlgRank;
        for (u32 i = 0; i < nSlices && deltaRoot + i * deltaSlice < templateRankSize_; ++i) {
            stepInfo.rxSliceIdxs.push_back(slice);
            slice = (slice + templateRankSize_ - deltaSlice) % templateRankSize_;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterNHRNDAInter::BuildSlices(
    const TemplateDataParams& tempAlgParams, const std::vector<u32>& sliceIdxs, void* buffer,
    std::vector<DataSlice>& slices) const
{
    for (u32 rank : sliceIdxs) {
        CHK_PRT_RET(
            rank >= templateRankSize_, HCCL_ERROR("[InsTempScatterNHRNDAInter] invalid slice rank[%u]", rank),
            HCCL_E_INTERNAL);
        const u64 size = tempAlgParams.allRankSliceSize[rank];
        if (size != 0) {
            const u64 offset = tempAlgParams.buffInfo.hcclBuffBaseOff + tempAlgParams.allRankDispls[rank];
            slices.emplace_back(buffer, offset, size, tempAlgParams.allRankProcessedDataCount[rank]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterNHRNDAInter::RunNHR(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource) const
{
    const u32 steps = GetNHRStepNum(templateRankSize_);
    for (u32 step = 0; step < steps; ++step) {
        AicpuNHRStepInfo info{};
        CHK_RET(GetStepInfo(step, steps, info));
        if (info.txSliceIdxs.empty() && info.rxSliceIdxs.empty()) {
            continue;
        }
        const bool send = !info.txSliceIdxs.empty();
        const u32 peer = send ? info.toRank : info.fromRank;
        const auto it = templateResource.channels.find(peer);
        CHK_PRT_RET(
            it == templateResource.channels.end() || it->second.empty(),
            HCCL_ERROR("[InsTempScatterNHRNDAInter] missing channel for peer[%u]", peer), HCCL_E_INTERNAL);
        const auto& channel = it->second[0];
        // Even an all-zero step must complete the paired ACK/DATA/FIN_ACK handshake.
        if (send) {
            std::vector<DataSlice> localSlices;
            std::vector<DataSlice> remoteSlices;
            CHK_RET(BuildSlices(tempAlgParams, info.txSliceIdxs, tempAlgParams.buffInfo.hcclBuff.addr, localSlices));
            CHK_RET(BuildSlices(tempAlgParams, info.txSliceIdxs, channel.remoteCclMem.addr, remoteSlices));
            CHK_RET(SendNdaSlices(channel, localSlices, remoteSlices, templateResource.threads[0]));
        } else {
            CHK_RET(RecvNdaSlices(channel, templateResource.threads[0]));
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempScatterNHRNDAInter", InsTempScatterNHRNDAInter);
} // namespace ops_hccl
