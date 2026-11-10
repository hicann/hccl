/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_scatter_omnipipe_nhr_nda.h"
#include "channel.h"
#include "alg_v2_template_register.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"

namespace ops_hccl {

InsTempScatterOmniPipeNHRNDA::InsTempScatterOmniPipeNHRNDA(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempScatterOmniPipeNHRNDA::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_RET(GetRes(resourceRequest));
    std::vector<HcclChannelDesc> channels;
    CHK_RET(CalcChannelRequestNhrHighestHostRoce(comm, param, topoInfo, subCommRanks_, channels));
    resourceRequest.channels.push_back(channels);
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHRNDA::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumOnMainThread = 0;
    return HCCL_SUCCESS;
}

u64 InsTempScatterOmniPipeNHRNDA::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return 1;
}

HcclResult
InsTempScatterOmniPipeNHRNDA::PreLocalCopy(const TemplateDataParams& tempAlgParams, const ThreadHandle& thread) const
{
    // Only the source rank stages user input. Forwarding ranks already hold the data in CCL memory.
    if (myRank_ != root_ || tempAlgParams.buffInfo.inBuffType != BufferType::INPUT) {
        return HCCL_SUCCESS;
    }
    const auto& step = tempAlgParams.stepSliceInfo;
    CHK_PRT_RET(step.stepSliceSize.empty(), HCCL_ERROR("[ScatterNDA] empty slice rows"), HCCL_E_PARA);
    const u32 row = myRank_ % step.stepSliceSize.size();
    const size_t pieces = step.stepSliceSize[row].size();
    CHK_PRT_RET(
        row >= step.inputOmniPipeSliceStride.size() || row >= step.outputOmniPipeSliceStride.size()
            || step.inputOmniPipeSliceStride[row].size() != pieces
            || step.outputOmniPipeSliceStride[row].size() != pieces,
        HCCL_ERROR("[ScatterNDA] invalid input slice layout"), HCCL_E_PARA);
    const u32 typeSize = HCCL_SIZE_TABLE[dataType_];
    for (size_t i = 0; i < pieces; ++i) {
        const u64 size = step.stepSliceSize[row][i];
        if (size == 0) {
            continue;
        }
        CHK_PTR_NULL(tempAlgParams.buffInfo.inputPtr);
        CHK_PTR_NULL(tempAlgParams.buffInfo.outputPtr);
        DataSlice src(
            tempAlgParams.buffInfo.inputPtr, step.buffInfo.inBuffBaseOff + step.inputOmniPipeSliceStride[row][i], size,
            size / typeSize);
        DataSlice dst(
            tempAlgParams.buffInfo.outputPtr, step.buffInfo.outBuffBaseOff + step.outputOmniPipeSliceStride[row][i],
            size, size / typeSize);
        CHK_RET(LocalCopy(thread, src, dst));
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHRNDA::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    if (templateRankSize_ == 1 || !doTask_.load(std::memory_order_relaxed)) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        subCommRanks_.empty() || subCommRanks_[0].empty() || templateRankSize_ != subCommRanks_[0].size(),
        HCCL_ERROR("[ScatterNDA] invalid sub-communicator"), HCCL_E_PARA);
    CHK_PRT_RET(templateResource.threads.empty(), HCCL_ERROR("[ScatterNDA] no execution thread"), HCCL_E_PARA);
    dataType_ = param.DataDes.dataType;
    CHK_RET(PreLocalCopy(tempAlgParams, templateResource.threads[0]));
    // Level-2 forwards slices from earlier outer steps, whose completion is ordered
    // by the executor's PostSyncInterThreads -> PreSyncInterThreads chain.
    // Root staging, notify waits and RoCE doorbells use this same thread; hcomm must
    // also order soft-PI updates on its RTSQ. Flush for eager execution without
    // a CPU-side join (unlike the separate DPU request path).
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    HcclResult ret = RunNHR(tempAlgParams, templateResource);
    // Restore batch mode on both the success and failure paths.
    const HcclResult restoreRet = static_cast<HcclResult>(HcommBatchModeStart(param.algTag));
    return ret == HCCL_SUCCESS ? restoreRet : ret;
}

HcclResult InsTempScatterOmniPipeNHRNDA::GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo& stepInfo) const
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

HcclResult InsTempScatterOmniPipeNHRNDA::BuildLocalSlices(
    const TemplateDataParams& tempAlgParams, const std::vector<u32>& sliceIdxs,
    std::vector<DataSlice>& localSlices) const
{
    const auto& step = tempAlgParams.stepSliceInfo;
    CHK_PRT_RET(step.stepSliceSize.empty(), HCCL_ERROR("[ScatterNDA] empty slice rows"), HCCL_E_PARA);
    const u32 row = myRank_ % step.stepSliceSize.size();
    const size_t pieces = step.stepSliceSize[row].size();
    const u32 peers = templateRankSize_ - 1;
    CHK_PRT_RET(
        peers == 0 || pieces % peers != 0 || row >= step.outputOmniPipeSliceStride.size()
            || step.outputOmniPipeSliceStride[row].size() != pieces,
        HCCL_ERROR("[ScatterNDA] invalid output slice layout"), HCCL_E_PARA);
    const size_t repeatNum = pieces / peers;
    u32 rootAlgRank = 0;
    CHK_RET(GetAlgRank(root_, subCommRanks_[0], rootAlgRank));
    const u32 typeSize = HCCL_SIZE_TABLE[dataType_];
    for (u32 rank : sliceIdxs) {
        if (rank == rootAlgRank) {
            continue; // The root slice stays local and has no slot in the transfer layout.
        }
        CHK_PRT_RET(
            rank >= templateRankSize_, HCCL_ERROR("[ScatterNDA] invalid peer slice[%u], root[%u]", rank, rootAlgRank),
            HCCL_E_INTERNAL);
        // The OmniPipe rows contain one group per remote rank, with the root group omitted.
        const u32 group = rank < rootAlgRank ? rank : rank - 1;
        for (size_t repeat = 0; repeat < repeatNum; ++repeat) {
            const size_t index = group * repeatNum + repeat;
            const u64 size = step.stepSliceSize[row][index];
            if (size != 0) {
                localSlices.emplace_back(
                    tempAlgParams.buffInfo.outputPtr,
                    step.buffInfo.outBuffBaseOff + step.outputOmniPipeSliceStride[row][index], size, size / typeSize);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult InsTempScatterOmniPipeNHRNDA::RunNHR(
    const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource) const
{
    const u32 steps = GetNHRStepNum(templateRankSize_);
    AicpuNHRStepInfo info{};
    std::vector<DataSlice> localSlices;
    std::vector<DataSlice> remoteSlices;
    for (u32 step = 0; step < steps; ++step) {
        CHK_RET(GetStepInfo(step, steps, info));
        if (info.txSliceIdxs.empty() && info.rxSliceIdxs.empty()) {
            continue;
        }
        // Scatter has one direction per rank at each NHR step.
        const bool send = !info.txSliceIdxs.empty();
        const u32 peer = send ? info.toRank : info.fromRank;
        const auto channelIt = templateResource.channels.find(peer);
        CHK_PRT_RET(
            channelIt == templateResource.channels.end() || channelIt->second.empty(),
            HCCL_ERROR("[ScatterNDA] missing channel for peer[%u]", peer), HCCL_E_INTERNAL);
        const auto& channel = channelIt->second[0];
        localSlices.clear();
        // Keep ACK/DATA/FIN_ACK paired even when this step contains only zero-sized pieces.
        HcclResult transferRet = HCCL_SUCCESS;
        if (send) {
            CHK_RET(BuildLocalSlices(tempAlgParams, info.txSliceIdxs, localSlices));
            remoteSlices.clear();
            remoteSlices.reserve(localSlices.size());
            for (const auto& slice : localSlices) {
                remoteSlices.emplace_back(channel.remoteCclMem.addr, slice.offset_, slice.size_, slice.count_);
            }
            transferRet = SendNdaSlices(channel, localSlices, remoteSlices, templateResource.threads[0]);
        } else {
            transferRet = RecvNdaSlices(channel, templateResource.threads[0]);
        }
        CHK_PRT_RET(
            transferRet != HCCL_SUCCESS,
            HCCL_ERROR(
                "[ScatterNDA] transfer failed rank[%u] root[%u] peer[%u] step[%u/%u] send[%d] "
                "thread[%llu] txSlices[%zu] ret[%d]",
                myRank_, root_, peer, step, steps, static_cast<int>(send),
                static_cast<unsigned long long>(templateResource.threads[0]), localSlices.size(), transferRet),
            transferRet);
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempScatterOmniPipeNHRNDA", InsTempScatterOmniPipeNHRNDA);
} // namespace ops_hccl
