/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_barrier_nhr_aicpu.h"
#include "alg_data_trans_wrapper.h"
#include "channel.h"
#include "template_utils.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {

InsTempBarrierNhrAicpu::InsTempBarrierNhrAicpu(const OpParam &param, const u32 rankId,
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks) {}

HcclResult InsTempBarrierNhrAicpu::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhr(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO("[InsTempBarrierNhrAicpu][CalcRes] level1Channels[%zu].", level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempBarrierNhrAicpu::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    return 0;
}

HcclResult InsTempBarrierNhrAicpu::KernelRun(const OpParam &param,
    const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    (void)param;
    (void)tempAlgParams;
    HCCL_INFO("[InsTempBarrierNhrAicpu] Run Start, rank[%u] rankSize[%u]", myRank_, templateRankSize_);

    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempBarrierNhrAicpu] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(RunNHRBarrier(templateResource.channels, templateResource.threads[0]));

    HCCL_INFO("[InsTempBarrierNhrAicpu] Run End");
    return HCCL_SUCCESS;
}

u32 InsTempBarrierNhrAicpu::GetRankFromMap(const uint32_t rankIdx) const
{
    return subCommRanks_[0].at(rankIdx);
}

HcclResult InsTempBarrierNhrAicpu::RunNHRBarrier(
    const std::map<u32, std::vector<ChannelInfo>> &channels, const ThreadHandle &thread)
{
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    uint32_t rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));

    for (uint32_t step = 0; step < nSteps; ++step) {
        uint32_t deltaRank = 1u << (nSteps - 1 - step);
        uint32_t recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
        uint32_t sendTo = (rankIdx + deltaRank) % templateRankSize_;

        auto rxIter = channels.find(GetRankFromMap(recvFrom));
        auto txIter = channels.find(GetRankFromMap(sendTo));
        CHK_PRT_RET(rxIter == channels.end() || txIter == channels.end(),
            HCCL_ERROR("[InsTempBarrierNhrAicpu] myRank[%u] channel not found (step=%u), recvFrom[%u] sendTo[%u]",
                myRank_, step, GetRankFromMap(recvFrom), GetRankFromMap(sendTo)),
            HcclResult::HCCL_E_INTERNAL);
        CHK_PRT_RET(rxIter->second.empty() || txIter->second.empty(),
            HCCL_ERROR("[InsTempBarrierNhrAicpu] myRank[%u] channel empty (step=%u)", myRank_, step),
            HcclResult::HCCL_E_INTERNAL);
        const auto &rxChannel = rxIter->second;
        const auto &txChannel = txIter->second;

        std::vector<DataSlice> emptySlices;
        if (txChannel[0].remoteRank == rxChannel[0].remoteRank) {
            TxRxChannels sendRecvChannels(txChannel[0], rxChannel[0]);
            TxRxSlicesList sendRecvSlicesList({emptySlices, emptySlices}, {emptySlices, emptySlices});
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] myRank[%u] SendRecvWrite failed (step=%u)", myRank_, step),
                HcclResult::HCCL_E_INTERNAL);
        } else if (txChannel[0].remoteRank < rxChannel[0].remoteRank) {
            SlicesList sendSliceList(emptySlices, emptySlices);
            DataInfo sendInfo(txChannel[0], sendSliceList);
            CHK_PRT_RET(SendWrite(sendInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] myRank[%u] Send failed (step=%u)", myRank_, step),
                HcclResult::HCCL_E_INTERNAL);

            SlicesList recvSliceList(emptySlices, emptySlices);
            DataInfo recvInfo(rxChannel[0], recvSliceList);
            CHK_PRT_RET(RecvWrite(recvInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] myRank[%u] Recv failed (step=%u)", myRank_, step),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            SlicesList recvSliceList(emptySlices, emptySlices);
            DataInfo recvInfo(rxChannel[0], recvSliceList);
            CHK_PRT_RET(RecvWrite(recvInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] myRank[%u] Recv failed (step=%u)", myRank_, step),
                HcclResult::HCCL_E_INTERNAL);

            SlicesList sendSliceList(emptySlices, emptySlices);
            DataInfo sendInfo(txChannel[0], sendSliceList);
            CHK_PRT_RET(SendWrite(sendInfo, thread),
                HCCL_ERROR("[InsTempBarrierNhrAicpu] myRank[%u] Send failed (step=%u)", myRank_, step),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempBarrierNhrAicpu", InsTempBarrierNhrAicpu);

}  // namespace ops_hccl
