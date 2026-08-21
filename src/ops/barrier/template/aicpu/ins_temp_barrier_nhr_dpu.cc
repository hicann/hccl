/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_barrier_nhr_dpu.h"
#include "alg_data_trans_wrapper.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "channel.h"
#include "template_utils.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {
InsTempBarrierNHRDPU::InsTempBarrierNHRDPU(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{
}

HcclResult InsTempBarrierNHRDPU::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;

    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestNhrHighestHostRoce(comm, param, topoInfo, subCommRanks_, level1Channels));
    resourceRequest.channels.push_back(level1Channels);
    HCCL_INFO("[InsTempBarrierNHRDPU][CalcRes] level1Channels[%u].", level1Channels.size());
    return HCCL_SUCCESS;
}

u64 InsTempBarrierNHRDPU::CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType)
{
    (void)inBufferType;
    (void)outBufferType;
    return 0;
}

HcclResult InsTempBarrierNHRDPU::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
{
    (void)tempAlgParams;
    HCCL_INFO("[InsTempBarrierNHRDPU] Run Start, rank[%u] rankSize[%u]", myRank_, templateRankSize_);

    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    if (templateResource.threads.size() < 1) {
        HCCL_ERROR("[InsTempBarrierNHRDPU] Rank[%u], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }

    // 转换成 eager-mode，保障 AICPU 指令下发执行完成
    if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempBarrierNHRDPU] failed set eager mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }

    if (HcommThreadSynchronize(templateResource.threads[0]) != 0) {
        HCCL_ERROR("[InsTempBarrierNHRDPU] HcommThreadSynchronize failed");
        return HCCL_E_INTERNAL;
    }

    DPURunInfo dpuRunInfo;
    dpuRunInfo.templateName = "InsTempBarrierNHRDPU";
    dpuRunInfo.tempAlgParams = tempAlgParams;
    dpuRunInfo.channels = templateResource.channels;
    dpuRunInfo.myRank = myRank_;
    dpuRunInfo.subCommRanks = subCommRanks_;
    auto dpuRunInfoSeqData = dpuRunInfo.Serialize();

    u32 sendMsgId = 0;
    if (HcommSendRequest(reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr), param.algTag,
            static_cast<void *>(dpuRunInfoSeqData.data()), dpuRunInfoSeqData.size(), &sendMsgId)
        != 0) {
        HCCL_ERROR("[InsTempBarrierNHRDPU] HcommSendRequest failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempBarrierNHRDPU] HcommSendRequest sendMsgId[%u]", sendMsgId);

    void *recvData = nullptr;
    u32 recvMsgId = 0;
    if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) != 0) {
        HCCL_ERROR("[InsTempBarrierNHRDPU] HcommWaitResponse failed");
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempBarrierNHRDPU] HcommWaitResponse recvMsgId[%u]", recvMsgId);

    if (recvMsgId != sendMsgId) {
        HCCL_ERROR("[InsTempBarrierNHRDPU] recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
        return HCCL_E_INTERNAL;
    }

    if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
        HCCL_ERROR("[InsTempBarrierNHRDPU] failed set batch mode, tag is %s.", param.algTag);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempBarrierNHRDPU] Run End");
    return HCCL_SUCCESS;
}

HcclResult InsTempBarrierNHRDPU::DPUKernelRun(const TemplateDataParams &tempAlgParams,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>> &subCommRanks)
{
    (void)tempAlgParams;
    myRank_ = myRank;
    templateRankSize_ = subCommRanks[0].size();
    subCommRanks_ = subCommRanks;
    CHK_RET(RunNHRBarrier(channels));
    return HCCL_SUCCESS;
}

u32 InsTempBarrierNHRDPU::GetRankFromMap(const uint32_t rankIdx) const
{
    return subCommRanks_[0].at(rankIdx);
}

HcclResult InsTempBarrierNHRDPU::RunNHRBarrier(const std::map<u32, std::vector<ChannelInfo>> &channels) const
{
#ifndef AICPU_COMPILE
    if (templateRankSize_ <= 1) {
        return HCCL_SUCCESS;
    }
    const uint32_t nSteps = GetNHRStepNum(templateRankSize_);

    uint32_t rankIdx = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], rankIdx));

    for (uint32_t step = 0; step < nSteps; ++step) {
        // 计算 NHR step 通信对（与 AllGather NHR DPU 一致）
        uint32_t deltaRank = 1u << (nSteps - 1 - step);
        uint32_t recvFrom = (rankIdx + templateRankSize_ - deltaRank) % templateRankSize_;
        uint32_t sendTo = (rankIdx + deltaRank) % templateRankSize_;

        auto rxChannel = channels.at(GetRankFromMap(recvFrom));
        auto txChannel = channels.at(GetRankFromMap(sendTo));

        // Barrier 语义：传空 slice vector，DPU wrapper 只做前后同步对，不搬运数据
        std::vector<DataSlice> emptySlices;
        if (txChannel[0].remoteRank == rxChannel[0].remoteRank) {
            TxRxChannels sendRecvChannels(txChannel[0], rxChannel[0]);
            TxRxSlicesList sendRecvSlicesList({emptySlices, emptySlices}, {emptySlices, emptySlices});
            SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

            CHK_PRT_RET(SendRecvWrite(sendRecvInfo),
                HCCL_ERROR("[InsTempBarrierNHRDPU] SendRecvWrite failed (step=%u)", step), HcclResult::HCCL_E_INTERNAL);
        } else if (txChannel[0].remoteRank < rxChannel[0].remoteRank) {
            SlicesList sendSliceList(emptySlices, emptySlices);
            DataInfo sendInfo(txChannel[0], sendSliceList);
            CHK_PRT_RET(SendWrite(sendInfo), HCCL_ERROR("[InsTempBarrierNHRDPU] Send failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);

            SlicesList recvSliceList(emptySlices, emptySlices);
            DataInfo recvInfo(rxChannel[0], recvSliceList);
            CHK_PRT_RET(RecvWrite(recvInfo), HCCL_ERROR("[InsTempBarrierNHRDPU] Recv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        } else {
            SlicesList recvSliceList(emptySlices, emptySlices);
            DataInfo recvInfo(rxChannel[0], recvSliceList);
            CHK_PRT_RET(RecvWrite(recvInfo), HCCL_ERROR("[InsTempBarrierNHRDPU] Recv failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);

            SlicesList sendSliceList(emptySlices, emptySlices);
            DataInfo sendInfo(txChannel[0], sendSliceList);
            CHK_PRT_RET(SendWrite(sendInfo), HCCL_ERROR("[InsTempBarrierNHRDPU] Send failed (step=%u)", step),
                HcclResult::HCCL_E_INTERNAL);
        }
    }
#endif
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempBarrierNHRDPU", InsTempBarrierNHRDPU);
} // namespace ops_hccl
