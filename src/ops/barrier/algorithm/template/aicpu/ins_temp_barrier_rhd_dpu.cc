/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_barrier_rhd_dpu.h"
#include "barrier_rhd_utils.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "alg_v2_template_register.h"

namespace ops_hccl {
InsTempBarrierRhdDpu::InsTempBarrierRhdDpu(
    const OpParam& param, u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

HcclResult InsTempBarrierRhdDpu::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    return CalcBarrierRhdResources(comm, param, topoInfo, subCommRanks_, true, resourceRequest);
}

HcclResult InsTempBarrierRhdDpu::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& resource)
{
    std::vector<BarrierRhdStep> steps;
    CHK_RET(BuildBarrierRhdSteps(myRank_, subCommRanks_, steps));
    if (steps.empty()) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        resource.isDpuThreadEnabled,
        HCCL_ERROR("[BarrierRHD] HostDPU requires the shared-memory request path, rank[%u].", myRank_),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(
        resource.threads.empty(), HCCL_ERROR("[BarrierRHD] missing main thread for rank[%u].", myRank_),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        resource.npu2DpuShmemPtr == nullptr || resource.dpu2NpuShmemPtr == nullptr,
        HCCL_ERROR("[BarrierRHD] missing request/response resources for rank[%u].", myRank_), HCCL_E_INTERNAL);
    CHK_RET(CheckBarrierRhdChannels(steps, resource.channels, ENDPOINT_LOC_TYPE_HOST));
    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    HcclResult ret = static_cast<HcclResult>(HcommThreadSynchronize(resource.threads[0]));
    if (ret == HCCL_SUCCESS) {
        ret = SendPlanRequest(param, tempAlgParams, resource);
    }
    // 旧路径在即时模式下完成请求和响应，再恢复批量模式。
    const HcclResult restoreRet = static_cast<HcclResult>(HcommBatchModeStart(param.algTag));
    if (restoreRet != HCCL_SUCCESS) {
        HCCL_ERROR("[BarrierRHD] failed to restore batch mode, rank[%u], ret[%d].", myRank_, restoreRet);
    }
    return ret != HCCL_SUCCESS ? ret : restoreRet;
}

HcclResult InsTempBarrierRhdDpu::SendPlanRequest(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& resource)
{
    DPURunInfo runInfo;
    runInfo.execTimeout = param.opConfig.execTimeout;
    runInfo.templateName = "InsTempBarrierRhdDpu";
    runInfo.tempAlgParams = tempAlgParams;
    runInfo.channels = resource.channels;
    runInfo.myRank = myRank_;
    runInfo.subCommRanks = subCommRanks_;
    const auto data = runInfo.Serialize();
    u32 sendMsgId = 0;
    CHK_RET(static_cast<HcclResult>(HcommSendRequest(
        reinterpret_cast<uint64_t>(resource.npu2DpuShmemPtr), param.algTag, data.data(), data.size(), &sendMsgId)));
    HCCL_INFO(
        "[BarrierRHD][LegacyHostDPU] rank[%u] tag[%s] template[%s] sendMsgId[%u].", myRank_, param.algTag,
        runInfo.templateName.c_str(), sendMsgId);
    void* response = nullptr;
    u32 recvMsgId = 0;
    CHK_RET(static_cast<HcclResult>(
        HcommWaitResponse(reinterpret_cast<uint64_t>(resource.dpu2NpuShmemPtr), response, 0, &recvMsgId)));
    HCCL_INFO(
        "[BarrierRHD][LegacyHostDPU] rank[%u] tag[%s] template[%s] sendMsgId[%u] recvMsgId[%u].", myRank_, param.algTag,
        runInfo.templateName.c_str(), sendMsgId, recvMsgId);
    CHK_PRT_RET(
        recvMsgId != sendMsgId,
        HCCL_ERROR("[BarrierRHD] response msgId[%u] differs from request[%u].", recvMsgId, sendMsgId), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult InsTempBarrierRhdDpu::DPUKernelRun(
    const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels, u32 myRank,
    const std::vector<std::vector<u32>>& subCommRanks)
{
#ifndef AICPU_COMPILE
    (void)tempAlgParams;
    std::vector<BarrierRhdStep> steps;
    CHK_RET(BuildBarrierRhdSteps(myRank, subCommRanks, steps));
    CHK_RET(CheckBarrierRhdChannels(steps, channels, ENDPOINT_LOC_TYPE_HOST));
    const std::vector<DataSlice> emptySlices;
    for (const auto& step : steps) {
        const auto& channel = channels.at(step.peerRank)[0];
        TxRxChannels pair(channel, channel);
        TxRxSlicesList slices({emptySlices, emptySlices}, {emptySlices, emptySlices});
        SendRecvInfo info(pair, slices);
        if (step.phase == RhdPhase::MAIN_AG) {
            HCCL_DEBUG(
                "[BarrierRHD][HostDPU] rank[%u] phase[%u] step[%u] peer[%u] virtualPeer[%u] algRank[%u] "
                "userRank[%u] payloadBytes=0.",
                myRank, static_cast<u32>(step.phase), step.step, step.peerRank, step.virtualPeer, step.algPeer,
                step.peerRank);
        } else {
            HCCL_DEBUG(
                "[BarrierRHD][HostDPU] rank[%u] phase[%u] step[%u] peer[%u] payloadBytes=0.", myRank,
                static_cast<u32>(step.phase), step.step, step.peerRank);
        }
        CHK_RET(DpuSendRecvWrite(info));
    }
    return HCCL_SUCCESS;
#else
    return HCCL_E_NOT_SUPPORT;
#endif
}

REGISTER_TEMPLATE_V2("InsTempBarrierRhdDpu", InsTempBarrierRhdDpu);
} // namespace ops_hccl
