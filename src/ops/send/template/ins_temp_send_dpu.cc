/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_send_dpu.h"

namespace ops_hccl {
InsTempSendDpu::InsTempSendDpu()
{}
// ! 已编码完成
InsTempSendDpu::InsTempSendDpu(const OpParam &param,
    const u32 rankId,  // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

// ! 已编码完成
InsTempSendDpu::~InsTempSendDpu()
{}

// ! 已编码完成
HcclResult InsTempSendDpu::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumPerThread = {};
    resourceRequest.notifyNumOnMainThread = 0;
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    resourceRequest.channels.push_back(level0Channels);
    return HCCL_SUCCESS;
}

// ! 基本编码完成，剩余数据序列化
HcclResult InsTempSendDpu::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, const TemplateResource &templateResource)
{
    // 得到thread和channel
    recvRank_ = param.sendRecvRemoteRank;
    thread_ = templateResource.threads[0];
    auto channelIter = templateResource.channels.find(recvRank_);
    if (channelIter == templateResource.channels.end() || channelIter->second.empty()) {
        HCCL_ERROR(
            "[InsTempSendDpu][KernelRun]my rank is [%d], receive rank [%u] channel not found!", myRank_, recvRank_);
        return HCCL_E_INTERNAL;
    }
    sendChannel_ = channelIter->second[0];
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    // 跨框流程要走dpu
    if (sendChannel_.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
        // aicpu先把inputbuffer的内容localcopy给cclbuffer
        DataSlice inputBuffer(
            tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_);
        DataSlice localCclBuffer(
            tempAlgParams.buffInfo.hcclBuff.addr, 0, processSize_, count_);  // cclbuffer不需要offset
        CHK_RET(LocalCopy(thread_, inputBuffer, localCclBuffer));            // 本端inputbuffer->本端ccl
        // 转换成eager-mode，保障AICPU指令下发执行完成
        if (HcommBatchModeEnd(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }
        if (HcommThreadSynchronize(thread_) != 0) {
            HCCL_ERROR("HcommThreadSynchronize failed");
            return HCCL_E_INTERNAL;
        }
        DPURunInfo dpuRunInfo;
        dpuRunInfo.templateName = "InsTempSendDpu";
        dpuRunInfo.tempAlgParams = tempAlgParams;
        dpuRunInfo.channels = templateResource.channels;
        dpuRunInfo.myRank = myRank_;
        dpuRunInfo.subCommRanks = subCommRanks_;
        u32 sendMsgId = 0;
        auto dpuRunInfoSeqData = dpuRunInfo.Serialize();

        if (HcommSendRequest(reinterpret_cast<uint64_t>(templateResource.npu2DpuShmemPtr),
                param.algTag,
                static_cast<void *>(dpuRunInfoSeqData.data()),
                dpuRunInfoSeqData.size(),
                &sendMsgId) != 0) {
            HCCL_ERROR("HcommSendRequest failed");
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("HcommSendRequest run over, sendMsgId[%u]", sendMsgId);
        // 等待DPU数据传输，然后回写结果回来
        void *recvData = nullptr;
        u32 recvMsgId = 0;
        if (HcommWaitResponse(reinterpret_cast<uint64_t>(templateResource.dpu2NpuShmemPtr), recvData, 0, &recvMsgId) !=
            0) {
            HCCL_ERROR("HcommWaitResponse failed");
            return HCCL_E_INTERNAL;
        }

        // 将执行模式转换回到batch
        if (HcommBatchModeStart(param.algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param.algTag);
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("HcommWaitResponse run over, recvMsgId[%u]", recvMsgId);

        if (recvMsgId != sendMsgId) {
            HCCL_ERROR("recvMsgId[%u] not equal to sendMsgId[%u]", recvMsgId, sendMsgId);
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("[InsTempSendDpu] Run End");
    } else if (sendChannel_.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
        // 直接发送到对端的cclbuffer上
        DataSlice inputBuffer(
            tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_);
        DataSlice remoteCclBuffer(tempAlgParams.buffInfo.outputPtr, 0, processSize_, count_);  // ccl buffer不需要offset
        // 发送
        SlicesList sendSlicesList({inputBuffer}, {remoteCclBuffer});
        DataInfo sendInfo(sendChannel_, sendSlicesList);
        CHK_PRT_RET(SendWrite(sendInfo, thread_),
            HCCL_ERROR("[InsTempSendDpu][KernelRun]Aicpu Run Send failed"),
            HcclResult::HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("[InsTempSendDpu][Kernel Run] location [%d] is not supported!", sendChannel_.locationType);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempSendDpu] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempSendDpu] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempSendDpu::DPUKernelRun(const TemplateDataParams &tempAlgParam,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>> &subCommRanks)
{
#ifndef AICPU_COMPILE
    if (subCommRanks.empty() || subCommRanks[0].size() < 2) {
        HCCL_ERROR("[InsTempSendDpu] [DPUKernelRun] no rank at all!");
    }
    u32 recvRank{0};
    for (auto rankId : subCommRanks[0]) {
        if (rankId != myRank) {
            recvRank = rankId;
            HCCL_INFO("[InsTempSendDpu] [DPUKernelRun] my rank is [%d],  receive rank is [%u].", myRank, recvRank);
        }
    }
    auto channelIter = channels.find(recvRank);
    if (channelIter == channels.end() || channelIter->second.empty()) {
        HCCL_ERROR(
            "[InsTempSendDpu] [DPUKernelRun] my rank is [%d],  receive rank [%u] channel not found!", myRank, recvRank);
        return HCCL_E_INTERNAL;
    }
    ChannelInfo linkSend = channelIter->second[0];
    DataSlice localCclBuffer(tempAlgParam.buffInfo.hcclBuff.addr, 0, tempAlgParam.sliceSize);
    DataSlice remoteCclBuffer(tempAlgParam.buffInfo.outputPtr, 0, tempAlgParam.sliceSize);  // ccl
    // 发送
    SlicesList sendSlicesList({localCclBuffer}, {remoteCclBuffer});
    DataInfo sendInfo(linkSend, sendSlicesList);
    CHK_PRT_RET(
        SendWrite(sendInfo), HCCL_ERROR("[InsTempSendDpu][DPUKernelRun] Run Send failed"), HcclResult::HCCL_E_INTERNAL);
    HCCL_INFO("[InsTempSendDpu][DPUKernelRun] Run Send sucess!");
#endif
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempSendDpu", InsTempSendDpu);
}  // namespace ops_hccl