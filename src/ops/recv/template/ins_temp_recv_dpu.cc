/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_recv_dpu.h"

namespace ops_hccl {
InsTempRecvDpu::InsTempRecvDpu()
{}
// ! 已编码完成
InsTempRecvDpu::InsTempRecvDpu(const OpParam &param,
    const u32 rankId,  // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)
{}

InsTempRecvDpu::~InsTempRecvDpu()
{}

HcclResult InsTempRecvDpu::CalcRes(
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
HcclResult InsTempRecvDpu::KernelRun(
    const OpParam &param, const TemplateDataParams &tempAlgParams, const TemplateResource &templateResource)
{
    // 得到thread和channel
    sendRank_ = param.sendRecvRemoteRank;
    thread_ = templateResource.threads[0];
    auto channelIter = templateResource.channels.find(sendRank_);
    if (channelIter == templateResource.channels.end() || channelIter->second.empty()) {
        HCCL_ERROR(
            "[InsTempRecvDpu][KernelRun] my rank is [%d], send rank [%u] channel not found!", myRank_, sendRank_);
        return HCCL_E_INTERNAL;
    }
    recvChannel_ = channelIter->second[0];
    threadNum_ = templateResource.threads.size();
    processSize_ = tempAlgParams.sliceSize;
    count_ = tempAlgParams.count;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    // 跨框流程要走dpu
    if (recvChannel_.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
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
        dpuRunInfo.templateName = "InsTempRecvDpu";
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
            HCCL_ERROR("HcommRecvRequest failed");
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO("HcommRecvRequest run over, sendMsgId[%u]", sendMsgId);
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
        // aicpu把cclbuffer的内容localcopy给outputbuffer
        DataSlice localCclBuffer(
            tempAlgParams.buffInfo.hcclBuff.addr, 0, processSize_, count_);  // cclbuffer不需要offset
        DataSlice outputBuffer(
            tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);

        CHK_RET(LocalCopy(thread_, localCclBuffer, outputBuffer));  // 本端ccl->本端output
        HCCL_INFO("[InsTempRecvDpu] Run End");
    } else if (recvChannel_.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
        // 接收对方inputbuffer的数据到本端cclbuffer
        DataSlice remoteInputBuffer(
            tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_);
        DataSlice localCclBuffer(
            tempAlgParams.buffInfo.hcclBuff.addr, 0, processSize_, count_);  // cclbuffer不需要offset
        // 发送
        SlicesList recvSlicesList({remoteInputBuffer}, {localCclBuffer});
        DataInfo recvInfo(recvChannel_, recvSlicesList);
        CHK_PRT_RET(RecvWrite(recvInfo, thread_),
            HCCL_ERROR("[InsTempRecvDpu][KernelRun]Aicpu Run Recv failed"),
            HcclResult::HCCL_E_INTERNAL);
        // 把cclbuffer的内容localcopy给outputbuffer
        DataSlice outputBuffer(
            tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);
        CHK_PRT_RET(LocalCopy(thread_, localCclBuffer, outputBuffer),
            HCCL_ERROR("[InsTempRecvDpu][KernelRun]Aicpu Run Recv failed"),
            HcclResult::HCCL_E_INTERNAL);  // 本端ccl->本端output
    } else {
        HCCL_ERROR("[InsTempRecvDpu][Kernel Run] location [%d] is not supported!", recvChannel_.locationType);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    if (threadNum_ < 1) {
        HCCL_ERROR("[InsTempRecvDpu] Rank [%d], required thread error.", myRank_);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[InsTempRecvDpu] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempRecvDpu::DPUKernelRun(const TemplateDataParams &tempAlgParam,
    const std::map<u32, std::vector<ChannelInfo>> &channels, const u32 myRank,
    const std::vector<std::vector<uint32_t>> &subCommRanks)
{
#ifndef AICPU_COMPILE
    if (subCommRanks.empty() || subCommRanks[0].size() < 2) {
        HCCL_ERROR("[InsTempRecvDpu] [DPUKernelRun] no rank at all!");
    }
    u32 sendRank{0};
    for (auto rankId : subCommRanks[0]) {
        if (rankId != myRank) {
            sendRank = rankId;
            HCCL_INFO("[InsTempRecvDpu] [DPUKernelRun] my rank is [%d],  send rank is [%u].", myRank, sendRank);
        }
    }
    auto channelIter = channels.find(sendRank);
    if (channelIter == channels.end() || channelIter->second.empty()) {
        HCCL_ERROR(
            "[InsTempRecvDpu] [DPUKernelRun] my rank is [%d], send rank [%u] channel not found!", myRank, sendRank);
        return HCCL_E_INTERNAL;
    }
    ChannelInfo linkRecv = channelIter->second[0];
    // 实际数据流向是send端ccl -> recv端ccl
    DataSlice remoteInputBuffer(tempAlgParam.buffInfo.inputPtr,
        tempAlgParam.buffInfo.inBuffBaseOff,
        tempAlgParam.sliceSize,
        tempAlgParam.count);
    DataSlice localCclBuffer(tempAlgParam.buffInfo.hcclBuff.addr,
        0,
        tempAlgParam.sliceSize,
        tempAlgParam.count);  // cclbuffer不需要offset
    // 发送
    SlicesList recvSlicesList({remoteInputBuffer}, {localCclBuffer});
    DataInfo recvInfo(linkRecv, recvSlicesList);
    CHK_PRT_RET(RecvWrite(recvInfo), HCCL_ERROR("[InsTempRecvDpu][DPUKernelRun] Run Recv failed"), HcclResult::HCCL_E_INTERNAL);
    HCCL_INFO("[InsTempRecvDpu][DPUKernelRun] Run Recv sucess!");
#endif
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE_V2("InsTempRecvDpu", InsTempRecvDpu);
}  // namespace ops_hccl