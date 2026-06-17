/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_batch_send_recv_dpu.h"

namespace ops_hccl
{
    constexpr u32 DPU_TIMEOUT = 180000;
    InsTempBatchSendRecvDpu::InsTempBatchSendRecvDpu()
    {
    }
    // ! 已编码完成
    InsTempBatchSendRecvDpu::InsTempBatchSendRecvDpu(const OpParam &param,
                                   const u32 rankId, // 传通信域的rankId，userRank
                                   const std::vector<std::vector<u32>> &subCommRanks)
        : InsAlgTemplateBase(param, rankId, subCommRanks)
    {
    }

    // ! 已编码完成
    InsTempBatchSendRecvDpu::~InsTempBatchSendRecvDpu()
    {
    }

    // ! 已编码完成
    HcclResult InsTempBatchSendRecvDpu::CalcRes(
        HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
    {
        HCCL_INFO("[InsTempBatchSendRecvDpu][CalcRes] Sucessfully calres!");
        return HCCL_SUCCESS;
    }

    // ! 基本编码完成，剩余数据序列化
    HcclResult InsTempBatchSendRecvDpu::KernelRun(
        const OpParam &param, const TemplateDataParams &tempAlgParams, TemplateResource &templateResource)
    {
        HCCL_INFO("[InsTempBatchSendRecvDpu] KernelRun enter.");
        // 得到thread和channel
        recvRank_ = param.sendRecvRemoteRank;
        threadNum_ = templateResource.threads.size();
        hcclbuffBlockMemSize_ = tempAlgParams.inputSliceStride;
        HCCL_INFO("[InsTempBatchSendRecvDpu] recvRank_[%u] param.sendRecvRemoteRank[%u]", recvRank_, param.sendRecvRemoteRank);
        if (threadNum_ < 1) {
            HCCL_ERROR("[InsTempBatchSendRecvDpu] Rank [%d], required thread error.", myRank_);
            return HCCL_E_INTERNAL;
        }
        if (tempAlgParams.opType >= BatchSendRecvOpType::DEFAULT) {
            HCCL_ERROR("[InsTempBatchSendRecvDpu] opType [%d] error.", tempAlgParams.opType);
            return HCCL_E_INTERNAL;
        }
        thread_ = templateResource.threads[0];
        subThread_ = templateResource.threads[1];
        auto channelIter = templateResource.channels.find(recvRank_);
        if (channelIter == templateResource.channels.end() || channelIter->second.empty()) {
            HCCL_ERROR(
                "[InsTempBatchSendRecvDpu][KernelRun]my rank is [%d], receive rank [%u] channel not found!", myRank_, recvRank_);
            return HCCL_E_INTERNAL;
        }
        sendRecvChannel_ = channelIter->second[0];
        subSendRecvChannel_ = channelIter->second[1];
        processSize_ = tempAlgParams.sliceSize;
        count_ = tempAlgParams.count;
        dataCount_ = param.DataDes.count;
        dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
        dataSize_ = dataCount_ * dataTypeSize_;
        dataType_ = param.DataDes.dataType;
        // 跨框流程要走dpu
        if (sendRecvChannel_.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
            if (tempAlgParams.opType == BatchSendRecvOpType::SEND) {
                // aicpu先把inputbuffer的内容localcopy给cclbuffer
                DataSlice inputBuffer(
                    tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_);
                DataSlice localCclBuffer(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_, processSize_, count_); // copy到对应本rank划分的cclbuffer
                CHK_RET(LocalCopy(thread_, inputBuffer, localCclBuffer));           // 本端inputbuffer->本端ccl
            }

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
            dpuRunInfo.templateName = "InsTempBatchSendRecvDpu";
            dpuRunInfo.tempAlgParams = tempAlgParams;
            dpuRunInfo.channels = templateResource.channels;
            dpuRunInfo.myRank = myRank_;
            std::vector<std::vector<uint32_t>> subCommRanks;
            subCommRanks.push_back({myRank_, recvRank_});
            dpuRunInfo.subCommRanks = subCommRanks;
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

            if (tempAlgParams.opType == BatchSendRecvOpType::RECV) {
                // aicpu把cclbuffer的内容localcopy给outputbuffer
                DataSlice localCclBuffer(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff + recvRank_ * hcclbuffBlockMemSize_, processSize_, count_); // 从对端rank对应划分的cclbuffer接收
                DataSlice outputBuffer(
                    tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);

                CHK_RET(LocalCopy(thread_, localCclBuffer, outputBuffer)); // 本端ccl->本端output
            }
            HCCL_INFO("[InsTempBatchSendRecvDpu] Run End");
        }
        else if (sendRecvChannel_.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
            if (tempAlgParams.opType == BatchSendRecvOpType::SEND) {
                // 直接发送到对端的cclbuffer上
                DataSlice inputBuffer(
                    tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_);
                DataSlice remoteCclBuffer(tempAlgParams.buffInfo.outputPtr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff + myRank_ * hcclbuffBlockMemSize_, processSize_, count_); // 发送到对端对应本rank划分的cclbuffer
                // 发送
                SlicesList sendSlicesList({inputBuffer}, {remoteCclBuffer});
                DataInfo sendInfo(recvRank_ < myRank_ ? sendRecvChannel_ : subSendRecvChannel_, sendSlicesList);
                CHK_PRT_RET(SendWrite(sendInfo, thread_),
                            HCCL_ERROR("[InsTempBatchSendRecvDpu][KernelRun]Aicpu Run Send failed"),
                            HcclResult::HCCL_E_INTERNAL);
            } else if (tempAlgParams.opType == BatchSendRecvOpType::RECV) {
                // 接收对方inputbuffer的数据到本端cclbuffer
                DataSlice remoteInputBuffer(
                    tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.inBuffBaseOff, processSize_, count_);
                DataSlice localCclBuffer(tempAlgParams.buffInfo.hcclBuff.addr,
                    tempAlgParams.buffInfo.hcclBuffBaseOff + recvRank_ * hcclbuffBlockMemSize_, processSize_, count_); // 从给对端rank划分的cclbuffer接收
                // 发送
                SlicesList recvSlicesList({remoteInputBuffer}, {localCclBuffer});
                DataInfo recvInfo(recvRank_ > myRank_ ? sendRecvChannel_ : subSendRecvChannel_, recvSlicesList);
                CHK_PRT_RET(RecvWrite(recvInfo, subThread_),
                            HCCL_ERROR("[InsTempRecvDpu][KernelRun]Aicpu Run Recv failed"),
                            HcclResult::HCCL_E_INTERNAL);
                // 把cclbuffer的内容localcopy给outputbuffer
                DataSlice outputBuffer(
                    tempAlgParams.buffInfo.outputPtr, tempAlgParams.buffInfo.outBuffBaseOff, processSize_, count_);
                CHK_PRT_RET(LocalCopy(subThread_, localCclBuffer, outputBuffer),
                            HCCL_ERROR("[InsTempRecvDpu][KernelRun]Aicpu Run Recv failed"),
                            HcclResult::HCCL_E_INTERNAL); // 本端ccl->本端output
            }
        }
        else {
            HCCL_ERROR("[InsTempBatchSendRecvDpu][Kernel Run] location [%d] is not supported!", sendRecvChannel_.locationType);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }

        HCCL_INFO("[InsTempBatchSendRecvDpu] Run End");
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsTempBatchSendRecvDpu::DPUKernelRun(const TemplateDataParams &tempAlgParam,
                                            const std::map<u32, std::vector<ChannelInfo>> &channels, const u32 myRank,
                                            const std::vector<std::vector<uint32_t>> &subCommRanks)
    {
#ifndef AICPU_COMPILE
        if (subCommRanks.empty() || subCommRanks[0].size() < 2) {
            HCCL_ERROR("[InsTempBatchSendRecvDpu] [DPUKernelRun] no rank at all!");
            return HCCL_E_PARA;
        }
        u32 sendRank{0};
        hcclbuffBlockMemSize_ = tempAlgParam.inputSliceStride;
        for (auto rankId : subCommRanks[0]) {
            if (rankId != myRank) {
                sendRank = rankId;
                HCCL_INFO("[InsTempBatchSendRecvDpu] [DPUKernelRun] my rank is [%d],  send rank is [%u].", myRank, sendRank);
            }
        }
        auto channelIter = channels.find(sendRank);
        if (channelIter == channels.end() || channelIter->second.empty()) {
            HCCL_ERROR(
                "[InsTempBatchSendRecvDpu] [DPUKernelRun] my rank is [%d], send rank [%u] channel not found!", myRank, sendRank);
            return HCCL_E_INTERNAL;
        }

        if (tempAlgParam.opType == BatchSendRecvOpType::RECORD) {
            ChannelInfo linkRecv = channelIter->second[0];
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(0, linkRecv.handle, NOTIFY_IDX_ACK)));
            return HCCL_SUCCESS;
        } else if (tempAlgParam.opType == BatchSendRecvOpType::SEND) {
            ChannelInfo linkSend = channelIter->second[0];
            DataSlice localCclBuffer(tempAlgParam.buffInfo.hcclBuff.addr,
                tempAlgParam.buffInfo.hcclBuffBaseOff + myRank * hcclbuffBlockMemSize_, tempAlgParam.sliceSize);
            DataSlice remoteCclBuffer(tempAlgParam.buffInfo.outputPtr,
                tempAlgParam.buffInfo.hcclBuffBaseOff + myRank * hcclbuffBlockMemSize_, tempAlgParam.sliceSize); // ccl
            // 发送
            SlicesList sendSlicesList({localCclBuffer}, {remoteCclBuffer});
            DataInfo sendInfo(linkSend, sendSlicesList);
            const std::vector<DataSlice> srcSlices = sendInfo.slices_.srcSlices_;
            const std::vector<DataSlice> dstSlices = sendInfo.slices_.dstSlices_;
            const ChannelInfo &sendChannel = sendInfo.channel_;
            u32 sliceNum = srcSlices.size();
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(0, sendChannel.handle, NOTIFY_IDX_ACK, DPU_TIMEOUT)));
            for (int i = 0; i < sliceNum; i++) {
                const DataSlice srcSlice = srcSlices[i];
                const DataSlice dstSlcie = dstSlices[i];
                void *dst = static_cast<void *>(static_cast<s8 *>(dstSlcie.addr_) + dstSlcie.offset_);
                void *src = static_cast<void *>(static_cast<s8 *>(srcSlice.addr_) + srcSlice.offset_);
                CHK_RET(static_cast<HcclResult>(
                    HcommWriteWithNotifyNbiOnThread(0, sendChannel.handle, dst, src, srcSlice.size_, NOTIFY_IDX_DATA_SIGNAL)));
            }
            return HCCL_SUCCESS;
        } else if (tempAlgParam.opType == BatchSendRecvOpType::RECV) {
            ChannelInfo linkRecv = channelIter->second[0];
            // 实际数据流向是send端ccl -> recv端ccl
            DataSlice remoteInputBuffer(tempAlgParam.buffInfo.inputPtr,
                                        tempAlgParam.buffInfo.inBuffBaseOff,
                                        tempAlgParam.sliceSize,
                                        tempAlgParam.count);
            DataSlice localCclBuffer(tempAlgParam.buffInfo.hcclBuff.addr,
                                    0,
                                    tempAlgParam.sliceSize,
                                    tempAlgParam.count); // cclbuffer不需要offset
            SlicesList recvSlicesList({remoteInputBuffer}, {localCclBuffer});
            DataInfo recvInfo(linkRecv, recvSlicesList);
            const std::vector<DataSlice> srcSlices = recvInfo.slices_.srcSlices_;
            const std::vector<DataSlice> dstSlices = recvInfo.slices_.dstSlices_;
            const ChannelInfo &recvChannel = recvInfo.channel_;
            u32 sliceNum = srcSlices.size();
            for (int i = 0; i < sliceNum; i++) {
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(0, recvChannel.handle, NOTIFY_IDX_DATA_SIGNAL, DPU_TIMEOUT)));
            }
            CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
            return HCCL_SUCCESS;
        } else if (tempAlgParam.opType == BatchSendRecvOpType::FENCE) {
            ChannelInfo linkRecv = channelIter->second[0];
            CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(0, linkRecv.handle)));
            CHK_RET(static_cast<HcclResult>(HcommFenceOnThread(0)));
        }

        HCCL_INFO("[InsTempBatchSendRecvDpu][DPUKernelRun] Run success!");
#endif
        return HCCL_SUCCESS;
    }

    REGISTER_TEMPLATE_V2("InsTempBatchSendRecvDpu", InsTempBatchSendRecvDpu);
} // namespace ops_hccl