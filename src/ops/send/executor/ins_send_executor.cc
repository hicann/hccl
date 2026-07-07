/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_send_executor.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
constexpr u32 P2P_CHANNEL_REPEAT_NUM = 2;
    std::string InsSendExecutor::Describe() const {
        return "Instruction based Send Executor.";
    }

    HcclResult InsSendExecutor::InitSendInfo(
        const HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo)
    {
        (void) comm;
        myRank_ = topoInfo->userRank;
        rankSize_ = topoInfo->userRankSize;
        devType_ = topoInfo->deviceType;
        remoteRank_ = param.sendRecvRemoteRank;
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);

        HCCL_INFO(
            "[InsSendExecutor][InitSendInfo] myRank [%u], remoteRank [%u], rankSize [%u], devType [%u], "
            "dataType [%u] dataTypeSize [%u]",
            myRank_, remoteRank_, rankSize_, devType_, dataType_, dataTypeSize_);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsSendExecutor::CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
    {
        // 初始化一些基本成员变量
        myRank_ = topoInfo->userRank;
        HCCL_DEBUG("[InsSendExecutor][CalcAlgHierarchyInfo][%d] Start.", myRank_);
        CHK_PRT_RET(
            (topoInfo->userRankSize == 0),
            HCCL_ERROR("[InsSendExecutor][CalcAlgHierarchyInfo] Rank [%d], rankSize is 0.", myRank_),
            HcclResult::HCCL_E_PARA);

        // AlgHierarchyInfoForAllLevel固定为一层
        algHierarchyInfo.infos.resize(1);
        algHierarchyInfo.infos[0].resize(1);
        algHierarchyInfo.infos[0][0].clear();
        for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
            algHierarchyInfo.infos[0][0].push_back(rankId);
        }
        
        HCCL_DEBUG("[InsSendExecutor][CalcAlgHierarchyInfo][%d] Success.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsSendExecutor::CalcRes(
        HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
    {
    #ifndef AICPU_COMPILE
        // 初始化一些基本成员变量
        InitSendInfo(comm, param, topoInfo);
        HCCL_DEBUG("[InsSendExecutor][CalcRes][%d]->[%d] Start.", myRank_, remoteRank_);

        resourceRequest.notifyNumOnMainThread = 0;
        resourceRequest.slaveThreadNum = 0;

        std::vector<HcclChannelDesc> level0Channels;
        bool isGroupEnabled = false;
        if (HcommIsSupportHcclGroupStatusGet()) {
            CHK_RET(HcclGroupStatusGet(&isGroupEnabled));
        }
        if (isGroupEnabled) {
            CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, remoteRank_, level0Channels, P2P_CHANNEL_REPEAT_NUM));
        } else {
            CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, remoteRank_, level0Channels));
        }
        resourceRequest.channels.push_back(level0Channels);

        HCCL_DEBUG("[InsSendExecutor][CalcRes][%d]->[%d] Success.", myRank_, remoteRank_);
    #endif
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsSendExecutor::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) {
        opMode_ = param.opMode;
        myRank_ = resCtx.topoInfo.userRank;
        remoteRank_ = param.sendRecvRemoteRank;
        // maxTmpMemSize_设定为ccl buffer的大小
        maxTmpMemSize_ = resCtx.cclMem.size;
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);
        dataSize_ = dataCount_ * dataTypeSize_;

        HCCL_DEBUG("[InsSendExecutor][Orchestrate][%d]->[%d] Start.", myRank_, remoteRank_);
        // 给channels_和threads_赋值
        const ThreadHandle &thread = resCtx.threads.at(0);
        auto channelIt = std::find_if(
            resCtx.channels.at(0).begin(), resCtx.channels.at(0).end(),
            [this](const ChannelInfo &channel_) {
                return channel_.remoteRank == remoteRank_;
            });
        CHK_PRT_RET(
            channelIt == resCtx.channels.at(0).end(),
            HCCL_ERROR("[InsSendExecutor][Orchestrate] Channel[%d]-[%d] not found.", myRank_, remoteRank_),
            HcclResult::HCCL_E_NOT_FOUND);
        const ChannelInfo &channel = *channelIt;
        
        // 判断是否为PCIE链路，如果是则使用read
        if (channel.protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
            isDmaRead_ = true;
        }

        if (opMode_ == OpMode::OFFLOAD) {
            CHK_RET(OrchestrateOffload(param, resCtx, thread, channel));
        } else {
            CHK_RET(OrchestrateOpbase(param, resCtx, thread, channel));
        }
        HCCL_DEBUG("[InsSendExecutor][Orchestrate][%d]->[%d] Success.", myRank_, remoteRank_);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsSendExecutor::OrchestrateP2p(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        ThreadHandle sendRecvThread) {
        opMode_ = param.opMode;
        myRank_ = resCtx.topoInfo.userRank;
        remoteRank_ = param.sendRecvRemoteRank;
        // maxTmpMemSize_设定为ccl buffer的大小
        maxTmpMemSize_ = resCtx.cclMem.size;
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);
        dataSize_ = dataCount_ * dataTypeSize_;
 
        HCCL_DEBUG("[InsSendExecutor][OrchestrateP2p][%d]->[%d] Start.", myRank_, remoteRank_);
        HCCL_INFO("[InsSendExecutor][OrchestrateP2p] resCtx channels size [%zu].", resCtx.channels.at(0).size());

        // 给channels_赋值
        const ChannelInfo &channel
            = (resCtx.channels.at(0).size() > 1)
                  ? ((myRank_ <= remoteRank_) ? resCtx.channels.at(0).at(0) : resCtx.channels.at(0).at(1))
                  : resCtx.channels.at(0).at(0);

        // 判断是否为PCIE链路，如果是则使用read
        if (channel.protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
            isDmaRead_ = true;
        }
 
        if (opMode_ == OpMode::OFFLOAD) {
            CHK_RET(OrchestrateOffload(param, resCtx, sendRecvThread, channel));
        } else {
            CHK_RET(OrchestrateOpbase(param, resCtx, sendRecvThread, channel));
        }
        HCCL_DEBUG("[InsSendExecutor][OrchestrateP2p][%d]->[%d] Success.", myRank_, remoteRank_);
 
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsSendExecutor::OrchestrateOffload(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const ThreadHandle &thread, const ChannelInfo &channel) {
        (void) resCtx;
        // 图模式本端可拿到对端output buffer地址，所以直接从本端input buffer到对端output buffer
        void *dstBufferPtr = static_cast<void *>(channel.remoteOutputGraphMode.addr);
        // UB传输最大数据量
        maxLoopTransSize_ = UB_MAX_DATA_SIZE;
        // 一次搬运最大数据个数
        maxLoopTransCount_ = maxLoopTransSize_ / dataTypeSize_;

        u64 dataCountToSend = dataCount_;
        u64 currentOffset = 0;
        std::vector<DataSlice> srcSlices;
        std::vector<DataSlice> dstSlices;
        HCCL_DEBUG("[InsSendExecutor][Orchestrate][%d]->[%d] OFFLOAD Generating tasks.", myRank_, remoteRank_);
        // 根据UB大小限制，对数据进行切分
        while (dataCountToSend > 0) {
            u64 transferCount = dataCountToSend > maxLoopTransCount_ ? maxLoopTransCount_ : dataCountToSend;
            u64 transferSize = transferCount * dataTypeSize_;
            srcSlices.emplace_back(param.inputPtr, currentOffset, transferSize, transferCount);
            dstSlices.emplace_back(dstBufferPtr, currentOffset, transferSize, transferCount);
            currentOffset = currentOffset + transferSize;
            dataCountToSend = dataCountToSend - transferCount;
        }
        SlicesList sendSlicesList{srcSlices, dstSlices};
        DataInfo sendInfo{channel, sendSlicesList};
        // 等待对端ready后，根据数据切片一片片往对端写，最后给对端发送fin信号
        if (isDmaRead_) {
            CHK_RET(SendRead(sendInfo, thread));
        } else {
            CHK_RET(SendWrite(sendInfo, thread));
        }

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsSendExecutor::OrchestrateOpbase(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const ThreadHandle &thread, const ChannelInfo &channel) {
        // UB和ccl Buffer取小为一次传输最大数据量
        maxLoopTransSize_ = std::min<u64>(UB_MAX_DATA_SIZE, channel.remoteCclMem.size);
        // 一次搬运最大数据个数
        maxLoopTransCount_ = maxLoopTransSize_ / dataTypeSize_;

        u64 dataCountToSend = dataCount_;
        u64 currentOffset = 0;
        HCCL_DEBUG("[InsSendExecutor][Orchestrate][%d]->[%d] OPBASE Generating tasks.", myRank_, remoteRank_);
        // 根据UB和ccl buffer大小限制，对数据进行切分
        while (dataCountToSend > 0) {
            u64 transferCount = dataCountToSend > maxLoopTransCount_ ? maxLoopTransCount_ : dataCountToSend;
            u64 transferSize = transferCount * dataTypeSize_;
            DataSlice inputSlice{param.inputPtr, currentOffset, transferSize, transferCount};
            // 因ccl buffer大小限制，每次往ccl buffer写一片数据，所以offset固定为0
            DataSlice remoteCclSlice{channel.remoteCclMem.addr, 0, transferSize, transferCount};
            if (isDmaRead_) {
                // Read模式下，先拷贝到自己的hcclBuffer上，再通知对端来读取数据
                DataSlice cclSlice{resCtx.cclMem.addr, 0, transferSize, transferCount};
                CHK_RET(LocalCopy(thread, inputSlice, cclSlice));
                SlicesList sendSlicesList{{cclSlice}, {remoteCclSlice}};
                DataInfo sendInfo{channel, sendSlicesList};
                CHK_RET(SendRead(sendInfo, thread));
            } else {
                // Write模式下，将数据直接写入对端cclBuffer
                SlicesList sendSlicesList{{inputSlice}, {remoteCclSlice}};
                DataInfo sendInfo{channel, sendSlicesList};
                CHK_RET(SendWrite(sendInfo, thread));
            }
            currentOffset = currentOffset + transferSize;
            dataCountToSend = dataCountToSend - transferCount;
        }

        return HcclResult::HCCL_SUCCESS;
    }

    REGISTER_EXECUTOR_IMPL(HcclCMDType::HCCL_CMD_SEND, InsSend, InsSendExecutor);
} // namespace ops_hccl
