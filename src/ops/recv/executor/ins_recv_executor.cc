/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_recv_executor.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

    std::string InsRecvExecutor::Describe() const {
        return "Instruction based Recv Executor.";
    }

    HcclResult InsRecvExecutor::InitCommInfo(
        HcclComm comm, const OpParam &param, const TopoInfo *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
    {
        myRank_ = topoInfo->userRank;
        rankSize_ = topoInfo->userRankSize;
        devType_ = topoInfo->deviceType;
        remoteRank_ = param.sendRecvRemoteRank;
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);

        HCCL_INFO(
            "[InsRecvExecutor][InitCommInfo] myRank [%u], remoteRank [%u], rankSize [%u], devType [%u], "
            "dataType [%u] dataTypeSize [%u]",
            myRank_, remoteRank_, rankSize_, devType_, dataType_, dataTypeSize_);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsRecvExecutor::CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfo *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
    {
        // 初始化一些基本成员变量
        myRank_ = topoInfo->userRank;
        HCCL_DEBUG("[InsRecvExecutor][CalcAlgHierarchyInfo][%d] Start.", myRank_);
        CHK_PRT_RET(
            topoInfo->userRankSize == 0,
            HCCL_ERROR("[InsRecvExecutor][CalcAlgHierarchyInfo] Rank [%d], rankSize is 0.", myRank_),
            HcclResult::HCCL_E_PARA);

        // AlgHierarchyInfoForAllLevel固定为一层
        algHierarchyInfo.infos.resize(1);
        algHierarchyInfo.infos[0].resize(1);
        algHierarchyInfo.infos[0][0].clear();
        for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
            algHierarchyInfo.infos[0][0].push_back(rankId);
        }

        HCCL_DEBUG("[InsRecvExecutor][CalcAlgHierarchyInfo][%d] Success.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsRecvExecutor::CalcRes(
        HcclComm comm, const OpParam &param, const TopoInfo *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
    {
        // 初始化一些基本成员变量
        InitCommInfo(comm, param, topoInfo, algHierarchyInfo);
        HCCL_DEBUG("[InsRecvExecutor][CalcRes][%d]<-[%d] Start.", myRank_, remoteRank_);

        resourceRequest.notifyNumOnMainThread = 0;
        resourceRequest.slaveThreadNum = 0;

        std::vector<HcclChannelDesc> level0Channels;
        CHK_RET(CreateChannelRequestByRankId(comm, myRank_, remoteRank_, level0Channels));
        resourceRequest.channels.push_back(level0Channels);

        HCCL_DEBUG("[InsRecvExecutor][CalcRes][%d]<-[%d] Success.", myRank_, remoteRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsRecvExecutor::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) {
        opMode_ = param.opMode;
        // ! 需要添加空指针判断，这里是暂时实现
        myRank_ = resCtx.topoInfo.userRank;
        remoteRank_ = param.sendRecvRemoteRank;
        HCCL_DEBUG("[InsRecvExecutor][Orchestrate][%d]<-[%d] Start.", myRank_, remoteRank_);

        // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
        maxTmpMemSize_ = resCtx.cclMem.size;
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);
        dataSize_ = dataCount_ * dataTypeSize_;

        // 给channels_和threads_赋值
        threads_ = resCtx.threads;
        const ThreadHandle &thread = threads_.at(0);
        auto channelIt = std::find_if(
            resCtx.channels.at(0).begin(), resCtx.channels.at(0).end(),
            [this](const ChannelInfo &channel_) {
                return channel_.remoteRank == remoteRank_;
            });
        CHK_PRT_RET(
            channelIt == resCtx.channels.at(0).end(),
            HCCL_ERROR("[InsRecvExecutor][Orchestrate] Channel[%d]-[%d] not found.", myRank_, remoteRank_),
            HcclResult::HCCL_E_NOT_FOUND);
        const ChannelInfo &channel = *channelIt;
        // 使用的是PUT DMAMode，所以无论图模式还是单算子，数据都是从对端的input buffer来
        // 此处channel.remoteInput不是对端input buffer
        // 但因为使用的是PUT模式此地址无用，且无法获取对端input buffer地址，此处仅示意作用
        void *srcBufferPtr = static_cast<void *>(channel.remoteInput.addr);
        void *dstBufferPtr = nullptr;
        if (opMode_ == OpMode::OFFLOAD) {
            // 图模式直接到本端output buffer
            dstBufferPtr = static_cast<void *>(param.outputPtr);
            // UB传输最大数据量
            maxLoopTransSize_ = UB_MAX_DATA_SIZE;
            // 一次搬运最大数据个数
            maxLoopTransCount_ = maxLoopTransSize_ / dataTypeSize_;

            u64 dataCountToRecv = dataCount_;
            u64 currentOffset = 0;
            std::vector<DataSlice> srcSlices;
            std::vector<DataSlice> dstSlices;
            HCCL_DEBUG("[InsRecvExecutor][Orchestrate][%d]<-[%d] OFFLOAD Generating tasks.", myRank_, remoteRank_);
            // 根据UB大小限制，对数据进行切分
            // 因使用的是PUT模式，此处循环和srcSlices、dstSlices其实无实际使用，仅示意
            while (dataCountToRecv > 0) {
                u64 transferCount = dataCountToRecv > maxLoopTransCount_ ? maxLoopTransCount_ : dataCountToRecv;
                u64 transferSize = transferCount * dataTypeSize_;
                srcSlices.emplace_back(srcBufferPtr, currentOffset, transferSize, transferCount);
                dstSlices.emplace_back(dstBufferPtr, currentOffset, transferSize, transferCount);
                currentOffset = currentOffset + transferSize;
                dataCountToRecv = dataCountToRecv - transferCount;
            }
            SlicesList recvSlicesList{srcSlices, dstSlices};
            DataInfo recvInfo{channel, recvSlicesList};
            // 给对端发送ready信号，最后等待对端发送fin信号
            CHK_RET(RecvWrite(recvInfo, thread));
        } else {
            // 单算子模式到本端ccl buffer(scratch buffer)
            dstBufferPtr = static_cast<void *>(resCtx.cclMem.addr);
            // UB和ccl Buffer取小为一次传输最大数据量
            maxLoopTransSize_ = std::min<u64>(UB_MAX_DATA_SIZE, maxTmpMemSize_);
            // 一次搬运最大数据个数
            maxLoopTransCount_ = maxLoopTransSize_ / dataTypeSize_;

            u64 dataCountToRecv = dataCount_;
            u64 currentOffset = 0;
            HCCL_DEBUG("[InsRecvExecutor][Orchestrate][%d]<-[%d] OPBASE Generating tasks.", myRank_, remoteRank_);
            // 根据UB和ccl buffer大小限制，对数据进行切分
            while (dataCountToRecv > 0) {
                u64 transferCount = dataCountToRecv > maxLoopTransCount_ ? maxLoopTransCount_ : dataCountToRecv;
                u64 transferSize = transferCount * dataTypeSize_;
                // 因使用的是PUT模式，此处srcSlices其实无实际使用，仅示意
                DataSlice srcSlice{srcBufferPtr, currentOffset, transferSize, transferCount};
                // 因ccl buffer大小限制，每次往ccl buffer写一片数据，所以offset固定为0
                DataSlice cclSlice{dstBufferPtr, 0, transferSize, transferCount};
                SlicesList recvSlicesList{{srcSlice}, {cclSlice}};
                DataInfo recvInfo{channel, recvSlicesList};
                // 因本端需要把ccl buffer数据localCopy到output buffer，所以此处每片数据都调用一次RecvWrite
                // 给对端发送ready信号后，等待对端发送fin信号
                CHK_RET(RecvWrite(recvInfo, thread));
                // 把ccl buffer数据localCopy到output buffer
                DataSlice outputSlice{param.outputPtr, currentOffset, transferSize, transferCount};
                CHK_RET(LocalCopy(thread, cclSlice, outputSlice));
                currentOffset = currentOffset + transferSize;
                dataCountToRecv = dataCountToRecv - transferCount;
            }
        }
        HCCL_DEBUG("[InsRecvExecutor][Orchestrate][%d]<-[%d] Success.", myRank_, remoteRank_);

        return HcclResult::HCCL_SUCCESS;
    }

    REGISTER_EXECUTOR_IMPL(HcclCMDType::HCCL_CMD_RECEIVE, InsRecv, InsRecvExecutor);
} // namespace ops_hccl
