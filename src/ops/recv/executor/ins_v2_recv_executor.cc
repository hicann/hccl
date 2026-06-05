/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_recv_executor.h"
#include "alg_data_trans_wrapper.h"
#include "hccl_aiv_utils.h"

namespace ops_hccl {

    std::string InsV2RecvExecutor::Describe() const
    {
        return "Instruction based Recv Executor.";
    }

    HcclResult InsV2RecvExecutor::InitRecvInfo(
        const HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
    {
        (void) comm;
        (void) algHierarchyInfo;
        myRank_ = topoInfo->userRank;
        rankSize_ = topoInfo->userRankSize;
        devType_ = topoInfo->deviceType;
        remoteRank_ = param.sendRecvRemoteRank;
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);

        HCCL_INFO(
            "[InsV2RecvExecutor][InitRecvInfo] myRank [%u], remoteRank [%u], rankSize [%u], devType [%u], "
            "dataType [%u] dataTypeSize [%u]",
            myRank_, remoteRank_, rankSize_, devType_, dataType_, dataTypeSize_);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsV2RecvExecutor::CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
    {
        // 初始化一些基本成员变量
        myRank_ = topoInfo->userRank;
        HCCL_DEBUG("[InsV2RecvExecutor][CalcAlgHierarchyInfo][%d] Start.", myRank_);
        CHK_PRT_RET(
            topoInfo->userRankSize == 0,
            HCCL_ERROR("[InsV2RecvExecutor][CalcAlgHierarchyInfo] Rank [%d], rankSize is 0.", myRank_),
            HcclResult::HCCL_E_PARA);

        // AlgHierarchyInfoForAllLevel固定为一层
        algHierarchyInfo.infos.resize(1);
        algHierarchyInfo.infos[0].resize(1);
        algHierarchyInfo.infos[0][0].clear();
        for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
            algHierarchyInfo.infos[0][0].push_back(rankId);
        }

        HCCL_DEBUG("[InsV2RecvExecutor][CalcAlgHierarchyInfo][%d] Success.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsV2RecvExecutor::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit) const
    {
        (void)dataSize;

        if (numBlocksLimit < 1) {
            HCCL_ERROR("[InsV2RecvExecutor] core num[%u] is less than 1", numBlocksLimit);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }

        numBlocks = numBlocksLimit;
        HCCL_INFO("[InsV2RecvExecutor] Actually use core num[%u]", numBlocks);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsV2RecvExecutor::CalcRes(
        HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
    {
        // 初始化一些基本成员变量
        InitRecvInfo(comm, param, topoInfo, algHierarchyInfo);
        HCCL_DEBUG("[InsV2RecvExecutor][CalcRes][%d]<-[%d] Start.", myRank_, remoteRank_);

        resourceRequest.notifyNumOnMainThread = 0;
        resourceRequest.slaveThreadNum = 0;

        std::vector<HcclChannelDesc> level0Channels;
        CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, remoteRank_, level0Channels));
        resourceRequest.channels.push_back(level0Channels);

        HCCL_DEBUG("[InsV2RecvExecutor][CalcRes][%d]<-[%d] Success.", myRank_, remoteRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsV2RecvExecutor::Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
    {
        opMode_ = param.opMode;
        myRank_ = resCtx.topoInfo.userRank;
        remoteRank_ = param.sendRecvRemoteRank;
        // maxTmpMemSize_设定为ccl buffer的大小
        maxTmpMemSize_ = resCtx.cclMem.size;
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);
        dataSize_ = dataCount_ * dataTypeSize_;

        HCCL_DEBUG("[InsV2RecvExecutor][Orchestrate][%d]<-[%d] Start.", myRank_, remoteRank_);
        CHK_RET(OrchestrateImpl(param, resCtx));
        HCCL_DEBUG("[InsV2RecvExecutor][Orchestrate][%d]<-[%d] Success.", myRank_, remoteRank_);

        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult InsV2RecvExecutor::OrchestrateImpl(const OpParam &param, const AlgResourceCtxSerializable &resCtx)
    {
        HCCL_INFO("[InsV2RecvExecutor][KernelRun] start: rank is %d, count is %u, dataType is %u, destRank is %u",
            myRank_, dataCount_, static_cast<u32>(dataType_), remoteRank_);

        u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
        u64 maxScratchDataSize = std::min(transportBoundDataSize, maxTmpMemSize_);
        u64 maxScratchDataCount = maxScratchDataSize / dataTypeSize_;
        CHK_PRT_RET(maxScratchDataCount == 0,
            HCCL_ERROR("[InsV2RecvExecutor][OrchestrateOpbase] maxScratchDataCount is 0"),
            HCCL_E_INTERNAL);

        u64 loopTimes = dataCount_ / maxScratchDataCount + static_cast<u64>(dataCount_ % maxScratchDataCount != 0);
        u64 processedDataCount = 0;
        for (u64 loop = 0; loop < loopTimes; loop++) {
            sliceId_++; // 自动增长sliceId，传入aivTag
            u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxScratchDataCount;
            HCCL_INFO("[InsV2RecvExecutor][OrchestrateOpbase] myRank[%u], loop[%llu] sliceId_[%llu] "
                "currDataCount[%llu], processedDataCount[%llu]",
                myRank_, loop, sliceId_, currDataCount, processedDataCount);

            AivOpArgs aivSendArgs;
            aivSendArgs.cmdType = HcclCMDType::HCCL_CMD_RECEIVE;
            aivSendArgs.input = reinterpret_cast<u64>(param.inputPtr);
            aivSendArgs.output = reinterpret_cast<u64>(param.outputPtr) + processedDataCount * dataTypeSize_;
            aivSendArgs.rank = u32(myRank_);
            aivSendArgs.dataType = dataType_;
            aivSendArgs.sliceId = sliceId_;
            aivSendArgs.sendRecvRemoteRank = remoteRank_;
            aivSendArgs.rankSize = resCtx.topoInfo.userRankSize;
            aivSendArgs.count = currDataCount; // 需要传输的数据量
            aivSendArgs.buffersIn = resCtx.aivCommInfoPtr;
            aivSendArgs.stream = param.stream;
            aivSendArgs.isOpBase = (opMode_ == OpMode::OPBASE);
            aivSendArgs.yRankSize = 0;
            aivSendArgs.zRankSize = 0;
            aivSendArgs.xRankSize = resCtx.topoInfo.userRankSize;

            CHK_RET(CalNumBlocks(aivSendArgs.numBlocks, currDataCount * dataTypeSize_, param.numBlocksLimit));

            aivSendArgs.repeatNum = 1; // 不重复
            aivSendArgs.inputSliceStride = 0;
            aivSendArgs.outputSliceStride = 0;
            aivSendArgs.inputRepeatStride = 0;
            aivSendArgs.outputRepeatStride = 0;

            CHK_RET(ExecuteKernelLaunch(aivSendArgs));
            processedDataCount += currDataCount;
        }

        HCCL_INFO("[InsV2RecvExecutor][KernelRun] end: rank[%d]", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    REGISTER_EXECUTOR_IMPL(HcclCMDType::HCCL_CMD_RECEIVE, AivRecv, InsV2RecvExecutor);

} // namespace ops_hccl
