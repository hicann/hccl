/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_send_sole_executor.h"
#include "ins_temp_send_dpu.h"

namespace ops_hccl
{
    template <typename InsAlgTemplate>
    InsV2SendSoleExecutor<InsAlgTemplate>::InsV2SendSoleExecutor()
    {
    }

    template <typename InsAlgTemplate>
    std::string InsV2SendSoleExecutor<InsAlgTemplate>::Describe() const
    {
        return "Instruction based Send Executor.";
    }

    template <typename InsAlgTemplate>
    HcclResult InsV2SendSoleExecutor<InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                                                              const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
    {
        // 变量检查
        CHK_PTR_NULL(topoInfo);
        // 初始化一些基本成员变量
        myRank_ = topoInfo->userRank;
        devType_ = topoInfo->deviceType;
        recvRank_ = param.sendRecvRemoteRank;
        dataType_ = param.DataDes.dataType;
        dataCount_ = param.DataDes.count;
        dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
        HCCL_INFO("[InsV2SendSoleExecutor][CalcRes] myRank [%u], receiveRank [%u], devType [%u],dataType[%u] "
                  "dataTypeSize[%u] ",
                  myRank_,
                  recvRank_,
                  devType_,
                  dataType_,
                  dataTypeSize_);
        HCCL_INFO("[InsV2SendSoleExecutor][CalcRes] algHierarchyInfo size is [%u]", algHierarchyInfo.infos.size());
        algHierarchyInfo_ = algHierarchyInfo;
        if (algHierarchyInfo_.infos.empty())
        {
            HCCL_ERROR("[InsV2SendSoleExecutor][CalcRes] algHierarchyInfo infos is empty!");
            return HCCL_E_PARA;
        }
        std::shared_ptr<InsAlgTemplate> insTemp =
            std::make_shared<InsAlgTemplate>(param, myRank_, algHierarchyInfo_.infos[0]);
        AlgResourceRequest resReq;
        // 进行资源计算
        // 得到通信所需的 resourceRequest.channels[0]
        CHK_RET(insTemp->CalcRes(comm, param, topoInfo, resReq));
        resourceRequest.channels.resize(1);
        resourceRequest.channels[0] = resReq.channels[0];
        return HCCL_SUCCESS;
    }

    template <typename InsAlgTemplate>
    HcclResult InsV2SendSoleExecutor<InsAlgTemplate>::CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
    {
        myRank_ = topoInfo->userRank;
        // AlgHierarchyInfoForAllLevel固定为一层
        CHK_PRT_RET((topoInfo->userRankSize == 0),
                    HCCL_ERROR("[InsV2SendSoleExecutor][CalcAlgHierarchyInfo] Rank [%u], rankSize is 0.", myRank_),
                    HcclResult::HCCL_E_PARA);

        algHierarchyInfo.infos.resize(1);
        algHierarchyInfo.infos[0].resize(1);
        algHierarchyInfo.infos[0][0].clear();
        for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++)
        {
            algHierarchyInfo.infos[0][0].push_back(rankId);
        }
        HCCL_INFO("[InsV2SendSoleExecutor][CalcAlgHierarchyInfo][%u] Success.", myRank_);
        return HCCL_SUCCESS;
    }

    template <typename InsAlgTemplate>
    HcclResult InsV2SendSoleExecutor<InsAlgTemplate>::Orchestrate(
        const OpParam &param, const AlgResourceCtxSerializable &resCtx)
    {
        HCCL_INFO("[InsV2SendSoleExecutor][Orchestrate] Orchestrate Start");
        myRank_ = resCtx.topoInfo.userRank;
        recvRank_ = param.sendRecvRemoteRank;
        dataCount_ = param.DataDes.count;
        dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
        dataSize_ = dataCount_ * dataTypeSize_;
        dataType_ = param.DataDes.dataType;
        if (resCtx.threads.empty())
        {
            HCCL_ERROR("[InsV2SendSoleExecutor][Orchestrate] threads is empty!");
            return HCCL_E_INTERNAL;
        }
        thread_ = resCtx.threads.front();
        algHierarchyInfo_.infos.resize(1);
        algHierarchyInfo_.infos[0].resize(1);
        algHierarchyInfo_.infos[0][0].push_back(myRank_);
        algHierarchyInfo_.infos[0][0].push_back(recvRank_);
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
        if (remoteRankToChannelInfo_.empty() || remoteRankToChannelInfo_[0].empty())
        {
            HCCL_ERROR("[InsV2SendSoleExecutor][Orchestrate] no channel found!");
            return HCCL_E_INTERNAL;
        }
        auto channelIter = remoteRankToChannelInfo_[0].find(recvRank_);
        if (channelIter == remoteRankToChannelInfo_[0].end() || channelIter->second.empty())
        {
            HCCL_ERROR("[InsV2SendSoleExecutor][Orchestrate] recv rank [%u] channel not found!", recvRank_);
            return HCCL_E_INTERNAL;
        }
        sendChannel_ = channelIter->second[0];
        // 构造template
        std::shared_ptr<InsAlgTemplate> algTemplate =
            std::make_shared<InsAlgTemplate>(param, myRank_, algHierarchyInfo_.infos[0]);
        // 处理一些buffer
        TemplateDataParams tempAlgParams;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.outputPtr = sendChannel_.remoteCclMem.addr; // 无论跨框还是框内 都要发送到对方的ccl上
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;                   // 整个ccl buffer
        // template资源
        TemplateResource templateResource;
        templateResource.channels = remoteRankToChannelInfo_[0];
        templateResource.threads = resCtx.threads;
        templateResource.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        templateResource.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
        u64 resDataSize = dataSize_;
        maxTmpMemSize_ = std::min<u64>(UB_MAX_DATA_SIZE, resCtx.cclMem.size); // maxTmpMemSize_取ub和ccl的最小值
        u64 maxRoundTransferSize = (maxTmpMemSize_ / dataTypeSize_) * dataTypeSize_;
        while (resDataSize > 0)
        {
            u64 transferSize = resDataSize > maxTmpMemSize_ ? maxRoundTransferSize : resDataSize;
            tempAlgParams.sliceSize = transferSize;
            tempAlgParams.count = transferSize / dataTypeSize_;
            CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateResource));
            tempAlgParams.buffInfo.inBuffBaseOff += transferSize;
            resDataSize = resDataSize - transferSize;
        }
        return HCCL_SUCCESS;
    }

#if !defined(HCCL_CANN_COMPAT_850)
    REGISTER_EXECUTOR_IMPL_NO_TOPOMATCH(HcclCMDType::HCCL_CMD_SEND, InsSendDPU, InsV2SendSoleExecutor, InsTempSendDpu);
#endif /* !HCCL_CANN_COMPAT_850 */
} // namespace ops_hccl