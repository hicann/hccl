/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include "ins_send_dpu_executor.h"
#include "alg_data_trans_wrapper.h"
#include "template/host_nic/ins_temp_send_host_nic_dpu.h"

namespace ops_hccl {
    template <typename InsAlgTemplate>
    std::string InsSendDpuExecutor<InsAlgTemplate>::Describe() const
    {
        return "Instruction based Send Dpu Executor.";
    }

    template <typename InsAlgTemplate>
    HcclResult InsSendDpuExecutor<InsAlgTemplate>::InitCommInfo(HcclComm comm, const OpParam &param,
        const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo)
    {
        (void) algHierarchyInfo;
        (void) comm;
        myRank_ = topoInfo->userRank;
        rankSize_ = topoInfo->userRankSize;
        devType_ = topoInfo->deviceType;
        remoteRank_ = param.sendRecvRemoteRank;
        dataCount_ = param.DataDes.count;
        dataType_ = param.DataDes.dataType;
        dataTypeSize_ = static_cast<u64>(DATATYPE_SIZE_TABLE[dataType_]);

        HCCL_INFO("[InsSendDpuExecutor][InitCommInfo] myRank [%u], remoteRank [%u], rankSize [%u], devType [%u], "
            "dataType [%u], dataTypeSize[%u]",
            myRank_, remoteRank_, rankSize_, devType_, dataType_, dataTypeSize_);
        
        return HcclResult::HCCL_SUCCESS;
    }

    template <typename InsAlgTemplate>
    HcclResult InsSendDpuExecutor<InsAlgTemplate>::CalcAlgHierarchyInfo(HcclComm comm,
        TopoInfoWithNetLayerDetails *topoInfo,  AlgHierarchyInfoForAllLevel &algHierarchyInfo)
    {
        // 初始化一些基本成员变量
        myRank_ = topoInfo->userRank;
        HCCL_DEBUG("[InsSendDpuExecutor][CalcAlgHierarchyInfo][%d] Start.", myRank_);
        CHK_PRT_RET((topoInfo->userRankSize == 0),
            HCCL_ERROR("[InsSendDpuExecutor][CalcAlgHierarchyInfo] Rank [%d], rankSize is 0.", myRank_),
            HcclResult::HCCL_E_PARA);
        
        // AlgHierarchyInfoForAllLevel固定为一层
        algHierarchyInfo.infos.resize(1);
        algHierarchyInfo.infos[0].resize(1);
        algHierarchyInfo.infos[0][0].clear();
        for (uint32_t rankId = 0; rankId < topoInfo->userRankSize; rankId++) {
            algHierarchyInfo.infos[0][0].push_back(rankId);
        }

        algHierarchyInfo_ = algHierarchyInfo;
        
        HCCL_DEBUG("[InsSendDpuExecutor][CalcAlgHierarchyInfo][%d] Success.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    template <typename InsAlgTemplate>
    HcclResult InsSendDpuExecutor<InsAlgTemplate>::CalcRes(HcclComm comm, const OpParam &param,
        const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
    {
        // 初始化一些基本成员变量
        InitCommInfo(comm, param, topoInfo, algHierarchyInfo);
        HCCL_DEBUG("[InsSendDpuExecutor][CalcRes][%d]->[%d] Start.", myRank_, remoteRank_);

        resourceRequest.notifyNumOnMainThread = 0;
        resourceRequest.slaveThreadNum = 0;

        std::vector<HcclChannelDesc> level0Channels;
        CHK_RET(CreateChannelRequestByRankId(comm, param, myRank_, remoteRank_, level0Channels));
        resourceRequest.channels.push_back(level0Channels);

        HCCL_DEBUG("[InsSendDpuExecutor][CalcRes][%d]->[%d] Success.", myRank_, remoteRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    template <typename InsAlgTemplate>
    HcclResult InsSendDpuExecutor<InsAlgTemplate>::Orchestrate(const OpParam &param,
        const AlgResourceCtxSerializable &resCtx)
    {
        HCCL_DEBUG("[InsSendDpuExecutor][Orchestrate][%d]->[%d] Start.", myRank_, remoteRank_);

        opMode_ = param.opMode;
        myRank_ = resCtx.topoInfo.userRank;
        remoteRank_ = param.sendRecvRemoteRank;

        // 获取Channel信息
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

        TemplateDataParams tempAlgParams;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.inputSize = param.inputSize;
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;

        // 构造template资源
        TemplateResource templateResource;
        templateResource.channels = remoteRankToChannelInfo_[0];
        templateResource.threads = resCtx.threads;
        templateResource.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        templateResource.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

        // 选择template
        std::shared_ptr<InsAlgTemplate> algTemplateDpu = std::make_shared<InsAlgTemplate>(param, myRank_,
            resCtx.algHierarchyInfo.infos[0]);
        
        CHK_RET(algTemplateDpu->KernelRun(param, tempAlgParams, templateResource));

        HCCL_DEBUG("[InsSendDpuExecutor][Orchestrate][%d]->[%d] Success.", myRank_, remoteRank_);
        return HcclResult::HCCL_SUCCESS;
    }

    template <typename InsAlgTemplate>
    HcclResult InsSendDpuExecutor<InsAlgTemplate>::OrchestrateWithThread(
        const OpParam &param, const AlgResourceCtxSerializable &resCtx, ThreadHandle sendRecvThread)
    {
        (void)sendRecvThread;
        return Orchestrate(param, resCtx);
    }

    // opv2流程使用opv2_insSendHostDpu算法名
    REGISTER_EXECUTOR_IMPL_NO_TOPOMATCH(HcclCMDType::HCCL_CMD_SEND, opv2_insSendHostDpu, InsSendDpuExecutor, \
        InsTempSendHostNicDpu);
} // namespace ops_hccl