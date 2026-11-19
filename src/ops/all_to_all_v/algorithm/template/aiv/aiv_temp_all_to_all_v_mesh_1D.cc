/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_aiv_utils.h"
#include "aiv/aiv_temp_all_to_all_v_mesh_1D.h"
#include "config_log.h"

namespace ops_hccl {

AivTempAlltoAllVMesh1D::AivTempAlltoAllVMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{}

AivTempAlltoAllVMesh1D::~AivTempAlltoAllVMesh1D() {}

HcclResult AivTempAlltoAllVMesh1D::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    u32 threadNum = 1;
    resourceRequest.slaveThreadNum = threadNum - 1;
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(1);
    }
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    std::vector<HcclChannelDesc> level0Channels;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMeshClosMultiJetty(comm, param, topoInfo, subCommRanks_, myChannelDescs, true));
        // AIV直驱URMA建链须为UB_CTP协议(WriteWithNotifyNbi/WriteValueNbi仅UB_CTP支持)
        CommProtocol expectedProtocol
            = (param.opExecuteConfig == OpExecuteConfig::AIV_URMA) ? COMM_PROTOCOL_UB_CTP : COMM_PROTOCOL_UB_MEM;
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == expectedProtocol) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[AivTempAlltoAllVMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    resRequest_ = resourceRequest;
    HCCL_WARNING("[AivTempAlltoAllVMesh1D] Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllVMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    if (numBlocksLimit == 0) {
        HCCL_ERROR("[AivTempAlltoAllVMesh1D] numBlocksLimit is 0");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    if (opExecuteConfig_ == OpExecuteConfig::AIV_URMA) {
        u32 urmaLimit = std::min(tempRankSize_, AIV_URMA_ALLTOALLV_GROUP_SIZE);
        numBlocks = std::min(numBlocksLimit, urmaLimit);
    } else if (numBlocksLimit < tempRankSize_) {
        numBlocks = numBlocksLimit;
    } else {
        // 多核一 rank，对齐到 rankSize 倍数
        numBlocks = numBlocksLimit / tempRankSize_ * tempRankSize_;
    }
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[AivTempAlltoAllVMesh1D] Actually use core num[%u], limit[%u]", numBlocks, numBlocksLimit);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAlltoAllVMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempAlltoAllVMesh1D] KernelRun start");

    opExecuteConfig_ = param.opExecuteConfig;
    IncSliceId(); // 自动增长sliceId，传入sliceId
    dataType_ = param.all2AllVDataDes.sendType;
    AivOpArgs aivAlltoAllVArgs;
    aivAlltoAllVArgs.cmdType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    aivAlltoAllVArgs.totalDataCount = param.dataCount;
    aivAlltoAllVArgs.input = reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivAlltoAllVArgs.output = reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivAlltoAllVArgs.rank = u32(myRank_);
    aivAlltoAllVArgs.rankSize = tempRankSize_;
    aivAlltoAllVArgs.count = tempAlgParams.sliceSize / HCCL_SIZE_TABLE[dataType_];
    aivAlltoAllVArgs.dataType = dataType_;
    aivAlltoAllVArgs.op = param.reduceType;
    aivAlltoAllVArgs.root = root_;
    aivAlltoAllVArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivAlltoAllVArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivAlltoAllVArgs.stream = param.stream;
    aivAlltoAllVArgs.hcclComm = static_cast<HcclComm>(param.hcclComm);
    aivAlltoAllVArgs.isOpBase = (param.opMode == OpMode::OPBASE);

    size_t count = tempRankSize_;
    std::copy(
        tempAlgParams.sendCounts.data(), tempAlgParams.sendCounts.data() + count,
        aivAlltoAllVArgs.extraArgs.sendCounts);
    std::copy(
        tempAlgParams.sdispls.data(), tempAlgParams.sdispls.data() + count, aivAlltoAllVArgs.extraArgs.sendDispls);
    std::copy(
        tempAlgParams.recvCounts.data(), tempAlgParams.recvCounts.data() + count,
        aivAlltoAllVArgs.extraArgs.recvCounts);
    std::copy(
        tempAlgParams.rdispls.data(), tempAlgParams.rdispls.data() + count, aivAlltoAllVArgs.extraArgs.recvDispls);

    u64 dataSize = tempAlgParams.sliceSize;
    CHK_RET(CalNumBlocks(aivAlltoAllVArgs.numBlocks, dataSize, param.numBlocksLimit));

    aivAlltoAllVArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivAlltoAllVArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivAlltoAllVArgs.repeatNum = tempAlgParams.repeatNum;
    aivAlltoAllVArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivAlltoAllVArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;
    // 单级算法无子通信域层级概念(AG两级为inter=1/intra=0), 固定0
    aivAlltoAllVArgs.subCommRankLevel = 0;
    if (opExecuteConfig_ == OpExecuteConfig::AIV_URMA) {
        aivAlltoAllVArgs.argsType = KernelArgsType::ARGS_TYPE_URMA;
    }

    CHK_RET(ExecuteKernelLaunch(aivAlltoAllVArgs));

    HCCL_INFO("[AivTempAlltoAllVMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
