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
#include "aiv/aiv_temp_broadcast_mesh_1D.h"
#include "config_log.h"

namespace ops_hccl {

std::vector<CostModelParam> AivTempBroadcastMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    // twoshot mesh：CLOS下portNum=6，MESH下portNum=1（netType由executor根据isNhr/isMultiLevel确定后传入）
    int portNum = (param.netType == CommTopo::COMM_TOPO_CLOS) ? 6 : 1;
    int kernelNum = 10;
    int taskNum = 0; // AIV的D=0
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    // twoshot: n = dataRatio / rankSize * 2（scatter阶段每轮发D/R，allgather阶段每轮发D/R，共2D/R）
    CostModelManager::Global()->CalcMeshParam(
        param.dataRatio * 2 / param.rankSize, param.netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        // 本地拷贝1份全量数据（root拷入、非root拷出，平均1份）
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AIV, B);
    }
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AIV, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AIV, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

AivTempBroadcastMesh1D::AivTempBroadcastMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{}

AivTempBroadcastMesh1D::~AivTempBroadcastMesh1D() {}

HcclResult AivTempBroadcastMesh1D::CalcRes(
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
        for (auto channel : myChannelDescs) {
            if (channel.channelProtocol == COMM_PROTOCOL_UB_MEM) {
                level0Channels.push_back(channel);
            }
        }
        HCCL_DEBUG("[AivTempBroadcastMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempBroadcastMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    if (numBlocksLimit == 0) {
        HCCL_ERROR("[AivTempBroadcastMesh1D] numBlocksLimit is 0");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    // 基于实测结果，小于等于512K且小于等于16P，走控核算法性能更优，控rankSize核
    if (dataSize <= SMALL_SIZE_512KB && tempRankSize_ <= BR_CTRL_CORE_LIMIT_RANK_SIZE) {
        numBlocks = std::min(numBlocksLimit, tempRankSize_);
    } else {
        numBlocks = numBlocksLimit;
    }
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[AivTempBroadcastMesh1D] Actually use core num[%u], limit[%u]", numBlocks, numBlocksLimit);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempBroadcastMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempBroadcastMesh1D] KernelRun start");

    IncSliceId(); // 自动增长sliceId，传入sliceId
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivBroadcastArgs;
    aivBroadcastArgs.cmdType = HcclCMDType::HCCL_CMD_BROADCAST;
    aivBroadcastArgs.input
        = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivBroadcastArgs.output
        = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivBroadcastArgs.rank = u32(myRank_);
    aivBroadcastArgs.rankSize = tempRankSize_;
    aivBroadcastArgs.count = tempAlgParams.sliceSize / HCCL_SIZE_TABLE[dataType_];
    aivBroadcastArgs.dataType = dataType_;
    aivBroadcastArgs.op = param.reduceType;
    aivBroadcastArgs.root = root_;
    aivBroadcastArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivBroadcastArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivBroadcastArgs.stream = param.stream;
    aivBroadcastArgs.isOpBase = (param.opMode == OpMode::OPBASE);

    u64 dataSize = tempAlgParams.sliceSize;
    CHK_RET(CalNumBlocks(aivBroadcastArgs.numBlocks, dataSize, param.numBlocksLimit));

    aivBroadcastArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivBroadcastArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivBroadcastArgs.repeatNum = tempAlgParams.repeatNum;
    aivBroadcastArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivBroadcastArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivBroadcastArgs));

    HCCL_INFO("[AivTempBroadcastMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
