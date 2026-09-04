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
#include "aiv/aiv_temp_reduce_scatter_mesh_1D.h"
#include "config_log.h"
#include "cost_model.h"

namespace ops_hccl {

AivTempReduceScatterMesh1D::AivTempReduceScatterMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{}

AivTempReduceScatterMesh1D::~AivTempReduceScatterMesh1D() {}

std::vector<CostModelParam> AivTempReduceScatterMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    // int portNum = (param.portNum.size() == 1) ? param.portNum[0] : (param.portNum[0] + param.portNum[1]);
    int portNum = (param.netType == CommTopo::COMM_TOPO_CLOS) ? 8 : 1;
    int kernelNum = 1;
    int taskNum = 5 * (param.rankSize - 1);
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    float B1 = 0.0f;
    float B2 = 0.0f;
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B1);
    } else {
        B1 = 0.0f;
    }
    CostModelManager::Global()->CalcLocalReduceParams(param.dataRatio, EngineType::AICPU, B2);
    B = B1 + (param.rankSize - 1) * B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AIV, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AIV, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

u64 AivTempReduceScatterMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = 2 * tempRankSize_;
    return scratchMultiple;
}

HcclResult AivTempReduceScatterMesh1D::CalcRes(
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
        HCCL_DEBUG("[AivTempReduceScatterMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempReduceScatterMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    numBlocks = numBlocksLimit;
    if (dataSize < REDUCE_SCATTER_SMALL_COUNT_512KB) { // 小数据量，走原来极致低时延的流程
        constexpr uint32_t stepNum = 2;
        if (numBlocks > stepNum * tempRankSize_) {
            numBlocks = stepNum * tempRankSize_;
        }
    }
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[AivTempReduceScatterMesh1D] Actually use core num[%u], limit[%u]", numBlocks, numBlocksLimit);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempReduceScatterMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempReduceScatterMesh1D] KernelRun start");

    IncSliceId(); // 自动增长sliceId，传入sliceId
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivReduceScatterArgs;
    aivReduceScatterArgs.cmdType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    aivReduceScatterArgs.input
        = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivReduceScatterArgs.output
        = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivReduceScatterArgs.rank = u32(myRank_);
    aivReduceScatterArgs.rankSize = tempRankSize_;
    aivReduceScatterArgs.count = tempAlgParams.sliceSize / HCCL_SIZE_TABLE[dataType_];
    aivReduceScatterArgs.dataType = dataType_;
    aivReduceScatterArgs.op = param.reduceType;
    aivReduceScatterArgs.root = root_;
    aivReduceScatterArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivReduceScatterArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivReduceScatterArgs.stream = param.stream;
    aivReduceScatterArgs.isOpBase = (param.opMode == OpMode::OPBASE);

    u64 dataSize = tempAlgParams.inputSliceStride;
    CHK_RET(CalNumBlocks(aivReduceScatterArgs.numBlocks, dataSize, param.numBlocksLimit));

    aivReduceScatterArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivReduceScatterArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivReduceScatterArgs.repeatNum = tempAlgParams.repeatNum;
    aivReduceScatterArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivReduceScatterArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivReduceScatterArgs));

    HCCL_INFO("[AivTempReduceScatterMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
