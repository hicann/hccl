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
#include "aiv/aiv_temp_reduce_mesh_1D.h"
#include "cost_model.h"

namespace ops_hccl {

std::vector<CostModelParam> AivTempReduceMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = (param.netType == CommTopo::COMM_TOPO_CLOS) ? 8 : 1;
    int kernelNum = 15;
    int taskNum = 5 * (param.rankSize - 1);
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    float B1 = 0.0f;
    float B2 = 0.0f;

    CostModelManager::Global()->CalcMeshParam(
        2 * param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B1);
    } else {
        B1 = 0.0f;
    }
    CostModelManager::Global()->CalcLocalReduceParams(param.dataRatio * (param.rankSize - 1), EngineType::AICPU, B2);
    B = B1 + B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AIV, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AIV, D);
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

AivTempReduceMesh1D::AivTempReduceMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{}

AivTempReduceMesh1D::~AivTempReduceMesh1D() {}

u64 AivTempReduceMesh1D::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // TwoShot场景部分数据量切分会有尾快，且需要确定性计算，需要2倍scratch才能保证数据不溢出
    u64 multiple = 2;
    HCCL_INFO("[AivTempReduceMesh1D] scratch multiple is [%llu]", multiple);
    return multiple;
}

HcclResult AivTempReduceMesh1D::CalcRes(
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
        HCCL_DEBUG("[AivTempReduceMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempReduceMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    HCCL_INFO("[AivTempReduceMesh1D] Limit core num[%u]", numBlocksLimit);
    numBlocks = numBlocksLimit;
    HCCL_INFO("[AivTempReduceMesh1D] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempReduceMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempReduceMesh1D] KernelRun start");

    IncSliceId(); // 自动增长sliceId，传入sliceId
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivReduceArgs;
    aivReduceArgs.cmdType = HcclCMDType::HCCL_CMD_REDUCE;
    aivReduceArgs.input = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivReduceArgs.output
        = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivReduceArgs.rank = u32(myRank_);
    aivReduceArgs.rankSize = tempRankSize_;
    aivReduceArgs.count = tempAlgParams.sliceSize / HCCL_SIZE_TABLE[dataType_];
    aivReduceArgs.dataType = dataType_;
    aivReduceArgs.op = param.reduceType;
    aivReduceArgs.root = root_;
    aivReduceArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivReduceArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivReduceArgs.stream = param.stream;
    aivReduceArgs.isOpBase = (param.opMode == OpMode::OPBASE);

    u64 dataSize = tempAlgParams.sliceSize;
    CHK_RET(CalNumBlocks(aivReduceArgs.numBlocks, dataSize, param.numBlocksLimit));

    aivReduceArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivReduceArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivReduceArgs.repeatNum = tempAlgParams.repeatNum;
    aivReduceArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivReduceArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivReduceArgs));

    HCCL_INFO("[AivTempReduceMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
