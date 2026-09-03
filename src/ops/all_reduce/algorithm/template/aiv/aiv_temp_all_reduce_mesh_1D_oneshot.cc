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
#include "aiv/aiv_temp_all_reduce_mesh_1D_oneshot.h"

namespace ops_hccl {

AivTempAllReduceMesh1DOneShot::AivTempAllReduceMesh1DOneShot(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{}

AivTempAllReduceMesh1DOneShot::~AivTempAllReduceMesh1DOneShot() {}

std::vector<CostModelParam> AivTempAllReduceMesh1DOneShot::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = (param.portNum.size() == 1) ? param.portNum[0] : (param.portNum[0] + param.portNum[1]);
    int kernelNum = 5;
    int taskNum = 5 * (param.rankSize - 1);
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    // 同时有localreduce和localcopy，所以需要调用两个接口获取两个步骤的B并相加
    float B1 = 0.0f;
    float B2 = 0.0f;
    CostModelManager::Global()->CalcMeshParam(1, param.netType, portNum, param.rankSize, A, param.isPod);
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(1, EngineType::AICPU, B1);
    } else {
        B1 = 0.0f;
    }
    CostModelManager::Global()->CalcLocalReduceParams(param.rankSize - 1, EngineType::AICPU, B2);
    B = B1 + B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AIV, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AIV, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    return params;
}

u64 AivTempAllReduceMesh1DOneShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    u64 scratchMultiple = tempRankSize_;
    return scratchMultiple;
}

HcclResult AivTempAllReduceMesh1DOneShot::CalcRes(
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
        HCCL_DEBUG("[AivTempAllReduceMesh1DOneShot::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempAllReduceMesh1DOneShot::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    if (numBlocksLimit >= (tempRankSize_ + 1)) {
        numBlocks = tempRankSize_ + 1;
    } else {
        // 如果要用更少的核心可以在这里折算，比如rankSize/2个核心
        numBlocks = numBlocksLimit;
    }
    HCCL_INFO("[AivTempAllReduceMesh1DOneShot] Actually use core num[%u]", numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAllReduceMesh1DOneShot::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempAllReduceMesh1DOneShot] KernelRun start");

    IncSliceId(); // 自动增长sliceId，传入sliceId
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivAllReduceArgs;
    aivAllReduceArgs.cmdType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    aivAllReduceArgs.input
        = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivAllReduceArgs.output
        = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivAllReduceArgs.rank = u32(myRank_);
    aivAllReduceArgs.rankSize = tempRankSize_;
    aivAllReduceArgs.count = tempAlgParams.sliceSize / HCCL_SIZE_TABLE[dataType_];
    aivAllReduceArgs.dataType = dataType_;
    aivAllReduceArgs.op = param.reduceType;
    aivAllReduceArgs.root = root_;
    aivAllReduceArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivAllReduceArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivAllReduceArgs.stream = param.stream;
    aivAllReduceArgs.isOpBase = (param.opMode == OpMode::OPBASE);

    CHK_RET(CalNumBlocks(aivAllReduceArgs.numBlocks, tempAlgParams.sliceSize, param.numBlocksLimit));

    aivAllReduceArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivAllReduceArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivAllReduceArgs.repeatNum = tempAlgParams.repeatNum;
    aivAllReduceArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivAllReduceArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivAllReduceArgs));

    HCCL_INFO("[AivTempAllReduceMesh1DOneShot] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
