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
#include "aiv/aiv_temp_scatter_mesh_1D.h"
#include "cost_model.h"
#include "config_log.h"

namespace ops_hccl {

std::vector<CostModelParam> AivTempScatterMesh1D::CalcCostCoeff(CalcCostCoeffParam param)
{
    // Mesh 算法走 CLOS 时取 portNum[0]（单通道语义，不求和）；MESH 分支 portNum 不参与
    int portNum = static_cast<int>(param.portNum[0]);
    int remoteSyncNum = 1; // 单 kernel 下发
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    // 跨框(匹配层为 CLOS)时执行侧按目标路由：框内目标走框内 mesh 直连（每对端独享链路，
    // 并发时间=单份），跨框目标共享 CLOS 端口。从匹配层往下找框内 mesh 层，
    // 找到才启用双平面并发取 max 的口径；找不到（每框单卡无 mesh 层）或匹配层
    // 本身即 mesh（单级）时维持单平面公式
    u32 meshRankNum = 0;
    if (param.netType == CommTopo::COMM_TOPO_CLOS && param.topoInfo != nullptr) {
        for (const auto& level : param.topoInfo->physicalLevels) {
            if (level.topoType == CommTopo::COMM_TOPO_1DMESH && level.localRanks.size() >= 2
                && level.localRanks.size() < param.rankSize) {
                meshRankNum = std::max(meshRankNum, static_cast<u32>(level.localRanks.size()));
            }
        }
    }
    if (meshRankNum >= 2) {
        u32 meshTargetNum = meshRankNum - 1;              // 框内目标份数
        u32 closTargetNum = param.rankSize - meshRankNum; // 跨框目标份数
        float aMesh = 0.0f;
        float aClos = 0.0f;
        // 框内平面：mesh 直连每对端独享带宽，nMesh 份并发，时间=单份
        CostModelManager::Global()->CalcMeshParam(
            param.dataRatio, CommTopo::COMM_TOPO_1DMESH, portNum, meshRankNum, aMesh, false);
        // 跨框平面：closTargetNum 份共享 CLOS 端口
        CostModelManager::Global()->CalcMeshParam(
            param.dataRatio, CommTopo::COMM_TOPO_CLOS, portNum, closTargetNum + 1, aClos, false);
        A = std::max(aMesh, aClos);
    } else {
        CostModelManager::Global()->CalcMeshParam(param.dataRatio, param.netType, portNum, param.rankSize, A, false);
    }
    // in-kernel 处理，无独立 local copy 阶段；executor 通过 buffer 组合控制（INPUT→OUTPUT 时按 1 份计）
    if (param.inputBuffer != BufferType::HCCL_BUFFER && param.outputBuffer != BufferType::HCCL_BUFFER) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
    }
    CostModelManager::Global()->CalcLatencyParams(remoteSyncNum, EngineType::AIV, C);
    CostModelManager::Global()->CalcLaunchParams(
        CostModelManager::CalcTransTaskNum(param.rankSize), EngineType::AIV,
        D); // AIV 的 D 恒为 0（kernel launch 无展开）

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

AivTempScatterMesh1D::AivTempScatterMesh1D(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{}

AivTempScatterMesh1D::~AivTempScatterMesh1D() {}

HcclResult AivTempScatterMesh1D::CalcRes(
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
        HCCL_DEBUG("[AivTempScatterMesh1D::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempScatterMesh1D::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    if (numBlocksLimit == 0) {
        HCCL_ERROR("[AivTempScatterMesh1D] numBlocksLimit is 0");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    numBlocks = numBlocksLimit;
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[AivTempScatterMesh1D] Actually use core num[%u], limit[%u]", numBlocks, numBlocksLimit);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempScatterMesh1D::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempScatterMesh1D] KernelRun start");

    IncSliceId(); // 自动增长sliceId，传入sliceId
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivScatterArgs;
    aivScatterArgs.cmdType = HcclCMDType::HCCL_CMD_SCATTER;
    aivScatterArgs.input
        = tempAlgParams.buffInfo.inBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.inputPtr);
    aivScatterArgs.output
        = tempAlgParams.buffInfo.outBuffBaseOff + reinterpret_cast<u64>(tempAlgParams.buffInfo.outputPtr);
    aivScatterArgs.rank = u32(myRank_);
    aivScatterArgs.rankSize = tempRankSize_;
    aivScatterArgs.count = tempAlgParams.sliceSize / HCCL_SIZE_TABLE[dataType_];
    aivScatterArgs.dataType = dataType_;
    aivScatterArgs.op = param.reduceType;
    aivScatterArgs.root = root_;
    aivScatterArgs.sliceId = static_cast<uint32_t>(sliceId_);
    aivScatterArgs.buffersIn = templateResource.aivCommInfoPtr;
    aivScatterArgs.stream = param.stream;
    aivScatterArgs.isOpBase = (param.opMode == OpMode::OPBASE);

    u64 dataSize = tempAlgParams.sliceSize;
    CHK_RET(CalNumBlocks(aivScatterArgs.numBlocks, dataSize, param.numBlocksLimit));

    aivScatterArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivScatterArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivScatterArgs.repeatNum = tempAlgParams.repeatNum;
    aivScatterArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivScatterArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivScatterArgs));

    HCCL_INFO("[AivTempScatterMesh1D] KernelRun finished");
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
