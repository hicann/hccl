/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "hccl_aiv_utils.h"
#include "aiv/aiv_temp_all_reduce_mesh_1D_twoshot.h"
#include "config_log.h"

namespace ops_hccl {
constexpr u32 TWO_PHASE_DATA_FACTOR = 2;

AivTempAllReduceMesh1DTwoShot::AivTempAllReduceMesh1DTwoShot(
    const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
    const std::vector<std::vector<u32>>& subCommRanks)
    : AivAlgTemplateBase(param, rankId, subCommRanks)
{}

AivTempAllReduceMesh1DTwoShot::~AivTempAllReduceMesh1DTwoShot() {}

std::vector<CostModelParam> AivTempAllReduceMesh1DTwoShot::CalcCostCoeff(CalcCostCoeffParam param)
{
    int portNum = param.portNum[0]; // 没有2+6
    int kernelNum = 2;
    // 第一步是ReduceScatter，
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    float B1 = 0.0f;
    float B2 = 0.0f;

    u32 level0RankSize = 0;
    if (param.topoInfo != nullptr) {
        level0RankSize = param.topoInfo->deviceNumPerModule;
    }
    bool isSymmetric
        = (param.topoInfo != nullptr && param.topoInfo->level0Symmetric && param.topoInfo->level1Symmetric);
    bool isMultiNode = (level0RankSize > 0 && level0RankSize < param.rankSize);
    if (isMultiNode && isSymmetric) {
        float A1 = 0.0f;
        int level0Port = 1;
        int level1Port = portNum;
        float A0 = 0.0f;
        // aiv预估与实测相差太大，这里先保持和pod计算一致
        CostModelManager::Global()->CalcMeshParam(
            TWO_PHASE_DATA_FACTOR * param.dataRatio, CommTopo::COMM_TOPO_1DMESH, level0Port, level0RankSize, A0, false);
        u32 level1RankSize = param.rankSize - level0RankSize;
        CostModelManager::Global()->CalcMeshParam(
            TWO_PHASE_DATA_FACTOR * param.dataRatio, CommTopo::COMM_TOPO_CLOS, level1Port, level1RankSize, A1, true);
        A = std::max(A0, A1);
    } else {
        CostModelManager::Global()->CalcMeshParam(
            TWO_PHASE_DATA_FACTOR * param.dataRatio, param.netType, portNum, param.rankSize, A, param.isPod);
    }

    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B1);
    } else {
        B1 = 0.0f;
    }
    CostModelManager::Global()->CalcLocalReduceParams(param.dataRatio * (param.rankSize - 1), EngineType::AICPU, B2);
    B = B1 + B2;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AIV, C);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f.", __func__, A, B, C, D);
    return params;
}

u64 AivTempAllReduceMesh1DTwoShot::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    // 少核路径 scratch = 镜像区 + 槽区，两段各约 dataSize。极小数据量+大通信规模(如 32B/32p)下
    // rankChunkStride 向上取整到 1 个元素，槽区可达 dataSize 数倍，4 倍贴边、加镜像区会溢出，故取 5 倍。
    // 对 loop 切分的影响：默认 HCCL_BUFFSIZE = 200MB 时 hcclBuff/5 = 40MB(4 倍时为 50MB)
    // 即为 maxDataSizePerLoop 的约束项(< host 侧 UB_MAX_DATA_SIZE 256MB)，
    // 因此 dataSize 落在 (40MB, 50MB] 时 loop 数会由 1 变 2。
    u32 multiplier = 5;
    return multiplier;
}

HcclResult AivTempAllReduceMesh1DTwoShot::CalcRes(
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
        HCCL_DEBUG("[AivTempAllReduceMesh1DTwoShot::CalcRes] Get Channel Success!");
    } else {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    }
    resourceRequest.channels.push_back(level0Channels);
    HCCL_WARNING("[AivTempAllReduceMesh1DTwoShot] Resource calculation is temporarily not performed in the template.");
    return HCCL_SUCCESS;
}

HcclResult AivTempAllReduceMesh1DTwoShot::CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit)
{
    (void)dataSize;
    if (numBlocksLimit >= (tempRankSize_ + 1)) {
        u32 coreNumPerRank = numBlocksLimit / (tempRankSize_ + 1);
        numBlocks = coreNumPerRank * (tempRankSize_ + 1);
    } else {
        // 如果要用更少的核心可以在这里折算，比如rankSize/2个核心
        numBlocks = numBlocksLimit;
    }
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[AivTempAllReduceMesh1DTwoShot] Actually use core num[%u], limit[%u]", numBlocks, numBlocksLimit);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAllReduceMesh1DTwoShot::CalNumBlocksWrite(u32& numBlocks, u32 numBlocksLimit)
{
    constexpr u32 TWO_RANK_SIZE = 2;
    constexpr u32 TWO_RANK_CORE_LIMIT = 24;
    constexpr u32 SMALL_RANK_SIZE = 4;
    constexpr u32 SMALL_RANK_CORE_LIMIT = 12;
    constexpr u32 MEDIUM_RANK_SIZE = 8;
    constexpr u32 MEDIUM_RANK_CORE_LIMIT = 24;
    constexpr u32 LARGE_RANK_CORE_LIMIT = 36;
    // 写语义按rank数分档控核：2卡单独24核、3~4卡12核、5~8卡24核、8卡以上36核，并受用户limit约束。
    u32 rankCoreLimit = LARGE_RANK_CORE_LIMIT;
    if (tempRankSize_ == TWO_RANK_SIZE) {
        rankCoreLimit = TWO_RANK_CORE_LIMIT;
    } else if (tempRankSize_ <= SMALL_RANK_SIZE) {
        rankCoreLimit = SMALL_RANK_CORE_LIMIT;
    } else if (tempRankSize_ <= MEDIUM_RANK_SIZE) {
        rankCoreLimit = MEDIUM_RANK_CORE_LIMIT;
    }
    numBlocks = std::min(numBlocksLimit, rankCoreLimit);
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[AivTempAllReduceMesh1DTwoShot] Write semantics use core num[%u], limit[%u], rank limit[%u]",
        numBlocks, numBlocksLimit, rankCoreLimit);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AivTempAllReduceMesh1DTwoShot::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource)
{
    HCCL_INFO("[AivTempAllReduceMesh1DTwoShot] KernelRun start");

    IncSliceId(); // 自动增长sliceId，传入sliceId
    dataType_ = param.DataDes.dataType;
    AivOpArgs aivAllReduceArgs;
    aivAllReduceArgs.cmdType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    // 核入口统一为aiv_allreduce_mesh1d_twoshot_*，读写语义由运行期isWrite分流，funcKey不随语义变化。
    aivAllReduceArgs.argsType = KernelArgsType::ARGS_TYPE_TWO_SHOT;
    FillCommonAivOpArgs(aivAllReduceArgs, param, tempAlgParams, templateResource);
    aivAllReduceArgs.isWrite = tempAlgParams.isWrite;

    if (tempAlgParams.isWrite) {
        CHK_RET(CalNumBlocksWrite(aivAllReduceArgs.numBlocks, param.numBlocksLimit));
    } else {
        CHK_RET(CalNumBlocks(aivAllReduceArgs.numBlocks, tempAlgParams.sliceSize, param.numBlocksLimit));
    }

    aivAllReduceArgs.inputSliceStride = tempAlgParams.inputSliceStride;
    aivAllReduceArgs.outputSliceStride = tempAlgParams.outputSliceStride;
    aivAllReduceArgs.repeatNum = tempAlgParams.repeatNum;
    aivAllReduceArgs.inputRepeatStride = tempAlgParams.inputRepeatStride;
    aivAllReduceArgs.outputRepeatStride = tempAlgParams.outputRepeatStride;

    CHK_RET(ExecuteKernelLaunch(aivAllReduceArgs));

    HCCL_INFO(
        "[AivTempAllReduceMesh1DTwoShot] KernelRun finished, isWrite[%d] numBlocks[%u]",
        static_cast<int>(aivAllReduceArgs.isWrite), aivAllReduceArgs.numBlocks);
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
