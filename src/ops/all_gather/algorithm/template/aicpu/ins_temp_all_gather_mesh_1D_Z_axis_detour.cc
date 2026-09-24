/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "cost_model.h"

namespace ops_hccl {
constexpr u32 MAX_RANK_SIZE_FOR_MESH_1D = 8;
constexpr int POD_TASK_NUM_MULTIPLE = 3;
constexpr int TASK_NUM_MULTIPLE = 2;

InsTempAllGatherMesh1D1DZAxisDetour::InsTempAllGatherMesh1D1DZAxisDetour(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempAllGatherMesh1D(param, rankId, subCommRanks)
{}
InsTempAllGatherMesh1D1DZAxisDetour::~InsTempAllGatherMesh1D1DZAxisDetour() {}

std::vector<CostModelParam> InsTempAllGatherMesh1D1DZAxisDetour::CalcCostCoeff(CalcCostCoeffParam param)
{
    if (param.rankSize <= 1 || param.rankSize > MAX_RANK_SIZE_FOR_MESH_1D) {
        return {};
    }
    // rankSize is the local mesh group size, including self, even for multi-server executors.
    const u32 level0Bandwidth = param.rankSize - 1;
    CommTopo netType0 = CommTopo::COMM_TOPO_1DMESH;
    CommTopo netType1 = CommTopo::COMM_TOPO_CLOS;
    int portNum0 = param.portNum.empty() ? 0 : static_cast<int>(param.portNum[0]);
    // level1 端口总数(Σ); 列表为空走默认 8(旧行为)
    int level1PortTotal = 8;
    const bool hasPhyLevelInfo
        = param.phyLevelIdxs.size() >= 2 && param.phyLevelNetTypes.size() >= 2 && param.phyLevelPortNums.size() >= 2;
    if (hasPhyLevelInfo) {
        netType0 = param.phyLevelNetTypes[0];
        netType1 = param.phyLevelNetTypes[1];
        portNum0 = param.phyLevelPortNums[0].empty() ? 0 : static_cast<int>(param.phyLevelPortNums[0][0]);
        level1PortTotal = 0;
        // 上层须为物理层1, 否则按 0 回退纯 fullmesh
        if (static_cast<u32>(param.phyLevelIdxs[1]) == 1) {
            for (u32 port : param.phyLevelPortNums[1]) {
                level1PortTotal += static_cast<int>(port);
            }
        }
    }
    if (level1PortTotal == 1) {
        level1PortTotal = 0;
    }
    int portNum1 = level1PortTotal;
    if (param.algName != nullptr && strcmp(param.algName, "AicpuAllGatherSoleMeshConcur") == 0) {
        portNum1 /= 2;
    }
    // 分流分母 = level1 端口Σ(与运行态 totalWeight 一致); 0 时全量走 level0
    float level0Ratio = (level1PortTotal > 0) ? static_cast<float>(level0Bandwidth)
                                                    / (level0Bandwidth + static_cast<u32>(level1PortTotal)) :
                                                1.0f;
    float nLevel0 = param.dataRatio * level0Ratio;
    float nLevel1 = param.dataRatio * (1.0f - level0Ratio);

    int kernelNum = (param.isPod ? 18 : 32);
    int taskNum
        = CostModelManager::CalcTransTaskNum(param.rankSize) + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2;
    taskNum = param.isPod ? taskNum * 3 : taskNum * 2;
    float A0 = 0.0f;
    float A1 = 0.0f;
    CostModelManager::Global()->CalcMeshParam(nLevel0, netType0, portNum0, param.rankSize, A0, param.isPod);
    // portNum1<=0(纯 fullmesh)时跳过 A1, 避免 0/0=NaN
    float A = A0;
    if (portNum1 > 0) {
        CostModelManager::Global()->CalcMeshParam(nLevel1, netType1, portNum1, param.rankSize, A1, false);
        A = std::max(A0, A1);
    }

    // B: 本地拷贝，两级合计处理完整数据
    float B = 0.0f;
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
    }
    if (param.inputBuffer != param.outputBuffer) {
        float B2 = 0.0f;
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B2);
        B += B2;
    }

    float C = 0.0f;
    float D = 0.0f;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    D = std::max(94e-6f, 0.5e-6f * taskNum);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG(
        "[%s] CalcCostCoeff A0=%f A1=%f A=%f B=%f C=%f D=%f (level0Ratio=%f).", __func__, A0, A1, A, B, C, D,
        level0Ratio);
    return params;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    HCCL_INFO("[InsTempAllGatherMesh1D1DZAxisDetour][CalcRes] start");
    CHK_PRT_RET(
        topoInfo == nullptr, HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour][CalcRes] topoInfo is nullptr"),
        HCCL_E_PARA);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));
    level0ChannelNumPerRank_ = level0Channels.empty() ? 0 : CalcChannelsPerRank(level0Channels);
    level1ChannelNumPerRank_ = level1Channels.empty() ? 0 : CalcChannelsPerRank(level1Channels);
    channelsPerRank_ = level0ChannelNumPerRank_ + level1ChannelNumPerRank_;
    std::vector<HcclChannelDesc> mergedChannels;
    HCCL_INFO("level0Channels[%d]level1Channels[%d]\n", level0Channels.size(), level1Channels.size());
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    HCCL_INFO("mergedChannels[%d]\n", mergedChannels.size());

    if (subCommRanks_.size() <= COMM_LEVEL0) {
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(
        channelsPerRank_ == 0, HCCL_ERROR("[InsTempAllGatherMesh1D1DZAxisDetour][CalcRes] channelsPerRank_ is 0"),
        HCCL_E_INTERNAL);
    CHK_RET(GetRes(resourceRequest));
    return HCCL_SUCCESS;
}

u64 InsTempAllGatherMesh1D1DZAxisDetour::GetThreadNum() const
{
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_) : 1;
    HCCL_INFO(
        "[InsTempAllGatherMesh1D1DZAxisDetour][GetThreadNum] templateRankSize_[%u] channelsPerRank_[%u] threadNum[%u]",
        templateRankSize_, channelsPerRank_, threadNum);
    return threadNum;
}

HcclResult InsTempAllGatherMesh1D1DZAxisDetour::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset)
{
    HCCL_INFO(
        "[InsTempAllGatherMesh1D1DZAxisDetour][CalcDataSplitByPortGroup] Run Start[%u][%u][%u]\n",
        level0ChannelNumPerRank_, level1ChannelNumPerRank_, templateRankSize_);
    // Thread resources use the maximum channel count; split data by this peer's actual channels.
    const u32 level1ChannelNum
        = channels.size() > level0ChannelNumPerRank_ ? static_cast<u32>(channels.size()) - level0ChannelNumPerRank_ : 0;
    return CalcDataSplitByBandwidthZAxisDetour(
        totalDataCount, dataTypeSize, channels, elemCountOut, sizeOut, elemOffset, level0ChannelNumPerRank_,
        level1ChannelNum, templateRankSize_);
}

HcclResult
InsTempAllGatherMesh1D1DZAxisDetour::SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRank(channels);
    level0ChannelNumPerRank_ = MESH_CHANNELS_NUM;
    level1ChannelNumPerRank_ = channelsPerRank_ - level0ChannelNumPerRank_;
    HCCL_INFO(
        "[InsTempAllGatherMesh1D1DZAxisDetour][SetchannelsPerRank], channelsPerRank_[%u], "
        "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], templateRankSize_[%u]",
        channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, templateRankSize_);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
