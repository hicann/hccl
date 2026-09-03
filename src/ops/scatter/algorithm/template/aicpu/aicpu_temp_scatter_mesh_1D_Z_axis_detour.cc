/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_temp_scatter_mesh_1D_Z_axis_detour.h"
#include "cost_model.h"

namespace ops_hccl {
std::vector<CostModelParam> AicpuTempScatterMesh1DZAxisDetour::CalcCostCoeff(CalcCostCoeffParam param)
{
    // Z 轴绕行双层并发取 max（design.md §2.3 目标公式，对齐 RS 同族模板）：
    // 层0（server 内 MESH）传一半数据，读 executor 注入的 portNum[0]（MESH 段为 {R-1}）；
    // 层1（跨 server CLOS）传另一半，portNum 按多通道语义（单元素取 [0]，两元素 multichannel 使能时求和）
    constexpr float meshRatio = 0.5f;
    // 层0 MESH：取 portNum[0]（MESH 分支 A=n/bw，值不参与计算）；
    // 层1 CLOS：Z 轴绕行使能多 channel，portNum 求和（{6,2} → 8；框架内不做 POD 减半）
    int portNum0 = (param.portNum.size() >= 1) ? static_cast<int>(param.portNum[0]) : 1;
    int portNum1 = (param.portNum.size() == 1) ? static_cast<int>(param.portNum[0]) :
                                                 static_cast<int>(param.portNum[0] + param.portNum[1]);
    int kernelNum = 1; // 单次下发
    // taskNum（B 方案定案）：传输 task = 3/次（send 单边通信），两层分开计：
    //   MESH 分量：3 × (R-1)；CLOS 分量：3 × (R-1) × closMultiplier（双元素=pod 双 die → 3，单元素 → 2）
    // local copy task = 1/份（PreCopy 1 份 + PostCopy 1 份，与 B 系数份数同口径）
    int remotes = static_cast<int>(param.rankSize - 1);
    int closMultiplier = (param.portNum.size() >= 2) ? 3 : 2;
    int transTaskNum = 3 * remotes * (1 + closMultiplier);
    int localCopyCount = 0;
    if (param.inputBuffer != BufferType::HCCL_BUFFER) {
        localCopyCount += 1; // PreCopy
    }
    if (param.outputBuffer != BufferType::HCCL_BUFFER) {
        localCopyCount += 1; // PostCopy
    }
    int taskNum = transTaskNum + localCopyCount;
    float A0 = 0.0f;
    float A1 = 0.0f;
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;

    CostModelManager::Global()->CalcMeshParam(
        param.dataRatio * meshRatio, CommTopo::COMM_TOPO_1DMESH, portNum0, param.rankSize, A0, param.isPod);
    // CLOS 层不传 isPod（禁用 POD 减半）：Z 轴绕行的跨 die 数据在执行侧已按物理 channel 切分并行
    // （CalcDataSplitByPortGroupZAxisDetour 把 level1 数据均分到各 channel），带宽口径=各 channel 并行
    // 实际吞吐；POD 减半是普通 NHR"双 channel 当单逻辑口的 2:1 收敛折半"，Z 轴绕行不走该收敛
    CostModelManager::Global()->CalcMeshParam(
        param.dataRatio * (1.0f - meshRatio), CommTopo::COMM_TOPO_CLOS, portNum1, param.rankSize, A1, false);
    A = std::max(A0, A1);
    // 与 Mesh1D 同款数据流：PreCopy(root 1 slice) + PostCopy(非root 1 slice) 两次串行搬运，按 buffer 判据分段
    float preCopyB = 0.0f;
    float postCopyB = 0.0f;
    if (param.inputBuffer != BufferType::HCCL_BUFFER) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, preCopyB);
    }
    if (param.outputBuffer != BufferType::HCCL_BUFFER) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, postCopyB);
    }
    B = preCopyB + postCopyB;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

AicpuTempScatterMesh1DZAxisDetour::AicpuTempScatterMesh1DZAxisDetour(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempScatterMesh1D(param, rankId, subCommRanks)
{}

AicpuTempScatterMesh1DZAxisDetour::~AicpuTempScatterMesh1DZAxisDetour() {}

HcclResult AicpuTempScatterMesh1DZAxisDetour::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(
        topoInfo == nullptr, HCCL_ERROR("[AicpuTempScatterMesh1DZAxisDetour][CalcRes] topoInfo is nullptr"),
        HCCL_E_PARA);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));
    std::vector<HcclChannelDesc> mergedChannels;
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    level0ChannelNumPerRank_ = level0Channels.empty() ? 0 : CalcChannelsPerRank(level0Channels);
    level1ChannelNumPerRank_ = level1Channels.empty() ? 0 : CalcChannelsPerRank(level1Channels);
    channelsPerRank_ = level0ChannelNumPerRank_ + level1ChannelNumPerRank_;
    CHK_RET(GetRes(resourceRequest));
    HCCL_DEBUG(
        "[AicpuTempScatterMesh1DZAxisDetour][CalcRes] myRank[%u], channelsPerRank_[%u], "
        "level0ChannelNum[%zu], level1ChannelNum[%zu], notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, channelsPerRank_, level0Channels.size(), level1Channels.size(), resourceRequest.notifyNumOnMainThread,
        resourceRequest.slaveThreadNum);
    HCCL_INFO(
        "[AicpuTempScatterMesh1DZAxisDetour][CalcRes]myRank[%u], channelsPerRank_[%u], "
        "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], level0DataRatio_[%.2f]",
        myRank_, channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return HCCL_SUCCESS;
}

u64 AicpuTempScatterMesh1DZAxisDetour::GetThreadNum() const
{
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_) : 1;
    HCCL_INFO(
        "[AicpuTempScatterMesh1DZAxisDetour][GetThreadNum] templateRankSize_[%u] channelsPerRank_[%u] threadNum[%u]",
        templateRankSize_, channelsPerRank_, threadNum);
    return threadNum;
}

HcclResult AicpuTempScatterMesh1DZAxisDetour::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset)
{
    HCCL_INFO("[AicpuTempScatterMesh1DZAxisDetour][CalcDataSplitByPortGroup] Run Start");
    return CalcDataSplitByPortGroupZAxisDetour(
        totalDataCount, dataTypeSize, channels, elemCountOut, sizeOut, elemOffset, level0ChannelNumPerRank_,
        level1ChannelNumPerRank_, level0DataRatio_);
}

HcclResult
AicpuTempScatterMesh1DZAxisDetour::SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    CHK_PRT_RET(channels.empty(), HCCL_ERROR("[SetchannelsPerRank] channels is empty."), HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRank(channels);
    if (channelsPerRank_ > 1) {
        level0ChannelNumPerRank_ = MESH_CHANNELS_NUM;
        level1ChannelNumPerRank_ = channelsPerRank_ - level0ChannelNumPerRank_;
        level0DataRatio_ = 0.5f;
    }
    HCCL_INFO(
        "[AicpuTempScatterMesh1DZAxisDetour][SetchannelsPerRank], channelsPerRank_[%u], "
        "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], level0DataRatio_[%.2f]",
        channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, level0DataRatio_);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
