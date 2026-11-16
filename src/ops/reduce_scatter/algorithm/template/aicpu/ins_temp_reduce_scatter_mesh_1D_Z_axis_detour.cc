/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "cost_model.h"

namespace ops_hccl {

InsTempReduceScatterMesh1DZAxisDetour::InsTempReduceScatterMesh1DZAxisDetour(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempReduceScatterMesh1D(param, rankId, subCommRanks)
{}

InsTempReduceScatterMesh1DZAxisDetour::~InsTempReduceScatterMesh1DZAxisDetour() {}

std::vector<CostModelParam> InsTempReduceScatterMesh1DZAxisDetour::CalcCostCoeff(CalcCostCoeffParam param)
{
    if (param.rankSize <= 1 || param.rankSize > 8) {
        return {};
    }
    // rankSize is the local mesh group size, including self, even for multi-server executors.
    const u32 level0Bandwidth = param.rankSize - 1;
    CommTopo netType0 = CommTopo::COMM_TOPO_1DMESH;
    CommTopo netType1 = CommTopo::COMM_TOPO_CLOS;
    int portNum0 = param.portNum.empty() ? 0 : static_cast<int>(param.portNum[0]);
    bool isSoleMeshConcur
        = (param.algName != nullptr && (strcmp(param.algName, "AicpuReduceScatterSoleMeshConcur") == 0));
    bool isPod = isSoleMeshConcur ? param.isPod : false;
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
    int portNum1 = level1PortTotal;
    // 分流分母 = level1 端口Σ(与运行态 totalWeight 一致); 0 时全量走 level0
    float level0Ratio = (level1PortTotal > 0) ? static_cast<float>(level0Bandwidth)
                                                    / (level0Bandwidth + static_cast<u32>(level1PortTotal)) :
                                                1.0f;
    float level1Ratio = 1.0f - level0Ratio;
    float nLevel0 = param.dataRatio * level0Ratio;
    float nLevel1 = param.dataRatio * level1Ratio;

    // A: 两级跨片传输代价取最大值（level0 和 level1 并行传输）
    // level0: server 内 mesh 组网，level1: 跨 server clos 组网
    int kernelNum = 27;
    // pod 先乘3,后续需要考虑server
    int taskNum
        = (CostModelManager::CalcTransTaskNum(param.rankSize) + CostModelManager::CalcSyncTaskNum(param.rankSize) * 2);
    taskNum = param.isPod ? taskNum * 3 : taskNum * 2;
    float A0 = 0.0f;
    float A1 = 0.0f;
    CostModelManager::Global()->CalcMeshParam(nLevel0, netType0, portNum0, param.rankSize, A0, param.isPod);
    // portNum1<=0(纯 fullmesh)时跳过 A1, 避免 0/0=NaN
    float A = A0;
    if (portNum1 > 0) {
        CostModelManager::Global()->CalcMeshParam(nLevel1, netType1, portNum1, param.rankSize, A1, isPod);
        A = std::max(A0, A1);
    }

    // B: 本地操作，两级合计处理完整数据，reduce (rankSize-1) 份
    float B1 = 0.0f;
    float B2 = 0.0f;
    if (param.inputBuffer != param.scratchBuffer) {
        CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B1);
    }
    CostModelManager::Global()->CalcLocalReduceParams(param.dataRatio, EngineType::AICPU, B2);
    float B = B1 + (param.rankSize - 1) * B2;

    float C = 0.0f;
    float D = 0.0f;
    CostModelManager::Global()->CalcLatencyParams(kernelNum, EngineType::AICPU, C);
    CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);

    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    HCCL_DEBUG("[%s] CalcCostCoeff A=%f B=%f C=%f D=%f (level0Ratio=%f).", __func__, A, B, C, D, level0Ratio);
    return params;
}

HcclResult InsTempReduceScatterMesh1DZAxisDetour::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    CHK_PRT_RET(
        topoInfo == nullptr, HCCL_ERROR("[InsTempReduceScatterMesh1DZAxisDetour][CalcRes] topoInfo is nullptr"),
        HCCL_E_PARA);
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));
    // host 平面通道（localEndpoint.loc.locType != DEVICE，
    // 如全 HOST hostdpu 形态下 netLayer1 的 host RoCE 对端）由 host 侧 COMM_ENGINE_CPU 建链
    // （HostCpuRoceChannel，op_common.cc HcclGetChannel 分流），不随 AICPU H2D 资源打包、
    // 不在设备侧重建 transport，其句柄对 channel notify/BatchTransfer 原语不可消费——
    // 执行期第 2 个 (对端, 通道) 的首个 notify 原语即失效（无显式报错的死等/进程失效）。
    // L0 双平面按设备侧可用性过滤：仅保留 DEVICE 端点的框间通道
    // （与 host 建链分流同判据：locType=DEVICE ⟺ AICPU_TS 建链 ⟺ 设备侧可执行）。
    // level1 全滤后 level1ChannelNumPerRank_=0，z 轴绕路按既有退化语义全量走框内
    // （CalcDataSplitByBandwidthZAxisDetour 的 level1==0 分支）；
    // 全 HOST hostdpu 形态的框间数据流由二级 L1 Dpu inter 模板（DPU 搬运协议）承担，语义自洽。
    level1Channels.erase(
        std::remove_if(
            level1Channels.begin(), level1Channels.end(),
            [](const HcclChannelDesc& desc) {
                return desc.localEndpoint.loc.locType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
            }),
        level1Channels.end());
    std::vector<HcclChannelDesc> mergedChannels;
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    level0ChannelNumPerRank_ = level0Channels.empty() ? 0 : CalcChannelsPerRank(level0Channels);
    level1ChannelNumPerRank_ = level1Channels.empty() ? 0 : CalcChannelsPerRank(level1Channels);
    channelsPerRank_ = level0ChannelNumPerRank_ + level1ChannelNumPerRank_;
    CHK_RET(GetRes(resourceRequest));
    HCCL_DEBUG(
        "[InsTempReduceScatterMesh1DZAxisDetour][CalcRes] myRank[%u], channelsPerRank_[%u], "
        "level0ChannelNum[%zu], level1ChannelNum[%zu], notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, channelsPerRank_, level0Channels.size(), level1Channels.size(), resourceRequest.notifyNumOnMainThread,
        resourceRequest.slaveThreadNum);
    HCCL_INFO(
        "[InsTempReduceScatterMesh1DZAxisDetour][CalcRes]myRank[%u], channelsPerRank_[%u], "
        "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], templateRankSize_[%u]",
        myRank_, channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, templateRankSize_);
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DZAxisDetour::GetThreadNum() const
{
    u32 threadNum = templateRankSize_ > 1 ? ((templateRankSize_ - 1) * channelsPerRank_ + 1) : 1;
    HCCL_INFO(
        "[InsTempReduceScatterMesh1DZAxisDetour][GetThreadNum] templateRankSize_[%u] channelsPerRank_[%u] "
        "threadNum[%u]",
        templateRankSize_, channelsPerRank_, threadNum);
    return threadNum;
}

HcclResult InsTempReduceScatterMesh1DZAxisDetour::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset)
{
    HCCL_INFO("[InsTempReduceScatterMesh1DZAxisDetour][CalcDataSplitByPortGroup] Run Start");
    return CalcDataSplitByBandwidthZAxisDetour(
        totalDataCount, dataTypeSize, channels, elemCountOut, sizeOut, elemOffset, level0ChannelNumPerRank_,
        level1ChannelNumPerRank_, templateRankSize_);
}

HcclResult
InsTempReduceScatterMesh1DZAxisDetour::SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    CHK_PRT_RET(
        channels.empty(), HCCL_ERROR("[InsTempReduceScatterMesh1DZAxisDetour][SetchannelsPerRank] channels is empty."),
        HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRank(channels);
    level0ChannelNumPerRank_ = MESH_CHANNELS_NUM;
    // 无符号下溢守卫：level0ChannelNumPerRank_ 固定为 MESH_CHANNELS_NUM，减法
    // channelsPerRank_ - level0ChannelNumPerRank_ 在 channelsPerRank_ < level0ChannelNumPerRank_ 时会
    // 静默下溢为巨值 level1ChannelNumPerRank_，后续带宽切分按错误通道数寻址（现 CalcChannelsPerRank 映射
    // 重载下限为 1 且 MESH_CHANNELS_NUM=1，该分支实际不可达，属对未来变更的防御性守卫）。按失败处理，
    // 对齐本函数既有 CHK_PRT_RET（channels.empty 守卫）错误返回风格。
    CHK_PRT_RET(
        channelsPerRank_ < level0ChannelNumPerRank_,
        HCCL_ERROR(
            "[InsTempReduceScatterMesh1DZAxisDetour][SetchannelsPerRank] channelsPerRank_[%u] < "
            "level0ChannelNumPerRank_[%u], level1 channel count underflow.",
            channelsPerRank_, level0ChannelNumPerRank_),
        HCCL_E_INTERNAL);
    level1ChannelNumPerRank_ = channelsPerRank_ - level0ChannelNumPerRank_;
    HCCL_INFO(
        "[InsTempReduceScatterMesh1DZAxisDetour][SetchannelsPerRank], channelsPerRank_[%u], "
        "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], templateRankSize_[%u]",
        channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, templateRankSize_);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
