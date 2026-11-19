/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_reduce_scatter_mesh_1D_meshchunk_concur.h"
#include "cost_model.h"
#include "channel.h"
#include "template_utils.h"

namespace ops_hccl {

namespace {
    // ---- CalcCostCoeff 标定常量与 A/B/C/D 分项计算 ----
    // phyLevel 信息缺失时的 CLOS 端口总数 fallback(与 ZAxisDetour 一致)
    constexpr int CLOS_PORT_TOTAL_FALLBACK = 8;
    // WriteReduce 逐字节净开销: 在途规约 DMA 读改写略慢于纯写, 真机 8p×2 标定实测为
    // 双路分流基准的 1.39 倍; 并发分流与消除串行 LocalReduce 的收益已由 B 项体现, 不重复计入
    constexpr float WRITE_REDUCE_NET_A_RATIO = 1.39f;
    // chunk 轮转 kernel 数: 真机 8p×2 小包固定开销标定反推
    constexpr int CHUNK_ROTATION_KERNEL_NUM = 232;

    // A 项: 双路分流取 max(对齐 ZAxisDetour 机制): mesh 直连与 CLOS 绕行并发, 分流比 = 运行态
    // 带宽权重, 瓶颈取慢路; CLOS 路 isPod 传 false(对齐 ZAxisDetour 非 sole 序列形态的约定):
    // CLOS 绕路路径不计 pod 端口减半惩罚, pod 场景由 A0(mesh 直连, 传 param.isPod)与净系数
    // 统一体现, 保证与老算法 A 项同一基准可比
    float CalcCostCoeffTermA(const CalcCostCoeffParam& param)
    {
        const u32 level0Bandwidth = param.rankSize - 1;
        CommTopo netType0 = CommTopo::COMM_TOPO_1DMESH;
        CommTopo netType1 = CommTopo::COMM_TOPO_CLOS;
        int portNum0 = param.portNum.empty() ? 0 : static_cast<int>(param.portNum[0]);
        int level1PortTotal = CLOS_PORT_TOTAL_FALLBACK;
        const bool hasPhyLevelInfo = param.phyLevelIdxs.size() >= 2 && param.phyLevelNetTypes.size() >= 2
                                     && param.phyLevelPortNums.size() >= 2;
        if (hasPhyLevelInfo) {
            netType0 = param.phyLevelNetTypes[0];
            netType1 = param.phyLevelNetTypes[1];
            portNum0 = param.phyLevelPortNums[0].empty() ? 0 : static_cast<int>(param.phyLevelPortNums[0][0]);
            level1PortTotal = 0;
            if (static_cast<u32>(param.phyLevelIdxs[1]) == 1) {
                for (u32 port : param.phyLevelPortNums[1]) {
                    level1PortTotal += static_cast<int>(port);
                }
            }
        }
        int portNum1 = level1PortTotal;
        float level0Ratio = (level1PortTotal > 0) ? static_cast<float>(level0Bandwidth)
                                                        / (level0Bandwidth + static_cast<u32>(level1PortTotal)) :
                                                    1.0f;
        float level1Ratio = 1.0f - level0Ratio;
        float nLevel0 = param.dataRatio * level0Ratio;
        float nLevel1 = param.dataRatio * level1Ratio;
        float A0 = 0.0f;
        float A1 = 0.0f;
        CostModelManager::Global()->CalcMeshParam(nLevel0, netType0, portNum0, param.rankSize, A0, param.isPod);
        float A = A0;
        if (portNum1 > 0) {
            CostModelManager::Global()->CalcMeshParam(nLevel1, netType1, portNum1, param.rankSize, A1, false);
            A = std::max(A0, A1);
        }
        return A * WRITE_REDUCE_NET_A_RATIO;
    }

    // B 项: 仅 PreCopy(write-reduce 在途规约消除串行 LocalReduce, 无 B2 项)
    float CalcCostCoeffTermB(const CalcCostCoeffParam& param)
    {
        float B = 0.0f;
        if (param.inputBuffer != param.scratchBuffer) {
            CostModelManager::Global()->CalcLocalCopyParams(param.dataRatio, EngineType::AICPU, B);
        }
        return B;
    }

    // C 项: chunk 轮转固定 kernel 数 × AICPU 单 kernel 时延
    float CalcCostCoeffTermC(const CalcCostCoeffParam& param)
    {
        (void)param; // C 项仅依赖标定常数, 与拓扑/数据量无关
        float C = 0.0f;
        CostModelManager::Global()->CalcLatencyParams(CHUNK_ROTATION_KERNEL_NUM, EngineType::AICPU, C);
        return C;
    }

    // D 项: chunk 轮转 task 数(n-1 步收发 + n-1 步同步, isPod ×3 否则 ×2)× AICPU 单 launch 开销
    float CalcCostCoeffTermD(const CalcCostCoeffParam& param)
    {
        int taskNum = CostModelManager::CalcTransTaskNum(param.rankSize) * (param.rankSize - 1)
                      + CostModelManager::CalcSyncTaskNum(param.rankSize) * (param.rankSize - 1);
        taskNum = param.isPod ? taskNum * 3 : taskNum * 2;
        float D = 0.0f;
        CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);
        return D;
    }
} // namespace

InsTempReduceScatterMesh1DMeshChunkConcur::InsTempReduceScatterMesh1DMeshChunkConcur(
    const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempReduceScatterMesh1DMeshChunk(param, rankId, subCommRanks)
{}

InsTempReduceScatterMesh1DMeshChunkConcur::~InsTempReduceScatterMesh1DMeshChunkConcur() {}

u64 InsTempReduceScatterMesh1DMeshChunkConcur::GetThreadNum() const
{
    // queIdx = frontRankShifted*C + channelIdx, 范围 [0, (R-1)*C-1](含主线 0), 恰需 (R-1)*C 个线程
    return templateRankSize_ > 1 ? (templateRankSize_ - 1) * channelsPerRank_ : 1;
}

HcclResult InsTempReduceScatterMesh1DMeshChunkConcur::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel0(comm, param, topoInfo, subCommRanks_, level0Channels));
    std::vector<HcclChannelDesc> level1Channels;
    CHK_RET(CalcChannelRequestMesh1DLevel1(comm, param, topoInfo, subCommRanks_, level1Channels));
    // 通道布局一致性校验(host 侧, 内核启动前): level1 链路允许缺失(linkRequired=false),
    // 各 peer 通道数可能不一致; 而 DoMeshChunk 的静态槽位(queIdx=peerOffset*C+ch)要求
    // 所有 peer 通道布局一致(不一致时轮转中途 E_PARA 或留空槽), 不一致拓扑直接干净拒绝。
    {
        std::map<u32, u32> l0Count;
        std::map<u32, u32> l1Count;
        for (const auto& desc : level0Channels) {
            l0Count[desc.remoteRank]++;
        }
        for (const auto& desc : level1Channels) {
            l1Count[desc.remoteRank]++;
        }
        for (const auto& entry : l0Count) {
            if (entry.second != MESH_CHANNELS_NUM) {
                HCCL_ERROR(
                    "[InsTempReduceScatterMesh1DMeshChunkConcur][CalcRes] rank[%u] has [%u] level0 channels, "
                    "expect [%u], inconsistent channel layout, not supported.",
                    entry.first, entry.second, MESH_CHANNELS_NUM);
                return HCCL_E_NOT_SUPPORT;
            }
        }
        if (!l1Count.empty()) {
            // 以全量 peer 为基准计数, 缺失按 0 计: 只互比 l1Count 内部会漏过混合布局(部分 0 条/部分 K 条);
            // 排除本 rank 自己(通道只建到其他 peer, self 的 l1Count 恒 0, 误比会拒掉正常拓扑)
            u32 expectedL1 = l1Count.begin()->second;
            for (const auto& peerRank : subCommRanks_[0]) {
                if (peerRank == myRank_) {
                    continue;
                }
                u32 actualL1 = 0;
                auto it = l1Count.find(peerRank);
                if (it != l1Count.end()) {
                    actualL1 = it->second;
                }
                if (actualL1 != expectedL1) {
                    HCCL_ERROR(
                        "[InsTempReduceScatterMesh1DMeshChunkConcur][CalcRes] rank[%u] has [%u] level1 channels, "
                        "expect [%u], inconsistent channel layout, not supported.",
                        peerRank, actualL1, expectedL1);
                    return HCCL_E_NOT_SUPPORT;
                }
            }
        }
    }
    std::vector<HcclChannelDesc> mergedChannels;
    mergedChannels.insert(mergedChannels.end(), level0Channels.begin(), level0Channels.end());
    mergedChannels.insert(mergedChannels.end(), level1Channels.begin(), level1Channels.end());
    resourceRequest.channels.push_back(mergedChannels);
    level0ChannelNumPerRank_ = level0Channels.empty() ? 0 : CalcChannelsPerRank(level0Channels);
    level1ChannelNumPerRank_ = level1Channels.empty() ? 0 : CalcChannelsPerRank(level1Channels);
    channelsPerRank_ = level0ChannelNumPerRank_ + level1ChannelNumPerRank_;
    // 极小数据下限守卫(host 侧, 内核启动前): 最小 peer 切片元素数(count/peer数) < 通道数时,
    // 任何通道切分必留空槽, 直接 NOT_SUPPORT 干净失败, 全 rank 同步。阈值随拓扑(通道数/
    // peer 数)与 dtype 自适应; 大数据路径(4K 对齐)下恒不触发。注: 资源缓存复用时本守卫
    // 不执行, 该场景由内核守卫兜底。
    if (templateRankSize_ > 1 && channelsPerRank_ > 0
        && param.DataDes.count / (templateRankSize_ - 1) < channelsPerRank_) {
        HCCL_ERROR(
            "[InsTempReduceScatterMesh1DMeshChunkConcur][CalcRes] unit[%llu] elements, min per-peer slice[%llu] < "
            "channel num[%u], data too small for concurrent split, not supported.",
            param.DataDes.count, param.DataDes.count / (templateRankSize_ - 1), channelsPerRank_);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_RET(GetRes(resourceRequest));
    HCCL_INFO(
        "[InsTempReduceScatterMesh1DMeshChunkConcur][CalcRes]myRank[%u], channelsPerRank_[%u], "
        "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], templateRankSize_[%u]",
        myRank_, channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, templateRankSize_);
    return HCCL_SUCCESS;
}

HcclResult
InsTempReduceScatterMesh1DMeshChunkConcur::SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    CHK_PRT_RET(
        channels.empty(),
        HCCL_ERROR("[InsTempReduceScatterMesh1DMeshChunkConcur][SetchannelsPerRank] channels is empty."),
        HCCL_E_INTERNAL);
    channelsPerRank_ = CalcChannelsPerRank(channels);
    level0ChannelNumPerRank_ = MESH_CHANNELS_NUM;
    level1ChannelNumPerRank_ = channelsPerRank_ - level0ChannelNumPerRank_;
    HCCL_INFO(
        "[InsTempReduceScatterMesh1DMeshChunkConcur][SetchannelsPerRank], channelsPerRank_[%u], "
        "level0ChannelNumPerRank_[%u], level1ChannelNumPerRank_[%u], templateRankSize_[%u]",
        channelsPerRank_, level0ChannelNumPerRank_, level1ChannelNumPerRank_, templateRankSize_);
    return HCCL_SUCCESS;
}

HcclResult InsTempReduceScatterMesh1DMeshChunkConcur::CalcDataSplitByPortGroup(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset)
{
    CHK_RET(CalcDataSplitByBandwidthZAxisDetour(
        totalDataCount, dataTypeSize, channels, elemCountOut, sizeOut, elemOffset, level0ChannelNumPerRank_,
        level1ChannelNumPerRank_, templateRankSize_));
    // 极小切片防退化: 加权取整可能把通道清零(如 per-peer 2 元素 × mesh 权重 → mesh 通道 0 个),
    // 0 尺寸通道该槽位全程零任务, 与静态线程/notify 槽位结构不兼容。有数据时从最大通道借元素
    // 保底每通道 ≥1; 元素不足以填满所有通道时返回 NOT_SUPPORT 干净失败。0 元素 peer 切片同理。
    if (totalDataCount == 0) {
        HCCL_ERROR("[InsTempReduceScatterMesh1DMeshChunkConcur][CalcDataSplitByPortGroup] per-peer slice is 0 "
                   "elements, data too small for concurrent split, not supported.");
        return HCCL_E_NOT_SUPPORT;
    }
    if (elemCountOut.size() > 1) {
        std::vector<u32> zeroIdx;
        for (u32 i = 0; i < elemCountOut.size(); i++) {
            if (elemCountOut[i] == 0) {
                zeroIdx.push_back(i);
            }
        }
        if (!zeroIdx.empty()) {
            if (totalDataCount < elemCountOut.size()) {
                HCCL_ERROR(
                    "[InsTempReduceScatterMesh1DMeshChunkConcur][CalcDataSplitByPortGroup] per-peer slice [%llu] "
                    "elements < channel num [%zu], data too small for concurrent split, not supported.",
                    totalDataCount, elemCountOut.size());
                return HCCL_E_NOT_SUPPORT;
            }
            for (u32 idx : zeroIdx) {
                u32 donor = 0;
                for (u32 j = 1; j < elemCountOut.size(); j++) {
                    if (elemCountOut[j] > elemCountOut[donor]) {
                        donor = j;
                    }
                }
                // 防御: 前置条件 total >= 通道数 下数学上不可达(和不变式保证必有 donor >= 2);
                // 上游分流函数若变更破坏不变式, 响亮报错优于静默留零通道
                if (elemCountOut[donor] <= 1) {
                    HCCL_ERROR(
                        "[InsTempReduceScatterMesh1DMeshChunkConcur][CalcDataSplitByPortGroup] borrow failed, "
                        "split sum invariant broken, totalDataCount[%llu], channelNum[%zu].",
                        totalDataCount, elemCountOut.size());
                    return HCCL_E_NOT_SUPPORT;
                }
                elemCountOut[donor] -= 1;
                elemCountOut[idx] = 1;
            }
            u64 offset = 0;
            for (u32 i = 0; i < elemCountOut.size(); i++) {
                sizeOut[i] = elemCountOut[i] * dataTypeSize;
                elemOffset[i] = offset;
                offset += sizeOut[i];
            }
        }
    }
    return HCCL_SUCCESS;
}

u64 InsTempReduceScatterMesh1DMeshChunkConcur::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    if (inBuffType == BufferType::INPUT && outBuffType == BufferType::HCCL_BUFFER) {
        return templateRankSize_;
    }
    // 其余(sole 等)组合: sole 场景 executor 传出的 stride 为 0, ScratchSlotBaseOff 的
    // per-algRank 槽位寻址退化为基址, 维持上游 n-1 scratch 契约(序列组合才需 n 槽)
    return templateRankSize_ - 1;
}

std::vector<CostModelParam> InsTempReduceScatterMesh1DMeshChunkConcur::CalcCostCoeff(CalcCostCoeffParam param)
{
    // rankSize 守卫: 超出机内 8p 形态无标定意义
    if (param.rankSize <= 1 || param.rankSize > 8) {
        return {};
    }
    const float A = CalcCostCoeffTermA(param);
    const float B = CalcCostCoeffTermB(param);
    const float C = CalcCostCoeffTermC(param);
    const float D = CalcCostCoeffTermD(param);
    std::vector<CostModelParam> params;
    params.push_back({A, B, C, D});
    return params;
}

} // namespace ops_hccl
