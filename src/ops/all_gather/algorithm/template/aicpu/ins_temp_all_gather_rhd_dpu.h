/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_RHD_DPU_H
#define INS_TEMP_ALL_GATHER_RHD_DPU_H

#pragma once

#include "dpu_alg_nhr_opt_wrapper.h"
#include "ins_temp_all_gather_rhd.h"

namespace ops_hccl {

// all_gather RHD 递归折半倍增的 host-DPU 版：AICPU 侧装载/卸载与协议交互，网络三相位
// （PRE 合并、MAIN 倍增、POST 回填）经共享内存卸载到 DPU 逐 run 直写对端 CCL 网格。
// 公有继承 InsTempAllGatherRHD 复用其全部同源数学（upstream 脚手架步骤表/σ 交错 owner 槽位
// 公式/装载与卸载），仅覆写资源面与执行面：
//   - CalcRes：CalcChannelRequestMesh1DHighestHostRoce 全互连申请（对域内全部 rank 申请最高
//     HOST+ROCE 层通道——全互连 ⊇ RHD 每步对端集，PRE/POST 配对端 t↔t+r 与 MAIN 对端
//     phys(v⊕2^k) 均为域内 rank）；
//   - 资源面：单线程、无从流、无主从 notify（DPU 数据面同步 = DpuBatchTransfer 三阶段
//     STEP_SYNC/DATA_SIGNAL）；
//   - KernelRun：AICPU 侧 LocalDataCopy → eager 切换 → HcommSendRequest 卸载 DPURunInfo →
//     HcommWaitResponse → HcommBatchModeStart → PostLocalCopy；enableRemoteMemAccess=true
//     防御性拒绝（remote 直写分支不接入 DPU 路径）；
//   - DPUKernelRun：按域重建 RHD 步骤表后执行三相位循环（skipPrePhase 与基类 RunAllGatherRHD
//     镜像门控），每步单对端 channels.at(peer)[0]、按步骤表展开 tx/rx 区段为 owner 槽位 run 表
//     （公式与基类 BuildStepSlice 逐字节同源），每步一次 DpuBatchTransfer。
// 绑定点：二级 DpuAllGatherSequenceMeshRHD（L1 inter）与三级 DpuAllGatherSequenceMeshConcurRHDRHD
// （L2 host）共用本 template。
class InsTempAllGatherRHDDPU : public InsTempAllGatherRHD {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::RHD};
    InsTempAllGatherRHDDPU() = default;
    explicit InsTempAllGatherRHDDPU(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllGatherRHDDPU() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather RHD (3-phase, host-DPU offload: AICPU side load/unload and "
                           "protocol interaction, network PRE/MAIN/POST offloaded to DPU with per-run direct "
                           "write to peer CCL grid) with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    u64 GetThreadNum() const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult DPUKernelRun(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32 myRank, const std::vector<std::vector<uint32_t>>& subCommRanks) override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub) override {}
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override {}

private:
    // AICPU 侧装载（数据面恒单 channel [0] 整份，无多通道切分；两分支=CCL 别名跨步槽位/紧凑中转，
    // 公式与基类 LocalDataCopy 同源、NCH=1 退化）
    HcclResult LocalDataCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    // AICPU 侧卸载（CCL 别名跳过；真拷贝恒等回搬，公式与基类 PostLocalCopy 同源、NCH=1 退化）
    HcclResult PostLocalCopy(const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    // DPU 侧三相位循环：按域重建 part1Info_/mainSteps_/preStep_/postStep_（镜像基类
    // PrepareKernelRun），rpt → PRE（skipPrePhase 镜像门控）→ MAIN → POST；收尾汇总打印本 rank
    // 发送 WQE 总数（局部变量+RunStepDpu 出参传递，不加实例成员——DPU 侧无实例级状态残留）
    HcclResult RunAllGatherRHDDpu(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels);
    // DPU 侧单步：对端换算（MAIN=虚拟序映射，PRE/POST=物理配对端）→ channels.at(peer)[0] →
    // 步骤表区段展开为 DpuTransferCtx → DpuBatchTransfer（空步骤全跳过）。wqeCnt 出参=本步
    // 发送 WQE 数（=txSrcSlices 条数，每 run 恰 1 条 DataSlice→1 条 WQE；对齐基类 RunStepRHD
    // 的 wqeCntTotal 出参签名模式；空步骤/空 tx 段计 0——函数入口清零，全路径定值）
    HcclResult RunStepDpu(
        const RhdStepInfo& step, const u64 rpt, const std::map<u32, std::vector<ChannelInfo>>& channels, u32& wqeCnt);
    // 步骤表 tx/rx 区段 → owner 槽位 run 表 → DataSlice 列表（run 划分公式与基类
    // BuildStepSlice 的 buildRuns 逐字节同源；NCH=1 无多通道切分，每 run 恰 1 条 slice）
    HcclResult
    BuildStepSlicesDpu(const ChannelInfo& peerChannel, const RhdStepInfo& step, const u64 rpt, DpuTransferCtx& ctx);
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_GATHER_RHD_DPU_H
