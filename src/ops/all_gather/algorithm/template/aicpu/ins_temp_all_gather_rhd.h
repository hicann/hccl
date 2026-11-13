/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_RHD_H
#define INS_TEMP_ALL_GATHER_RHD_H

#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

// RHD（Recursive Halving-Doubling）递归折半倍增模板：消费 upstream RHD 公共脚手架
// （template_utils RHD 函数族，PR 3854）。配对 (t ↔ t+r)（r = p − 2^⌊log₂p⌋），AllGather 无
// root（恒传 INVALID_VALUE_RANKID → 代表恒低位 t）；σ 交错布局（RhdUnitStartOfRank：
// algRank t→unit 2t、t+r→unit 2t+1、block a→a）。三相位：PRE（被合并方发自身份额段、代表接收）
// → 主体 ⌊log₂P⌋ 步倍增（仅有效 rank：!inPart1 || isRepresentative，被合并方不参与）→
// POST（代表发全量、被合并方接收）；AG 族取 (SHARE, FULL)，2^ 幂时 PRE/POST 输出空步骤。
// σ 感知 OUT 写映射（方案乙=owner 槽位 staging）：方向 B 的"直写 OUT 网格+整段单 WQE"
// 在 σ 交错布局下结构性不可行——连续 σ unit run 的 OUT 目标在 part1 区交错不邻接
// （unit 2t→槽 t、unit 2t+1→槽 t+r）。本模板 staging 恒为 rank 槽位网格：网络区段
// （σ unit 空间）逐 unit 反查 owner=σ⁻¹(u) 落 owner 槽位（收发双方同映射同偏移），
// run 按"owner 槽位字节邻接"合并——part1 区逐 unit 断 run；block 区（含 2r−1/2r 边界）
// 紧凑网格（stride==S）整段合并单 WQE、稀疏网格（三级 L2 OSS=S·d1 等）逐 unit run，
// 末 run 端点恒在 loop 预算内（前任务 F2 经验继承）。CCL 别名/remote 绑定下网络直写
// 最终 OUT 网格（零后拷贝）；真拷贝绑定紧凑 CCL 中转+恒等回搬。r=0（2^ 幂）σ 恒等
// 退化为整段合并单 WQE（先行 HD 已验证路径形态），结构同源于仓内 halving-doubling 模板族。
class InsTempAllGatherRHD : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::RHD};
    InsTempAllGatherRHD() = default;
    explicit InsTempAllGatherRHD(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempAllGatherRHD() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather recursive-halving-doubling (3-phase, upstream scaffold pairing "
                           "t<->t+r with sigma-interleaved unit space, owner-slot staging with per-owner run "
                           "splitting — single WQE per contiguous owner run, per-unit runs on interleaved part1 / "
                           "strided grids) with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;

    HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>& channels) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;
    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

protected:
    u32 GetRankFromMap(const u32 algRankIdx) const;
    TemplateDataParams tempAlgParams_;
    bool isDmaRead_{false};
    // CCL 别名绑定判定（outBuffType==HCCL_BUFFER，nhr.cc:170 先例）：网络直写 OUT 网格、后拷贝跳过
    bool isOutputCclBuff_{false};
    // 以下成员/方法由 DPU 卸载版 InsTempAllGatherRHDDPU 复用（private→protected 提升，零语义变更）：
    // 槽位/owner 反查闭式与域步骤表是同源数学的共享面，DPU 版 KernelRun 侧重建、DPUKernelRun 侧消费
    // owner 槽位落点（三情形，参数为 algRank——σ 反查后的 owner，非 σ unit）：
    //   情形 A（isOutputCclBuff_）：CCL 别名绑定 OUT≡SCR 同缓冲同网格，网络直写输出网格
    //     = hcclBuffBaseOff + owner·OSS + rpt·ORS（二级 inter/三级 L1 紧凑、三级 L2 跨步 OSS=S·d1
    //     同式；nhr.cc:249-252/488-490 同款）；
    //   情形 B（真拷贝绑定）：紧凑 CCL 中转 = hcclBuffBaseOff + owner·S + rpt·S·N；
    //   remote（enableRemoteMemAccess_，symMem/图模式）：OUT 用户缓冲 =
    //     outBuffBaseOff + owner·OSS + rpt·ORS
    u64 RhdSlotOffset(const u64 owner, const u64 rpt) const;
    // σ⁻¹（RhdUnitStartOfRank 反查，upstream 脚手架未提供的闭式）：unit u → 持有者 algRank。
    //   u ≥ 2r → u；u < 2r 偶 → u/2（代表 t）；u < 2r 奇 → u/2 + r（被合并方 t+r）
    u32 RhdOwnerOfUnit(const u32 unit) const;
    // run 划分闭式（提取自基类 BuildStepSlice 与 DPU 版 BuildStepSlicesDpu 两处同源 lambda
    // unitLenOf/buildRuns，PR#3933 评审消除双份维护的静默漂移风险）：
    //   段（σ unit 区间 [segStart, segEnd)）→ 逐 unit 反查 owner=σ⁻¹(u) → owner 槽位落点
    //   run(u) = [RhdSlotOffset(owner(u), rpt), unitLen(u)) → 字节邻接合并出 run 表；
    //   依赖 tempAlgParams_（sliceSize/tailSize）、scopeRankSize_（tailUnit）、part1Info_（σ 反查），
    //   DPU 卸载版复用的前置条件由 RunAllGatherRHDDpu 域重建序保证（tempAlgParams_ 先行赋值、
    //   scopeRankSize_/part1Info_ 先于 BuildStepSlicesDpu 重建）
    void
    RhdBuildRuns(const u32 segStart, const u32 segEnd, const u64 rpt, std::vector<std::pair<u64, u64>>& runs) const;
    // RHD 域信息与步骤表（upstream 脚手架类型；每次 KernelRun 按当前域重算，无实例级状态残留）
    RhdPart1Info part1Info_{};
    std::vector<RhdStepInfo> mainSteps_;
    RhdStepInfo preStep_{};
    RhdStepInfo postStep_{};
    // 算法作用域物理规模 p（upstream RhdPart1Info 无 rankSize 字段；upstream virtualRank 对
    // 被合并方恒有效，主体参与性判据用 isRepresentative，见 PrepareKernelRun）
    u32 scopeRankSize_{0};

private:
    HcclResult CheckKernelRunParams(const TemplateResource& templateResource);
    HcclResult PrepareKernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, const TemplateResource& templateResource);
    HcclResult SyncInterThreads(const std::vector<ThreadHandle>& threads, bool isPreSync);
    HcclResult RunChannelMainLoop(const TemplateResource& templateResource);
    HcclResult PrepareDataSplitForMultiChannel(const TemplateResource& templateResource);
    HcclResult LocalDataCopy(const std::vector<ThreadHandle>& threads, const u32& channelIdx);
    HcclResult PostLocalCopy(const ThreadHandle& thread, const u32& channelIdx);
    HcclResult BuildStepSlice(
        const ChannelInfo& peerChannel, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const RhdStepInfo& step, const u64 rpt, const u32& channelIdx, std::vector<DataSlice>& txSrcSlices,
        std::vector<DataSlice>& txDstSlices, std::vector<DataSlice>& rxSrcSlices, std::vector<DataSlice>& rxDstSlices);
    HcclResult RunStepRHD(
        const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32& channelIdx, const RhdStepInfo& step, const u64 rpt, u32& wqeCntTotal);
    HcclResult RunAllGatherRHD(
        const std::vector<ThreadHandle>& threads, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32& channelIdx, u32& totalWqeNum);
    u32 maxChannelsPerRank_ = 1;
    u32 totalWqeNum_{0}; // 本 rank 单份（每 rpt 份独立计数，RPT 不计入）发送 WQE 数（channel 0 口径）
    std::vector<u64> dataSplit_;
    std::vector<u64> dataOffset_;
    std::vector<u64> dataSplitTail_;
    std::vector<u64> dataOffsetTail_;
};

} // namespace ops_hccl

#endif // INS_TEMP_ALL_GATHER_RHD_H
