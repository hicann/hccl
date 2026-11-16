/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_SCATTER_SEQUENCE_EXECUTOR_3LEVEL_DPU_H
#define HCCLV2_INS_V2_SCATTER_SEQUENCE_EXECUTOR_3LEVEL_DPU_H

#include "executor_common_ops.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"
#include "topo_match_base_v2.h"
#include "topo_match_two_level.h"
#include "topo_match_three_level.h"

namespace ops_hccl {

// Scatter DPU 版三级 Sequence 执行器：下沉序 L0 绕路（框内，root 所在框）→ L1 DevRHD（仅 root
// 所在 pod 的 rank 执行）→ L2 DpuRHD（全员）。骨架照抄 ScatterSequenceAicpu3LevelExecutor
// （参数生成/root-pod 门控/skipLevel 退化），两处 DPU 差异：L2 参数按 DPU 变长契约 allRank*
// 重写（均匀退化）、GenTempResource 补 shmem 透传（照 AG/RS 三级 DPU 先例）。
// cost 接口按 preferred-only 家族口径：固定占位（不再 probe）。
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
class InsV2ScatterSequenceExecutor3LevelDpu : public InsCollAlgBase {
public:
    explicit InsV2ScatterSequenceExecutor3LevelDpu();
    ~InsV2ScatterSequenceExecutor3LevelDpu() override = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;
    AlgNetMeta GetAlgNetMeta(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const override;

protected:
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    void GenIntraTemplateParams(
        TemplateDataParams& tempAlgParamsIntra, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const;
    void GenInterTemplateParams1(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const;
    // L2(DpuRHD) 参数：在 AICPU 三级同名函数基础上补 allRank* 变长契约（均匀退化）
    void GenInterTemplateParams2(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const;
    template <typename InsAlgTemplate>
    HcclResult GenTempResource(
        const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource) const;

    uint32_t rankSizeLevel0_{0};
    uint32_t rankSizeLevel1_{0};
    uint32_t rankSizeLevel2_{0};
    bool skipLevel1_{false};
    bool skipLevel2_{false};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<ThreadHandle> threads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
};
} // namespace ops_hccl

#endif // HCCLV2_INS_V2_SCATTER_SEQUENCE_EXECUTOR_3LEVEL_DPU_H
