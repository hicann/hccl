/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_TWO_SHOT_SOLE_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_REDUCE_TWO_SHOT_SOLE_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_1d.h"
#include "topo_match_base.h"
#include "topo_match_base_v2.h"
#include "topo_match_one_level.h"
#include <type_traits>

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2AllReduceTwoShotSoleExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllReduceTwoShotSoleExecutor();
    ~InsV2AllReduceTwoShotSoleExecutor() override = default;

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;
    AlgNetMeta GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const override;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

protected:
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    void GenBaseTempAlgParams(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParamsReduceScatter,
        TemplateDataParams& tempAlgParamsAllGather) const;
    void GenTempAlgParamsReduceScatter(
        const u64 loop, const u64 currDataCount, const u64 processedDataCount,
        TemplateDataParams& tempAlgParamsReduceScatter) const;
    void GenTempAlgParamsAllGather(
        const u64 loop, const u64 currDataCount, const u64 processedDataCount,
        TemplateDataParams& tempAlgParamsAllGather) const;
    template <typename InsAlgTemplate>
    HcclResult GenTempResource(
        const AlgResourceCtxSerializable& resCtx, const std::shared_ptr<InsAlgTemplate>& algTemplate,
        TemplateResource& tempResource) const;

    u64 outCclBuffSize_{0};
    u64 inCclBuffSize_{0};
    u64 outCclBuffOffset_{0};
    u64 inCclBuffOffset_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
};
} // namespace ops_hccl

#endif
