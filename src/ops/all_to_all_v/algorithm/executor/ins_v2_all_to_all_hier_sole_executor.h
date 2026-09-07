/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_V2_ALL_TO_ALL_HIER_SOLE_EXECUTOR_H
#define INS_V2_ALL_TO_ALL_HIER_SOLE_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_base.h"
#include "aicpu/hier/ins_temp_all_to_all_hier_stage_base.h"
#include "aicpu/hier/alltoall_stage_template_registry.h"

namespace ops_hccl {

constexpr u32 FULLMESH_THRESHOLD = 16;

template <typename AlgTopoMatch>
class InsV2AlltoAllHierSoleExecutor : public InsCollAlgBase {
public:
    explicit InsV2AlltoAllHierSoleExecutor();
    ~InsV2AlltoAllHierSoleExecutor() override = default;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;
    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;
    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;

protected:
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult SetStageParams(
        TemplateDataParams& tempAlgParams, u32 stageIndex, u32 totalStages, u64 loop, u64 currDataCount,
        u64 processedDataCount, u64 maxDataCountPerLoop, u32 totalRankSize, u32 stageRankSize);
    HcclResult FillTemplateResource(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateResource& templateRes, u32 stageIndex);
    std::vector<std::string> GetDefaultStageAlgos(u32 levelNum);
    HcclResult BuildStageTemplates(const OpParam& param);

private:
    std::vector<std::shared_ptr<InsTempAlltoAllHierStageBase>> stageTemplates_;
    std::vector<std::string> stageAlgos_;
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
    u32 totalStages_{0};
    u32 totalRankSize_{0};
    u64 dataTypeSize_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_INT8};
};

} // namespace ops_hccl

#endif // !INS_V2_ALL_TO_ALL_HIER_SOLE_EXECUTOR_H
