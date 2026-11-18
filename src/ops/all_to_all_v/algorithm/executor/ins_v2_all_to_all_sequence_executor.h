/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_V2_ALL_TO_ALL_SEQUENCE_EXECUTOR_H
#define INS_V2_ALL_TO_ALL_SEQUENCE_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_base.h"
#include "aicpu/hier/ins_temp_all_to_all_hier_stage_base.h"
#include "aicpu/hier/alltoall_stage_template_registry.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu/hier/ccu_temp_all_to_all_hier_stage_base.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

namespace ops_hccl {

constexpr u32 FULLMESH_THRESHOLD = 16;

template <typename AlgTopoMatch>
class InsV2AlltoAllSequenceExecutor : public InsCollAlgBase {
public:
    explicit InsV2AlltoAllSequenceExecutor();
    ~InsV2AlltoAllSequenceExecutor() override = default;
    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;
    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;
    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;
    AlgNetMeta GetAlgNetMeta(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const override;

protected:
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult SetStageParams(
        TemplateDataParams& tempAlgParams, u32 stageIndex, u32 totalStages, u64 loop, u64 currDataCount,
        u64 processedDataCount, u64 maxDataCountPerLoop, u32 totalRankSize, u32 stageRankSize);
    HcclResult FillTemplateResource(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateResource& templateRes, u32 stageIndex);
    HcclResult GetStageAlgos(const OpParam& param, u32 levelNum, std::vector<std::string>& stageAlgos);
    HcclResult BuildStageTemplates(const OpParam& param);
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
    // CCU 快速下发：单轮（loopTimes==1）场景缓存 kernel submitInfo 上下文，后续同签名算子直接重放
    HcclResult FastLaunch(const OpParam& param, const CcuFastLaunchCtx* resCtx) override;
    HcclResult FastLaunchSaveCtx(
        const OpParam& param, const std::vector<std::vector<CcuKernelSubmitInfo>>& stageSubmitInfos,
        u32 notifyNumOnMainThread) const;
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

private:
    // 按引擎创建第 k 个 stage 模板（k 为执行序，stageIndex=totalStages-1-k）
    HcclResult CreateStageTemplate(u32 k, const OpParam& param);
    u32 GetStageTemplateRankSize(u32 k, const OpParam& param) const;
    HcclResult RunStageTemplate(
        u32 k, const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateRes);
    HcclResult RestoreStageCcuKernels(const AlgResourceCtxSerializable& resCtx);

    std::vector<std::shared_ptr<InsTempAlltoAllHierStageBase>> stageTemplates_; // AICPU stage 模板
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
    std::vector<std::shared_ptr<CcuTempAlltoAllHierStageBase>> ccuStageTemplates_; // CCU stage 模板
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
    std::vector<std::string> stageAlgos_;
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<std::vector<CcuKernelHandle>> stageCcuKernels_; // CCU：按 stageIndex 落位的 kernel 切片
    std::vector<ThreadHandle> threads_;
    u32 totalStages_{0};
    u32 totalRankSize_{0};
    u64 dataTypeSize_{0};
    HcclDataType dataType_{HCCL_DATA_TYPE_INT8};
};

} // namespace ops_hccl

#endif // !INS_V2_ALL_TO_ALL_HIER_SEQUENCE_EXECUTOR_H
