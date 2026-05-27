/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_REDUCE_PARALLEL_EXECUTOR_H
#define HCCL_REDUCE_PARALLEL_EXECUTOR_H

#include <array>
#include "common_alg_template_base.h"
#include "executor_v2_base.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2,
    typename AlgTemplate3>
class ReduceParallelExecutor : public InsCollAlgBase {
public:
    static constexpr u32 dataSplitPart_{2};
    static constexpr u32 stageSize_{2};
    static constexpr u32 stepSize_{2};

    explicit ReduceParallelExecutor();
    ~ReduceParallelExecutor() override = default;

    std::string Describe() const override
    {
        return "Reduce Parallel Executor.";
    }

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

private:
    uint64_t GetRankSize(const std::vector<std::vector<u32>> &vTopo) const;
    HcclResult CalcLocalRoot();

    HcclResult PrepareResForStage(u32 stage);
    HcclResult PrepareResForStage2(u32 stage);
    TemplateDataParams GenDataParamsTempAlg(u32 dataSliceIdx, u32 stageIdx, u32 stepIdx, bool isInter);

    HcclResult OrchestrateImpl();
    HcclResult OrchestrateLoop(u32 loopTimes, u64 maxCountPerLoop);
    HcclResult OrchestrateStep(u32 stageIdx, u32 stepIdx);
    HcclResult RunTemplate(u32 dataSliceIdx, u32 stageIdx, u32 stepIdx, bool isInter);

#ifndef AICPU_COMPILE
    HcclResult FastLaunch(const OpParam &param, const CcuFastLaunchCtx *resCtx) override;
    HcclResult FastLaunchSaveCtx();
#endif

    u32 intraLocalRankSize_{0};     // server内算法rankSize
    u32 interLocalRankSize_{0};     // server间算法rankSize
    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    u32 intraLocalRoot_{0};     // server内算法root
    u32 interLocalRoot_{0};     // server间算法root

    ThreadHandle mainThread_;
    std::vector<ThreadHandle> templateMainThreads_;
    std::vector<u32> syncNotifyOnTemplates_;
    std::vector<u32> syncNotifyOnMain_;

    std::vector<std::vector<std::vector<u32>>> vTopo_;
    std::vector<u32> virtRanks_;
    std::array<std::map<u32, u32>, dataSplitPart_> virtRankMap_;

    std::vector<ThreadHandle> intraThreads_;
    std::vector<ThreadHandle> interThreads_;

    u32 ccuKernelLaunchNumRSIntra0_{0};
    u32 ccuKernelLaunchNumRSInter0_{0};
    u32 ccuKernelLaunchNumRSIntra1_{0};
    u32 ccuKernelLaunchNumRSInter1_{0};
    u32 ccuKernelLaunchNumAGIntra0_{0};
    u32 ccuKernelLaunchNumAGInter0_{0};
    u32 ccuKernelLaunchNumAGIntra1_{0};
    u32 ccuKernelLaunchNumAGInter1_{0};

    std::map<u32, std::vector<ChannelInfo>> intraLinks_;
    std::map<u32, std::vector<ChannelInfo>> interLinks_;

    std::vector<ThreadHandle> threads_;

    std::array<std::array<std::shared_ptr<CommonAlgTemplateBase>, dataSplitPart_>, stageSize_> algTemplatePtrArr_{{}};

    OpParam param_;
    AlgResourceCtxSerializable resCtx_;
    std::array<TemplateResource, 4> tempAlgResArr_{};
    std::array<u64, dataSplitPart_> dataOffsetPerLoop_{0, 0};
    std::array<u64, dataSplitPart_> dataCountPerLoop_{0, 0};
    std::vector<std::vector<u32>> temp0HierarchyInfo_;
    std::vector<std::vector<u32>> temp1HierarchyInfo_;
    double multipleDimensionSplitRatio_{0.5};
};
}  // namespace ops_hccl

#endif