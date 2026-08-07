/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_GATHER_PARALLEL_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_GATHER_PARALLEL_EXECUTOR_H

#include "executor_common_ops.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2AllGatherParallelExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllGatherParallelExecutor();
    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
#ifndef AICPU_COMPILE
    HcclResult FastLaunch(const OpParam& param, const CcuFastLaunchCtx* ctx) override;
    HcclResult FastLaunchSaveCtx(
        const OpParam& param, const TemplateResource& templateAlgResIntra, const TemplateResource& templateAlgResInter,
        u32 notifyNumOnMainThread);
#endif

protected:
    HcclResult CalcLocalRankSize();
    HcclResult InitExectorInfo(const OpParam& param);

    HcclResult InitCommInfo(
        HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfo& algHierarchyInfo);
    HcclResult OrchestrateLoop(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, InsAlgTemplate0& tempAlgIntra,
        InsAlgTemplate1& tempAlgInter);
    void GenTemplateAlgParamsIntra0(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 dataOffset,
        const u64 dataCountPerLoopAxis0, const u64 scratchOffset, TemplateDataParams& tempAlgParamsIntra0) const;
    void GenTemplateAlgParamsIntra1(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 dataOffset,
        const u64 dataCountPerLoopAxis1, const u64 scratchOffset, TemplateDataParams& tempAlgParamsIntra1) const;
    void GenTemplateAlgParamsInter0(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 dataOffset,
        const u64 dataCountPerLoopAxis0, const u64 scratchOffset, TemplateDataParams& tempAlgParamsInter0) const;
    void GenTemplateAlgParamsInter1(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, const u64 dataOffset,
        const u64 dataCountPerLoopAxis1, const u64 scratchOffset, TemplateDataParams& tempAlgParamsInter1) const;
    void GetParallelDataSplit(std::vector<float>& splitDataSize) const;
    HcclResult PrepareResForTemplate(InsAlgTemplate0& tempAlgIntra, InsAlgTemplate1& tempAlgInter);
    uint64_t GetRankSize(const std::vector<std::vector<u32>>& vTopo) const;

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    u32 ccuKernelLaunchNumIntra0_{0};
    u32 ccuKernelLaunchNumInter0_{0};
    u32 ccuKernelLaunchNumIntra1_{0};
    u32 ccuKernelLaunchNumInter1_{0};

    ThreadHandle mainThread_;
    std::vector<ThreadHandle> templateMainThreads_;
    std::vector<u32> syncNotifyOnTemplates_;
    std::vector<u32> syncNotifyOnMain_;

    std::vector<ThreadHandle> intraThreads_;
    std::vector<ThreadHandle> interThreads_;
    std::map<u32, std::vector<ChannelInfo>> intraLinkMap_;
    std::map<u32, std::vector<ChannelInfo>> interLinkMap_;
    std::vector<ThreadHandle> threads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<std::vector<u32>> intraHierarchyInfo_;
    std::vector<std::vector<u32>> interHierarchyInfo_;
    double multipleDimensionSplitRatio_{0.5};
    MultipleDimensionSplitRatioSource multipleDimensionSplitRatioSource_
        = MultipleDimensionSplitRatioSource::BUILTIN_FORMULA;
};
} // namespace ops_hccl

#endif
