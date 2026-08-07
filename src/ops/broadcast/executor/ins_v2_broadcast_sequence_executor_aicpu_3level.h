/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_BROADCAST_SEQ_EXECUTOR_AICPU_3LEVEL_H
#define HCCLV2_INS_V2_BROADCAST_SEQ_EXECUTOR_AICPU_3LEVEL_H

#include "executor_common_ops.h"
#include "topo_match_1d.h"
#include "topo_match_base.h"
#include "topo_match_ubx.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
class BroadcastSequenceMesh1dNHRNHRExecutor : public InsCollAlgBase {
public:
    explicit BroadcastSequenceMesh1dNHRNHRExecutor();
    ~BroadcastSequenceMesh1dNHRNHRExecutor() = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult InitExecutorInfo(const OpParam& param, const AlgResourceCtxSerializable& resCtx);

    // 三层Scatter标量分片生成
    void GenTempAlgParamsScatterL0(u64 currDataCount, u64 processedDataCount, TemplateDataParams& params) const;
    void GenTempAlgParamsScatterL1(u64 level1TotalCnt, u64 l0SliceByte, TemplateDataParams& params) const;
    void
    GenTempAlgParamsScatterL2(u64 level2TotalCnt, u64 l0SliceByte, u64 l1SliceByte, TemplateDataParams& params) const;

    // 三层AllGather标量分片生成
    void GenTempAlgParamsAGL2(
        const u64 sliceSize, const u64 tailSize, TemplateDataParams& params, u64 l0SliceByte, u64 l1SliceByte) const;
    void
    GenTempAlgParamsAGL1(const u64 sliceSize, const u64 tailSize, TemplateDataParams& params, u64 l0SliceByte) const;
    void GenTempAlgParamsAGL0(
        const u64 processedDataCount, const u64 sliceSize, const u64 tailSize,
        TemplateDataParams& tempAlgParamsStepFour) const;

    template <typename InsAlgTemplate>
    HcclResult GenTempResource(
        const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource) const;

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSizeLevel2_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t rankIdxLevel2_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_; // 相当于之前的std::vector<InsQuePtr> tempInsQue_;

    u64 myRank_{0};
    u64 rankSize_{0};
    u64 dataCount_{0};
    u64 dataTypeSize_{0};
    u64 dataSize_{0};

    bool skipLevel1_{false};
    bool skipLevel2_{false};
};
} // namespace ops_hccl

#endif
