/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_REDUCE_PARALLEL_EXECUTOR_H
#define HCCLV2_INS_V2_REDUCE_PARALLEL_EXECUTOR_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_base.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2ReduceParallelExecutor : public InsCollAlgBase {
public:
    explicit InsV2ReduceParallelExecutor();
    ~InsV2ReduceParallelExecutor() = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfo *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfo *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter);

    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

private:
    void GenTemplateAlgParams0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, u64 dataOffset,
        u64 dataCountPerLoopAxis0, u64 scratchOffset, TemplateDataParams &tempAlgParams0) const;
    void GenTemplateAlgParamsIntra1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, u64 dataOffset,
        u64 dataCountPerLoopAxis1, u64 scratchOffset, u64 othScratchOffset, TemplateDataParams &tempAlgParams1) const;
    void GenTemplateAlgParamsInter1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, u64 dataOffset,
        u64 dataCountPerLoopAxis1, u64 scratchOffset, u64 othScratchOffset, TemplateDataParams &tempAlgParams1) const;
    HcclResult PrepareResForTemplate(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter);
    void GetParallelDataSplit(std::vector<float> &splitDataSize) const;
    uint64_t GetRankSize(const std::vector<std::vector<u32>> &vTopo);

    HcclResult CalcLocalRoot();

    u32 intraLocalRankSize_{0};  // server内算法rankSize
    u32 interLocalRankSize_{0};  // server间算法rankSize
    u32 intraLocalRoot_ = UINT32_MAX;
    u32 interLocalRoot_ = UINT32_MAX;

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    std::vector<ThreadHandle> intraThreads_;
    std::vector<ThreadHandle> interThreads_;
    std::map<u32, std::vector<ChannelInfo>> intraLinkMap_;
    std::map<u32, std::vector<ChannelInfo>> interLinkMap_;
};
}  // namespace ops_hccl

#endif