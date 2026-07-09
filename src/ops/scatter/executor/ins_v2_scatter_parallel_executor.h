/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_SCATTER_PARALLEL_EXECUTOR_H
#define HCCLV2_INS_V2_SCATTER_PARALLEL_EXECUTOR_H

#include "alg_param.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_multilevel.h"
#include "topo_match_pcie_mix.h"
#include "topo_match_ubx.h"
#include "channel.h"
#include "alg_v2_template_base.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2ScatterParallelExecutor : public InsCollAlgBase {
public:
    explicit InsV2ScatterParallelExecutor();
    ~InsV2ScatterParallelExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;
    HcclResult PreSyncInterTemplates();
    HcclResult PostSyncInterTemplates();
#ifndef AICPU_COMPILE
    HcclResult FastLaunch(const OpParam &param, const CcuFastLaunchCtx *ctx) override;
    HcclResult FastLaunchSaveCtx(const OpParam &param, const TemplateResource &templateAlgResIntra,
                                 const TemplateResource &templateAlgResInter, u32 notifyNumOnMainThread);
#endif
protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult PrepareResForTemplate(InsAlgTemplate0 &tempAlgIntra);
    HcclResult GenInsQuesHost(
        const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &tempAlgIntra, InsAlgTemplate1 &tempAlgInter);
    void GenTemplateAlgParamsIntra0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 dataCountPerLoopAxis0, const u64 scratchOffset,
        TemplateDataParams &tempAlgParamsIntra0) const;
    void GenTemplateAlgParamsIntra1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 dataCountPerLoopAxis1, const u64 scratchOffset,
        TemplateDataParams &tempAlgParamsIntra1) const;
    void GenTemplateAlgParamsInter0(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 dataCountPerLoopAxis0, const u64 scratchOffset,
        TemplateDataParams &tempAlgParamsInter0) const;
    void GenTemplateAlgParamsInter1(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset, const u64 dataCountPerLoopAxis1, const u64 scratchOffset,
        TemplateDataParams &tempAlgParamsInter1) const;

    void GetParallelDataSplit(std::vector<double> &splitDataSize) const;
    uint64_t GetRankSize(const std::vector<std::vector<u32>> &vTopo) const;

    u32 ccuKernelLaunchNumIntra0_{0};
    u32 ccuKernelLaunchNumInter0_{0};
    u32 ccuKernelLaunchNumIntra1_{0};
    u32 ccuKernelLaunchNumInter1_{0};

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    std::vector<ThreadHandle> requiredThreads_;
    std::vector<ThreadHandle> intraThreads_;
    std::vector<ThreadHandle> interThreads_;
    std::map<u32, std::vector<ChannelInfo>> intraChannelInfo_;
    std::map<u32, std::vector<ChannelInfo>> interChannelInfo_;

    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
    std::vector<std::vector<u32>> temp0HierarchyInfo_;
    std::vector<std::vector<u32>> temp1HierarchyInfo_;
};
}  // namespace ops_hccl

#endif