/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_ALL_GATHER_CONCURR_EXECUTOR_H
#define HCCLV2_INS_ALL_GATHER_CONCURR_EXECUTOR_H

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
#include "topo_match_ubx.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2AllGatherConcurrentExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllGatherConcurrentExecutor();
    ~InsV2AllGatherConcurrentExecutor() = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       const AlgHierarchyInfoForAllLevel &algHierarchyInfo,
                       AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                                    AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

private:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
                               InsAlgTemplate0 &algTemplateMesh, InsAlgTemplate1 &algTemplateNhr);

    HcclResult InitCommInfo(const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                            const AlgHierarchyInfoForAllLevel &algHierarchyInfo);

    void GetParallelDataSplit(std::vector<float> &splitDataSize) const;

    void GenTemplateAlgParams(const OpParam &param, const AlgResourceCtxSerializable &resCtx, const u64 dataOffset,
                                  const u64 dataCountPerLoopMesh, const u64 scratchOffset,
                                  TemplateDataParams &tempAlgParamsMesh) const;

    HcclResult PrepareResForTemplate(const OpParam &param, const AlgResourceCtxSerializable &resCtx, InsAlgTemplate0 &algTemplateMesh, InsAlgTemplate1 &algTemplateNhr);

    std::vector<ThreadHandle> threads_;  // 相当于之前的std::vector<InsQuePtr> tempInsQue_;
    std::vector<ThreadHandle> tmp0Threads_;
    std::vector<ThreadHandle> tmp1Threads_;
    ThreadHandle mainThread_;
    std::vector<ThreadHandle> templateMainThreads_;
    std::vector<u32> syncNotifyOnTemplates_;
    std::vector<u32> syncNotifyOnMain_;

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::map<u32, std::vector<ChannelInfo>> tmp0LinkMap_;
    std::map<u32, std::vector<ChannelInfo>> tmp1LinkMap_;
    std::vector<CcuKernelHandle> tmp0CcuKernels_;
    std::vector<CcuKernelHandle> tmp1CcuKernels_;
};
}

#endif  // HCCLV2_INS_ALL_GATHER_CONCURR_EXECUTOR_H