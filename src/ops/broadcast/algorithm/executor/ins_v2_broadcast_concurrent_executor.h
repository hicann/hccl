/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_BROADCAST_CONCURRENT_EXECUTOR_H
#define HCCLV2_INS_V2_BROADCAST_CONCURRENT_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_1d.h"
#include "topo_match_base.h"
#include "topo_match_ubx.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2BroadcastConcurrentExecutor : public InsCollAlgBase {
public:
    explicit InsV2BroadcastConcurrentExecutor();
    ~InsV2BroadcastConcurrentExecutor() final = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

private:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult InitExectorInfo(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult
    PrepareThreadFromTemplate(std::shared_ptr<InsAlgTemplate0>& tempAlg0, std::shared_ptr<InsAlgTemplate1>& tempAlg1);
    void GenTempAlgParams(
        const u64 dataOffset, const u64 dataCountforTemp, const u64 maxCountPerLoop,
        TemplateDataParams& tempAlgParams) const;

    std::vector<ThreadHandle> threads_;
    std::vector<ThreadHandle> temp0Threads_;
    ThreadHandle temp0ThreadMain_ = 0;
    std::vector<ThreadHandle> temp1Threads_;
    ThreadHandle temp1ThreadMain_ = 0;
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
};
} // namespace ops_hccl

#endif
