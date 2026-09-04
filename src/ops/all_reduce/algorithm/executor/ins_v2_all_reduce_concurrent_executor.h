/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_CONCURRENT_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_REDUCE_CONCURRENT_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_base.h"
#include "topo_match_ubx.h"
#include "topo_match_base_v2.h"
#include "topo_match_concurrent_v2.h"
#include <type_traits>

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2AllReduceConcurrentExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllReduceConcurrentExecutor();
    ~InsV2AllReduceConcurrentExecutor() override = default;

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;
    AlgNetMeta GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const override;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

/* *************** 资源计算 *************** */
// 这些函数为ExecutorBase纯虚函数，必须重写
#ifndef AICPU_COMPILE
    HcclResult FastLaunch(const OpParam& param, const CcuFastLaunchCtx* ctx) override;
    HcclResult FastLaunchSaveCtx(
        const OpParam& param, const TemplateResource& templateAlgRes0, const TemplateResource& templateAlgRes1,
        u32 notifyNumOnMainThread);
#endif
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);

    HcclResult CalcChannelRequest(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const std::vector<std::vector<u32>>& subCommRanks, std::vector<HcclChannelDesc>& channelDescs,
        CommTopo topo) const;

    std::vector<ThreadHandle> threads_;
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<ThreadHandle> temp0Threads_;
    ThreadHandle temp0ThreadMain_ = 0;
    std::vector<ThreadHandle> temp1Threads_;
    ThreadHandle temp1ThreadMain_ = 0;
};
} // namespace ops_hccl

#endif
