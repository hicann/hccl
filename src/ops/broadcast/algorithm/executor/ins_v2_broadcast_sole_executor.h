/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_BROADCAST_SOLE_EXECUTOR_H
#define HCCLV2_INS_V2_BROADCAST_SOLE_EXECUTOR_H

#include "executor_common_ops.h"
#include "topo_match_1d.h"
#include "topo_match_base.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2BroadcastSoleExecutor : public InsCollAlgBase {
public:
    explicit InsV2BroadcastSoleExecutor();
    ~InsV2BroadcastSoleExecutor() final = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
#ifndef AICPU_COMPILE
    HcclResult FastLaunch(const OpParam& param, const CcuFastLaunchCtx* fastLaunchCtx) override;
    HcclResult
    FastLaunchSaveCtx(const OpParam& param, const TemplateResource& templateAlgRes, u32 notifyNumOnMainThread) const;
#endif
private:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);

    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
};
} // namespace ops_hccl

#endif
