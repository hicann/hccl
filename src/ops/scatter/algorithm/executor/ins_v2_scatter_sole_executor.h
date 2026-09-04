/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_SCATTER_SOLE_EXECUTOR_H
#define HCCLV2_INS_V2_SCATTER_SOLE_EXECUTOR_H

#include "alg_param.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_1d.h"
#include "topo_match_multilevel.h"
#include "topo_match_base_v2.h"
#include "topo_match_one_level.h"
#include <type_traits>

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2ScatterSoleExecutor : public InsCollAlgBase {
public:
    explicit InsV2ScatterSoleExecutor();
    ~InsV2ScatterSoleExecutor() override = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;
    AlgNetMeta GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const override;
#ifndef AICPU_COMPILE
    HcclResult FastLaunch(const OpParam& param, const CcuFastLaunchCtx* fastLaunchCtx) override;
    HcclResult
    FastLaunchSaveCtx(const OpParam& param, const TemplateResource& templateAlgRes, u32 notifyNumOnMainThread) const;
#endif

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);

    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_; // 相当于之前的std::vector<InsQuePtr> tempInsQue_;
};
} // namespace ops_hccl

#endif
