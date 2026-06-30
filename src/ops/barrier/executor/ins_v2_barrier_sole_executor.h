/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_V2_BARRIER_SOLE_EXECUTOR_H
#define INS_V2_BARRIER_SOLE_EXECUTOR_H

#pragma once

#include "executor_v2_base.h"
#include "alg_v2_template_base.h"

namespace ops_hccl {

// Barrier 单级编排：单个模板执行一次，纯同步无数据搬运。
template <typename AlgTopoMatch, typename InsAlgTemplate>
class InsV2BarrierSoleExecutor : public InsCollAlgBase {
public:
    InsV2BarrierSoleExecutor() {}
    ~InsV2BarrierSoleExecutor() override = default;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo,
                                    AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

private:
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
};

}  // namespace ops_hccl

#endif  // INS_V2_BARRIER_SOLE_EXECUTOR_H
