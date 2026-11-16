/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_EXECUTOR_V2_BASE_H
#define BSR_HOST_TEST_EXECUTOR_V2_BASE_H

// InsCollAlgBase 的桩：只保留 BSR executor 真正触及的基类成员（myRank_ / rankSize_）与虚接口。
// 生产基类在此之外还有大量 cost / 保序 / CCU 设施，被测路径一个都不走。

#include <string>
#include <vector>

#include "alg_param.h"
#include "log.h"
#include "template_utils.h"

namespace ops_hccl {

class InsCollAlgBase {
public:
    InsCollAlgBase() = default;
    virtual ~InsCollAlgBase() = default;

    virtual std::string Describe() const { return "InsCollAlgBase"; }

    virtual std::vector<CostModelParam>
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
    {
        (void)comm;
        (void)topoInfo;
        (void)algName;
        (void)param;
        return {};
    }

    virtual AlgNetMeta
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
    {
        (void)param;
        (void)algName;
        return BuildDefaultMeshNetMeta(topoInfo);
    }

    virtual HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
        = 0;

    virtual HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
        = 0;

    virtual HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) = 0;

protected:
    static AlgNetMeta BuildDefaultMeshNetMeta(const TopoInfoWithNetLayerDetails* topoInfo)
    {
        AlgNetMeta meta;
        meta.netTypes.push_back(CommTopo::COMM_TOPO_1DMESH);
        meta.intraGroupMode = CostAggMode::SUM;
        meta.groupSizes = {1};
        meta.dataRatios = {1.0f};
        meta.rankSizes = {(topoInfo != nullptr) ? topoInfo->userRankSize : 1};
        return meta;
    }

    u32 myRank_ = INVALID_VALUE_RANKID;
    u32 rankSize_ = 0;
};

} // namespace ops_hccl

#endif // BSR_HOST_TEST_EXECUTOR_V2_BASE_H
