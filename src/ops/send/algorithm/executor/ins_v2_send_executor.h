/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INS_V2_SEND_EXECUTOR_H
#define HCCL_INS_V2_SEND_EXECUTOR_H

#include "alg_param.h"
#include "topo_host.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {
class InsV2SendExecutor : public InsCollAlgBase {
public:
    std::string Describe() const override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

    // 资源计算
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    // 算法编排
    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;

    AlgNetMeta GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const override;

protected:
    HcclResult InitSendInfo(const HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo);
    // 图模式&单算子
    HcclResult OrchestrateImpl(const OpParam& param, const AlgResourceCtxSerializable& resCtx);

    HcclResult CalNumBlocks(u32& numBlocks, u64 dataSize, u32 numBlocksLimit) const;

    // 单算子|图模式
    OpMode opMode_;
    u32 remoteRank_;
    // 一次搬运最大数据量
    u64 maxLoopTransSize_;
    // 一次搬运最大数据个数
    u64 maxLoopTransCount_;
    u32 sliceId_{0};
};

} // namespace ops_hccl

#endif // #ifndef HCCL_INS_V2_SEND_EXECUTOR_H
