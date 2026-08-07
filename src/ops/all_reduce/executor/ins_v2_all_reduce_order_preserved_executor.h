/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_ORDER_PRESERVED_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_REDUCE_ORDER_PRESERVED_EXECUTOR_H

#include "alg_param.h"
#include "ins_v2_reduce_scatter_order_preserved_executor.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_1d.h"

namespace ops_hccl {

struct OrderPreservedAllReduceMemInfo {
    u64 sizePerBlock{0};
    std::vector<u64> groupSize;
    bool scratchMemFlag;
    u64 totalSize{0};
};

template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
class InsV2AllReduceOrderPreservedExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllReduceOrderPreservedExecutor();
    ~InsV2AllReduceOrderPreservedExecutor() override = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitExecutorInfo(const OpParam& param);
    HcclResult CalcSizePerBlock(const OpParam& param);
    HcclResult CalcGroupSlices(const OpParam& param);

    template <typename InsAlgTemplate>
    HcclResult GenTempResource(
        const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource);

    void InitTemplateDataParams(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParams);
    void PrintTemplateResource(const TemplateResource& templateAlgRes);
    void PrintTemplateDataParams(const TemplateDataParams& tempAlgParams);

    HcclResult RunReduceScatter(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 currDataCount, u64 processedDataCount,
        std::shared_ptr<InsAlgTemplateRS> rsTempAlg, TemplateResource& rsTemplateAlgRes);
    HcclResult RunAllGather(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64 currDataCount, u64 processedDataCount,
        std::shared_ptr<InsAlgTemplateAG> agTempAlg, TemplateResource& agTemplateAlgRes);

    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;

    OrderPreservedAllReduceMemInfo memInfo_;
    bool deterministicStrict_{false};

    u64 outCclBuffSize_{0};
    u64 inCclBuffSize_{0};
    u64 outCclBuffOffset_{0};
    u64 inCclBuffOffset_{0};
};
} // namespace ops_hccl

#endif
