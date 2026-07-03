/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_REDUCE_SCATTER_SEQUENCE_EXECUTOR_3LEVEL_H
#define HCCLV2_INS_V2_REDUCE_SCATTER_SEQUENCE_EXECUTOR_3LEVEL_H

#include "executor_common_ops.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
class InsV2ReduceScatterSequenceExecutor3Level : public InsCollAlgBase {
public:
    explicit InsV2ReduceScatterSequenceExecutor3Level();
    ~InsV2ReduceScatterSequenceExecutor3Level() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable& resCtx) override;

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                            const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    void GenIntraTemplateParams(TemplateDataParams &tempAlgParamsIntra, const u64 processedDataCount,
        const u64 currDataCount, const u64 loop) const;
    void GenInterTemplateParams1(TemplateDataParams &tempAlgParamsInter, const u64 processedDataCount,
        const u64 currDataCount, const u64 loop) const;
    void GenInterTemplateParams2(TemplateDataParams &tempAlgParamsInter, const u64 processedDataCount,
        const u64 currDataCount, const u64 loop) const;
    template <typename InsAlgTemplate>
    HcclResult GenTempResource(const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempResource) const;

    uint32_t rankSizeLevel0_{0};
    uint32_t rankSizeLevel1_{0};
    uint32_t rankSizeLevel2_{0};
    bool skipLevel1_{false};

    uint32_t rankIdxLevel0_{0};
    uint32_t rankIdxLevel1_{0};
    uint32_t rankIdxLevel2_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;
};
}

#endif
