/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_REDUCE_SCATTER_SEQUENCE_EXECUTOR_AICPU_H
#define HCCLV2_INS_V2_REDUCE_SCATTER_SEQUENCE_EXECUTOR_AICPU_H

#include "executor_common_ops.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2ReduceScatterSequenceExecutorAicpu : public InsCollAlgBase {
public:
    explicit InsV2ReduceScatterSequenceExecutorAicpu();
    ~InsV2ReduceScatterSequenceExecutorAicpu() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;
    
    HcclResult CalcAlgHierarchyInfo(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
                                    AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
#ifndef AICPU_COMPILE
HcclResult FastLaunch(const OpParam &param, const CcuFastLaunchCtx *resCtx) override;
HcclResult FastLaunchSaveCtx(const OpParam &param, const TemplateResource &templateAlgRes0,
                             const TemplateResource &templateAlgRes1, u32 notifyNumOnMainThread);
#endif
protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                            const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    void GenInterTemplateParams(TemplateDataParams &tempAlgParamsInter, const u64 processedDataCount,
        const u64 currDataCount, const u64 loop) const;
    void GenIntraTemplateParams(TemplateDataParams &tempAlgParamsIntra, const u64 processedDataCount,
        const u64 currDataCount, const u64 loop) const;
    template <typename InsAlgTemplate>
    HcclResult GenTempResource(const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempReousrce) const;

    uint32_t rankSizeLevel0_{0};
    uint32_t rankSizeLevel1_{0};

    uint32_t rankIdxLevel0_{0};
    uint32_t rankIdxLevel1_{0};
    
    u32 ccuKernelLaunchNumInter_{0};
    u32 ccuKernelLaunchNumIntra_{0};
    
    u64 scratchBlockSize_{0};
    CommEngine engine_{CommEngine::COMM_ENGINE_AICPU};

    std::vector<CcuKernelHandle> interCcuKernels_;
    std::vector<CcuKernelHandle> intraCcuKernels_;

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;               
};
}

#endif