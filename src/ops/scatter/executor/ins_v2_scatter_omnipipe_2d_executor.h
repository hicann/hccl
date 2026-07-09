/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_SCATTER_EXECUTOR_CCU_V2_SCATTER_OMNIPIPE_EXECUTOR_H_
#define OPS_HCCL_SRC_OPS_SCATTER_EXECUTOR_CCU_V2_SCATTER_OMNIPIPE_EXECUTOR_H_

#include "topo_match_ubx.h"
#include "topo_match_multilevel.h"
#include "executor_v2_base.h"
#include "alg_data_trans_wrapper.h"
#include "template_utils.h"
#include "log.h"
#include "workflow.h"
#include "utils.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTempLevel0, typename InsAlgTempLevel1>
class InsV2ScatterOmniPipe2DExecutor : public InsCollAlgBase {
public:
    explicit InsV2ScatterOmniPipe2DExecutor();
    ~InsV2ScatterOmniPipe2DExecutor() override = default;
    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;
    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
    HcclResult GetRes(AlgResourceRequest &resourceRequest) const;

protected:
    u32 rankSizeLevel0_ = 0;
    u32 rankSizeLevel1_ = 0;
    u32 rankIdxLevel0_ = 0;
    u32 rankIdxLevel1_ = 0;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> localThreads_;
    bool isSameXAxisAsRoot = false; // 和root同x轴
    bool isSameYAxisAsRoot = false; // 和root同y轴

    // 线程管理
    std::vector<ThreadHandle> level0Threads_;
    std::vector<ThreadHandle> level1Threads_;
    ThreadHandle controlThread_;
    std::vector<ThreadHandle> templateMainThreads_;
    std::vector<ThreadHandle> templateLocalCopyThreads_;
    std::vector<u32> notifyIdxControlToTemplates_;
    std::vector<u32> notifyIdxTemplatesToControl_;
    HcclResult InitCommInfo(const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo);
    // 单步数据切片信息生成templateParam
    HcclResult GenTempAlgParamsIn2HCCLBuff(TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo,
        u64 processedDataCount, const AlgResourceCtxSerializable &resCtx, const OpParam &param);
    HcclResult GenTempAlgParamsHCCLBuff2HCCLBuff(TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo,
        u64 processedDataCount, const AlgResourceCtxSerializable &resCtx, const OpParam &param);

    // 为模板准备资源
    HcclResult PrepareResForTemplate(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        InsAlgTempLevel0 &algTempLevel0, InsAlgTempLevel1 &algTempLevel1);
    // 主执行函数
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
};

} // namespace ops_hccl

#endif // OPS_HCCL_SRC_OPS_SCATTER_EXECUTOR_CCU_V2_SCATTER_OMNIPIPE_EXECUTOR_H_