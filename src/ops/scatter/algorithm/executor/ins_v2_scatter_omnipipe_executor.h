/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_SCATTER_OMNIPIPE_EXECUTOR_H
#define HCCLV2_INS_V2_SCATTER_OMNIPIPE_EXECUTOR_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "omnipipe_scatter_data_slice_calc.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
class InsV2ScatterOmniPipeExecutor : public InsCollAlgBase {
public:
    explicit InsV2ScatterOmniPipeExecutor();
    ~InsV2ScatterOmniPipeExecutor() override = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
    HcclResult RestoreChannelMap(
        const AlgResourceCtxSerializable& resCtx,
        std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const override;

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult PrepareResForTemplateLevel(u32 level, const std::shared_ptr<InsAlgTemplateBase>& tempBase);

    // 单步数据切片信息生成templateParam
    HcclResult GenTempAlgParamsIn2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param);
    HcclResult GenTempAlgParamsHCCLBuff2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param);

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSizeLevel2_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t rankIdxLevel2_{0};
    bool isSameXAxisAsRoot = false; // 和root同x轴
    bool isSameYAxisAsRoot = false; // 和root同y轴
    bool isSameZAxisAsRoot = false; // 和root同z轴
    bool isSameSerAsRoot = false;   // 和root同机的节点
    ThreadHandle controlThread_;
    std::vector<std::vector<ThreadHandle>> levelThreads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    // xy两轴间同步使用
    std::vector<u32> notifyIdxCtrlToTempLevel01_;
    std::vector<u32> notifyIdxTempToCtrlLevel01_;
    std::vector<u32> notifyIdxCtrlToTempLevel2_;
    std::vector<u32> notifyIdxTempToCtrlLevel2_;
    std::vector<ThreadHandle> tempMainThreadsLevel2_;
    std::vector<ThreadHandle> tempMainThreadsLevel01_;

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<ThreadHandle> threads_; // 相当于之前的std::vector<InsQuePtr> tempInsQue_;

    enum class TopoType { UBX_2LEVEL, THREE_LEVEL };
    TopoType topoType_ = TopoType::UBX_2LEVEL;

    HcclResult BuildSubCommAndTempMap(
        const OpParam& param, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const TopoInfoWithNetLayerDetails* topoInfo);

    std::vector<std::vector<u32>> subCommRanks0_;
    std::vector<std::vector<u32>> subCommRanks1_;
    std::vector<std::vector<u32>> subCommRanks2_;
    std::shared_ptr<InsAlgTemplate0> tempLevel0_;
    std::shared_ptr<InsAlgTemplate1> tempLevel1_;
    std::shared_ptr<InsAlgTemplate2> tempLevel2_;
    OmniNeedSetStepNum omniNeedSetStepNum_ = OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
};
} // namespace ops_hccl

#endif // HCCLV2_INS_V2_SCATTER_OMNIPIPE_EXECUTOR_H
