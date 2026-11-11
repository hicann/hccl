/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_GATHER_V_OMNIPIPE_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_GATHER_V_OMNIPIPE_EXECUTOR_H

#include "alg_param.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs.h"
#include "omnipipe_data_slice_calc.h"
#include "topo_match_base_v2.h"
#include "topo_match_two_level.h"
#include <type_traits>

namespace ops_hccl {
/*
 * AllGatherV OmniPipe 两级执行器：复用 AllGather 均匀流水线模板，变长语义由执行器消化：
 * 1. 预拷：各rank把有效数据（counts[rank]）拷入ccl buff的本rank均匀槽位（槽位大小取各rank counts最大值）；
 * 2. 通信：模板按均匀槽位做标准 AllGather（无变长感知），仅mesh×nhr两级（TopoMatchTwoLevel）；
 * 3. trim：从均匀槽位只提取各rank有效字节，按displs紧凑回拷到user output，padding全部丢弃。
 */
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
class InsV2AllGatherVOmniPipeExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllGatherVOmniPipeExecutor();
    ~InsV2AllGatherVOmniPipeExecutor() override = default;
    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

protected:
    HcclResult GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo) const;

    HcclResult InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult OrchestrateLoop(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx,
        std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap);
    HcclResult RestoreChannelMap(
        const AlgResourceCtxSerializable& resCtx,
        std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const override;
    HcclResult PrepareResForTemplateLevel(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase);
    HcclResult CalcResLevel(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        std::shared_ptr<InsAlgTemplateBase> tempAlg, AlgResourceRequest& resourceRequest) const;

private:
    HcclResult BuildSubCommAndTempMap(
        const OpParam& param, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
        std::vector<std::vector<u32>>& subCommRanks2, std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
        const TopoInfoWithNetLayerDetails* topoInfo);
    // 从param.varData解析counts/displs，并计算均匀槽位大小dataCount_（各rank counts最大值）
    HcclResult ParseCountsAndDispls(const OpParam& param);
    // 构造供均匀AllGather模板使用的param视图：填充DataDes（与vDataDes共享union，模板只认DataDes）
    OpParam MakeUniformParam(const OpParam& param) const;

    std::vector<u64> counts_; // 各rank有效数据量（元素个数）
    std::vector<u64> displs_; // 各rank数据在output中的起始偏移（元素个数）
    std::vector<uint64_t> rankSizeLevel_;
    std::vector<uint64_t> rankIdxLevel_;
    OpMode opMode_;

    ThreadHandle controlThread_;
    std::vector<ThreadHandle> threads_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;

    std::vector<u32> ntfIdxCtrlToTempXY_;
    std::vector<u32> ntfIdxTempToCtrlXY_;

    std::vector<std::vector<ThreadHandle>> levelThreads_;

    std::vector<ThreadHandle> tempMainThreadsXY_;
    AlgHierarchyInfoForAllLevel algHierarchyInfo_;

    OmniNeedSetStepNum omniNeedSetStepNum_ = OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
};
} // namespace ops_hccl
#endif
