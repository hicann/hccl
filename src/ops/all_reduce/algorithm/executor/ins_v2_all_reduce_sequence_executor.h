/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_EXECUTOR_H
#define HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_EXECUTOR_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "utils.h"
#include "log.h"
#include "alg_v2_template_base.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_multilevel.h"
#include "topo_match_base_v2.h"
#include "topo_match_two_level.h"
#include <type_traits>
#include <memory>
#include "ins_v2_all_reduce_sequence_rhd_tail.h"

namespace ops_hccl {

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
class InsV2AllReduceSequenceExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllReduceSequenceExecutor();
    ~InsV2AllReduceSequenceExecutor() override = default;

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;
    AlgNetMeta GetAlgNetMeta(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const override;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

protected:
    /* *************** 算法编排 *************** */
    // 单次编排的局部状态，不跨算子保留模板或参数。
    template <typename InsAlgTemplate>
    struct TemplateStage {
        std::shared_ptr<InsAlgTemplate> algorithm;
        TemplateDataParams params;
        TemplateResource resource;
    };

    struct LoopContext {
        ArSeqRhdTail tail;
        TemplateStage<InsAlgTemplate0> rsIntra;
        TemplateStage<InsAlgTemplate1> rsInter;
        TemplateStage<InsAlgTemplate2> agInter;
        TemplateStage<InsAlgTemplate3> agIntra;
        u64 maxCountPerLoop{0};
        u64 loopTimes{0};
        u32 totalRankAlign{0};
        HcclMem cclInMem{};
        HcclMem cclOutMem{};
    };

    struct LoopState {
        u64 loop{0};
        u64 processedDataCount{0};
        u64 currDataCount{0};
        u64 validDataCount{0};
        u64 chunkCount{0};
        u64 chunkSize{0};
        u64 subSliceCount{0};
        u64 subSliceSize{0};
    };

    HcclResult PrepareLoopContext(const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx);
    HcclResult PrepareRhdLoop(const OpParam& param, LoopContext& ctx);
    HcclResult RunRhdReduceScatterIntra(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunRhdReduceScatterInter(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunRhdAllGatherInter(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx, LoopState& state);
    HcclResult RunRhdAllGatherIntra(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx, LoopState& state);
    HcclResult RunRhdLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx);
    // ===== 默认（非 RHD）执行路径 =====
    // 走此路径的两类情形：
    //   1) 注册绑定的模板不是 RHD 族（Mesh / NHR / MeshMesh 等既有算法），默认选路或显式配置选中该注册；
    //   2) 显式配置的 RHD 未通过准入（层级数与配置形态不符 / L0 非对称 / dtype、PROD 不支持等），
    //      选路打印 WARNING 后回退自动选路，由既有算法注册执行——即"回退执行"落在此路径。
    // 与 RunRhd* 的差异：按各模板既有参数契约填参（stride/sliceSize 或 SplitData 的 allRank*），
    // 不做 RHD 的 σ 布局、rhdSkipPost/skipPrePhase 及代表侧桥接处理。
    HcclResult RunDefaultReduceScatterIntra(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunDefaultReduceScatterInter(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunDefaultAllGatherInter(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunDefaultAllGatherIntra(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunDefaultLoop(const OpParam& param, LoopContext& ctx);
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult InitExecutorInfo(const OpParam& param);
    HcclResult SplitData(const u64& dataCount, const uint64_t& rankSize, TemplateDataParams& tempAlgParams);
    u64 RoundUp(const u64 dividend, const u64 divisor);

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::vector<u32>> intraHierarchyInfo_;
    std::vector<std::vector<u32>> interHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_; // 相当于之前的std::vector<InsQuePtr> tempInsQue_;
};
} // namespace ops_hccl

#endif
