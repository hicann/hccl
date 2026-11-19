/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_EXECUTOR_DPU_3LEVEL_H
#define HCCLV2_INS_V2_ALL_REDUCE_SEQUENCE_EXECUTOR_DPU_3LEVEL_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "utils.h"
#include "log.h"
#include "sal.h"
#include "config_log.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "topo_match_multilevel.h"
#include "topo_match_base_v2.h"
#include "topo_match_two_level.h"
#include "topo_match_three_level.h"
#include "ins_v2_all_reduce_sequence_3level_utils.h"
#include <type_traits>
#include <memory>
#include "ins_v2_all_reduce_sequence_rhd_tail.h"

namespace ops_hccl {

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
class InsV2AllReduceSequenceExecutorDpu3Level : public InsV2AllReduceSequence3LevelBase {
public:
    explicit InsV2AllReduceSequenceExecutorDpu3Level();
    ~InsV2AllReduceSequenceExecutorDpu3Level() override = default;

    std::vector<CostModelParam> CalcCostCoeff(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param) override;
    AlgNetMeta GetAlgNetMeta(
        const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const override;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */
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
        TemplateStage<InsAlgTemplate0> rsLevel0;
        TemplateStage<InsAlgTemplate1> rsLevel1;
        TemplateStage<InsAlgTemplate2> rsLevel2;
        TemplateStage<InsAlgTemplate3> agLevel2;
        TemplateStage<InsAlgTemplate4> agLevel1;
        TemplateStage<InsAlgTemplate5> agLevel0;
        u32 scratchMult2{0};
        u64 scratchMultiplier{0};
        u64 maxCountPerLoop{0};
        u32 totalRankAlign{0};
    };

    struct LoopState {
        u64 loop{0};
        u64 processedDataCount{0};
        u64 currDataCount{0};
        u64 validDataCount{0};
        u64 chunkCount{0};
        u64 subSliceCount1{0};
        u64 subSliceSize1{0};
        u64 subSliceCount2{0};
        u64 subSliceSize2{0};
        u64 sliceSizeRSL1{0};
        u64 tailSizeRSL1{0};
        u64 symMemBaseOffRSL1{0};
    };

    HcclResult PrepareLoopContext(const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx);
    HcclResult PrepareLoopBuffers(const AlgResourceCtxSerializable& resCtx, LoopContext& ctx);
    HcclResult PrepareRhdLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx);
    HcclResult RunRhdReduceScatterLevel1(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunRhdReduceScatterLevel2(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunRhdAllGatherLevel2(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunRhdAllGatherLevel1(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunRhdLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx);
    HcclResult PrepareDefaultSymmetricMemory(const OpParam& param, LoopContext& ctx);
    HcclResult SelectDefaultLoopCount(LoopContext& ctx, LoopState& state);
    // ===== 默认（非 RHD）执行路径 =====
    // 走此路径的两类情形：
    //   1) 注册绑定的模板不是 RHD 族（Mesh / NHR / MeshMesh 等既有算法），默认选路或显式配置选中该注册；
    //   2) 显式配置的 RHD 未通过准入（层级数与配置形态不符 / L0 非对称 / dtype、PROD 不支持等），
    //      选路打印 WARNING 后回退自动选路，由既有算法注册执行——即"回退执行"落在此路径。
    // 与 RunRhd* 的差异：按各模板既有参数契约填参（stride/sliceSize 或 SplitData 的 allRank*），
    // 不做 RHD 的 σ 布局、rhdSkipPost/skipPrePhase 及代表侧桥接处理。
    HcclResult RunDefaultReduceScatterLevels01(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunDefaultReduceScatterLevel2(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunDefaultAllGather(const OpParam& param, LoopContext& ctx, LoopState& state);
    HcclResult RunDefaultLoop(const OpParam& param, LoopContext& ctx);
    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);
    HcclResult GenTempAlgParamsRSL2(
        const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL1, const u64 tailSizeRSL1,
        const u64 symMemBaseOffRSL1, TemplateDataParams& tempAlgParamsRSL2) const;
    void GenTempAlgParamsAGL2(
        const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL2, const u64 tailSizeRSL2,
        const u64 sliceSizeRSL1, const u64 symMemBaseOffRSL1, TemplateDataParams& tempAlgParamsAGL2) const;
    template <typename InsAlgTemplate>
    HcclResult GenTempResource(
        const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource) const;

    // DpuInter 变长分片契约：按 rankSize 把 dataCount 切成 allRank* 三件套（参照二级 AR）
    u64 RoundUp(const u64 dividend, const u64 divisor) const;
    HcclResult SplitData(const u64& dataCount, const uint64_t& rankSize, TemplateDataParams& tempAlgParams) const;
};
} // namespace ops_hccl

#endif
