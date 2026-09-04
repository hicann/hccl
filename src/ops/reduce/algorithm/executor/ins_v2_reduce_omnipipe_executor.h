/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_V2_REDUCE_OMNIPIPE_EXECUTOR_H
#define HCCLV2_CCU_V2_REDUCE_OMNIPIPE_EXECUTOR_H

#include "executor_common_ops.h"
#include "ccu_alg_template_base.h"
#include "omnipipe_gather_data_slice_calc.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_base_v2.h"
#include "topo_match_two_level.h"
#include <type_traits>
#include "executor_v2_base.h"       // 引入InsCollAlgBase基类
#include "alg_data_trans_wrapper.h" // for localCopy in Executor
#include "template_utils.h"         // for stepSliceInfo
#include "log.h"
#include "utils.h"

namespace ops_hccl {

template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
class InsV2ReduceOmniPipeExecutor : public InsCollAlgBase {
public:
    explicit InsV2ReduceOmniPipeExecutor();
    ~InsV2ReduceOmniPipeExecutor() override = default;

    HcclResult Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

    HcclResult CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& algAttrs) override;

protected:
    /* *************** 算法编排 *************** */
    HcclResult InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);
    HcclResult CalcResLevel(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resReqlevel, AlgResourceRequest& resourceReq, const int& curLevel);

    HcclResult InitSubCommRanks(
        std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo);

    HcclResult GenTemplateAlgParamsByDimData(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount);

    // 单步数据切片信息生成templateParam
    HcclResult GenTempAlgParamsIn2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param);
    HcclResult GenTempAlgParamsHCCLBuff2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param);

    HcclResult OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx);

    std::vector<ThreadHandle> threads_; // 相当于之前的std::vector<InsQuePtr> tempInsQue_;

    // 计算RS/G在Level0(mesh)/Level1(clos)的等效带宽，Level1按(rankSizeLevel1_-1)均摊
    HcclResult CalcEndpointBandwidth(
        std::vector<double>& endpointAttrBwAvgRS, std::vector<double>& endpointAttrBwAvgG, const OpParam& param);

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t rootx{0};
    uint64_t rooty{0};

    enum OmnipipeARLevel {
        OMNIPIPE_RS_LEVEL0 = 0,
        OMNIPIPE_RS_LEVEL1 = 1,
        OMNIPIPE_AG_LEVEL0 = 2,
        OMNIPIPE_AG_LEVEL1 = 3,
        OMNIPIPE_AR_LEVEL_NUM = 4
    };

    /// 对角算法专用
private:
    bool isSameXAxisAsRoot = false;
    bool isSameYAxisAsRoot = false;
};
} // namespace ops_hccl

#endif
