/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_V2_BROADCAST_OMNIPIPE_2D_EXECUTOR_H
#define HCCLV2_INS_V2_BROADCAST_OMNIPIPE_2D_EXECUTOR_H

#include "executor_common_ops.h"
#include "ccu_alg_template_base.h"
#include "omnipipe_scatter_data_slice_calc.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename CcuScatterAlgTemplateX, typename CcuScatterAlgTemplateY,
    typename CcuAgAlgTemplateX, typename CcuAgAlgTemplateY>
class InsV2BroadcastOmniPipe2dExecutor : public InsCollAlgBase {
public:
    explicit InsV2BroadcastOmniPipe2dExecutor();
    ~InsV2BroadcastOmniPipe2dExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

protected:
    /* *************** 算法编排 *************** */
    HcclResult InitCommInfo(const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo);

    HcclResult CalcResLevel(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resReqlevel, AlgResourceRequest &resourceReq, const int &curLevel);

    HcclResult InitSubCommRanks(std::vector<std::vector<u32>> &subCommRanks0,
        std::vector<std::vector<u32>> &subCommRanks1, const AlgHierarchyInfoForAllLevel &algHierarchyInfo);

    HcclResult GenTemplateAlgParamsByDimData(
        TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount);

    // 初始化4个维度的TemplateResource（ScatterX/Y、AgX/Y），按ccuKernelNum偏移切分ccuKernels
    HcclResult InitTemplateResources(const AlgResourceCtxSerializable &resCtx,
        TemplateResource &templateResourceScatterX, TemplateResource &templateResourceScatterY,
        TemplateResource &templateResourceAgX, TemplateResource &templateResourceAgY);

    // 计算SC/AG在Level0(mesh)/Level1(clos)的等效带宽，Level1按(rankSizeLevel1_-1)均摊
    HcclResult CalcEndpointBandwidth(
        std::vector<double> &endpointAttrBwAvgSC, std::vector<double> &endpointAttrBwAvgAG);

    // 计算数据切分: 每rank总量、每rank每loop量、单loop最大count、loop次数
    struct LoopSplitData {
        std::vector<u64> allRankSplitData;                       // 每个rank切分的总count
        std::vector<std::vector<u64>> multiLoopAllRankSplitData; // 每个rank每个loop切分的count
        u64 maxCountPerLoop{0};                                  // 每个loop最大count
        u32 loopTimes{0};                                        // loop次数
    };
    HcclResult CalcLoopSplitData(u64 maxTmpMemSize, LoopSplitData &loopSplitData);

    // 初始化OmniPipeSliceParam默认值（dataSizePerLoop/dataWholeSize/level配置等）
    HcclResult InitSliceParam(const OpParam &param, const std::vector<u64> &allRankSplitData,
        const std::vector<std::vector<u64>> &multiLoopAllRankSplitData, OmniPipeSliceParam &sliceParam);

    // 每轮loop按需重算Scatter/AG的OmniPipeSliceInfo（仅loop==0或与上轮切分不同时重算）
    HcclResult PrepareSliceInfoForLoop(u64 loop, u64 root, const std::vector<u64> &allRankSplitData,
        const std::vector<std::vector<u64>> &multiLoopAllRankSplitData, const std::vector<double> &endpointAttrBwAvgSC,
        const std::vector<double> &endpointAttrBwAvgAG, OmniPipeSliceParam &sliceParam,
        OmniPipeSliceInfo &omniPipeSliceInfoSC, OmniPipeSliceInfo &omniPipeSliceInfoAG);

    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);

    // 单步数据切片信息生成templateParam
    HcclResult GenTempAlgParamsIn2HCCLBuff(TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo,
        u64 processedDataCount, const AlgResourceCtxSerializable &resCtx, const OpParam &param);

    HcclResult GenTempAlgParamsHCCLBuff2HCCLBuff(TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo,
        u64 processedDataCount, const AlgResourceCtxSerializable &resCtx, const OpParam &param);

    std::vector<ThreadHandle> threads_;

private:
    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    enum OmnipipeARLevel {
        OMNIPIPE_SC_LEVEL0 = 0,
        OMNIPIPE_SC_LEVEL1 = 1,
        OMNIPIPE_AG_LEVEL0 = 2,
        OMNIPIPE_AG_LEVEL1 = 3,
        OMNIPIPE_AR_LEVEL_NUM = 4
    };

    bool isSameXAxisAsRoot = false; // 和root同x轴
    bool isSameYAxisAsRoot = false; // 和root同y轴
};
} // namespace ops_hccl

#endif