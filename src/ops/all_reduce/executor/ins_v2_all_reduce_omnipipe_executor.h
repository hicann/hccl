/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_ALL_REDUCE_OMNIPIPE_EXECUTOR_H
#define HCCLV2_INS_ALL_REDUCE_OMNIPIPE_EXECUTOR_H

#include "alg_param.h"
#include "topo_host.h"
#include "channel.h"
#include "alg_v2_template_base.h"
#include "executor_v2_base.h"
#include "coll_alg_v2_exec_registry.h"
#include "utils.h"
#include "log.h"
#include "workflow.h"
#include "sal.h"
#include "config_log.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"
#include "topo_match_pcie_mix.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {
template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsRsAlgTemplateZ,
    typename InsAgAlgTemplateX, typename InsAgAlgTemplateY, typename InsAgAlgTemplateZ>
class InsV2AllReduceOmniPipeExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllReduceOmniPipeExecutor();
    ~InsV2AllReduceOmniPipeExecutor() override = default;

    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;

    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;

protected:
    /* *************** 算法编排 *************** */
    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult InitCommInfo(
        HcclComm comm, const OpParam &param, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo);
    HcclResult InitExectorInfo(const OpParam &param, const AlgResourceCtxSerializable &resCtx);
    HcclResult GenTemplateAlgParamsByDimData(TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo) const;

    HcclResult RestoreChannelMap(const AlgResourceCtxSerializable &resCtx,
        std::vector<std::map<u32, std::vector<ChannelInfo>>> &rankIdToChannelInfo) const override;

    HcclResult InitOmniPipeScratchParam(OmniPipeScratchParam& scratchParam, const OpParam &param,
        const std::vector<double>& endpointAttrBwNew,
        std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap) const;

    HcclResult InitOmniPipeSliceParam(OmniPipeSliceParam& sliceParam, const OpParam &param,
        const std::vector<double>& endpointAttrBwNew,
        std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap, u64 maxCountPerLoop) const;

    HcclResult CalcResLevel(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const std::shared_ptr<InsAlgTemplateBase> tempAlg, AlgResourceRequest &resourceRequest, bool addChannel) const;

    HcclResult PrepareResForTemplateLevelRS(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase);
    HcclResult PrepareResForTemplateLevelAG(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase);

    HcclResult InitSubCommRanks(std::vector<std::vector<u32>>& subCommRanks0,
        std::vector<std::vector<u32>>& subCommRanks1,
        std::vector<std::vector<u32>>& subCommRanks2,
        const TopoInfoWithNetLayerDetails* topoInfo);

    HcclResult InitTemplate(const OpParam &param, std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
        const std::vector<std::vector<u32>>& subCommRanks0,
        const std::vector<std::vector<u32>>& subCommRanks1,
        const std::vector<std::vector<u32>>& subCommRanks2);

    HcclResult InitTemplateParams(const OpParam &param, const AlgResourceCtxSerializable &resCtx,
        const std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
        std::map<u32, TemplateResource>& tempResMap,
        std::map<u32, TemplateDataParams>& tempAlgParamMap);

    HcclResult DoLocalCopy(const TemplateDataParams &tempAlgParams, const ThreadHandle &thread,
        const std::vector<u64>& allRankSplitData, const std::vector<u64>& curLoopAllRankSplitData) const;

    HcclResult ClacOmniBandwidthInSever(const AlgResourceCtxSerializable &resCtx, std::vector<double>& bdvec);

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSizeLevel2_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t rankIdxLevel2_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;  // 相当于之前的std::vector<InsQuePtr> tempInsQue_;

    ThreadHandle controlThread_ = 0;

    std::vector<ThreadHandle> tempMainThreadsLevel01RS_;
    std::vector<u32> ntfIdxCtrlToTempLevel01RS_;
    std::vector<u32> ntfIdxTempToCtrlLevel01RS_;

    std::vector<ThreadHandle> tempMainThreadsLevel2RS_;
    std::vector<u32> ntfIdxCtrlToTempLevel2RS_;
    std::vector<u32> ntfIdxTempToCtrlLevel2RS_;

    std::vector<std::vector<ThreadHandle>> levelThreadsRS_; //new
    std::vector<std::vector<ThreadHandle>> levelThreadsAG_; // new

    std::vector<ThreadHandle> tempMainThreadsLevel01AG_;
    std::vector<u32> ntfIdxCtrlToTempLevel01AG_;
    std::vector<u32> ntfIdxTempToCtrlLevel01AG_;
    std::vector<ThreadHandle> tempMainThreadsLevel2AG_;
    std::vector<u32> ntfIdxCtrlToTempLevel2AG_;
    std::vector<u32> ntfIdxTempToCtrlLevel2AG_;
    OmniNeedSetStepNum omniNeedSetStepNum_ = OmniNeedSetStepNum::OMNIPIPE_DEFAULT;

    enum OmnipipeARLevel{
        OMNIPIPE_RS_LEVEL0 = 0,
        OMNIPIPE_RS_LEVEL1 = 1,
        OMNIPIPE_RS_LEVEL2 = 2,
        OMNIPIPE_AG_LEVEL0 = 3,
        OMNIPIPE_AG_LEVEL1 = 4,
        OMNIPIPE_AG_LEVEL2 = 5,
        OMNIPIPE_AR_LEVEL_NUM = 6
    };

    enum class TopoType { UBX_2LEVEL, THREE_LEVEL };
    TopoType topoType_ = TopoType::UBX_2LEVEL;

    HcclResult BuildSubCommAndTempMap(
        const OpParam& param,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        std::vector<std::vector<u32>>& subCommRanks0,
        std::vector<std::vector<u32>>& subCommRanks1,
        std::vector<std::vector<u32>>& subCommRanks2,
        std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
        const TopoInfoWithNetLayerDetails* topoInfo);

    std::vector<std::vector<u32>> subCommRanks0_;
    std::vector<std::vector<u32>> subCommRanks1_;
    std::vector<std::vector<u32>> subCommRanks2_;
};
}  // namespace ops_hccl

#endif