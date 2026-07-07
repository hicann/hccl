/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCLV2_CCU_V2_ALL_REDUCE_OMNIPIPE_EXECUTOR_H
#define HCCLV2_CCU_V2_ALL_REDUCE_OMNIPIPE_EXECUTOR_H

#include "executor_common_ops.h"
#include "ccu_alg_template_base.h"
#include "omnipipe_data_slice_calc.h"
#include "topo_match_base.h"
#include "topo_match_multilevel.h"
#include "topo_match_ubx.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsRsAlgTemplateX, typename InsRsAlgTemplateY, typename InsAgAlgTemplateX, typename InsAgAlgTemplateY>
class InsV2AllReduceOmniPipe2dExecutor : public InsCollAlgBase {
public:
    explicit InsV2AllReduceOmniPipe2dExecutor();
    ~InsV2AllReduceOmniPipe2dExecutor() = default;
 
    HcclResult Orchestrate(const OpParam &param, const AlgResourceCtxSerializable &resCtx) override;
 
    /* *************** 资源计算 *************** */
    // 这些函数为ExecutorBase纯虚函数，必须重写
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest) override;

    HcclResult CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo) override;
 
protected:
    /* *************** 算法编排 *************** */
    HcclResult InitCommInfo(const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo, const AlgHierarchyInfoForAllLevel &algHierarchyInfo);
    HcclResult CalcResLevel(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                std::shared_ptr<CcuAlgTemplateBase> tempAlg, AlgResourceRequest& resourceReq, const int& curLevel);
    
    HcclResult InitTemplate(const OpParam& param, std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap,
                const std::vector<std::vector<u32>>& subCommRanks0, const std::vector<std::vector<u32>>& subCommRanks1);

    HcclResult InitTemplateParamsCommon(const OpParam& param, TemplateDataParams& templateDataParams);

    HcclResult InitTemplateParams(const OpParam& param, const AlgResourceCtxSerializable& resCtx,
                const std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap,
                std::map<u32, TemplateResource>& tempResMap,
                std::map<u32, TemplateDataParams>& tempAlgParamMap);

    HcclResult InitOmniPipeSliceParam(OmniPipeSliceParam& sliceParam, const OpParam& param,
                const std::vector<double>& endpointAttrBwAvg,
                std::map<u32, std::shared_ptr<CcuAlgTemplateBase>>& tempMap);
    
    HcclResult GenTemplateAlgParamsByDimData(TemplateDataParams &tempAlgParams, StepSliceInfo &stepSliceInfo, u64 processedDataCount);

    HcclResult OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx);
    HcclResult CalcSliceInfoAllReduce(u64 dataCount);
    u64 RoundUp(const u64 dividend, const u64 divisor) const;

    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_;  // 相当于之前的std::vector<InsQuePtr> tempInsQue_;

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};

    enum OmnipipeARLevel{
        OMNIPIPE_RS_LEVEL0 = 0,
        OMNIPIPE_RS_LEVEL1 = 1,
        OMNIPIPE_AG_LEVEL0 = 2,
        OMNIPIPE_AG_LEVEL1 = 3,
        OMNIPIPE_AR_LEVEL_NUM = 4
    };

/// 对角算法专用
private:
    ThreadHandle controlThread_;

    std::vector<std::vector<ThreadHandle>> levelThreads_;
    std::vector<u32> ntfIdxCtrlToTempLevel01RS_;
    std::vector<u32> ntfIdxTempToCtrlLevel01RS_;

    std::vector<std::vector<ThreadHandle>> levelThreadsAG_;
    std::vector<ThreadHandle> tempMainThreadsLevel01AG_;
    std::vector<u32> ntfIdxCtrlToTempLevel01AG_;
    std::vector<u32> ntfIdxTempToCtrlLevel01AG_;

    std::vector<ThreadHandle> tempMainThreadsLevel0AG_;
    std::vector<u32> ntfIdxCtrlToTempLevel0AG_;
    std::vector<u32> ntfIdxTempToCtrlLevel0AG_;
};
}

#endif