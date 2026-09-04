/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_barrier_sole_executor.h"
#include "ins_temp_barrier_nhr_aicpu.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs_registry.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BarrierSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BarrierSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BarrierSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    myRank_ = topoInfo->userRank;
    CHK_PRT_RET(
        algHierarchyInfo.infos.empty(),
        HCCL_ERROR("[InsV2BarrierSoleExecutor][CalcRes] myRank[%u] algHierarchyInfo.infos is empty", myRank_),
        HCCL_E_INTERNAL);
    InsAlgTemplate tempAlg(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    CHK_RET(tempAlg.CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2BarrierSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BarrierSoleExecutor][Orchestrate] Start");
    myRank_ = resCtx.topoInfo.userRank;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    CHK_PRT_RET(
        resCtx.algHierarchyInfo.infos.empty(),
        HCCL_ERROR("[InsV2BarrierSoleExecutor][Orchestrate] myRank[%u] algHierarchyInfo.infos is empty", myRank_),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        remoteRankToChannelInfo_.empty(),
        HCCL_ERROR("[InsV2BarrierSoleExecutor][Orchestrate] myRank[%u] remoteRankToChannelInfo_ is empty", myRank_),
        HCCL_E_INTERNAL);

    TemplateDataParams tempDataParams{};
    tempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempDataParams.repeatNum = 1;

    InsAlgTemplate tempAlg(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);

    TemplateResource templateResource;
    templateResource.channels = remoteRankToChannelInfo_[0];
    templateResource.threads = resCtx.threads;

    CHK_RET(tempAlg.KernelRun(param, tempDataParams, templateResource));

    HCCL_INFO("[InsV2BarrierSoleExecutor][Orchestrate] End");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
std::vector<CostModelParam> InsV2BarrierSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    return {{0.0f, 0.0f, 1.0f, 0.0f}};
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
AlgNetMeta InsV2BarrierSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)param;
    u32 rankSize = (topoInfo != nullptr) ? topoInfo->userRankSize : 1;
    AlgNetMeta meta;
    meta.netTypes.push_back(CommTopo::COMM_TOPO_1DMESH);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1};
    meta.dataRatios = {1.0f};
    meta.rankSizes = {rankSize};
    return meta;
}

REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_BARRIER, AicpuBarrierSoleNHR, InsV2BarrierSoleExecutor, TopoMatchOneLevel,
    InsTempBarrierNhrAicpu);
REGISTER_ALG_ATTRS(AicpuBarrierSoleNHR, topo.supportLevel0Topos = LEVEL0_TOPO_ANY;);

} // namespace ops_hccl
