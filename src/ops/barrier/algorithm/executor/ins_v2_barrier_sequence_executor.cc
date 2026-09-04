/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_barrier_sequence_executor.h"
#include "ins_temp_barrier_mesh_1D.h"
#include "ins_temp_barrier_nhr_dpu.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs_registry.h"

namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    algHierarchyInfo_ = algHierarchyInfo;

    InsAlgTemplate0 intraTempAlg(param, myRank_, algHierarchyInfo.infos[0]);
    InsAlgTemplate1 interTempAlg(param, myRank_, algHierarchyInfo.infos[1]);

    AlgResourceRequest resReqIntra;
    CHK_RET(intraTempAlg.CalcRes(comm, param, topoInfo, resReqIntra));
    AlgResourceRequest resReqInter;
    CHK_RET(interTempAlg.CalcRes(comm, param, topoInfo, resReqInter));

    resourceRequest.slaveThreadNum = std::max(resReqIntra.slaveThreadNum, resReqInter.slaveThreadNum);
    resourceRequest.notifyNumPerThread = resReqIntra.notifyNumPerThread; // DPU 目前无 notify
    resourceRequest.notifyNumOnMainThread
        = std::max(resReqIntra.notifyNumOnMainThread, resReqInter.notifyNumOnMainThread);

    resourceRequest.channels = {resReqIntra.channels[0], resReqInter.channels[0]};
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BarrierSequenceExecutor][Orchestrate] Start");
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // Barrier 不依赖 dataCount，直接构造一份空的 TemplateDataParams 给两个模板。
    // 两个模板内部只用 channel/thread 资源，不读 buffInfo / sliceSize / count。
    TemplateDataParams interTempDataParams{};
    interTempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    interTempDataParams.repeatNum = 1;

    TemplateDataParams intraTempDataParams{};
    intraTempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    intraTempDataParams.repeatNum = 1;

    InsAlgTemplate1 interTempAlg(param, myRank_, algHierarchyInfo_.infos[1]);
    InsAlgTemplate0 intraTempAlg(param, myRank_, algHierarchyInfo_.infos[0]);

    TemplateResource templateResourceInter;
    templateResourceInter.channels = remoteRankToChannelInfo_[1];
    templateResourceInter.threads = resCtx.threads;
    templateResourceInter.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceInter.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

    TemplateResource templateResourceIntra;
    templateResourceIntra.channels = remoteRankToChannelInfo_[0];
    templateResourceIntra.threads = resCtx.threads;
    templateResourceIntra.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateResourceIntra.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;

    // 先框间（DPU）再框内（AICPU），与 AllGather SequenceExecutor 顺序一致。
    CHK_RET(interTempAlg.KernelRun(param, interTempDataParams, templateResourceInter));
    CHK_RET(intraTempAlg.KernelRun(param, intraTempDataParams, templateResourceIntra));

    HCCL_INFO("[InsV2BarrierSequenceExecutor][Orchestrate] End");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
std::vector<CostModelParam> InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    return {{0.0f, 0.0f, 1.0f, 0.0f}};
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
AlgNetMeta InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetAlgNetMeta(
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

REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_BARRIER, DpuBarrierSequenceMeshNHR, InsV2BarrierSequenceExecutor, TopoMatchTwoLevel,
    InsTempBarrierMesh1D, InsTempBarrierNHRDPU);
REGISTER_ALG_ATTRS(DpuBarrierSequenceMeshNHR, topo.isSupportLevel0PcieMix = true; topo.isHostDpuOnly = true;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS);
} // namespace ops_hccl
