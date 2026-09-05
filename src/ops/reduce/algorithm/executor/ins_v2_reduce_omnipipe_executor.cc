/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_omnipipe_executor.h"

#include <algorithm>
#include <string>

#include "alg_attrs_registry.h"
#include "alg_data_trans_wrapper.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_omnipipe_nhr1d_mem2mem.h"
#include "ccu_temp_reduce_scatter_omnipipe_mesh1d.h"
#include "ccu_temp_gather_omnipipe_mesh_1d_mem2mem.h"
#include "ccu_temp_gather_omnipipe_nhr1d_mem2mem.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
namespace ops_hccl {
constexpr u64 OMNI_UBX_MS_DATA_SIZE = 64 * 1024 * 1024; // MS模式UBX流水算法数据量下限，与selector保持一致
constexpr u64 OMNI2D_UBX_REDUCE_DATA_SIZE = 128 * 1024 * 1024; // UBX机型ccu并行/流水算法数据量分界，与selector保持一致
constexpr u32 OMNIPIPE_MIN_THREAD_NUM = 2;
constexpr u32 OMNIPIPE_MIN_CCU_KERNEL_NUM = 4;
namespace {
    constexpr double OMNIPIPE_FIXED_UB_UTILIZATION = 0.85;
    constexpr double GBPS_TO_BYTES_PER_SECOND = 1000.0 * 1000.0 * 1000.0;

    struct OmniPipeStageCost {
        double transferCoeff = 0.0;
        float syncCost = 0.0f;
        u64 stepNum = 0;
        bool reachesMaxStep = false;
    };

    bool CalcOmniPipe2dCostAxes(const TopoInfoWithNetLayerDetails* topoInfo, u64& meshRankSize, u64& closRankSize)
    {
        if (topoInfo == nullptr || topoInfo->topoInstDetailsOfLayer.empty()) {
            return false;
        }
        const auto& rankNumForTopoType = topoInfo->topoInstDetailsOfLayer[0].rankNumForTopoType;
        auto meshIt = rankNumForTopoType.find(CommTopo::COMM_TOPO_1DMESH);
        auto closIt = rankNumForTopoType.find(CommTopo::COMM_TOPO_CLOS);
        if (meshIt == rankNumForTopoType.end() || meshIt->second.empty() || closIt == rankNumForTopoType.end()
            || closIt->second.empty() || meshIt->second[0] == 0 || closIt->second[0] % meshIt->second[0] != 0) {
            return false;
        }
        meshRankSize = meshIt->second[0];
        closRankSize = closIt->second[0] / meshRankSize;
        return closRankSize > 0 && meshRankSize * closRankSize == topoInfo->userRankSize;
    }

    u64 CalcStepNumByAxes(
        double firstBandwidth, double secondBandwidth, u64 firstRankSize, u64 secondRankSize, u64 maxStepNum,
        bool isReduceScatter)
    {
        const bool firstIsSlow = firstBandwidth <= secondBandwidth;
        const double slowBandwidth = firstIsSlow ? firstBandwidth : secondBandwidth;
        const double fastBandwidth = firstIsSlow ? secondBandwidth : firstBandwidth;
        const u64 slowRankSize = firstIsSlow ? firstRankSize : secondRankSize;
        const u64 fastRankSize = firstIsSlow ? secondRankSize : firstRankSize;
        return isReduceScatter ?
                   CalcReducescatterStepNum2D(slowBandwidth, fastBandwidth, slowRankSize, fastRankSize, maxStepNum) :
                   CalcAllgatherStepNum2D(slowBandwidth, fastBandwidth, slowRankSize, fastRankSize, maxStepNum);
    }

    float CalcTemplateLatency(u32 taskNum)
    {
        float latency = 0.0f;
        CostModelManager::Global()->CalcLatencyParams(taskNum, EngineType::CCU, latency);
        return latency;
    }

    OmniPipeStageCost CalcStageCost(
        u64 meshRankSize, u64 closRankSize, u64 totalRankSize, double meshBandwidth, double closBandwidth,
        bool isReduceScatter)
    {
        OmniPipeStageCost stage;
        const double closPlanBandwidth = closRankSize > 1 ? closBandwidth / (closRankSize - 1) : closBandwidth;
        const u64 maxStepNum = isReduceScatter ?
                                   static_cast<u64>(SetMaxStepNumOmni(OmniNeedSetStepNum::OMNIPIPE_DEFAULT) + 1) :
                                   MAX_STEP_NUM;
        stage.stepNum = CalcStepNumByAxes(
            meshBandwidth, closPlanBandwidth, meshRankSize, closRankSize, maxStepNum, isReduceScatter);
        stage.reachesMaxStep = stage.stepNum == maxStepNum;

        const bool meshActive = meshRankSize > 1;
        const bool closActive = closRankSize > 1;
        if (meshActive && closActive) {
            if (stage.reachesMaxStep) {
                stage.transferCoeff = meshBandwidth <= closPlanBandwidth ? (meshRankSize - 1) / meshBandwidth :
                                                                           (closRankSize - 1) / closBandwidth;
            } else {
                stage.transferCoeff = (totalRankSize - 1) / (meshBandwidth + closBandwidth);
            }
        } else if (meshActive) {
            stage.transferCoeff = (meshRankSize - 1) / meshBandwidth;
        } else if (closActive) {
            stage.transferCoeff = (closRankSize - 1) / closBandwidth;
        }

        const float meshLatency = meshActive ? CalcTemplateLatency(1) : 0.0f;
        const float nhrLatency = closActive ? CalcTemplateLatency(GetNHRStepNum(static_cast<u32>(closRankSize))) : 0.0f;
        stage.syncCost = 2.0f * static_cast<float>(stage.stepNum) * std::max(meshLatency, nhrLatency);
        return stage;
    }
} // namespace
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    InsV2ReduceOmniPipeExecutor()
{}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
std::vector<CostModelParam>
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    if (topoInfo == nullptr || algName == nullptr) {
        HCCL_ERROR("[%s] topoInfo or algName is null.", __func__);
        return {};
    }
    u64 meshRankSize = 1;
    u64 closRankSize = 1;
    if (!CalcOmniPipe2dCostAxes(topoInfo, meshRankSize, closRankSize)) {
        HCCL_WARNING("[%s] unable to derive OmniPipe axes for algName[%s].", __func__, algName);
        return {};
    }

    const bool isCcuMs = std::string(algName) == "CcuMSReducePipeLineMeshNHR";
    const double rsMeshBandwidth
        = (isCcuMs ? BW_OMNI_UBX_CCU_MS_RS_MESH : BW_OMNI_UBX_CCU_SCHED_RS_MESH) / OMNIPIPE_FIXED_UB_UTILIZATION;
    const double rsClosBandwidth
        = (isCcuMs ? BW_OMNI_UBX_CCU_MS_RS_CLOS : BW_OMNI_UBX_CCU_SCHED_R_RS_CLOS) / OMNIPIPE_FIXED_UB_UTILIZATION;
    const double gatherMeshBandwidth
        = (isCcuMs ? BW_OMNI_UBX_CCU_MS_SCHED_G_MESH : BW_OMNI_UBX_CCU_SCHED_G_MESH) / OMNIPIPE_FIXED_UB_UTILIZATION;
    const double gatherClosBandwidth
        = (isCcuMs ? BW_OMNI_UBX_CCU_MS_SCHED_G_CLOS : BW_OMNI_UBX_CCU_SCHED_G_CLOS) / OMNIPIPE_FIXED_UB_UTILIZATION;

    const OmniPipeStageCost rsCost
        = CalcStageCost(meshRankSize, closRankSize, topoInfo->userRankSize, rsMeshBandwidth, rsClosBandwidth, true);
    const OmniPipeStageCost gatherCost = CalcStageCost(
        meshRankSize, closRankSize, topoInfo->userRankSize, gatherMeshBandwidth, gatherClosBandwidth, false);

    CostModelParam costParam{};
    costParam.A = static_cast<float>(
        (rsCost.transferCoeff + gatherCost.transferCoeff) / topoInfo->userRankSize / GBPS_TO_BYTES_PER_SECOND);
    CostModelManager::Global()->CalcLocalCopyParams(1.0f, EngineType::CCU, costParam.B);
    costParam.C = rsCost.syncCost + gatherCost.syncCost;

    HCCL_INFO(
        "[%s] algName[%s] axes[%llu,%llu] rsStep[%llu] rsMaxStep[%d] gatherStep[%llu] gatherMaxStep[%d] "
        "Ufixed[%f] A[%e] B[%e] C[%e].",
        __func__, algName, meshRankSize, closRankSize, rsCost.stepNum, rsCost.reachesMaxStep, gatherCost.stepNum,
        gatherCost.reachesMaxStep, OMNIPIPE_FIXED_UB_UTILIZATION, costParam.A, costParam.B, costParam.C);
    return {costParam};
}

template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
AlgNetMeta
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)topoInfo;
    AlgNetMeta meta;
    meta.netTypes = {CommTopo::COMM_TOPO_1DMESH};
    meta.groupSizes = {1};
    return meta;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    if (algHierarchyInfo.infos.empty()) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos[0] is invalid (empty or size < 2).", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rootx = param.root % rankSizeLevel0_;
    rooty = param.root / rankSizeLevel0_;
    bool isRoot = (myRank_ == param.root);
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx && !isRoot);
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty && !isRoot);
    HCCL_DEBUG(
        "[%s]myRank[%u] rankSize[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] "
        "rankIdxLevel1[%u] devType[%u] dataCount[%u] dataType[%u] dataTypeSize[%u]",
        __func__, myRank_, rankSize_, rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_, devType_,
        dataCount_, dataType_, dataTypeSize_);
    return HcclResult::HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    CalcResLevel(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resReqlevel, AlgResourceRequest& resourceReq, const int& curLevel)
{
    resourceReq.slaveThreadNum += resReqlevel.slaveThreadNum;               // 从流数 一般是0
    resourceReq.notifyNumOnMainThread += resReqlevel.notifyNumOnMainThread; // 一般是0
    resourceReq.notifyNumPerThread.insert(
        resourceReq.notifyNumPerThread.end(), // 一般是0
        resReqlevel.notifyNumPerThread.begin(), resReqlevel.notifyNumPerThread.end());
    // 资源组的值一样就一起申请，资源组的值不一样就串行申请，前一个销毁后后一个申请
    HCCL_DEBUG("[%s] currTemplate has [%d] kernels.", __func__, resReqlevel.ccuKernelNum[0]);
    if (curLevel == OMNIPIPE_RS_LEVEL0 || curLevel == OMNIPIPE_RS_LEVEL1) {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo& info) {
            info.resGroup = 0;
        });
    } else {
        std::for_each(resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end(), [](CcuKernelInfo& info) {
            info.resGroup = 1;
        });
    }
    resourceReq.ccuKernelInfos.insert(
        resourceReq.ccuKernelInfos.end(), resReqlevel.ccuKernelInfos.begin(), resReqlevel.ccuKernelInfos.end());
    resourceReq.ccuKernelNum.insert(
        resourceReq.ccuKernelNum.end(), resReqlevel.ccuKernelNum.begin(), resReqlevel.ccuKernelNum.end());
    return HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    InitSubCommRanks(
        std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    subCommRanks1.clear();
    subCommRanks0.clear();
    if (algHierarchyInfo.infos.empty()) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos[0] is invalid (empty or size < 2).", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    subCommRanks0.push_back(algHierarchyInfo.infos[0][0]);
    subCommRanks1.push_back(algHierarchyInfo.infos[1][0]);
    return HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    HCCL_DEBUG("start CalcRes");
    CHK_RET(InitCommInfo(param, topoInfo, algHierarchyInfo));
    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    CHK_RET(InitSubCommRanks(subCommRanks0, subCommRanks1, algHierarchyInfo));
    CcuRsAlgTemplateX rsAlgTempLevelX(param, myRank_, subCommRanks0);
    CcuRsAlgTemplateY rsAlgTempLevelY(param, myRank_, subCommRanks1);
    CcuGAlgTemplateX gAlgTempLevelX(param, myRank_, subCommRanks0);
    CcuGAlgTemplateY gAlgTempLevelY(param, myRank_, subCommRanks1);
    // 计算调用每一个template的资源
    resourceRequest.slaveThreadNum = 0; // ccu内部没有从流和notify
    resourceRequest.notifyNumOnMainThread = 0;
    AlgResourceRequest resRsReqLevelX;
    CHK_RET(rsAlgTempLevelX.CalcRes(comm, param, topoInfo, resRsReqLevelX));
    AlgResourceRequest resRsReqLevelY;
    CHK_RET(rsAlgTempLevelY.CalcRes(comm, param, topoInfo, resRsReqLevelY));
    AlgResourceRequest resGReqLevelX;
    CHK_RET(gAlgTempLevelX.CalcRes(comm, param, topoInfo, resGReqLevelX));
    AlgResourceRequest resGReqLevelY;
    gAlgTempLevelY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
    CHK_RET(gAlgTempLevelY.CalcRes(comm, param, topoInfo, resGReqLevelY));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resRsReqLevelX, resourceRequest, 0));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resRsReqLevelY, resourceRequest, 1));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resGReqLevelX, resourceRequest, 2));
    CHK_RET(CalcResLevel(comm, param, topoInfo, resGReqLevelY, resourceRequest, 3));
    resourceRequest.slaveThreadNum += 1; // 需要一个主流和一个从流来并行2d
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    HCCL_DEBUG(
        "[%s] slaveThreadNum:%d, notifyNumOnMainThread:%d", __func__, resourceRequest.slaveThreadNum,
        resourceRequest.notifyNumOnMainThread);
    HCCL_DEBUG("end CalcRes");
    return HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_DEBUG("[%s] start", __func__);
    threads_ = resCtx.threads;
    HCCL_DEBUG("[%s] threads size: %u", __func__, threads_.size());
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    maxTmpMemSize_ = resCtx.cclMem.size;
    if (resCtx.algHierarchyInfo.infos.empty()) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos[0] is invalid (empty or size < 2).", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankSizeLevel0_ = resCtx.algHierarchyInfo.infos[0][0].size();
    if (rankSizeLevel0_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel0 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankSizeLevel1_ = resCtx.algHierarchyInfo.infos[1][0].size();
    if (rankSizeLevel1_ == 0) {
        HCCL_ERROR("[%s] rankSizeLevel1 is 0", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rootx = param.root % rankSizeLevel0_;
    rooty = param.root / rankSizeLevel0_;
    bool isRoot = (myRank_ == param.root);
    isSameYAxisAsRoot = (rankIdxLevel0_ == rootx && !isRoot);
    isSameXAxisAsRoot = (rankIdxLevel1_ == rooty && !isRoot);
    HCCL_DEBUG(
        "[%s] myRank[%u] rankSizeLevel0[%u] rankSizeLevel1[%u] rankIdxLevel0[%u] rankIdxLevel1[%u]", __func__, myRank_,
        rankSizeLevel0_, rankSizeLevel1_, rankIdxLevel0_, rankIdxLevel1_);
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ReduceOmniPipeExecutor][Orchestrate]errNo[0x%016llx] executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}
// 将计算出的单步slice信息初始化到templateParam中
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    GenTemplateAlgParamsByDimData(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount)
{
    tempAlgParams.count = 0;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff + processedDataCount * dataTypeSize_;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    return HcclResult::HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    GenTempAlgParamsIn2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param)
{
    tempAlgParams.count = 0;
    tempAlgParams.dataType = dataType_;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = param.inputPtr;
    stepSliceInfo.buffInfo.inputSize = param.inputSize;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.inBuffType = BufferType::INPUT;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    u64 inputOffset = stepSliceInfo.buffInfo.inBuffBaseOff + processedDataCount * dataTypeSize_;
    u64 outputOffset = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();
    tempAlgParams.sliceSize = 0;
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff = inputOffset;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = outputOffset;
    tempAlgParams.buffInfo.inBuffBaseOff = inputOffset;
    tempAlgParams.buffInfo.outBuffBaseOff = outputOffset;
    return HcclResult::HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    GenTempAlgParamsHCCLBuff2HCCLBuff(
        TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo, u64 processedDataCount,
        const AlgResourceCtxSerializable& resCtx, const OpParam& param)
{
    tempAlgParams.count = 0;
    tempAlgParams.dataType = dataType_;
    stepSliceInfo.buffInfo.hcclBuff = resCtx.cclMem;
    stepSliceInfo.buffInfo.inputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.inputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.outputPtr = resCtx.cclMem.addr;
    stepSliceInfo.buffInfo.outputSize = resCtx.cclMem.size;
    stepSliceInfo.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    stepSliceInfo.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo = stepSliceInfo.buffInfo;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    tempAlgParams.stepSliceInfo.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.stepSliceInfo.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.inputSliceStride = 0;
    tempAlgParams.outputSliceStride = 0;
    tempAlgParams.localCopyFlag = 0;
    tempAlgParams.sliceSize = 0;
    tempAlgParams.repeatNum = stepSliceInfo.stepCount.size();
    return HcclResult::HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    CalcEndpointBandwidth(
        std::vector<double>& endpointAttrBwAvgRS, std::vector<double>& endpointAttrBwAvgG, const OpParam& param)
{
    // RS带宽: Level0走mesh, Level1走clos（按rankSizeLevel1_-1均摊）
    double eqBwLevel0RS = BW_OMNI_DEFAULT;
    double eqBwLevel1RS = BW_OMNI_DEFAULT;
    double eqBwLevel0G = BW_OMNI_DEFAULT;
    double eqBwLevel1G = BW_OMNI_DEFAULT;
    if (param.opExecuteConfig == OpExecuteConfig::CCU_SCHED) {
        eqBwLevel0RS = BW_OMNI_UBX_CCU_SCHED_RS_MESH;
        eqBwLevel1RS = BW_OMNI_UBX_CCU_SCHED_R_RS_CLOS;
        eqBwLevel0G = BW_OMNI_UBX_CCU_SCHED_G_MESH;
        eqBwLevel1G = BW_OMNI_UBX_CCU_SCHED_G_CLOS;
    } else if (param.opExecuteConfig == OpExecuteConfig::CCU_MS) {
        eqBwLevel0RS = BW_OMNI_UBX_CCU_MS_RS_MESH;
        eqBwLevel1RS = BW_OMNI_UBX_CCU_MS_RS_CLOS;
        eqBwLevel0G = BW_OMNI_UBX_CCU_MS_SCHED_G_MESH;
        eqBwLevel1G = BW_OMNI_UBX_CCU_MS_SCHED_G_CLOS;
    }
    eqBwLevel1RS = rankSizeLevel1_ > 1 ? eqBwLevel1RS / (rankSizeLevel1_ - 1) : eqBwLevel1RS;
    endpointAttrBwAvgRS = {eqBwLevel0RS, eqBwLevel1RS, 1.0};
    // G带宽: Level0走mesh, Level1走clos（按rankSizeLevel1_-1均摊）
    eqBwLevel1G = rankSizeLevel1_ > 1 ? eqBwLevel1G / (rankSizeLevel1_ - 1) : eqBwLevel1G;
    endpointAttrBwAvgG = {eqBwLevel0G, eqBwLevel1G, 1.0};
    HCCL_DEBUG(
        "[%s] eqBwLevel0RS:%f, eqBwLevel1RS:%f, eqBwLevel0G:%f, eqBwLevel1G:%f", __func__, eqBwLevel0RS, eqBwLevel1RS,
        eqBwLevel0G, eqBwLevel1G);
    return HCCL_SUCCESS;
}
template <
    typename AlgTopoMatch, typename CcuRsAlgTemplateX, typename CcuRsAlgTemplateY, typename CcuGAlgTemplateX,
    typename CcuGAlgTemplateY>
HcclResult
InsV2ReduceOmniPipeExecutor<AlgTopoMatch, CcuRsAlgTemplateX, CcuRsAlgTemplateY, CcuGAlgTemplateX, CcuGAlgTemplateY>::
    OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_DEBUG("[%s] Start.", __func__);
    // 初始化通信域subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    if (resCtx.algHierarchyInfo.infos.empty()) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos[0] is invalid (empty or size < 2).", __func__);
        return HcclResult::HCCL_E_PARA;
    }
    subCommRanks0.push_back(resCtx.algHierarchyInfo.infos[0][0]);
    subCommRanks1.push_back(resCtx.algHierarchyInfo.infos[1][0]);

    bool isRoot = (myRank_ == param.root);
    // 初始化template
    CcuRsAlgTemplateX rsAlgTempX(param, myRank_, subCommRanks0);
    CcuRsAlgTemplateY rsAlgTempY(param, myRank_, subCommRanks1);
    CcuGAlgTemplateX gAlgTempX(param, myRank_, subCommRanks0);
    CcuGAlgTemplateY gAlgTempY(param, myRank_, subCommRanks1);
    rootx = param.root % rankSizeLevel0_;
    rooty = param.root / rankSizeLevel0_;
    gAlgTempX.SetRoot(rankIdxLevel1_ * rankSizeLevel0_ + rootx);
    gAlgTempY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
    // 公共参数初始化
    TemplateDataParams tempAlgParamsCommon;
    tempAlgParamsCommon.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsCommon.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsCommon.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsCommon.buffInfo.hcclBuffSize = resCtx.cclMem.size;
    // 资源模板初始化
    TemplateResource templateResourceCommon;
    TemplateResource templateResourceRsX = templateResourceCommon;
    CHK_PRT_RET(
        resCtx.threads.size() < OMNIPIPE_MIN_THREAD_NUM || resCtx.ccuKernelNum.size() < OMNIPIPE_MIN_CCU_KERNEL_NUM
            || resCtx.ccuKernels.size() < static_cast<size_t>(resCtx.ccuKernelNum[0]) + resCtx.ccuKernelNum[1]
                                              + resCtx.ccuKernelNum[2] + resCtx.ccuKernelNum[3],
        HCCL_ERROR(
            "[%s] resCtx resource not enough. threads.size[%zu], ccuKernelNum.size[%zu], ccuKernels.size[%zu].",
            __func__, resCtx.threads.size(), resCtx.ccuKernelNum.size(), resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);
    templateResourceRsX.threads.push_back(resCtx.threads[0]);
    templateResourceRsX.ccuKernels.insert(
        templateResourceRsX.ccuKernels.end(), resCtx.ccuKernels.begin(),
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
    TemplateResource templateResourceRsY = templateResourceCommon;
    templateResourceRsY.threads.push_back(resCtx.threads[1]);
    templateResourceRsY.ccuKernels.insert(
        templateResourceRsY.ccuKernels.end(), resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    TemplateResource templateResourceGX = templateResourceCommon;
    templateResourceGX.threads.push_back(resCtx.threads[0]);
    templateResourceGX.ccuKernels.insert(
        templateResourceGX.ccuKernels.end(),
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1],
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1] + resCtx.ccuKernelNum[2]);
    TemplateResource templateResourceGY = templateResourceCommon;
    templateResourceGY.threads.push_back(resCtx.threads[1]);
    templateResourceGY.ccuKernels.insert(
        templateResourceGY.ccuKernels.end(),
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1] + resCtx.ccuKernelNum[2],
        resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1] + resCtx.ccuKernelNum[2]
            + resCtx.ccuKernelNum[3]);
    // 1、计算带宽 平均带宽还是总带宽,如果是总带宽这边要处理成平均带宽
    std::vector<double> endpointAttrBwAvgRS;
    std::vector<double> endpointAttrBwAvgG;
    CHK_RET(CalcEndpointBandwidth(endpointAttrBwAvgRS, endpointAttrBwAvgG, param));
    // 2.1 获取每个rank切分的数据量count
    auto allRankSplitData = OmniPipeSplitData(rankSize_, dataCount_, dataTypeSize_);
    for (int i = 0; i < allRankSplitData.size(); i++) {
        HCCL_DEBUG("[%s] rankId[%d], allRankSplitData[%d]:%d", __func__, myRank_, i, allRankSplitData[i]);
    }
    // 2.2 计算loop次数
    maxTmpMemSize_ = resCtx.cclMem.size;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 scratchBoundDataSize = maxTmpMemSize_ / rankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    HCCL_DEBUG(
        "[%s] myRank[%u] transportBoundDataSize[%u] scratchBoundDataSize[%u]", __func__, myRank_,
        transportBoundDataSize, scratchBoundDataSize);
    u64 maxCountPerLoop = std::min(transportBoundDataSize, scratchBoundDataSize) / dataTypeSize_;
    CHK_PRT_RET(maxCountPerLoop == 0, HCCL_ERROR("[%s] maxCountPerLoop is 0", __func__), HCCL_E_INTERNAL);
    HCCL_DEBUG("[%s] myRank[%u] maxCountPerLoop[%u]", __func__, myRank_, maxCountPerLoop);
    u32 loopTimes = allRankSplitData[0] / maxCountPerLoop + ((allRankSplitData[0] % maxCountPerLoop == 0) ? 0 : 1);
    HCCL_DEBUG("[%s] myRank[%u] loopTimes[%u]", __func__, myRank_, loopTimes);
    // 2.3 获取每个rank，每个loop切分的数据量count
    auto multiLoopAllRankSplitData
        = OmniPipeSplitRankDataLoop(allRankSplitData, maxCountPerLoop, loopTimes, dataTypeSize_);
    HCCL_DEBUG("[%s]maxCountPerLoop[%u], loopTimes[%u]", __func__, maxCountPerLoop, loopTimes);
    for (int i = 0; i < multiLoopAllRankSplitData.size(); i++) {
        for (int j = 0; j < multiLoopAllRankSplitData[i].size(); j++) {
            HCCL_DEBUG(
                "rankId[%d],allRankSplitData[%d][%d]:%d multiLoopAllRankSplitData[%d][%d]:%d", myRank_, i, j,
                allRankSplitData[i], i, j, multiLoopAllRankSplitData[i][j]);
        }
    }
    for (int i = 0; i < allRankSplitData.size(); i++) {
        HCCL_DEBUG("[%s]xx rankId[%d], allRankSplitData[%d]:%d", __func__, myRank_, i, allRankSplitData[i]);
    }
    // 3.1 计算n-1次loop的slice信息
    u64 perLoopSize = multiLoopAllRankSplitData[0][0] * dataTypeSize_;
    perLoopSize = dataSize_ > perLoopSize ? perLoopSize : dataSize_;
    HCCL_DEBUG(
        "[%s] perLoopSize[%u] dataSize_[%u] allRankSplitData[%u]", __func__, perLoopSize, dataSize_,
        allRankSplitData[myRank_]);
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, allRankSplitData[myRank_] * dataTypeSize_);
    OmniPipeSliceParam sliceParam;
    sliceParam.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[0], dataTypeSize_);
    sliceParam.dataWholeSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
    sliceParam.levelRankId = {rankIdxLevel0_, rankIdxLevel1_, 0};
    sliceParam.levelRankSize = {rankSizeLevel0_, rankSizeLevel1_, 1};
    std::vector<u64> levelAlgType{1, 0, 1};
    sliceParam.levelAlgType = levelAlgType;
    sliceParam.dataTypeSize = dataTypeSize_;
    sliceParam.opMode = param.opMode;
    sliceParam.engine = CommEngine::COMM_ENGINE_CCU;
    // 3.2 进行一次loop的数据处理
    u64 processedDataCount = 0;
    TemplateDataParams tempRsAlgParamsX = tempAlgParamsCommon;
    TemplateDataParams tempRsAlgParamsY = tempAlgParamsCommon;
    TemplateDataParams tempGAlgParamsX = tempAlgParamsCommon;
    TemplateDataParams tempGAlgParamsY = tempAlgParamsCommon;
    OmniPipeSliceInfo omniPipeSliceInfoRS;
    OmniPipeSliceInfo omniPipeSliceInfoG;
    std::vector<u64> processedDataCountTmp(rankSize_, 0);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        gAlgTempX.SetRoot(rankIdxLevel1_ * rankSizeLevel0_ + rootx);
        gAlgTempY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
        CHK_PRT_RET(
            multiLoopAllRankSplitData.size() <= loop,
            HCCL_ERROR("[InsV2ReduceOmniPipeExecutor][Orchestrate] multiLoopAllRankSplitData.size() <= loop"),
            HCCL_E_PARA);
        if (loop == 0 || !isSameLoop(multiLoopAllRankSplitData[loop - 1], multiLoopAllRankSplitData[loop])) {
            sliceParam.dataSizePerLoop = CalcCountToDataSize(multiLoopAllRankSplitData[loop], dataTypeSize_);
            sliceParam.dataWholeSize = CalcCountToDataSize(allRankSplitData, dataTypeSize_);
            sliceParam.endpointAttrBw = endpointAttrBwAvgRS;
            omniPipeSliceInfoRS = CalcRSOmniPipeSliceInfo(sliceParam);
            sliceParam.endpointAttrBw = endpointAttrBwAvgG;
            omniPipeSliceInfoG = CalcGatherOmniPipeSliceInfo(sliceParam);
            HCCL_DEBUG(
                "[%s] endpointAttrBwAvgRS0:%f, endpointAttrBwAvgRS1:%f, "
                "endpointAttrBwAvgG0:%f, endpointAttrBwAvgG1:%f",
                __func__, endpointAttrBwAvgRS[0], endpointAttrBwAvgRS[1], endpointAttrBwAvgG[0], endpointAttrBwAvgG[1]);
        }
        u64 currDataCount = multiLoopAllRankSplitData[loop][myRank_];
        for (int i = 0; i < omniPipeSliceInfoG.dataSliceLevel0.size(); ++i) {
            for (int j = 0; j < omniPipeSliceInfoG.dataSliceLevel0[i].inputOmniPipeSliceStride.size(); ++j) {
                for (int k = 0; k < omniPipeSliceInfoG.dataSliceLevel0[i].inputOmniPipeSliceStride[j].size(); k++) {
                    HCCL_DEBUG(
                        "[dataSliceLevel0] myRank[%u] inputOmniPipeSliceStride[%u][%u][%u]=[%u] sliceSize=[%u]",
                        myRank_, i, j, k, omniPipeSliceInfoG.dataSliceLevel0[i].inputOmniPipeSliceStride[j][k],
                        omniPipeSliceInfoG.dataSliceLevel0[i].stepSliceSize[j][k]);
                }
            }
        }
        for (int i = 0; i < omniPipeSliceInfoG.dataSliceLevel1.size(); ++i) {
            for (int j = 0; j < omniPipeSliceInfoG.dataSliceLevel1[i].inputOmniPipeSliceStride.size(); ++j) {
                for (int k = 0; k < omniPipeSliceInfoG.dataSliceLevel1[i].inputOmniPipeSliceStride[j].size(); k++) {
                    HCCL_DEBUG(
                        "[dataSliceLevel1] myRank[%u] inputOmniPipeSliceStride[%u][%u][%u]=[%u] sliceSize=[%u]",
                        myRank_, i, j, k, omniPipeSliceInfoG.dataSliceLevel1[i].inputOmniPipeSliceStride[j][k],
                        omniPipeSliceInfoG.dataSliceLevel1[i].stepSliceSize[j][k]);
                }
            }
        }
        HCCL_DEBUG(
            "[%s] dataCount_ %llu, processedDataCount %llu, maxCountPerLoop %llu, currDataCount %llu", __func__,
            dataCount_, processedDataCount, maxCountPerLoop, currDataCount);
        // 4.1 RS的通信步数
        auto level0StepCountRS = omniPipeSliceInfoRS.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] myRank[%u] level0StepCountRS[%u]", __func__, myRank_, level0StepCountRS);
        if (omniPipeSliceInfoRS.isEmpty()) {
            HCCL_DEBUG("[%s] myRank[%u] omniPipeSliceInfo is Empty!", __func__, myRank_);
        } else {
            auto l0StepNum = omniPipeSliceInfoRS.dataSliceLevel0;
            auto l1StepNum = omniPipeSliceInfoRS.dataSliceLevel1;
            HCCL_DEBUG("[%s] myRank[%u] L0 stepNum[%u]", __func__, myRank_, l0StepNum.size());
            HCCL_DEBUG("[%s] myRank[%u] L1 stepNum[%u]", __func__, myRank_, l1StepNum.size());
        }
        // 4.2 RS for内层2d
        // template间同步所需信息计算
        ThreadHandle mainThread = threads_[0];
        std::vector<ThreadHandle> syncThreads{threads_[1]};
        std::vector<u32> notifyIdxesMainToSub{0};
        std::vector<u32> notifyIdxesSubToMain{0};
        for (auto i = 0; i < level0StepCountRS; ++i) {
            // 第一步开始前同步
            CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
            // level0
            GenTemplateAlgParamsByDimData(tempRsAlgParamsX, omniPipeSliceInfoRS.dataSliceLevel0[i], processedDataCount);
            CHK_RET(rsAlgTempX.KernelRun(param, tempRsAlgParamsX, templateResourceRsX));
            // 第一步做完后回到主流做尾同步
            // level1
            GenTemplateAlgParamsByDimData(tempRsAlgParamsY, omniPipeSliceInfoRS.dataSliceLevel1[i], processedDataCount);
            CHK_RET(rsAlgTempY.KernelRun(param, tempRsAlgParamsY, templateResourceRsY));
            CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        }
        // 4.3 GATHER for内层2d
        u32 level0StepCountAG = omniPipeSliceInfoG.dataSliceLevel0.size();
        HCCL_DEBUG("[%s] level0StepCountAG %u", __func__, level0StepCountAG);
        for (u32 i = 0; i < level0StepCountAG; i++) {
            // 初始化机内template param
            // 开始前同步
            CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
            CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                tempGAlgParamsX, omniPipeSliceInfoG.dataSliceLevel0[i], processedDataCount, resCtx, param));
            CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                tempGAlgParamsY, omniPipeSliceInfoG.dataSliceLevel1[i], processedDataCount, resCtx, param));
            gAlgTempX.SetRoot(rankIdxLevel1_ * rankSizeLevel0_ + rootx);
            gAlgTempY.SetRoot(param.root / rankSizeLevel0_ * rankSizeLevel0_ + rankIdxLevel0_);
            HCCL_DEBUG(
                "[%s][KernelRun] myRank[%u] rankIdxLevel1_[%u], rankSizeLevel0_[%u], rootx[%u] param.root[%u]",
                __func__, myRank_, rankIdxLevel1_, rankSizeLevel0_, rootx, param.root);
            // NHR算法时，root的同y轴都需要执行y轴任务
            if (isSameYAxisAsRoot || myRank_ == param.root) {
                gAlgTempY.ifDoTask_ = true;
            } else {
                gAlgTempY.ifDoTask_ = false;
            }
            if (i == 0) { // 第一步
                // 第一步nhr全部卡doTask=true,其他的只有root和root同列的doTask=true
                gAlgTempY.ifDoTask_ = true;
                HCCL_DEBUG("[%s][KernelRun] firstStep start.", __func__);
            } else if (i == level0StepCountAG - 1) { // 最后一步
                HCCL_DEBUG("[%s][KernelRun] lastStep start.", __func__);
                // ----------------第n步----------------
                // 如果当前卡是root的同x轴节点 mesh ccl->usrOut
                // 如果当前卡是root的同y轴节点 nhr ccl->usrOut
                if (isSameYAxisAsRoot && !isRoot) {
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                        tempGAlgParamsY, omniPipeSliceInfoG.dataSliceLevel1[i], processedDataCount, resCtx, param));
                    gAlgTempX.UnsetRoot(myRank_);
                } else if (isSameXAxisAsRoot && !isRoot) {
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                        tempGAlgParamsX, omniPipeSliceInfoG.dataSliceLevel0[i], processedDataCount, resCtx, param));
                } else if (!isRoot) {
                    gAlgTempX.UnsetRoot(myRank_);
                }
            } else { // 中间的所有步
                HCCL_DEBUG("[%s][KernelRun] middleStep start.", __func__);
                // ----------------第2 ~ n-1步----------------
                // 如果当前卡是root的同x轴节点 mesh usrIn->usrOut
                // 如果当前卡是root的同y轴节点 nhr buff->usrOut
                // 如果当前卡是斜对角节点 nhr buff->buff
                if (isSameYAxisAsRoot && !isRoot) {
                    CHK_RET(GenTempAlgParamsHCCLBuff2HCCLBuff(
                        tempGAlgParamsY, omniPipeSliceInfoG.dataSliceLevel1[i], processedDataCount, resCtx, param));
                } else if (isSameXAxisAsRoot && !isRoot) {
                    CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                        tempGAlgParamsX, omniPipeSliceInfoG.dataSliceLevel0[i], processedDataCount, resCtx, param));
                } else if (!isRoot) {
                    CHK_RET(GenTempAlgParamsIn2HCCLBuff(
                        tempGAlgParamsX, omniPipeSliceInfoG.dataSliceLevel0[i], processedDataCount, resCtx, param));
                }
                HCCL_DEBUG("[%s][KernelRun] middleStep end.", __func__);
            }
            HCCL_DEBUG("[%s][KernelRun] start.", __func__);
            CHK_RET(gAlgTempX.KernelRun(param, tempGAlgParamsX, templateResourceGX));
            CHK_RET(gAlgTempY.KernelRun(param, tempGAlgParamsY, templateResourceGY));
            // 第一步做完后回到主流做尾同步
            CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
        }
        // 4.4 Gather本地拷贝
        if (myRank_ == param.root) { // loop偏移 + 外部卡偏移
            HCCL_DEBUG(
                "[%s] Gather local copy start, myRank[%d], currDataCount %llu, processedDataCount %llu"
                " dataSize_ %llu",
                __func__, myRank_, dataCount_, processedDataCount, dataSize_);
            ThreadHandle mainThread = threads_[0];
            std::vector<ThreadHandle> syncThreads{threads_[1]};
            std::vector<u32> notifyIdxesMainToSub{0};
            std::vector<u32> notifyIdxesSubToMain{0};
            u64 rankOffset = 0;
            u64 rankLoopOffset = 0;
            CHK_RET(PreSyncInterThreads(mainThread, syncThreads, notifyIdxesMainToSub));
            for (u32 i = 0; i < rankSize_; i++) {
                if (loop != 0) {
                    processedDataCountTmp[i] = processedDataCountTmp[i] + multiLoopAllRankSplitData[loop - 1][i];
                    HCCL_DEBUG(
                        "processedDataCountTmp[%lu]:[%lu] multiloop[%lu][%lu]:[%lu] ", i, processedDataCountTmp[i],
                        loop, i, multiLoopAllRankSplitData[loop][i]);
                }
            }
            for (u32 i = 0; i < rankSize_; i++) {
                u64 currDataCountTmp = multiLoopAllRankSplitData[loop][i];
                if (currDataCountTmp == 0) {
                    rankOffset += allRankSplitData[i] * dataTypeSize_;
                    continue;
                }
                HCCL_DEBUG("[%s] currDataCountTmp is %llu", __func__, currDataCountTmp);
                TemplateDataParams tempAlgParamLocalCopy = tempAlgParamsCommon;
                tempAlgParamLocalCopy.localCopyFlag = 1;
                tempAlgParamLocalCopy.dataType = dataType_;
                tempAlgParamLocalCopy.buffInfo.outputSize = param.outputSize;
                tempAlgParamLocalCopy.buffInfo.hcclBuff = resCtx.cclMem;
                tempAlgParamLocalCopy.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
                tempAlgParamLocalCopy.inputSliceStride = 0;
                tempAlgParamLocalCopy.outputSliceStride = 0;
                tempAlgParamLocalCopy.count = currDataCountTmp;
                tempAlgParamLocalCopy.sliceSize = currDataCountTmp * dataTypeSize_;
                tempAlgParamLocalCopy.buffInfo.outputPtr = param.outputPtr;
                tempAlgParamLocalCopy.buffInfo.outBuffType = BufferType::OUTPUT;
                tempAlgParamLocalCopy.buffInfo.outBuffBaseOff = rankOffset + processedDataCountTmp[i] * dataTypeSize_;
                tempAlgParamLocalCopy.stepSliceInfo.buffInfo.outBuffBaseOff
                    = rankOffset + processedDataCountTmp[i] * dataTypeSize_;
                if (i == param.root) {
                    tempAlgParamLocalCopy.buffInfo.inputPtr = param.inputPtr;
                    tempAlgParamLocalCopy.buffInfo.inputSize = param.inputSize;
                    tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::INPUT;
                    tempAlgParamLocalCopy.buffInfo.inBuffBaseOff
                        = rankOffset + processedDataCountTmp[i] * dataTypeSize_;
                    tempAlgParamLocalCopy.stepSliceInfo.buffInfo.inBuffBaseOff
                        = rankOffset + processedDataCountTmp[i] * dataTypeSize_;
                } else {
                    tempAlgParamLocalCopy.buffInfo.inputPtr = resCtx.cclMem.addr;
                    tempAlgParamLocalCopy.buffInfo.inputSize = resCtx.cclMem.size;
                    tempAlgParamLocalCopy.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
                    tempAlgParamLocalCopy.buffInfo.inBuffBaseOff = rankLoopOffset;
                    tempAlgParamLocalCopy.stepSliceInfo.buffInfo.inBuffBaseOff = rankLoopOffset;
                }
                HCCL_DEBUG(
                    "[%s] myRank[%u] inBuffBaseOff[%lu] outBuffBaseOff[%lu] sliceSize[%lu] "
                    "processedDataCount[%lu] rankOffset[%lu] rankLoopOffset[%lu]",
                    __func__, myRank_, tempAlgParamLocalCopy.buffInfo.inBuffBaseOff,
                    tempAlgParamLocalCopy.buffInfo.outBuffBaseOff, tempAlgParamLocalCopy.sliceSize, processedDataCount,
                    rankOffset, rankLoopOffset);
                CHK_RET(gAlgTempX.KernelRun(param, tempAlgParamLocalCopy, templateResourceGX));
                rankOffset += allRankSplitData[i] * dataTypeSize_;
                rankLoopOffset += multiLoopAllRankSplitData[loop][i] * dataTypeSize_;
            }
            CHK_RET(PostSyncInterThreads(mainThread, syncThreads, notifyIdxesSubToMain));
            HCCL_DEBUG("[%s] Gather local copy end", __func__);
        }
        processedDataCount += maxCountPerLoop;
    }
    HCCL_DEBUG("[%s][OrchestrateLoop] Endxx.", __func__);
    return HCCL_SUCCESS;
}
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, CcuSchedReducePipeLineMeshNHR, InsV2ReduceOmniPipeExecutor, TopoMatchTwoLevel,
    CcuTempReduceScatterOmniPipeMesh1DMem2Mem, CcuTempReduceScatterOmniPipeNHR1DMem2Mem,
    CcuTempGatherOmniPipeMesh1DMem2Mem, CcuTempGatherOmniPipeNHR1DMem2Mem);
REGISTER_ALG_ATTRS(
    CcuSchedReducePipeLineMeshNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS; topo.maxTopoLevelNum = 1;
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
    };);

#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, CcuMSReducePipeLineMeshNHR, InsV2ReduceOmniPipeExecutor, TopoMatchTwoLevel,
    CcuTempReduceScatterOmniPipeMesh1D, CcuTempReduceScatterOmniPipeNHR1DMem2Mem, CcuTempGatherOmniPipeMesh1DMem2Mem,
    CcuTempGatherOmniPipeNHR1DMem2Mem);
REGISTER_ALG_ATTRS(
    CcuMSReducePipeLineMeshNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS; topo.maxTopoLevelNum = 1;
    topo.isSupportLevel0PcieMix = true; op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_INT8_AND_64BIT;
    // 非全连接UBX场景在通信域初始化时过滤，不满足时该算法不参与选择
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
    });

#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
} // namespace ops_hccl
