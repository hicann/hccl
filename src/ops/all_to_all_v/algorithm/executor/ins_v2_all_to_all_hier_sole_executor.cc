/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_to_all_hier_sole_executor.h"
#include "coll_alg_v2_exec_registry.h"
#include "dev_type.h"
#include "topo_match_nlevel.h"
#include "alg_attrs_registry.h"

namespace ops_hccl {

template <typename AlgTopoMatch>
InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::InsV2AlltoAllHierSoleExecutor()
{}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    totalRankSize_ = topoInfo->userRankSize;
    totalStages_ = topoInfo->topoLevelNums;

    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    HCCL_INFO(
        "[InsV2AlltoAllHierSoleExecutor][CalcAlgHierarchyInfo] myRank[%u], rankSize[%u], "
        "totalStages[%u]",
        myRank_, rankSize_, totalStages_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
std::vector<std::string> InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::GetDefaultStageAlgos(u32 levelNum)
{
    std::vector<std::string> stageAlgos(levelNum);
    for (u32 k = 0; k < levelNum; k++) {
        stageAlgos[k] = "MeshStage";
    }
    HCCL_INFO("[InsV2AlltoAllHierSoleExecutor][GetDefaultStageAlgos] levelNum[%u], all MeshStage", levelNum);
    return stageAlgos;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::BuildStageTemplates(const OpParam& param)
{
    stageAlgos_ = GetDefaultStageAlgos(totalStages_);
    stageTemplates_.resize(totalStages_);

    for (u32 k = 0; k < totalStages_; k++) {
        u32 stageIndex = totalStages_ - 1 - k;

        auto& subCommRanks = algHierarchyInfo_.infos[stageIndex];
        stageTemplates_[k] = AlltoAllStageTemplateRegistry::Instance().Create(
            stageAlgos_[k], param, myRank_, subCommRanks, stageIndex);
        if (stageTemplates_[k] == nullptr) {
            HCCL_ERROR(
                "[InsV2AlltoAllHierSoleExecutor][BuildStageTemplates] Fail to create stage "
                "template[%u], algoName[%s]",
                k, stageAlgos_[k].c_str());
            return HCCL_E_INTERNAL;
        }
        stageTemplates_[k]->SetStageRole(stageIndex, totalStages_);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    algHierarchyInfo_ = algHierarchyInfo;
    totalStages_ = static_cast<u32>(algHierarchyInfo_.infos.size());
    stageAlgos_ = GetDefaultStageAlgos(totalStages_);
    stageTemplates_.resize(totalStages_);

    u32 maxSlaveThreadNum = 0;
    u32 maxNotifyNumOnMainThread = 0;
    resourceRequest.channels.resize(totalStages_);

    for (u32 k = 0; k < totalStages_; k++) {
        u32 stageIndex = totalStages_ - 1 - k;

        auto& subCommRanks = algHierarchyInfo_.infos[stageIndex];
        stageTemplates_[k] = AlltoAllStageTemplateRegistry::Instance().Create(
            stageAlgos_[k], param, myRank_, subCommRanks, stageIndex);
        if (stageTemplates_[k] == nullptr) {
            HCCL_ERROR(
                "[InsV2AlltoAllHierSoleExecutor][CalcRes] Failed to create stage template[%u], "
                "algoName[%s]",
                k, stageAlgos_[k].c_str());
            return HCCL_E_INTERNAL;
        }
        stageTemplates_[k]->SetStageRole(stageIndex, totalStages_);

        AlgResourceRequest stageResReq;
        CHK_RET(stageTemplates_[k]->CalcRes(comm, param, topoInfo, stageResReq));

        maxSlaveThreadNum = std::max(maxSlaveThreadNum, stageResReq.slaveThreadNum);
        maxNotifyNumOnMainThread = std::max(maxNotifyNumOnMainThread, stageResReq.notifyNumOnMainThread);

        if (!stageResReq.channels.empty()) {
            resourceRequest.channels[stageIndex] = stageResReq.channels[0];
        }

        HCCL_INFO(
            "[InsV2AlltoAllHierSoleExecutor][CalcRes] stage[%u], stageIndex[%u], algoName[%s], "
            "slaveThreadNum[%u], channelNum[%u]",
            k, stageIndex, stageAlgos_[k].c_str(), stageResReq.slaveThreadNum,
            static_cast<u32>(stageResReq.channels.empty() ? 0 : stageResReq.channels[0].size()));
    }

    resourceRequest.slaveThreadNum = maxSlaveThreadNum;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(maxSlaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = maxNotifyNumOnMainThread;

    HCCL_INFO(
        "[InsV2AlltoAllHierSoleExecutor][CalcRes] totalStages[%u], slaveThreadNum[%u], "
        "notifyNumOnMainThread[%u]",
        totalStages_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult
InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    totalRankSize_ = resCtx.topoInfo.userRankSize;
    totalStages_ = static_cast<u32>(resCtx.algHierarchyInfo.infos.size());
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = HCCL_SIZE_TABLE[dataType_];

    CHK_RET(BuildStageTemplates(param));

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    return OrchestrateLoop(param, resCtx);
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::SetStageParams(
    TemplateDataParams& tempAlgParams, u32 stageIndex, u32 totalStages, u64 loop, u64 currDataCount,
    u64 processedDataCount, u64 maxDataCountPerLoop, u32 totalRankSize, u32 stageRankSize)
{
    (void)loop;
    (void)totalRankSize;

    tempAlgParams.count = currDataCount;
    tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

    u32 otherDimRankSize = (stageRankSize > 0) ? (totalRankSize_ / stageRankSize) : 1;
    tempAlgParams.inputSliceStride = maxDataCountPerLoop * otherDimRankSize * dataTypeSize_;
    tempAlgParams.outputSliceStride = static_cast<u64>(otherDimRankSize);

    if (stageIndex == totalStages - 1) {
        tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
        tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    } else if (stageIndex == 0) {
        tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    } else {
        tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    }

    HCCL_DEBUG(
        "[InsV2AlltoAllHierSoleExecutor][SetStageParams] stageIndex[%u], count[%llu], "
        "sliceSize[%llu], inputSliceStride[%llu], inBuffType[%d], outBuffType[%d]",
        stageIndex, currDataCount, tempAlgParams.sliceSize, tempAlgParams.inputSliceStride,
        static_cast<int>(tempAlgParams.buffInfo.inBuffType), static_cast<int>(tempAlgParams.buffInfo.outBuffType));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::FillTemplateResource(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateResource& templateRes, u32 stageIndex)
{
    (void)param;
    templateRes.threads = threads_;
    if (stageIndex < remoteRankToChannelInfo_.size()) {
        templateRes.channels = remoteRankToChannelInfo_[stageIndex];
    }
    templateRes.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateRes.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
    u64 totalCount = *(reinterpret_cast<const u64*>(param.all2AllVDataDes.sendCounts));
    u64 cclBuffSize = resCtx.cclMem.size;

    u64 maxDataSizePerLoop = cclBuffSize / totalRankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    CHK_PRT_RET(
        maxDataSizePerLoop == 0,
        HCCL_ERROR(
            "[InsV2AlltoAllHierSoleExecutor][OrchestrateLoop] maxDataSizePerLoop is 0, "
            "cclBuffSize[%llu], totalRankSize[%u], dataTypeSize[%llu]",
            cclBuffSize, totalRankSize_, dataTypeSize_),
        HCCL_E_INTERNAL);

    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    u64 loopTimes = totalCount / maxDataCountPerLoop + static_cast<u64>(totalCount % maxDataCountPerLoop != 0);
    u64 processedDataCount = 0;

    HCCL_INFO(
        "[InsV2AlltoAllHierSoleExecutor][OrchestrateLoop] totalCount[%llu], cclBuffSize[%llu], "
        "maxDataCountPerLoop[%llu], loopTimes[%llu], totalStages[%u], totalRankSize[%u]",
        totalCount, cclBuffSize, maxDataCountPerLoop, loopTimes, totalStages_, totalRankSize_);

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? (totalCount - processedDataCount) : maxDataCountPerLoop;

        TemplateDataParams tempAlgParams;
        tempAlgParams.dataType = dataType_;
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParams.buffInfo.hcclBuffSize = cclBuffSize;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.outputPtr = param.outputPtr;

        for (u32 k = 0; k < totalStages_; k++) {
            u32 stageIndex = totalStages_ - 1 - k;
            u32 stageRankSize = stageTemplates_[k]->GetTemplateRankSize();

            SetStageParams(
                tempAlgParams, stageIndex, totalStages_, loop, currDataCount, processedDataCount, maxDataCountPerLoop,
                totalRankSize_, stageRankSize);

            TemplateResource templateRes;
            FillTemplateResource(param, resCtx, templateRes, stageIndex);

            CHK_RET(stageTemplates_[k]->KernelRun(param, tempAlgParams, templateRes));
        }
        processedDataCount += currDataCount;
    }

    HCCL_INFO("[InsV2AlltoAllHierSoleExecutor][OrchestrateLoop] done");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
std::vector<CostModelParam> InsV2AlltoAllHierSoleExecutor<AlgTopoMatch>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    // 临时占位值，后续由MDE提供实测标定数据
    // A: 跨卡传输时间系数, B: 本地传输时间系数, C: 基本时延常数项, D: launch开销系数
    return {CostModelParam{1000.0f, 0.0f, 1000.0f, 0.0f}};
}

REGISTER_EXECUTOR_BY_TOPO(
    HcclCMDType::HCCL_CMD_ALLTOALL, AicpuAllToAllSoleMeshHier, InsV2AlltoAllHierSoleExecutor, TopoMatchNLevel);

REGISTER_ALG_ATTRS(AicpuAllToAllSoleMeshHier, topo.minTopoLevelNum = 2; topo.maxTopoLevelNum = 4;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;);

} // namespace ops_hccl
