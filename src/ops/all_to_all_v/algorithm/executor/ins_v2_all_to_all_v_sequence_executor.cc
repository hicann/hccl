/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_to_all_v_sequence_executor.h"
#include "coll_alg_v2_exec_registry.h"
#include "dev_type.h"
#include "topo_match_two_level.h"
#include "alg_attrs_registry.h"
#include "exec_timeout_manager.h"
#include "hcomm_primitives.h"

namespace ops_hccl {

template <typename AlgTopoMatch>
InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::InsV2AlltoAllVSequenceExecutor()
{}
template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    totalRankSize_ = topoInfo->userRankSize;
    totalStages_ = topoInfo->topoLevelNums;

    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    HCCL_INFO(
        "[InsV2AlltoAllVSequenceExecutor][CalcAlgHierarchyInfoV2] myRank[%u], rankSize[%u], "
        "totalStages[%u]",
        myRank_, rankSize_, totalStages_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
std::vector<std::string> InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::GetDefaultStageAlgos(u32 levelNum)
{
    std::vector<std::string> stageAlgos(levelNum);
    for (u32 k = 0; k < levelNum; k++) {
        stageAlgos[k] = "MeshVStage";
    }
    HCCL_INFO("[InsV2AlltoAllVSequenceExecutor][GetDefaultStageAlgos] levelNum[%u], all MeshVStage", levelNum);
    return stageAlgos;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::BuildStageTemplates(const OpParam& param)
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
                "[InsV2AlltoAllVSequenceExecutor][BuildStageTemplates] Fail to create stage "
                "template[%u], algoName[%s]",
                k, stageAlgos_[k].c_str());
            return HCCL_E_INTERNAL;
        }
        stageTemplates_[k]->SetStageRole(stageIndex, totalStages_);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::BuildMetaStage(const OpParam& param)
{
    // MetaStage双组{intra, inter}直接构造：基类自动推导templateRankSize_ = serverSize * serverNum = N；
    // 不经registry、不注册，不调SetStageRole(stageIndex仅占位传0)
    CHK_PRT_RET(
        algHierarchyInfo_.infos.size() < 2 || algHierarchyInfo_.infos[0].empty()
            || algHierarchyInfo_.infos[0][0].empty() || algHierarchyInfo_.infos[1].empty()
            || algHierarchyInfo_.infos[1][0].empty(),
        HCCL_ERROR(
            "[InsV2AllToAllVSequenceExecutor][BuildMetaStage] expect 2-level topo with non-empty groups, "
            "levelNum[%zu]",
            algHierarchyInfo_.infos.size()),
        HCCL_E_INTERNAL);

    // infos为三维(level->组列表->rank)，TopoMatchTwoLevel每级恰好一个组，取infos[level][0]
    std::vector<std::vector<u32>> metaSubCommRanks = {algHierarchyInfo_.infos[0][0], algHierarchyInfo_.infos[1][0]};
    metaStage_ = std::make_shared<InsTempAlltoAllVHierMetaStage>(param, myRank_, metaSubCommRanks, 0);
    if (metaStage_ == nullptr) {
        HCCL_ERROR("[InsV2AlltoAllVSequenceExecutor][BuildMetaStage] Fail to create meta stage");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    algHierarchyInfo_ = algHierarchyInfo;
    totalStages_ = static_cast<u32>(algHierarchyInfo_.infos.size());
    stageAlgos_ = GetDefaultStageAlgos(totalStages_);
    stageTemplates_.resize(totalStages_);

    u32 maxSlaveThreadNum = 0;
    u32 maxNotifyNumOnMainThread = 0;
    std::vector<u32> mergedNotifyNumPerThread;
    resourceRequest.channels.resize(totalStages_);

    for (u32 k = 0; k < totalStages_; k++) {
        u32 stageIndex = totalStages_ - 1 - k;

        auto& subCommRanks = algHierarchyInfo_.infos[stageIndex];
        stageTemplates_[k] = AlltoAllStageTemplateRegistry::Instance().Create(
            stageAlgos_[k], param, myRank_, subCommRanks, stageIndex);
        if (stageTemplates_[k] == nullptr) {
            HCCL_ERROR(
                "[InsV2AlltoAllVSequenceExecutor][CalcRes] Failed to create stage template[%u], "
                "algoName[%s]",
                k, stageAlgos_[k].c_str());
            return HCCL_E_INTERNAL;
        }
        stageTemplates_[k]->SetStageRole(stageIndex, totalStages_);

        AlgResourceRequest stageResReq;
        CHK_RET(stageTemplates_[k]->CalcRes(comm, param, topoInfo, stageResReq));

        maxSlaveThreadNum = std::max(maxSlaveThreadNum, stageResReq.slaveThreadNum);
        maxNotifyNumOnMainThread = std::max(maxNotifyNumOnMainThread, stageResReq.notifyNumOnMainThread);
        // 逐线程归并各stage的notify需求(取max)，不假设每线程固定1个
        if (stageResReq.notifyNumPerThread.size() > mergedNotifyNumPerThread.size()) {
            mergedNotifyNumPerThread.resize(stageResReq.notifyNumPerThread.size(), 0);
        }
        for (u32 t = 0; t < stageResReq.notifyNumPerThread.size(); t++) {
            mergedNotifyNumPerThread[t] = std::max(mergedNotifyNumPerThread[t], stageResReq.notifyNumPerThread[t]);
        }

        if (!stageResReq.channels.empty()) {
            resourceRequest.channels[stageIndex] = stageResReq.channels[0];
        }

        HCCL_INFO(
            "[InsV2AlltoAllVSequenceExecutor][CalcRes] stage[%u], stageIndex[%u], algoName[%s], "
            "slaveThreadNum[%u], channelNum[%u]",
            k, stageIndex, stageAlgos_[k].c_str(), stageResReq.slaveThreadNum,
            static_cast<u32>(stageResReq.channels.empty() ? 0 : stageResReq.channels[0].size()));
    }

    resourceRequest.slaveThreadNum = maxSlaveThreadNum;
    resourceRequest.notifyNumPerThread = mergedNotifyNumPerThread;
    resourceRequest.notifyNumPerThread.resize(maxSlaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = maxNotifyNumOnMainThread;

    u32 totalChannelNum = 0;
    for (const auto& ch : resourceRequest.channels) {
        if (!ch.empty()) {
            totalChannelNum += static_cast<u32>(ch.size());
        }
    }
    HCCL_INFO(
        "[InsV2AlltoAllVSequenceExecutor][CalcRes] totalStages[%u], slaveThreadNum[%u], "
        "notifyNumOnMainThread[%u], totalChannelNum[%u]",
        totalStages_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, totalChannelNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    totalRankSize_ = resCtx.topoInfo.userRankSize;
    totalStages_ = static_cast<u32>(resCtx.algHierarchyInfo.infos.size());
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    // V/VC统一：alltoallvc在host侧已转为4数组，sendType == recvType
    dataType_ = param.all2AllVDataDes.sendType;
    // 防御性校验：入口层已校验sendType，此处避免非法值越界查表/后续除零
    CHK_PRT_RET(
        dataType_ >= HCCL_DATA_TYPE_RESERVED || HCCL_SIZE_TABLE[dataType_] == 0,
        HCCL_ERROR("[InsV2AlltoAllVSequenceExecutor][Orchestrate] invalid dataType[%u]", static_cast<u32>(dataType_)),
        HCCL_E_PARA);
    dataTypeSize_ = HCCL_SIZE_TABLE[dataType_];

    CHK_RET(BuildStageTemplates(param));
    CHK_RET(BuildMetaStage(param));

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    return OrchestrateLoop(param, resCtx);
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::SetStageParams(
    TemplateDataParams& tempAlgParams, u32 stageIndex, u32 totalStages, u64 loop, u64 currDataCount,
    u64 processedDataCount, u64 maxDataCountPerLoop, u32 totalRankSize, u32 stageRankSize)
{
    (void)loop;
    (void)totalRankSize;

    tempAlgParams.count = currDataCount;
    tempAlgParams.sliceSize = currDataCount * dataTypeSize_;
    // V版：4数组传原始值(sdispls/rdispls为元素单位)，轮内余量按processedDataCount计算
    tempAlgParams.processedDataCount = processedDataCount;
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
        "[InsV2AlltoAllVSequenceExecutor][SetStageParams] stageIndex[%u], count[%llu], "
        "sliceSize[%llu], inputSliceStride[%llu], inBuffType[%d], outBuffType[%d]",
        stageIndex, currDataCount, tempAlgParams.sliceSize, tempAlgParams.inputSliceStride,
        static_cast<int>(tempAlgParams.buffInfo.inBuffType), static_cast<int>(tempAlgParams.buffInfo.outBuffType));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::FillTemplateResource(
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
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::FillMetaTemplateResource(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateResource& templateRes)
{
    (void)param;
    templateRes.threads = threads_;
    // 合并两级channels(intra+inter)：rank全局唯一，MetaStage按remoteRank直接查找
    for (const auto& stageChannels : remoteRankToChannelInfo_) {
        for (const auto& entry : stageChannels) {
            templateRes.channels[entry.first] = entry.second;
        }
    }
    templateRes.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateRes.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::RunMetaPhase(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, u64& maxCount)
{
    TemplateDataParams metaParams;
    metaParams.dataType = dataType_;
    metaParams.buffInfo.hcclBuff = resCtx.cclMem;
    metaParams.buffInfo.hcclBuffSize = resCtx.cclMem.size;
    metaParams.buffInfo.inputPtr = param.inputPtr;
    metaParams.buffInfo.outputPtr = param.outputPtr;

    TemplateResource metaRes;
    CHK_RET(FillMetaTemplateResource(param, resCtx, metaRes));

    CHK_RET(metaStage_->KernelRun(param, metaParams, metaRes));

    CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
    CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
    for (const auto& thread : threads_) {
        CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, ExecTimeoutManager::Instance().GetExecTimeout())));
    }

    // 扫描cclbuff头部的全矩阵(N*N*u64，AICPU直读)，取全局最大单点count；矩阵随后被数据轮覆盖
    const u64* matrix = reinterpret_cast<const u64*>(resCtx.cclMem.addr);
    u64 entryNum = static_cast<u64>(totalRankSize_) * totalRankSize_;
    maxCount = 0;
    for (u64 i = 0; i < entryNum; i++) {
        maxCount = std::max(maxCount, matrix[i]);
    }

    HCCL_INFO("[InsV2AlltoAllVSequenceExecutor][RunMetaPhase] maxCount[%llu]", maxCount);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    // V/VC统一：4数组(各totalRankSize_个u64)已在param.varData就绪，AICPU直读
    CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
    CHK_PTR_NULL(param.all2AllVDataDes.recvCounts);
    CHK_PTR_NULL(param.all2AllVDataDes.sdispls);
    CHK_PTR_NULL(param.all2AllVDataDes.rdispls);
    const u64* sendCountsPtr = reinterpret_cast<const u64*>(param.all2AllVDataDes.sendCounts);
    const u64* recvCountsPtr = reinterpret_cast<const u64*>(param.all2AllVDataDes.recvCounts);
    const u64* sdisplsPtr = reinterpret_cast<const u64*>(param.all2AllVDataDes.sdispls);
    const u64* rdisplsPtr = reinterpret_cast<const u64*>(param.all2AllVDataDes.rdispls);
    std::vector<u64> sendCountsVec(sendCountsPtr, sendCountsPtr + totalRankSize_);
    std::vector<u64> recvCountsVec(recvCountsPtr, recvCountsPtr + totalRankSize_);
    std::vector<u64> sdisplsVec(sdisplsPtr, sdisplsPtr + totalRankSize_);
    std::vector<u64> rdisplsVec(rdisplsPtr, rdisplsPtr + totalRankSize_);

    u64 cclBuffSize = resCtx.cclMem.size;

    // chunk = cclbuff单slot容量(元素数)：每轮每slot最多搬运chunk个元素
    u64 chunkBytes = cclBuffSize / totalRankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    CHK_PRT_RET(
        chunkBytes == 0,
        HCCL_ERROR(
            "[InsV2AlltoAllVSequenceExecutor][OrchestrateLoop] chunkBytes is 0, cclBuffSize[%llu], "
            "totalRankSize[%u]",
            cclBuffSize, totalRankSize_),
        HCCL_E_INTERNAL);
    u64 chunk = chunkBytes / dataTypeSize_;
    CHK_PRT_RET(
        chunk == 0,
        HCCL_ERROR(
            "[InsV2AlltoAllVSequenceExecutor][OrchestrateLoop] chunk is 0, chunkBytes[%llu], "
            "dataTypeSize[%llu]",
            chunkBytes, dataTypeSize_),
        HCCL_E_INTERNAL);

    // Phase 0：MetaStage两级收集sendCounts全矩阵(overlay在cclbuff头部)，扫描取maxCount
    u64 maxCount = 0;
    CHK_RET(RunMetaPhase(param, resCtx, maxCount));

    u64 roundCount = (maxCount + chunk - 1) / chunk; // maxCount为0时roundCount为0，无数据搬运

    HCCL_INFO(
        "[InsV2AlltoAllVSequenceExecutor][OrchestrateLoop] maxCount[%llu], cclBuffSize[%llu], "
        "chunk[%llu], roundCount[%llu], totalStages[%u], totalRankSize[%u]",
        maxCount, cclBuffSize, chunk, roundCount, totalStages_, totalRankSize_);

    for (u64 loop = 0; loop < roundCount; loop++) {
        u64 processedDataCount = loop * chunk;

        TemplateDataParams tempAlgParams;
        tempAlgParams.dataType = dataType_;
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParams.buffInfo.hcclBuffSize = cclBuffSize;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.outputPtr = param.outputPtr;
        // 4数组传原始值(元素单位)，轮偏移由SetStageParams的processedDataCount/inBuffBaseOff承载
        tempAlgParams.sendCounts = sendCountsVec;
        tempAlgParams.recvCounts = recvCountsVec;
        tempAlgParams.sdispls = sdisplsVec;
        tempAlgParams.rdispls = rdisplsVec;

        for (u32 k = 0; k < totalStages_; k++) {
            u32 stageIndex = totalStages_ - 1 - k;
            u32 stageRankSize = stageTemplates_[k]->GetTemplateRankSize();

            SetStageParams(
                tempAlgParams, stageIndex, totalStages_, loop, chunk, processedDataCount, chunk, totalRankSize_,
                stageRankSize);

            TemplateResource templateRes;
            FillTemplateResource(param, resCtx, templateRes, stageIndex);

            CHK_RET(stageTemplates_[k]->KernelRun(param, tempAlgParams, templateRes));
        }
    }

    HCCL_INFO("[InsV2AlltoAllVSequenceExecutor][OrchestrateLoop] done");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
std::vector<CostModelParam> InsV2AlltoAllVSequenceExecutor<AlgTopoMatch>::CalcCostCoeff(
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

// alltoallv与alltoallvc共用同一executor(V/VC参数在host侧已统一为4数组)
REGISTER_EXECUTOR_BY_TOPO(
    HcclCMDType::HCCL_CMD_ALLTOALLV, AicpuAllToAllVSequenceMeshMesh, InsV2AlltoAllVSequenceExecutor, TopoMatchTwoLevel);

REGISTER_EXECUTOR_BY_TOPO(
    HcclCMDType::HCCL_CMD_ALLTOALLVC, AicpuAllToAllVCSequenceMeshMesh, InsV2AlltoAllVSequenceExecutor,
    TopoMatchTwoLevel);

REGISTER_ALG_ATTRS(
    AicpuAllToAllVSequenceMeshMesh, topo.minTopoLevelNum = 2; topo.maxTopoLevelNum = 3;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->level0Symmetric;
    };);

REGISTER_ALG_ATTRS(
    AicpuAllToAllVCSequenceMeshMesh, topo.minTopoLevelNum = 2; topo.maxTopoLevelNum = 3;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    // 同AicpuAllToAllVSequenceMeshMesh：拒绝非对称两级拓扑
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->level0Symmetric;
    };);

} // namespace ops_hccl
