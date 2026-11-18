/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_to_all_sequence_executor.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu/hier/ccu_alltoall_stage_template_registry.h"
#include "ccu/hier/ccu_temp_all_to_all_hier_last_stage.h"
#include "ccu/hier/ccu_temp_all_to_all_hier_first_stage.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
#include "coll_alg_v2_exec_registry.h"
#include "dev_type.h"
#include "topo_match_two_level.h"
#include "alg_attrs_registry.h"

namespace ops_hccl {

constexpr u32 A2A_HIER_UNROLL_MIN = 2;
constexpr u32 A2A_SEQ_MAX_RANK_SIZE = 1024;
constexpr u32 A2A_SEQ_CCU_MAX_PEER_NUM = 128;          // CCU 单 kernel 通道数上限
constexpr u32 A2A_SEQ_CC_KERNEL_MAX_INTER = 2;         // CCU 框间段 kernel 上界（2+6 双 kernel）
constexpr u32 A2A_SEQ_CC_KERNEL_INTRA = 1;             // CCU 板内段 kernel 数
constexpr u32 A2A_SEQ_AICPU_KERNEL_NUM_PER_STAGE = 12; // AICPU 每段 kernel 任务数
constexpr float A2A_SEQ_B_CONST = 2e-6f;               // AICPU 序列编排常量（AG seqBConst 惯例）

static CostModelParam
CalcA2ASegCost(u32 groupSize, float dataRatio, CommTopo netType, int portNum, bool isPod, bool isCcu, int kernelNum)
{
    float A = 0.0f;
    float B = 0.0f;
    float C = 0.0f;
    float D = 0.0f;
    CostModelManager::Global()->CalcMeshParam(dataRatio, netType, portNum, groupSize, A, isPod);
    CostModelManager::Global()->CalcLatencyParams(kernelNum, isCcu ? EngineType::CCU : EngineType::AICPU, C);
    if (!isCcu) {
        CostModelManager::Global()->CalcLocalCopyParams(dataRatio, EngineType::AICPU, B);
        C += A2A_SEQ_B_CONST;
        int taskNum
            = CostModelManager::CalcTransTaskNum(groupSize) + CostModelManager::CalcSyncTaskNum(groupSize) * 2 + 1;
        CostModelManager::Global()->CalcLaunchParams(taskNum, EngineType::AICPU, D);
    }
    return {A, B, C, D};
}

template <typename AlgTopoMatch>
InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::InsV2AlltoAllSequenceExecutor()
{}
template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    totalRankSize_ = topoInfo->userRankSize;

    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    algHierarchyInfo_ = algHierarchyInfo;
    // totalStages_ 取算法级数（infos 维度）：三级物理拓扑经 TwoLevel 压缩后恒为 2
    totalStages_ = static_cast<u32>(algHierarchyInfo_.infos.size());
    HCCL_INFO(
        "[InsV2AlltoAllSequenceExecutor][CalcAlgHierarchyInfoV2] myRank[%u], rankSize[%u], "
        "topoLevelNums[%u], algoStages[%u]%s",
        myRank_, rankSize_, topoInfo->topoLevelNums, totalStages_,
        (topoInfo->topoLevelNums != totalStages_) ? " (flattened)" : "");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::GetStageAlgos(
    const OpParam& param, u32 levelNum, std::vector<std::string>& stageAlgos)
{
    CHK_PRT_RET(
        levelNum < 2 || algHierarchyInfo_.infos.size() < levelNum,
        HCCL_ERROR(
            "[InsV2AlltoAllSequenceExecutor][%s] invalid stage plan: levelNum[%u] (expect >= 2), hierInfos[%zu], "
            "engine[%d].",
            __func__, levelNum, algHierarchyInfo_.infos.size(), static_cast<int>(param.engine)),
        HcclResult::HCCL_E_INTERNAL);
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        constexpr u32 CCU_SEQUENCE_LEVEL_NUM = 2;
        CHK_PRT_RET(
            levelNum != CCU_SEQUENCE_LEVEL_NUM,
            HCCL_ERROR(
                "[InsV2AlltoAllSequenceExecutor][%s] CCU engine only supports 2-level sequence, levelNum[%u].",
                __func__, levelNum),
            HcclResult::HCCL_E_INTERNAL);
        stageAlgos = {"FirstMeshStage", "LastMeshStage"};
        HCCL_INFO(
            "[InsV2AlltoAllSequenceExecutor][%s] CCU engine, levelNum[%u], firstStage[%s], lastStage[%s]", __func__,
            levelNum, stageAlgos[0].c_str(), stageAlgos[1].c_str());
        return HCCL_SUCCESS;
    }
    stageAlgos.assign(levelNum, "MeshStage");
    HCCL_INFO("[InsV2AlltoAllSequenceExecutor][%s] levelNum[%u], stage[%s]", __func__, levelNum, stageAlgos[0].c_str());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::CreateStageTemplate(u32 k, const OpParam& param)
{
    u32 stageIndex = totalStages_ - 1 - k;
    auto& subCommRanks = algHierarchyInfo_.infos[stageIndex];
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        ccuStageTemplates_[k] = CcuAlltoAllStageTemplateRegistry::Instance().Create(
            stageAlgos_[k], param, myRank_, subCommRanks, stageIndex);
        if (ccuStageTemplates_[k] == nullptr) {
            HCCL_ERROR(
                "[InsV2AlltoAllSequenceExecutor][%s] Fail to create CCU stage template[%u], algoName[%s]", __func__, k,
                stageAlgos_[k].c_str());
            return HCCL_E_INTERNAL;
        }
        ccuStageTemplates_[k]->SetStageRole(stageIndex, totalStages_);
        return HCCL_SUCCESS;
    }
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
    stageTemplates_[k]
        = AlltoAllStageTemplateRegistry::Instance().Create(stageAlgos_[k], param, myRank_, subCommRanks, stageIndex);
    if (stageTemplates_[k] == nullptr) {
        HCCL_ERROR(
            "[InsV2AlltoAllSequenceExecutor][%s] Fail to create stage template[%u], algoName[%s]", __func__, k,
            stageAlgos_[k].c_str());
        return HCCL_E_INTERNAL;
    }
    stageTemplates_[k]->SetStageRole(stageIndex, totalStages_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
u32 InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::GetStageTemplateRankSize(u32 k, const OpParam& param) const
{
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        return ccuStageTemplates_[k]->GetTemplateRankSize();
    }
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
    return stageTemplates_[k]->GetTemplateRankSize();
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::RunStageTemplate(
    u32 k, const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateRes)
{
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        return ccuStageTemplates_[k]->KernelRun(param, tempAlgParams, templateRes);
    }
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
    return stageTemplates_[k]->KernelRun(param, tempAlgParams, templateRes);
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::BuildStageTemplates(const OpParam& param)
{
    CHK_RET(GetStageAlgos(param, totalStages_, stageAlgos_));
    stageTemplates_.resize(totalStages_);
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
    ccuStageTemplates_.resize(totalStages_);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

    for (u32 k = 0; k < totalStages_; k++) {
        CHK_RET(CreateStageTemplate(k, param));
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    algHierarchyInfo_ = algHierarchyInfo;
    totalStages_ = static_cast<u32>(algHierarchyInfo_.infos.size());
    CHK_RET(GetStageAlgos(param, totalStages_, stageAlgos_));
    stageTemplates_.resize(totalStages_);
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
    ccuStageTemplates_.resize(totalStages_);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

    u32 maxSlaveThreadNum = 0;
    u32 maxNotifyNumOnMainThread = 0;

    for (u32 k = 0; k < totalStages_; k++) {
        CHK_RET(CreateStageTemplate(k, param));
        u32 stageIndex = totalStages_ - 1 - k;

        AlgResourceRequest stageResReq;
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
        if (param.engine == CommEngine::COMM_ENGINE_CCU) {
            CHK_RET(ccuStageTemplates_[k]->CalcRes(comm, param, topoInfo, stageResReq));
        } else
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
        {
            CHK_RET(stageTemplates_[k]->CalcRes(comm, param, topoInfo, stageResReq));
        }

        maxSlaveThreadNum = std::max(maxSlaveThreadNum, stageResReq.slaveThreadNum);
        maxNotifyNumOnMainThread = std::max(maxNotifyNumOnMainThread, stageResReq.notifyNumOnMainThread);

        if (param.engine == CommEngine::COMM_ENGINE_CCU) {
            resourceRequest.ccuKernelInfos.insert(
                resourceRequest.ccuKernelInfos.end(), stageResReq.ccuKernelInfos.begin(),
                stageResReq.ccuKernelInfos.end());
            resourceRequest.ccuKernelNum.push_back(static_cast<u32>(stageResReq.ccuKernelInfos.size()));
            if (stageResReq.dieSplitRatio > 0.0) {
                resourceRequest.dieSplitRatio = stageResReq.dieSplitRatio;
            }
        } else {
            resourceRequest.channels.resize(totalStages_);
            if (!stageResReq.channels.empty()) {
                resourceRequest.channels[stageIndex] = stageResReq.channels[0];
            }
        }

        HCCL_INFO(
            "[InsV2AlltoAllSequenceExecutor][CalcRes] stage[%u], stageIndex[%u], algoName[%s], "
            "slaveThreadNum[%u], channelNum[%u], ccuKernelNum[%zu]",
            k, stageIndex, stageAlgos_[k].c_str(), stageResReq.slaveThreadNum,
            static_cast<u32>(stageResReq.channels.empty() ? 0 : stageResReq.channels[0].size()),
            stageResReq.ccuKernelInfos.size());
    }

    resourceRequest.slaveThreadNum = maxSlaveThreadNum;
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(maxSlaveThreadNum, 1);
    resourceRequest.notifyNumOnMainThread = maxNotifyNumOnMainThread;

    u32 totalChannelNum = 0;
    for (const auto& ch : resourceRequest.channels) {
        if (!ch.empty()) {
            totalChannelNum += static_cast<u32>(ch.size());
        }
    }
    HCCL_INFO(
        "[InsV2AlltoAllSequenceExecutor][CalcRes] totalStages[%u], slaveThreadNum[%u], "
        "notifyNumOnMainThread[%u], totalChannelNum[%u], totalCcuKernelNum[%zu]",
        totalStages_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, totalChannelNum,
        resourceRequest.ccuKernelInfos.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::RestoreStageCcuKernels(const AlgResourceCtxSerializable& resCtx)
{
    CHK_PRT_RET(
        resCtx.ccuKernelNum.size() < totalStages_,
        HCCL_ERROR(
            "[InsV2AlltoAllSequenceExecutor][%s] ccuKernelNum size[%zu] < totalStages[%u].", __func__,
            resCtx.ccuKernelNum.size(), totalStages_),
        HcclResult::HCCL_E_INTERNAL);
    stageCcuKernels_.assign(totalStages_, {});
    u32 kernelOffset = 0;
    for (u32 k = 0; k < totalStages_; k++) {
        u32 kernelCnt = resCtx.ccuKernelNum[k];
        CHK_PRT_RET(
            kernelOffset + kernelCnt > resCtx.ccuKernels.size(),
            HCCL_ERROR(
                "[InsV2AlltoAllSequenceExecutor][%s] stage[%u] kernel slice out of range: offset[%u], cnt[%u], "
                "total[%zu].",
                __func__, k, kernelOffset, kernelCnt, resCtx.ccuKernels.size()),
            HcclResult::HCCL_E_INTERNAL);
        if (kernelCnt > 0) {
            u32 stageIndex = totalStages_ - 1 - k;
            stageCcuKernels_[stageIndex].assign(
                resCtx.ccuKernels.begin() + kernelOffset, resCtx.ccuKernels.begin() + kernelOffset + kernelCnt);
        }
        kernelOffset += kernelCnt;
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult
InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
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

    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreStageCcuKernels(resCtx));
    } else {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    return OrchestrateLoop(param, resCtx);
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::SetStageParams(
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
        "[InsV2AlltoAllSequenceExecutor][SetStageParams] stageIndex[%u], count[%llu], "
        "sliceSize[%llu], inputSliceStride[%llu], inBuffType[%d], outBuffType[%d]",
        stageIndex, currDataCount, tempAlgParams.sliceSize, tempAlgParams.inputSliceStride,
        static_cast<int>(tempAlgParams.buffInfo.inBuffType), static_cast<int>(tempAlgParams.buffInfo.outBuffType));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::FillTemplateResource(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateResource& templateRes, u32 stageIndex)
{
    templateRes.threads = threads_;
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        CHK_PRT_RET(
            stageIndex >= stageCcuKernels_.size(),
            HCCL_ERROR(
                "[InsV2AlltoAllSequenceExecutor][%s] stageIndex[%u] >= ccuKernels size[%zu].", __func__, stageIndex,
                stageCcuKernels_.size()),
            HcclResult::HCCL_E_INTERNAL);
        templateRes.ccuKernels = stageCcuKernels_[stageIndex];
    } else {
        if (stageIndex < remoteRankToChannelInfo_.size()) {
            templateRes.channels = remoteRankToChannelInfo_[stageIndex];
        }
    }
    templateRes.dieSplitRatio = resCtx.dieSplitRatio;
    templateRes.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    templateRes.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
    u64 totalCount = *(reinterpret_cast<const u64*>(param.all2AllVDataDes.sendCounts));
    u64 cclBuffSize = resCtx.cclMem.size;

    u64 maxDataSizePerLoop = cclBuffSize / totalRankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    CHK_PRT_RET(
        maxDataSizePerLoop == 0,
        HCCL_ERROR(
            "[InsV2AlltoAllSequenceExecutor][%s] maxDataSizePerLoop is 0, "
            "cclBuffSize[%llu], totalRankSize[%u], dataTypeSize[%llu]",
            __func__, cclBuffSize, totalRankSize_, dataTypeSize_),
        HCCL_E_INTERNAL);

    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    u64 loopTimes = totalCount / maxDataCountPerLoop + static_cast<u64>(totalCount % maxDataCountPerLoop != 0);
    u64 processedDataCount = 0;

    HCCL_INFO(
        "[InsV2AlltoAllSequenceExecutor][%s] totalCount[%llu], cclBuffSize[%llu], "
        "maxDataCountPerLoop[%llu], loopTimes[%llu], totalStages[%u], totalRankSize[%u]",
        __func__, totalCount, cclBuffSize, maxDataCountPerLoop, loopTimes, totalStages_, totalRankSize_);

    std::vector<std::vector<CcuKernelSubmitInfo>> firstLoopSubmitInfos(totalStages_);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? (totalCount - processedDataCount) : maxDataCountPerLoop;

        TemplateDataParams tempAlgParams;
        tempAlgParams.dataType = dataType_;
        tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParams.buffInfo.hcclBuffSize = cclBuffSize;
        tempAlgParams.buffInfo.inputPtr = param.inputPtr;
        tempAlgParams.buffInfo.outputPtr = param.outputPtr;
        tempAlgParams.buffInfo.inputSize = param.inputSize;
        tempAlgParams.buffInfo.outputSize = param.outputSize;

        for (u32 k = 0; k < totalStages_; k++) {
            u32 stageIndex = totalStages_ - 1 - k;
            u32 stageRankSize = GetStageTemplateRankSize(k, param);

            SetStageParams(
                tempAlgParams, stageIndex, totalStages_, loop, currDataCount, processedDataCount, maxDataCountPerLoop,
                totalRankSize_, stageRankSize);

            TemplateResource templateRes;
            CHK_RET(FillTemplateResource(param, resCtx, templateRes, stageIndex));

            CHK_RET(RunStageTemplate(k, param, tempAlgParams, templateRes));
            if (loop == 0) {
                firstLoopSubmitInfos[k] = templateRes.submitInfos;
            }
        }
        processedDataCount += currDataCount;
    }

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, firstLoopSubmitInfos, resCtx.notifyNumOnMainThread));
    }
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

    HCCL_INFO("[InsV2AlltoAllSequenceExecutor][%s] done", __func__);
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::FastLaunchSaveCtx(
    const OpParam& param, const std::vector<std::vector<CcuKernelSubmitInfo>>& stageSubmitInfos,
    u32 notifyNumOnMainThread) const
{
    u32 threadNum = static_cast<u32>(threads_.size());
    u32 ccuKernelNum = 0;
    std::vector<u32> ccuKernelNumList;
    for (const auto& submitInfos : stageSubmitInfos) {
        ccuKernelNumList.push_back(static_cast<u32>(submitInfos.size()));
        ccuKernelNum += static_cast<u32>(submitInfos.size());
    }
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2AlltoAllSequenceExecutor][%s] ccu kernel num is 0, no need to save.", __func__);
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[InsV2AlltoAllSequenceExecutor][%s] threadNum[%u], ccuKernelNum[%u]", __func__, threadNum, ccuKernelNum);

    return FastLaunchSaveCtxTwoTemplate(
        param, threadNum, ccuKernelNum, threads_, ccuKernelNumList, stageSubmitInfos, notifyNumOnMainThread);
}

template <typename AlgTopoMatch>
HcclResult InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::FastLaunch(const OpParam& param, const CcuFastLaunchCtx* resCtx)
{
    // 契约校验：两级各必发 kernel（首∈[1,2]、末恒1），缺级会被模板级 FastLaunch "just success" 静默跳过，须硬失败拦截；
    // ccuKernelNum 为定长数组、save 侧恒写 [0]/[1]，无越界/脏读
    CHK_PRT_RET(
        resCtx->threadNum == 0 || resCtx->ccuKernelNum[0] < 1 || resCtx->ccuKernelNum[1] < 1,
        HCCL_ERROR(
            "[InsV2AlltoAllSequenceExecutor][FastLaunch] invalid ctx: threadNum[%u], ccuKernelNum[%u]+[%u].",
            resCtx->threadNum, resCtx->ccuKernelNum[0], resCtx->ccuKernelNum[1]),
        HcclResult::HCCL_E_INTERNAL);

    HCCL_INFO(
        "[InsV2AlltoAllSequenceExecutor][FastLaunch] Start, threadNum[%u], firstKernelNum[%u], lastKernelNum[%u]",
        resCtx->threadNum, resCtx->ccuKernelNum[0], resCtx->ccuKernelNum[1]);
    TemplateFastLaunchCtx tempFastLaunchCtxFirst;
    TemplateFastLaunchCtx tempFastLaunchCtxLast;

    ThreadHandle* threads = resCtx->GetThreadHandlePtr();
    threads_.assign(threads, threads + resCtx->threadNum);
    CcuKernelSubmitInfo* ccuKernelSubmitInfos = resCtx->GetCcuKernelSubmitInfoPtr();

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxFirst, param.inputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxFirst.threads = threads_;
    tempFastLaunchCtxFirst.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + resCtx->ccuKernelNum[0]);
    ccuKernelSubmitInfos += resCtx->ccuKernelNum[0];

    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxLast, param.inputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxLast.threads = threads_;
    tempFastLaunchCtxLast.ccuKernelSubmitInfos.assign(
        ccuKernelSubmitInfos, ccuKernelSubmitInfos + resCtx->ccuKernelNum[1]);

    CcuTempAlltoAllHierFirstStage firstStageAlg{};
    CHK_RET(firstStageAlg.FastLaunch(param, tempFastLaunchCtxFirst));
    CcuTempAlltoAllHierLastStage lastStageAlg{};
    CHK_RET(lastStageAlg.FastLaunch(param, tempFastLaunchCtxLast));

    HCCL_INFO("[InsV2AlltoAllSequenceExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

template <typename AlgTopoMatch>
std::vector<CostModelParam> InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AlltoAllSequenceExecutor][CalcCostCoeff]",
            TopoProbeScene::PROBE_CALC_COST_COEFF)) {
        return {};
    }
    // infos 经 MatchTopoForProbe 成功定形恒为 2 层 × 每层 1 组；守卫为跨模块契约消费点防御（异常 → 可控剔除）
    CHK_PRT_RET(
        algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2 || algHierarchyInfo.infos[0].empty()
            || algHierarchyInfo.infos[1].empty(),
        HCCL_WARNING("[InsV2AlltoAllSequenceExecutor][CalcCostCoeff] algHierarchyInfo infos is empty, skip."), {});
    u32 d0 = static_cast<u32>(algHierarchyInfo.infos[0][0].size());
    u32 d1 = static_cast<u32>(algHierarchyInfo.infos[1][0].size());
    bool isCcu = (algName != nullptr && std::string(algName).rfind("CcuSched", 0) == 0);
    if (isCcu && (d0 < A2A_HIER_UNROLL_MIN || d1 < A2A_HIER_UNROLL_MIN)) {
        HCCL_INFO(
            "[InsV2AlltoAllSequenceExecutor][%s] algName[%s] skipped: d0[%u]/d1[%u] < %u.", __func__, algName, d0, d1,
            A2A_HIER_UNROLL_MIN);
        return {};
    }
    if (isCcu && (d0 - 1 > A2A_SEQ_CCU_MAX_PEER_NUM || d1 - 1 > A2A_SEQ_CCU_MAX_PEER_NUM)) {
        HCCL_INFO(
            "[InsV2AlltoAllSequenceExecutor][%s] algName[%s] skipped: d0[%u]/d1[%u] exceed CCU channel limit, "
            "max peer num[%u].",
            __func__, algName, d0, d1, A2A_SEQ_CCU_MAX_PEER_NUM);
        return {};
    }
    CHK_PRT_RET(
        algHierarchyInfo.physicalIdxForAlgoLevels.size() != TOPO_LEVEL_NUM_2,
        HCCL_WARNING("[InsV2AlltoAllSequenceExecutor][CalcCostCoeff] physicalIdxForAlgoLevels size != 2."), {});
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;

    std::vector<std::vector<CommTopo>> phyLevelNetTypes(physIdx.size());
    std::vector<std::vector<std::vector<u32>>> phyLevelPortNums(physIdx.size());
    AppendPhysicalLevelInfo(topoInfo, physIdx, phyLevelNetTypes, phyLevelPortNums);
    CHK_PRT_RET(
        phyLevelPortNums[0].empty() || phyLevelPortNums[1].empty() || phyLevelPortNums[0][0].empty()
            || phyLevelPortNums[1][0].empty(),
        HCCL_WARNING("[InsV2AllToAllSequenceExecutor][CalcCostCoeff] portNum is empty, skip."), {});
    auto sumPort = [](const std::vector<u32>& ports) {
        int sum = 0;
        for (auto p : ports) {
            sum += static_cast<int>(p);
        }
        return sum > 0 ? sum : 1;
    };

    int aicpuKernelNum = static_cast<int>(A2A_SEQ_AICPU_KERNEL_NUM_PER_STAGE);
    std::vector<CostModelParam> params;
    params.push_back(CalcA2ASegCost(
        d0, 1.0f * d1, phyLevelNetTypes[0][0], sumPort(phyLevelPortNums[0][0]), topoInfo->isPod, isCcu,
        isCcu ? static_cast<int>(A2A_SEQ_CC_KERNEL_INTRA) : aicpuKernelNum));
    params.push_back(CalcA2ASegCost(
        d1, 1.0f * d0, phyLevelNetTypes[1][0], sumPort(phyLevelPortNums[1][0]), topoInfo->isPod, isCcu,
        isCcu ? static_cast<int>(A2A_SEQ_CC_KERNEL_MAX_INTER) : aicpuKernelNum));
    return params;
}

template <typename AlgTopoMatch>
AlgNetMeta InsV2AlltoAllSequenceExecutor<AlgTopoMatch>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AlltoAllSequenceExecutor][GetAlgNetMeta]",
            TopoProbeScene::PROBE_GET_ALG_NET_META)) {
        return {};
    }
    CHK_PRT_RET(
        algHierarchyInfo.physicalIdxForAlgoLevels.size() != TOPO_LEVEL_NUM_2,
        HCCL_WARNING("[InsV2AlltoAllSequenceExecutor][GetAlgNetMeta] physicalIdxForAlgoLevels size != 2."), {});

    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    std::vector<std::vector<CommTopo>> phyLevelNetTypes(physIdx.size());
    std::vector<std::vector<std::vector<u32>>> phyLevelPortNums(physIdx.size());
    AppendPhysicalLevelInfo(topoInfo, physIdx, phyLevelNetTypes, phyLevelPortNums);
    if (phyLevelNetTypes[0].empty() || phyLevelNetTypes[1].empty()) {
        HCCL_WARNING("[InsV2AlltoAllSequenceExecutor][GetAlgNetMeta] netType is empty, skip.");
        return {};
    }
    CHK_PRT_RET(
        algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2 || algHierarchyInfo.infos[0].empty()
            || algHierarchyInfo.infos[1].empty(),
        HCCL_WARNING("[InsV2AlltoAllSequenceExecutor][GetAlgNetMeta] algHierarchyInfo infos is empty, skip."), {});
    u32 d0 = static_cast<u32>(algHierarchyInfo.infos[0][0].size());
    u32 d1 = static_cast<u32>(algHierarchyInfo.infos[1][0].size());

    // 段序 [L0, L1] 与 CalcCostCoeff 两段实算对齐，dataRatios 为每段每对端块数 {d1, d0}
    AlgNetMeta meta;
    meta.netTypes = {phyLevelNetTypes[0][0], phyLevelNetTypes[1][0]};
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1};
    meta.dataRatios = {1.0f * d1, 1.0f * d0};
    meta.rankSizes = {d0, d1};
    return meta;
}

REGISTER_EXECUTOR_BY_TOPO(
    HcclCMDType::HCCL_CMD_ALLTOALL, AicpuAllToAllSequenceMeshMesh, InsV2AlltoAllSequenceExecutor, TopoMatchTwoLevel);

REGISTER_ALG_ATTRS(AicpuAllToAllSequenceMeshMesh, topo.minTopoLevelNum = 2; topo.maxTopoLevelNum = 2;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;);

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
// 三级物理拓扑经 TwoLevel 压缩为两级算法视图（打平）
REGISTER_EXECUTOR_BY_TOPO(
    HcclCMDType::HCCL_CMD_ALLTOALL, CcuSchedAllToAllSequenceMeshMesh, InsV2AlltoAllSequenceExecutor, TopoMatchTwoLevel);

REGISTER_ALG_ATTRS(CcuSchedAllToAllSequenceMeshMesh, topo.minTopoLevelNum = 2; topo.maxTopoLevelNum = 3;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D; topo.maxSupportRankSize = A2A_SEQ_MAX_RANK_SIZE;
                   op.isSupportInplace = false);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif

} // namespace ops_hccl
