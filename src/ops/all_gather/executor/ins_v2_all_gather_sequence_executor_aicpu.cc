/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_sequence_executor_aicpu.h"
#include "topo_match_multilevel.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_temp_all_gather_nhr.h"
#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#endif
#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 2;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(HcclComm comm,
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void) comm;
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;

    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][InitCommInfo] myRank[%u], rankSize[%u], dataType[%u], dataTypeSize[%u]",
        myRank_, rankSize_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;

    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(HcclComm comm,
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    InitCommInfo(comm, param, topoInfo, algHierarchyInfo);
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR("[InsV2AllGatherSequenceExecutorAicpu] algHierarchyInfo size should be %u", SEQUENCE_EXECUTOR_LEVEL_NUM);
        return HCCL_E_INTERNAL;
    }
    // 第一步框间NHR
    InsAlgTemplate1 interTempAlg(param, myRank_, algHierarchyInfo.infos[1]);
    // 第二步框内Mesh
    InsAlgTemplate0 intraTempAlg(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqIntra;
    intraTempAlg.CalcRes(comm, param, topoInfo, resReqIntra);
    AlgResourceRequest resReqInter;
    interTempAlg.CalcRes(comm, param, topoInfo, resReqInter);

    // 分级算法，slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max(resReqIntra.slaveThreadNum, resReqInter.slaveThreadNum);
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqIntra.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqIntra.notifyNumPerThread[i]);
        }
        if (i < resReqInter.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqInter.notifyNumPerThread[i]);
        }
    }
    resourceRequest.notifyNumOnMainThread = std::max(resReqIntra.notifyNumOnMainThread, resReqInter.notifyNumOnMainThread);

    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        // ccu: 合并两个template的ccuKernelInfos
        HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][CalcRes] intraTemplate has [%d] kernels, interTemplate has [%d] kernels.",
                  resReqIntra.ccuKernelNum[0], resReqInter.ccuKernelNum[0]);
        resourceRequest.ccuKernelNum.emplace_back(resReqInter.ccuKernelNum[0]);
        resourceRequest.ccuKernelNum.emplace_back(resReqIntra.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              resReqInter.ccuKernelInfos.begin(), resReqInter.ccuKernelInfos.end());
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
                                              resReqIntra.ccuKernelInfos.begin(), resReqIntra.ccuKernelInfos.end());
    } else {
        resourceRequest.channels = {resReqIntra.channels[0], resReqInter.channels[0]};
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;
    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    engine_ = param.engine;
    // ccu路径无channel数据，跳过RestoreChannelMap
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    // 构建template
    InsAlgTemplate1 interTempAlg(param, myRank_, algHierarchyInfo_.infos[1]);
    InsAlgTemplate0 intraTempAlg(param, myRank_, algHierarchyInfo_.infos[0]);

    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        // ccu: 分配ccuKernels，线程与aicpu一致（共用）
        interCcuKernels_.assign(resCtx.ccuKernels.begin(), resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
        intraCcuKernels_.assign(resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0],
                               resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    }

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, intraTempAlg, interTempAlg);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllGatherSequenceExecutorAicpu][Orchestrate]errNo[0x%016llx] Orchestrate failed",
            HCCL_ERROR_CODE(ret)), ret);
    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][Orchestrate] Orchestrate End");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInterTemplateParams(
    TemplateDataParams &interTempDataParams, const u64 processedDataCount, const u64 currDataCount, const u64 loop) const
{
    interTempDataParams.count = currDataCount;
    interTempDataParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    interTempDataParams.buffInfo.outBuffBaseOff = 0; // 第一阶段不做后拷贝
    interTempDataParams.buffInfo.hcclBuffBaseOff = 0;

    interTempDataParams.sliceSize = currDataCount * dataTypeSize_;
    interTempDataParams.tailSize = interTempDataParams.sliceSize;

    interTempDataParams.inputSliceStride = 0;
    interTempDataParams.outputSliceStride = 0; // 第一阶段不做后拷贝
    interTempDataParams.repeatNum = 1;
    interTempDataParams.inputRepeatStride = 0;
    interTempDataParams.outputRepeatStride = 0;

    if (engine_ == CommEngine::COMM_ENGINE_CCU) {
        interTempDataParams.buffInfo.outBuffBaseOff = rankIdxLevel0_ * dataSize_ + processedDataCount * dataTypeSize_;
        interTempDataParams.outputSliceStride = dataSize_ * rankSizeLevel0_;
    }

    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu] loop[%llu] interTempDataParams.inputSliceStride[%llu] "
        "interTempDataParams.outputSliceStride[%llu] interTempDataParams.sliceSize[%llu] "
        "interTempDataParams.buffInfo.inBuffBaseOff[%llu] interTempDataParams.buffInfo.outBuffBaseOff[%llu] "
        "interTempDataParams.repeatNum[%llu] interTempDataParams.inputRepeatStride[%llu] "
        "interTempDataParams.outputRepeatStride[%llu]", loop, interTempDataParams.inputSliceStride,
        interTempDataParams.outputSliceStride, interTempDataParams.sliceSize,
        interTempDataParams.buffInfo.inBuffBaseOff, interTempDataParams.buffInfo.outBuffBaseOff,
        interTempDataParams.repeatNum, interTempDataParams.inputRepeatStride, interTempDataParams.outputRepeatStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenIntraTemplateParams(
    TemplateDataParams &intraTempDataParams, const u64 processedDataCount, const u64 currDataCount, const u64 loop) const
{
    intraTempDataParams.count = currDataCount;
    intraTempDataParams.buffInfo.inBuffBaseOff = 0; // 第二阶段的input就是ccl buffer
    intraTempDataParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    intraTempDataParams.buffInfo.hcclBuffBaseOff = 0;

    intraTempDataParams.sliceSize = currDataCount * dataTypeSize_;
    intraTempDataParams.tailSize = intraTempDataParams.sliceSize;
    // 这里的stride当成传统意义上的stride间隔
    intraTempDataParams.inputSliceStride = 0;
    intraTempDataParams.outputSliceStride = dataSize_;

    intraTempDataParams.repeatNum = rankSizeLevel1_;
    intraTempDataParams.inputRepeatStride = currDataCount * dataTypeSize_;
    intraTempDataParams.outputRepeatStride = dataSize_ * rankSizeLevel0_;

    if (engine_ == CommEngine::COMM_ENGINE_CCU) {
        intraTempDataParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_; 
        intraTempDataParams.inputSliceStride = dataSize_;
        intraTempDataParams.inputRepeatStride = dataSize_ * rankSizeLevel0_;
    }

    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu] loop[%llu] intraTempDataParams.inputSliceStride[%llu] "
        "intraTempDataParams.outputSliceStride[%llu] intraTempDataParams.sliceSize[%llu] "
        "intraTempDataParams.buffInfo.inBuffBaseOff[%llu] intraTempDataParams.buffInfo.outBuffBaseOff[%llu] "
        "intraTempDataParams.repeatNum[%llu] intraTempDataParams.inputRepeatStride[%llu] "
        "intraTempDataParams.outputRepeatStride[%llu]", loop, intraTempDataParams.inputSliceStride,
        intraTempDataParams.outputSliceStride, intraTempDataParams.sliceSize,
        intraTempDataParams.buffInfo.inBuffBaseOff, intraTempDataParams.buffInfo.outBuffBaseOff,
        intraTempDataParams.repeatNum, intraTempDataParams.inputRepeatStride, intraTempDataParams.outputRepeatStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
template <typename InsAlgTemplate>
HcclResult InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTempResource
    (const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
    const InsAlgTemplate &algTemplate, TemplateResource &tempReousrce) const
{
    AlgResourceRequest req;
    algTemplate.GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR("[InsV2AllGatherSequenceExecutorAicpu][GenTempResource] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]", channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempReousrce.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempReousrce.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}


template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    InsAlgTemplate0 &intraTempAlg, InsAlgTemplate1 &interTempAlg)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][OrchestrateLoop] Start");

    // 框间template
    TemplateDataParams interTempDataParams;
    interTempDataParams.buffInfo.inputPtr = param.inputPtr;
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        // ccu: 用output作为中转，减少一次拷贝
        interTempDataParams.buffInfo.outputPtr = param.outputPtr;
        interTempDataParams.buffInfo.outBuffType = BufferType::OUTPUT;
    } else {
        interTempDataParams.buffInfo.outputPtr = resCtx.cclMem.addr;
        interTempDataParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    }
    interTempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    interTempDataParams.buffInfo.inBuffType = BufferType::INPUT;
    interTempDataParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框间template（aicpu路径需要SetchannelsPerRank）
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        interTempAlg.SetchannelsPerRank(remoteRankToChannelInfo_[1]);
    }

    // 框内template
    TemplateDataParams intraTempDataParams;
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        // ccu: 用output作为中转，input=output，isInputOutputEqual=1，跳过GroupCopy
        intraTempDataParams.buffInfo.inputPtr = param.outputPtr;
        intraTempDataParams.buffInfo.inBuffType = BufferType::OUTPUT;
    } else {
        intraTempDataParams.buffInfo.inputPtr = resCtx.cclMem.addr;
        intraTempDataParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    }
    intraTempDataParams.buffInfo.outputPtr = param.outputPtr;
    intraTempDataParams.buffInfo.hcclBuff = resCtx.cclMem;
    intraTempDataParams.buffInfo.outBuffType = BufferType::OUTPUT;
    intraTempDataParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框内template（aicpu路径需要SetchannelsPerRank）
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        intraTempAlg.SetchannelsPerRank(remoteRankToChannelInfo_[0]);
    }

    // ccl buffer按框数切分
    u32 templateScratchMultiplier = interTempAlg.CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);

    // 构造框间template资源
    TemplateResource templateResourceInter;
    // 构造框内template资源
    TemplateResource templateResourceIntra;

    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        // ccu: ccuKernels按template分，线程与aicpu一致共用threads_
        templateResourceInter.ccuKernels = interCcuKernels_;
        templateResourceInter.threads = threads_;
        templateResourceIntra.ccuKernels = intraCcuKernels_;
        templateResourceIntra.threads = threads_;
    } else {
        CHK_RET(GenTempResource(resCtx, 1, interTempAlg, templateResourceInter));
        CHK_RET(GenTempResource(resCtx, 0, intraTempAlg, templateResourceIntra));
    }
    
    u64 maxCountPerLoop = 0;
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        maxCountPerLoop = UB_MAX_DATA_SIZE / dataTypeSize_;
    } else {
        if (templateScratchMultiplier == 0) {
            HCCL_ERROR("[%s] templateScratchMultiplier is 0, division by zero.", __func__);
            return HCCL_E_INTERNAL;
        }
        maxCountPerLoop = interTempDataParams.buffInfo.hcclBuff.size / templateScratchMultiplier /
            HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    }
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        // 框间的数据偏移和搬运计算
        GenInterTemplateParams(interTempDataParams, processedDataCount, currDataCount, loop);
        CHK_RET(SplitData(currDataCount, rankSizeLevel1_, interTempDataParams));
        CHK_RET(interTempAlg.KernelRun(param, interTempDataParams, templateResourceInter));

        // 框内的数据偏移和搬运量计算
        GenIntraTemplateParams(intraTempDataParams, processedDataCount, currDataCount, loop);
        CHK_RET(intraTempAlg.KernelRun(param, intraTempDataParams, templateResourceIntra));

        processedDataCount += currDataCount;
    }

#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        ccuKernelLaunchNumInter_ = templateResourceInter.submitInfos.size();
        ccuKernelLaunchNumIntra_ = templateResourceIntra.submitInfos.size();
        CHK_RET(FastLaunchSaveCtx(param, templateResourceInter, templateResourceIntra, resCtx.notifyNumOnMainThread));
    }
#endif

    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &templateAlgResInter, const TemplateResource &templateAlgResIntra, u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][FastLaunchSaveCtx] loopTimes==1, save fast launch ctx.");
    u32 threadNum = threads_.size();
    u32 ccuKernelNum = ccuKernelLaunchNumInter_ + ccuKernelLaunchNumIntra_;
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][FastLaunchSaveCtx] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][FastLaunchSaveCtx] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    // sequence执行: 先inter再intra, ccuKernelNumList = {inter, intra}
    std::vector<u32> ccuKernelNumList = {ccuKernelLaunchNumInter_, ccuKernelLaunchNumIntra_};
    std::vector<std::vector<CcuKernelSubmitInfo>> submitInfosList = {templateAlgResInter.submitInfos, templateAlgResIntra.submitInfos};
    return FastLaunchSaveCtxTwoTemplate(param, threadNum, ccuKernelNum, threads_, ccuKernelNumList, submitInfosList, notifyNumOnMainThread);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunch(
        const OpParam &param, const CcuFastLaunchCtx *resCtx)
{
    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][FastLaunch] Start");
    InsAlgTemplate1 interTempAlg{};
    InsAlgTemplate0 intraTempAlg{};

    TemplateFastLaunchCtx tempFastLaunchCtxInter;
    TemplateFastLaunchCtx tempFastLaunchCtxIntra;

    ThreadHandle *threads = resCtx->GetThreadHandlePtr();
    threads_.assign(threads, threads + resCtx->threadNum);

    CcuKernelSubmitInfo *ccuKernelSubmitInfos = resCtx->GetCcuKernelSubmitInfoPtr();

    // 第一步: 框间NHR（ccu模式下用output作为中转）
    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][FastLaunch] inter ccuKernelNum[%llu]", resCtx->ccuKernelNum[0]);
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter, param.inputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxInter.threads = threads_;
    tempFastLaunchCtxInter.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + resCtx->ccuKernelNum[0]);
    ccuKernelSubmitInfos += resCtx->ccuKernelNum[0];
    if (resCtx->ccuKernelNum[0] > 0) {
        CHK_RET(interTempAlg.FastLaunch(param, tempFastLaunchCtxInter));
    }

    // 第二步: 框内Mesh（ccu模式下input=output，isInputOutputEqual=1）
    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][FastLaunch] intra ccuKernelNum[%llu]", resCtx->ccuKernelNum[1]);
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra, param.outputPtr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxIntra.threads = threads_;
    tempFastLaunchCtxIntra.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + resCtx->ccuKernelNum[1]);
    if (resCtx->ccuKernelNum[1] > 0) {
        CHK_RET(intraTempAlg.FastLaunch(param, tempFastLaunchCtxIntra));
    }

    HCCL_INFO("[InsV2AllGatherSequenceExecutorAicpu][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllGatherSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::SplitData(
    const u64 dataCount, const u64 rankSize, TemplateDataParams &tempAlgParams)
{
    u32 sliceNum = rankSize;
    tempAlgParams.allRankSliceSize.clear();
    tempAlgParams.allRankDispls.clear();
    tempAlgParams.allRankProcessedDataCount.clear();
    tempAlgParams.allRankSliceSize.reserve(sliceNum);
    tempAlgParams.allRankDispls.reserve(sliceNum);
    tempAlgParams.allRankProcessedDataCount.reserve(sliceNum);

    u64 sliceSize = dataCount * dataTypeSize_;
    for (u32 i = 0; i < sliceNum; i++) {
        tempAlgParams.allRankDispls.emplace_back(i * sliceSize);
        tempAlgParams.allRankSliceSize.emplace_back(sliceSize);
        tempAlgParams.allRankProcessedDataCount.emplace_back(dataCount);
    }
    return HCCL_SUCCESS;
}

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER,
                               InsAllGatherSequenceNHRMesh1D,
                               InsV2AllGatherSequenceExecutorAicpu,
                               TopoMatchMultilevel,
                               InsTempAllGatherMesh1D1DZAxisDetour,
                               InsTempAllGatherNHR);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#ifndef AICPU_COMPILE
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLGATHER, CcuAllGatherSequenceMeshMesh,
    InsV2AllGatherSequenceExecutorAicpu, TopoMatchMultilevel,
    CcuTempAllGatherMesh1DMem2Mem, CcuTempAllGatherMesh1DMem2Mem);
#endif
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
}