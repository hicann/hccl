/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_sequence_executor_aicpu.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "alg_data_trans_wrapper.h"

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ccu_temp_reduce_scatter_mesh_1D_mem2mem.h"
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif

namespace ops_hccl {

// 序列执行器需要的层级数
constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 2;
constexpr u32 CCL_MEM_HALF_DIVISOR = 2;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2ReduceScatterSequenceExecutorAicpu()
{
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutorAicpu][InitCommInfo] myRank [%u], rankSize [%u], redOp [%u], "
        "dataType [%u] dataTypeSize [%u]", myRank_, rankSize_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
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
HcclResult InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[InsV2ReduceScatterSequenceExecutorAicpu] CalcRes start");
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR("[InsV2ReduceScatterSequenceExecutorAicpu] algHierarchyInfo size should be %u", SEQUENCE_EXECUTOR_LEVEL_NUM);
        return HCCL_E_INTERNAL;
    }
    // 第一步框内mesh
    std::shared_ptr<InsAlgTemplate0> intraTempAlg = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    // 第二步框间NHR
    std::shared_ptr<InsAlgTemplate1> interTempAlg = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);

    AlgResourceRequest resReqIntra;
    AlgResourceRequest resReqInter;
    CHK_RET(intraTempAlg->CalcRes(comm, param, topoInfo, resReqIntra));
    CHK_RET(interTempAlg->CalcRes(comm, param, topoInfo, resReqInter));

    // 分级算法，slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum = std::max(resReqInter.slaveThreadNum, resReqIntra.slaveThreadNum);
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqInter.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqInter.notifyNumPerThread[i]);
        }
        if (i < resReqIntra.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i], resReqIntra.notifyNumPerThread[i]);
        }
    }
    resourceRequest.notifyNumOnMainThread = std::max(resReqInter.notifyNumOnMainThread, resReqIntra.notifyNumOnMainThread);
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutorAicpu] notifyNumOnMainThread is %u", resourceRequest.notifyNumOnMainThread);
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        // ccu
        HCCL_INFO("[InsV2ReduceScatterSequenceExecutorAicpu] ccu intraTemplate has %d channels, interTemplate has %d channels",
            resReqIntra.ccuKernelNum[0], resReqInter.ccuKernelNum[0]);
        resourceRequest.ccuKernelNum.emplace_back(resReqIntra.ccuKernelNum[0]);
        resourceRequest.ccuKernelNum.emplace_back(resReqInter.ccuKernelNum[0]);
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
            resReqIntra.ccuKernelInfos.begin(), resReqIntra.ccuKernelInfos.end());
        resourceRequest.ccuKernelInfos.insert(resourceRequest.ccuKernelInfos.end(),
            resReqInter.ccuKernelInfos.begin(), resReqInter.ccuKernelInfos.end());
    } else {
        resourceRequest.channels.resize(SEQUENCE_EXECUTOR_LEVEL_NUM);
        resourceRequest.channels[0] = resReqIntra.channels[0];
        resourceRequest.channels[1] = resReqInter.channels[0];
        HCCL_INFO("[InsV2ReduceScatterSequenceExecutorAicpu] slaveThreadNum is [%u], notifyNumOnMainThread is [%u], "\
            "level 1 chanel size [%u], level 2 channel size [%u]",
            resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread,
            resourceRequest.channels[0].size(), resourceRequest.channels[1].size());
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataCount_ = param.DataDes.count;
    dataTypeSize_ =  SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = myRank_ / algHierarchyInfo_.infos[0][0].size();

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    // ccu无channel数据，跳过RestoreChannelMap
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }
    
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        // ccu
        intraCcuKernels_.assign(resCtx.ccuKernels.begin(), resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0]);
        interCcuKernels_.assign(resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0], 
            resCtx.ccuKernels.begin() + resCtx.ccuKernelNum[0] + resCtx.ccuKernelNum[1]);
    }
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ReduceScatterSequenceExecutorAicpu][Orchestrate] errNo[0x%016llx] "\
            "Reduce scatter executor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenIntraTemplateParams(
    TemplateDataParams &tempAlgParamsIntra, const u64 processedDataCount, const u64 currDataCount, const u64 loop) const
{
    tempAlgParamsIntra.count = currDataCount;
    tempAlgParamsIntra.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsIntra.buffInfo.outBuffBaseOff = 0; // 从input搬运到ccl，最终输出到ccl上面
    if (engine_ == CommEngine::COMM_ENGINE_CCU) {
        tempAlgParamsIntra.buffInfo.hcclBuffBaseOff = scratchBlockSize_;
    } else {
        tempAlgParamsIntra.buffInfo.hcclBuffBaseOff = 0;
    }

    tempAlgParamsIntra.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsIntra.tailSize = tempAlgParamsIntra.sliceSize;

    tempAlgParamsIntra.inputSliceStride = dataSize_; // ccl按照分给每个rank的数据量偏移量
    tempAlgParamsIntra.outputSliceStride = currDataCount * dataTypeSize_;
    // 框内需要做框数次重复
    tempAlgParamsIntra.repeatNum = rankSizeLevel1_;
    tempAlgParamsIntra.inputRepeatStride = rankSizeLevel0_ * dataSize_;
    tempAlgParamsIntra.outputRepeatStride = rankSizeLevel0_ * currDataCount * dataTypeSize_;

    HCCL_INFO("[InsV2ReduceScatterSequenceExecutorAicpu] loop[%llu] tempAlgParamsIntra.inputSliceStride[%llu] "
    "tempAlgParamsIntra.outputSliceStride[%llu] tempAlgParamsIntra.sliceSize[%llu] "
    "tempAlgParamsIntra.buffInfo.inBuffBaseOff[%llu] tempAlgParamsIntra.buffInfo.outBuffBaseOff[%llu] "
    "tempAlgParamsIntra.repeatNum[%llu] tempAlgParamsIntra.inputRepeatStride[%llu] "
    "tempAlgParamsIntra.outputRepeatStride[%llu]", loop, tempAlgParamsIntra.inputSliceStride,
    tempAlgParamsIntra.outputSliceStride, tempAlgParamsIntra.sliceSize,
    tempAlgParamsIntra.buffInfo.inBuffBaseOff, tempAlgParamsIntra.buffInfo.outBuffBaseOff,
    tempAlgParamsIntra.repeatNum, tempAlgParamsIntra.inputRepeatStride, tempAlgParamsIntra.outputRepeatStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenInterTemplateParams(
    TemplateDataParams &tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount, const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = rankIdxLevel0_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    if (engine_ == CommEngine::COMM_ENGINE_CCU) {
        tempAlgParamsInter.buffInfo.hcclBuffBaseOff = scratchBlockSize_;
    } else {
        tempAlgParamsInter.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * currDataCount * dataTypeSize_;
    }
    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = rankSizeLevel0_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputSliceStride = 0;
    // 不需要重复
    tempAlgParamsInter.repeatNum = 1;
    tempAlgParamsInter.inputRepeatStride = 0;
    tempAlgParamsInter.outputRepeatStride = 0;

    HCCL_INFO("[InsV2ReduceScatterSequenceExecutorAicpu] loop[%llu] tempAlgParamsInter.inputSliceStride[%llu] "
    "tempAlgParamsInter.outputSliceStride[%llu] tempAlgParamsInter.sliceSize[%llu] "
    "tempAlgParamsInter.buffInfo.inBuffBaseOff[%llu] tempAlgParamsInter.buffInfo.outBuffBaseOff[%llu] "
    "tempAlgParamsInter.repeatNum[%llu] tempAlgParamsInter.inputRepeatStride[%llu] "
    "tempAlgParamsInter.outputRepeatStride[%llu]", loop, tempAlgParamsInter.inputSliceStride,
    tempAlgParamsInter.outputSliceStride, tempAlgParamsInter.sliceSize,
    tempAlgParamsInter.buffInfo.inBuffBaseOff, tempAlgParamsInter.buffInfo.outBuffBaseOff,
    tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride, tempAlgParamsInter.outputRepeatStride);
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
template <typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTempResource
    (const AlgResourceCtxSerializable &resCtx, const u32 channelLevelIdx,
    const std::shared_ptr<InsAlgTemplate> &algTemplate, TemplateResource &tempReousrce) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR("[InsV2ReduceScatterSequenceExecutorAicpu][GenTempResource] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]", channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempReousrce.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempReousrce.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    engine_ = param.engine;
    scratchBlockSize_ = resCtx.cclMem.size / CCL_MEM_HALF_DIVISOR;
    // 框内模板参数，input搬运到ccl，最终规约到ccl
    TemplateDataParams tempAlgParamsIntra;
    tempAlgParamsIntra.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsIntra.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsIntra.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsIntra.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsIntra.buffInfo.hcclBuff = resCtx.cclMem;

    // 构建框内template
    std::shared_ptr<InsAlgTemplate0> algTemplateIntra = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    
    // aicpu需要设置channelsPerRank
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        algTemplateIntra->SetchannelsPerRank(remoteRankToChannelInfo_[0]);
    }

    // 框间模板参数，ccl写到对端ccl，最终搬运到output上
    TemplateDataParams tempAlgParamsInter;
    tempAlgParamsInter.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsInter.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsInter.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsInter.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsInter.buffInfo.hcclBuff = resCtx.cclMem;

    // 构建框间template
    std::shared_ptr<InsAlgTemplate1> algTemplateInter = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
    
    // aicpu需要设置channelsPerRank
    if (param.engine != CommEngine::COMM_ENGINE_CCU) {
        algTemplateInter->SetchannelsPerRank(remoteRankToChannelInfo_[1]);
    }

    u32 templateScratchMultiplierIntra = algTemplateIntra->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplierInter = algTemplateInter->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    u32 templateScratchMultiplier = 1;
    templateScratchMultiplier = templateScratchMultiplierIntra * templateScratchMultiplierInter;

    TemplateResource templateResourceIntra, templateResourceInter;
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        templateResourceIntra.ccuKernels = intraCcuKernels_;
        templateResourceIntra.threads = threads_;
        templateResourceInter.ccuKernels = interCcuKernels_;
        templateResourceInter.threads = threads_;
    } else {
        CHK_RET(GenTempResource(resCtx, 0, algTemplateIntra, templateResourceIntra));
        CHK_RET(GenTempResource(resCtx, 1, algTemplateInter, templateResourceInter));
    }

    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 maxCountPerLoop = 0;
    if (param.engine == CommEngine::COMM_ENGINE_CCU) {
        maxCountPerLoop = scratchBlockSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN 
            * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
        maxCountPerLoop = std::min<u64>(maxCountPerLoop, UB_MAX_DATA_SIZE / dataTypeSize_);
    } else {
        maxCountPerLoop = tempAlgParamsInter.buffInfo.hcclBuff.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN
            * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    }
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        // ----------- 框内数据搬运 -----------
        // 框内的数据偏移和搬运计算
        GenIntraTemplateParams(tempAlgParamsIntra, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateIntra->KernelRun(param, tempAlgParamsIntra, templateResourceIntra));

        // ----------- 框间数据搬运 -----------
        // 框间的数据偏移和搬运量计算
        GenInterTemplateParams(tempAlgParamsInter, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateInter->KernelRun(param, tempAlgParamsInter, templateResourceInter));
        processedDataCount += currDataCount;
    }

#ifndef AICPU_COMPILE
    if (loopTimes == 1 && param.engine == CommEngine::COMM_ENGINE_CCU && param.opMode != OpMode::OFFLOAD) {
        CHK_RET(FastLaunchSaveCtx(param, templateResourceIntra, templateResourceInter, resCtx.notifyNumOnMainThread));
    }
#endif
    return HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunchSaveCtx(
    const OpParam &param, const TemplateResource &templateAlgResIntra, const TemplateResource &templateAlgResInter, u32 notifyNumOnMainThread)
{
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor] loopTimes==1, save fast launch ctx.");
    u32 threadNum = threads_.size();
    u32 ccuKernelNum = templateAlgResInter.submitInfos.size() + templateAlgResIntra.submitInfos.size();
    if (ccuKernelNum < 1) {
        HCCL_INFO("[InsV2ReduceScatterSequenceExecutor] ccu kernel num is 0, no need to save.");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor][HcclEngineCtxCreate] threadNum[%llu], ccuKernelNum[%llu]", threadNum, ccuKernelNum);

    std::vector<u32> ccuKernelNumList = {static_cast<u32>(templateAlgResIntra.submitInfos.size()),
                                         static_cast<u32>(templateAlgResInter.submitInfos.size())};
    std::vector<std::vector<CcuKernelSubmitInfo>> submitInfosList = {templateAlgResIntra.submitInfos, templateAlgResInter.submitInfos};
    return FastLaunchSaveCtxTwoTemplate(param, threadNum, ccuKernelNum, threads_, ccuKernelNumList, submitInfosList, notifyNumOnMainThread);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2ReduceScatterSequenceExecutorAicpu<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::FastLaunch(
        const OpParam &param, const CcuFastLaunchCtx *ctx)
{
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor][FastLaunch] Start");
    InsAlgTemplate1 tempAlgInter{};
    InsAlgTemplate0 tempAlgIntra{};
    
    TemplateFastLaunchCtx tempFastLaunchCtxInter, tempFastLaunchCtxIntra;

    ThreadHandle *threads = ctx->GetThreadHandlePtr();
    threads_.assign(threads, threads + ctx->threadNum);

    CcuKernelSubmitInfo *ccuKernelSubmitInfos = ctx->GetCcuKernelSubmitInfoPtr();
    
    // 框间template
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor][FastLaunch] Intra ccuKernelNum[%llu]", ctx->ccuKernelNum[0]);
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxIntra, param.inputPtr, param.hcclBuff.addr, param.hcclBuff));
    tempFastLaunchCtxIntra.threads = threads_;
    tempFastLaunchCtxIntra.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[0]);
    ccuKernelSubmitInfos += ctx->ccuKernelNum[0];
    if (ctx->ccuKernelNum[0] > 0) {
        CHK_RET(tempAlgIntra.FastLaunch(param, tempFastLaunchCtxIntra));
    }
    
    // 框内template
    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor][FastLaunch] Inter ccuKernelNum[%llu]", ctx->ccuKernelNum[1]);
    CHK_RET(SetTempFastLaunchAddr(tempFastLaunchCtxInter, param.hcclBuff.addr, param.outputPtr, param.hcclBuff));
    tempFastLaunchCtxInter.threads = threads_;
    tempFastLaunchCtxInter.ccuKernelSubmitInfos.assign(ccuKernelSubmitInfos, ccuKernelSubmitInfos + ctx->ccuKernelNum[1]);
    if (ctx->ccuKernelNum[1] > 0) {
        CHK_RET(tempAlgInter.FastLaunch(param, tempFastLaunchCtxInter));
    }

    HCCL_INFO("[InsV2ReduceScatterSequenceExecutor][FastLaunch] End.");
    return HCCL_SUCCESS;
}
#endif
 

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
                                InsReduceScatterSequenceMesh1DNhr,
                                InsV2ReduceScatterSequenceExecutorAicpu,
                                TopoMatchMultilevel,
                                InsTempReduceScatterMesh1DZAxisDetour,
                                InsTempReduceScatterNHR);
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */

#ifndef AICPU_COMPILE
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
                               CcuReduceScatterSequenceMeshMesh,
                               InsV2ReduceScatterSequenceExecutorAicpu,
                               TopoMatchMultilevel,
                               CcuTempReduceScatterMesh1DMem2Mem,
                               CcuTempReduceScatterMesh1DMem2Mem);
#endif /* CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0) */
#endif
}