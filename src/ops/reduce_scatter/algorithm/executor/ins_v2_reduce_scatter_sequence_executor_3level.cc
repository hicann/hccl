/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_reduce_scatter_sequence_executor_3level.h"
#include "cost_model.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "alg_data_trans_wrapper.h"

#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 3;
constexpr u32 OMNIPIPE_LEVEL2_IDX = 2;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2ReduceScatterSequenceExecutor3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2ReduceScatterSequenceExecutor3Level()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level][InitCommInfo] myRank [%u], rankSize [%u], redOp [%u], "
        "dataType [%u] dataTypeSize [%u]",
        myRank_, rankSize_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[InsV2ReduceScatterSequenceExecutor3Level][CalcRes] myRank[%u] start", myRank_);
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] algHierarchyInfo size should be %u", myRank_,
            SEQUENCE_EXECUTOR_LEVEL_NUM);
        return HCCL_E_INTERNAL;
    }
    skipLevel1_ = (algHierarchyInfo.infos[1][0].size() == 1);
    std::shared_ptr<InsAlgTemplate0> tempAlgLevel0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> tempAlgLevel1
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> tempAlgLevel2
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);

    AlgResourceRequest resReq0;
    AlgResourceRequest resReq1;
    AlgResourceRequest resReq2;
    CHK_RET(tempAlgLevel0->CalcRes(comm, param, topoInfo, resReq0));
    if (skipLevel1_) {
        HCCL_INFO(
            "[InsV2ReduceScatterSequenceExecutor3Level][CalcRes] myRank[%u] level1 rankSize is 1, skip level1 CalcRes",
            myRank_);
    } else {
        CHK_RET(tempAlgLevel1->CalcRes(comm, param, topoInfo, resReq1));
    }
    CHK_RET(tempAlgLevel2->CalcRes(comm, param, topoInfo, resReq2));

    if (skipLevel1_) {
        resourceRequest.slaveThreadNum = std::max(resReq0.slaveThreadNum, resReq2.slaveThreadNum);
    } else {
        resourceRequest.slaveThreadNum
            = std::max({resReq0.slaveThreadNum, resReq1.slaveThreadNum, resReq2.slaveThreadNum});
    }
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReq0.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq0.notifyNumPerThread[i]);
        }
        if (!skipLevel1_ && i < resReq1.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq1.notifyNumPerThread[i]);
        }
        if (i < resReq2.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq2.notifyNumPerThread[i]);
        }
    }
    if (skipLevel1_) {
        resourceRequest.notifyNumOnMainThread = std::max(resReq0.notifyNumOnMainThread, resReq2.notifyNumOnMainThread);
    } else {
        resourceRequest.notifyNumOnMainThread
            = std::max({resReq0.notifyNumOnMainThread, resReq1.notifyNumOnMainThread, resReq2.notifyNumOnMainThread});
    }
    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] notifyNumOnMainThread is %u", myRank_,
        resourceRequest.notifyNumOnMainThread);
    resourceRequest.channels.resize(SEQUENCE_EXECUTOR_LEVEL_NUM);
    if (resReq0.channels.empty() || resReq2.channels.empty()) {
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] channels empty, level0[%u] level2[%u]", myRank_,
            resReq0.channels.size(), resReq2.channels.size());
        return HCCL_E_INTERNAL;
    }
    if (!skipLevel1_ && resReq1.channels.empty()) {
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] channels empty, level1[%u]", myRank_,
            resReq1.channels.size());
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[0] = resReq0.channels[0];
    if (!skipLevel1_) {
        resourceRequest.channels[1] = resReq1.channels[0];
    }
    resourceRequest.channels[2] = resReq2.channels[0];
    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] slaveThreadNum is [%u], notifyNumOnMainThread is [%u], "
        "level0 chanel size [%u], level1 channel size [%u], level2 channel size [%u]",
        myRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumPerThread, resourceRequest.channels[0].size(),
        resourceRequest.channels[1].size(), resourceRequest.channels[2].size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    if (algHierarchyInfo_.infos.size() < TOPO_LEVEL_NUM_3 || algHierarchyInfo_.infos[0][0].empty()
        || algHierarchyInfo_.infos[1][0].empty() || algHierarchyInfo_.infos[2][0].empty()) {
        HCCL_ERROR("[%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = (myRank_ / algHierarchyInfo_.infos[0][0].size()) % algHierarchyInfo_.infos[1][0].size();
    rankIdxLevel2_ = myRank_ / (algHierarchyInfo_.infos[0][0].size() * algHierarchyInfo_.infos[1][0].size());

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    if (skipLevel1_) {
        HCCL_INFO(
            "[InsV2ReduceScatterSequenceExecutor3Level] [Orchestrate] myRank[%u] level1 rankSize is 1, skip level1",
            myRank_);
    }
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level] [Orchestrate] myRank_[%u] rankIdxLevel0_[%u] "
        "rankIdxLevel1_[%u] rankIdxLevel2_[%u] rankSizeLevel0_[%u] rankSizeLevel1_[%u] "
        "rankSizeLevel2_[%u]",
        myRank_, rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_, rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_);
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level][Orchestrate] myRank[%u] errNo[0x%016llx] "
            "Reduce scatter executor kernel run failed",
            myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenIntraTemplateParams(
        TemplateDataParams& tempAlgParamsIntra, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsIntra.count = currDataCount;
    tempAlgParamsIntra.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsIntra.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsIntra.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsIntra.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsIntra.tailSize = tempAlgParamsIntra.sliceSize;

    tempAlgParamsIntra.inputSliceStride = dataSize_;
    tempAlgParamsIntra.outputSliceStride = currDataCount * dataTypeSize_;
    tempAlgParamsIntra.repeatNum = rankSizeLevel1_ * rankSizeLevel2_;
    tempAlgParamsIntra.inputRepeatStride = rankSizeLevel0_ * dataSize_;
    tempAlgParamsIntra.outputRepeatStride = rankSizeLevel0_ * currDataCount * dataTypeSize_;

    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] loop[%llu] Intra inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsIntra.inputSliceStride, tempAlgParamsIntra.outputSliceStride,
        tempAlgParamsIntra.sliceSize, tempAlgParamsIntra.buffInfo.inBuffBaseOff,
        tempAlgParamsIntra.buffInfo.outBuffBaseOff, tempAlgParamsIntra.repeatNum, tempAlgParamsIntra.inputRepeatStride,
        tempAlgParamsIntra.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenInterTemplateParams1(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = rankIdxLevel0_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.outBuffBaseOff
        = (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_)) * currDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * currDataCount * dataTypeSize_;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = rankSizeLevel0_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = rankSizeLevel2_;
    tempAlgParamsInter.inputRepeatStride = rankSizeLevel0_ * rankSizeLevel1_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputRepeatStride = rankSizeLevel0_ * rankSizeLevel1_ * currDataCount * dataTypeSize_;

    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] loop[%llu] Inter1 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride,
        tempAlgParamsInter.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenInterTemplateParams2(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff
        = (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_)) * currDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff
        = (rankIdxLevel0_ + (rankIdxLevel1_ * rankSizeLevel0_)) * currDataCount * dataTypeSize_;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = rankSizeLevel0_ * rankSizeLevel1_ * currDataCount * dataTypeSize_;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = 1;
    tempAlgParamsInter.inputRepeatStride = 0;
    tempAlgParamsInter.outputRepeatStride = 0;

    HCCL_INFO(
        "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] loop[%llu] Inter2 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride,
        tempAlgParamsInter.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
template <typename InsAlgTemplate>
HcclResult InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenTempResource(
        const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level][GenTempResource] myRank[%u] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]",
            myRank_, channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    TemplateDataParams tempAlgParamsLevel0;
    tempAlgParamsLevel0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsLevel0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel0.buffInfo.hcclBuff = resCtx.cclMem;

    std::shared_ptr<InsAlgTemplate0> algTemplateLevel0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    if (rankSizeLevel0_ > 1) {
        CHK_RET(algTemplateLevel0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));
    }

    TemplateDataParams tempAlgParamsLevel1;
    tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.hcclBuff = resCtx.cclMem;

    std::shared_ptr<InsAlgTemplate1> algTemplateLevel1
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
    if (!skipLevel1_) {
        CHK_RET(algTemplateLevel1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    TemplateDataParams tempAlgParamsLevel2;
    tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsLevel2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsLevel2.buffInfo.hcclBuff = resCtx.cclMem;

    std::shared_ptr<InsAlgTemplate2> algTemplateLevel2
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
    CHK_RET(algTemplateLevel2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));

    u32 templateScratchMultiplier0 = algTemplateLevel0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier1
        = skipLevel1_ ? 1 : algTemplateLevel1->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier2
        = algTemplateLevel2->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    u32 templateScratchMultiplier
        = templateScratchMultiplier0 * templateScratchMultiplier1 * templateScratchMultiplier2;

    TemplateResource templateResource0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateLevel0, templateResource0));
    TemplateResource templateResource1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateLevel1, templateResource1));
    }
    TemplateResource templateResource2;
    CHK_RET(GenTempResource(resCtx, OMNIPIPE_LEVEL2_IDX, algTemplateLevel2, templateResource2));

    if (templateScratchMultiplier == 0) {
        HCCL_ERROR("[%s] templateScratchMultiplier is 0, division by zero.", __func__);
        return HCCL_E_INTERNAL;
    }
    u64 maxCountPerLoop = tempAlgParamsLevel2.buffInfo.hcclBuff.size / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN
                          * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[InsV2ReduceScatterSequenceExecutor3Level] myRank[%u] maxCountPerLoop is 0, "
            "scratchMultiplier[%u] too large for cclBuffSize[%llu]",
            myRank_, templateScratchMultiplier, tempAlgParamsLevel2.buffInfo.hcclBuff.size);
        return HCCL_E_INTERNAL;
    }
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        GenIntraTemplateParams(tempAlgParamsLevel0, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateLevel0->KernelRun(param, tempAlgParamsLevel0, templateResource0));

        if (!skipLevel1_) {
            GenInterTemplateParams1(tempAlgParamsLevel1, processedDataCount, currDataCount, loop);
            CHK_RET(algTemplateLevel1->KernelRun(param, tempAlgParamsLevel1, templateResource1));
        }

        GenInterTemplateParams2(tempAlgParamsLevel2, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateLevel2->KernelRun(param, tempAlgParamsLevel2, templateResource2));
        processedDataCount += currDataCount;
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
std::vector<CostModelParam>
InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)algName;
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // TODO: unused for now, costmodel fallback
    (void)algHierarchyInfo;
    // TODO: CalcAlgHierarchyInfo(comm, topoInfo, algHierarchyInfo);
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = true;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    u32 rankSizeLevel2 = rs.level2;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    // TODO: CommTopo netTypeLevel1 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[1]);
    CommTopo netTypeLevel1 = CommTopo::COMM_TOPO_CLOS;
    // TODO: CommTopo netTypeLevel2 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[2]);
    CommTopo netTypeLevel2 = CommTopo::COMM_TOPO_CLOS;
    // TODO: std::vector<u32> portNumLevel0 = GetPortNumLevel(topoInfo, algHierarchyInfo.index[0]);
    std::vector<u32> portNumLevel0 = {1};
    // TODO: std::vector<u32> portNumLevel1 = GetPortNumLevel(topoInfo, algHierarchyInfo.index[1]);
    std::vector<u32> portNumLevel1 = {8};
    // TODO: std::vector<u32> portNumLevel2 = GetPortNumLevel(topoInfo, algHierarchyInfo.index[2]);
    std::vector<u32> portNumLevel2 = {8};
    HCCL_INFO(
        "[CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, rankSizeLevel2=%d, "
        "portNumLevel0=%d, portNumLevel1=%d, portNumLevel2=%d, netTypeLevel0=%d, netTypeLevel1=%d, netTypeLevel2=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0, portNumLevel1, portNumLevel2,
        static_cast<int>(netTypeLevel0), static_cast<int>(netTypeLevel1), static_cast<int>(netTypeLevel2));
    std::vector<CostModelParam> params
        = [rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, rankSize, portNumLevel0, portNumLevel1, portNumLevel2,
           netTypeLevel0, netTypeLevel1, netTypeLevel2, isPod] {
              std::vector<CostModelParam> v;
              // Step1: 框内 RS（level0 mesh，全量输入的 1/rankSizeLevel0）
              auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f * rankSizeLevel1 * rankSizeLevel2 / rankSize, netTypeLevel0, BufferType::INPUT,
                  BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel0, isPod});
              // Step2: 框间 RS（level1 NHR，level0 归约后的 1/level0 份）
              auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel1, 1.0f * rankSizeLevel2 / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel1, isPod});
              // Step3: 跨 super-pod RS（level2 NHR，level1 归约后的 1/(level0*level1) 份）
              auto p2 = InsAlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel2, 1.0f / rankSize, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::OUTPUT,
                  BufferType::HCCL_BUFFER, portNumLevel2, isPod});
              // 任一 template 未实现 CalcCostCoeff（返回空）则整个算法不参与 CostModel
              if (p0.empty() || p1.empty() || p2.empty()) {
                  HCCL_WARNING(
                      "[InsV2ReduceScatterSequenceExecutor3Level] CalcCostCoeff incomplete, skip "
                      "(p0=%zu p1=%zu p2=%zu).",
                      p0.size(), p1.size(), p2.size());
                  return v;
              }
              v.insert(v.end(), p0.begin(), p0.end());
              v.insert(v.end(), p1.begin(), p1.end());
              v.insert(v.end(), p2.begin(), p2.end());
              return v;
          }();
    return params;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
AlgNetMeta InsV2ReduceScatterSequenceExecutor3Level<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)param;
    auto rs = CostModelManager::Global()->CalcRankSizeByTopo(topoInfo);
    u32 rankSizeLevel0 = rs.level0;
    u32 rankSizeLevel1 = rs.level1;
    u32 rankSizeLevel2 = rs.level2;
    u32 rankSize = topoInfo->userRankSize;
    // TODO: CommTopo netTypeLevel0 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[0]);
    CommTopo netTypeLevel0 = CommTopo::COMM_TOPO_1DMESH;
    // TODO: CommTopo netTypeLevel1 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[1]);
    CommTopo netTypeLevel1 = CommTopo::COMM_TOPO_CLOS;
    // TODO: CommTopo netTypeLevel2 = GetNetTypeLevel(topoInfo, algHierarchyInfo.index[2]);
    CommTopo netTypeLevel2 = CommTopo::COMM_TOPO_CLOS;
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel2);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1, 1};
    meta.dataRatios
        = {1.0f * rankSizeLevel1 * rankSizeLevel2 / rankSize, 1.0f * rankSizeLevel2 / rankSize, 1.0f / rankSize};
    meta.rankSizes = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel2};
    return meta;
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE_SCATTER, AicpuReduceScatterSequenceMeshConcurNHRNHR,
    InsV2ReduceScatterSequenceExecutor3Level, TopoMatchMultilevel, InsTempReduceScatterMesh1DZAxisDetour,
    InsTempReduceScatterNHR, InsTempReduceScatterNHR);
REGISTER_ALG_ATTRS(AicpuReduceScatterSequenceMeshConcurNHRNHR, topo.minTopoLevelNum = 3; topo.maxTopoLevelNum = 3;
                   op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT);

} // namespace ops_hccl
