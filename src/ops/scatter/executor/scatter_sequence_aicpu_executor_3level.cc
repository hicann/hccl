/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_sequence_aicpu_executor_3level.h"
#include "aicpu_temp_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_scatter_nhr.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 3;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
ScatterSequenceAicpu3LevelExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::ScatterSequenceAicpu3LevelExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
ScatterSequenceAicpu3LevelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[ScatterSequenceAicpu3LevelExecutor][InitCommInfo] myRank [%u], rankSize [%u], "
        "dataType [%u] dataTypeSize [%u]",
        myRank_, rankSize_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult ScatterSequenceAicpu3LevelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
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
HcclResult ScatterSequenceAicpu3LevelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[ScatterSequenceAicpu3LevelExecutor][CalcRes] myRank[%u] start", myRank_);
    InitCommInfo(param, topoInfo, algHierarchyInfo);

    skipLevel1_ = (algHierarchyInfo.infos[1][0].size() == 1);
    skipLevel2_ = (algHierarchyInfo.infos.size() == 2);
    std::shared_ptr<InsAlgTemplate0> tempAlgLevel0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> tempAlgLevel1
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> tempAlgLevel2;
    if (!skipLevel2_) {
        tempAlgLevel2 = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);
    }

    AlgResourceRequest resReq0;
    AlgResourceRequest resReq1;
    AlgResourceRequest resReq2;
    CHK_RET(tempAlgLevel0->CalcRes(comm, param, topoInfo, resReq0));
    if (skipLevel1_) {
        HCCL_INFO(
            "[ScatterSequenceAicpu3LevelExecutor][CalcRes] myRank[%u] level1 rankSize is 1, skip level1 CalcRes",
            myRank_);
    } else {
        CHK_RET(tempAlgLevel1->CalcRes(comm, param, topoInfo, resReq1));
    }
    if (skipLevel2_) {
        HCCL_INFO(
            "[ScatterSequenceAicpu3LevelExecutor][CalcRes] myRank[%u] level2 rankSize is 1, skip level2 CalcRes",
            myRank_);
    } else {
        CHK_RET(tempAlgLevel2->CalcRes(comm, param, topoInfo, resReq2));
    }

    auto maxThreadNum = resReq0.slaveThreadNum;
    if (!skipLevel1_) {
        maxThreadNum = std::max(maxThreadNum, resReq1.slaveThreadNum);
    }
    if (!skipLevel2_) {
        maxThreadNum = std::max(maxThreadNum, resReq2.slaveThreadNum);
    }
    resourceRequest.slaveThreadNum = maxThreadNum;
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
        if (!skipLevel2_ && i < resReq2.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReq2.notifyNumPerThread[i]);
        }
    }
    auto maxNotifyNum = resReq0.notifyNumOnMainThread;
    if (!skipLevel1_) {
        maxNotifyNum = std::max(maxNotifyNum, resReq1.notifyNumOnMainThread);
    }
    if (!skipLevel2_) {
        maxNotifyNum = std::max(maxNotifyNum, resReq2.notifyNumOnMainThread);
    }
    resourceRequest.notifyNumOnMainThread = maxNotifyNum;
    HCCL_INFO(
        "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] notifyNumOnMainThread is %u", myRank_,
        resourceRequest.notifyNumOnMainThread);
    resourceRequest.channels.resize(SEQUENCE_EXECUTOR_LEVEL_NUM);
    if (resReq0.channels.empty()) {
        HCCL_ERROR(
            "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] channels empty, level0[%u] level2[%u]", myRank_,
            resReq0.channels.size(), resReq2.channels.size());
        return HCCL_E_INTERNAL;
    }
    if (!skipLevel1_ && resReq1.channels.empty()) {
        HCCL_ERROR(
            "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] channels empty, level1[%u]", myRank_,
            resReq1.channels.size());
        return HCCL_E_INTERNAL;
    }
    if (!skipLevel2_ && resReq2.channels.empty()) {
        HCCL_ERROR(
            "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] channels empty, level2[%u]", myRank_,
            resReq2.channels.size());
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[0] = resReq0.channels[0];
    if (!skipLevel1_) {
        resourceRequest.channels[1] = resReq1.channels[0];
    }
    if (!skipLevel2_) {
        resourceRequest.channels[2] = resReq2.channels[0];
    }
    HCCL_INFO(
        "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] slaveThreadNum is [%u], notifyNumOnMainThread is [%u], "
        "level0 channel size [%u], level1 channel size [%u], level2 channel size [%u]",
        myRank_, resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread,
        resourceRequest.channels[0].size(), resourceRequest.channels[1].size(), resourceRequest.channels[2].size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
ScatterSequenceAicpu3LevelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    if (algHierarchyInfo_.infos.size() < 2 || algHierarchyInfo_.infos[0].empty() || algHierarchyInfo_.infos[1].empty()
        || algHierarchyInfo_.infos[0][0].empty() || algHierarchyInfo_.infos[1][0].empty()) {
        HCCL_ERROR("[%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = (myRank_ / algHierarchyInfo_.infos[0][0].size()) % algHierarchyInfo_.infos[1][0].size();
    rankIdxLevel2_ = myRank_ / (algHierarchyInfo_.infos[0][0].size() * algHierarchyInfo_.infos[1][0].size());

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();

    skipLevel1_ = (rankSizeLevel1_ == 1);
    skipLevel2_ = (algHierarchyInfo_.infos.size() == 2);
    if (skipLevel1_) {
        HCCL_INFO(
            "[ScatterSequenceAicpu3LevelExecutor] [Orchestrate] myRank[%u] level1 rankSize is 1, skip level1", myRank_);
    }
    if (skipLevel2_) {
        rankSizeLevel2_ = 1;
        HCCL_INFO(
            "[ScatterSequenceAicpu3LevelExecutor] [Orchestrate] myRank[%u] level2 rankSize is 1, skip level2", myRank_);
    } else {
        rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
    }
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    HCCL_INFO(
        "[ScatterSequenceAicpu3LevelExecutor] [Orchestrate] myRank_[%u] rankIdxLevel0_[%u] "
        "rankIdxLevel1_[%u] rankIdxLevel2_[%u] rankSizeLevel0_[%u] rankSizeLevel1_[%u] "
        "rankSizeLevel2_[%u]",
        myRank_, rankIdxLevel0_, rankIdxLevel1_, rankIdxLevel2_, rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_);
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ScatterSequenceAicpu3LevelExecutor][Orchestrate] myRank[%u] errNo[0x%016llx] "
            "scatter excutor kernel run failed",
            myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void ScatterSequenceAicpu3LevelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
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
    tempAlgParamsIntra.outputSliceStride = 0;
    tempAlgParamsIntra.repeatNum = rankSizeLevel1_ * rankSizeLevel2_;
    tempAlgParamsIntra.inputRepeatStride = rankSizeLevel0_ * dataSize_;
    tempAlgParamsIntra.outputRepeatStride = currDataCount * dataTypeSize_;

    HCCL_INFO(
        "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] loop[%llu] Intra inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsIntra.inputSliceStride, tempAlgParamsIntra.outputSliceStride,
        tempAlgParamsIntra.sliceSize, tempAlgParamsIntra.buffInfo.inBuffBaseOff,
        tempAlgParamsIntra.buffInfo.outBuffBaseOff, tempAlgParamsIntra.repeatNum, tempAlgParamsIntra.inputRepeatStride,
        tempAlgParamsIntra.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void ScatterSequenceAicpu3LevelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenInterTemplateParams1(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = skipLevel2_ ? processedDataCount * dataTypeSize_ : 0;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = 0;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = rankSizeLevel2_;
    tempAlgParamsInter.inputRepeatStride = 0;

    HCCL_INFO(
        "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] loop[%llu] Inter1 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride,
        tempAlgParamsInter.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
void ScatterSequenceAicpu3LevelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenInterTemplateParams2(
        TemplateDataParams& tempAlgParamsInter, const u64 processedDataCount, const u64 currDataCount,
        const u64 loop) const
{
    tempAlgParamsInter.count = currDataCount;
    tempAlgParamsInter.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsInter.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsInter.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsInter.sliceSize = currDataCount * dataTypeSize_;
    tempAlgParamsInter.tailSize = tempAlgParamsInter.sliceSize;

    tempAlgParamsInter.inputSliceStride = 0;
    tempAlgParamsInter.outputSliceStride = 0;
    tempAlgParamsInter.repeatNum = 1;
    tempAlgParamsInter.inputRepeatStride = 0;

    HCCL_INFO(
        "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] loop[%llu] Inter2 inputSliceStride[%llu] "
        "outputSliceStride[%llu] sliceSize[%llu] inBuffBaseOff[%llu] outBuffBaseOff[%llu] "
        "repeatNum[%llu] inputRepeatStride[%llu] outputRepeatStride[%llu]",
        myRank_, loop, tempAlgParamsInter.inputSliceStride, tempAlgParamsInter.outputSliceStride,
        tempAlgParamsInter.sliceSize, tempAlgParamsInter.buffInfo.inBuffBaseOff,
        tempAlgParamsInter.buffInfo.outBuffBaseOff, tempAlgParamsInter.repeatNum, tempAlgParamsInter.inputRepeatStride,
        tempAlgParamsInter.outputRepeatStride);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
template <typename InsAlgTemplate>
HcclResult
ScatterSequenceAicpu3LevelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GenTempResource(
    const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
    const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR(
            "[ScatterSequenceAicpu3LevelExecutor][GenTempResource] myRank[%u] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]",
            myRank_, channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
ScatterSequenceAicpu3LevelExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    u32 podSize = rankSizeLevel0_ * rankSizeLevel1_;
    u32 rootPodStartRank = param.root / podSize * podSize;
    u32 rootPodEndRank = rootPodStartRank + podSize - 1;
    TemplateDataParams tempAlgParamsLevel0;
    tempAlgParamsLevel0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsLevel0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel0.buffInfo.hcclBuff = resCtx.cclMem;
    if (skipLevel1_ && skipLevel2_) {
        tempAlgParamsLevel0.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsLevel0.buffInfo.outputPtr = param.outputPtr;
    }

    std::shared_ptr<InsAlgTemplate0> algTemplateLevel0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    if (rankSizeLevel0_ > 1) {
        CHK_RET(algTemplateLevel0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));
    }

    TemplateDataParams tempAlgParamsLevel1;
    tempAlgParamsLevel1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel1.buffInfo.hcclBuff = resCtx.cclMem;
    if (skipLevel2_) {
        tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::OUTPUT;
        tempAlgParamsLevel1.buffInfo.outputPtr = param.outputPtr;
    } else {
        tempAlgParamsLevel1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsLevel1.buffInfo.outputPtr = resCtx.cclMem.addr;
    }

    std::shared_ptr<InsAlgTemplate1> algTemplateLevel1 = nullptr;
    if (!skipLevel1_) {
        algTemplateLevel1 = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
        algTemplateLevel1->SetRoot((param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_));
        CHK_RET(algTemplateLevel1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    TemplateDataParams tempAlgParamsLevel2;
    tempAlgParamsLevel2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsLevel2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsLevel2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsLevel2.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsLevel2.buffInfo.hcclBuff = resCtx.cclMem;

    std::shared_ptr<InsAlgTemplate2> algTemplateLevel2 = nullptr;
    if (!skipLevel2_) {
        algTemplateLevel2 = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
        algTemplateLevel2->SetRoot(param.root / podSize * podSize + myRank_ % podSize);
        CHK_RET(algTemplateLevel2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));
    }

    u32 templateScratchMultiplier0 = algTemplateLevel0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier1
        = skipLevel1_ ? 1 : algTemplateLevel1->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 templateScratchMultiplier2
        = skipLevel2_ ? 1 : algTemplateLevel2->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::OUTPUT);
    u32 templateScratchMultiplier
        = templateScratchMultiplier0 * templateScratchMultiplier1 * templateScratchMultiplier2;

    TemplateResource templateResource0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateLevel0, templateResource0));
    TemplateResource templateResource1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateLevel1, templateResource1));
    }
    TemplateResource templateResource2;
    if (!skipLevel2_) {
        CHK_RET(GenTempResource(resCtx, 2, algTemplateLevel2, templateResource2));
    }

    if (templateScratchMultiplier == 0) {
        HCCL_ERROR(
            "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] templateScratchMultiplier is 0, "
            "cannot compute maxCountPerLoop.",
            myRank_);
        return HCCL_E_INTERNAL;
    }
    u64 maxCountPerLoop = tempAlgParamsLevel0.buffInfo.hcclBuff.size / templateScratchMultiplier / AICPU_ALIGN_SIZE
                          * AICPU_ALIGN_SIZE / dataTypeSize_;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[ScatterSequenceAicpu3LevelExecutor] myRank[%u] maxCountPerLoop is 0, "
            "scratchMultiplier[%u] too large for cclBuffSize[%llu]",
            myRank_, templateScratchMultiplier, tempAlgParamsLevel0.buffInfo.hcclBuff.size);
        return HCCL_E_INTERNAL;
    }
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        GenIntraTemplateParams(tempAlgParamsLevel0, processedDataCount, currDataCount, loop);
        CHK_RET(algTemplateLevel0->KernelRun(param, tempAlgParamsLevel0, templateResource0));

        if (!skipLevel1_ && myRank_ >= rootPodStartRank && myRank_ <= rootPodEndRank) {
            GenInterTemplateParams1(tempAlgParamsLevel1, processedDataCount, currDataCount, loop);
            CHK_RET(algTemplateLevel1->KernelRun(param, tempAlgParamsLevel1, templateResource1));
        }
        if (!skipLevel2_) {
            GenInterTemplateParams2(tempAlgParamsLevel2, processedDataCount, currDataCount, loop);
            CHK_RET(algTemplateLevel2->KernelRun(param, tempAlgParamsLevel2, templateResource2));
        }
        processedDataCount += currDataCount;
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_SCATTER, AicpuScatterSequenceMesh1DNHRNHR, ScatterSequenceAicpu3LevelExecutor,
    TopoMatchMultilevel, AicpuTempScatterMesh1DZAxisDetour, InsTempScatterNHR, InsTempScatterNHR);

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_SCATTER, AicpuScatterSequenceMesh1DNHR, ScatterSequenceAicpu3LevelExecutor,
    TopoMatchMultilevel, AicpuTempScatterMesh1DZAxisDetour, InsTempScatterNHR, InsTempScatterNHR);
} // namespace ops_hccl
