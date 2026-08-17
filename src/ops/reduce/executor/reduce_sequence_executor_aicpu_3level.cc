/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_sequence_executor_aicpu_3level.h"
#include <algorithm>
#include "alg_data_trans_wrapper.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"

namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 3;
constexpr u32 SEQUENCE_EXECUTOR_MIN_LEVEL_NUM = 2;
constexpr u32 LEVEL2_IDX = 2;

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4,
    AlgTemplate5>::ReduceSequenceExecutorAicpu3Level()
{}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
HcclResult ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    InitCommInfo(
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
        "[ReduceSequenceExecutorAicpu3Level][InitCommInfo] myRank [%u], rankSize [%u], redOp [%u], "
        "dataType [%u] dataTypeSize [%llu]",
        myRank_, rankSize_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
HcclResult ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
HcclResult ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    if (algHierarchyInfo.infos.size() < SEQUENCE_EXECUTOR_MIN_LEVEL_NUM
        || algHierarchyInfo.infos.size() > SEQUENCE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR(
            "[ReduceSequenceExecutorAicpu3Level] algHierarchyInfo size[%u] should be 2 or 3",
            algHierarchyInfo.infos.size());
        return HCCL_E_INTERNAL;
    }
    if (algHierarchyInfo.infos[0].empty() || algHierarchyInfo.infos[1].empty() || algHierarchyInfo.infos[0][0].empty()
        || algHierarchyInfo.infos[1][0].empty()
        || ((algHierarchyInfo.infos.size() == SEQUENCE_EXECUTOR_LEVEL_NUM)
            && (algHierarchyInfo.infos[2].empty() || algHierarchyInfo.infos[2][0].empty()))) {
        HCCL_ERROR("[%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    skipLevel2_ = (algHierarchyInfo.infos.size() == SEQUENCE_EXECUTOR_MIN_LEVEL_NUM);
    if (!skipLevel2_) {
        rankSizeLevel2_ = algHierarchyInfo.infos[2][0].size();
        skipLevel2_ = (rankSizeLevel2_ == 1);
    }

    std::shared_ptr<AlgTemplate0> rsL0TempAlg
        = std::make_shared<AlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<AlgTemplate1> rsL1TempAlg;
    if (!skipLevel1_) {
        rsL1TempAlg = std::make_shared<AlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    }
    std::shared_ptr<AlgTemplate2> rsL2TempAlg;
    std::shared_ptr<AlgTemplate3> agL2TempAlg;
    if (!skipLevel2_) {
        rsL2TempAlg = std::make_shared<AlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);
        agL2TempAlg = std::make_shared<AlgTemplate3>(param, myRank_, algHierarchyInfo.infos[2]);
    }
    std::shared_ptr<AlgTemplate4> agL1TempAlg;
    if (!skipLevel1_) {
        agL1TempAlg = std::make_shared<AlgTemplate4>(param, myRank_, algHierarchyInfo.infos[1]);
    }
    std::shared_ptr<AlgTemplate5> agL0TempAlg
        = std::make_shared<AlgTemplate5>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqRSL0;
    AlgResourceRequest resReqRSL1;
    AlgResourceRequest resReqRSL2;
    AlgResourceRequest resReqAGL2;
    AlgResourceRequest resReqAGL1;
    AlgResourceRequest resReqAGL0;
    CHK_RET(rsL0TempAlg->CalcRes(comm, param, topoInfo, resReqRSL0));
    if (!skipLevel1_) {
        CHK_RET(rsL1TempAlg->CalcRes(comm, param, topoInfo, resReqRSL1));
        CHK_RET(agL1TempAlg->CalcRes(comm, param, topoInfo, resReqAGL1));
    }
    if (!skipLevel2_) {
        CHK_RET(rsL2TempAlg->CalcRes(comm, param, topoInfo, resReqRSL2));
        CHK_RET(agL2TempAlg->CalcRes(comm, param, topoInfo, resReqAGL2));
    }
    CHK_RET(agL0TempAlg->CalcRes(comm, param, topoInfo, resReqAGL0));

    std::vector<AlgResourceRequest> activeReqs = {resReqRSL0, resReqAGL0};
    if (!skipLevel1_) {
        activeReqs.push_back(resReqRSL1);
        activeReqs.push_back(resReqAGL1);
    }
    if (!skipLevel2_) {
        activeReqs.push_back(resReqRSL2);
        activeReqs.push_back(resReqAGL2);
    }
    resourceRequest.slaveThreadNum = 0;
    for (auto& req : activeReqs) {
        resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, req.slaveThreadNum);
    }
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        for (auto& req : activeReqs) {
            if (i < req.notifyNumPerThread.size()) {
                resourceRequest.notifyNumPerThread[i]
                    = std::max(resourceRequest.notifyNumPerThread[i], req.notifyNumPerThread[i]);
            }
        }
    }
    u32 mainNotifyNum = 0;
    for (auto& req : activeReqs) {
        mainNotifyNum = std::max(mainNotifyNum, req.notifyNumOnMainThread);
    }
    resourceRequest.notifyNumOnMainThread = mainNotifyNum;

    resourceRequest.channels.resize(algHierarchyInfo.infos.size());
    if (resReqRSL0.channels.empty()) {
        HCCL_ERROR("[ReduceSequenceExecutorAicpu3Level] level0 channels empty");
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[0] = resReqRSL0.channels[0];
    if (!skipLevel1_) {
        if (resReqRSL1.channels.empty()) {
            HCCL_ERROR("[ReduceSequenceExecutorAicpu3Level] level1 channels empty");
            return HCCL_E_INTERNAL;
        }
        resourceRequest.channels[1] = resReqRSL1.channels[0];
    }
    if (!skipLevel2_) {
        if (resReqRSL2.channels.empty()) {
            HCCL_ERROR("[ReduceSequenceExecutorAicpu3Level] level2 channels empty");
            return HCCL_E_INTERNAL;
        }
        resourceRequest.channels[LEVEL2_IDX] = resReqRSL2.channels[0];
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
HcclResult ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4,
    AlgTemplate5>::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[ReduceSequenceExecutorAicpu3Level][Orchestrate] Orchestrate Start");
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    CHK_PRT_RET(
        algHierarchyInfo_.infos.size() < SEQUENCE_EXECUTOR_MIN_LEVEL_NUM
            || algHierarchyInfo_.infos.size() > SEQUENCE_EXECUTOR_LEVEL_NUM,
        HCCL_ERROR(
            "[ReduceSequenceExecutorAicpu3Level] algHierarchyInfo size[%u] should be 2 or 3",
            algHierarchyInfo_.infos.size()),
        HCCL_E_INTERNAL);

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    skipLevel2_ = (algHierarchyInfo_.infos.size() == SEQUENCE_EXECUTOR_MIN_LEVEL_NUM);
    if (!skipLevel2_) {
        rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
        skipLevel2_ = (rankSizeLevel2_ == 1);
    }
    if (skipLevel1_) {
        HCCL_INFO("[ReduceSequenceExecutorAicpu3Level][Orchestrate] level1 rankSize is 1, skip level1");
    }
    if (skipLevel2_) {
        HCCL_INFO("[ReduceSequenceExecutorAicpu3Level][Orchestrate] skip level2");
    }
    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = (myRank_ / algHierarchyInfo_.infos[0][0].size()) % algHierarchyInfo_.infos[1][0].size();

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[ReduceSequenceExecutorAicpu3Level][Orchestrate]errNo[0x%016llx] "
            "Reduce executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
void ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    GenBaseTempAlgParams(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParamsRSL0,
        TemplateDataParams& tempAlgParamsRSL1, TemplateDataParams& tempAlgParamsRSL2,
        TemplateDataParams& tempAlgParamsAGL2, TemplateDataParams& tempAlgParamsAGL1,
        TemplateDataParams& tempAlgParamsAGL0) const
{
    tempAlgParamsRSL0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsRSL0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsRSL0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsRSL0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsRSL0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsRSL0.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsRSL1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsRSL1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsRSL1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsRSL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsRSL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsRSL1.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsRSL2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsRSL2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsRSL2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsRSL2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsRSL2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsRSL2.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsAGL2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL2.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL2.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL2.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsAGL1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL1.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsAGL0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAGL0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAGL0.buffInfo.hcclBuff = resCtx.cclMem;
    return;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
void ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    GenTempAlgParamsRSL0(
        const u64 loop, const u64 currDataCount, const u64 processedDataCount,
        TemplateDataParams& tempAlgParamsRSL0) const
{
    tempAlgParamsRSL0.count = currDataCount;
    tempAlgParamsRSL0.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsRSL0.buffInfo.outBuffBaseOff = rsResultBuffOffset_;
    tempAlgParamsRSL0.buffInfo.hcclBuffBaseOff = meshCommBuffOffset_;

    tempAlgParamsRSL0.sliceSize = currDataCount / rankSizeLevel0_ * dataTypeSize_;
    tempAlgParamsRSL0.tailSize = (currDataCount / rankSizeLevel0_ + currDataCount % rankSizeLevel0_) * dataTypeSize_;

    tempAlgParamsRSL0.inputSliceStride = tempAlgParamsRSL0.sliceSize;
    tempAlgParamsRSL0.outputSliceStride = 0;

    HCCL_INFO(
        "[ReduceSequenceExecutorAicpu3Level] loop [%llu] RSL0.inputSliceStride [%llu], "
        "RSL0.outputSliceStride [%llu], RSL0.sliceSize [%llu], RSL0.tailSize [%llu], "
        "RSL0.inBuffBaseOff [%llu], RSL0.outBuffBaseOff [%llu]",
        loop, tempAlgParamsRSL0.inputSliceStride, tempAlgParamsRSL0.outputSliceStride, tempAlgParamsRSL0.sliceSize,
        tempAlgParamsRSL0.tailSize, tempAlgParamsRSL0.buffInfo.inBuffBaseOff,
        tempAlgParamsRSL0.buffInfo.outBuffBaseOff);

    tempAlgParamsRSL0.repeatNum = 1;
    tempAlgParamsRSL0.inputRepeatStride = 0;
    tempAlgParamsRSL0.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
void ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    GenTempAlgParamsRSL1(
        const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL0, const u64 tailSizeRSL0,
        TemplateDataParams& tempAlgParamsRSL1) const
{
    tempAlgParamsRSL1.count = currDataCount;
    if (rankIdxLevel0_ == rankSizeLevel0_ - 1) {
        u64 tailCountRSL0 = tailSizeRSL0 / dataTypeSize_;
        tempAlgParamsRSL1.sliceSize = tailCountRSL0 / rankSizeLevel1_ * dataTypeSize_;
        tempAlgParamsRSL1.tailSize = tempAlgParamsRSL1.sliceSize + tailCountRSL0 % rankSizeLevel1_ * dataTypeSize_;
    } else {
        u64 sliceCountRSL0 = sliceSizeRSL0 / dataTypeSize_;
        tempAlgParamsRSL1.sliceSize = sliceCountRSL0 / rankSizeLevel1_ * dataTypeSize_;
        tempAlgParamsRSL1.tailSize = tempAlgParamsRSL1.sliceSize + sliceCountRSL0 % rankSizeLevel1_ * dataTypeSize_;
    }
    tempAlgParamsRSL1.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsRSL1.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsRSL1.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsRSL1.inputSliceStride = tempAlgParamsRSL1.sliceSize;
    tempAlgParamsRSL1.outputSliceStride = tempAlgParamsRSL1.sliceSize;

    HCCL_INFO(
        "[ReduceSequenceExecutorAicpu3Level] loop [%llu] RSL1.inputSliceStride [%llu], "
        "RSL1.outputSliceStride [%llu], RSL1.sliceSize [%llu], RSL1.tailSize [%llu], "
        "RSL1.inBuffBaseOff [%llu], RSL1.outBuffBaseOff [%llu]",
        loop, tempAlgParamsRSL1.inputSliceStride, tempAlgParamsRSL1.outputSliceStride, tempAlgParamsRSL1.sliceSize,
        tempAlgParamsRSL1.tailSize, tempAlgParamsRSL1.buffInfo.inBuffBaseOff,
        tempAlgParamsRSL1.buffInfo.outBuffBaseOff);

    tempAlgParamsRSL1.repeatNum = 1;
    tempAlgParamsRSL1.inputRepeatStride = 0;
    tempAlgParamsRSL1.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
void ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    GenTempAlgParamsRSL2(
        const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL1, const u64 tailSizeRSL1,
        TemplateDataParams& tempAlgParamsRSL2) const
{
    tempAlgParamsRSL2.count = currDataCount;
    if (rankIdxLevel1_ == rankSizeLevel1_ - 1) {
        u64 tailCountRSL1 = tailSizeRSL1 / dataTypeSize_;
        tempAlgParamsRSL2.sliceSize = tailCountRSL1 / rankSizeLevel2_ * dataTypeSize_;
        tempAlgParamsRSL2.tailSize = tempAlgParamsRSL2.sliceSize + tailCountRSL1 % rankSizeLevel2_ * dataTypeSize_;
    } else {
        u64 sliceCountRSL1 = sliceSizeRSL1 / dataTypeSize_;
        tempAlgParamsRSL2.sliceSize = sliceCountRSL1 / rankSizeLevel2_ * dataTypeSize_;
        tempAlgParamsRSL2.tailSize = tempAlgParamsRSL2.sliceSize + sliceCountRSL1 % rankSizeLevel2_ * dataTypeSize_;
    }
    tempAlgParamsRSL2.buffInfo.inBuffBaseOff = rankIdxLevel1_ * sliceSizeRSL1;
    tempAlgParamsRSL2.buffInfo.outBuffBaseOff = rankIdxLevel1_ * sliceSizeRSL1;
    tempAlgParamsRSL2.buffInfo.hcclBuffBaseOff = rankIdxLevel1_ * sliceSizeRSL1;

    tempAlgParamsRSL2.inputSliceStride = tempAlgParamsRSL2.sliceSize;
    tempAlgParamsRSL2.outputSliceStride = tempAlgParamsRSL2.sliceSize;

    HCCL_INFO(
        "[ReduceSequenceExecutorAicpu3Level] loop [%llu] RSL2.inputSliceStride [%llu], "
        "RSL2.outputSliceStride [%llu], RSL2.sliceSize [%llu], RSL2.tailSize [%llu], "
        "RSL2.inBuffBaseOff [%llu], RSL2.outBuffBaseOff [%llu]",
        loop, tempAlgParamsRSL2.inputSliceStride, tempAlgParamsRSL2.outputSliceStride, tempAlgParamsRSL2.sliceSize,
        tempAlgParamsRSL2.tailSize, tempAlgParamsRSL2.buffInfo.inBuffBaseOff,
        tempAlgParamsRSL2.buffInfo.outBuffBaseOff);

    tempAlgParamsRSL2.repeatNum = 1;
    tempAlgParamsRSL2.inputRepeatStride = 0;
    tempAlgParamsRSL2.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
void ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    GenTempAlgParamsAGL2(
        const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL2, const u64 tailSizeRSL2,
        const u64 sliceSizeRSL1, TemplateDataParams& tempAlgParamsAGL2) const
{
    tempAlgParamsAGL2.count = currDataCount;
    tempAlgParamsAGL2.buffInfo.inBuffBaseOff = rankIdxLevel1_ * sliceSizeRSL1;
    tempAlgParamsAGL2.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsAGL2.buffInfo.hcclBuffBaseOff = rankIdxLevel1_ * sliceSizeRSL1;

    tempAlgParamsAGL2.sliceSize = sliceSizeRSL2;
    tempAlgParamsAGL2.tailSize = tailSizeRSL2;

    tempAlgParamsAGL2.inputSliceStride = tempAlgParamsAGL2.sliceSize;
    tempAlgParamsAGL2.outputSliceStride = sliceSizeRSL1;

    HCCL_INFO(
        "[ReduceSequenceExecutorAicpu3Level] loop [%llu] AGL2.inputSliceStride [%llu], "
        "AGL2.outputSliceStride [%llu], AGL2.sliceSize [%llu], AGL2.tailSize [%llu], "
        "AGL2.inBuffBaseOff [%llu], AGL2.outBuffBaseOff [%llu]",
        loop, tempAlgParamsAGL2.inputSliceStride, tempAlgParamsAGL2.outputSliceStride, tempAlgParamsAGL2.sliceSize,
        tempAlgParamsAGL2.tailSize, tempAlgParamsAGL2.buffInfo.inBuffBaseOff,
        tempAlgParamsAGL2.buffInfo.outBuffBaseOff);

    tempAlgParamsAGL2.repeatNum = 1;
    tempAlgParamsAGL2.inputRepeatStride = 0;
    tempAlgParamsAGL2.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
void ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    GenTempAlgParamsAGL1(
        const u64 loop, const u64 currDataCount, const u64 sliceSize, const u64 tailSize,
        TemplateDataParams& tempAlgParamsAGL1) const
{
    tempAlgParamsAGL1.count = currDataCount;
    tempAlgParamsAGL1.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsAGL1.buffInfo.outBuffBaseOff = 0;
    tempAlgParamsAGL1.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsAGL1.sliceSize = sliceSize;
    tempAlgParamsAGL1.tailSize = tailSize;

    tempAlgParamsAGL1.inputSliceStride = tempAlgParamsAGL1.sliceSize;
    tempAlgParamsAGL1.outputSliceStride = 0;

    HCCL_INFO(
        "[ReduceSequenceExecutorAicpu3Level] loop [%llu] AGL1.inputSliceStride [%llu], "
        "AGL1.outputSliceStride [%llu], AGL1.sliceSize [%llu], AGL1.tailSize [%llu], "
        "AGL1.inBuffBaseOff [%llu], AGL1.outBuffBaseOff [%llu]",
        loop, tempAlgParamsAGL1.inputSliceStride, tempAlgParamsAGL1.outputSliceStride, tempAlgParamsAGL1.sliceSize,
        tempAlgParamsAGL1.tailSize, tempAlgParamsAGL1.buffInfo.inBuffBaseOff,
        tempAlgParamsAGL1.buffInfo.outBuffBaseOff);

    tempAlgParamsAGL1.repeatNum = 1;
    tempAlgParamsAGL1.inputRepeatStride = 0;
    tempAlgParamsAGL1.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
void ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    GenTempAlgParamsAGL0(
        const u64 loop, const u64 currDataCount, const u64 sliceSize, const u64 tailSize,
        TemplateDataParams& tempAlgParamsAGL0) const
{
    tempAlgParamsAGL0.count = currDataCount;
    tempAlgParamsAGL0.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsAGL0.buffInfo.outBuffBaseOff = meshCommBuffOffset_;
    tempAlgParamsAGL0.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsAGL0.sliceSize = sliceSize;
    tempAlgParamsAGL0.tailSize = tailSize;

    tempAlgParamsAGL0.inputSliceStride = 0;
    tempAlgParamsAGL0.outputSliceStride = tempAlgParamsAGL0.sliceSize;

    HCCL_INFO(
        "[ReduceSequenceExecutorAicpu3Level] loop [%llu] AGL0.inputSliceStride [%llu], "
        "AGL0.outputSliceStride [%llu], AGL0.sliceSize [%llu], AGL0.tailSize [%llu], "
        "AGL0.inBuffBaseOff [%llu], AGL0.outBuffBaseOff [%llu]",
        loop, tempAlgParamsAGL0.inputSliceStride, tempAlgParamsAGL0.outputSliceStride, tempAlgParamsAGL0.sliceSize,
        tempAlgParamsAGL0.tailSize, tempAlgParamsAGL0.buffInfo.inBuffBaseOff,
        tempAlgParamsAGL0.buffInfo.outBuffBaseOff);

    tempAlgParamsAGL0.repeatNum = 1;
    tempAlgParamsAGL0.inputRepeatStride = 0;
    tempAlgParamsAGL0.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
template <typename AlgTemplate>
HcclResult ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    GenTempResource(
        const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<AlgTemplate>& algTemplate, TemplateResource& tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR(
            "[ReduceSequenceExecutorAicpu3Level][GenTempResource] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]",
            channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
HcclResult ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4,
    AlgTemplate5>::OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[ReduceSequenceExecutorAicpu3Level][OrchestrateLoop] Start");

    TemplateDataParams tempAlgParamsRSL0;
    TemplateDataParams tempAlgParamsRSL1;
    TemplateDataParams tempAlgParamsRSL2;
    TemplateDataParams tempAlgParamsAGL2;
    TemplateDataParams tempAlgParamsAGL1;
    TemplateDataParams tempAlgParamsAGL0;
    GenBaseTempAlgParams(
        param, resCtx, tempAlgParamsRSL0, tempAlgParamsRSL1, tempAlgParamsRSL2, tempAlgParamsAGL2, tempAlgParamsAGL1,
        tempAlgParamsAGL0);

    std::shared_ptr<AlgTemplate0> algTemplateRSL0
        = std::make_shared<AlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateRSL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    std::shared_ptr<AlgTemplate1> algTemplateRSL1;
    if (!skipLevel1_) {
        algTemplateRSL1 = std::make_shared<AlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
        CHK_RET(algTemplateRSL1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    std::shared_ptr<AlgTemplate2> algTemplateRSL2;
    if (!skipLevel2_) {
        algTemplateRSL2 = std::make_shared<AlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
        CHK_RET(algTemplateRSL2->SetchannelsPerRank(remoteRankToChannelInfo_[LEVEL2_IDX]));
    }

    std::shared_ptr<AlgTemplate3> algTemplateAGL2;
    if (!skipLevel2_) {
        algTemplateAGL2 = std::make_shared<AlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[2]);
        CHK_RET(algTemplateAGL2->SetchannelsPerRank(remoteRankToChannelInfo_[LEVEL2_IDX]));
    }

    std::shared_ptr<AlgTemplate4> algTemplateAGL1;
    if (!skipLevel1_) {
        algTemplateAGL1 = std::make_shared<AlgTemplate4>(param, myRank_, algHierarchyInfo_.infos[1]);
        CHK_RET(algTemplateAGL1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    std::shared_ptr<AlgTemplate5> algTemplateAGL0
        = std::make_shared<AlgTemplate5>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateAGL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    TemplateResource templateResourceRSL0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateRSL0, templateResourceRSL0));
    TemplateResource templateResourceRSL1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateRSL1, templateResourceRSL1));
    }
    TemplateResource templateResourceRSL2;
    if (!skipLevel2_) {
        CHK_RET(GenTempResource(resCtx, LEVEL2_IDX, algTemplateRSL2, templateResourceRSL2));
    }
    TemplateResource templateResourceAGL2;
    if (!skipLevel2_) {
        CHK_RET(GenTempResource(resCtx, LEVEL2_IDX, algTemplateAGL2, templateResourceAGL2));
    }
    TemplateResource templateResourceAGL1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateAGL1, templateResourceAGL1));
    }
    TemplateResource templateResourceAGL0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateAGL0, templateResourceAGL0));

    u64 scratchMultiplier = algTemplateRSL0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 cclBuffSliceNum = scratchMultiplier + 1;
    // 来自local reduce的地址需要按照数据类型对齐，防止精度问题
    cclBuffSliceSize_ = resCtx.cclMem.size / cclBuffSliceNum / dataTypeSize_ * dataTypeSize_;
    rsResultBuffSize_ = cclBuffSliceSize_;
    meshCommBuffSize_ = scratchMultiplier * cclBuffSliceSize_;
    rsResultBuffOffset_ = 0;
    meshCommBuffOffset_ = rsResultBuffSize_;
    u32 totalRankAlign = rankSizeLevel0_ * rankSizeLevel1_;
    if (!skipLevel2_) {
        totalRankAlign *= rankSizeLevel2_;
    }
    u64 maxCountPerLoop
        = meshCommBuffSize_ / AICPU_ALIGN_SIZE * AICPU_ALIGN_SIZE / dataTypeSize_ / totalRankAlign * totalRankAlign;
    CHK_PRT_RET(
        maxCountPerLoop == 0, HCCL_ERROR("[ReduceSequenceExecutorAicpu3Level] maxCountPerLoop is 0"), HCCL_E_INTERNAL);
    u64 processedDataCount = 0;
    u64 loop = 0;
    while (processedDataCount < dataCount_) {
        u64 remaining = dataCount_ - processedDataCount;
        u64 currDataCount;
        if (remaining <= maxCountPerLoop) {
            currDataCount = remaining;
            u64 q = currDataCount / rankSizeLevel0_;
            u64 r = currDataCount % rankSizeLevel0_;
            u64 tailSize = (q + r) * dataTypeSize_;
            if (tailSize > rsResultBuffSize_ && q > 0) {
                u64 maxTailElements = rsResultBuffSize_ / dataTypeSize_;
                if (maxTailElements == 0) {
                    HCCL_ERROR(
                        "[ReduceSequenceExecutorAicpu3Level] rsResultBuffSize_[%llu] is smaller than "
                        "dataTypeSize_[%llu], buffer too small",
                        rsResultBuffSize_, dataTypeSize_);
                    return HCCL_E_INTERNAL;
                }
                u64 newQ = q - 1;
                u64 newR;
                if (newQ >= maxTailElements) {
                    newQ = maxTailElements;
                    newR = 0;
                } else {
                    newR = std::min(static_cast<u64>(rankSizeLevel0_ - 1), maxTailElements - newQ);
                }
                currDataCount = newQ * rankSizeLevel0_ + newR;
            }
        } else {
            currDataCount = maxCountPerLoop;
        }

        // ----------- RSL0: level0 ReduceScatter -----------
        GenTempAlgParamsRSL0(loop, currDataCount, processedDataCount, tempAlgParamsRSL0);
        CHK_RET(algTemplateRSL0->KernelRun(param, tempAlgParamsRSL0, templateResourceRSL0));

        // ----------- RSL1: level1 ReduceScatter -----------
        u64 sliceSizeRSL1 = tempAlgParamsRSL0.sliceSize;
        u64 tailSizeRSL1 = tempAlgParamsRSL0.tailSize;
        if (!skipLevel1_) {
            GenTempAlgParamsRSL1(
                loop, currDataCount, tempAlgParamsRSL0.sliceSize, tempAlgParamsRSL0.tailSize, tempAlgParamsRSL1);
            CHK_RET(algTemplateRSL1->KernelRun(param, tempAlgParamsRSL1, templateResourceRSL1));
            sliceSizeRSL1 = tempAlgParamsRSL1.sliceSize;
            tailSizeRSL1 = tempAlgParamsRSL1.tailSize;
        } else {
            sliceSizeRSL1 = tempAlgParamsRSL0.sliceSize;
            tailSizeRSL1
                = (rankIdxLevel0_ == rankSizeLevel0_ - 1) ? tempAlgParamsRSL0.tailSize : tempAlgParamsRSL0.sliceSize;
        }

        // ----------- RSL2: level2 ReduceScatter -----------
        if (!skipLevel2_) {
            GenTempAlgParamsRSL2(loop, currDataCount, sliceSizeRSL1, tailSizeRSL1, tempAlgParamsRSL2);
            CHK_RET(algTemplateRSL2->KernelRun(param, tempAlgParamsRSL2, templateResourceRSL2));

            // ----------- AGL2: level2 AllGather -----------
            GenTempAlgParamsAGL2(
                loop, currDataCount, tempAlgParamsRSL2.sliceSize, tempAlgParamsRSL2.tailSize, sliceSizeRSL1,
                tempAlgParamsAGL2);
            CHK_RET(algTemplateAGL2->KernelRun(param, tempAlgParamsAGL2, templateResourceAGL2));
        }

        // ----------- AGL1: level1 AllGather -----------
        if (!skipLevel1_) {
            GenTempAlgParamsAGL1(loop, currDataCount, sliceSizeRSL1, tailSizeRSL1, tempAlgParamsAGL1);
            CHK_RET(algTemplateAGL1->KernelRun(param, tempAlgParamsAGL1, templateResourceAGL1));
        }

        // ----------- AGL0: level0 AllGather -----------
        GenTempAlgParamsAGL0(
            loop, currDataCount, tempAlgParamsRSL0.sliceSize, tempAlgParamsRSL0.tailSize, tempAlgParamsAGL0);
        CHK_RET(algTemplateAGL0->KernelRun(param, tempAlgParamsAGL0, templateResourceAGL0));

        if (myRank_ == param.root) {
            const DataSlice srcSlice(resCtx.cclMem.addr, meshCommBuffOffset_, currDataCount * dataTypeSize_);
            const DataSlice dstSlice(
                param.outputPtr, processedDataCount * dataTypeSize_, currDataCount * dataTypeSize_);
            CHK_RET(LocalCopy(threads_.at(0), srcSlice, dstSlice));
        }

        processedDataCount += currDataCount;
        loop++;
    }
    HCCL_INFO("[ReduceSequenceExecutorAicpu3Level][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, AicpuReduceSequenceMesh1DNHRNHR, ReduceSequenceExecutorAicpu3Level,
    TopoMatchMultilevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterNHR, InsTempReduceScatterNHR,
    InsTempAllGatherNHR, InsTempAllGatherNHR, InsTempAllGatherMesh1D1DZAxisDetour);

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, AicpuReduceSequenceMesh1DNHR, ReduceSequenceExecutorAicpu3Level, TopoMatchMultilevel,
    InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterNHR, InsTempReduceScatterNHR, InsTempAllGatherNHR,
    InsTempAllGatherNHR, InsTempAllGatherMesh1D1DZAxisDetour);

} // namespace ops_hccl
