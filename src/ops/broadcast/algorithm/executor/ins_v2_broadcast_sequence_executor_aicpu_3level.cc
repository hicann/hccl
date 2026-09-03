/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_sequence_executor_aicpu_3level.h"
#include "ins_temp_scatter_nhr.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "aicpu_temp_scatter_mesh_1D_Z_axis_detour.h"

#include "alg_attrs_registry.h"
namespace ops_hccl {
// 当前sequence支持两级和三级组网

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::BroadcastSequenceMesh1dNHRNHRExecutor()
{}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    InitCommInfo(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[BroadcastSequenceMesh1dNHRNHRExecutor][InitCommInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]", myRank_,
        rankSize_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
template <typename InsAlgTemplate>
HcclResult BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempResource(
        const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR(
            "[BroadcastSequenceMesh1dNHRNHRExecutor][GenTempResource] myRank[%u] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%u]",
            myRank_, channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    CalcAlgHierarchyInfo(
        HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    if (algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2 || algHierarchyInfo.infos[0].empty()
        || algHierarchyInfo.infos[1].empty() || algHierarchyInfo.infos[0][0].empty()
        || algHierarchyInfo.infos[1][0].empty()) {
        HCCL_ERROR("[%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    skipLevel2_ = (algHierarchyInfo.infos.size() == TOPO_LEVEL_NUM_2);
    if (skipLevel2_) {
        rankSizeLevel2_ = 1;
    } else {
        rankSizeLevel2_ = algHierarchyInfo.infos[2][0].size();
    }
    HCCL_INFO(
        "[BroadcastSequenceMesh1dNHRNHRExecutor][CalcRes] rankSizeLevel0 [%u], rankSizeLevel1 [%u], rankSizeLevel2 "
        "[%u]",
        rankSizeLevel0_, rankSizeLevel1_, rankSizeLevel2_);

    // L0/L1 模板必创建
    std::shared_ptr<InsAlgTemplate0> ScatterL0TempAlg
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> ScatterL1TempAlg
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate4> agL1TempAlg
        = std::make_shared<InsAlgTemplate4>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate5> agL0TempAlg
        = std::make_shared<InsAlgTemplate5>(param, myRank_, algHierarchyInfo.infos[0]);

    // L2模板、资源请求对象，默认为空
    std::shared_ptr<InsAlgTemplate2> ScatterL2TempAlg;
    std::shared_ptr<InsAlgTemplate3> agL2TempAlg;
    AlgResourceRequest resReqScatterL0;
    AlgResourceRequest resReqScatterL1;
    AlgResourceRequest resReqScatterL2;
    AlgResourceRequest resReqAGL2;
    AlgResourceRequest resReqAGL1;
    AlgResourceRequest resReqAGL0;

    // L0/L1 必走层级计算资源
    CHK_RET(ScatterL0TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL0));
    CHK_RET(ScatterL1TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL1));
    CHK_RET(agL1TempAlg->CalcRes(comm, param, topoInfo, resReqAGL1));
    CHK_RET(agL0TempAlg->CalcRes(comm, param, topoInfo, resReqAGL0));

    // 仅三层拓扑才创建L2模板、计算资源
    if (!skipLevel2_) {
        ScatterL2TempAlg = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);
        agL2TempAlg = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[2]);
        CHK_RET(ScatterL2TempAlg->CalcRes(comm, param, topoInfo, resReqScatterL2));
        CHK_RET(agL2TempAlg->CalcRes(comm, param, topoInfo, resReqAGL2));
    }

    // slaveThreadNum：先取必选层级最大值，L2存在再合并
    resourceRequest.slaveThreadNum = std::max(
        {resReqScatterL0.slaveThreadNum, resReqScatterL1.slaveThreadNum, resReqAGL1.slaveThreadNum,
         resReqAGL0.slaveThreadNum});
    if (!skipLevel2_) {
        resourceRequest.slaveThreadNum
            = std::max({resourceRequest.slaveThreadNum, resReqScatterL2.slaveThreadNum, resReqAGL2.slaveThreadNum});
    }

    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);

    // 更新notify逻辑
    auto UpdateNotify = [&](const AlgResourceRequest& req) {
        for (u32 i = 0; i < req.notifyNumPerThread.size() && i < resourceRequest.notifyNumPerThread.size(); ++i) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], req.notifyNumPerThread[i]);
        }
    };
    UpdateNotify(resReqScatterL0);
    UpdateNotify(resReqScatterL1);
    UpdateNotify(resReqAGL1);
    UpdateNotify(resReqAGL0);
    if (!skipLevel2_) {
        UpdateNotify(resReqScatterL2);
        UpdateNotify(resReqAGL2);
    }

    // notifyNumOnMainThread
    resourceRequest.notifyNumOnMainThread = std::max(
        {resReqScatterL0.notifyNumOnMainThread, resReqScatterL1.notifyNumOnMainThread, resReqAGL1.notifyNumOnMainThread,
         resReqAGL0.notifyNumOnMainThread});
    if (!skipLevel2_) {
        resourceRequest.notifyNumOnMainThread = std::max(
            {resourceRequest.notifyNumOnMainThread, resReqScatterL2.notifyNumOnMainThread,
             resReqAGL2.notifyNumOnMainThread});
    }

    // channels默认3个长度，两层拓扑复用L1通道填充channels[2]，实际用不到
    u64 channelsSize = 3;
    resourceRequest.channels.resize(channelsSize);
    resourceRequest.channels[0] = resReqScatterL0.channels[0];
    resourceRequest.channels[1] = resReqScatterL1.channels[0];
    if (!skipLevel2_) {
        resourceRequest.channels[2] = resReqScatterL2.channels[0];
    } else {
        resourceRequest.channels[2] = resReqScatterL1.channels[0];
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[BroadcastSequenceMesh1dNHRNHRExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    CHK_RET(InitExecutorInfo(param, resCtx));
    threads_ = resCtx.threads;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[BroadcastSequenceMesh1dNHRNHRExecutor][Orchestrate]errNo[0x%016llx] Broadcast excutor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::InitExecutorInfo(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    skipLevel2_ = (algHierarchyInfo_.infos.size() == TOPO_LEVEL_NUM_2);
    skipLevel1_ = (rankSizeLevel1_ == 1);
    if (skipLevel2_) {
        rankSizeLevel2_ = 1;
    } else {
        rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
    }
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = (myRank_ / rankSizeLevel0_) % rankSizeLevel1_;
    rankIdxLevel2_ = myRank_ / (rankSizeLevel0_ * rankSizeLevel1_);

    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;

    HCCL_INFO(
        "[BroadcastSequenceMesh1dNHRNHRExecutor][InitExecutorInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]",
        +myRank_, rankSize_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::GenTempAlgParamsScatterL0(u64 currDataCount, u64 processedDataCount, TemplateDataParams& params)
    const
{
    u64 sliceCnt = currDataCount / rankSizeLevel0_;
    u64 remCnt = currDataCount % rankSizeLevel0_;
    params.sliceSize = sliceCnt * dataTypeSize_;
    params.tailSize = (sliceCnt + remCnt) * dataTypeSize_;

    params.count = currDataCount;
    params.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    params.buffInfo.outBuffBaseOff = 0;
    params.buffInfo.hcclBuffBaseOff = 0;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;

    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::GenTempAlgParamsScatterL1(u64 level1TotalCnt, u64 l0SliceByte, TemplateDataParams& params) const
{
    u64 sliceCnt = level1TotalCnt / rankSizeLevel1_;
    u64 remCnt = level1TotalCnt % rankSizeLevel1_;
    params.sliceSize = sliceCnt * dataTypeSize_;
    params.tailSize = (sliceCnt + remCnt) * dataTypeSize_;

    params.count = level1TotalCnt;
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.outBuffBaseOff = rankIdxLevel0_ * l0SliceByte;
    params.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * l0SliceByte;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;

    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempAlgParamsScatterL2(u64 level2TotalCnt, u64 l0SliceByte, u64 l1SliceByte, TemplateDataParams& params) const
{
    u64 sliceCnt = level2TotalCnt / rankSizeLevel2_;
    u64 remCnt = level2TotalCnt % rankSizeLevel2_;
    params.sliceSize = sliceCnt * dataTypeSize_;
    params.tailSize = (sliceCnt + remCnt) * dataTypeSize_;

    params.count = level2TotalCnt;
    params.buffInfo.inBuffBaseOff = 0;
    params.buffInfo.outBuffBaseOff = rankIdxLevel0_ * l0SliceByte + rankIdxLevel1_ * l1SliceByte;
    params.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * l0SliceByte + rankIdxLevel1_ * l1SliceByte;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;

    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempAlgParamsAGL2(
        const u64 sliceSize, const u64 tailSize, TemplateDataParams& params, u64 l0SliceByte, u64 l1SliceByte) const
{
    params.buffInfo.inBuffBaseOff = rankIdxLevel0_ * l0SliceByte + rankIdxLevel1_ * l1SliceByte;
    params.buffInfo.outBuffBaseOff = rankIdxLevel0_ * l0SliceByte + rankIdxLevel1_ * l1SliceByte;
    params.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * l0SliceByte + rankIdxLevel1_ * l1SliceByte;
    // 与上一步框间Scatter数据量一致
    params.sliceSize = sliceSize;
    params.tailSize = tailSize;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorAicpu] params.inputSliceStride [%u],"
        "params.outputSliceStride [%u] params.sliceSize [%u], params.tailSize [%u], "
        "params.buffInfo.inBuffBaseOff [%u], params.buffInfo.outBuffBaseOff [%u]",
        params.inputSliceStride, params.outputSliceStride, params.sliceSize, params.tailSize,
        params.buffInfo.inBuffBaseOff, params.buffInfo.outBuffBaseOff);

    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempAlgParamsAGL1(const u64 sliceSize, const u64 tailSize, TemplateDataParams& params, u64 l0SliceByte) const
{
    params.buffInfo.inBuffBaseOff = rankIdxLevel0_ * l0SliceByte;
    params.buffInfo.outBuffBaseOff = rankIdxLevel0_ * l0SliceByte;
    params.buffInfo.hcclBuffBaseOff = rankIdxLevel0_ * l0SliceByte;

    params.sliceSize = sliceSize;
    params.tailSize = tailSize;

    params.inputSliceStride = params.sliceSize;
    params.outputSliceStride = params.sliceSize;

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorAicpu] params.inputSliceStride [%u], "
        "params.outputSliceStride [%u], params.sliceSize [%u], params.tailSize [%u], "
        "params.buffInfo.inBuffBaseOff [%u], params.buffInfo.outBuffBaseOff [%u]",
        params.inputSliceStride, params.outputSliceStride, params.sliceSize, params.tailSize,
        params.buffInfo.inBuffBaseOff, params.buffInfo.outBuffBaseOff);

    params.repeatNum = 1;
    params.inputRepeatStride = 0;
    params.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempAlgParamsAGL0(
        const u64 processedDataCount, const u64 sliceSize, const u64 tailSize,
        TemplateDataParams& tempAlgParamsStepFour) const
{
    tempAlgParamsStepFour.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsStepFour.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsStepFour.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsStepFour.sliceSize = sliceSize;
    tempAlgParamsStepFour.tailSize = tailSize;

    tempAlgParamsStepFour.inputSliceStride = tempAlgParamsStepFour.sliceSize;
    tempAlgParamsStepFour.outputSliceStride = tempAlgParamsStepFour.sliceSize;

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorAicpu] tempAlgParamsStepFour.inputSliceStride [%u], "
        "tempAlgParamsStepFour.outputSliceStride [%u], tempAlgParamsStepFour.sliceSize [%u], "
        "tempAlgParamsStepFour.tailSize [%u], "
        "tempAlgParamsStepFour.buffInfo.inBuffBaseOff [%u], tempAlgParamsStepFour.buffInfo.outBuffBaseOff [%u]",
        tempAlgParamsStepFour.inputSliceStride, tempAlgParamsStepFour.outputSliceStride,
        tempAlgParamsStepFour.sliceSize, tempAlgParamsStepFour.tailSize, tempAlgParamsStepFour.buffInfo.inBuffBaseOff,
        tempAlgParamsStepFour.buffInfo.outBuffBaseOff);

    tempAlgParamsStepFour.repeatNum = 1;
    tempAlgParamsStepFour.inputRepeatStride = 0;
    tempAlgParamsStepFour.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult BroadcastSequenceMesh1dNHRNHRExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[BroadcastSequenceMesh1dNHRNHRExecutor][OrchestrateLoop] Start");
    u32 podSize = rankSizeLevel0_ * rankSizeLevel1_;
    u32 rootPodStartRank = param.root / podSize * podSize;
    u32 rootPodEndRank = rootPodStartRank + podSize - 1;
    // scatter L0
    TemplateDataParams tempAlgParamsScatterL0;
    tempAlgParamsScatterL0.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsScatterL0.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsScatterL0.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsScatterL0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    std::shared_ptr<InsAlgTemplate0> algTemplateScatterL0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateScatterL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    // scatter L1
    TemplateDataParams tempAlgParamsScatterL1;
    tempAlgParamsScatterL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterL1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsScatterL1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterL1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    std::shared_ptr<InsAlgTemplate1> algTemplateScatterL1
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
    if (!skipLevel1_) {
        CHK_RET(algTemplateScatterL1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }
    std::shared_ptr<InsAlgTemplate2> algTemplateScatterL2;
    std::shared_ptr<InsAlgTemplate3> algTemplateAllGatherL2;
    TemplateDataParams tempAlgParamsScatterL2;
    TemplateDataParams tempAlgParamsAllGatherL2;
    if (!skipLevel2_) {
        // scatter L2
        tempAlgParamsScatterL2.buffInfo.inputPtr = resCtx.cclMem.addr;
        tempAlgParamsScatterL2.buffInfo.outputPtr = resCtx.cclMem.addr;
        tempAlgParamsScatterL2.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsScatterL2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsScatterL2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsScatterL2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        algTemplateScatterL2 = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
        CHK_RET(algTemplateScatterL2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));
        // AG L2
        tempAlgParamsAllGatherL2.buffInfo.inputPtr = resCtx.cclMem.addr;
        tempAlgParamsAllGatherL2.buffInfo.outputPtr = resCtx.cclMem.addr;
        tempAlgParamsAllGatherL2.buffInfo.hcclBuff = resCtx.cclMem;
        tempAlgParamsAllGatherL2.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsAllGatherL2.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        tempAlgParamsAllGatherL2.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        algTemplateAllGatherL2 = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[2]);
        CHK_RET(algTemplateAllGatherL2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));
    }
    // AG L1
    TemplateDataParams tempAlgParamsAllGatherL1;
    tempAlgParamsAllGatherL1.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL1.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL1.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsAllGatherL1.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAllGatherL1.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAllGatherL1.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    std::shared_ptr<InsAlgTemplate4> algTemplateAllGatherL1
        = std::make_shared<InsAlgTemplate4>(param, myRank_, algHierarchyInfo_.infos[1]);
    if (!skipLevel1_) {
        CHK_RET(algTemplateAllGatherL1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }
    // AG L0
    TemplateDataParams tempAlgParamsAllGatherL0;
    tempAlgParamsAllGatherL0.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherL0.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsAllGatherL0.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsAllGatherL0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAllGatherL0.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsAllGatherL0.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    std::shared_ptr<InsAlgTemplate5> algTemplateAllGatherL0
        = std::make_shared<InsAlgTemplate5>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateAllGatherL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    // 构造Scatter L0 template资源
    TemplateResource templateScatterResourceL0;
    templateScatterResourceL0.channels = remoteRankToChannelInfo_[0];
    templateScatterResourceL0.threads = resCtx.threads;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateScatterL0, templateScatterResourceL0));
    // 构造Scatter L1 template资源
    TemplateResource templateScatterResourceL1;
    templateScatterResourceL1.channels = remoteRankToChannelInfo_[1];
    templateScatterResourceL1.threads = resCtx.threads;
    CHK_RET(GenTempResource(resCtx, 1, algTemplateScatterL1, templateScatterResourceL1));
    // 构造Scatter L2 template资源
    TemplateResource templateScatterResourceL2;
    // 构造Allgather L2 template资源
    TemplateResource templateAllgatherResourceL2;
    if (!skipLevel2_) {
        templateScatterResourceL2.channels = remoteRankToChannelInfo_[2];
        templateScatterResourceL2.threads = resCtx.threads;
        CHK_RET(GenTempResource(resCtx, 2, algTemplateScatterL2, templateScatterResourceL2));
        templateAllgatherResourceL2.channels = remoteRankToChannelInfo_[2];
        templateAllgatherResourceL2.threads = resCtx.threads;
        CHK_RET(GenTempResource(resCtx, 2, algTemplateAllGatherL2, templateAllgatherResourceL2));
    }

    // 构造Allgather L1 template资源
    TemplateResource templateAllgatherResourceL1;
    templateAllgatherResourceL1.channels = remoteRankToChannelInfo_[1];
    templateAllgatherResourceL1.threads = resCtx.threads;
    CHK_RET(GenTempResource(resCtx, 1, algTemplateAllGatherL1, templateAllgatherResourceL1));

    // 构造Allgather L0 template资源
    TemplateResource templateAllgatherResourceL0;
    templateAllgatherResourceL0.channels = remoteRankToChannelInfo_[0];
    templateAllgatherResourceL0.threads = resCtx.threads;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateAllGatherL0, templateAllgatherResourceL0));

    // 中转内存单次最多能够接受的output count，注意是count不是size
    u64 dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    u64 dataCount_ = param.DataDes.count;
    u64 maxCountPerLoop
        = tempAlgParamsScatterL0.buffInfo.hcclBuff.size / AICPU_ALIGN_SIZE * AICPU_ALIGN_SIZE / dataTypeSize_;
    CHK_PRT_RET(
        maxCountPerLoop == 0, HCCL_ERROR("[%s] maxCountPerLoop is 0, dataTypeSize_[%llu].", __func__, dataTypeSize_),
        HCCL_E_INTERNAL);
    // 计算loopTimes
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    // 已处理的元素数
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        // ---------------------- Scatter L0 标量分片 ----------------------
        GenTempAlgParamsScatterL0(currDataCount, processedDataCount, tempAlgParamsScatterL0);
        algTemplateScatterL0->SetRoot(param.root);
        u32 root = param.root;
        u32 rootIdx0 = param.root % rankSizeLevel0_;
        u32 rootIdx1 = (param.root / rankSizeLevel0_) % rankSizeLevel1_;
        u32 rootIdx2 = param.root / (rankSizeLevel0_ * rankSizeLevel1_);
        if (rankIdxLevel2_ == rootIdx2 && rankIdxLevel1_ == rootIdx1) {
            CHK_RET(algTemplateScatterL0->KernelRun(param, tempAlgParamsScatterL0, templateScatterResourceL0));
        }
        // ---------------------- Scatter L1 标量分片 ----------------------
        u64 l1SliceByte = tempAlgParamsScatterL0.sliceSize;
        u64 l1TotalCnt = currDataCount / rankSizeLevel0_;
        u64 l1TailCnt = tempAlgParamsScatterL0.tailSize / dataTypeSize_;
        root = param.root - rootIdx0 + rankIdxLevel0_;
        bool layer1IsTail = false;
        if ((root % rankSizeLevel0_) == (algHierarchyInfo_.infos[0][0].back() % rankSizeLevel0_)) {
            GenTempAlgParamsScatterL1(l1TailCnt, l1SliceByte, tempAlgParamsScatterL1);
            layer1IsTail = true;
        } else {
            GenTempAlgParamsScatterL1(l1TotalCnt, l1SliceByte, tempAlgParamsScatterL1);
        }
        algTemplateScatterL1->SetRoot(root);
        rootIdx0 = root % rankSizeLevel0_;
        rootIdx1 = (root / rankSizeLevel0_) % rankSizeLevel1_;
        rootIdx2 = root / (rankSizeLevel0_ * rankSizeLevel1_);
        if (l1TotalCnt != 0 || l1TailCnt != 0) {
            if (rankIdxLevel2_ == rootIdx2) {
                if (!skipLevel1_) {
                    CHK_RET(algTemplateScatterL1->KernelRun(param, tempAlgParamsScatterL1, templateScatterResourceL1));
                }
            }
        }
        // ---------------------- Scatter L2 标量分片 ----------------------
        if (!skipLevel2_) {
            root = rootIdx2 * (rankSizeLevel0_ * rankSizeLevel1_) + rankIdxLevel1_ * rankSizeLevel0_ + rankIdxLevel0_;
            if ((root % (rankSizeLevel0_ * rankSizeLevel1_))
                == (algHierarchyInfo_.infos[1][0].back() % (rankSizeLevel0_ * rankSizeLevel1_))) {
                u64 l2SliceByte = tempAlgParamsScatterL1.sliceSize;
                GenTempAlgParamsScatterL2(
                    tempAlgParamsScatterL1.tailSize / dataTypeSize_, l1SliceByte, l2SliceByte, tempAlgParamsScatterL2);
            } else {
                u64 l2SliceByte = tempAlgParamsScatterL1.sliceSize;
                GenTempAlgParamsScatterL2(
                    tempAlgParamsScatterL1.sliceSize / dataTypeSize_, l1SliceByte, l2SliceByte, tempAlgParamsScatterL2);
            }
            algTemplateScatterL2->SetRoot(root);
            if (tempAlgParamsScatterL1.tailSize != 0 || tempAlgParamsScatterL1.sliceSize != 0) {
                CHK_RET(algTemplateScatterL2->KernelRun(param, tempAlgParamsScatterL2, templateScatterResourceL2));
            }
        }
        // ---------------------- AllGather L2 ----------------------
        if (!skipLevel2_) {
            GenTempAlgParamsAGL2(
                tempAlgParamsScatterL2.sliceSize, tempAlgParamsScatterL2.tailSize, tempAlgParamsAllGatherL2,
                tempAlgParamsScatterL0.sliceSize, tempAlgParamsScatterL1.sliceSize);

            CHK_RET(algTemplateAllGatherL2->KernelRun(param, tempAlgParamsAllGatherL2, templateAllgatherResourceL2));
        }
        // ---------------------- AllGather L1 ----------------------
        GenTempAlgParamsAGL1(
            tempAlgParamsScatterL1.sliceSize, tempAlgParamsScatterL1.tailSize, tempAlgParamsAllGatherL1,
            tempAlgParamsScatterL0.sliceSize);
        if (!skipLevel1_) {
            CHK_RET(algTemplateAllGatherL1->KernelRun(param, tempAlgParamsAllGatherL1, templateAllgatherResourceL1));
        }
        // ---------------------- AllGather L0 ----------------------
        GenTempAlgParamsAGL0(
            processedDataCount, tempAlgParamsScatterL0.sliceSize, tempAlgParamsScatterL0.tailSize,
            tempAlgParamsAllGatherL0);
        CHK_RET(algTemplateAllGatherL0->KernelRun(param, tempAlgParamsAllGatherL0, templateAllgatherResourceL0));
        processedDataCount += currDataCount;
    }

    HCCL_INFO("[BroadcastSequenceMesh1dNHRNHRExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_ALG_ATTRS(AicpuBroadcastSequenceMeshConcurNHRNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
                   topo.minTopoLevelNum = 3; topo.maxTopoLevelNum = 3; topo.isSupportLevel1Nhr = false;
                   op.unsupportedDataTypes = UNSUPPORTED_64BIT;);
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_BROADCAST, AicpuBroadcastSequenceMeshConcurNHRNHR, BroadcastSequenceMesh1dNHRNHRExecutor,
    TopoMatchMultilevel,
    AicpuTempScatterMesh1DZAxisDetour,    // Scatter L0 (框内, Z轴绕路)
    InsTempScatterNHR,                    // Scatter L1 (框间)
    InsTempScatterNHR,                    // Scatter L2 (跨超节点)
    InsTempAllGatherNHR,                  // AllGather L2 (跨超节点)
    InsTempAllGatherNHR,                  // AllGather L1 (框间)
    InsTempAllGatherMesh1D1DZAxisDetour); // AllGather L0 (框内, Z轴绕路)

REGISTER_ALG_ATTRS(AicpuBroadcastSequenceMeshConcurNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
                   topo.minTopoLevelNum = 3; topo.maxTopoLevelNum = 3; topo.isSupportLevel1Nhr = false;
                   op.unsupportedDataTypes = UNSUPPORTED_64BIT;);
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_BROADCAST, AicpuBroadcastSequenceMeshConcurNHR, BroadcastSequenceMesh1dNHRNHRExecutor,
    TopoMatchMultilevel,
    AicpuTempScatterMesh1DZAxisDetour,    // Scatter L0 (框内, Z轴绕路)
    InsTempScatterNHR,                    // Scatter L1 (框间)
    InsTempScatterNHR,                    // Scatter L2 (跨超节点)
    InsTempAllGatherNHR,                  // AllGather L2 (跨超节点)
    InsTempAllGatherNHR,                  // AllGather L1 (框间)
    InsTempAllGatherMesh1D1DZAxisDetour); // AllGather L0 (框内, Z轴绕路)
} // namespace ops_hccl
