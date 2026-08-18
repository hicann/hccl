/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_two_shot_sole_executor.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"

namespace ops_hccl {

constexpr u32 SOLE_EXECUTOR_LEVEL_NUM = 1;

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
std::vector<CostModelParam>
InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName)
{
    (void)comm;
    (void)algName;
    u32 rankSize = topoInfo->userRankSize;
    std::vector<CostModelParam> params = [rankSize, algName] {
        std::vector<CostModelParam> v;
        auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{rankSize, 1.0f, AlgNetType::MESH, true, algName});
        v.insert(v.end(), p0.begin(), p0.end());
        auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{rankSize, 1.0f, AlgNetType::MESH, true, algName});
        v.insert(v.end(), p1.begin(), p1.end());
        return v;
    }();
    return params;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
AlgNetMeta InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo) const
{
    (void)topoInfo;
    AlgNetMeta meta;
    meta.netTypes.push_back(AlgNetType::MESH);
    meta.netTypes.push_back(AlgNetType::MESH);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InsV2AllReduceTwoShotSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    reduceOp_ = param.reduceType;
    dataCount_ = param.DataDes.count;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[InsV2AllReduceTwoShotSoleExecutor][InitCommInfo] myRank [%u], rankSize [%u], redOp [%u], "
        "dataType [%u] dataTypeSize [%u]",
        myRank_, rankSize_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    HCCL_DEBUG("[InsV2AllReduceTwoShotSoleExecutor] CalcRes start");
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    if (algHierarchyInfo.infos.size() != SOLE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR("[InsV2AllReduceTwoShotSoleExecutor] algHierarchyInfo size should be %u", SOLE_EXECUTOR_LEVEL_NUM);
        return HCCL_E_INTERNAL;
    }

    std::shared_ptr<InsAlgTemplate0> reduceScatterTempAlg
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> allGatherTempAlg
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqReduceScatter;
    AlgResourceRequest resReqAllGather;
    CHK_RET(reduceScatterTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatter));
    CHK_RET(allGatherTempAlg->CalcRes(comm, param, topoInfo, resReqAllGather));

    resourceRequest.slaveThreadNum = std::max(resReqReduceScatter.slaveThreadNum, resReqAllGather.slaveThreadNum);
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqReduceScatter.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReqReduceScatter.notifyNumPerThread[i]);
        }
        if (i < resReqAllGather.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReqAllGather.notifyNumPerThread[i]);
        }
    }
    resourceRequest.notifyNumOnMainThread
        = std::max(resReqReduceScatter.notifyNumOnMainThread, resReqAllGather.notifyNumOnMainThread);
    resourceRequest.channels.resize(SOLE_EXECUTOR_LEVEL_NUM);
    resourceRequest.channels[0] = resReqReduceScatter.channels[0];
    HCCL_INFO(
        "[InsV2AllReduceTwoShotSoleExecutor] slaveThreadNum is [%u], notifyNumOnMainThread is [%u], "
        "channel size [%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread, resourceRequest.channels[0].size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceTwoShotSoleExecutor][Orchestrate] Orchestrate Start");
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllReduceTwoShotSoleExecutor][Orchestrate] errNo[0x%016llx] "
            "AllReduce executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenBaseTempAlgParams(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateDataParams& tempAlgParamsReduceScatter,
    TemplateDataParams& tempAlgParamsAllGather) const
{
    tempAlgParamsReduceScatter.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsReduceScatter.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatter.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsReduceScatter.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsReduceScatter.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsReduceScatter.buffInfo.hcclBuff = resCtx.cclMem;

    tempAlgParamsAllGather.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAllGather.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsAllGather.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAllGather.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGather.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsAllGather.buffInfo.hcclBuff = resCtx.cclMem;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTempAlgParamsReduceScatter(
    const u64 loop, const u64 currDataCount, const u64 processedDataCount,
    TemplateDataParams& tempAlgParamsReduceScatter) const
{
    tempAlgParamsReduceScatter.count = currDataCount;
    tempAlgParamsReduceScatter.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsReduceScatter.buffInfo.outBuffBaseOff = outCclBuffOffset_;
    tempAlgParamsReduceScatter.buffInfo.hcclBuffBaseOff = inCclBuffOffset_;

    tempAlgParamsReduceScatter.sliceSize = currDataCount / rankSize_ * dataTypeSize_;
    tempAlgParamsReduceScatter.tailSize = (currDataCount / rankSize_ + currDataCount % rankSize_) * dataTypeSize_;

    tempAlgParamsReduceScatter.inputSliceStride = tempAlgParamsReduceScatter.sliceSize;
    tempAlgParamsReduceScatter.outputSliceStride = 0;

    HCCL_INFO(
        "[InsV2AllReduceTwoShotSoleExecutor] loop [%u] tempAlgParamsReduceScatter.inputSliceStride [%u], "
        "tempAlgParamsReduceScatter.outputSliceStride [%u], tempAlgParamsReduceScatter.sliceSize [%u], "
        "tempAlgParamsReduceScatter.tailSize [%u], tempAlgParamsReduceScatter.buffInfo.inBuffBaseOff [%u], "
        "tempAlgParamsReduceScatter.buffInfo.outBuffBaseOff [%u]",
        loop, tempAlgParamsReduceScatter.inputSliceStride, tempAlgParamsReduceScatter.outputSliceStride,
        tempAlgParamsReduceScatter.sliceSize, tempAlgParamsReduceScatter.tailSize,
        tempAlgParamsReduceScatter.buffInfo.inBuffBaseOff, tempAlgParamsReduceScatter.buffInfo.outBuffBaseOff);

    tempAlgParamsReduceScatter.repeatNum = 1;
    tempAlgParamsReduceScatter.inputRepeatStride = 0;
    tempAlgParamsReduceScatter.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
void InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTempAlgParamsAllGather(
    const u64 loop, const u64 currDataCount, const u64 processedDataCount,
    TemplateDataParams& tempAlgParamsAllGather) const
{
    tempAlgParamsAllGather.count = currDataCount;
    tempAlgParamsAllGather.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsAllGather.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    tempAlgParamsAllGather.buffInfo.hcclBuffBaseOff = 0;

    tempAlgParamsAllGather.sliceSize = currDataCount / rankSize_ * dataTypeSize_;
    tempAlgParamsAllGather.tailSize = (currDataCount / rankSize_ + currDataCount % rankSize_) * dataTypeSize_;

    tempAlgParamsAllGather.inputSliceStride = 0;
    tempAlgParamsAllGather.outputSliceStride = tempAlgParamsAllGather.sliceSize;

    HCCL_INFO(
        "[InsV2AllReduceTwoShotSoleExecutor] loop [%u] tempAlgParamsAllGather.inputSliceStride [%u], "
        "tempAlgParamsAllGather.outputSliceStride [%u], tempAlgParamsAllGather.sliceSize [%u], "
        "tempAlgParamsAllGather.tailSize [%u], tempAlgParamsAllGather.buffInfo.inBuffBaseOff [%u], "
        "tempAlgParamsAllGather.buffInfo.outBuffBaseOff [%u]",
        loop, tempAlgParamsAllGather.inputSliceStride, tempAlgParamsAllGather.outputSliceStride,
        tempAlgParamsAllGather.sliceSize, tempAlgParamsAllGather.tailSize,
        tempAlgParamsAllGather.buffInfo.inBuffBaseOff, tempAlgParamsAllGather.buffInfo.outBuffBaseOff);

    tempAlgParamsAllGather.repeatNum = 1;
    tempAlgParamsAllGather.inputRepeatStride = 0;
    tempAlgParamsAllGather.outputRepeatStride = 0;
    return;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
template <typename InsAlgTemplate>
HcclResult InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::GenTempResource(
    const AlgResourceCtxSerializable& resCtx, const std::shared_ptr<InsAlgTemplate>& algTemplate,
    TemplateResource& tempResource) const
{
    AlgResourceRequest req;
    algTemplate->GetRes(req);
    if (remoteRankToChannelInfo_.empty()) {
        HCCL_ERROR("[InsV2AllReduceTwoShotSoleExecutor][GenTempResource] remoteRankToChannelInfo_ is empty");
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[0];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1>
HcclResult InsV2AllReduceTwoShotSoleExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceTwoShotSoleExecutor][OrchestrateLoop] Start");

    TemplateDataParams tempAlgParamsReduceScatter;
    TemplateDataParams tempAlgParamsAllGather;
    GenBaseTempAlgParams(param, resCtx, tempAlgParamsReduceScatter, tempAlgParamsAllGather);

    std::shared_ptr<InsAlgTemplate0> algTemplateReduceScatter
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    algTemplateReduceScatter->SetchannelsPerRank(remoteRankToChannelInfo_[0]);

    std::shared_ptr<InsAlgTemplate1> algTemplateAllGather
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[0]);
    algTemplateAllGather->SetchannelsPerRank(remoteRankToChannelInfo_[0]);

    TemplateResource templateResourceReduceScatter;
    CHK_RET(GenTempResource(resCtx, algTemplateReduceScatter, templateResourceReduceScatter));

    TemplateResource templateResourceAllGather;
    CHK_RET(GenTempResource(resCtx, algTemplateAllGather, templateResourceAllGather));

    u32 buffSize = 2;
    outCclBuffSize_ = tempAlgParamsReduceScatter.buffInfo.hcclBuff.size / buffSize;
    inCclBuffSize_ = tempAlgParamsReduceScatter.buffInfo.hcclBuff.size - outCclBuffSize_;
    outCclBuffOffset_ = 0;
    inCclBuffOffset_ = outCclBuffSize_;

    u64 maxCountPerLoop
        = inCclBuffOffset_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_ / rankSize_ * rankSize_;

    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        GenTempAlgParamsReduceScatter(loop, currDataCount, processedDataCount, tempAlgParamsReduceScatter);
        CHK_RET(algTemplateReduceScatter->KernelRun(param, tempAlgParamsReduceScatter, templateResourceReduceScatter));

        GenTempAlgParamsAllGather(loop, currDataCount, processedDataCount, tempAlgParamsAllGather);
        CHK_RET(algTemplateAllGather->KernelRun(param, tempAlgParamsAllGather, templateResourceAllGather));

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2AllReduceTwoShotSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSoleMeshConcur, InsV2AllReduceTwoShotSoleExecutor, TopoMatch1D,
    InsTempReduceScatterMesh1DZAxisDetour, InsTempAllGatherMesh1D1DZAxisDetour);
} // namespace ops_hccl
