/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_broadcast_sequence_rhd_executor.h"
#include "executor_common_ops.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "aicpu_temp_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_scatter_rhd.h"
#include "ins_temp_scatter_rhd_dpu_inter_node.h"
#include "ins_temp_all_gather_rhd.h"
#include "ins_temp_all_gather_rhd_dpu.h"
#include "ins_v2_broadcast_sequence_executor_aicpu_3level.h"
#include "broadcast_sequence_executor_utils.h"
#include "alg_attrs_registry.h"
#include "op_common.h"

namespace ops_hccl {

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    InsV2BroadcastSequenceRhdExecutor()
{}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    InitCommInfo(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    CHK_PTR_NULL(topoInfo);
    CHK_PRT_RET(
        algHierarchyInfo.infos.size() != ALGO_LEVEL_NUM_TWO || algHierarchyInfo.infos[0].size() != 1
            || algHierarchyInfo.infos[1].size() != 1 || algHierarchyInfo.infos[0][0].empty()
            || algHierarchyInfo.infos[1][0].empty(),
        HCCL_ERROR("[BroadcastSequence] invalid hierarchy"), HCCL_E_PARA);
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[InsV2BroadcastSequenceRhdExecutor][InitCommInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]", myRank_,
        rankSize_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    CHK_PTR_NULL(topoInfo);
    CHK_PRT_RET(
        algAttrs.algoTypes.size() != ALGO_LEVEL_NUM_TWO,
        HCCL_ERROR("[BroadcastSequence] expected two algorithm levels"), HCCL_E_NOT_SUPPORT);
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
std::vector<CostModelParam>
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    return {{0.0f, 0.0f, 1.0f, 0.0f}};
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
AlgNetMeta
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)topoInfo;
    (void)param;
    (void)algName;
    AlgNetMeta meta;
    meta.netTypes.push_back(CommTopo::COMM_TOPO_1DMESH);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1};
    return meta;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_RET(InitCommInfo(comm, param, topoInfo, algHierarchyInfo));
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    HCCL_INFO(
        "[InsV2BroadcastSequenceRhdExecutor][CalcRes] rankSizeLevel0 [%u], rankSizeLevel1 [%u]", rankSizeLevel0_,
        rankSizeLevel1_);

    std::shared_ptr<InsAlgTemplate0> intraScatterTempAlg
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> interScatterTempAlg
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate2> interAllGatherTempAlg
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[1]);
    std::shared_ptr<InsAlgTemplate3> intraAllGatherTempAlg
        = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[0]);

    AlgResourceRequest resReqScatterIntra;
    AlgResourceRequest resReqAllGatherIntra;
    AlgResourceRequest resReqScatterInter;
    AlgResourceRequest resReqAllGatherInter;

    CHK_RET(intraScatterTempAlg->CalcRes(comm, param, topoInfo, resReqScatterIntra));
    CHK_RET(intraAllGatherTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherIntra));
    CHK_RET(interScatterTempAlg->CalcRes(comm, param, topoInfo, resReqScatterInter));
    CHK_RET(interAllGatherTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherInter));

    // 各层 Scatter/AllGather 顺序复用同一批线程与 notify，因此按层取并集语义合并。
    resourceRequest.slaveThreadNum = std::max(
        {resReqScatterIntra.slaveThreadNum, resReqAllGatherIntra.slaveThreadNum, resReqScatterInter.slaveThreadNum,
         resReqAllGatherInter.slaveThreadNum});
    resourceRequest.notifyNumPerThread.clear();
    broadcast_sequence_executor::MergeNotifyNumPerThread(
        resourceRequest.notifyNumPerThread, resReqScatterIntra.notifyNumPerThread);
    broadcast_sequence_executor::MergeNotifyNumPerThread(
        resourceRequest.notifyNumPerThread, resReqAllGatherIntra.notifyNumPerThread);
    broadcast_sequence_executor::MergeNotifyNumPerThread(
        resourceRequest.notifyNumPerThread, resReqScatterInter.notifyNumPerThread);
    broadcast_sequence_executor::MergeNotifyNumPerThread(
        resourceRequest.notifyNumPerThread, resReqAllGatherInter.notifyNumPerThread);
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum, 0);
    resourceRequest.notifyNumOnMainThread = std::max(
        {resReqScatterIntra.notifyNumOnMainThread, resReqAllGatherIntra.notifyNumOnMainThread,
         resReqScatterInter.notifyNumOnMainThread, resReqAllGatherInter.notifyNumOnMainThread});

    resourceRequest.channels.resize(2);
    CHK_PRT_RET(
        broadcast_sequence_executor::ValidateChannelRequestCompatible(
            resReqScatterIntra.channels[0], resReqAllGatherIntra.channels[0])
            != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2BroadcastSequenceRhdExecutor][CalcRes] intra-level Scatter/AllGather channel requests conflict."),
        HCCL_E_PARA);
    CHK_PRT_RET(
        broadcast_sequence_executor::ValidateChannelRequestCompatible(
            resReqScatterInter.channels[0], resReqAllGatherInter.channels[0])
            != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2BroadcastSequenceRhdExecutor][CalcRes] inter-level Scatter/AllGather channel requests conflict."),
        HCCL_E_PARA);
    CHK_RET(broadcast_sequence_executor::MergeChannelRequest(
        resReqScatterIntra.channels[0], resReqAllGatherIntra.channels[0], resourceRequest.channels[0]));
    CHK_RET(broadcast_sequence_executor::MergeChannelRequest(
        resReqScatterInter.channels[0], resReqAllGatherInter.channels[0], resourceRequest.channels[1]));

    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceRhdExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    CHK_RET(InitExecutorInfo(param, resCtx));
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2BroadcastSequenceRhdExecutor][Orchestrate]errNo[0x%016llx] Broadcast executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    InitExecutorInfo(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    if (algHierarchyInfo_.infos.size() < TOPO_LEVEL_NUM_2 || algHierarchyInfo_.infos[0].empty()
        || algHierarchyInfo_.infos[1].empty() || algHierarchyInfo_.infos[0][0].empty()
        || algHierarchyInfo_.infos[1][0].empty()) {
        HCCL_ERROR("[InsV2BroadcastSequenceRhdExecutor][%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = myRank_ / algHierarchyInfo_.infos[0][0].size();

    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];

    HCCL_INFO(
        "[InsV2BroadcastSequenceRhdExecutor][InitExecutorInfo] myRank [%u], rankSize [%u], dataTypeSize [%u]", +myRank_,
        rankSize_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2BroadcastSequenceRhdExecutor][OrchestrateLoop] Start");

    // 不区分CCL-IN 与 CCL-OUT
    // 声明框内Scatter templateargs
    TemplateDataParams tempAlgParamsScatterIntra;
    tempAlgParamsScatterIntra.buffInfo.inputPtr = param.inputPtr;
    tempAlgParamsScatterIntra.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterIntra.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsScatterIntra.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParamsScatterIntra.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterIntra.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框内Scatter template
    std::shared_ptr<InsAlgTemplate0> algTemplateScatterIntra
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);

    // 声明框间Scatter templateargs
    TemplateDataParams tempAlgParamsScatterInter;
    tempAlgParamsScatterInter.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterInter.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsScatterInter.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsScatterInter.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterInter.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsScatterInter.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框间Scatter template
    std::shared_ptr<InsAlgTemplate1> algTemplateScatterInter
        = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框间AllGather templateargs
    TemplateDataParams tempAlgParamsAllGatherInter;
    tempAlgParamsAllGatherInter.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherInter.buffInfo.outputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherInter.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsAllGatherInter.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAllGatherInter.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAllGatherInter.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框间AllGather template
    std::shared_ptr<InsAlgTemplate2> algTemplateAllGatherInter
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[1]);

    // 声明框内AllGather templateargs
    TemplateDataParams tempAlgParamsAllGatherIntra;
    tempAlgParamsAllGatherIntra.buffInfo.inputPtr = resCtx.cclMem.addr;
    tempAlgParamsAllGatherIntra.buffInfo.outputPtr = param.outputPtr;
    tempAlgParamsAllGatherIntra.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParamsAllGatherIntra.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParamsAllGatherIntra.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParamsAllGatherIntra.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框内AllGather template
    std::shared_ptr<InsAlgTemplate3> algTemplateAllGatherIntra
        = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[0]);

    auto PrepareTemplateResource
        = [&](TemplateResource& resource, const auto& scatterTemplate, const auto& allGatherTemplate, const u32 level) {
              const auto& channels = remoteRankToChannelInfo_[level];
              SetTemplateCommResource(resource, channels, resCtx);
              CHK_RET(broadcast_sequence_executor::SetChannelsPerRankIfNotEmpty(scatterTemplate, channels));
              const size_t requiredThreads
                  = std::max<size_t>({1, scatterTemplate->GetThreadNum(), allGatherTemplate->GetThreadNum()});
              if (resource.threads.size() > requiredThreads) {
                  resource.threads.resize(requiredThreads);
              }
              return HCCL_SUCCESS;
          };
    // 构造框内template资源
    TemplateResource templateResourceIntra;
    CHK_RET(PrepareTemplateResource(templateResourceIntra, algTemplateScatterIntra, algTemplateAllGatherIntra, 0));
    // 构造框间template资源
    TemplateResource templateResourceInter;
    CHK_RET(PrepareTemplateResource(templateResourceInter, algTemplateScatterInter, algTemplateAllGatherInter, 1));
    templateResourceInter.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
    templateResourceInter.dpuThread = resCtx.dpuThread;
    templateResourceInter.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
    templateResourceInter.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;
    // Scatter/AllGather 复用同层通道资源；两套模板都必须按实际 peer 通道数切分数据。
    CHK_RET(broadcast_sequence_executor::SetChannelsPerRankIfNotEmpty(
        algTemplateAllGatherIntra, remoteRankToChannelInfo_[0]));
    CHK_RET(broadcast_sequence_executor::SetChannelsPerRankIfNotEmpty(
        algTemplateAllGatherInter, remoteRankToChannelInfo_[1]));

    // 中转内存单次最多能够接受的output count，注意是count不是size
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataCount_ = param.DataDes.count;
    u64 maxCountPerLoop = tempAlgParamsScatterIntra.buffInfo.hcclBuff.size / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN
                          / dataTypeSize_;
    // 计算loopTimes
    CHK_PRT_RET(
        maxCountPerLoop == 0, HCCL_ERROR("[%s] maxCountPerLoop is 0, dataTypeSize_[%llu].", __func__, dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);
    u64 processedDataCount = 0;

    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;

        // ----------- 框内Scatter数据搬运 -----------
        // 框内Scatter的数据偏移和搬运量计算
        tempAlgParamsScatterIntra.count = currDataCount;
        tempAlgParamsScatterIntra.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsScatterIntra.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterIntra.buffInfo.hcclBuffBaseOff = 0;

        CHK_RET(SplitData(
            currDataCount, rankSizeLevel0_, tempAlgParamsScatterIntra)); // 计算每个卡对应位置的offset,count,size
        CHK_PRT_RET(
            tempAlgParamsScatterIntra.allRankSliceSize.size() != rankSizeLevel0_,
            HCCL_ERROR(
                "[InsV2BroadcastSequenceRhdExecutor][tempAlgParamsScatterIntra] slice num[%zu] is not equal to rank "
                "size[%u].",
                tempAlgParamsScatterIntra.allRankSliceSize.size(), rankSizeLevel0_),
            HcclResult::HCCL_E_INTERNAL);

        tempAlgParamsScatterIntra.sliceSize = tempAlgParamsScatterIntra.allRankSliceSize.front();
        tempAlgParamsScatterIntra.tailSize = tempAlgParamsScatterIntra.allRankSliceSize.back();
        tempAlgParamsScatterIntra.inputSliceStride = tempAlgParamsScatterIntra.sliceSize;
        tempAlgParamsScatterIntra.outputSliceStride = tempAlgParamsScatterIntra.sliceSize;

        // 不需要重复
        tempAlgParamsScatterIntra.repeatNum = 1;
        tempAlgParamsScatterIntra.inputRepeatStride = 0;
        tempAlgParamsScatterIntra.outputRepeatStride = 0;
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateScatterIntra->KernelRun(param, tempAlgParamsScatterIntra, templateResourceIntra));

        // ----------- 框间Scatter数据搬运 -----------
        // 框间的数据偏移和搬运计算
        tempAlgParamsScatterInter.count = tempAlgParamsScatterIntra.allRankProcessedDataCount.at(rankIdxLevel0_);
        tempAlgParamsScatterInter.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsScatterInter.buffInfo.outBuffBaseOff = 0;
        tempAlgParamsScatterInter.buffInfo.hcclBuffBaseOff
            = tempAlgParamsScatterIntra.allRankDispls.at(rankIdxLevel0_); // 将框内的切片偏移传到框间
        tempAlgParamsScatterInter.root = (param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_);

        tempAlgParamsScatterInter.sliceSize = 0;
        tempAlgParamsScatterInter.tailSize = 0;

        CHK_RET(SplitData(tempAlgParamsScatterInter.count, rankSizeLevel1_, tempAlgParamsScatterInter));
        CHK_PRT_RET(
            tempAlgParamsScatterInter.allRankSliceSize.size() != rankSizeLevel1_,
            HCCL_ERROR(
                "[InsV2BroadcastSequenceRhdExecutor][tempAlgParamsScatterInter] slice num[%zu] is not equal to rank "
                "size[%u].",
                tempAlgParamsScatterInter.allRankSliceSize.size(), rankSizeLevel1_),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_INFO(
            "[InsV2BroadcastSequenceRhdExecutor][SplitData][ScatterInter] count[%llu] slicenum[%u]",
            tempAlgParamsScatterInter.count, rankSizeLevel1_);

        tempAlgParamsScatterInter.sliceSize = tempAlgParamsScatterInter.allRankSliceSize.front();
        tempAlgParamsScatterInter.tailSize = tempAlgParamsScatterInter.allRankSliceSize.back();
        tempAlgParamsScatterInter.inputSliceStride = tempAlgParamsScatterInter.sliceSize;
        tempAlgParamsScatterInter.outputSliceStride = tempAlgParamsScatterInter.sliceSize;

        HCCL_DEBUG(
            "[InsV2BroadcastSequenceRhdExecutor] loop [%llu] tempAlgParamsScatterInter.inputSliceStride [%llu],"
            "tempAlgParamsScatterInter.outputSliceStride [%llu] tempAlgParamsScatterInter.sliceSize [%llu]",
            loop, tempAlgParamsScatterInter.inputSliceStride, tempAlgParamsScatterInter.outputSliceStride,
            tempAlgParamsScatterInter.sliceSize);
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceRhdExecutor] loop [%llu] tempAlgParamsScatterInter.buffInfo.inBuffBaseOff [%llu],"
            "tempAlgParamsScatterInter.buffInfo.outBuffBaseOff [%llu]",
            loop, tempAlgParamsScatterInter.buffInfo.inBuffBaseOff, tempAlgParamsScatterInter.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsScatterInter.repeatNum = 1;
        tempAlgParamsScatterInter.inputRepeatStride = 0;
        tempAlgParamsScatterInter.outputRepeatStride = 0;
        // 标准模板自行处理空步骤；所有 rank 均参与 Scatter MAIN/POST 相位。
        CHK_RET(algTemplateScatterInter->KernelRun(param, tempAlgParamsScatterInter, templateResourceInter));

        // ----------- 框间AllGather数据搬运 -----------
        // 框间的数据偏移和搬运计算
        tempAlgParamsAllGatherInter.count
            = tempAlgParamsScatterIntra.allRankProcessedDataCount.at(rankIdxLevel0_); // 沿用框间Scatter的切片结果
        const u64 l0SliceOffset = tempAlgParamsScatterIntra.allRankDispls.at(rankIdxLevel0_);
        tempAlgParamsAllGatherInter.buffInfo.inBuffBaseOff = l0SliceOffset;
        tempAlgParamsAllGatherInter.buffInfo.outBuffBaseOff = 0;
        // AG L1 的输入与输出均为 CCL：本 rank 数据从对应 L0 槽位出发，网络落到同基址网格。
        tempAlgParamsAllGatherInter.buffInfo.hcclBuffBaseOff = l0SliceOffset;
        tempAlgParamsAllGatherInter.root
            = (param.root / rankSizeLevel0_) * rankSizeLevel0_ + (myRank_ % rankSizeLevel0_);

        tempAlgParamsAllGatherInter.allRankDispls = tempAlgParamsScatterInter.allRankDispls;
        tempAlgParamsAllGatherInter.allRankSliceSize = tempAlgParamsScatterInter.allRankSliceSize;
        tempAlgParamsAllGatherInter.allRankProcessedDataCount = tempAlgParamsScatterInter.allRankProcessedDataCount;

        tempAlgParamsAllGatherInter.sliceSize = tempAlgParamsScatterInter.sliceSize;
        tempAlgParamsAllGatherInter.tailSize = tempAlgParamsScatterInter.tailSize;
        tempAlgParamsAllGatherInter.inputSliceStride = tempAlgParamsAllGatherInter.sliceSize;
        tempAlgParamsAllGatherInter.outputSliceStride = tempAlgParamsAllGatherInter.sliceSize;

        HCCL_DEBUG(
            "[InsV2BroadcastSequenceRhdExecutor] loop [%llu] tempAlgParamsAllGatherInter.inputSliceStride [%llu],"
            "tempAlgParamsAllGatherInter.outputSliceStride [%llu] tempAlgParamsAllGatherInter.sliceSize [%llu]",
            loop, tempAlgParamsAllGatherInter.inputSliceStride, tempAlgParamsAllGatherInter.outputSliceStride,
            tempAlgParamsAllGatherInter.sliceSize);
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceRhdExecutor] loop [%llu] tempAlgParamsAllGatherInter.buffInfo.inBuffBaseOff [%llu],"
            "tempAlgParamsAllGatherInter.buffInfo.outBuffBaseOff [%llu]",
            loop, tempAlgParamsAllGatherInter.buffInfo.inBuffBaseOff,
            tempAlgParamsAllGatherInter.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsAllGatherInter.repeatNum = 1;
        tempAlgParamsAllGatherInter.inputRepeatStride = 0;
        tempAlgParamsAllGatherInter.outputRepeatStride = 0;
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceRhdExecutor] loop [%llu] tempAlgParamsAllGatherInter.repeatNum [%llu],"
            "tempAlgParamsAllGatherInter.inputRepeatStride [%llu], tempAlgParamsAllGatherInter.outputRepeatStride "
            "[%llu]",
            loop, tempAlgParamsAllGatherInter.repeatNum, tempAlgParamsAllGatherInter.inputRepeatStride,
            tempAlgParamsAllGatherInter.outputRepeatStride);
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateAllGatherInter->KernelRun(param, tempAlgParamsAllGatherInter, templateResourceInter));

        // ----------- 框内AllGather数据搬运 -----------
        // 框内的数据偏移和搬运计算
        tempAlgParamsAllGatherIntra.count = currDataCount;
        tempAlgParamsAllGatherIntra.buffInfo.inBuffBaseOff = 0;
        tempAlgParamsAllGatherIntra.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParamsAllGatherIntra.buffInfo.hcclBuffBaseOff = 0;

        tempAlgParamsAllGatherIntra.allRankDispls
            = tempAlgParamsScatterIntra.allRankDispls; // 沿用框内Scatter的切片结果
        tempAlgParamsAllGatherIntra.allRankSliceSize = tempAlgParamsScatterIntra.allRankSliceSize;
        tempAlgParamsAllGatherIntra.allRankProcessedDataCount = tempAlgParamsScatterIntra.allRankProcessedDataCount;

        tempAlgParamsAllGatherIntra.sliceSize = tempAlgParamsScatterIntra.sliceSize;
        tempAlgParamsAllGatherIntra.tailSize = tempAlgParamsScatterIntra.tailSize;
        tempAlgParamsAllGatherIntra.inputSliceStride = tempAlgParamsAllGatherIntra.sliceSize;
        tempAlgParamsAllGatherIntra.outputSliceStride = tempAlgParamsAllGatherIntra.sliceSize;

        HCCL_DEBUG(
            "[InsV2BroadcastSequenceRhdExecutor] loop [%llu] tempAlgParamsAllGatherIntra.inputSliceStride [%llu],"
            "tempAlgParamsAllGatherIntra.outputSliceStride [%llu] tempAlgParamsAllGatherIntra.sliceSize [%llu]",
            loop, tempAlgParamsAllGatherIntra.inputSliceStride, tempAlgParamsAllGatherIntra.outputSliceStride,
            tempAlgParamsAllGatherIntra.sliceSize);
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceRhdExecutor] loop [%llu] tempAlgParamsAllGatherIntra.buffInfo.inBuffBaseOff [%llu],"
            "tempAlgParamsAllGatherIntra.buffInfo.outBuffBaseOff [%llu]",
            loop, tempAlgParamsAllGatherIntra.buffInfo.inBuffBaseOff,
            tempAlgParamsAllGatherIntra.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParamsAllGatherIntra.repeatNum = 1;
        tempAlgParamsAllGatherIntra.inputRepeatStride = 0;
        tempAlgParamsAllGatherIntra.outputRepeatStride = 0;
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceRhdExecutor] loop [%llu] tempAlgParamsAllGatherIntra.repeatNum [%llu],"
            "tempAlgParamsAllGatherIntra.inputRepeatStride [%llu], tempAlgParamsAllGatherIntra.outputRepeatStride "
            "[%llu]",
            loop, tempAlgParamsAllGatherIntra.repeatNum, tempAlgParamsAllGatherIntra.inputRepeatStride,
            tempAlgParamsAllGatherIntra.outputRepeatStride);
        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplateAllGatherIntra->KernelRun(param, tempAlgParamsAllGatherIntra, templateResourceIntra));

        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2BroadcastSequenceRhdExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2BroadcastSequenceRhdExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    SplitData(const u64& dataCount, const uint64_t& rankSize, TemplateDataParams& tempAlgParams)
{
    u32 sliceNum = rankSize;
    broadcast_sequence_executor::FillFloorTailSlices(dataCount, dataTypeSize_, sliceNum, tempAlgParams);
    for (u32 i = 0; i < tempAlgParams.allRankSliceSize.size(); ++i) {
        HCCL_DEBUG(
            "[InsV2BroadcastSequenceRhdExecutor] SliceInfo: offset[%llu] size[%llu] count[%llu]",
            tempAlgParams.allRankDispls.at(i), tempAlgParams.allRankSliceSize.at(i),
            tempAlgParams.allRankProcessedDataCount.at(i));
    }

    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_BROADCAST, DpuBroadcastSequenceMeshConcurRHD, InsV2BroadcastSequenceRhdExecutor,
    TopoMatchTwoLevel, AicpuTempScatterMesh1DZAxisDetour, InsTempScatterRhdDpuInterNode, InsTempAllGatherRHDDPU,
    InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(
    DpuBroadcastSequenceMeshConcurRHD, topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
    topo.isHostDpuOnly = true; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D; topo.isSupportLevel1Nhr = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    };);

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_BROADCAST, DpuBroadcastSequenceMeshConcurRHDRHD, BroadcastSequenceMesh1dNHRNHRExecutor,
    TopoMatchThreeLevel, AicpuTempScatterMesh1DZAxisDetour, InsTempScatterRHD, InsTempScatterRhdDpuInterNode,
    InsTempAllGatherRHDDPU, InsTempAllGatherRHD, InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(
    DpuBroadcastSequenceMeshConcurRHDRHD, topo.minTopoLevelNum = TOPO_LEVEL_NUM_3;
    topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.isHostDpuOnly = true; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    };);
} // namespace ops_hccl
