/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_sequence_executor.h"
#include "executor_common_ops.h"
#include "ins_temp_reduce_scatter_mesh_1D_intra.h"
#include "ins_temp_reduce_scatter_mesh_1D_dpu_inter.h"
#include "ins_temp_reduce_scatter_mesh_1D_nda_inter.h"
#include "ins_temp_reduce_scatter_nhr_dpu.h"
#include "ins_temp_all_gather_nhr_dpu_inter.h"
#include "ins_temp_all_gather_nhr_nda_inter.h"
#include "ins_temp_all_gather_mesh_1D_intra.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_v2_all_reduce_sequence_rhd_tail.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_all_gather_rhd_dpu.h"
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "ins_temp_reduce_scatter_rhd_dpu_inter.h"
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
#include "dpu_inline_reduce_selector.h"

namespace ops_hccl {
// ! 已经编码完成
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
std::vector<CostModelParam>
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AllReduceSequenceExecutor][CalcCostCoeff]",
            TopoProbeScene::PROBE_CALC_COST_COEFF)) {
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 physIdxLevel0 = static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]);
    u32 physIdxLevel1 = (algHierarchyInfo.physicalIdxForAlgoLevels.size() > 1) ?
                            static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[1][0]) :
                            physIdxLevel0;

    CommTopo netTypeLevel1 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel1);
    CommTopo netTypeLevel0 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel0);

    std::vector<u32> portNumLevel0 = GetPhysicalLevelPortNums(topoInfo, physIdxLevel0);
    std::vector<u32> portNumLevel1 = GetPhysicalLevelPortNums(topoInfo, physIdxLevel1);

    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;

    if (portNumLevel1.empty()) {
        HCCL_WARNING("[InsV2AllReduceSequenceExecutor][CalcCostCoeff] portNum is empty");
        return {};
    }

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutor][CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, "
        "netTypeLevel0=%d, netTypeLevel1=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, static_cast<int>(netTypeLevel0), static_cast<int>(netTypeLevel1));
    std::vector<CostModelParam> params = [rankSize, rankSizeLevel0, rankSizeLevel1, portNumLevel0, portNumLevel1,
                                          netTypeLevel0, netTypeLevel1, isPod] {
        std::vector<CostModelParam> v;
        auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel0, 1.0f / rankSizeLevel0, netTypeLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel0, isPod});
        v.insert(v.end(), p0.begin(), p0.end());
        auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel1, 1.0f / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel1, isPod});
        v.insert(v.end(), p1.begin(), p1.end());
        auto p2 = InsAlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel1, 1.0f / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
            BufferType::HCCL_BUFFER, portNumLevel1, isPod});
        v.insert(v.end(), p2.begin(), p2.end());
        auto p3 = InsAlgTemplate3::CalcCostCoeff(CalcCostCoeffParam{
            rankSizeLevel0, 1.0f / rankSizeLevel0, netTypeLevel0, BufferType::HCCL_BUFFER, BufferType::OUTPUT,
            BufferType::HCCL_BUFFER, portNumLevel0, isPod});
        v.insert(v.end(), p3.begin(), p3.end());
        return v;
    }();
    return params;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
AlgNetMeta
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AllReduceSequenceExecutor][GetAlgNetMeta]",
            TopoProbeScene::PROBE_GET_ALG_NET_META)) {
        return {};
    }

    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 physIdxLevel0 = static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[0][0]);
    u32 physIdxLevel1 = (algHierarchyInfo.physicalIdxForAlgoLevels.size() > 1) ?
                            static_cast<u32>(algHierarchyInfo.physicalIdxForAlgoLevels[1][0]) :
                            physIdxLevel0;

    CommTopo netTypeLevel0 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel0);
    CommTopo netTypeLevel1 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel1);

    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;

    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel0);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1, 1, 1};
    meta.dataRatios = {1.0f / rankSizeLevel0, 1.0f / rankSize, 1.0f / rankSize, 1.0f / rankSizeLevel0};
    meta.rankSizes = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel1, rankSizeLevel0};
    return meta;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
InsV2AllReduceSequenceExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::InsV2AllReduceSequenceExecutor()
{}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    InitCommonCommInfo(param, topoInfo);

    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutor][InitCommInfo] myRank [%u], rankSize [%u], devType [%u], redOp [%u], "
        "dataType [%u] dataTypeSize [%u]",
        myRank_, rankSize_, devType_, reduceOp_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

// 实例化实际执行以来AutoMatchMeshNhr这个类的实现
template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);

    if (algHierarchyInfo.infos.empty() || algHierarchyInfo.infos.size() < TOPO_LEVEL_NUM_2) {
        HCCL_ERROR("[%s] algHierarchyInfo.infos is invalid (empty or size < 2).", __func__);
        return HCCL_E_PARA;
    }
    std::vector<std::vector<u32>> intraHierarchyInfo;
    std::vector<std::vector<u32>> interHierarchyInfo;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix
        && algHierarchyInfo.infos[0].size() > 1) {
        intraHierarchyInfo = {algHierarchyInfo.infos[0][1]};
        interHierarchyInfo = algHierarchyInfo.infos[1];
    } else {
        intraHierarchyInfo = algHierarchyInfo.infos[0];
        interHierarchyInfo = algHierarchyInfo.infos[1];
    }

    std::shared_ptr<InsAlgTemplate0> reduceScatterIntraTempAlg
        = std::make_shared<InsAlgTemplate0>(param, myRank_, intraHierarchyInfo);
    std::shared_ptr<InsAlgTemplate1> reduceScatterInterTempAlg
        = std::make_shared<InsAlgTemplate1>(param, myRank_, interHierarchyInfo);
    std::shared_ptr<InsAlgTemplate2> allGatherInterTempAlg
        = std::make_shared<InsAlgTemplate2>(param, myRank_, interHierarchyInfo);
    std::shared_ptr<InsAlgTemplate3> allGatherIntraTempAlg
        = std::make_shared<InsAlgTemplate3>(param, myRank_, intraHierarchyInfo);

    AlgResourceRequest resReqReduceScatterIntra;
    AlgResourceRequest resReqReduceScatterInter;
    AlgResourceRequest resReqAllGatherInter;
    AlgResourceRequest resReqAllGatherIntra;
    CHK_RET(reduceScatterIntraTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterIntra));
    CHK_RET(reduceScatterInterTempAlg->CalcRes(comm, param, topoInfo, resReqReduceScatterInter));
    CHK_RET(allGatherInterTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherInter));
    CHK_RET(allGatherIntraTempAlg->CalcRes(comm, param, topoInfo, resReqAllGatherIntra));

    // step1、2、3、4为串行，因此slaveThread和对应notify可以复用
    resourceRequest.slaveThreadNum
        = std::max(resReqReduceScatterIntra.slaveThreadNum, resReqReduceScatterInter.slaveThreadNum);
    resourceRequest.notifyNumPerThread.clear();
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        if (i < resReqReduceScatterIntra.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReqReduceScatterIntra.notifyNumPerThread[i]);
        }
        if (i < resReqReduceScatterInter.notifyNumPerThread.size()) {
            resourceRequest.notifyNumPerThread[i]
                = std::max(resourceRequest.notifyNumPerThread[i], resReqReduceScatterInter.notifyNumPerThread[i]);
        }
    }
    resourceRequest.notifyNumOnMainThread
        = std::max(resReqReduceScatterIntra.notifyNumOnMainThread, resReqReduceScatterInter.notifyNumOnMainThread);

    resourceRequest.channels = {resReqReduceScatterIntra.channels[0], resReqReduceScatterInter.channels[0]};
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutor][Orchestrate] Orchestrate Start");
    // 参数填充
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;

    if (algHierarchyInfo_.infos.empty() || algHierarchyInfo_.infos.size() < TOPO_LEVEL_NUM_2
        || algHierarchyInfo_.infos[0][0].empty() || algHierarchyInfo_.infos[1][0].empty()) {
        HCCL_ERROR("[%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix
        && resCtx.algHierarchyInfo.infos[0].size() > 1) {
        intraHierarchyInfo_ = {resCtx.algHierarchyInfo.infos[0][1]};
        interHierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    } else {
        intraHierarchyInfo_ = resCtx.algHierarchyInfo.infos[0];
        interHierarchyInfo_ = resCtx.algHierarchyInfo.infos[1];
    }
    rankSizeLevel0_ = intraHierarchyInfo_[0].size();
    rankSizeLevel1_ = interHierarchyInfo_[0].size();
    rankIdxLevel0_ = myRank_ % rankSizeLevel0_;
    rankIdxLevel1_ = myRank_ / rankSizeLevel0_;
    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutor][Orchestrate]errNo[0x%016llx] AllReduce executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    PrepareLoopContext(const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx)
{
    // 将ccl-buffer分成ccl-in和ccl-out 2部分区分使用
    void* cclInAddr = resCtx.cclMem.addr;
    ctx.cclInMem = {resCtx.cclMem.type, cclInAddr, resCtx.cclMem.size / 2};
    void* cclOutAddr = static_cast<void*>(static_cast<s8*>(resCtx.cclMem.addr) + resCtx.cclMem.size / 2);
    ctx.cclOutMem = {resCtx.cclMem.type, cclOutAddr, resCtx.cclMem.size / 2};

    // 声明框内ReduceScatterMesh1D的templateargs
    ctx.rsIntra.params.buffInfo.inputPtr = param.inputPtr;
    ctx.rsIntra.params.buffInfo.outputPtr = ctx.cclOutMem.addr;
    ctx.rsIntra.params.buffInfo.hcclBuff = ctx.cclInMem;
    ctx.rsIntra.params.buffInfo.inBuffType = BufferType::INPUT;
    ctx.rsIntra.params.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    ctx.rsIntra.params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框内ReduceScatterMesh1D的template
    ctx.rsIntra.algorithm = std::make_shared<InsAlgTemplate0>(param, myRank_, intraHierarchyInfo_);

    // 声明框间ReduceScatterDpu的templateargs
    ctx.rsInter.params.buffInfo.inputPtr = ctx.cclOutMem.addr;
    ctx.rsInter.params.buffInfo.outputPtr = ctx.cclOutMem.addr;
    ctx.rsInter.params.buffInfo.hcclBuff = ctx.cclInMem;
    ctx.rsInter.params.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    ctx.rsInter.params.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    ctx.rsInter.params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框间ReduceScatterDpu的template
    ctx.rsInter.algorithm = std::make_shared<InsAlgTemplate1>(param, myRank_, interHierarchyInfo_);

    // 声明框间AllGatherNhr的templateargs，ccl-out搬运到ccl-out
    ctx.agInter.params.buffInfo.inputPtr = ctx.cclOutMem.addr;
    ctx.agInter.params.buffInfo.outputPtr = ctx.cclOutMem.addr;
    ctx.agInter.params.buffInfo.hcclBuff = ctx.cclInMem;
    ctx.agInter.params.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    ctx.agInter.params.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    ctx.agInter.params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框间AllGatherNhrDPU的template
    ctx.agInter.algorithm = std::make_shared<InsAlgTemplate2>(param, myRank_, interHierarchyInfo_);

    // 声明框内AllGatherMesh1D的templateargs，ccl-out搬运到user-out
    ctx.agIntra.params.buffInfo.inputPtr = ctx.cclOutMem.addr;
    ctx.agIntra.params.buffInfo.outputPtr = param.outputPtr;
    ctx.agIntra.params.buffInfo.hcclBuff = ctx.cclInMem;
    ctx.agIntra.params.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    ctx.agIntra.params.buffInfo.outBuffType = BufferType::OUTPUT;
    ctx.agIntra.params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 构建框内AllGatherMesh1D的template
    ctx.agIntra.algorithm = std::make_shared<InsAlgTemplate3>(param, myRank_, intraHierarchyInfo_);

    // 构造框内ReduceScatterMesh1D的template资源
    SetTemplateCommResource(ctx.rsIntra.resource, remoteRankToChannelInfo_[0], resCtx);
    // 构造框间ReduceScatterDpu的template资源
    SetTemplateCommResource(ctx.rsInter.resource, remoteRankToChannelInfo_[1], resCtx);
    ctx.rsInter.resource.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
    ctx.rsInter.resource.dpuThread = resCtx.dpuThread;
    ctx.rsInter.resource.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
    ctx.rsInter.resource.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;
    // 构造框间AllGatherNhrDPU的template资源
    SetTemplateCommResource(ctx.agInter.resource, remoteRankToChannelInfo_[1], resCtx);
    ctx.agInter.resource.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
    ctx.agInter.resource.dpuThread = resCtx.dpuThread;
    ctx.agInter.resource.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
    ctx.agInter.resource.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;
    // 构造框内AllGatherMesh1D的template资源
    SetTemplateCommResource(ctx.agIntra.resource, remoteRankToChannelInfo_[0], resCtx);

    // 中转内存单次最多能够接受的output count，注意是count不是size
    ctx.maxCountPerLoop = ctx.rsIntra.params.buffInfo.hcclBuff.size / 2 / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN
                          / dataTypeSize_; // 这边看前面有/10*10，不知道要不要加上
    CHK_PRT_RET(
        ctx.maxCountPerLoop == 0,
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutor][OrchestrateLoop] maxCountPerLoop is 0, cclMem too small, "
            "dataTypeSize[%llu].",
            dataTypeSize_),
        HcclResult::HCCL_E_INTERNAL);
    // 计算loopTimes
    ctx.loopTimes = dataCount_ / ctx.maxCountPerLoop + static_cast<u64>(dataCount_ % ctx.maxCountPerLoop != 0);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    PrepareRhdLoop(const OpParam& param, LoopContext& ctx)
{
    // RHD 使用均匀槽；保留 RS/AG 配对、线程裁剪和按 rank 对齐的容量限制。
    CHK_PRT_RET(
        param.supportSymmetricMemory,
        HCCL_ERROR(
            "[%s] algName[%s] RHD sequence not support symmetric memory, fallback by reselecting HCCL_ALGO.", __func__,
            param.algName),
        HcclResult::HCCL_E_NOT_SUPPORT);
    // CalcRes 与执行期使用不同模板实例，需刷新通道数以匹配线程和 notify 数量。
    // 单 rank 层级无通道，跳过刷新。
    if (rankSizeLevel0_ > 1) {
        CHK_RET(ctx.rsIntra.algorithm->SetchannelsPerRank(remoteRankToChannelInfo_[0]));
        CHK_RET(ctx.agIntra.algorithm->SetchannelsPerRank(remoteRankToChannelInfo_[0]));
    }
    // 各阶段复用共享线程池；Step4 仅接收自身所需线程，保持从线程与 notify 数量一致。
    const u64 allGatherThreadNum = ctx.agIntra.algorithm->GetThreadNum();
    CHK_PRT_RET(
        allGatherThreadNum == 0 || ctx.agIntra.resource.threads.size() < allGatherThreadNum,
        HCCL_ERROR(
            "[%s] Step4 AllGather threads insufficient, available[%zu], required[%llu].", __func__,
            ctx.agIntra.resource.threads.size(), allGatherThreadNum),
        HcclResult::HCCL_E_INTERNAL);
    ctx.agIntra.resource.threads.resize(allGatherThreadNum);
    if (rankSizeLevel1_ > 1) {
        CHK_RET(ctx.rsInter.algorithm->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
        CHK_RET(ctx.agInter.algorithm->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }
    ctx.totalRankAlign = static_cast<u32>(rankSizeLevel0_ * rankSizeLevel1_);
    ctx.maxCountPerLoop
        = ctx.maxCountPerLoop / ctx.totalRankAlign * ctx.totalRankAlign; // 均匀槽对齐（同 aicpu 二级口径）
    CHK_PRT_RET(
        ctx.maxCountPerLoop == 0,
        HCCL_ERROR(
            "[%s] algName[%s] maxCountPerLoop is 0 after rank alignment, cclMem too small for RHD.", __func__,
            param.algName),
        HcclResult::HCCL_E_INTERNAL);
    ctx.loopTimes = dataCount_ / ctx.maxCountPerLoop + static_cast<u64>(dataCount_ % ctx.maxCountPerLoop != 0);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunRhdReduceScatterIntra(const OpParam& param, LoopContext& ctx, LoopState& state)
{
    // Step1：框内 MeshConcur ReduceScatter，输出到 CCL-OUT，scratch 使用 CCL-IN。
    // 显式设置均匀分片参数，避免模板因 sliceSize/tailSize 为零而跳过执行。
    state.chunkCount = state.currDataCount / rankSizeLevel0_; // 本 rank L0 chunk（元素数，整除均匀）
    state.chunkSize = state.chunkCount * dataTypeSize_;
    ctx.rsIntra.params.count = state.currDataCount;
    if (state.validDataCount != state.currDataCount) {
        ctx.rsIntra.params.buffInfo.inputPtr = ctx.tail.storage.addr;
        ctx.rsIntra.params.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        ctx.rsIntra.params.buffInfo.inBuffBaseOff = ctx.tail.inputOffset;
    } else {
        ctx.rsIntra.params.buffInfo.inBuffBaseOff = state.processedDataCount * dataTypeSize_;
    }
    ctx.rsIntra.params.buffInfo.outBuffBaseOff = 0;  // CCL-OUT（cclOutMem 基址）
    ctx.rsIntra.params.buffInfo.hcclBuffBaseOff = 0; // CCL-IN scratch（cclInMem 基址）
    CHK_RET(SplitData(state.currDataCount, rankSizeLevel0_, ctx.rsIntra.params));
    ctx.rsIntra.params.sliceSize = state.chunkSize;
    ctx.rsIntra.params.tailSize = state.chunkSize;         // 本轮均匀槽 tail==slice
    ctx.rsIntra.params.inputSliceStride = state.chunkSize; // IN[base+i·ISS] = 本 rank chunk 的第 i 子片
    ctx.rsIntra.params.outputSliceStride = 0;              // OUT[base] = 本 rank chunk（无 alg-rank 项）
    ctx.rsIntra.params.repeatNum = 1;
    ctx.rsIntra.params.inputRepeatStride = 0;
    ctx.rsIntra.params.outputRepeatStride = 0;
    CHK_RET(ctx.rsIntra.algorithm->KernelRun(param, ctx.rsIntra.params, ctx.rsIntra.resource));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunRhdReduceScatterInter(const OpParam& param, LoopContext& ctx, LoopState& state)
{
    // Step2：框间 RHD ReduceScatter，输入/输出使用 CCL-OUT，scratch 使用 CCL-IN。
    state.subSliceCount = state.chunkCount / rankSizeLevel1_; // 本轮均匀槽
    state.subSliceSize = state.subSliceCount * dataTypeSize_;
    ctx.rsInter.params.count = state.chunkCount;
    ctx.rsInter.params.sliceSize = state.subSliceSize; // stride 契约字段（device RHD 同式；本模板均匀退化等价）
    ctx.rsInter.params.tailSize = state.subSliceSize;
    ctx.rsInter.params.inputSliceStride = state.subSliceSize; // 模板自推导 allRankDispls[j] = j·ISS
    ctx.rsInter.params.outputSliceStride = state.subSliceSize;
    ctx.rsInter.params.buffInfo.inBuffBaseOff = 0;  // 输入 = ccl-out[0] 本 rank chunk
    ctx.rsInter.params.buffInfo.outBuffBaseOff = 0; // 输出 = ccl-out[0] 本 rank 子份额（终写无 alg-rank 项）
    ctx.rsInter.params.buffInfo.hcclBuffBaseOff = 0; // scratch = ccl-in 基址区（acc+recv 双区，二级不隔离）
    ctx.rsInter.params.repeatNum = 1;
    ctx.rsInter.params.inputRepeatStride = 0;
    ctx.rsInter.params.outputRepeatStride = 0;
    ctx.rsInter.params.rhdSkipPost = true;
    // allRank* 由 RhdDpuInter 按 inputSliceStride 推导，executor 不预填。
    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutor] loop [%llu] RHD StepTwo chunkCount[%llu] subSliceCount[%llu] "
        "subSliceSize[%llu]",
        state.loop, state.chunkCount, state.subSliceCount, state.subSliceSize);
    CHK_RET(ctx.rsInter.algorithm->KernelRun(param, ctx.rsInter.params, ctx.rsInter.resource));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunRhdAllGatherInter(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx, LoopState& state)
{
    // Step3：框间 RHD AllGather，跳过 PRE。
    ctx.agInter.params.count = state.chunkCount;
    ctx.agInter.params.sliceSize = state.subSliceSize;
    ctx.agInter.params.tailSize = state.subSliceSize;
    ctx.agInter.params.inputSliceStride = 0; // 装载源 = ccl-out[0] 本 rank 子份额（无 alg-rank 项）
    ctx.agInter.params.outputSliceStride = state.subSliceSize; // 网格 [owner·SS]
    // 输入与 scratch 统一基址，使模板能够识别并跳过同址拷贝。
    ctx.agInter.params.buffInfo.inputPtr = resCtx.cclMem.addr;
    ctx.agInter.params.buffInfo.inBuffBaseOff = resCtx.cclMem.size / 2;
    // AG 输出与 scratch 共用 CCL-OUT 网格，均使用相对完整 CCL 的偏移。
    ctx.agInter.params.buffInfo.outBuffBaseOff = resCtx.cclMem.size / 2;
    ctx.agInter.params.buffInfo.hcclBuffBaseOff = resCtx.cclMem.size / 2; // CCL 绝对偏移（对端对称）
    ctx.agInter.params.buffInfo.hcclBuff = resCtx.cclMem; // 基址区视图（与 remoteCclMem 同基准）
    ctx.agInter.params.repeatNum = 1;
    ctx.agInter.params.inputRepeatStride = 0;
    ctx.agInter.params.outputRepeatStride = 0;
    ctx.agInter.params.skipPrePhase = true;
    CHK_RET(ctx.agInter.algorithm->KernelRun(param, ctx.agInter.params, ctx.agInter.resource));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunRhdAllGatherIntra(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx, LoopState& state)
{
    // Step4：从 CCL-OUT 网格执行框内 AllGather，回写用户输出。
    // 输入与 scratch 同基址、同偏移，跳过模板的前置本地拷贝。
    ctx.agIntra.params.count = state.currDataCount;
    ctx.agIntra.params.buffInfo.inputPtr = resCtx.cclMem.addr; // 基址视图（偏移承载区域）
    ctx.agIntra.params.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    ctx.agIntra.params.buffInfo.inBuffBaseOff = resCtx.cclMem.size / 2; // CCL-OUT 网格基址
    if (state.validDataCount != state.currDataCount) {
        ctx.agIntra.params.buffInfo.outputPtr = ctx.tail.storage.addr;
        ctx.agIntra.params.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
        ctx.agIntra.params.buffInfo.outBuffBaseOff = ctx.tail.outputOffset;
    } else {
        ctx.agIntra.params.buffInfo.outBuffBaseOff = state.processedDataCount * dataTypeSize_;
    }
    ctx.agIntra.params.buffInfo.hcclBuff = resCtx.cclMem;
    ctx.agIntra.params.buffInfo.hcclBuffBaseOff = resCtx.cclMem.size / 2; // 与 inBuffBaseOff 同址
    ctx.agIntra.params.allRankDispls = ctx.rsIntra.params.allRankDispls;
    ctx.agIntra.params.allRankSliceSize = ctx.rsIntra.params.allRankSliceSize;
    ctx.agIntra.params.allRankProcessedDataCount = ctx.rsIntra.params.allRankProcessedDataCount;
    ctx.agIntra.params.sliceSize = state.chunkSize;
    ctx.agIntra.params.tailSize = state.chunkSize;
    ctx.agIntra.params.inputSliceStride = 0;                // 本 rank chunk 在网格 [myIdx·OSS]
    ctx.agIntra.params.outputSliceStride = state.chunkSize; // 输出网格 [owner·chunkSize]
    ctx.agIntra.params.repeatNum = 1;
    ctx.agIntra.params.inputRepeatStride = 0;
    ctx.agIntra.params.outputRepeatStride = 0;
    CHK_RET(ctx.agIntra.algorithm->KernelRun(param, ctx.agIntra.params, ctx.agIntra.resource));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunRhdLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx, LoopContext& ctx)
{
    CHK_RET(PrepareRhdLoop(param, ctx));
    LoopState state;
    while (state.processedDataCount < dataCount_) {
        const ArSeqRhdLoopPlan plan
            = ArSeqRhdPlanLoop(dataCount_ - state.processedDataCount, ctx.maxCountPerLoop, ctx.totalRankAlign);
        state.validDataCount = plan.validCount;
        state.currDataCount = plan.currDataCount;
        if (plan.paddedTail) {
            CHK_RET(ArSeqRhdStageTail(param, ctx.tail, state.processedDataCount, state.validDataCount, threads_[0]));
        }
        CHK_RET(RunRhdReduceScatterIntra(param, ctx, state));
        CHK_RET(RunRhdReduceScatterInter(param, ctx, state));
        // 将代表 scratch 中的被合并方份额放入 CCL-OUT 的 AG owner 槽。
        CHK_RET(ArSeqRhdBridgeMergedShare(
            static_cast<u32>(rankSizeLevel1_), static_cast<u32>(rankIdxLevel1_), state.subSliceSize,
            state.subSliceCount, resCtx.cclMem.addr, 0, resCtx.cclMem.addr, resCtx.cclMem.size / 2, threads_[0]));
        CHK_RET(RunRhdAllGatherInter(param, resCtx, ctx, state));
        CHK_RET(RunRhdAllGatherIntra(param, resCtx, ctx, state));
        if (plan.paddedTail) {
            CHK_RET(
                ArSeqRhdCopyTailOutput(param, ctx.tail, state.processedDataCount, state.validDataCount, threads_[0]));
        }
        state.processedDataCount += state.validDataCount;
        state.loop++;
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunDefaultReduceScatterIntra(const OpParam& param, LoopContext& ctx, LoopState& state)
{
    // ----------- 框内ReduceScatter数据搬运 -----------
    // 框内的数据偏移和搬运计算
    ctx.rsIntra.params.count = state.currDataCount;
    ctx.rsIntra.params.buffInfo.inBuffBaseOff = state.processedDataCount * dataTypeSize_;
    ctx.rsIntra.params.buffInfo.outBuffBaseOff = 0;  // CCL-OUT
    ctx.rsIntra.params.buffInfo.hcclBuffBaseOff = 0; // CCL-IN
    CHK_RET(
        SplitData(state.currDataCount, rankSizeLevel0_, ctx.rsIntra.params)); // 计算每个卡对应位置的offset,count,size
    CHK_PRT_RET(
        ctx.rsIntra.params.allRankSliceSize.size() != rankSizeLevel0_,
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutor][tempAlgParamsStepOne] slice num[%zu] is not equal to rank size[%u].",
            ctx.rsIntra.params.allRankSliceSize.size(), rankSizeLevel0_),
        HcclResult::HCCL_E_INTERNAL);
    ctx.rsIntra.params.sliceSize = 0; // 没用到，template里面用SplitData算了
    ctx.rsIntra.params.tailSize = 0;  // 没用到
    // 这里的stride当成传统意义上的stride间隔
    ctx.rsIntra.params.inputSliceStride = 0;  // 没用到
    ctx.rsIntra.params.outputSliceStride = 0; // 没用到

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutor] loop [%llu] tempAlgParamsStepOne.inputSliceStride [%llu],"
        "tempAlgParamsStepOne.outputSliceStride [%llu] tempAlgParamsStepOne.sliceSize [%llu]",
        state.loop, ctx.rsIntra.params.inputSliceStride, ctx.rsIntra.params.outputSliceStride,
        ctx.rsIntra.params.sliceSize);
    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutor] loop [%llu] tempAlgParamsStepOne.buffInfo.inBuffBaseOff [%llu],"
        "tempAlgParamsStepOne.buffInfo.outBuffBaseOff [%llu]",
        state.loop, ctx.rsIntra.params.buffInfo.inBuffBaseOff, ctx.rsIntra.params.buffInfo.outBuffBaseOff);
    // 不需要重复
    ctx.rsIntra.params.repeatNum = 1;
    ctx.rsIntra.params.inputRepeatStride = 0;
    ctx.rsIntra.params.outputRepeatStride = 0;
    // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
    CHK_RET(ctx.rsIntra.algorithm->KernelRun(param, ctx.rsIntra.params, ctx.rsIntra.resource));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunDefaultReduceScatterInter(const OpParam& param, LoopContext& ctx, LoopState& state)
{
    // ----------- 框间ReduceScatterDpu数据搬运 -----------
    // 框间的数据偏移和搬运量计算
    ctx.rsInter.params.count = ctx.rsIntra.params.allRankProcessedDataCount.at(rankIdxLevel0_);
    if (ctx.rsInter.params.count != 0) { // 如果卡里没有数据，不参与框间，让该框有数据的其他卡参与跨框
        ctx.rsInter.params.buffInfo.inBuffBaseOff = 0;   // ccl-out偏移量，每次更新，所以是0
        ctx.rsInter.params.buffInfo.outBuffBaseOff = 0;  // ccl-out偏移量，每次更新，所以是0
        ctx.rsInter.params.buffInfo.hcclBuffBaseOff = 0; // ccl-in
        ctx.rsInter.params.sliceSize = 0;
        ctx.rsInter.params.tailSize = 0;
        CHK_RET(SplitData(ctx.rsInter.params.count, rankSizeLevel1_, ctx.rsInter.params));
        CHK_PRT_RET(
            ctx.rsInter.params.allRankSliceSize.size() != rankSizeLevel1_,
            HCCL_ERROR(
                "[InsV2AllReduceSequenceExecutor][tempAlgParamsStepTwo] slice num[%zu] is not "
                "equal to rank size[%u].",
                ctx.rsInter.params.allRankSliceSize.size(), rankSizeLevel1_),
            HcclResult::HCCL_E_INTERNAL);
        // 这里的stride当成传统意义上的stride 间隔
        ctx.rsInter.params.inputSliceStride = 0;  // 没用到
        ctx.rsInter.params.outputSliceStride = 0; // 没用到
        //
        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutor] loop [%llu] tempAlgParamsStepTwo.inputSliceStride [%llu],"
            "tempAlgParamsStepTwo.outputSliceStride [%llu] "
            "tempAlgParamsStepTwo.sliceSize [%llu]",
            state.loop, ctx.rsInter.params.inputSliceStride, ctx.rsInter.params.outputSliceStride,
            ctx.rsInter.params.sliceSize);
        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutor] loop [%llu] "
            "tempAlgParamsStepTwo.buffInfo.inBuffBaseOff [%llu],"
            "tempAlgParamsStepTwo.buffInfo.outBuffBaseOff [%llu]",
            state.loop, ctx.rsInter.params.buffInfo.inBuffBaseOff, ctx.rsInter.params.buffInfo.outBuffBaseOff);

        ctx.rsInter.params.repeatNum = 1;
        ctx.rsInter.params.inputRepeatStride = 0;
        ctx.rsInter.params.outputRepeatStride = 0;

        CHK_RET(ctx.rsInter.algorithm->KernelRun(param, ctx.rsInter.params, ctx.rsInter.resource));
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunDefaultAllGatherInter(const OpParam& param, LoopContext& ctx, LoopState& state)
{
    // ----------- 框间AllGatherNhr数据搬运 -----------
    // 框间的数据偏移和搬运量计算
    ctx.agInter.params.count = ctx.rsIntra.params.allRankProcessedDataCount.at(rankIdxLevel0_);
    if (ctx.rsInter.params.count != 0) {
        ctx.agInter.params.buffInfo.inBuffBaseOff = 0;   // ccl-out偏移量，每次更新，所以是0
        ctx.agInter.params.buffInfo.outBuffBaseOff = 0;  // ccl-out偏移量，每次更新，所以是0
        ctx.agInter.params.buffInfo.hcclBuffBaseOff = 0; // ccl-in
        ctx.agInter.params.allRankDispls = ctx.rsInter.params.allRankDispls;
        ctx.agInter.params.allRankSliceSize = ctx.rsInter.params.allRankSliceSize;
        ctx.agInter.params.allRankProcessedDataCount = ctx.rsInter.params.allRankProcessedDataCount;

        ctx.agInter.params.sliceSize = 0;
        ctx.agInter.params.tailSize = 0;
        // 这里的stride当成传统意义上的stride 间隔
        ctx.agInter.params.inputSliceStride = 0;
        ctx.agInter.params.outputSliceStride = 0;

        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutor] loop [%llu] tempAlgParamsStepThree.inputSliceStride [%llu],"
            "tempAlgParamsStepThree.outputSliceStride [%llu] tempAlgParamsStepThree.sliceSize [%llu]",
            state.loop, ctx.agInter.params.inputSliceStride, ctx.agInter.params.outputSliceStride,
            ctx.agInter.params.sliceSize);
        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutor] loop [%llu] tempAlgParamsStepThree.buffInfo.inBuffBaseOff [%llu],"
            "tempAlgParamsStepThree.buffInfo.outBuffBaseOff [%llu]",
            state.loop, ctx.agInter.params.buffInfo.inBuffBaseOff, ctx.agInter.params.buffInfo.outBuffBaseOff);

        ctx.agInter.params.repeatNum = 1;
        ctx.agInter.params.inputRepeatStride = 0;
        ctx.agInter.params.outputRepeatStride = 0;

        CHK_RET(ctx.agInter.algorithm->KernelRun(param, ctx.agInter.params, ctx.agInter.resource));
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunDefaultAllGatherIntra(const OpParam& param, LoopContext& ctx, LoopState& state)
{
    // ----------- 框内AllGatherMesh数据搬运 -----------
    // 框内的数据偏移和搬运计算
    ctx.agIntra.params.count = state.currDataCount;
    ctx.agIntra.params.buffInfo.inBuffBaseOff = 0;                                         // CCL-OUT
    ctx.agIntra.params.buffInfo.outBuffBaseOff = state.processedDataCount * dataTypeSize_; // USR-OUT
    ctx.agIntra.params.buffInfo.hcclBuffBaseOff = 0;                                       // CCL-IN
    ctx.agIntra.params.allRankDispls = ctx.rsIntra.params.allRankDispls;
    ctx.agIntra.params.allRankSliceSize = ctx.rsIntra.params.allRankSliceSize;
    ctx.agIntra.params.allRankProcessedDataCount = ctx.rsIntra.params.allRankProcessedDataCount;

    ctx.agIntra.params.sliceSize = 0;
    ctx.agIntra.params.tailSize = 0;
    // 这里的stride当成传统意义上的stride间隔
    ctx.agIntra.params.inputSliceStride = 0;
    ctx.agIntra.params.outputSliceStride = 0;

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutor] loop [%llu] tempAlgParamsStepFour.inputSliceStride [%llu],"
        "tempAlgParamsStepFour.outputSliceStride [%llu] tempAlgParamsStepFour.sliceSize [%llu]",
        state.loop, ctx.agIntra.params.inputSliceStride, ctx.agIntra.params.outputSliceStride,
        ctx.agIntra.params.sliceSize);
    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutor] loop [%llu] tempAlgParamsStepFour.buffInfo.inBuffBaseOff [%llu],"
        "tempAlgParamsStepFour.buffInfo.outBuffBaseOff [%llu]",
        state.loop, ctx.agIntra.params.buffInfo.inBuffBaseOff, ctx.agIntra.params.buffInfo.outBuffBaseOff);

    ctx.agIntra.params.repeatNum = 1;
    ctx.agIntra.params.inputRepeatStride = 0;
    ctx.agIntra.params.outputRepeatStride = 0;

    CHK_RET(ctx.agIntra.algorithm->KernelRun(param, ctx.agIntra.params, ctx.agIntra.resource));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RunDefaultLoop(const OpParam& param, LoopContext& ctx)
{
    LoopState state;
    for (state.loop = 0; state.loop < ctx.loopTimes; state.loop++) {
        state.currDataCount = (state.loop == ctx.loopTimes - 1) ? dataCount_ - state.processedDataCount :
                                                                  ctx.maxCountPerLoop; // 判断是最后一轮，就处理尾块长度
        CHK_RET(RunDefaultReduceScatterIntra(param, ctx, state));
        CHK_RET(RunDefaultReduceScatterInter(param, ctx, state));
        CHK_RET(RunDefaultAllGatherInter(param, ctx, state));
        CHK_RET(RunDefaultAllGatherIntra(param, ctx, state));
        state.processedDataCount += state.currDataCount;
    }
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutor][OrchestrateLoop] Start");
    LoopContext ctx;
    AlgResourceCtxSerializable loopResCtx = resCtx;
    if constexpr (IsRhdTemplate<InsAlgTemplate1>::value) {
        CHK_RET(ArSeqRhdPrepareTailWorkspace(
            dataCount_, rankSizeLevel0_ * rankSizeLevel1_, dataTypeSize_, loopResCtx.cclMem, ctx.tail));
    }
    CHK_RET(PrepareLoopContext(param, loopResCtx, ctx));
    if constexpr (IsRhdTemplate<InsAlgTemplate1>::value) {
        CHK_RET(RunRhdLoop(param, loopResCtx, ctx));
    } else {
        // 非 RHD 绑定的默认路径：既包括默认选路命中既有算法注册，
        // 也包括显式配置的 RHD 因准入失败（WARNING 回退自动选路）后由既有算法执行的情况。
        CHK_RET(RunDefaultLoop(param, ctx));
    }
    HCCL_INFO("[InsV2AllReduceSequenceExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
u64 InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    RoundUp(const u64 dividend, const u64 divisor)
{
    if (divisor == 0) {
        HCCL_WARNING("[InsV2AllReduceSequenceExecutor][RoundUp] divisor is 0.");
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3>
HcclResult
InsV2AllReduceSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3>::
    SplitData(const u64& dataCount, const uint64_t& rankSize, TemplateDataParams& tempAlgParams)
{
    u32 sliceNum = rankSize;
    tempAlgParams.allRankSliceSize.clear();
    tempAlgParams.allRankProcessedDataCount.clear();
    tempAlgParams.allRankDispls.clear();
    tempAlgParams.allRankSliceSize.reserve(sliceNum);
    tempAlgParams.allRankDispls.reserve(sliceNum);
    tempAlgParams.allRankProcessedDataCount.reserve(sliceNum);

    u64 offsetCount = 0;
    u64 offsetSize = 0;
    u64 sliceCount = RoundUp(dataCount, sliceNum);
    u64 sliceSize = sliceCount * dataTypeSize_;

    for (u32 sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        if (dataCount - offsetCount >= sliceCount) {
            tempAlgParams.allRankSliceSize.emplace_back(sliceSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(sliceCount);
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            offsetCount += sliceCount;
            offsetSize = offsetCount * dataTypeSize_;
        } else {
            u64 curSliceCount = dataCount - offsetCount;
            u64 curSliceSize = curSliceCount * dataTypeSize_;
            tempAlgParams.allRankSliceSize.emplace_back(curSliceSize);
            tempAlgParams.allRankProcessedDataCount.emplace_back(curSliceCount);
            tempAlgParams.allRankDispls.emplace_back(offsetSize);
            offsetCount = dataCount;
            offsetSize = offsetCount * dataTypeSize_;
        }
    }

    for (u32 i = 0; i < tempAlgParams.allRankSliceSize.size(); ++i) {
        HCCL_DEBUG(
            "[InsV2AllReduceSequenceExecutor] SliceInfo: offset[%llu] size[%llu] count[%llu]",
            tempAlgParams.allRankDispls.at(i), tempAlgParams.allRankSliceSize.at(i),
            tempAlgParams.allRankProcessedDataCount.at(i));
    }

    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_ALLREDUCE, DpuAllReduceSequenceMeshMesh, InsV2AllReduceSequenceExecutor, TopoMatchTwoLevel,
    InsTempReduceScatterMesh1DIntra, InsTempReduceScatterMesh1dDpuInter, InsTempAllGatherNhrDpuInter,
    InsTempAllGatherMesh1dIntra);
REGISTER_ALG_ATTRS(
    DpuAllReduceSequenceMeshMesh,
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
    topo.isSupportLevel0PcieMix = true; topo.isHostDpuOnly = true; topo.isSupportLevel1Nhr = true;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS || topo->deviceNumPerModule == 1 || topo->level0PcieMix) {
            return true;
        }
        // UBX非对称
        return topo->Level1Nhr;
    };
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !CanUseDpuInlineReduce(opParam, topo);
    };);

REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_ALLREDUCE, DpuAllReduceSequenceMeshNHRNDA, InsV2AllReduceSequenceExecutor, TopoMatchTwoLevel,
    InsTempReduceScatterMesh1DIntra, InsTempReduceScatterMesh1dNdaInter, InsTempAllGatherNhrNdaInter,
    InsTempAllGatherMesh1dIntra);
REGISTER_ALG_ATTRS(
    DpuAllReduceSequenceMeshNHRNDA, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
    topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.isHostDpuOnly = true;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !topo->level0PcieMix;
    });
REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_ALLREDUCE, DpuAllReduceSequenceMeshNHR, InsV2AllReduceSequenceExecutor, TopoMatchTwoLevel,
    InsTempReduceScatterMesh1DIntra, InsTempReduceScatterNhrDpu, InsTempAllGatherNhrDpuInter,
    InsTempAllGatherMesh1dIntra);
REGISTER_ALG_ATTRS(
    DpuAllReduceSequenceMeshNHR,
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
    topo.isSupportLevel0PcieMix = true; topo.isHostDpuOnly = true; topo.isSupportLevel1Nhr = true;
    topo.supportDevTypes = {HcclDevType::DEV_TYPE_950, HcclDevType::DEV_TYPE_960};
    topo.topoPrecondCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (!CheckDpuInlineReduceTopo(topo)) {
            return false;
        }
        if (topo->level0Topo != Level0Shape::MESH_1D_CLOS || topo->deviceNumPerModule == 1 || topo->level0PcieMix) {
            return true;
        }
        return topo->Level1Nhr;
    };
    op.isSupportProd = false; op.unsupportedDataTypes = UNSUPPORTED_64BIT;);

// 二级 HostDPU RHD：L0 MeshConcur，L1 RS/AG 使用 DPU 模板；仅显式配置时参与选路。
// 启用 Level1Nhr，避免 level0 退化为单 rank 时被选路硬过滤。
#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXECUTOR_BY_FOUR_TEMPS(
    HcclCMDType::HCCL_CMD_ALLREDUCE, DpuAllReduceSequenceMeshConcurRHD, InsV2AllReduceSequenceExecutor,
    TopoMatchTwoLevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterRhdDpuInter, InsTempAllGatherRHDDPU,
    InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(
    DpuAllReduceSequenceMeshConcurRHD, topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D; topo.isHostDpuOnly = true; topo.isSupportLevel0PcieMix = true;
    topo.isSupportLevel1Nhr = true; op.isSupportProd = false;
    op.unsupportedDataTypes
    = {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP64};
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    };);
#endif // CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
} // namespace ops_hccl
