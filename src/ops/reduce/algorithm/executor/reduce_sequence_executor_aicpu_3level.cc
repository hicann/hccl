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
#include "executor_common_ops.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"
#include "template_utils.h"
#include "alg_data_trans_wrapper.h"
#include <algorithm>
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_mesh_1D.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ins_temp_reduce_scatter_aicpu_reduce_nhr_pcie.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_rhd.h"
#include "ins_temp_all_gather_rhd.h"
#include "ins_temp_reduce_scatter_rhd_dpu_inter.h"
#include "ins_temp_all_gather_rhd_dpu.h"
#include "reduce_sequence_rhd_tail.h"
#include "alg_attrs_registry.h"

namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 3;
constexpr u32 SEQUENCE_EXECUTOR_MIN_LEVEL_NUM = 2;
constexpr u32 LEVEL2_IDX = 2;

/* 三级 L2 模板 DPU 卸载变体判定（编译期）：
 * RSL2=RhdDpuInter / AGL2=AG RHDDPU 时，RSL2 scratch 为 acc+recv 双区（2×Σunit），须与
 * rsResult 输入/输出区物理隔离——若沿用三同基（in/out/hccl 共用本 rank L1 子份额基址），
 * recv 区会溢出覆盖相邻服务器的 rsResult 窗口，L2 末段 rx 数据滞留被 AGL1 汇聚读走
 * （l1dev 实测 409：72 源仅合并 24）。DPU 变体下 scratch 迁 meshComm 独立区
 * （meshCommBuffOffset_，OrchestrateLoop 已预算），AGL2 保持 CCL 别名网格（rsResult 同位），
 * 装载 ISS=0 读 DpuInter 平铺终写。AICPU/device 变体维持 0913d1eb 既定三同基布局零变化。 */
template <typename T>
struct Reduce3LIsDpuOffloadTemplate : std::false_type {};
template <>
struct Reduce3LIsDpuOffloadTemplate<InsTempReduceScatterRhdDpuInter> : std::true_type {};
template <>
struct Reduce3LIsDpuOffloadTemplate<InsTempAllGatherRHDDPU> : std::true_type {};

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
std::vector<CostModelParam> ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[ReduceSequenceExecutorAicpu3Level][CalcCostCoeff]",
            TopoProbeScene::PROBE_CALC_COST_COEFF)) {
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    // TwoLevel 匹配只填 2 层 physicalIdx, 层数不足时高层复用 [0][0] 兜底, 避免 vector 越界
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    u32 physIdxLevel0 = static_cast<u32>(physIdx[0][0]);
    u32 physIdxLevel1 = (physIdx.size() > 1) ? static_cast<u32>(physIdx[1][0]) : physIdxLevel0;
    u32 physIdxLevel2 = (physIdx.size() > 2) ? static_cast<u32>(physIdx[2][0]) : physIdxLevel0;
    CommTopo netTypeLevel1 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel1);
    CommTopo netTypeLevel2 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel2);
    std::vector<u32> portNumLevel0 = GetPhysicalLevelPortNums(topoInfo, physIdxLevel0);
    std::vector<u32> portNumLevel1 = GetPhysicalLevelPortNums(topoInfo, physIdxLevel1);
    std::vector<u32> portNumLevel2 = GetPhysicalLevelPortNums(topoInfo, physIdxLevel2);
    CommTopo netTypeLevel0 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel0);
    if (portNumLevel0.empty() || portNumLevel1.empty() || portNumLevel2.empty()) {
        HCCL_WARNING("[ReduceSequenceExecutorAicpu3Level][CalcCostCoeff] portNum is empty");
        return {};
    }
    HCCL_INFO(
        "[ReduceSequenceExecutorAicpu3Level][CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, "
        "rankSizeLevel2=%d, "
        "portNumLevel0=%d, portNumLevel1=%d, portNumLevel2=%d, netTypeLevel0=%d, netTypeLevel1=%d, netTypeLevel2=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0, portNumLevel1, portNumLevel2,
        static_cast<int>(netTypeLevel0), static_cast<int>(netTypeLevel1), static_cast<int>(netTypeLevel2));
    std::vector<CostModelParam> params
        = [rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0, portNumLevel1, portNumLevel2,
           netTypeLevel0, netTypeLevel1, netTypeLevel2, isPod, algName] {
              std::vector<CostModelParam> v;
              auto p0 = AlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f * rankSizeLevel1 * rankSizeLevel2 / rankSize, netTypeLevel0, BufferType::INPUT,
                  BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel0, isPod});
              v.insert(v.end(), p0.begin(), p0.end());
              auto p1 = AlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel1, 1.0f * rankSizeLevel2 / rankSize, netTypeLevel1, BufferType::INPUT,
                  BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel1, isPod});
              v.insert(v.end(), p1.begin(), p1.end());
              auto p2 = AlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel2, 1.0f / rankSize, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel2, isPod});
              v.insert(v.end(), p2.begin(), p2.end());
              auto p3 = AlgTemplate3::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel2, 1.0f / rankSize, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel2, isPod});
              v.insert(v.end(), p3.begin(), p3.end());
              auto p4 = AlgTemplate4::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel1, 1.0f * rankSizeLevel2 / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, portNumLevel1, isPod});
              v.insert(v.end(), p4.begin(), p4.end());
              auto p5 = AlgTemplate5::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f * rankSizeLevel1 * rankSizeLevel2 / rankSize, netTypeLevel0,
                  BufferType::HCCL_BUFFER, BufferType::OUTPUT, BufferType::HCCL_BUFFER, portNumLevel0, isPod});
              v.insert(v.end(), p5.begin(), p5.end());
              // 任一段代价为空则整体返回空，避免缺段条目低估成本
              if (p0.empty() || p1.empty() || p2.empty() || p3.empty() || p4.empty() || p5.empty()) {
                  HCCL_WARNING(
                      "[ReduceSequenceExecutorAicpu3Level][CalcCostCoeff] some template cost coeff is empty, skip "
                      "this algo in costmodel, algName[%s], p0[%zu] p1[%zu] p2[%zu] p3[%zu] p4[%zu] p5[%zu].",
                      algName != nullptr ? algName : "(null)", p0.size(), p1.size(), p2.size(), p3.size(), p4.size(),
                      p5.size());
                  return std::vector<CostModelParam>{};
              }
              return v;
          }();
    return params;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
AlgNetMeta ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4,
    AlgTemplate5>::GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName)
    const
{
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[ReduceSequenceExecutorAicpu3Level][GetAlgNetMeta]",
            TopoProbeScene::PROBE_GET_ALG_NET_META)) {
        return {};
    }
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    u32 rankSize = topoInfo->userRankSize;
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    u32 physIdxLevel0 = static_cast<u32>(physIdx[0][0]);
    u32 physIdxLevel1 = (physIdx.size() > 1) ? static_cast<u32>(physIdx[1][0]) : physIdxLevel0;
    u32 physIdxLevel2 = (physIdx.size() > 2) ? static_cast<u32>(physIdx[2][0]) : physIdxLevel0;
    CommTopo netTypeLevel1 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel1);
    CommTopo netTypeLevel2 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel2);
    CommTopo netTypeLevel0 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel0);
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel2);
    meta.netTypes.push_back(netTypeLevel2);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel0);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1, 1, 1, 1, 1};
    meta.dataRatios
        = {1.0f * rankSizeLevel1 * rankSizeLevel2 / rankSize,
           1.0f * rankSizeLevel2 / rankSize,
           1.0f / rankSize,
           1.0f / rankSize,
           1.0f * rankSizeLevel2 / rankSize,
           1.0f * rankSizeLevel1 * rankSizeLevel2 / rankSize};
    meta.rankSizes = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, rankSizeLevel2, rankSizeLevel1, rankSizeLevel0};
    return meta;
}

template <
    typename AlgTopoMatch, typename AlgTemplate0, typename AlgTemplate1, typename AlgTemplate2, typename AlgTemplate3,
    typename AlgTemplate4, typename AlgTemplate5>
HcclResult ReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, AlgTemplate0, AlgTemplate1, AlgTemplate2, AlgTemplate3, AlgTemplate4, AlgTemplate5>::
    InitCommInfo(
        const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    InitRankAndDataInfo(param, topoInfo);

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
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
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
            "[ReduceSequenceExecutorAicpu3Level] algHierarchyInfo size[%zu] should be 2 or 3",
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
    rankSize_ = resCtx.topoInfo.userRankSize;
    myRank_ = resCtx.topoInfo.userRank;

    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;

    CHK_PRT_RET(
        algHierarchyInfo_.infos.size() < SEQUENCE_EXECUTOR_MIN_LEVEL_NUM
            || algHierarchyInfo_.infos.size() > SEQUENCE_EXECUTOR_LEVEL_NUM,
        HCCL_ERROR(
            "[ReduceSequenceExecutorAicpu3Level] algHierarchyInfo size[%zu] should be 2 or 3",
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
    rankIdxLevel2_ = myRank_ / (algHierarchyInfo_.infos[0][0].size() * algHierarchyInfo_.infos[1][0].size());
    isRootRank_ = (myRank_ == param.root);

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
    SetTemplateBuffInfo(
        tempAlgParamsRSL0, BufferType::INPUT, BufferType::HCCL_BUFFER, param.inputPtr, resCtx.cclMem.addr,
        resCtx.cclMem);

    SetTemplateBuffInfo(
        tempAlgParamsRSL1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);

    SetTemplateBuffInfo(
        tempAlgParamsRSL2, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);

    SetTemplateBuffInfo(
        tempAlgParamsAGL2, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);

    SetTemplateBuffInfo(
        tempAlgParamsAGL1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
        resCtx.cclMem);

    if (isRootRank_) {
        SetTemplateBuffInfo(
            tempAlgParamsAGL0, BufferType::HCCL_BUFFER, BufferType::OUTPUT, resCtx.cclMem.addr, param.outputPtr,
            resCtx.cclMem);
    } else {
        SetTemplateBuffInfo(
            tempAlgParamsAGL0, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER, resCtx.cclMem.addr, resCtx.cclMem.addr,
            resCtx.cclMem);
    }
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
    if constexpr (Reduce3LIsDpuOffloadTemplate<AlgTemplate2>::value) {
        // DPU 卸载变体（经 L2=RhdDpuInter 判定；L1 模板 device RHD 两种变体同名不可判）：
        // RSL1（device RHD）scratch 同样隔离到 meshComm 区
        // （hcclBuffBaseOff = meshCommBuffOffset_）——其 acc+recv 双区与 rsResult 输入区同基时
        // recv 溢出会覆盖相邻区域，破坏后续相位读源。
        // AICPU/device 三级维持三同基（0913d1eb 绿基线）不变。
        tempAlgParamsRSL1.buffInfo.hcclBuffBaseOff = meshCommBuffOffset_;
    } else {
        tempAlgParamsRSL1.buffInfo.hcclBuffBaseOff = 0;
    }

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
    if constexpr (Reduce3LIsDpuOffloadTemplate<AlgTemplate2>::value) {
        // DPU 卸载变体（RhdDpuInter）：scratch（acc+recv 双区 2×Σunit）迁 meshComm 独立区，
        // 与 rsResult 输入/输出区物理隔离
        // （hcclBuffBaseOff = meshCommBuffOffset_）。三同基布局下 recv 区溢出覆盖相邻服务器
        // rsResult 窗口，末段 rx 滞留被 AGL1 汇聚读走（409：72 源仅 24）。
        tempAlgParamsRSL2.buffInfo.hcclBuffBaseOff = meshCommBuffOffset_;
        // 终写槽位预烘焙：DpuInter 终写为平铺式（OUT[base] 无 alg-rank 项，模板契约不可改），
        // 但 AGL2 网格交换要求每个 rank 的终态 unit 落在**自己的 owner 槽** [base+σ(myIdx)·SS2]。
        // 平铺时三个 pod 成员的 flat 写同一 [base] 互相覆盖；AGL2 装载(ISS=0)读 [inBase] 也
        // 只能读到该处的一个 unit；其余槽为 L1 陈旧数据。AGL2 的 MAIN 全段交换会把 block 侧
        // 陈旧槽(=本 pod 的 24 源 L1 结果)覆盖 rep 侧就位的 72 源终值——409 的 24=pod2
        // 精确来源。预烘焙 σ(myIdx)·sliceSize 进 outBase/inBase（AGL2 同式），flat 终写与
        // AGL2 装载各自落到本 rank 自己的 owner 槽，MAIN 交换的每个槽在两侧语义一致。
        tempAlgParamsRSL2.buffInfo.outBuffBaseOff
            = rankIdxLevel1_ * sliceSizeRSL1
              + static_cast<u64>(
                    RhdUnitStartOfRank(static_cast<u32>(rankSizeLevel2_), static_cast<u32>(rankIdxLevel2_)))
                    * tempAlgParamsRSL2.sliceSize;
    } else {
        tempAlgParamsRSL2.buffInfo.hcclBuffBaseOff = rankIdxLevel1_ * sliceSizeRSL1;
    }

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
    // allRank* 变长分片契约（RhdDpuInter L2 模板消费，均匀退化与 stride 契约等价；stride 字段
    // 保留供 device RHD L1 消费，两套并存）：
    // L2 组内逐超节点份额表（尾 rank 取 tailSize），allRankDispls 为本 rank 本地输入布局
    tempAlgParamsRSL2.allRankSliceSize.clear();
    tempAlgParamsRSL2.allRankDispls.clear();
    tempAlgParamsRSL2.allRankProcessedDataCount.clear();
    for (u32 j = 0; j < rankSizeLevel2_; ++j) {
        u64 segSize = (j == rankSizeLevel2_ - 1) ? tempAlgParamsRSL2.tailSize : tempAlgParamsRSL2.sliceSize;
        tempAlgParamsRSL2.allRankSliceSize.push_back(segSize);
        tempAlgParamsRSL2.allRankDispls.push_back(j * tempAlgParamsRSL2.inputSliceStride);
        tempAlgParamsRSL2.allRankProcessedDataCount.push_back(segSize / dataTypeSize_);
    }
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
    // 三同基（in/out/hccl = 本 rank 的 L1 子份额窗口）：AGL2 汇聚网格与 RSL2 scratch 同基，
    // σ 槽位装载/卸载仿射闭环（对齐 AllReduce 三级 AGL2 同区同基写法）
    tempAlgParamsAGL2.buffInfo.outBuffBaseOff = rankIdxLevel1_ * sliceSizeRSL1;
    tempAlgParamsAGL2.buffInfo.hcclBuffBaseOff = rankIdxLevel1_ * sliceSizeRSL1;

    tempAlgParamsAGL2.sliceSize = sliceSizeRSL2;
    tempAlgParamsAGL2.tailSize = tailSizeRSL2;

    if constexpr (Reduce3LIsDpuOffloadTemplate<AlgTemplate3>::value) {
        // DPU 卸载变体（AG RHDDPU）：装载 ISS=0——src=[inBase]（与本 rank RSL2 终写
        // outBase 同式预烘焙的 σ(myIdx)·SS2 槽位）=本 rank 自己的终态 unit，装载写入
        // 网格 owner 槽 [hcclBase+myIdx·OSS]（同槽）——MAIN 全段交换时每槽在收发两侧
        // 语义一致，不再被对端 L1 陈旧槽覆盖（409 根因）。device 变体维持 ISS=SS2 不变。
        tempAlgParamsAGL2.inputSliceStride = 0;
        tempAlgParamsAGL2.buffInfo.inBuffBaseOff
            = rankIdxLevel1_ * sliceSizeRSL1
              + static_cast<u64>(
                    RhdUnitStartOfRank(static_cast<u32>(rankSizeLevel2_), static_cast<u32>(rankIdxLevel2_)))
                    * tempAlgParamsAGL2.sliceSize;
    } else {
        tempAlgParamsAGL2.inputSliceStride = tempAlgParamsAGL2.sliceSize;
    }
    tempAlgParamsAGL2.outputSliceStride = tempAlgParamsAGL2.sliceSize;

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
    tempAlgParamsAGL1.outputSliceStride = tempAlgParamsAGL1.sliceSize;

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
        const u64 loop, const u64 currDataCount, const u64 processedDataCount, const u64 sliceSize, const u64 tailSize,
        TemplateDataParams& tempAlgParamsAGL0) const
{
    tempAlgParamsAGL0.count = currDataCount;
    tempAlgParamsAGL0.buffInfo.inBuffBaseOff = 0;
    tempAlgParamsAGL0.buffInfo.outBuffBaseOff = isRootRank_ ? processedDataCount * dataTypeSize_ : meshCommBuffOffset_;
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
            "than remoteRankToChannelInfo_.size()[%zu]",
            channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    // host-DPU 模板（RHD DPU inter）经共享内存下发 DPURunInfo/等待回执，shmem 双字段须显式
    // 透传（AICPU 侧无 dlsym 查版本能力，漏传退化为 Fence；对齐 AllReduce DPU 三级 executor）
    tempResource.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    tempResource.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
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

    // 非整除数据尾 staging（仅 DPU 卸载实例，
    // 模板零改动）：RhdDpuInter 均匀 σ-unit 契约只覆盖 totalRankAlign 整除数据，非整除余量直接入链
    // 会在 L2 每 pod-slice 丢 8/4 字节尾、AG 尾感知 WQE 携带发送方陈旧 L1 尾字节直达输出
    // （409：72 源仅合 24）。CCL 末端预留输入/输出双 rankAlign tile，预算分区前收缩工作区；
    // 每轮只跑整除对齐前缀，余量单独一轮 staged（补位车道取有效输入复制初始化），仅回拷有效字节。
    u32 totalRankAlign = rankSizeLevel0_ * rankSizeLevel1_;
    if (!skipLevel2_) {
        totalRankAlign *= rankSizeLevel2_;
    }
    AlgResourceCtxSerializable loopResCtx = resCtx;
    [[maybe_unused]] ReduceRhdTail rhdTail{};
    if constexpr (Reduce3LIsDpuOffloadTemplate<AlgTemplate2>::value) {
        CHK_RET(ReduceRhdPrepareTailWorkspace(dataCount_, totalRankAlign, dataTypeSize_, loopResCtx.cclMem, rhdTail));
    }

    TemplateDataParams tempAlgParamsRSL0;
    TemplateDataParams tempAlgParamsRSL1;
    TemplateDataParams tempAlgParamsRSL2;
    TemplateDataParams tempAlgParamsAGL2;
    TemplateDataParams tempAlgParamsAGL1;
    TemplateDataParams tempAlgParamsAGL0;
    GenBaseTempAlgParams(
        param, loopResCtx, tempAlgParamsRSL0, tempAlgParamsRSL1, tempAlgParamsRSL2, tempAlgParamsAGL2,
        tempAlgParamsAGL1, tempAlgParamsAGL0);

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

    // RHD DPU 三级专属：L0 detour 通道按设备侧可用性过滤 + AGL0 线程池裁剪
    // （移植自二级 DPU 修复 a4e24c3b 前身 e6c2f713 的同款问题——hostdpu 组网下
    // remoteRankToChannelInfo_[0] 为 detour CalcRes 请求的 level0+level1 合并集，
    // level1 平面落在 HOST 时 raw 透传会使 z 轴绕路把数据相位发往 host 侧通道，
    // AICPU 设备核消费不了 host 句柄 → 悬空 Record（checker 103/大图 901）。
    // L1=DEVICE 的组网（AICPU 三级验证形态）该过滤为恒等变换，零行为差。
    if constexpr (Reduce3LIsDpuOffloadTemplate<AlgTemplate2>::value) {
        std::map<u32, std::vector<ChannelInfo>> deviceOnlyLevel0;
        for (const auto& rankChannelPair : remoteRankToChannelInfo_[0]) {
            std::vector<ChannelInfo> kept;
            for (const auto& channel : rankChannelPair.second) {
                if (channel.locationType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
                    kept.push_back(channel);
                }
            }
            if (!kept.empty()) {
                deviceOnlyLevel0.emplace(rankChannelPair.first, std::move(kept));
            }
        }
        // 设备侧过滤后为空（全 HOST 链路的仿真形态）时回退原始通道表
        const std::map<u32, std::vector<ChannelInfo>>& level0ChannelsForDetour
            = deviceOnlyLevel0.empty() ? remoteRankToChannelInfo_[0] : deviceOnlyLevel0;
        if (rankSizeLevel0_ > 1) {
            CHK_RET(algTemplateRSL0->SetchannelsPerRank(level0ChannelsForDetour));
            CHK_RET(algTemplateAGL0->SetchannelsPerRank(level0ChannelsForDetour));
        }
    }

    TemplateResource templateResourceRSL0;
    CHK_RET(GenTempResource(loopResCtx, 0, algTemplateRSL0, templateResourceRSL0));
    TemplateResource templateResourceRSL1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(loopResCtx, 1, algTemplateRSL1, templateResourceRSL1));
    }
    TemplateResource templateResourceRSL2;
    if (!skipLevel2_) {
        CHK_RET(GenTempResource(loopResCtx, LEVEL2_IDX, algTemplateRSL2, templateResourceRSL2));
    }
    TemplateResource templateResourceAGL2;
    if (!skipLevel2_) {
        CHK_RET(GenTempResource(loopResCtx, LEVEL2_IDX, algTemplateAGL2, templateResourceAGL2));
    }
    TemplateResource templateResourceAGL1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(loopResCtx, 1, algTemplateAGL1, templateResourceAGL1));
    }
    TemplateResource templateResourceAGL0;
    CHK_RET(GenTempResource(loopResCtx, 0, algTemplateAGL0, templateResourceAGL0));
    // AGL0 线程池裁剪（同二级修复）：AG 侧 detour 的 GetThreadNum 语义（不含主流）与
    // 其自身 CalcRes 请求自洽；共享池须裁剪到模板期望数量，否则 PreSyncInterThreads
    // 的 subThreads(池-1) vs notifyIdx(GetThreadNum-1) 错配、AGL0 相位整体失败
    if constexpr (Reduce3LIsDpuOffloadTemplate<AlgTemplate2>::value) {
        const u64 rhdAgl0ThreadNum = algTemplateAGL0->GetThreadNum();
        if (templateResourceAGL0.threads.size() > rhdAgl0ThreadNum) {
            templateResourceAGL0.threads.resize(static_cast<size_t>(rhdAgl0ThreadNum));
        }
    }

    u64 scratchMultiplier = algTemplateRSL0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 cclBuffSliceNum = scratchMultiplier + 1;
    // 来自local reduce的地址需要按照数据类型对齐，防止精度问题
    // 分区预算须用尾 staging 收缩后的工作区（非整除数据下 CCL 末端 2*slotBytes 已被
    // 预留为 tile：用全量 size 计算会把落盘区推进预留区，padded 尾轮的 staging 输入
    // 会在同轮 RSL0 读走前被 mesh comm 写入覆盖，静默错值——loopResCtx.cclMem.size
    // 已扣除预留额度，整除时两者相等零行为差）
    cclBuffSliceSize_ = loopResCtx.cclMem.size / cclBuffSliceNum / dataTypeSize_ * dataTypeSize_;
    rsResultBuffSize_ = cclBuffSliceSize_;
    meshCommBuffSize_ = scratchMultiplier * cclBuffSliceSize_;
    rsResultBuffOffset_ = 0;
    meshCommBuffOffset_ = rsResultBuffSize_;
    u64 maxCountPerLoop
        = meshCommBuffSize_ / AICPU_ALIGN_SIZE * AICPU_ALIGN_SIZE / dataTypeSize_ / totalRankAlign * totalRankAlign;
    CHK_PRT_RET(
        maxCountPerLoop == 0, HCCL_ERROR("[ReduceSequenceExecutorAicpu3Level] maxCountPerLoop is 0"), HCCL_E_INTERNAL);
    u64 processedDataCount = 0;
    u64 loop = 0;
    while (processedDataCount < dataCount_) {
        u64 remaining = dataCount_ - processedDataCount;
        u64 currDataCount;
        [[maybe_unused]] u64 validDataCount = 0;
        bool paddedTailLoop = false;
        if constexpr (Reduce3LIsDpuOffloadTemplate<AlgTemplate2>::value) {
            // DPU 卸载实例：均匀槽整除契约——每轮只跑 totalRankAlign 整除前缀；末尾余量单独
            // 一轮 staged 进隔离 tile（整链跑在补位后的对齐槽上），仅回拷有效字节
            validDataCount = ReduceRhdSelectLoopCount(remaining, maxCountPerLoop, totalRankAlign);
            currDataCount = std::max(validDataCount, static_cast<u64>(totalRankAlign));
            paddedTailLoop = validDataCount != currDataCount;
            if (paddedTailLoop) {
                CHK_RET(ReduceRhdStageTail(param, rhdTail, processedDataCount, validDataCount, loopResCtx.threads[0]));
            }
        } else if (remaining <= maxCountPerLoop) {
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
        if (paddedTailLoop) {
            // 尾轮：输入重定向到 staging tile（有效+补位车道已就位）
            tempAlgParamsRSL0.buffInfo.inputPtr = rhdTail.storage.addr;
            tempAlgParamsRSL0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
            tempAlgParamsRSL0.buffInfo.inBuffBaseOff = rhdTail.inputOffset;
        }
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
            loop, currDataCount, processedDataCount, tempAlgParamsRSL0.sliceSize, tempAlgParamsRSL0.tailSize,
            tempAlgParamsAGL0);
        if (paddedTailLoop && isRootRank_) {
            // 尾轮：root 终写重定向到输出 tile（Reduce 仅 root 有用户输出），随后仅回拷有效字节
            tempAlgParamsAGL0.buffInfo.outputPtr = rhdTail.storage.addr;
            tempAlgParamsAGL0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
            tempAlgParamsAGL0.buffInfo.outBuffBaseOff = rhdTail.outputOffset;
        }
        CHK_RET(algTemplateAGL0->KernelRun(param, tempAlgParamsAGL0, templateResourceAGL0));

        if (paddedTailLoop) {
            if (isRootRank_) {
                CHK_RET(
                    ReduceRhdCopyTailOutput(param, rhdTail, processedDataCount, validDataCount, loopResCtx.threads[0]));
            }
            processedDataCount += validDataCount;
        } else {
            processedDataCount += currDataCount;
        }
        loop++;
    }
    HCCL_INFO("[ReduceSequenceExecutorAicpu3Level][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, AicpuReduceSequenceMeshConcurNHRNHR, ReduceSequenceExecutorAicpu3Level,
    TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterNHR, InsTempReduceScatterNHR,
    InsTempAllGatherNHR, InsTempAllGatherNHR, InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(AicpuReduceSequenceMeshConcurNHRNHR, topo.minTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; op.unsupportedDataTypes = UNSUPPORTED_64BIT;
                   op.isSupportInplace = false; op.isSupportProd = false);

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, AicpuReduceSequenceMeshConcurNHR, ReduceSequenceExecutorAicpu3Level,
    TopoMatchTwoLevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterNHR, InsTempReduceScatterNHR,
    InsTempAllGatherNHR, InsTempAllGatherNHR, InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(
    AicpuReduceSequenceMeshConcurNHR, topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
    op.unsupportedDataTypes = UNSUPPORTED_64BIT; op.isSupportInplace = false; op.isSupportProd = false;
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        u64 perRankSize = opParam.DataDes.count * DATATYPE_SIZE_TABLE[opParam.DataDes.dataType];
        return perRankSize > 4ULL * 1024 * 1024 * 1024;
    });
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, AicpuReduceSequenceNHRAicpuReduceNHR, ReduceSequenceExecutorAicpu3Level,
    TopoMatchTwoLevel, InsTempReduceScatterMesh1D, InsTempReduceScatterAicpuReduceNHRPcie,
    InsTempReduceScatterAicpuReduceNHRPcie, InsTempAllGatherNHR, InsTempAllGatherNHR, InsTempAllGatherMesh1D);
REGISTER_ALG_ATTRS(
    AicpuReduceSequenceNHRAicpuReduceNHR, topo.minTopoLevelNum = TOPO_LEVEL_NUM_2;
    topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_CLOS | LEVEL0_TOPO_MESH_1D_CLOS;
    topo.isSupportLevel1Nhr = true; topo.isSupportLevel0PcieMix = false; op.isSupportInplace = false;
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        return Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;
    });

// RHD（Recursive Halving-Doubling）系列：L0 复用 MeshConcur z 轴绕路模板，L1/L2 相位模板取
// ReduceScatter/AllGather 的 RHD 模板（非 2 幂支持，五参数 stride 契约与本
// executor 的 GenTempAlgParamsRSL1/RSL2/AGL1/AGL2 一致）。
// RHD 仅经 HCCL_ALGO 显式配置可选：topoCustomCheck 恒 false 使默认选路（软策略）过滤本算法，
// 显式配置 reduce:sequence{level0=meshconcur,level1=rhd[,level2=rhd]} 打 preferred 后软策略
// 跳过该检查仍可选，默认选路零变化。
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, AicpuReduceSequenceMeshConcurRHD, ReduceSequenceExecutorAicpu3Level,
    TopoMatchTwoLevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterRHD, InsTempReduceScatterRHD,
    InsTempAllGatherRHD, InsTempAllGatherRHD, InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(
    AicpuReduceSequenceMeshConcurRHD, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D; topo.isSupportLevel1Nhr = true;
    topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2;
    op.unsupportedDataTypes = UNSUPPORTED_64BIT; op.isSupportInplace = false; op.isSupportProd = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    });

// RHD 三级（Aicpu 引擎）：形态对齐 AicpuReduceSequenceMeshConcurNHRNHR，L1/L2 均为 AICPU RHD 相位模板
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, AicpuReduceSequenceMeshConcurRHDRHD, ReduceSequenceExecutorAicpu3Level,
    TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterRHD, InsTempReduceScatterRHD,
    InsTempAllGatherRHD, InsTempAllGatherRHD, InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(
    AicpuReduceSequenceMeshConcurRHDRHD, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D; topo.isSupportLevel1Nhr = true;
    topo.minTopoLevelNum = TOPO_LEVEL_NUM_3; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
    op.unsupportedDataTypes = UNSUPPORTED_64BIT; op.isSupportInplace = false; op.isSupportProd = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    });

// RHD 三级（Dpu 引擎）：L0 绕路 + L1 框间 AICPU RHD 相位模板 + L2 跨超节点 host DPU RHD inter
// 模板，模板链形态对齐 DpuAllReduceSequenceMeshConcurNHRMesh（RS 下沉链 L0/L1/L2 + Gather
// 回升链 L2/L1/L0）；RS_L2/GA_L2 为 allRank* 变长契约（GenTempAlgParamsRSL2/AGL2 需补填），
// DPU 模板经共享内存下发 DPURunInfo（GenTempResource 需透传 shmem 双字段）。
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, DpuReduceSequenceMeshConcurRHDRHD, ReduceSequenceExecutorAicpu3Level,
    TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterRHD,
    InsTempReduceScatterRhdDpuInter, InsTempAllGatherRHDDPU, InsTempAllGatherRHD, InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(
    DpuReduceSequenceMeshConcurRHDRHD, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D; topo.isHostDpuOnly = true;
    topo.isSupportLevel1Nhr = true; topo.minTopoLevelNum = TOPO_LEVEL_NUM_3; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
    op.unsupportedDataTypes = UNSUPPORTED_64BIT; op.isSupportInplace = false; op.isSupportProd = false;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    };
    // 物理硬前提（对齐兄弟算法的同款 topoPrecondCheck）：
    // L1 框间层须为设备侧 CLOS——本算法 L1 绑 device 版 RHD 模板，L1 为 HOST 链路时其
    // device 通道为空，且实测 host 通道组合产生悬空 Record（checker 103，l1dev/hostdpu
    // 双向实证）。违反前提的组网（如 hostdpu 全 HOST 三层布局）在选择期即被过滤，
    // preferred 配置静默回退自动选路（对齐 DpuReduceScatterSequenceMeshConcurRHDRHD 行为）。
    topo.topoPrecondCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->physicalLevels.size() == TOPO_LEVEL_NUM_3
               && topo->physicalLevels[1].topoType == CommTopo::COMM_TOPO_CLOS
               && topo->physicalLevels[1].locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    };);

// PCIE标卡机型的Reduce分层算法，归约在AICPU上完成，支持fp64/uint64
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_REDUCE, AicpuReduceSequenceMeshNHRAicpuReduce, ReduceSequenceExecutorAicpu3Level,
    TopoMatchTwoLevel, InsTempReduceScatterMesh1D, InsTempReduceScatterAicpuReduceNHRPcie,
    InsTempReduceScatterAicpuReduceNHRPcie, InsTempAllGatherNHR, InsTempAllGatherNHR, InsTempAllGatherMesh1D);
REGISTER_ALG_ATTRS(
    AicpuReduceSequenceMeshNHRAicpuReduce, topo.maxTopoLevelNum = TOPO_LEVEL_NUM_1;
    topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS; topo.isSupportLevel0PcieMix = true;
    topo.isSupportLevel1Nhr = true; op.isSupportProd = true;
    op.isSupportInplace = true; // 输入先暂存到cclMem，最后才写回输出，inplace安全
    op.opCustomCheck = [](const OpParam& opParam, const TopoInfoWithNetLayerDetails* topo) -> bool {
        return Is64BitDataType(opParam.DataDes.dataType) || opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD;
    };
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        if (topo->level0Topo == Level0Shape::MESH_1D_CLOS) {
            return topo->level0PcieMix
                   && !AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH);
        }
        return false;
    });
} // namespace ops_hccl
