/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_sequence_executor_aicpu_3level.h"
#include "executor_common_ops.h"
#include "alg_attrs_registry.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_rhd.h"
#include "ins_temp_all_gather_rhd.h"
#include "ins_v2_all_reduce_sequence_rhd_tail.h"
#include "alg_data_trans_wrapper.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"

namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 3;
constexpr u32 OMNIPIPE_LEVEL2_IDX = 2;

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
std::vector<CostModelParam> InsV2AllReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AllReduceSequenceExecutorAicpu3Level][CalcCostCoeff]",
            TopoProbeScene::PROBE_CALC_COST_COEFF)) {
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    // algo level 对应物理层(MeshConcur 为 2元组)逐层下传, 供跨物理层模板用; 层数不足时高层复用低层
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    CHK_PRT_RET(
        physIdx.size() != TOPO_LEVEL_NUM_3,
        HCCL_WARNING(
            "[InsV2AllReduceSequenceExecutorAicpu3Level][CalcCostCoeff] physicalIdxForAlgoLevels size[%zu] != 3.",
            physIdx.size()),
        {});
    std::vector<std::vector<CommTopo>> phyLevelNetTypes(physIdx.size());
    std::vector<std::vector<std::vector<u32>>> phyLevelPortNums(physIdx.size());
    for (u32 lvl = 0; lvl < physIdx.size(); lvl++) {
        for (PhysicalLevelIndex phyIdx : physIdx[lvl]) {
            phyLevelNetTypes[lvl].push_back(GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(phyIdx)));
            phyLevelPortNums[lvl].push_back(GetPhysicalLevelPortNums(topoInfo, static_cast<u32>(phyIdx)));
        }
    }
    CommTopo netTypeLevel0 = phyLevelNetTypes[0][0];
    CommTopo netTypeLevel1 = phyLevelNetTypes[1][0];
    CommTopo netTypeLevel2 = phyLevelNetTypes[2][0];
    const std::vector<u32>& portNumLevel0 = phyLevelPortNums[0][0];
    const std::vector<u32>& portNumLevel1 = phyLevelPortNums[1][0];
    const std::vector<u32>& portNumLevel2 = phyLevelPortNums[2][0];
    if (portNumLevel0.empty() || portNumLevel1.empty() || portNumLevel2.empty()) {
        HCCL_WARNING("[InsV2AllReduceSequenceExecutorAicpu3Level][CalcCostCoeff] portNum is empty");
        return {};
    }
    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorAicpu3Level][CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, "
        "rankSizeLevel2=%d, "
        "portNumLevel0=%d, portNumLevel1=%d, portNumLevel2=%d, netTypeLevel0=%d, netTypeLevel1=%d, netTypeLevel2=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0, portNumLevel1, portNumLevel2,
        static_cast<int>(netTypeLevel0), static_cast<int>(netTypeLevel1), static_cast<int>(netTypeLevel2));
    std::vector<CostModelParam> params
        = [rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0, portNumLevel1, portNumLevel2,
           netTypeLevel0, netTypeLevel1, netTypeLevel2, isPod, physIdx, phyLevelNetTypes, phyLevelPortNums] {
              std::vector<CostModelParam> v;
              auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f / rankSize, netTypeLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel0, isPod, nullptr, nullptr, nullptr, 1u, physIdx[0],
                  phyLevelNetTypes[0], phyLevelPortNums[0]});
              v.insert(v.end(), p0.begin(), p0.end());
              auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel1, 1.0f / rankSize, netTypeLevel1, BufferType::INPUT, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel1, isPod, nullptr, nullptr, nullptr, 1u, physIdx[1],
                  phyLevelNetTypes[1], phyLevelPortNums[1]});
              v.insert(v.end(), p1.begin(), p1.end());
              auto p2 = InsAlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel2, 1.0f / rankSize, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel2, isPod, nullptr, nullptr, nullptr, 1u, physIdx[2],
                  phyLevelNetTypes[2], phyLevelPortNums[2]});
              v.insert(v.end(), p2.begin(), p2.end());
              auto p3 = InsAlgTemplate3::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f / rankSize, netTypeLevel0, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel0, isPod, nullptr, nullptr, nullptr, 1u, physIdx[0],
                  phyLevelNetTypes[0], phyLevelPortNums[0]});
              v.insert(v.end(), p3.begin(), p3.end());
              auto p4 = InsAlgTemplate4::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel1, 1.0f / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel1, isPod, nullptr, nullptr, nullptr, 1u, physIdx[1],
                  phyLevelNetTypes[1], phyLevelPortNums[1]});
              v.insert(v.end(), p4.begin(), p4.end());
              auto p5 = InsAlgTemplate5::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel2, 1.0f / rankSize, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::OUTPUT,
                  BufferType::HCCL_BUFFER, portNumLevel2, isPod, nullptr, nullptr, nullptr, 1u, physIdx[2],
                  phyLevelNetTypes[2], phyLevelPortNums[2]});
              v.insert(v.end(), p5.begin(), p5.end());
              return v;
          }();
    return params;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
AlgNetMeta InsV2AllReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AllReduceSequenceExecutorAicpu3Level][GetAlgNetMeta]",
            TopoProbeScene::PROBE_GET_ALG_NET_META)) {
        return {};
    }
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    u32 rankSize = topoInfo->userRankSize;
    CHK_PRT_RET(
        algHierarchyInfo.physicalIdxForAlgoLevels.empty(),
        HCCL_WARNING("[InsV2AllReduceSequenceExecutorAicpu3Level][GetAlgNetMeta] physicalIdxForAlgoLevels is empty."),
        {});
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    u32 physIdxLevel0 = static_cast<u32>(physIdx[0][0]);
    u32 physIdxLevel1 = (physIdx.size() > 1) ? static_cast<u32>(physIdx[1][0]) : physIdxLevel0;
    u32 physIdxLevel2 = (physIdx.size() > 2) ? static_cast<u32>(physIdx[2][0]) : physIdxLevel0;
    CommTopo netTypeLevel0 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel0);
    CommTopo netTypeLevel1 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel1);
    CommTopo netTypeLevel2 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel2);
    AlgNetMeta meta;
    meta.netTypes.push_back(netTypeLevel0);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel2);
    meta.netTypes.push_back(netTypeLevel0);
    meta.netTypes.push_back(netTypeLevel1);
    meta.netTypes.push_back(netTypeLevel2);
    meta.intraGroupMode = CostAggMode::SUM;
    meta.groupSizes = {1, 1, 1, 1, 1, 1};
    meta.dataRatios = {1.0f / rankSizeLevel0, 1.0f / rankSize, 1.0f / rankSize,
                       1.0f / rankSize,       1.0f / rankSize, 1.0f / rankSizeLevel0};
    meta.rankSizes = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, rankSizeLevel2, rankSizeLevel1, rankSizeLevel0};
    return meta;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
InsV2AllReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::InsV2AllReduceSequenceExecutorAicpu3Level()
{}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2AllReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    CalcAlgHierarchyInfoV2(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2AllReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    SequenceInitCommInfo(param, topoInfo, algHierarchyInfo, "InsV2AllReduceSequenceExecutorAicpu3Level");
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutorAicpu3Level] algHierarchyInfo size should be %u",
            SEQUENCE_EXECUTOR_LEVEL_NUM);
        return HCCL_E_INTERNAL;
    }
    if (algHierarchyInfo.infos[0].empty() || algHierarchyInfo.infos[1].empty() || algHierarchyInfo.infos[2].empty()
        || algHierarchyInfo.infos[0][0].empty() || algHierarchyInfo.infos[1][0].empty()
        || algHierarchyInfo.infos[2][0].empty()) {
        HCCL_ERROR("[%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo.infos[2][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);

    std::shared_ptr<InsAlgTemplate0> rsL0TempAlg
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo.infos[0]);
    std::shared_ptr<InsAlgTemplate1> rsL1TempAlg;
    if (!skipLevel1_) {
        rsL1TempAlg = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo.infos[1]);
    }
    std::shared_ptr<InsAlgTemplate2> rsL2TempAlg
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo.infos[2]);
    std::shared_ptr<InsAlgTemplate3> agL2TempAlg
        = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo.infos[2]);
    std::shared_ptr<InsAlgTemplate4> agL1TempAlg;
    if (!skipLevel1_) {
        agL1TempAlg = std::make_shared<InsAlgTemplate4>(param, myRank_, algHierarchyInfo.infos[1]);
    }
    std::shared_ptr<InsAlgTemplate5> agL0TempAlg
        = std::make_shared<InsAlgTemplate5>(param, myRank_, algHierarchyInfo.infos[0]);

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
    CHK_RET(rsL2TempAlg->CalcRes(comm, param, topoInfo, resReqRSL2));
    CHK_RET(agL2TempAlg->CalcRes(comm, param, topoInfo, resReqAGL2));
    CHK_RET(agL0TempAlg->CalcRes(comm, param, topoInfo, resReqAGL0));

    std::vector<AlgResourceRequest> activeReqs = {resReqRSL0, resReqRSL2, resReqAGL2, resReqAGL0};
    if (!skipLevel1_) {
        activeReqs.push_back(resReqRSL1);
        activeReqs.push_back(resReqAGL1);
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

    resourceRequest.channels.resize(SEQUENCE_EXECUTOR_LEVEL_NUM);
    if (resReqRSL0.channels.empty()) {
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorAicpu3Level] level0 channels empty");
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[0] = resReqRSL0.channels[0];
    if (!skipLevel1_) {
        if (resReqRSL1.channels.empty()) {
            HCCL_ERROR("[InsV2AllReduceSequenceExecutorAicpu3Level] level1 channels empty");
            return HCCL_E_INTERNAL;
        }
        resourceRequest.channels[1] = resReqRSL1.channels[0];
    }
    if (resReqRSL2.channels.empty()) {
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorAicpu3Level] level2 channels empty");
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[2] = resReqRSL2.channels[0];
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2AllReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu3Level][Orchestrate] Orchestrate Start");
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;

    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataCount_ = param.DataDes.count;
    dataSize_ = dataCount_ * dataTypeSize_;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    threads_ = resCtx.threads;
    supportSymmetricMemory_ = param.supportSymmetricMemory;
    if (supportSymmetricMemory_) {
        inputOffset_ = param.inputOffset;
        outputOffset_ = param.outputOffset;
        inputSymWindow_ = param.inputSymWindow;
        outputSymWindow_ = param.outputSymWindow;
    }

    if (algHierarchyInfo_.infos.size() < TOPO_LEVEL_NUM_3 || algHierarchyInfo_.infos[0][0].empty()
        || algHierarchyInfo_.infos[1][0].empty() || algHierarchyInfo_.infos[2][0].empty()) {
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorAicpu3Level][%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    if (skipLevel1_) {
        HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu3Level][Orchestrate] level1 rankSize is 1, skip level1");
    }
    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = (myRank_ / algHierarchyInfo_.infos[0][0].size()) % algHierarchyInfo_.infos[1][0].size();
    rankIdxLevel2_ = myRank_ / (algHierarchyInfo_.infos[0][0].size() * algHierarchyInfo_.infos[1][0].size());

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutorAicpu3Level][Orchestrate]errNo[0x%016llx] "
            "AllReduce executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2AllReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempAlgParamsRSL2(
        const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL1, const u64 tailSizeRSL1,
        const u64 symMemBaseOffRSL1, TemplateDataParams& tempAlgParamsRSL2) const
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
    u64 symMemBaseOff
        = supportSymmetricMemory_ ? symMemBaseOffRSL1 + rankIdxLevel1_ * sliceSizeRSL1 : rankIdxLevel1_ * sliceSizeRSL1;
    tempAlgParamsRSL2.buffInfo.inBuffBaseOff = symMemBaseOff;
    tempAlgParamsRSL2.buffInfo.outBuffBaseOff = symMemBaseOff;
    tempAlgParamsRSL2.buffInfo.hcclBuffBaseOff = supportSymmetricMemory_ ? 0 : rankIdxLevel1_ * sliceSizeRSL1;

    tempAlgParamsRSL2.inputSliceStride = tempAlgParamsRSL2.sliceSize;
    tempAlgParamsRSL2.outputSliceStride = tempAlgParamsRSL2.sliceSize;

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorAicpu3Level] loop [%llu] RSL2.inputSliceStride [%llu], "
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
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2AllReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempAlgParamsAGL2(
        const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL2, const u64 tailSizeRSL2,
        const u64 sliceSizeRSL1, const u64 symMemBaseOffRSL1, TemplateDataParams& tempAlgParamsAGL2) const
{
    tempAlgParamsAGL2.count = currDataCount;
    u64 inBaseOff
        = supportSymmetricMemory_ ? symMemBaseOffRSL1 + rankIdxLevel1_ * sliceSizeRSL1 : rankIdxLevel1_ * sliceSizeRSL1;
    u64 outBaseOff = supportSymmetricMemory_ ? symMemBaseOffRSL1 + rankIdxLevel1_ * sliceSizeRSL1 : 0;
    tempAlgParamsAGL2.buffInfo.inBuffBaseOff = inBaseOff;
    tempAlgParamsAGL2.buffInfo.outBuffBaseOff = outBaseOff;
    tempAlgParamsAGL2.buffInfo.hcclBuffBaseOff = supportSymmetricMemory_ ? 0 : rankIdxLevel1_ * sliceSizeRSL1;

    tempAlgParamsAGL2.sliceSize = sliceSizeRSL2;
    tempAlgParamsAGL2.tailSize = tailSizeRSL2;

    tempAlgParamsAGL2.inputSliceStride = tempAlgParamsAGL2.sliceSize;
    tempAlgParamsAGL2.outputSliceStride = tempAlgParamsAGL2.sliceSize;

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorAicpu3Level] loop [%llu] AGL2.inputSliceStride [%llu], "
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
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
template <typename InsAlgTemplate>
HcclResult InsV2AllReduceSequenceExecutorAicpu3Level<
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
            "[InsV2AllReduceSequenceExecutorAicpu3Level][GenTempResource] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%zu]",
            channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2AllReduceSequenceExecutorAicpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu3Level][OrchestrateLoop] Start");
    const u32 totalRankAlign = rankSizeLevel0_ * rankSizeLevel1_ * rankSizeLevel2_;
    // RHD 尾块隔离：先收缩 workspace 并预留两块 rank-align tile，再做后续 scratch 分区
    // 与 maxCountPerLoop 计算；收缩后的 cclMem 同时进入模板参数（host/kernel 使用同一尺寸）。
    AlgResourceCtxSerializable loopResCtx = resCtx;
    ArSeqRhdTail rhdTail{};
    if constexpr (IsRhdTemplate<InsAlgTemplate1>::value) {
        CHK_RET(ArSeqRhdPrepareTailWorkspace(dataCount_, totalRankAlign, dataTypeSize_, loopResCtx.cclMem, rhdTail));
    }

    TemplateDataParams tempAlgParamsRSL0;
    TemplateDataParams tempAlgParamsRSL1;
    TemplateDataParams tempAlgParamsRSL2;
    TemplateDataParams tempAlgParamsAGL2;
    TemplateDataParams tempAlgParamsAGL1;
    TemplateDataParams tempAlgParamsAGL0;
    ArSeq3LevelGenBaseBuffInfo(
        param, loopResCtx, tempAlgParamsRSL0, tempAlgParamsRSL1, tempAlgParamsRSL2, tempAlgParamsAGL2,
        tempAlgParamsAGL1, tempAlgParamsAGL0);

    std::shared_ptr<InsAlgTemplate0> algTemplateRSL0
        = std::make_shared<InsAlgTemplate0>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateRSL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    std::shared_ptr<InsAlgTemplate1> algTemplateRSL1;
    if (!skipLevel1_) {
        algTemplateRSL1 = std::make_shared<InsAlgTemplate1>(param, myRank_, algHierarchyInfo_.infos[1]);
        CHK_RET(algTemplateRSL1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    std::shared_ptr<InsAlgTemplate2> algTemplateRSL2
        = std::make_shared<InsAlgTemplate2>(param, myRank_, algHierarchyInfo_.infos[2]);
    CHK_RET(algTemplateRSL2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));

    std::shared_ptr<InsAlgTemplate3> algTemplateAGL2
        = std::make_shared<InsAlgTemplate3>(param, myRank_, algHierarchyInfo_.infos[2]);
    CHK_RET(algTemplateAGL2->SetchannelsPerRank(remoteRankToChannelInfo_[2]));

    std::shared_ptr<InsAlgTemplate4> algTemplateAGL1;
    if (!skipLevel1_) {
        algTemplateAGL1 = std::make_shared<InsAlgTemplate4>(param, myRank_, algHierarchyInfo_.infos[1]);
        CHK_RET(algTemplateAGL1->SetchannelsPerRank(remoteRankToChannelInfo_[1]));
    }

    std::shared_ptr<InsAlgTemplate5> algTemplateAGL0
        = std::make_shared<InsAlgTemplate5>(param, myRank_, algHierarchyInfo_.infos[0]);
    CHK_RET(algTemplateAGL0->SetchannelsPerRank(remoteRankToChannelInfo_[0]));

    TemplateResource templateResourceRSL0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateRSL0, templateResourceRSL0));
    TemplateResource templateResourceRSL1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateRSL1, templateResourceRSL1));
    }
    TemplateResource templateResourceRSL2;
    CHK_RET(GenTempResource(resCtx, OMNIPIPE_LEVEL2_IDX, algTemplateRSL2, templateResourceRSL2));
    TemplateResource templateResourceAGL2;
    CHK_RET(GenTempResource(resCtx, OMNIPIPE_LEVEL2_IDX, algTemplateAGL2, templateResourceAGL2));
    TemplateResource templateResourceAGL1;
    if (!skipLevel1_) {
        CHK_RET(GenTempResource(resCtx, 1, algTemplateAGL1, templateResourceAGL1));
    }
    TemplateResource templateResourceAGL0;
    CHK_RET(GenTempResource(resCtx, 0, algTemplateAGL0, templateResourceAGL0));

    u64 scratchMultiplier = algTemplateRSL0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 cclBuffSliceNum = scratchMultiplier + 1;
    // 限制来自local reduce的地址需要按照数据类型宽度对齐
    cclBuffSliceSize_ = loopResCtx.cclMem.size / cclBuffSliceNum / dataTypeSize_ * dataTypeSize_;
    rsResultBuffSize_ = cclBuffSliceSize_;
    meshCommBuffSize_ = scratchMultiplier * cclBuffSliceSize_;
    rsResultBuffOffset_ = 0;
    meshCommBuffOffset_ = rsResultBuffSize_;
    u64 maxCountPerLoop = meshCommBuffSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_
                          / totalRankAlign * totalRankAlign;
    u64 processedDataCount = 0;
    u64 loop = 0;
    if constexpr (IsRhdTemplate<InsAlgTemplate1>::value) {
        // 尾块预留收缩后 workspace 可能变小；归零时无法推进单轮，直接报错而非死循环。
        CHK_PRT_RET(
            maxCountPerLoop == 0,
            HCCL_ERROR(
                "[InsV2AllReduceSequenceExecutorAicpu3Level] maxCountPerLoop is 0, cclBuffSize[%llu], "
                "scratchMultiplier[%llu], dataTypeSize[%llu].",
                static_cast<u64>(loopResCtx.cclMem.size), scratchMultiplier, static_cast<u64>(dataTypeSize_)),
            HcclResult::HCCL_E_INTERNAL);
        // L1/L2 RS/AG 使用均匀分片，scratch 与 rsResult 输入区隔离。
        // RSL1 保留 POST，为下层提供输入；RSL2 跳过 POST，AGL2 经代表桥接后跳过 PRE。
        // AGL1 保留 PRE，重新收集经 L2 汇聚的份额。
        // 每轮只跑 totalRankAlign 对齐前缀；末尾余量单独一轮 staged 到隔离 tile，对齐 DPU 方案。
        // AICPU 路径不支持对称内存。
        CHK_PRT_RET(
            param.supportSymmetricMemory,
            HCCL_ERROR(
                "[%s] algName[%s] RHD sequence not support symmetric memory, fallback by reselecting HCCL_ALGO.",
                __func__, param.algName),
            HcclResult::HCCL_E_NOT_SUPPORT);
        while (processedDataCount < dataCount_) {
            const ArSeqRhdLoopPlan plan
                = ArSeqRhdPlanLoop(dataCount_ - processedDataCount, maxCountPerLoop, totalRankAlign);
            if (plan.paddedTail) {
                CHK_RET(ArSeqRhdStageTail(param, rhdTail, processedDataCount, plan.validCount, threads_[0]));
            }

            // RSL0：框内 MeshConcur ReduceScatter。
            ArSeq3LevelGenRSL0Params(
                MakeArSeq3LevelCtx(), loop, plan.currDataCount, processedDataCount, tempAlgParamsRSL0,
                "InsV2AllReduceSequenceExecutorAicpu3Level");
            if (plan.paddedTail) {
                tempAlgParamsRSL0.buffInfo.inputPtr = rhdTail.storage.addr;
                tempAlgParamsRSL0.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
                tempAlgParamsRSL0.buffInfo.inBuffBaseOff = rhdTail.inputOffset;
            }
            CHK_RET(algTemplateRSL0->KernelRun(param, tempAlgParamsRSL0, templateResourceRSL0));

            // RSL1：保留 POST，为 L2 提供各 rank 的份额。
            u64 chunkCount = plan.currDataCount / rankSizeLevel0_;
            u64 subSliceCount1 = chunkCount / rankSizeLevel1_;
            u64 subSliceSize1 = subSliceCount1 * dataTypeSize_;
            if (!skipLevel1_) {
                tempAlgParamsRSL1.count = chunkCount;
                tempAlgParamsRSL1.sliceSize = subSliceSize1;
                tempAlgParamsRSL1.tailSize = subSliceSize1;
                tempAlgParamsRSL1.inputSliceStride = subSliceSize1;  // IN[base+i·ISS]：chunk 的第 i 子片
                tempAlgParamsRSL1.outputSliceStride = subSliceSize1; // OUT[base+myIdx·OSS]：我的子份额
                tempAlgParamsRSL1.buffInfo.inBuffBaseOff = 0;        // rsResult 区（RSL0 输出位）
                tempAlgParamsRSL1.buffInfo.outBuffBaseOff = 0;
                tempAlgParamsRSL1.buffInfo.hcclBuffBaseOff = meshCommBuffOffset_; // scratch 与输入区隔离
                tempAlgParamsRSL1.repeatNum = 1;
                tempAlgParamsRSL1.inputRepeatStride = 0;
                tempAlgParamsRSL1.outputRepeatStride = 0;
                CHK_RET(algTemplateRSL1->KernelRun(param, tempAlgParamsRSL1, templateResourceRSL1));
            }

            // RSL2：跳过 POST，由代表桥接到 AGL2。
            u64 subSliceCount2 = subSliceCount1 / rankSizeLevel2_;
            u64 subSliceSize2 = subSliceCount2 * dataTypeSize_;
            tempAlgParamsRSL2.count = subSliceCount1;
            tempAlgParamsRSL2.sliceSize = subSliceSize2;
            tempAlgParamsRSL2.tailSize = subSliceSize2;
            tempAlgParamsRSL2.inputSliceStride = subSliceSize2;
            tempAlgParamsRSL2.outputSliceStride = subSliceSize2;
            tempAlgParamsRSL2.buffInfo.inBuffBaseOff = rankIdxLevel1_ * subSliceSize1; // 我的 L1 子份额位
            tempAlgParamsRSL2.buffInfo.outBuffBaseOff = rankIdxLevel1_ * subSliceSize1;
            // RSL2 scratch 位于 rsResult 活跃区之后，所有 rank 使用相同基址。
            tempAlgParamsRSL2.buffInfo.hcclBuffBaseOff = meshCommBuffOffset_;
            tempAlgParamsRSL2.repeatNum = 1;
            tempAlgParamsRSL2.inputRepeatStride = 0;
            tempAlgParamsRSL2.outputRepeatStride = 0;
            tempAlgParamsRSL2.rhdSkipPost = true;
            HCCL_INFO(
                "[InsV2AllReduceSequenceExecutorAicpu3Level] loop [%llu] RHD chunkCount[%llu] subSliceCount1[%llu] "
                "subSliceCount2[%llu] RSL2BaseOff[%llu]",
                loop, chunkCount, subSliceCount1, subSliceCount2, tempAlgParamsRSL2.buffInfo.inBuffBaseOff);
            CHK_RET(algTemplateRSL2->KernelRun(param, tempAlgParamsRSL2, templateResourceRSL2));

            // 将被合并方份额放入代表的 AGL2 owner 槽。
            CHK_RET(ArSeqRhdBridgeMergedShare(
                static_cast<u32>(rankSizeLevel2_), static_cast<u32>(rankIdxLevel2_), subSliceSize2, subSliceCount2,
                loopResCtx.cclMem.addr, meshCommBuffOffset_, loopResCtx.cclMem.addr, rankIdxLevel1_ * subSliceSize1,
                threads_[0]));

            // AGL2：跳过 PRE，输出与 scratch 共用网格。
            tempAlgParamsAGL2.count = subSliceCount1;
            tempAlgParamsAGL2.sliceSize = subSliceSize2;
            tempAlgParamsAGL2.tailSize = subSliceSize2;
            tempAlgParamsAGL2.inputSliceStride = subSliceSize2;  // 装载源 = [base+myIdx·ISS]（RSL2 输出同位）
            tempAlgParamsAGL2.outputSliceStride = subSliceSize2; // 汇聚网格 [base+owner·OSS]
            tempAlgParamsAGL2.buffInfo.inBuffBaseOff = rankIdxLevel1_ * subSliceSize1;
            tempAlgParamsAGL2.buffInfo.outBuffBaseOff = rankIdxLevel1_ * subSliceSize1;
            tempAlgParamsAGL2.buffInfo.hcclBuffBaseOff = rankIdxLevel1_ * subSliceSize1; // out/hccl 同区同基
            tempAlgParamsAGL2.repeatNum = 1;
            tempAlgParamsAGL2.inputRepeatStride = 0;
            tempAlgParamsAGL2.outputRepeatStride = 0;
            tempAlgParamsAGL2.skipPrePhase = true;
            CHK_RET(algTemplateAGL2->KernelRun(param, tempAlgParamsAGL2, templateResourceAGL2));

            // AGL1：保留 PRE，收集经 L2 汇聚的份额。
            if (!skipLevel1_) {
                tempAlgParamsAGL1.count = chunkCount;
                tempAlgParamsAGL1.sliceSize = subSliceSize1;
                tempAlgParamsAGL1.tailSize = subSliceSize1;
                tempAlgParamsAGL1.inputSliceStride = subSliceSize1; // 装载源 = [myIdx·SS1]（AGL2 输出位）
                tempAlgParamsAGL1.outputSliceStride = subSliceSize1;
                tempAlgParamsAGL1.buffInfo.inBuffBaseOff = 0;
                tempAlgParamsAGL1.buffInfo.outBuffBaseOff = 0;
                tempAlgParamsAGL1.buffInfo.hcclBuffBaseOff = 0;
                tempAlgParamsAGL1.repeatNum = 1;
                tempAlgParamsAGL1.inputRepeatStride = 0;
                tempAlgParamsAGL1.outputRepeatStride = 0;
                CHK_RET(algTemplateAGL1->KernelRun(param, tempAlgParamsAGL1, templateResourceAGL1));
            }

            // AGL0：框内 MeshConcur AllGather，回写用户输出（尾块先落隔离 tile，再回拷有效字节）。
            ArSeq3LevelGenAGL0Params(
                MakeArSeq3LevelCtx(), loop, plan.currDataCount, processedDataCount, tempAlgParamsRSL0.sliceSize,
                tempAlgParamsRSL0.tailSize, 0, tempAlgParamsAGL0, "InsV2AllReduceSequenceExecutorAicpu3Level");
            if (plan.paddedTail) {
                tempAlgParamsAGL0.buffInfo.outputPtr = rhdTail.storage.addr;
                tempAlgParamsAGL0.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
                tempAlgParamsAGL0.buffInfo.outBuffBaseOff = rhdTail.outputOffset;
            }
            CHK_RET(algTemplateAGL0->KernelRun(param, tempAlgParamsAGL0, templateResourceAGL0));
            if (plan.paddedTail) {
                CHK_RET(ArSeqRhdCopyTailOutput(param, rhdTail, processedDataCount, plan.validCount, threads_[0]));
            }

            processedDataCount += plan.validCount;
            loop++;
        }
    } else {
        if (param.supportSymmetricMemory) {
            maxCountPerLoop = dataCount_;
            tempAlgParamsRSL0.buffInfo.outputPtr = param.inputPtr;
            tempAlgParamsRSL0.buffInfo.outBuffType = BufferType::INPUT;
            tempAlgParamsRSL1.buffInfo.inputPtr = param.inputPtr;
            tempAlgParamsRSL1.buffInfo.inBuffType = BufferType::INPUT;
            tempAlgParamsRSL1.buffInfo.outputPtr = param.inputPtr;
            tempAlgParamsRSL1.buffInfo.outBuffType = BufferType::INPUT;
            tempAlgParamsRSL2.buffInfo.inputPtr = param.inputPtr;
            tempAlgParamsRSL2.buffInfo.inBuffType = BufferType::INPUT;
            tempAlgParamsRSL2.buffInfo.outputPtr = param.inputPtr;
            tempAlgParamsRSL2.buffInfo.outBuffType = BufferType::INPUT;
            tempAlgParamsAGL2.buffInfo.inputPtr = param.outputPtr;
            tempAlgParamsAGL2.buffInfo.inBuffType = BufferType::OUTPUT;
            tempAlgParamsAGL2.buffInfo.outputPtr = param.outputPtr;
            tempAlgParamsAGL2.buffInfo.outBuffType = BufferType::OUTPUT;
            tempAlgParamsAGL2.enableRemoteMemAccess = true;
            tempAlgParamsAGL1.buffInfo.inputPtr = param.outputPtr;
            tempAlgParamsAGL1.buffInfo.inBuffType = BufferType::OUTPUT;
            tempAlgParamsAGL1.buffInfo.outputPtr = param.outputPtr;
            tempAlgParamsAGL1.buffInfo.outBuffType = BufferType::OUTPUT;
            tempAlgParamsAGL1.enableRemoteMemAccess = true;
            tempAlgParamsAGL0.buffInfo.inputPtr = param.outputPtr;
            tempAlgParamsAGL0.buffInfo.inBuffType = BufferType::OUTPUT;
            tempAlgParamsAGL0.enableRemoteMemAccess = true;
            HCCL_INFO(
                "[InsV2AllReduceSequenceExecutorAicpu3Level][OrchestrateLoop] %s: symmetric memory enabled",
                param.algName);
        }
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
                            "[InsV2AllReduceSequenceExecutorAicpu3Level] rsResultBuffSize_[%llu] is smaller than "
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
            ArSeq3LevelGenRSL0Params(
                MakeArSeq3LevelCtx(), loop, currDataCount, processedDataCount, tempAlgParamsRSL0,
                "InsV2AllReduceSequenceExecutorAicpu3Level");
            CHK_RET(algTemplateRSL0->KernelRun(param, tempAlgParamsRSL0, templateResourceRSL0));

            // ----------- RSL1: level1 ReduceScatter -----------
            u64 sliceSizeRSL1 = tempAlgParamsRSL0.sliceSize;
            u64 tailSizeRSL1 = tempAlgParamsRSL0.tailSize;
            u64 symMemBaseOffRSL1 = 0;
            if (!skipLevel1_) {
                ArSeq3LevelGenRSL1Params(
                    MakeArSeq3LevelCtx(), loop, currDataCount, tempAlgParamsRSL0.sliceSize, tempAlgParamsRSL0.tailSize,
                    processedDataCount, tempAlgParamsRSL1, "InsV2AllReduceSequenceExecutorAicpu3Level");
                CHK_RET(algTemplateRSL1->KernelRun(param, tempAlgParamsRSL1, templateResourceRSL1));
                sliceSizeRSL1 = tempAlgParamsRSL1.sliceSize;
                tailSizeRSL1 = tempAlgParamsRSL1.tailSize;
                symMemBaseOffRSL1 = tempAlgParamsRSL1.buffInfo.inBuffBaseOff;
            } else {
                sliceSizeRSL1 = tempAlgParamsRSL0.sliceSize;
                tailSizeRSL1 = (rankIdxLevel0_ == rankSizeLevel0_ - 1) ? tempAlgParamsRSL0.tailSize :
                                                                         tempAlgParamsRSL0.sliceSize;
                symMemBaseOffRSL1 = supportSymmetricMemory_ ? processedDataCount * dataTypeSize_
                                                                  + rankIdxLevel0_ * tempAlgParamsRSL0.sliceSize :
                                                              0;
            }

            // ----------- RSL2: level2 ReduceScatter -----------
            GenTempAlgParamsRSL2(
                loop, currDataCount, sliceSizeRSL1, tailSizeRSL1, symMemBaseOffRSL1, tempAlgParamsRSL2);
            CHK_RET(algTemplateRSL2->KernelRun(param, tempAlgParamsRSL2, templateResourceRSL2));

            // 对称内存路径：RS 完成后 input[mySlice] 已是归约结果，拷贝到 output[mySlice] 供 AG 阶段使用
            if (param.supportSymmetricMemory && currDataCount > 0) {
                u64 mySliceSize = (rankIdxLevel2_ == rankSizeLevel2_ - 1) ? tempAlgParamsRSL2.tailSize :
                                                                            tempAlgParamsRSL2.sliceSize;
                u64 mySliceOffset
                    = tempAlgParamsRSL2.buffInfo.inBuffBaseOff + rankIdxLevel2_ * tempAlgParamsRSL2.sliceSize;
                DataSlice copySrcSlice(param.inputPtr, mySliceOffset, mySliceSize, mySliceSize / dataTypeSize_);
                DataSlice copyDstSlice(param.outputPtr, mySliceOffset, mySliceSize, mySliceSize / dataTypeSize_);
                CHK_RET(LocalCopy(threads_[0], copySrcSlice, copyDstSlice));
            }

            // ----------- AGL2: level2 AllGather -----------
            GenTempAlgParamsAGL2(
                loop, currDataCount, tempAlgParamsRSL2.sliceSize, tempAlgParamsRSL2.tailSize, sliceSizeRSL1,
                symMemBaseOffRSL1, tempAlgParamsAGL2);
            CHK_RET(algTemplateAGL2->KernelRun(param, tempAlgParamsAGL2, templateResourceAGL2));

            // ----------- AGL1: level1 AllGather -----------
            if (!skipLevel1_) {
                ArSeq3LevelGenAGL1Params(
                    MakeArSeq3LevelCtx(), loop, currDataCount, sliceSizeRSL1, tailSizeRSL1, symMemBaseOffRSL1,
                    tempAlgParamsAGL1, "InsV2AllReduceSequenceExecutorAicpu3Level");
                CHK_RET(algTemplateAGL1->KernelRun(param, tempAlgParamsAGL1, templateResourceAGL1));
            }

            // ----------- AGL0: level0 AllGather -----------
            ArSeq3LevelGenAGL0Params(
                MakeArSeq3LevelCtx(), loop, currDataCount, processedDataCount, tempAlgParamsRSL0.sliceSize,
                tempAlgParamsRSL0.tailSize, symMemBaseOffRSL1, tempAlgParamsAGL0,
                "InsV2AllReduceSequenceExecutorAicpu3Level");
            CHK_RET(algTemplateAGL0->KernelRun(param, tempAlgParamsAGL0, templateResourceAGL0));

            processedDataCount += currDataCount;
            loop++;
        }
    } // else（非 RHD：既有 NHR 六步路径）
    HCCL_INFO("[InsV2AllReduceSequenceExecutorAicpu3Level][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSequenceMeshConcurNHRNHR, InsV2AllReduceSequenceExecutorAicpu3Level,
    TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterNHR, InsTempReduceScatterNHR,
    InsTempAllGatherNHR, InsTempAllGatherNHR, InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(AicpuAllReduceSequenceMeshConcurNHRNHR, topo.minTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
                   op.isSupportProd = false; op.unsupportedDataTypes
                                             = {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64,
                                                HcclDataType::HCCL_DATA_TYPE_FP64});

// 三级 AICPU RHD：L0 MeshConcur，L1/L2 RHD；仅显式配置时参与选路。
// L1 必须为设备侧 CLOS，L2 必须为设备侧链路，避免 RHD 模板过滤 ROCE 后无可用通道。
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_ALLREDUCE, AicpuAllReduceSequenceMeshConcurRHDRHD, InsV2AllReduceSequenceExecutorAicpu3Level,
    TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterRHD, InsTempReduceScatterRHD,
    InsTempAllGatherRHD, InsTempAllGatherRHD, InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(
    AicpuAllReduceSequenceMeshConcurRHDRHD, topo.minTopoLevelNum = TOPO_LEVEL_NUM_3;
    topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
    // 允许 level0 退化为单 rank 的拓扑通过 Level1Nhr 硬过滤。
    topo.isSupportLevel1Nhr = true;
    op.isSupportProd = false;
    op.unsupportedDataTypes
    = {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP64};
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) -> bool {
        return false;
    };
    topo.topoPrecondCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->physicalLevels.size() == TOPO_LEVEL_NUM_3
               && topo->physicalLevels[1].topoType == CommTopo::COMM_TOPO_CLOS
               && topo->physicalLevels[1].locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE
               && topo->physicalLevels[2].locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    };);

} // namespace ops_hccl
