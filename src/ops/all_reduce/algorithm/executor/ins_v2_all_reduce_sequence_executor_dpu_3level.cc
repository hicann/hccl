/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_reduce_sequence_executor_dpu_3level.h"
#include "executor_common_ops.h"
#include "ins_temp_reduce_scatter_mesh_1D_Z_axis_detour.h"
#include "ins_temp_reduce_scatter_nhr.h"
#include "ins_temp_reduce_scatter_mesh_1D_dpu_inter.h"
#include "ins_temp_all_gather_nhr_dpu_inter.h"
#include "ins_temp_all_gather_nhr.h"
#include "ins_temp_all_gather_mesh_1D.h"
#include "ins_temp_all_gather_mesh_1D_Z_axis_detour.h"
#include "alg_data_trans_wrapper.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"

namespace ops_hccl {

constexpr u32 SEQUENCE_EXECUTOR_LEVEL_NUM = 3;
constexpr u32 OMNIPIPE_LEVEL2_IDX = 2;

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
std::vector<CostModelParam> InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    CalcCostCoeff(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    // 三级 topomatch 失败时回退二级（ST topoLevelNums 可能为 2 但物理上有 3 级）
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AllReduceSequenceExecutorDpu3Level][CalcCostCoeff]",
            TopoProbeScene::PROBE_CALC_COST_COEFF)) {
        return {};
    }
    u32 rankSize = topoInfo->userRankSize;
    bool isPod = topoInfo->isPod;
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    bool skipL1 = (rankSizeLevel1 < 2);
    // 准入：d2 >= 2（冗余但无害，topomatch 已兜底）；三级无 d2>=2 则该级退化、无 DPU 通信
    if (rankSizeLevel2 < 2) {
        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutorDpu3Level][CalcCostCoeff] rankSizeLevel2[%u] < 2, skip.", rankSizeLevel2);
        return {};
    }
    // algo level 对应物理层(MeshConcur 为 2元组)逐层下传, 供跨物理层模板用; 层数不足时高层复用低层
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    CHK_PRT_RET(
        physIdx.size() != TOPO_LEVEL_NUM_3,
        HCCL_WARNING(
            "[InsV2AllReduceSequenceExecutorDpu3Level][CalcCostCoeff] physicalIdxForAlgoLevels size[%zu] != 3.",
            physIdx.size()),
        {});
    // 直接读 physicalLevels[].locType（不改基类）：区分同形态(CLOS)落在 DEVICE(设备/UB) 还是 HOST(RoCE)
    const auto getLocType = [topoInfo](u32 levelIdx) -> EndpointLocType {
        if (topoInfo == nullptr || levelIdx >= topoInfo->physicalLevels.size()) {
            return EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
        }
        return topoInfo->physicalLevels[levelIdx].locType;
    };
    std::vector<std::vector<CommTopo>> phyLevelNetTypes(physIdx.size());
    std::vector<std::vector<std::vector<u32>>> phyLevelPortNums(physIdx.size());
    std::vector<std::vector<EndpointLocType>> phyLevelLocTypes(physIdx.size());
    for (u32 lvl = 0; lvl < physIdx.size(); lvl++) {
        for (PhysicalLevelIndex phyIdx : physIdx[lvl]) {
            phyLevelNetTypes[lvl].push_back(GetPhysicalLevelTopoType(topoInfo, static_cast<u32>(phyIdx)));
            phyLevelPortNums[lvl].push_back(GetPhysicalLevelPortNums(topoInfo, static_cast<u32>(phyIdx)));
            phyLevelLocTypes[lvl].push_back(getLocType(static_cast<u32>(phyIdx)));
        }
    }
    CommTopo netTypeLevel0 = phyLevelNetTypes[0][0];
    CommTopo netTypeLevel1 = phyLevelNetTypes[1][0];
    CommTopo netTypeLevel2 = phyLevelNetTypes[2][0];
    const std::vector<u32>& portNumLevel0 = phyLevelPortNums[0][0];
    const std::vector<u32>& portNumLevel1 = phyLevelPortNums[1][0];
    const std::vector<u32>& portNumLevel2 = phyLevelPortNums[2][0];
    // 准入：中间层(L1)网型须为 CLOS（skipL1 时 L1 无通信，不校验）
    if (!skipL1 && netTypeLevel1 != CommTopo::COMM_TOPO_CLOS) {
        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutorDpu3Level][CalcCostCoeff] middle level netType[%u] != CLOS, skip.",
            static_cast<int>(netTypeLevel1));
        return {};
    }
    // 准入：中间层(L1)必须落在设备侧(DEVICE)（skipL1 时 L1 无通信，不校验）
    if (!skipL1 && phyLevelLocTypes[1][0] != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE) {
        HCCL_INFO(
            "[InsV2AllReduceSequenceExecutorDpu3Level][CalcCostCoeff] middle level locType[%u] != DEVICE, skip.",
            static_cast<int>(phyLevelLocTypes[1][0]));
        return {};
    }
    if (portNumLevel0.empty() || (!skipL1 && portNumLevel1.empty()) || portNumLevel2.empty()) {
        HCCL_WARNING("[InsV2AllReduceSequenceExecutorDpu3Level][CalcCostCoeff] portNum is empty");
        return {};
    }
    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorDpu3Level][CalcCostCoeff] rankSize=%d, rankSizeLevel0=%d, rankSizeLevel1=%d, "
        "rankSizeLevel2=%d, portNumLevel0=%zu, portNumLevel1=%zu, portNumLevel2=%zu, netTypeLevel0=%d, "
        "netTypeLevel1=%d, netTypeLevel2=%d, skipL1=%d",
        rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0.size(), portNumLevel1.size(),
        portNumLevel2.size(), static_cast<int>(netTypeLevel0), static_cast<int>(netTypeLevel1),
        static_cast<int>(netTypeLevel2), static_cast<int>(skipL1));
    std::vector<CostModelParam> params
        = [rankSize, rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, portNumLevel0, portNumLevel1, portNumLevel2,
           netTypeLevel0, netTypeLevel1, netTypeLevel2, isPod, physIdx, phyLevelNetTypes, phyLevelPortNums, skipL1] {
              std::vector<CostModelParam> v;
              auto p0 = InsAlgTemplate0::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f / rankSizeLevel0, netTypeLevel0, BufferType::INPUT, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel0, isPod, nullptr, nullptr, nullptr, 1u, physIdx[0],
                  phyLevelNetTypes[0], phyLevelPortNums[0]});
              v.insert(v.end(), p0.begin(), p0.end());
              if (!skipL1) {
                  auto p1 = InsAlgTemplate1::CalcCostCoeff(CalcCostCoeffParam{
                      rankSizeLevel1, 1.0f / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
                      BufferType::HCCL_BUFFER, portNumLevel1, isPod, nullptr, nullptr, nullptr, 1u, physIdx[1],
                      phyLevelNetTypes[1], phyLevelPortNums[1]});
                  v.insert(v.end(), p1.begin(), p1.end());
              }
              auto p2 = InsAlgTemplate2::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel2, 1.0f / rankSize, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel2, isPod, nullptr, nullptr, nullptr, 1u, physIdx[2],
                  phyLevelNetTypes[2], phyLevelPortNums[2]});
              v.insert(v.end(), p2.begin(), p2.end());
              // R11 纠正：AGL2(p3) 用 Level2 参数（AICPU 三级误用 Level0）
              auto p3 = InsAlgTemplate3::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel2, 1.0f / rankSize, netTypeLevel2, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
                  BufferType::HCCL_BUFFER, portNumLevel2, isPod, nullptr, nullptr, nullptr, 1u, physIdx[2],
                  phyLevelNetTypes[2], phyLevelPortNums[2]});
              v.insert(v.end(), p3.begin(), p3.end());
              if (!skipL1) {
                  auto p4 = InsAlgTemplate4::CalcCostCoeff(CalcCostCoeffParam{
                      rankSizeLevel1, 1.0f / rankSize, netTypeLevel1, BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER,
                      BufferType::HCCL_BUFFER, portNumLevel1, isPod, nullptr, nullptr, nullptr, 1u, physIdx[1],
                      phyLevelNetTypes[1], phyLevelPortNums[1]});
                  v.insert(v.end(), p4.begin(), p4.end());
              }
              // R11 纠正：AGL0(p5) 用 Level0 参数（AICPU 三级误用 Level2）
              auto p5 = InsAlgTemplate5::CalcCostCoeff(CalcCostCoeffParam{
                  rankSizeLevel0, 1.0f / rankSizeLevel0, netTypeLevel0, BufferType::HCCL_BUFFER, BufferType::OUTPUT,
                  BufferType::HCCL_BUFFER, portNumLevel0, isPod, nullptr, nullptr, nullptr, 1u, physIdx[0],
                  phyLevelNetTypes[0], phyLevelPortNums[0]});
              v.insert(v.end(), p5.begin(), p5.end());
              return v;
          }();
    return params;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
AlgNetMeta InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GetAlgNetMeta(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    AlgHierarchyInfoForAllLevel algHierarchyInfo;
    if (!MatchTopoForProbe<AlgTopoMatch>(
            topoInfo, algHierarchyInfo, algName, "[InsV2AllReduceSequenceExecutorDpu3Level][GetAlgNetMeta]",
            TopoProbeScene::PROBE_GET_ALG_NET_META)) {
        return {};
    }
    u32 rankSizeLevel0 = algHierarchyInfo.infos[0][0].size();
    u32 rankSizeLevel1 = (algHierarchyInfo.infos.size() > 1) ? algHierarchyInfo.infos[1][0].size() : 1;
    u32 rankSizeLevel2 = (algHierarchyInfo.infos.size() > 2) ? algHierarchyInfo.infos[2][0].size() : 1;
    u32 rankSize = topoInfo->userRankSize;
    CHK_PRT_RET(
        algHierarchyInfo.physicalIdxForAlgoLevels.empty(),
        HCCL_WARNING("[InsV2AllReduceSequenceExecutorDpu3Level][GetAlgNetMeta] physicalIdxForAlgoLevels is empty."),
        {});
    const auto& physIdx = algHierarchyInfo.physicalIdxForAlgoLevels;
    u32 physIdxLevel0 = static_cast<u32>(physIdx[0][0]);
    u32 physIdxLevel1 = (physIdx.size() > 1) ? static_cast<u32>(physIdx[1][0]) : physIdxLevel0;
    u32 physIdxLevel2 = (physIdx.size() > 2) ? static_cast<u32>(physIdx[2][0]) : physIdxLevel0;
    CommTopo netTypeLevel0 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel0);
    CommTopo netTypeLevel1 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel1);
    CommTopo netTypeLevel2 = GetPhysicalLevelTopoType(topoInfo, physIdxLevel2);
    AlgNetMeta meta;
    bool skipL1 = (rankSizeLevel1 < 2);
    // R11 纠正：netTypes p3(AGL2)=nt2、p5(AGL0)=nt0（AICPU 三级 p3=nt0/p5=nt2 交换）
    meta.netTypes.push_back(netTypeLevel0);
    if (!skipL1) {
        meta.netTypes.push_back(netTypeLevel1);
    }
    meta.netTypes.push_back(netTypeLevel2);
    meta.netTypes.push_back(netTypeLevel2);
    if (!skipL1) {
        meta.netTypes.push_back(netTypeLevel1);
    }
    meta.netTypes.push_back(netTypeLevel0);
    meta.intraGroupMode = CostAggMode::SUM;
    if (!skipL1) {
        meta.groupSizes = {1, 1, 1, 1, 1, 1};
        meta.dataRatios = {1.0f / rankSizeLevel0, 1.0f / rankSize, 1.0f / rankSize,
                           1.0f / rankSize,       1.0f / rankSize, 1.0f / rankSizeLevel0};
        meta.rankSizes
            = {rankSizeLevel0, rankSizeLevel1, rankSizeLevel2, rankSizeLevel2, rankSizeLevel1, rankSizeLevel0};
        meta.algoTypes = {AlgoType::MESH_CONCUR, AlgoType::NHR, AlgoType::MESH,
                          AlgoType::NHR,         AlgoType::NHR, AlgoType::MESH_CONCUR};
    } else {
        meta.groupSizes = {1, 1, 1, 1};
        meta.dataRatios = {1.0f / rankSizeLevel0, 1.0f / rankSize, 1.0f / rankSize, 1.0f / rankSizeLevel0};
        meta.rankSizes = {rankSizeLevel0, rankSizeLevel2, rankSizeLevel2, rankSizeLevel0};
        meta.algoTypes = {AlgoType::MESH_CONCUR, AlgoType::MESH, AlgoType::NHR, AlgoType::MESH_CONCUR};
    }
    return meta;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::InsV2AllReduceSequenceExecutorDpu3Level()
{}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2AllReduceSequenceExecutorDpu3Level<
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
HcclResult InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    SequenceInitCommInfo(param, topoInfo, algHierarchyInfo, "InsV2AllReduceSequenceExecutorDpu3Level");
    if (algHierarchyInfo.infos.size() != SEQUENCE_EXECUTOR_LEVEL_NUM) {
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutorDpu3Level] algHierarchyInfo size should be %u",
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
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorDpu3Level] level0 channels empty");
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[0] = resReqRSL0.channels[0];
    if (!skipLevel1_) {
        if (resReqRSL1.channels.empty()) {
            HCCL_ERROR("[InsV2AllReduceSequenceExecutorDpu3Level] level1 channels empty");
            return HCCL_E_INTERNAL;
        }
        resourceRequest.channels[1] = resReqRSL1.channels[0];
    }
    if (resReqRSL2.channels.empty()) {
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorDpu3Level] level2 channels empty");
        return HCCL_E_INTERNAL;
    }
    resourceRequest.channels[2] = resReqRSL2.channels[0];
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::Orchestrate(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutorDpu3Level][Orchestrate] Orchestrate Start");
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
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorDpu3Level][%s] invalid algHierarchyInfo infos.", __func__);
        return HCCL_E_PARA;
    }
    rankSizeLevel0_ = algHierarchyInfo_.infos[0][0].size();
    rankSizeLevel1_ = algHierarchyInfo_.infos[1][0].size();
    rankSizeLevel2_ = algHierarchyInfo_.infos[2][0].size();
    skipLevel1_ = (rankSizeLevel1_ == 1);
    if (skipLevel1_) {
        HCCL_INFO("[InsV2AllReduceSequenceExecutorDpu3Level][Orchestrate] level1 rankSize is 1, skip level1");
    }
    rankIdxLevel0_ = myRank_ % algHierarchyInfo_.infos[0][0].size();
    rankIdxLevel1_ = (myRank_ / algHierarchyInfo_.infos[0][0].size()) % algHierarchyInfo_.infos[1][0].size();
    rankIdxLevel2_ = myRank_ / (algHierarchyInfo_.infos[0][0].size() * algHierarchyInfo_.infos[1][0].size());

    CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutorDpu3Level][Orchestrate]errNo[0x%016llx] "
            "AllReduce executor kernel run failed",
            HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempAlgParamsRSL2(
        const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL1, const u64 tailSizeRSL1,
        const u64 symMemBaseOffRSL1, TemplateDataParams& tempAlgParamsRSL2) const
{
    // count 由 sliceSizeRSL1/tailSizeRSL1 推出（本 rank 经 RSL1 后的输出分片），currDataCount 不再单独使用
    (void)currDataCount;
    // DpuInter 变长分片契约：count 取本 rank 经 RSL1 后的输出分片，SplitData(count, d2) 填 allRank*
    // （替代 AICPU 设备 NHR 的 stride/sliceSize 契约，照二级 AR StepTwo）
    u64 count = (rankIdxLevel1_ == rankSizeLevel1_ - 1) ? tailSizeRSL1 / dataTypeSize_ : sliceSizeRSL1 / dataTypeSize_;
    tempAlgParamsRSL2.count = count;
    if (count != 0) {
        CHK_RET(SplitData(count, rankSizeLevel2_, tempAlgParamsRSL2));
    }

    // 输入/输出区在 rsResult 区（本 rank 的 RSL1 输出分片位置，沿用 AICPU 源块基址）
    u64 symMemBaseOff
        = supportSymmetricMemory_ ? symMemBaseOffRSL1 + rankIdxLevel1_ * sliceSizeRSL1 : rankIdxLevel1_ * sliceSizeRSL1;
    tempAlgParamsRSL2.buffInfo.inBuffBaseOff = symMemBaseOff;
    tempAlgParamsRSL2.buffInfo.outBuffBaseOff = symMemBaseOff;
    // R1/R3：DPU scratch 复用与 rsResult 隔离的 meshComm 区
    tempAlgParamsRSL2.buffInfo.hcclBuffBaseOff = meshCommBuffOffset_;

    // DpuInter 不消费 stride/sliceSize/tailSize（allRank* 变长分片契约）
    tempAlgParamsRSL2.sliceSize = 0;
    tempAlgParamsRSL2.tailSize = 0;
    tempAlgParamsRSL2.inputSliceStride = 0;
    tempAlgParamsRSL2.outputSliceStride = 0;

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorDpu3Level] loop [%llu] RSL2.count [%llu], RSL2.inBuffBaseOff [%llu], "
        "RSL2.outBuffBaseOff [%llu], RSL2.hcclBuffBaseOff [%llu]",
        loop, tempAlgParamsRSL2.count, tempAlgParamsRSL2.buffInfo.inBuffBaseOff,
        tempAlgParamsRSL2.buffInfo.outBuffBaseOff, tempAlgParamsRSL2.buffInfo.hcclBuffBaseOff);

    tempAlgParamsRSL2.repeatNum = 1;
    tempAlgParamsRSL2.inputRepeatStride = 0;
    tempAlgParamsRSL2.outputRepeatStride = 0;
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
void InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempAlgParamsAGL2(
        const u64 loop, const u64 currDataCount, const u64 sliceSizeRSL2, const u64 tailSizeRSL2,
        const u64 sliceSizeRSL1, const u64 symMemBaseOffRSL1, TemplateDataParams& tempAlgParamsAGL2) const
{
    (void)currDataCount;
    (void)sliceSizeRSL2;
    (void)tailSizeRSL2;
    (void)sliceSizeRSL1;
    // count 与 allRank* 均由 OrchestrateLoop 从 RSL2 复用（不重新 SplitData，R10）；此处只设基址/契约字段
    // 输入区读 RSL2 结果（rsResult 本 rank 位置），输出区写回本 L1 代表的 rsResult 区供 AG L1 读取
    // outBaseOff 必须按 rankIdxLevel1_ 错开：AGL1（NHR）在 L1 代表间 gather，代表 i 读 cclMem[i*sliceSizeRSL1]；
    // 若 outBaseOff=0，所有 L1 代表写同一位置 → AGL1 只能读到代表 0 的数据，跨 L1 代表的数据丢失（combine d0*d1 而非全
    // rank）
    u64 inBaseOff
        = supportSymmetricMemory_ ? symMemBaseOffRSL1 + rankIdxLevel1_ * sliceSizeRSL1 : rankIdxLevel1_ * sliceSizeRSL1;
    u64 outBaseOff
        = supportSymmetricMemory_ ? symMemBaseOffRSL1 + rankIdxLevel1_ * sliceSizeRSL1 : rankIdxLevel1_ * sliceSizeRSL1;
    tempAlgParamsAGL2.buffInfo.inBuffBaseOff = inBaseOff;
    tempAlgParamsAGL2.buffInfo.outBuffBaseOff = outBaseOff;
    tempAlgParamsAGL2.buffInfo.hcclBuffBaseOff = meshCommBuffOffset_;

    // DpuInter 不消费 stride/sliceSize/tailSize
    tempAlgParamsAGL2.sliceSize = 0;
    tempAlgParamsAGL2.tailSize = 0;
    tempAlgParamsAGL2.inputSliceStride = 0;
    tempAlgParamsAGL2.outputSliceStride = 0;

    HCCL_INFO(
        "[InsV2AllReduceSequenceExecutorDpu3Level] loop [%llu] AGL2.count [%llu], AGL2.inBuffBaseOff [%llu], "
        "AGL2.outBuffBaseOff [%llu], AGL2.hcclBuffBaseOff [%llu]",
        loop, tempAlgParamsAGL2.count, tempAlgParamsAGL2.buffInfo.inBuffBaseOff,
        tempAlgParamsAGL2.buffInfo.outBuffBaseOff, tempAlgParamsAGL2.buffInfo.hcclBuffBaseOff);

    // R2：AG Inter 模板隐式依赖 repeatNum=1（sliceSize=0 时 repeatNum>1 会使 scratchRepeatStride=0、所有 repeat
    // 写同一槽位 → 覆盖）
    tempAlgParamsAGL2.repeatNum = 1;
    tempAlgParamsAGL2.inputRepeatStride = 0;
    tempAlgParamsAGL2.outputRepeatStride = 0;
    return;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
template <typename InsAlgTemplate>
HcclResult InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::
    GenTempResource(
        const AlgResourceCtxSerializable& resCtx, const u32 channelLevelIdx,
        const std::shared_ptr<InsAlgTemplate>& algTemplate, TemplateResource& tempResource) const
{
    AlgResourceRequest req;
    // 序列内全部模板（含 DPU Inter）均已 override GetRes：编排期按 CalcRes 口径重建线程/Notify 请求；
    // 失败即真实异常（模板漏 override 或 GetRes 实现出错），报错返回，不再静默降级为 slaveThreadNum=0
    CHK_PRT_RET(
        algTemplate->GetRes(req) != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllReduceSequenceExecutorDpu3Level][GenTempResource] template GetRes failed."),
        HCCL_E_INTERNAL);
    if (channelLevelIdx >= remoteRankToChannelInfo_.size()) {
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutorDpu3Level][GenTempResource] channelLevelIdx[%u] should be lower"
            "than remoteRankToChannelInfo_.size()[%zu]",
            channelLevelIdx, remoteRankToChannelInfo_.size());
        return HCCL_E_INTERNAL;
    }
    tempResource.channels = remoteRankToChannelInfo_[channelLevelIdx];
    tempResource.threads.assign(resCtx.threads.begin(), resCtx.threads.begin() + 1 + req.slaveThreadNum);
    // Task2：DPU 共享内存透传（AICPU 三级缺这两行；不透传则 L2 DPU 模板 KernelRun 读到野值崩溃）
    tempResource.npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
    tempResource.dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
    tempResource.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
    tempResource.dpuThread = resCtx.dpuThread;
    tempResource.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
    tempResource.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::OrchestrateLoop(const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AllReduceSequenceExecutorDpu3Level][OrchestrateLoop] Start");

    TemplateDataParams tempAlgParamsRSL0;
    TemplateDataParams tempAlgParamsRSL1;
    TemplateDataParams tempAlgParamsRSL2;
    TemplateDataParams tempAlgParamsAGL2;
    TemplateDataParams tempAlgParamsAGL1;
    TemplateDataParams tempAlgParamsAGL0;
    ArSeq3LevelGenBaseBuffInfo(
        param, resCtx, tempAlgParamsRSL0, tempAlgParamsRSL1, tempAlgParamsRSL2, tempAlgParamsAGL2, tempAlgParamsAGL1,
        tempAlgParamsAGL0);

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

    // R3：scratchMultiple 取三级乘积 d0*d1*d2（AICPU 三级只用 d0，不含 d2 因子；DPU L2 scratch 需 d2 槽位）
    u32 scratchMult0 = algTemplateRSL0->CalcScratchMultiple(BufferType::INPUT, BufferType::HCCL_BUFFER);
    u32 scratchMult1
        = skipLevel1_ ? 1 : algTemplateRSL1->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u32 scratchMult2 = algTemplateRSL2->CalcScratchMultiple(BufferType::HCCL_BUFFER, BufferType::HCCL_BUFFER);
    u64 scratchMultiplier
        = static_cast<u64>(scratchMult0) * static_cast<u64>(scratchMult1) * static_cast<u64>(scratchMult2);
    u32 cclBuffSliceNum = static_cast<u32>(scratchMultiplier) + 1;
    // 限制来自local reduce的地址需要按照数据类型宽度对齐
    cclBuffSliceSize_ = resCtx.cclMem.size / cclBuffSliceNum / dataTypeSize_ * dataTypeSize_;
    rsResultBuffSize_ = cclBuffSliceSize_;
    meshCommBuffSize_ = scratchMultiplier * cclBuffSliceSize_;
    rsResultBuffOffset_ = 0;
    meshCommBuffOffset_ = rsResultBuffSize_;
    // 单 loop 处理量约束：rsResult 区需容纳 RSL0 输出 count/d0，meshComm 区需容纳各模板 scratch。
    // scratchMultiplier 取 d0*d1*d2（含 DPU L2 的 d2 因子），故单份 slice 必须按 cclMem/scratchMultiplier 收敛；
    // 若沿用 AICPU 的 meshCommBuffSize_ 公式，M>d0 时 maxCountPerLoop 偏大 → RSL0 输出越出 rsResult 区。
    u32 totalRankAlign = rankSizeLevel0_ * rankSizeLevel1_ * rankSizeLevel2_;
    u64 maxCountPerLoop = resCtx.cclMem.size / scratchMultiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN
                          / dataTypeSize_ / totalRankAlign * totalRankAlign;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[InsV2AllReduceSequenceExecutorDpu3Level] maxCountPerLoop is 0, cclBuffSize[%llu], "
            "scratchMultiplier[%llu], dataTypeSize[%llu]",
            static_cast<u64>(resCtx.cclMem.size), scratchMultiplier, static_cast<u64>(dataTypeSize_));
        return HCCL_E_INTERNAL;
    }
    u64 processedDataCount = 0;
    u64 loop = 0;
    if (param.supportSymmetricMemory) {
        // 对称路径输入输出均在对称窗口，不受 cclBuffer 分区限制，可放开单轮处理量上限；每轮实际
        // 处理量由下方尾块收缩逻辑隐式约束（RSL0 尾分片 ≤ rsResultBuffSize_），RSL2/AGL2（DPU 模板，
        // 收发槽位在 cclBuffer 的 meshComm 区）的每轮槽位需求随之收敛，不会越出 cclBuffer。
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
            "[InsV2AllReduceSequenceExecutorDpu3Level][OrchestrateLoop] %s: symmetric memory enabled", param.algName);
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
                        "[InsV2AllReduceSequenceExecutorDpu3Level] rsResultBuffSize_[%llu] is smaller than "
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
            "InsV2AllReduceSequenceExecutorDpu3Level");
        CHK_RET(algTemplateRSL0->KernelRun(param, tempAlgParamsRSL0, templateResourceRSL0));

        // ----------- RSL1: level1 ReduceScatter -----------
        u64 sliceSizeRSL1 = tempAlgParamsRSL0.sliceSize;
        u64 tailSizeRSL1 = tempAlgParamsRSL0.tailSize;
        u64 symMemBaseOffRSL1 = 0;
        if (!skipLevel1_) {
            ArSeq3LevelGenRSL1Params(
                MakeArSeq3LevelCtx(), loop, currDataCount, tempAlgParamsRSL0.sliceSize, tempAlgParamsRSL0.tailSize,
                processedDataCount, tempAlgParamsRSL1, "InsV2AllReduceSequenceExecutorDpu3Level");
            CHK_RET(algTemplateRSL1->KernelRun(param, tempAlgParamsRSL1, templateResourceRSL1));
            sliceSizeRSL1 = tempAlgParamsRSL1.sliceSize;
            tailSizeRSL1 = tempAlgParamsRSL1.tailSize;
            symMemBaseOffRSL1 = tempAlgParamsRSL1.buffInfo.inBuffBaseOff;
        } else {
            sliceSizeRSL1 = tempAlgParamsRSL0.sliceSize;
            tailSizeRSL1
                = (rankIdxLevel0_ == rankSizeLevel0_ - 1) ? tempAlgParamsRSL0.tailSize : tempAlgParamsRSL0.sliceSize;
            symMemBaseOffRSL1 = supportSymmetricMemory_ ?
                                    processedDataCount * dataTypeSize_ + rankIdxLevel0_ * tempAlgParamsRSL0.sliceSize :
                                    0;
        }

        // ----------- RSL2: level2 ReduceScatter (DPU) -----------
        CHK_RET(GenTempAlgParamsRSL2(
            loop, currDataCount, sliceSizeRSL1, tailSizeRSL1, symMemBaseOffRSL1, tempAlgParamsRSL2));
        // count==0 的 rank 跳过 L2 通信（同二级 AR StepTwo）
        if (tempAlgParamsRSL2.count != 0) {
            CHK_RET(algTemplateRSL2->KernelRun(param, tempAlgParamsRSL2, templateResourceRSL2));
        }

        // 对称内存路径：RS 完成后结果在 inputPtr[outBuffBaseOff]（DPU PostLocalReduce 写到 outBuffBaseOff），
        // 拷贝到 outputPtr 供 AG 阶段使用（R7：对称路径基址语义与 AICPU 设备 NHR 不同，须 ST 验证）
        if (param.supportSymmetricMemory && tempAlgParamsRSL2.count != 0) {
            u64 mySliceSize = tempAlgParamsRSL2.allRankSliceSize.at(rankIdxLevel2_);
            u64 mySliceOffset = tempAlgParamsRSL2.buffInfo.outBuffBaseOff;
            if (mySliceSize > 0) {
                DataSlice copySrcSlice(param.inputPtr, mySliceOffset, mySliceSize, mySliceSize / dataTypeSize_);
                DataSlice copyDstSlice(param.outputPtr, mySliceOffset, mySliceSize, mySliceSize / dataTypeSize_);
                CHK_RET(LocalCopy(threads_[0], copySrcSlice, copyDstSlice));
            }
        }

        // ----------- AGL2: level2 AllGather (DPU) -----------
        // R10：AGL2 复用 RSL2 的 allRank* 与 count（不重新 SplitData）
        tempAlgParamsAGL2.allRankDispls = tempAlgParamsRSL2.allRankDispls;
        tempAlgParamsAGL2.allRankSliceSize = tempAlgParamsRSL2.allRankSliceSize;
        tempAlgParamsAGL2.allRankProcessedDataCount = tempAlgParamsRSL2.allRankProcessedDataCount;
        tempAlgParamsAGL2.count = tempAlgParamsRSL2.count;
        GenTempAlgParamsAGL2(
            loop, currDataCount, tempAlgParamsRSL2.sliceSize, tempAlgParamsRSL2.tailSize, sliceSizeRSL1,
            symMemBaseOffRSL1, tempAlgParamsAGL2);
        if (tempAlgParamsRSL2.count != 0) {
            CHK_RET(algTemplateAGL2->KernelRun(param, tempAlgParamsAGL2, templateResourceAGL2));
        }

        // ----------- AGL1: level1 AllGather -----------
        if (!skipLevel1_) {
            ArSeq3LevelGenAGL1Params(
                MakeArSeq3LevelCtx(), loop, currDataCount, sliceSizeRSL1, tailSizeRSL1, symMemBaseOffRSL1,
                tempAlgParamsAGL1, "InsV2AllReduceSequenceExecutorDpu3Level");
            CHK_RET(algTemplateAGL1->KernelRun(param, tempAlgParamsAGL1, templateResourceAGL1));
        }

        // ----------- AGL0: level0 AllGather -----------
        ArSeq3LevelGenAGL0Params(
            MakeArSeq3LevelCtx(), loop, currDataCount, processedDataCount, tempAlgParamsRSL0.sliceSize,
            tempAlgParamsRSL0.tailSize, symMemBaseOffRSL1, tempAlgParamsAGL0,
            "InsV2AllReduceSequenceExecutorDpu3Level");
        CHK_RET(algTemplateAGL0->KernelRun(param, tempAlgParamsAGL0, templateResourceAGL0));

        processedDataCount += currDataCount;
        loop++;
    }
    HCCL_INFO("[InsV2AllReduceSequenceExecutorDpu3Level][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
u64 InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::RoundUp(const u64 dividend, const u64 divisor) const
{
    if (divisor == 0) {
        HCCL_WARNING("[InsV2AllReduceSequenceExecutorDpu3Level][RoundUp] divisor is 0.");
        return dividend;
    }
    return (dividend + divisor - 1) / divisor;
}

template <
    typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2,
    typename InsAlgTemplate3, typename InsAlgTemplate4, typename InsAlgTemplate5>
HcclResult InsV2AllReduceSequenceExecutorDpu3Level<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2, InsAlgTemplate3, InsAlgTemplate4,
    InsAlgTemplate5>::SplitData(const u64& dataCount, const uint64_t& rankSize, TemplateDataParams& tempAlgParams) const
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
            "[InsV2AllReduceSequenceExecutorDpu3Level] SliceInfo: offset[%llu] size[%llu] count[%llu]",
            tempAlgParams.allRankDispls.at(i), tempAlgParams.allRankSliceSize.at(i),
            tempAlgParams.allRankProcessedDataCount.at(i));
    }

    return HcclResult::HCCL_SUCCESS;
}

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_ALLREDUCE, DpuAllReduceSequenceMeshConcurNHRMesh, InsV2AllReduceSequenceExecutorDpu3Level,
    TopoMatchThreeLevel, InsTempReduceScatterMesh1DZAxisDetour, InsTempReduceScatterNHR,
    InsTempReduceScatterMesh1dDpuInter, InsTempAllGatherNhrDpuInter, InsTempAllGatherNHR,
    InsTempAllGatherMesh1D1DZAxisDetour);
REGISTER_ALG_ATTRS(DpuAllReduceSequenceMeshConcurNHRMesh, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
                   topo.isHostDpuOnly = true; topo.isSupportLevel0PcieMix = true; op.isSupportProd = false;
                   op.unsupportedDataTypes
                   = {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64,
                      HcclDataType::HCCL_DATA_TYPE_FP64});

} // namespace ops_hccl
