/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_barrier_sequence_executor.h"
#include <algorithm>
#include <array>
#include <utility>
#include "executor_common_ops.h"
#include "ins_temp_barrier_mesh_1D.h"
#include "ins_temp_barrier_nhr_dpu.h"
#include "ins_temp_barrier_rhd_aicpu.h"
#include "ins_temp_barrier_rhd_dpu.h"
#include "topo_match_three_level.h"
#include "coll_alg_v2_exec_registry.h"
#include "alg_attrs_registry.h"
#ifndef AICPU_COMPILE
#include <cstdlib>
#include "alg_env_config.h"
#include "barrier_rhd_config.h"
#include "cost_model.h"
#include "op_common.h"
#endif

namespace ops_hccl {

namespace {
    constexpr u32 BARRIER_TWO_LEVELS = 2;
    constexpr u32 BARRIER_THREE_LEVELS = 3;

    HcclResult CheckBarrierHierarchy(const AlgHierarchyInfoForAllLevel& hierarchy, u32 myRank, u32 levelNum)
    {
        CHK_PRT_RET(
            hierarchy.infos.size() != levelNum,
            HCCL_ERROR("[BarrierSequence] expected[%u] levels, got[%zu].", levelNum, hierarchy.infos.size()),
            HCCL_E_INTERNAL);
        // 二层和三层拓扑匹配均为本 rank 每层生成一个子通信域，Rail 和 GCD 拆分也遵循此约定。
        for (u32 levelIdx = 0; levelIdx < levelNum; ++levelIdx) {
            const auto& level = hierarchy.infos[levelIdx];
            CHK_PRT_RET(
                level.size() != 1 || level[0].empty()
                    || std::find(level[0].begin(), level[0].end(), myRank) == level[0].end(),
                HCCL_ERROR(
                    "[BarrierSequence] invalid subcomm: rank[%u] level[%u] subcommCount[%zu].", myRank, levelIdx,
                    level.size()),
                HCCL_E_PARA);
        }
        return HCCL_SUCCESS;
    }

    template <typename Template>
    HcclResult CalcBarrierLevelRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        const std::vector<std::vector<u32>>& subCommRanks, AlgResourceRequest& request)
    {
        Template temp(param, topoInfo->userRank, subCommRanks);
        CHK_RET(temp.CalcRes(comm, param, topoInfo, request));
        CHK_PRT_RET(
            request.channels.size() != 1 || request.notifyNumPerThread.size() != request.slaveThreadNum,
            HCCL_ERROR("[BarrierSequence] invalid template resource request."), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    template <typename Template>
    HcclResult RunBarrierLevel(
        const OpParam& param, const AlgResourceCtxSerializable& resCtx,
        const std::map<u32, std::vector<ChannelInfo>>& channels, u32 level)
    {
        Template temp(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[level]);
        TemplateResource resource;
        SetTemplateCommResource(resource, channels, resCtx);
        resource.isDpuThreadEnabled = resCtx.isDpuThreadEnabled;
        resource.dpuThread = resCtx.dpuThread;
        resource.dpuThreadNotifyIdxOnMain = resCtx.notifyNumOnMainThread + 1;
        resource.isHcommChannelDrainSupported = resCtx.isHcommChannelDrainSupported;
        TemplateDataParams data{};
        data.buffInfo.hcclBuff = resCtx.cclMem;
        data.repeatNum = 1;
        HCCL_INFO(
            "[BarrierSequence] rank[%u] level[%u] template[%s].", resCtx.topoInfo.userRank, level,
            temp.Describe().c_str());
        return temp.KernelRun(param, data, resource);
    }

#ifndef AICPU_COMPILE
    bool HasExplicitBarrierRhdConfig(HcclComm comm, const char* algName, OpExecuteConfig engine)
    {
        std::string config;
        if (HcclGetHcclAlgo(comm, config) != HCCL_SUCCESS || config.empty()) {
            config = GetEnv("HCCL_ALGO");
        }
        HcclAlgoParser parser;
        if (config.empty() || config == "EmptyString" || parser.Parser(config) != HCCL_SUCCESS) {
            return false;
        }
        const auto& allAlgos = *GetAllAlgos();
        std::vector<const char*> barrierNames;
        for (int i = 0; i < allAlgos.count; ++i) {
            const auto& alg = allAlgos.algElements[i];
            if (alg.opType == HcclCMDType::HCCL_CMD_BARRIER && alg.algName != nullptr) {
                barrierNames.push_back(alg.algName);
            }
        }
        // 支持 Barrier 的候选引擎中，HOSTCPU 模式仅查 Dpu，其余模式也允许 Aicpu。
        const std::vector<std::string> enginePrefixes = engine == OpExecuteConfig::HOSTCPU ?
                                                            std::vector<std::string>{"Dpu"} :
                                                            std::vector<std::string>{"Aicpu", "Dpu"};
        return IsBarrierRhdExplicitlyConfigured(parser, algName, barrierNames, enginePrefixes);
    }

    TopoAttrs MakeBarrierRhdTopo(u8 levelNum, bool hostDpu, OpExecuteConfig engine, std::vector<AlgoType> algoTypes)
    {
        TopoAttrs topo;
        topo.minTopoLevelNum = levelNum;
        topo.maxTopoLevelNum = levelNum;
        topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D;
        topo.isHostDpuOnly = hostDpu;
        topo.supportDevTypes = {HcclDevType::DEV_TYPE_950};
        // 在选型阶段预检分组和端点位置，条件不满足时回退到默认选路。
        topo.topoPrecondCheck = [levelNum, hostDpu, engine,
                                 algoTypes = std::move(algoTypes)](const TopoInfoWithNetLayerDetails* topoInfo) {
            if (topoInfo == nullptr) {
                return false;
            }
            // 主线开启 DPU 线程时不会申请共享内存，当前 HostDPU RHD 只支持旧路径。
            if (hostDpu && std::getenv("ST_DPU_THREAD_NEW_FLOW") != nullptr) {
                HCCL_INFO("[BarrierRHD] HostDPU requires the shared-memory request path.");
                return false;
            }
            AlgAttrs attrs;
            attrs.engine = engine;
            attrs.algoTypes = algoTypes;
            AlgHierarchyInfoForAllLevel hierarchy;
            TopoInfoWithNetLayerDetails topo = *topoInfo;
            HcclResult matchRet;
            if (levelNum == BARRIER_TWO_LEVELS) {
                TopoMatchTwoLevel matcher;
                matchRet = matcher.MatchTopo(&topo, hierarchy, attrs);
            } else {
                TopoMatchThreeLevel matcher;
                matchRet = matcher.MatchTopo(&topo, hierarchy, attrs);
            }
            if (matchRet != HCCL_SUCCESS || hierarchy.physicalIdxForAlgoLevels.size() != levelNum) {
                return false;
            }
            for (u32 level = 0; level < levelNum; ++level) {
                const auto& sources = hierarchy.physicalIdxForAlgoLevels[level];
                if (sources.size() != 1) {
                    return false;
                }
                const u32 physicalIdx = static_cast<u32>(sources[0]);
                if (physicalIdx >= topo.physicalLevels.size()) {
                    return false;
                }
                const auto& physical = topo.physicalLevels[physicalIdx];
                if (hostDpu && level == levelNum - 1) {
                    if (physical.locType != ENDPOINT_LOC_TYPE_HOST
                        || std::find(
                               physical.protocols.begin(), physical.protocols.end(), CommProtocol::COMM_PROTOCOL_ROCE)
                               == physical.protocols.end()) {
                        return false;
                    }
                } else if (physical.locType != ENDPOINT_LOC_TYPE_DEVICE) {
                    return false;
                }
            }
            return true;
        };
        // 默认选路过滤 RHD，显式配置由 executor 校验。
        topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails*) {
            return false;
        };
        return topo;
    }
#endif
} // namespace

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);
    constexpr u32 LEVEL_NUM = std::is_void<InsAlgTemplate2>::value ? BARRIER_TWO_LEVELS : BARRIER_THREE_LEVELS;
    CHK_RET(CheckBarrierHierarchy(algHierarchyInfo, topoInfo->userRank, LEVEL_NUM));
    std::array<AlgResourceRequest, LEVEL_NUM> requests{};
    CHK_RET(CalcBarrierLevelRes<InsAlgTemplate0>(comm, param, topoInfo, algHierarchyInfo.infos[0], requests[0]));
    CHK_RET(CalcBarrierLevelRes<InsAlgTemplate1>(comm, param, topoInfo, algHierarchyInfo.infos[1], requests[1]));
    if constexpr (!std::is_void<InsAlgTemplate2>::value) {
        CHK_RET(CalcBarrierLevelRes<InsAlgTemplate2>(comm, param, topoInfo, algHierarchyInfo.infos[2], requests[2]));
    }
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.channels.clear();
    for (const auto& request : requests) {
        resourceRequest.slaveThreadNum = std::max(resourceRequest.slaveThreadNum, request.slaveThreadNum);
        resourceRequest.notifyNumOnMainThread
            = std::max(resourceRequest.notifyNumOnMainThread, request.notifyNumOnMainThread);
        resourceRequest.channels.push_back(request.channels[0]);
    }
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 0);
    for (const auto& request : requests) {
        for (u32 index = 0; index < request.slaveThreadNum; ++index) {
            resourceRequest.notifyNumPerThread[index]
                = std::max(resourceRequest.notifyNumPerThread[index], request.notifyNumPerThread[index]);
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    constexpr u32 LEVEL_NUM = std::is_void<InsAlgTemplate2>::value ? BARRIER_TWO_LEVELS : BARRIER_THREE_LEVELS;
    CHK_RET(CheckBarrierHierarchy(resCtx.algHierarchyInfo, resCtx.topoInfo.userRank, LEVEL_NUM));
    CHK_PRT_RET(
        resCtx.channels.size() != LEVEL_NUM, HCCL_ERROR("[BarrierSequence] incomplete serialized channel resources."),
        HCCL_E_INTERNAL);
    std::vector<std::map<u32, std::vector<ChannelInfo>>> channels;
    CHK_RET(RestoreChannelMap(resCtx, channels));
    CHK_PRT_RET(
        channels.size() != LEVEL_NUM || resCtx.threads.empty(),
        HCCL_ERROR("[BarrierSequence] incomplete channel/thread resources."), HCCL_E_INTERNAL);
    if constexpr (!std::is_void<InsAlgTemplate2>::value) {
        CHK_RET(RunBarrierLevel<InsAlgTemplate2>(param, resCtx, channels[2], 2));
    }
    CHK_RET(RunBarrierLevel<InsAlgTemplate1>(param, resCtx, channels[1], 1));
    return RunBarrierLevel<InsAlgTemplate0>(param, resCtx, channels[0], 0);
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
std::vector<CostModelParam>
InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
#ifndef AICPU_COMPILE
    if constexpr (
        std::is_same_v<InsAlgTemplate1, InsTempBarrierRhdAicpu>
        || std::is_same_v<InsAlgTemplate1, InsTempBarrierRhdDpu>) {
        if (!HasExplicitBarrierRhdConfig(comm, algName, param.opExecuteConfig)) {
            return {};
        }
    }
#endif
    return {{0.0f, 0.0f, 1.0f, 0.0f}};
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
AlgNetMeta InsV2BarrierSequenceExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)algName;
    (void)param;
    return BuildDefaultMeshNetMeta(topoInfo);
}

REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_BARRIER, DpuBarrierSequenceMeshNHR, InsV2BarrierSequenceExecutor, TopoMatchTwoLevel,
    InsTempBarrierMesh1D, InsTempBarrierNHRDPU);
REGISTER_ALG_ATTRS(DpuBarrierSequenceMeshNHR, topo.isSupportLevel0PcieMix = true; topo.isHostDpuOnly = true;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
                   topo.isSupportLevel1Nhr = true;);

REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_BARRIER, AicpuBarrierSequenceMeshRHD, InsV2BarrierSequenceExecutor, TopoMatchTwoLevel,
    InsTempBarrierMesh1D, InsTempBarrierRhdAicpu);
REGISTER_ALG_ATTRS(AicpuBarrierSequenceMeshRHD,
                   topo = MakeBarrierRhdTopo(BARRIER_TWO_LEVELS, false, a.engine, a.algoTypes););

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_BARRIER, AicpuBarrierSequenceMeshRHDRHD, InsV2BarrierSequenceExecutor, TopoMatchThreeLevel,
    InsTempBarrierMesh1D, InsTempBarrierRhdAicpu, InsTempBarrierRhdAicpu);
REGISTER_ALG_ATTRS(AicpuBarrierSequenceMeshRHDRHD,
                   topo = MakeBarrierRhdTopo(BARRIER_THREE_LEVELS, false, a.engine, a.algoTypes););

REGISTER_EXECUTOR_BY_TWO_TEMPS(
    HcclCMDType::HCCL_CMD_BARRIER, DpuBarrierSequenceMeshRHD, InsV2BarrierSequenceExecutor, TopoMatchTwoLevel,
    InsTempBarrierMesh1D, InsTempBarrierRhdDpu);
REGISTER_ALG_ATTRS(DpuBarrierSequenceMeshRHD,
                   topo = MakeBarrierRhdTopo(BARRIER_TWO_LEVELS, true, a.engine, a.algoTypes););

REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_BARRIER, DpuBarrierSequenceMeshRHDRHD, InsV2BarrierSequenceExecutor, TopoMatchThreeLevel,
    InsTempBarrierMesh1D, InsTempBarrierRhdAicpu, InsTempBarrierRhdDpu);
REGISTER_ALG_ATTRS(DpuBarrierSequenceMeshRHDRHD,
                   topo = MakeBarrierRhdTopo(BARRIER_THREE_LEVELS, true, a.engine, a.algoTypes););
} // namespace ops_hccl
