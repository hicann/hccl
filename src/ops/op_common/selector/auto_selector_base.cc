/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_selector_base.h"
#include "selector_registry.h"
#include "op_common.h"

namespace ops_hccl {

SelectorStatus
AutoSelectorBase::Select(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo, std::string& selectAlgName) const
{
    HCCL_DEBUG("[AutoSelectorBase][%s] start, OpExecuteConfig is %d.", __func__, opParam.opExecuteConfig);
    std::map<HcclCMDType, std::vector<HcclAlgoType>> configAlgMap = GetExternalInputHcclAlgoConfigAllType();
    SelectorStatus ret = SelectorStatus::NOT_MATCH;
    bool hostDPUOnly = false;
    if ((CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) == HCCL_SUCCESS) && hostDPUOnly) {
        opParam.opExecuteConfig = OpExecuteConfig::HOSTCPU;
        opParam.engine = CommEngine::COMM_ENGINE_CPU;
        return SelectDPUAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
    }
    if (opParam.opExecuteConfig == OpExecuteConfig::CCU_MS) {
        ret = SelectCcuMsAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        if (ret == SelectorStatus::NOT_MATCH) {
            opParam.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
        } else {
            return ret;
        }
    }
    if (opParam.opExecuteConfig == OpExecuteConfig::CCU_SCHED) {
        ret = SelectCcuScheduleAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        if (ret == SelectorStatus::NOT_MATCH) {
            opParam.opExecuteConfig = OpExecuteConfig::CCU_FAIL;
        } else {
            return ret;
        }
    }
    if (ProcessAivConfig(opParam, topoInfo, configAlgMap, selectAlgName, ret)) {
        return ret;
    }
    if (IsStarsState(opParam.opExecuteConfig)) {
        // 需要回退AIV的场景下，回退AIV算法
        if (IsRollBackAiv(opParam, topoInfo)) {
            HCCL_INFO("[Algo][AutoSelectorBase] Need to roll back AIV algo");
            opParam.opExecuteConfig = OpExecuteConfig::AIV_ONLY;
            (void)ProcessAivConfig(opParam, topoInfo, configAlgMap, selectAlgName, ret);
            HCCL_INFO(
                "[Algo][AutoSelectorBase] The selected algo is %s, OpExecuteConfig is %d.", selectAlgName.c_str(),
                opParam.opExecuteConfig);
            return ret;
        }
        ret = SelectAicpuAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        if (ret == SelectorStatus::MATCH) {
            opParam.opExecuteConfig = OpExecuteConfig::AICPU_TS;
        }
    }
    HCCL_INFO(
        "[Algo][AutoSelectorBase] The selected algo is %s, OpExecuteConfig is %d.", selectAlgName.c_str(),
        opParam.opExecuteConfig);
    return ret;
}

bool AutoSelectorBase::IsRollBackAiv(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo) const
{
    // Mesh类算法场景，ATU资源受限，切换为AIV算法
    bool isAllToAllOps = opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL
                         || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
                         || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC;
    bool isInt64ReduceOps = opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64
                            && (opParam.opType == HcclCMDType::HCCL_CMD_ALLREDUCE
                                || opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
                                || opParam.opType == HcclCMDType::HCCL_CMD_REDUCE);
    bool isPcieMeshScene
        = topoInfo->level0PcieMix && topoInfo->level0BigClosRange && (isAllToAllOps || isInt64ReduceOps);

    // P2P算子场景，ATU资源受限，使用PCIE链路的切换为AIV算法
    bool isP2pOps = (opParam.opType == HcclCMDType::HCCL_CMD_SEND) || (opParam.opType == HcclCMDType::HCCL_CMD_RECEIVE);
    bool isLayer0AllConnetedWithMesh = IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH);
    bool isPcieP2pScene
        = topoInfo->level0PcieMix && topoInfo->serverNum == 1 && isP2pOps && !isLayer0AllConnetedWithMesh;

    bool isRollBackAiv = false;
    if (isPcieMeshScene) {
        HCCL_INFO(
            "[AutoSelectorBase] Need to rollback aiv, isPcieMeshScene[%d]: isAllToAllOps[%d] isInt64ReduceOps[%d]",
            isPcieMeshScene, isAllToAllOps, isInt64ReduceOps);
        isRollBackAiv = true;
    } else if (isPcieP2pScene) {
        HCCL_INFO(
            "[AutoSelectorBase] Need to rollback aiv, isPcieP2pScene[%d]: isP2pOps[%d] isLayer0AllConnetedWithMesh[%u]",
            isPcieP2pScene, isP2pOps, isLayer0AllConnetedWithMesh);
        isRollBackAiv = true;
    }
    return isRollBackAiv;
}

bool AutoSelectorBase::IsStarsState(const OpExecuteConfig& opExecuteConfig) const
{
    return (
        opExecuteConfig == OpExecuteConfig::AICPU_TS || opExecuteConfig == OpExecuteConfig::HOSTCPU_TS
        || opExecuteConfig == OpExecuteConfig::CCU_FAIL);
}

bool AutoSelectorBase::IsDefaultAlg(const HcclAlgoType algoType) const
{
    return (algoType == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) || (algoType == HcclAlgoType::HCCL_ALGO_TYPE_NA);
}

bool AutoSelectorBase::IsSmallData(const u64 dataSize) const { return dataSize < SMALL_COUNT_512KB; }

bool AutoSelectorBase::IsLargeData(const u64 dataSize) const { return dataSize >= LARGE_COUNT_1024KB; }

bool AutoSelectorBase::IsSmallDataCCU(const u64 dataSize, const u64 rankSize) const
{
    if (rankSize == 0) {
        HCCL_WARNING("the selector is not set RankSize");
    }
    return (dataSize <= CCU_PARALLEL_MAX_DATA_SIZE) ? true : false;
}

u32 AutoSelectorBase::CalcFrameNum(const TopoInfoWithNetLayerDetails* topoInfo)
{
    u32 frameNum = 0;
    if (topoInfo->topoLevelNums <= 1 || topoInfo->netLayerDetails.instSizeListOfLayer[0].empty()) {
        return frameNum;
    }
    u32 gcd = topoInfo->netLayerDetails.instSizeListOfLayer[0][0];
    for (size_t i = 1; i < topoInfo->netLayerDetails.instSizeListOfLayer[0].size(); ++i) {
        u32 a = gcd;
        u32 b = topoInfo->netLayerDetails.instSizeListOfLayer[0][i];
        while (b != 0) {
            u32 r = a % b;
            a = b;
            b = r;
        }
        gcd = a;
        if (gcd == 1) {
            break;
        }
    }
    frameNum = (gcd > 0) ? topoInfo->userRankSize / gcd : 0;
    return frameNum;
}

SelectorStatus AutoSelectorBase::SelectCcuMsAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)opParam;
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectCcuScheduleAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)opParam;
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)opParam;
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAivAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)opParam;
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)opParam;
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

bool AutoSelectorBase::IsLayerAllConnetedWithTopo(
    const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType)
{
    CHK_PRT_RET(
        topoInfo->netLayerDetails.localNetInsSizeOfLayer.size() <= netLayer,
        HCCL_WARNING(
            "[BaseSelector][IsLayerAllConnetedWithTopo] localNetInsSizeOfLayer size[%u] <= netLayer[%u]",
            topoInfo->netLayerDetails.localNetInsSizeOfLayer.size(), netLayer),
        false);
    u32 localRankSize = topoInfo->netLayerDetails.localNetInsSizeOfLayer[netLayer];

    CHK_PRT_RET(
        topoInfo->topoInstDetailsOfLayer.size() <= netLayer,
        HCCL_WARNING(
            "[BaseSelector][IsLayerAllConnetedWithTopo] topoInstDetailsOfLayer size[%u] <= netLayer[%u]",
            topoInfo->topoInstDetailsOfLayer.size(), netLayer),
        false);

    auto rankNumForTopoTypeItr = topoInfo->topoInstDetailsOfLayer[netLayer].rankNumForTopoType.find(topoType);
    if (rankNumForTopoTypeItr == topoInfo->topoInstDetailsOfLayer[netLayer].rankNumForTopoType.end()) {
        return false;
    }

    for (auto topoRankNum : rankNumForTopoTypeItr->second) {
        if (topoRankNum == localRankSize) {
            return true;
        }
    }
    return false;
}

HcclResult AutoSelectorBase::CheckMeshNumEqualToClosNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isEqual)
{
    const auto& topoInstDetails = topoInfo->topoInstDetailsOfLayer;

    // 检查topoInstDetails是否为空
    CHK_PRT_RET(
        topoInstDetails.empty(),
        HCCL_ERROR("[BaseSelector][CheckMeshNumEqualToClosNum] topoInstDetailsOfLayer0 size is zero."),
        HCCL_E_INTERNAL);

    const auto& rankNumMap = topoInstDetails[0].rankNumForTopoType;
    auto closItr = rankNumMap.find(COMM_TOPO_CLOS);
    auto meshItr = rankNumMap.find(COMM_TOPO_1DMESH);
    CHK_PRT_RET(
        closItr == rankNumMap.end() || closItr->second.empty() || meshItr == rankNumMap.end()
            || meshItr->second.empty(),
        HCCL_ERROR("[BaseSelector][CheckMeshNumEqualToClosNum] topoInstDetailsOfLayer0 size is zero."),
        HCCL_E_INTERNAL);

    // 获取CLOS和1DMESH拓扑的rank数量并比较是否相等
    isEqual = (closItr->second[0] == meshItr->second[0]);
    return HCCL_SUCCESS;
}

HcclResult
AutoSelectorBase::CheckClosNumMultipleOfMeshNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isMultiple)
{
    const auto& topoInstDetails = topoInfo->topoInstDetailsOfLayer;
    // 检查topoInstDetails是否为空
    CHK_PRT_RET(
        topoInstDetails.empty(),
        HCCL_ERROR("[BaseSelector][CheckClosNumMultipleOfMeshNum] topoInstDetailsOfLayer0 size is zero."),
        HCCL_E_INTERNAL);

    const auto& rankNumMap = topoInstDetails[0].rankNumForTopoType;
    auto closItr = rankNumMap.find(COMM_TOPO_CLOS);
    auto meshItr = rankNumMap.find(COMM_TOPO_1DMESH);
    CHK_PRT_RET(
        closItr == rankNumMap.end() || closItr->second.empty() || meshItr == rankNumMap.end()
            || meshItr->second.empty(),
        HCCL_ERROR("[BaseSelector][CheckClosNumMultipleOfMeshNum] topoInstDetailsOfLayer0 size is zero."),
        HCCL_E_INTERNAL);

    // 获取CLOS和1DMESH拓扑的rank数量
    const auto closRankNums = closItr->second[0];
    const auto meshRankNums = meshItr->second[0];

    // 检查CLOS数量是否大于1DMESH数量且是1DMESH数量的倍数
    isMultiple = (meshRankNums > 1) && (closRankNums > meshRankNums) && (closRankNums % meshRankNums == 0);
    return HCCL_SUCCESS;
}

bool AutoSelectorBase::IsTwoLevelNetLayer(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    CHK_PRT_RET(
        topoInfo == nullptr, HCCL_WARNING("[AutoSelectorBase][IsTwoLevelNetLayer] topoInfo is nullptr."), false);
    // hostDPU场景不走二级网络算法
    bool hostDPUOnly = false;
    if ((CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) == HCCL_SUCCESS) && hostDPUOnly) {
        HCCL_INFO("[AutoSelectorBase][IsTwoLevelNetLayer] host DPU only, not two level net layer.");
        return false;
    }
    if (topoInfo->netLayerDetails.netLayerNum <= 1) {
        HCCL_INFO(
            "[AutoSelectorBase][IsTwoLevelNetLayer] netLayerNum[%u] <= 1, not two level net layer.",
            topoInfo->netLayerDetails.netLayerNum);
        return false;
    }
    u32 level1Idx = topoInfo->netLayerDetails.netLayers[1];
    bool hasLevel1Clos = topoInfo->topoInstDetailsOfLayer.size() > level1Idx
                         && topoInfo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.find(COMM_TOPO_CLOS)
                                != topoInfo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.end();
    if (!hasLevel1Clos) {
        HCCL_INFO(
            "[AutoSelectorBase][IsTwoLevelNetLayer] level1[%u] has no CLOS topo, not two level net layer.", level1Idx);
        return false;
    }
    if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.size() < 1
        || topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] <= 1) {
        HCCL_INFO(
            "[AutoSelectorBase][IsTwoLevelNetLayer] level0 localNetInsSizeOfLayer[%zu] <= 1, not two level net layer.",
            topoInfo->netLayerDetails.localNetInsSizeOfLayer.size());
        return false;
    }
    HCCL_INFO(
        "[AutoSelectorBase][IsTwoLevelNetLayer] topoLevelNums[%u], netLayerNum[%u], level0Topo[MESH_1D], "
        "level1Idx[%u] has CLOS, level0LocalNetInsSize[%u], is two level net layer.",
        topoInfo->topoLevelNums, topoInfo->netLayerDetails.netLayerNum, level1Idx,
        topoInfo->netLayerDetails.localNetInsSizeOfLayer[0]);
    return true;
}

bool AutoSelectorBase::IsDevType960()
{
    HcclDevType deviceType;
    HcclGetDeviceType(deviceType);
    return deviceType == HcclDevType::DEV_TYPE_960;
}

bool AutoSelectorBase::IsInputOutputOverlap(const OpParam& opParam) const
{
    CHK_PRT_RET(
        opParam.inputPtr == nullptr || opParam.outputPtr == nullptr,
        HCCL_INFO("[Algo][AutoSelectorBase][IsInputOutputOverlap] The input or output buffer is null. Not overlap."),
        false);

    u64 inputDataSize = opParam.inputSize;
    u64 outputDataSize = opParam.outputSize;

    CHK_PRT_RET(
        inputDataSize == 0 || outputDataSize == 0,
        // 不存在overlap情况
        HCCL_INFO("[Algo][AutoSelectorBase][IsInputOutputOverlap] The input or output buffer size is 0. Not overlap."),
        false);

    uintptr_t inputStart = reinterpret_cast<uintptr_t>(opParam.inputPtr);
    uintptr_t outputStart = reinterpret_cast<uintptr_t>(opParam.outputPtr);
    uintptr_t inputEnd = inputStart + inputDataSize - 1;
    uintptr_t outputEnd = outputStart + outputDataSize - 1;

    HCCL_DEBUG(
        "[Algo][AutoSelectorBase][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], "
        "outputEnd[%llu].",
        inputStart, inputEnd, outputStart, outputEnd);

    CHK_PRT_RET(
        inputStart <= outputEnd && outputStart <= inputEnd,
        HCCL_INFO(
            "[Algo][AutoSelectorBase][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], "
            "outputEnd[%llu]. Overlap detected.",
            inputStart, inputEnd, outputStart, outputEnd),
        true);

    HCCL_DEBUG("[Algo][AutoSelectorBase][IsInputOutputOverlap]No overlap between input and output memory.");
    return false;
}

bool AutoSelectorBase::ProcessAivConfig(
    OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName,
    SelectorStatus& ret) const
{
    if (opParam.opExecuteConfig != OpExecuteConfig::AIV && opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY) {
        return false;
    }

    if (topoInfo->topLevelUboe) {
        opParam.opExecuteConfig = OpExecuteConfig::CCU_FAIL;
        return false;
    }

    ret = SelectAivAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
    if (ret == SelectorStatus::NOT_MATCH) {
        if (opParam.opExecuteConfig == OpExecuteConfig::AIV_ONLY) {
            return true;
        }
        opParam.opExecuteConfig = OpExecuteConfig::CCU_FAIL;
        return false;
    }

    return true;
}

} // namespace ops_hccl
