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

SelectorStatus AutoSelectorBase::Select(OpParam &opParam, TopoInfoWithNetLayerDetails* topoInfo,
                                        std::string &selectAlgName) const
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
        // level0是PCIE混合的场景，且CLOS规模大于8，alltoall算子选择AIV_ONLY算法
        if (topoInfo->level0PcieMix && topoInfo->level0BigClosRange &&
            (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL ||
             opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV ||
             opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC)) {
            opParam.opExecuteConfig = OpExecuteConfig::AIV_ONLY;
            (void)ProcessAivConfig(opParam, topoInfo, configAlgMap, selectAlgName, ret);
            HCCL_INFO("[Algo][AutoSelectorBase] The selected algo is %s, OpExecuteConfig is %d.",
                selectAlgName.c_str(), opParam.opExecuteConfig);
            return ret;
        }
        ret = SelectAicpuAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        if (ret == SelectorStatus::MATCH) {
            opParam.opExecuteConfig = OpExecuteConfig::AICPU_TS;
        }
    }
    HCCL_INFO("[Algo][AutoSelectorBase] The selected algo is %s, OpExecuteConfig is %d.",
        selectAlgName.c_str(), opParam.opExecuteConfig);
    return ret;
}

bool AutoSelectorBase::IsStarsState(const OpExecuteConfig &opExecuteConfig) const
{
    return (opExecuteConfig == OpExecuteConfig::AICPU_TS ||
            opExecuteConfig == OpExecuteConfig::HOSTCPU_TS ||
            opExecuteConfig == OpExecuteConfig::CCU_FAIL);
}

bool AutoSelectorBase::IsDefaultAlg(const HcclAlgoType algoType) const
{
    return (algoType ==  HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) || (algoType ==  HcclAlgoType::HCCL_ALGO_TYPE_NA);
}

bool AutoSelectorBase::IsSmallData(const u64 dataSize) const
{
    return dataSize < SMALL_COUNT_512KB;
}

bool AutoSelectorBase::IsLargeData(const u64 dataSize) const
{
    return dataSize >= LARGE_COUNT_1024KB;
}

bool AutoSelectorBase::IsSmallDataCCU(const u64 dataSize, const u64 rankSize) const
{
    if (rankSize == 0) {
        HCCL_WARNING("the selector is not set RankSize");
    } 
    return (dataSize <= CCU_PARALLEL_MAX_DATA_SIZE) ? true : false;
}

SelectorStatus AutoSelectorBase::SelectCcuMsAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                 const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                 std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectCcuScheduleAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAicpuAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                                 const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                 std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAivAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                               const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                               std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectDPUAlgo(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam &opParam,
                                               const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                               std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

bool AutoSelectorBase::IsLayerAllConnetedWithTopo(const TopoInfoWithNetLayerDetails *topoInfo, const u32 netLayer, const CommTopo topoType) const
{
    CHK_PRT_RET(topoInfo->netLayerDetails.localNetInsSizeOfLayer.size() <= netLayer,
        HCCL_WARNING("[BaseSelector][IsLayerAllConnetedWithTopo] localNetInsSizeOfLayer size[%u] <= netLayer[%u]",
        topoInfo->netLayerDetails.localNetInsSizeOfLayer.size(), netLayer), false);
    u32 localRankSize = topoInfo->netLayerDetails.localNetInsSizeOfLayer[netLayer];

    CHK_PRT_RET(topoInfo->topoInstDetailsOfLayer.size() <= netLayer,
        HCCL_WARNING("[BaseSelector][IsLayerAllConnetedWithTopo] topoInstDetailsOfLayer size[%u] <= netLayer[%u]",
        topoInfo->topoInstDetailsOfLayer.size(), netLayer), false);

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

HcclResult AutoSelectorBase::CheckMeshNumEqualToClosNum(const TopoInfoWithNetLayerDetails *topoInfo, bool &isEqual) const
{
    const auto& topoInstDetails = topoInfo->topoInstDetailsOfLayer;

    // 检查topoInstDetails是否为空
    CHK_PRT_RET(topoInstDetails.empty(),
        HCCL_ERROR("[BaseSelector][CheckMeshNumEqualToClosNum] topoInstDetailsOfLayer0 size is zero."), HCCL_E_INTERNAL);

    const auto& rankNumMap = topoInstDetails[0].rankNumForTopoType;
    auto closItr = rankNumMap.find(COMM_TOPO_CLOS);
    auto meshItr = rankNumMap.find(COMM_TOPO_1DMESH);
    CHK_PRT_RET(closItr == rankNumMap.end() || closItr->second.empty() ||
                meshItr == rankNumMap.end() || meshItr->second.empty(),
        HCCL_ERROR("[BaseSelector][CheckMeshNumEqualToClosNum] topoInstDetailsOfLayer0 size is zero."), HCCL_E_INTERNAL);

    // 获取CLOS和1DMESH拓扑的rank数量并比较是否相等
    isEqual = (closItr->second[0] == meshItr->second[0]);
    return HCCL_SUCCESS;
}

HcclResult AutoSelectorBase::CheckClosNumMultipleOfMeshNum(const TopoInfoWithNetLayerDetails *topoInfo, bool &isMultiple) const
{
    const auto& topoInstDetails = topoInfo->topoInstDetailsOfLayer;
    // 检查topoInstDetails是否为空
    CHK_PRT_RET(topoInstDetails.empty(),
        HCCL_ERROR("[BaseSelector][CheckClosNumMultipleOfMeshNum] topoInstDetailsOfLayer0 size is zero."), HCCL_E_INTERNAL);

    const auto& rankNumMap = topoInstDetails[0].rankNumForTopoType;
    auto closItr = rankNumMap.find(COMM_TOPO_CLOS);
    auto meshItr = rankNumMap.find(COMM_TOPO_1DMESH);
    CHK_PRT_RET(closItr == rankNumMap.end() || closItr->second.empty() ||
                meshItr == rankNumMap.end() || meshItr->second.empty(),
        HCCL_ERROR("[BaseSelector][CheckClosNumMultipleOfMeshNum] topoInstDetailsOfLayer0 size is zero."), HCCL_E_INTERNAL);

    // 获取CLOS和1DMESH拓扑的rank数量
    const auto closRankNums = closItr->second[0];
    const auto meshRankNums = meshItr->second[0];

    // 检查CLOS数量是否大于1DMESH数量且是1DMESH数量的倍数
    isMultiple = (meshRankNums > 1) && (closRankNums > meshRankNums) && (closRankNums % meshRankNums == 0);
    return HCCL_SUCCESS;
}

bool AutoSelectorBase::IsTwoLevelNetLayer(const TopoInfoWithNetLayerDetails *topoInfo) const
{
    CHK_PRT_RET(topoInfo == nullptr,
        HCCL_WARNING("[AutoSelectorBase][IsTwoLevelNetLayer] topoInfo is nullptr."), false);
    if (topoInfo->netLayerDetails.netLayerNum <= 1) {
        HCCL_INFO("[AutoSelectorBase][IsTwoLevelNetLayer] netLayerNum[%u] <= 1, not two level net layer.",
            topoInfo->netLayerDetails.netLayerNum);
        return false;
    }
    u32 level1Idx = topoInfo->netLayerDetails.netLayers[1];
    bool hasLevel1Clos = topoInfo->topoInstDetailsOfLayer.size() > level1Idx &&
        topoInfo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.find(COMM_TOPO_CLOS) !=
            topoInfo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.end();
    if (!hasLevel1Clos) {
        HCCL_INFO("[AutoSelectorBase][IsTwoLevelNetLayer] level1[%u] has no CLOS topo, not two level net layer.", level1Idx);
        return false;
    }
    if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.size() < 1 ||
        topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] <= 1) {
        HCCL_INFO("[AutoSelectorBase][IsTwoLevelNetLayer] level0 localNetInsSizeOfLayer[%zu] <= 1, not two level net layer.",
            topoInfo->netLayerDetails.localNetInsSizeOfLayer.size());
        return false;
    }
    HCCL_INFO("[AutoSelectorBase][IsTwoLevelNetLayer] topoLevelNums[%u], netLayerNum[%u], level0Topo[MESH_1D], "
        "level1Idx[%u] has CLOS, level0LocalNetInsSize[%u], is two level net layer.",
        topoInfo->topoLevelNums, topoInfo->netLayerDetails.netLayerNum,
        level1Idx, topoInfo->netLayerDetails.localNetInsSizeOfLayer[0]);
    return true;
}

bool AutoSelectorBase::IsInputOutputOverlap(const OpParam &opParam) const
{
    CHK_PRT_RET(opParam.inputPtr == nullptr || opParam.outputPtr == nullptr,
        HCCL_INFO("[Algo][AutoSelectorBase][IsInputOutputOverlap] The input or output buffer is null. Not overlap."),
        false);

    u64 inputDataSize = opParam.inputSize;
    u64 outputDataSize = opParam.outputSize;

    CHK_PRT_RET(inputDataSize == 0 || outputDataSize == 0,
        // 不存在overlap情况
        HCCL_INFO("[Algo][AutoSelectorBase][IsInputOutputOverlap] The input or output buffer size is 0. Not overlap."),
        false);

    uintptr_t inputStart = reinterpret_cast<uintptr_t>(opParam.inputPtr);
    uintptr_t outputStart = reinterpret_cast<uintptr_t>(opParam.outputPtr);
    uintptr_t inputEnd = inputStart + inputDataSize - 1;
    uintptr_t outputEnd = outputStart + outputDataSize - 1;

    HCCL_DEBUG("[Algo][AutoSelectorBase][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], "
               "outputEnd[%llu].",
        inputStart, inputEnd, outputStart, outputEnd);

    CHK_PRT_RET(inputStart <= outputEnd && outputStart <= inputEnd,
        HCCL_INFO("[Algo][AutoSelectorBase][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], "
                  "outputEnd[%llu]. Overlap detected.",
            inputStart,
            inputEnd,
            outputStart,
            outputEnd),
        true);

    HCCL_DEBUG("[Algo][AutoSelectorBase][IsInputOutputOverlap]No overlap between input and output memory.");
    return false;
}

bool AutoSelectorBase::ProcessAivConfig(OpParam &opParam, TopoInfoWithNetLayerDetails* topoInfo,
                                        const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                        std::string &selectAlgName, SelectorStatus &ret) const
{
    if (opParam.opExecuteConfig != OpExecuteConfig::AIV && opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY) {
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

}