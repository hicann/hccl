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

namespace ops_hccl {

SelectorStatus AutoSelectorBase::Select(OpParam &opParam, TopoInfo* topoInfo,
                                        std::string &selectAlgName, OpExecuteConfig &opExecuteConfig)
{
    HCCL_DEBUG("[AutoSelectorBase][%s] start", __func__);
    std::map<HcclCMDType, std::vector<HcclAlgoType>> configAlgMap = GetExternalInputHcclAlgoConfigAllType();
    SelectorStatus ret = SelectorStatus::NOT_MATCH;
    bool hostDPUOnly = false;
    if ((CheckHostDPUOnly(topoInfo, opParam, hostDPUOnly) == HCCL_SUCCESS) && hostDPUOnly) {
        opExecuteConfig = OpExecuteConfig::HOSTCPU;
        opParam.engine = CommEngine::COMM_ENGINE_CPU;
        return SelectDPUAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
    }
    if (opParam.opExecuteConfig == OpExecuteConfig::CCU_MS) {
        ret = SelectCcuMsAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        if (ret == SelectorStatus::NOT_MATCH) {
            opExecuteConfig = OpExecuteConfig::CCU_SCHED;
        } else {
            opExecuteConfig = OpExecuteConfig::CCU_MS;
            return ret;
        }
    }
    if (opParam.opExecuteConfig == OpExecuteConfig::CCU_SCHED) {
        ret = SelectCcuScheduleAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        if (ret == SelectorStatus::NOT_MATCH) {
            opExecuteConfig = OpExecuteConfig::CCU_FAIL;
        } else {
            opExecuteConfig = OpExecuteConfig::CCU_SCHED;
            return ret;
        }
    }
    if (opParam.opExecuteConfig == OpExecuteConfig::AIV) {
        opParam.engine = CommEngine::COMM_ENGINE_AIV;
        ret = SelectAivAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        if (ret == SelectorStatus::NOT_MATCH) {
            opExecuteConfig = OpExecuteConfig::CCU_FAIL;
        } else {
            opExecuteConfig = OpExecuteConfig::AIV;
            return ret;
        }
    }
    if (IsStarsState(opParam.opExecuteConfig)) {
        ret = SelectAicpuAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        if (ret == SelectorStatus::MATCH) {
            if (opParam.opMode == OpMode::OPBASE) {
                opExecuteConfig = OpExecuteConfig::AICPU_TS;
            } else {
                opExecuteConfig = OpExecuteConfig::HOSTCPU_TS;
            }
        }
    }
    HCCL_INFO("[Algo][AutoSelectorBase] The selected algo is %s.", selectAlgName.c_str());
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

SelectorStatus AutoSelectorBase::SelectCcuMsAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                                 const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                 std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectCcuScheduleAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                                    const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                    std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAicpuAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                                 const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                                 std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAivAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                               const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                               std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectDPUAlgo(TopoInfo* topoInfo, OpParam &opParam,
                                               const std::map<HcclCMDType, std::vector<HcclAlgoType>> &configAlgMap,
                                               std::string &selectAlgName) const
{
    (void)topoInfo;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

// 判断通过最高一个level的网络全部没有device的可达链路，并且有host的可达链路
HcclResult AutoSelectorBase::CheckHostDPUOnly(const TopoInfo* topoInfo, const OpParam &opParam, bool &hostDPUOnly) const
{
    hostDPUOnly = false;
    HCCL_INFO("Start CheckHostDPUOnly");
    // 只有一个server，不使用DPU
    if (topoInfo->serverNum == 1) {
        HCCL_INFO("Not using hostdpu because serverNum is 1");
        return HCCL_SUCCESS;
    }

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(opParam.hcclComm, &netLayers, &netLayerNum));
    if ((netLayers == nullptr) || (netLayerNum == 0)) {
        HCCL_WARNING("HcclRankGraphGetLayers fail");
        return HCCL_E_INTERNAL;
    }

    bool hostDPU = false;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        uint32_t netLayer = netLayers[layerIdx];
        // 只校验最后一个level
        if (netLayer < (topoInfo->topoLevelNums - 1)) {
            HCCL_INFO("Skip checking layer[%u], topoLevelNums is [%u]", netLayer, topoInfo->topoLevelNums);
            continue;
        }
        uint32_t *topoInsts = nullptr;
        uint32_t topoInsNum = 0;
        CHK_RET(HcclRankGraphGetTopoInstsByLayer(opParam.hcclComm, netLayer, &topoInsts, &topoInsNum));
        if ((topoInsts == nullptr) || (topoInsNum == 0)) {
            HCCL_WARNING("HcclRankGraphGetTopoInstsByLayer fail, netLayer[%u]", netLayer);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t topoInsIdx = 0; topoInsIdx < topoInsNum; topoInsIdx++) {
            uint32_t topoInstId = topoInsts[topoInsIdx];
            HCCL_INFO("Start checking topoInstId[%u]", topoInstId);
            CommTopo topoType;
            CHK_RET(HcclRankGraphGetTopoType(opParam.hcclComm, netLayer, topoInstId, &topoType));
            if (topoType != COMM_TOPO_CLOS) {
                HCCL_INFO("Not using hostdpu because topo type is not COMM_TOPO_CLOS");
                continue;
            }
            uint32_t *ranks = nullptr;
            uint32_t rankNum = 0;
            CHK_RET(HcclRankGraphGetRanksByTopoInst(opParam.hcclComm, netLayer, topoInstId, &ranks, &rankNum));
            // 校验当前rank与其他所有rank连通
            if (rankNum != topoInfo->userRankSize) {
                HCCL_INFO("Not using hostdpu because current rank is not fully connected to all other ranks");
                continue;
            }
            uint32_t endPointNums = 0;
            CHK_RET(HcclRankGraphGetEndpointNum(opParam.hcclComm, netLayer, topoInstId, &endPointNums));
            EndpointDesc endPointDescs[endPointNums];
            CHK_RET(HcclRankGraphGetEndpointDesc(opParam.hcclComm, netLayer, topoInstId, &endPointNums, endPointDescs));
            for (uint32_t endPointIdx = 0; endPointIdx < endPointNums; endPointIdx++) {
                EndpointDesc endPointDesc = endPointDescs[endPointIdx];
                if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
                    HCCL_INFO("Not using hostdpu because there is links on device in netLayer[%u] in endPointIdx[%u]",
                        netLayer, endPointIdx);
                    return HCCL_SUCCESS;
                } else if (endPointDesc.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
                    HCCL_INFO("Found a host endPoint in netLayer[%u] endPointIdx[%u]", netLayer, endPointIdx);
                    hostDPU = true;
                }
            }
        }
    }
    if (hostDPU) {
        HCCL_INFO("Using host dpu trans.");
        hostDPUOnly = true;
    }
    return HCCL_SUCCESS;
}

}