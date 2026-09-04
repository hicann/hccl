/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_gather_omnipipe_executor.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "alg_data_trans_wrapper.h"
#include "alg_param.h"
#include "ins_temp_all_gather_omnipipe_mesh_1D.h"
#include "ins_temp_all_gather_omnipipe_nhr_dpu.h"
#include "ins_temp_all_gather_omnipipe_nhr.h"
#include "omnipipe_template_utils.h"
#include "template_utils.h"
#include "alg_attrs_registry.h"
#include "auto_selector_base.h"

namespace ops_hccl {
constexpr u32 ALG_HIERARCHY_NUM3 = 3;
constexpr u32 RANK_LEVEL_2 = 2;
constexpr u32 RANK_LEVEL_4 = 4;
namespace {
    constexpr double OMNIPIPE_FIXED_UB_UTILIZATION = 0.85;
    constexpr double GBPS_TO_BYTES_PER_SECOND = 1000.0 * 1000.0 * 1000.0;

    struct OmniPipeCostAxes {
        u64 mesh = 1;
        u64 clos = 1;
        u64 third = 1;
    };

    bool CalcOmniPipeCostAxes(const TopoInfoWithNetLayerDetails* topoInfo, OmniPipeCostAxes& axes)
    {
        if (topoInfo == nullptr || topoInfo->userRankSize == 0) {
            return false;
        }

        if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS || topoInfo->level0PcieMix) {
            if (topoInfo->topoInstDetailsOfLayer.empty()) {
                return false;
            }
            const auto& rankNumForTopoType = topoInfo->topoInstDetailsOfLayer[0].rankNumForTopoType;
            auto meshIt = rankNumForTopoType.find(CommTopo::COMM_TOPO_1DMESH);
            auto closIt = rankNumForTopoType.find(CommTopo::COMM_TOPO_CLOS);
            if (meshIt == rankNumForTopoType.end() || meshIt->second.empty() || closIt == rankNumForTopoType.end()
                || closIt->second.empty() || meshIt->second[0] == 0 || closIt->second[0] % meshIt->second[0] != 0) {
                return false;
            }
            axes.mesh = meshIt->second[0];
            axes.clos = closIt->second[0] / axes.mesh;
        } else {
            const auto& localSizes = topoInfo->netLayerDetails.localNetInsSizeOfLayer;
            if (localSizes.empty() || localSizes[0] == 0) {
                return false;
            }
            axes.mesh = localSizes[0];
            if (topoInfo->topoLevelNums > 1) {
                if (localSizes.size() < 2 || localSizes[1] < axes.mesh || localSizes[1] % axes.mesh != 0) {
                    return false;
                }
                axes.clos = localSizes[1] / axes.mesh;
            }
        }

        const u64 xyRankSize = axes.mesh * axes.clos;
        if (xyRankSize == 0 || topoInfo->userRankSize % xyRankSize != 0) {
            return false;
        }
        axes.third = topoInfo->userRankSize / xyRankSize;
        return axes.third > 0;
    }

    u64 CalcStepNumByAxes(
        double firstBandwidth, double secondBandwidth, u64 firstRankSize, u64 secondRankSize, u64 maxStepNum)
    {
        if (firstBandwidth <= secondBandwidth) {
            return CalcAllgatherStepNum2D(firstBandwidth, secondBandwidth, firstRankSize, secondRankSize, maxStepNum);
        }
        return CalcAllgatherStepNum2D(secondBandwidth, firstBandwidth, secondRankSize, firstRankSize, maxStepNum);
    }

    float CalcTemplateLatency(u32 taskNum, EngineType engine)
    {
        float latency = 0.0f;
        CostModelManager::Global()->CalcLatencyParams(taskNum, engine, latency);
        return latency;
    }

    float CalcDpuTemplateLatency(int stepNum, int syncNum, int channelNum, int sndRcvnum)
    {
        float latency = 0.0f;
        CostModelManager::Global()->CalcDpuLatencyParams(stepNum, syncNum, channelNum, sndRcvnum, latency);
        return latency;
    }
} // namespace

constexpr u32 MAX_RANK_NUM_FOR_CONCURRENT_ALGO = 4;
constexpr u64 OMNI_PCIE_AG_DATA_SIZE = 4 * 1024 * 1024; // pcie/UBX机型并行与流水算法的数据量分界，与selector保持一致

constexpr u32 DEVICE_NUM_PER_MODULE_8 = 8;
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
InsV2AllGatherOmniPipeExecutor<
    AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InsV2AllGatherOmniPipeExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::InitCommInfo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    myRank_ = topoInfo->userRank;
    rankSize_ = topoInfo->userRankSize;
    devType_ = topoInfo->deviceType;
    dataType_ = param.DataDes.dataType;
    dataCount_ = param.DataDes.count;
    opMode_ = param.opMode;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    algHierarchyInfo_ = algHierarchyInfo;
    HCCL_INFO(
        "[InsV2AllGatherOmniPipeExecutor][InitCommInfo] initialize communication metadata, "
        "rank[%u], rankSize[%u], devType[%u], dataType[%u], dataTypeSize[%u].",
        myRank_, rankSize_, devType_, dataType_, dataTypeSize_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::BuildSubCommAndTempMap(
    const OpParam& param, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    std::vector<std::vector<u32>>& subCommRanks0, std::vector<std::vector<u32>>& subCommRanks1,
    std::vector<std::vector<u32>>& subCommRanks2, std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap,
    const TopoInfoWithNetLayerDetails* topoInfo)
{
    HCCL_INFO(
        "[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap] build sub-communicators from "
        "algorithm hierarchy, hierarchy[%s].",
        ThreeDVecToStrOmni(algHierarchyInfo_.infos).c_str());
    // 统一按 infos 层级数赋值 subCommRanks（同位卡过滤已由 topoMatch 完成）
    if (algHierarchyInfo_.infos.size() >= 1 && !algHierarchyInfo_.infos[0].empty()) {
        subCommRanks0 = algHierarchyInfo_.infos[0];
    } else {
        subCommRanks0.emplace_back(std::vector<u32>{myRank_});
    }
    if (algHierarchyInfo_.infos.size() >= 2 && !algHierarchyInfo_.infos[1].empty()) {
        subCommRanks1 = algHierarchyInfo_.infos[1];
    } else {
        subCommRanks1.emplace_back(std::vector<u32>{myRank_});
    }
    if (algHierarchyInfo_.infos.size() >= 3 && !algHierarchyInfo_.infos[2].empty()
        && !algHierarchyInfo_.infos[2][0].empty()) {
        subCommRanks2 = algHierarchyInfo_.infos[2];
    } else {
        subCommRanks2.emplace_back(std::vector<u32>{myRank_});
    }
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS && !topoInfo->level0PcieMix) {
        HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][BuildSubCommAndTempMap] UBX specific optimization flags.");
        omniNeedSetStepNum_ = (subCommRanks1[0].size() == RANK_LEVEL_4) ? OmniNeedSetStepNum::OMNIPIPE_UBX_16P :
                                                                          OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
        omniUbxLastStepRead_ = true;
        if (subCommRanks2[0].size() > 1) {
            omniUbxLastStepRead_ = false;
            omniNeedSetStepNum_ = OmniNeedSetStepNum::OMNIPIPE_UBX_32P;
        }
    }
    rankSizeLevel_[OMNIPIPE_LEVEL0] = subCommRanks0[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL1] = subCommRanks1[0].size();
    rankSizeLevel_[OMNIPIPE_LEVEL2] = subCommRanks2[0].size();
    tempMap.clear();
    if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
        tempMap[OMNIPIPE_LEVEL0] = std::make_shared<InsAlgTemplate0>(param, myRank_, subCommRanks0);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
        tempMap[OMNIPIPE_LEVEL1] = std::make_shared<InsAlgTemplate1>(param, myRank_, subCommRanks1);
    }
    if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
        tempMap[OMNIPIPE_LEVEL2] = std::make_shared<InsAlgTemplate2>(param, myRank_, subCommRanks2);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    (void)comm;
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, AlgAttrs{}));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
std::vector<CostModelParam>
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    if (topoInfo == nullptr || algName == nullptr) {
        HCCL_ERROR("[%s] topoInfo or algName is null.", __func__);
        return {};
    }

    OmniPipeCostAxes axes;
    if (!CalcOmniPipeCostAxes(topoInfo, axes)) {
        HCCL_WARNING("[%s] unable to derive OmniPipe axes for algName[%s].", __func__, algName);
        return {};
    }

    OmniNeedSetStepNum needSetStepNum = OmniNeedSetStepNum::OMNIPIPE_DEFAULT;
    if (axes.clos == RANK_LEVEL_4) {
        needSetStepNum = OmniNeedSetStepNum::OMNIPIPE_UBX_16P;
    }
    if (axes.third > 1) {
        needSetStepNum = OmniNeedSetStepNum::OMNIPIPE_UBX_32P;
    }

    double meshBandwidth = BW_OMNI_DEFAULT / OMNIPIPE_FIXED_UB_UTILIZATION;
    double closBandwidth = BW_OMNI_DEFAULT / OMNIPIPE_FIXED_UB_UTILIZATION;
    double thirdBandwidth = BW_OMNI_UBX_ROCE / OMNIPIPE_FIXED_UB_UTILIZATION;
    if (topoInfo->level0PcieMix) {
        if (axes.clos == RANK_LEVEL_2) {
            closBandwidth = BW_OMNI_PCIE_EIGHT_AG_CLOS / OMNIPIPE_FIXED_UB_UTILIZATION;
        } else if (axes.clos == RANK_LEVEL_4) {
            closBandwidth = BW_OMNI_PCIE_SIXTEEN_AG_CLOS / OMNIPIPE_FIXED_UB_UTILIZATION;
        }
    } else if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        closBandwidth = BW_OMNI_UBX_AG_CLOS / OMNIPIPE_FIXED_UB_UTILIZATION;
    }
    double costMeshBandwidth = meshBandwidth;
    double costClosBandwidth = closBandwidth;
    const bool useUbx2dCostBandwidth = !topoInfo->level0PcieMix && topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS
                                       && axes.mesh > 1 && axes.clos > 1 && axes.third == 1;
    if (useUbx2dCostBandwidth) {
        costMeshBandwidth = BW_OMNI_UBX_2D_COST_AG_MESH / OMNIPIPE_FIXED_UB_UTILIZATION;
        costClosBandwidth = BW_OMNI_UBX_2D_COST_AG_CLOS / OMNIPIPE_FIXED_UB_UTILIZATION;
    }

    CostModelParam costParam{};
    const u64 maxStepNum = static_cast<u64>(SetMaxStepNumOmni(needSetStepNum));
    const double meshPlanBandwidth = meshBandwidth;
    const double closPlanBandwidth = axes.clos > 1 ? closBandwidth / (axes.clos - 1) : closBandwidth;
    const u64 innerStepNum = CalcStepNumByAxes(meshPlanBandwidth, closPlanBandwidth, axes.mesh, axes.clos, maxStepNum);

    double xyBandwidth = meshPlanBandwidth;
    if (axes.mesh > 1 && axes.clos > 1) {
        if (meshPlanBandwidth <= closPlanBandwidth) {
            xyBandwidth = CalcBandwidth2D(meshPlanBandwidth, closPlanBandwidth, axes.mesh, axes.clos, maxStepNum);
        } else {
            xyBandwidth = CalcBandwidth2D(closPlanBandwidth, meshPlanBandwidth, axes.clos, axes.mesh, maxStepNum);
        }
    } else if (axes.clos > 1) {
        xyBandwidth = closPlanBandwidth;
    }

    const double thirdPlanBandwidth = axes.third > 1 ? thirdBandwidth / (axes.third - 1) : thirdBandwidth;
    const u64 outerStepNum
        = CalcStepNumByAxes(xyBandwidth, thirdPlanBandwidth, axes.mesh * axes.clos, axes.third, maxStepNum);
    const u64 xyStepNum = innerStepNum * outerStepNum;
    const u64 thirdStepNum = axes.third > 1 ? outerStepNum : 0;
    const bool innerReachesMax = innerStepNum == maxStepNum;
    const bool thirdActive = axes.third > 1;
    const bool outerReachesMax = thirdActive && outerStepNum == maxStepNum;
    const bool thirdIsOuterSlow = thirdActive && xyBandwidth > thirdPlanBandwidth;

    const bool meshActive = axes.mesh > 1;
    const bool closActive = axes.clos > 1;
    double transferCoeff = 0.0;
    if (meshActive && closActive) {
        if (innerReachesMax) {
            transferCoeff = meshPlanBandwidth <= closPlanBandwidth ? (axes.mesh - 1) / costMeshBandwidth :
                                                                     (axes.clos - 1) / costClosBandwidth;
        } else {
            transferCoeff = (topoInfo->userRankSize - 1) / (costMeshBandwidth + costClosBandwidth);
        }
    } else if (meshActive) {
        transferCoeff = (axes.mesh - 1) / costMeshBandwidth;
    } else if (closActive) {
        transferCoeff = (axes.clos - 1) / costClosBandwidth;
    }
    if (thirdActive) {
        if (outerReachesMax) {
            transferCoeff = thirdIsOuterSlow ? 1.0 / thirdPlanBandwidth : 1.0 / xyBandwidth;
        } else {
            double activeBandwidth = thirdBandwidth;
            activeBandwidth += meshActive ? meshBandwidth : 0.0;
            activeBandwidth += closActive ? closBandwidth : 0.0;
            transferCoeff = (topoInfo->userRankSize - 1) / activeBandwidth;
        }
    }
    costParam.A = static_cast<float>(transferCoeff / GBPS_TO_BYTES_PER_SECOND);

    const bool symmetricMemory = std::string(algName) == "AicpuAllGatherPipeLine";
    const float copyRatio = symmetricMemory ? 1.0f : static_cast<float>(topoInfo->userRankSize + 1);
    CostModelManager::Global()->CalcLocalCopyParams(copyRatio, EngineType::AICPU, costParam.B);

    const float meshLatency = meshActive ? CalcTemplateLatency(1, EngineType::AICPU) : 0.0f;
    const float nhrLatency
        = closActive ? CalcTemplateLatency(GetNHRStepNum(static_cast<u32>(axes.clos)), EngineType::AICPU) : 0.0f;
    const float thirdLatency
        = axes.third > 1 ? CalcDpuTemplateLatency(
                               GetNHRStepNum(static_cast<u32>(axes.third)), 2, 1, static_cast<u32>(axes.third) - 1) :
                           0.0f;

    costParam.C = 2.0f * static_cast<float>(xyStepNum) * std::max(meshLatency, nhrLatency)
                  + static_cast<float>(thirdStepNum) * thirdLatency;

    HCCL_INFO(
        "[%s] algName[%s] axes[%llu,%llu,%llu] steps[%llu,%llu] planBandwidth[%f,%f] "
        "costBandwidth[%f,%f] xyBandwidth[%f] thirdPlanBandwidth[%f] "
        "innerMax[%d] outerMax[%d] thirdIsOuterSlow[%d] transferCoeff[%f] Ufixed[%f] A[%e] B[%e] C[%e].",
        __func__, algName, axes.mesh, axes.clos, axes.third, xyStepNum, thirdStepNum, meshBandwidth, closBandwidth,
        costMeshBandwidth, costClosBandwidth, xyBandwidth, thirdPlanBandwidth, innerReachesMax, outerReachesMax,
        thirdIsOuterSlow, transferCoeff, OMNIPIPE_FIXED_UB_UTILIZATION, costParam.A, costParam.B, costParam.C);
    return {costParam};
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
AlgNetMeta
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param) const
{
    (void)topoInfo;
    AlgNetMeta meta;
    meta.netTypes = {CommTopo::COMM_TOPO_1DMESH};
    meta.groupSizes = {1};
    return meta;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 初始化一些基本成员变量
    InitCommInfo(param, topoInfo, algHierarchyInfo);
    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;
    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(BuildSubCommAndTempMap(
        param, algHierarchyInfo, subCommRanks0, subCommRanks1, subCommRanks2, tempMap, topoInfo));

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1])
                                     / rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);

    for (auto& temp : tempMap) {
        CHK_RET(CalcResLevel(comm, param, topoInfo, temp.second, resourceRequest));
    }
    HCCL_INFO(
        "[InsV2AllGatherOmniPipeExecutor][CalcRes] finish calculating template resources, "
        "templateCount[%zu].",
        tempMap.size());

    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::CalcResLevel(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    std::shared_ptr<InsAlgTemplateBase> tempAlg, AlgResourceRequest& resourceRequest) const
{
    AlgResourceRequest resReqlevel;
    CHK_RET(tempAlg->CalcRes(comm, param, topoInfo, resReqlevel));
    resourceRequest.slaveThreadNum += resReqlevel.slaveThreadNum + 1;
    resourceRequest.notifyNumOnMainThread += 1;
    resourceRequest.notifyNumPerThread.emplace_back(
        resReqlevel.notifyNumOnMainThread + 1); // temp2控制流：从流数量+主控制流
    resourceRequest.notifyNumPerThread.insert(
        resourceRequest.notifyNumPerThread.end(), resReqlevel.notifyNumPerThread.begin(),
        resReqlevel.notifyNumPerThread.end());
    // 对称路径将各层通道合并到 channels[0]，使公共资源层只发起一次 HcclChannelAcquire。
    // 对称内存句柄会随这次建链统一交换；普通路径仍按层保存通道，保持原有资源布局。
    if (!resReqlevel.channels.empty()) {
        if (param.supportSymmetricMemory) {
            if (resourceRequest.channels.empty()) {
                resourceRequest.channels.resize(1);
            }
            resourceRequest.channels[0].insert(
                resourceRequest.channels[0].end(), resReqlevel.channels[0].begin(), resReqlevel.channels[0].end());
        } else {
            resourceRequest.channels.emplace_back(resReqlevel.channels[0]);
        }
    }
    return HCCL_SUCCESS;
}

// 该函数必须按照level0、level1、level2的顺序调用
template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    PrepareResForTemplateLevel(u32 level, std::shared_ptr<InsAlgTemplateBase>& tempBase)
{
    u32 levelThreadNum = tempBase->GetThreadNum();
    if (level == OMNIPIPE_LEVEL0) {
        levelThreads_[OMNIPIPE_LEVEL0].assign(threads_.begin() + 1, threads_.begin() + 1 + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL0].at(0));
    } else if (level == OMNIPIPE_LEVEL1) {
        levelThreads_[OMNIPIPE_LEVEL1].assign(
            threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size(),
            threads_.begin() + 1 + levelThreads_[0].size() + levelThreadNum);
        tempMainThreadsXY_.push_back(levelThreads_[OMNIPIPE_LEVEL1].at(0));
    } else if (level == OMNIPIPE_LEVEL2) {
        levelThreads_[OMNIPIPE_LEVEL2].assign(
            threads_.begin() + 1 + levelThreads_[OMNIPIPE_LEVEL0].size() + levelThreads_[OMNIPIPE_LEVEL1].size(),
            threads_.end());
        tempMainThreadsZ_.push_back(levelThreads_[OMNIPIPE_LEVEL2].at(0));
    }

    // 获取当前template各自的主thread上有多少notify
    AlgResourceRequest levelTempRequest;
    CHK_RET(tempBase->GetRes(levelTempRequest));
    if (level < OMNIPIPE_LEVEL2) {
        ntfIdxCtrlToTempXY_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlXY_.push_back(tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
    } else {
        ntfIdxCtrlToTempZ_.push_back(levelTempRequest.notifyNumOnMainThread);
        ntfIdxTempToCtrlZ_.push_back(tempMainThreadsXY_.size() + tempMainThreadsZ_.size() - 1);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    dataCount_ = param.DataDes.count;
    dataTypeSize_ = HCCL_SIZE_TABLE[param.DataDes.dataType];
    dataSize_ = dataCount_ * dataTypeSize_;
    dataType_ = param.DataDes.dataType;
    reduceOp_ = param.reduceType;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    maxTmpMemSize_ = resCtx.cclMem.size; // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn

    // 计算subCommRanks
    std::vector<std::vector<u32>> subCommRanks0;
    std::vector<std::vector<u32>> subCommRanks1;
    std::vector<std::vector<u32>> subCommRanks2;
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>> tempMap;

    rankSizeLevel_.resize(OMNIPIPE_LEVEL_NUM);
    rankIdxLevel_.resize(OMNIPIPE_LEVEL_NUM);

    CHK_RET(BuildSubCommAndTempMap(
        param, algHierarchyInfo_, subCommRanks0, subCommRanks1, subCommRanks2, tempMap, &resCtx.topoInfo));

    rankIdxLevel_[OMNIPIPE_LEVEL0] = myRank_ % rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL1] = myRank_ % (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1])
                                     / rankSizeLevel_[OMNIPIPE_LEVEL0];
    rankIdxLevel_[OMNIPIPE_LEVEL2] = myRank_ / (rankSizeLevel_[OMNIPIPE_LEVEL0] * rankSizeLevel_[OMNIPIPE_LEVEL1]);

    // 为temp分配thread
    threads_ = resCtx.threads;
    controlThread_ = threads_.at(0);
    levelThreads_.resize(OMNIPIPE_LEVEL_NUM);

    // 对称路径的建链结果扁平存入 channels[0]，普通路径仍按层保存；遍历全部集合后，
    // 根据本 rank 与对端 rank 所属的子通信域重新归层，可同时兼容两种资源布局。
    const std::vector<const std::vector<std::vector<u32>>*> subCommsByLevel
        = {&subCommRanks0, &subCommRanks1, &subCommRanks2};
    CHK_RET(ClassifyOmniPipeChannelsByLevel(
        myRank_, resCtx.channels, subCommsByLevel, rankSizeLevel_, remoteRankToChannelInfo_));
    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS && !resCtx.topoInfo.level0PcieMix) {
        if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
            tempMap[OMNIPIPE_LEVEL1]->SetchannelsPerRank(remoteRankToChannelInfo_[1]);
        }
    }
    for (auto& temp : tempMap) {
        CHK_RET(PrepareResForTemplateLevel(temp.first, temp.second));
    }
    // 算法展开
    HcclResult ret = OrchestrateLoop(param, resCtx, tempMap);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[InsV2AllGatherOmniPipeExecutor][Orchestrate] all-gather execution failed, "
            "rank[%u], errorCode[0x%016llx].",
            myRank_, HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::
    GenTemplateAlgParamsByDimData(TemplateDataParams& tempAlgParams, StepSliceInfo& stepSliceInfo) const
{
    CHK_RET(FillOmniPipeTemplateAlgParams(tempAlgParams, stepSliceInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::OrchestrateLoop(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx,
    std::map<u32, std::shared_ptr<InsAlgTemplateBase>>& tempMap)
{
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop] start all-gather pipeline loops, rank[%u].", myRank_);
    // 带宽赋值
    double bw_ag_l0 = BW_OMNI_DEFAULT;
    double bw_ag_l1 = BW_OMNI_DEFAULT;
    double bw_ag_l2 = BW_OMNI_UBX_ROCE;

    if (resCtx.topoInfo.level0PcieMix) { // PCIE
        if (rankSizeLevel_[OMNIPIPE_LEVEL1] == RANK_LEVEL_2) {
            bw_ag_l1 = BW_OMNI_PCIE_EIGHT_CLOS;
        } else if (rankSizeLevel_[OMNIPIPE_LEVEL1] == RANK_LEVEL_4) {
            bw_ag_l1 = BW_OMNI_PCIE_SIXTEEN_CLOS;
        }
        // UBX
    } else if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        bw_ag_l1 = BW_OMNI_UBX_AG_CLOS;
    }
    std::vector<double> endpointAttrBw{bw_ag_l0, bw_ag_l1, bw_ag_l2};

    // 计算等价带宽
    double eqBw0 = endpointAttrBw[0]; // L0 mesh
    double eqBw1 = endpointAttrBw[1]; // L1 NHR
    double eqBw2 = endpointAttrBw[2]; // L2 NHR

    HCCL_DEBUG(
        "[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop] initialize per-level equivalent "
        "bandwidth, level0[%f], level1[%f], level2[%f].",
        eqBw0, eqBw1, eqBw2);

    // level0为mesh,等价mesh为其本身
    // level1为nhr
    // level2, ranksize = 1
    eqBw1 = rankSizeLevel_[OMNIPIPE_LEVEL1] > 1 ? eqBw1 / (rankSizeLevel_[OMNIPIPE_LEVEL1] - 1) : eqBw1;
    eqBw2 = rankSizeLevel_[OMNIPIPE_LEVEL2] > 1 ? eqBw2 / (rankSizeLevel_[OMNIPIPE_LEVEL2] - 1) : eqBw2;

    std::vector<double> endpointAttrBwNew{eqBw0, eqBw1, eqBw2};
    u64 scratchBoundDataSize = maxTmpMemSize_ / rankSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / dataTypeSize_;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    u64 maxCountPerLoop = std::min(scratchBoundDataSize, transportBoundDataSize);
    // 对称路径不占用紧凑的 ccl scratch，按完整 user output 布局将 dataCount_ 作为单个 executor loop。
    if (param.supportSymmetricMemory && dataCount_ > 0) {
        maxCountPerLoop = dataCount_;
    }
    CHK_PRT_RET(
        maxCountPerLoop == 0,
        HCCL_ERROR(
            "[%s] maxCountPerLoop is 0, maxTmpMemSize_[%llu], rankSize_[%u], dataTypeSize_[%llu]", __func__,
            maxTmpMemSize_, rankSize_, dataTypeSize_),
        HCCL_E_INTERNAL);
    u64 loopTimes = dataCount_ / maxCountPerLoop + static_cast<u64>(dataCount_ % maxCountPerLoop != 0);

    u64 perLoopSize = maxCountPerLoop * dataTypeSize_;
    std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
    std::vector<u64> dataWholeSize(rankSize_, perLoopSize);

    for (int i = 0; i < rankSize_; i++) {
        dataSizePerLoop.push_back(perLoopSize);
        dataWholeSize.push_back(perLoopSize);
    }

    OmniPipeSliceParam omniPipeSliceParam;
    omniPipeSliceParam.levelRankSize
        = {rankSizeLevel_[OMNIPIPE_LEVEL0], rankSizeLevel_[OMNIPIPE_LEVEL1], rankSizeLevel_[OMNIPIPE_LEVEL2]};
    omniPipeSliceParam.endpointAttrBw = endpointAttrBwNew;
    omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
    omniPipeSliceParam.dataTypeSize = dataTypeSize_;
    omniPipeSliceParam.levelRankId
        = {rankIdxLevel_[OMNIPIPE_LEVEL0], rankIdxLevel_[OMNIPIPE_LEVEL1], rankIdxLevel_[OMNIPIPE_LEVEL2]};
    omniPipeSliceParam.opMode = opMode_;
    omniPipeSliceParam.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    omniPipeSliceParam.dataWholeSize = dataWholeSize;
    omniPipeSliceParam.needSetStepNum = omniNeedSetStepNum_;
    if (resCtx.topoInfo.level0PcieMix
        && param.opConfig.multipleDimensionSplitRatioSource != MultipleDimensionSplitRatioSource::BUILTIN_FORMULA) {
        omniPipeSliceParam.multipleDimensionSplitRatio = param.opConfig.multipleDimensionSplitRatio;
    }

    OmniPipeSliceInfo alignSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);

    // localcopy使用
    OmniPipeSliceParam localcopySliceParam;
    localcopySliceParam.levelRankSize
        = {rankSizeLevel_[OMNIPIPE_LEVEL0], rankSizeLevel_[OMNIPIPE_LEVEL1], rankSizeLevel_[OMNIPIPE_LEVEL2]};
    localcopySliceParam.endpointAttrBw = endpointAttrBwNew;
    localcopySliceParam.dataSizePerLoop = dataSizePerLoop;
    localcopySliceParam.dataTypeSize = dataTypeSize_;
    localcopySliceParam.levelRankId
        = {rankIdxLevel_[OMNIPIPE_LEVEL0], rankIdxLevel_[OMNIPIPE_LEVEL1], rankIdxLevel_[OMNIPIPE_LEVEL2]};
    localcopySliceParam.opMode = opMode_;
    localcopySliceParam.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    std::vector<u64> dataWholeSizeLocalcopy(rankSize_, dataSize_);
    localcopySliceParam.dataWholeSize = dataWholeSizeLocalcopy; // 这里用整体数据量算一遍
    localcopySliceParam.needSetStepNum = omniNeedSetStepNum_;
    if (resCtx.topoInfo.level0PcieMix
        && param.opConfig.multipleDimensionSplitRatioSource != MultipleDimensionSplitRatioSource::BUILTIN_FORMULA) {
        localcopySliceParam.multipleDimensionSplitRatio = param.opConfig.multipleDimensionSplitRatio;
    }

    OmniPipeSliceInfo localcopySliceInfo = CalcAGOmniPipeSliceInfo(localcopySliceParam);

    // 4、计算第n次的loop的slice信息
    OmniPipeSliceInfo tailSliceInfo;
    OmniPipeSliceInfo localcopyTailSliceInfo;
    if (dataCount_ % maxCountPerLoop != 0) {
        u64 perLoopSize = (dataCount_ % maxCountPerLoop) * dataTypeSize_;
        std::vector<u64> dataSizePerLoop(rankSize_, perLoopSize);
        std::vector<u64> dataWholeSize(rankSize_, perLoopSize);
        omniPipeSliceParam.dataSizePerLoop = dataSizePerLoop;
        omniPipeSliceParam.dataWholeSize = dataWholeSize;
        tailSliceInfo = CalcAGOmniPipeSliceInfo(omniPipeSliceParam);
        // // 尾块也一样
        localcopySliceParam.dataSizePerLoop = dataSizePerLoop;
        localcopySliceParam.dataWholeSize = dataWholeSizeLocalcopy;
        localcopyTailSliceInfo = CalcAGOmniPipeSliceInfo(localcopySliceParam);
    }

    u64 processedDataCount = 0;
    OmniPipeSliceInfo omniPipeSliceInfo;
    OmniPipeSliceInfo omniPipeSliceLocalcopyInfo;

    std::map<u32, TemplateResource> tempResMap;
    std::map<u32, TemplateDataParams> tempAlgParamMap;

    for (auto& temp : tempMap) {
        tempResMap[temp.first].channels = remoteRankToChannelInfo_[temp.first];
        tempResMap[temp.first].threads = levelThreads_[temp.first];
        tempAlgParamMap[temp.first].buffInfo.hcclBuff = resCtx.cclMem;
        tempResMap[temp.first].npu2DpuShmemPtr = resCtx.npu2DpuShmemPtr;
        tempResMap[temp.first].dpu2NpuShmemPtr = resCtx.dpu2NpuShmemPtr;
        // 下发用户输入输出地址和对称内存开关，模板据此选择 ccl scratch 或 user output 数据面。
        tempAlgParamMap[temp.first].buffInfo.inputPtr = param.inputPtr;
        tempAlgParamMap[temp.first].buffInfo.outputPtr = param.outputPtr;
        tempAlgParamMap[temp.first].enableRemoteMemAccess = param.supportSymmetricMemory;
    }
    HCCL_DEBUG(
        "[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop] split operation into executor loops, "
        "loopCount[%llu], maxCountPerLoop[%llu], symmetric[%d].",
        loopTimes, maxCountPerLoop, param.supportSymmetricMemory);
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxCountPerLoop;
        DataSlice src(param.inputPtr, processedDataCount * dataTypeSize_, currDataCount * dataTypeSize_, currDataCount);
        // 对称路径先把本 rank 输入放入 user output 的本 rank 分片，后续直接在各 rank 的 output
        // 对称窗口间交换；普通路径仍放入 ccl scratch 的本 rank 紧凑分片。
        void* initDstPtr = param.supportSymmetricMemory ? param.outputPtr : resCtx.cclMem.addr;
        u64 initDstOffset = param.supportSymmetricMemory ? (myRank_ * dataCount_ + processedDataCount) * dataTypeSize_ :
                                                           myRank_ * currDataCount * dataTypeSize_;
        DataSlice dst(initDstPtr, initDstOffset, currDataCount * dataTypeSize_, currDataCount);
        CHK_RET(LocalCopy(controlThread_, src, dst));

        if (loop == loopTimes - 1 && dataCount_ % maxCountPerLoop != 0) {
            omniPipeSliceInfo = tailSliceInfo;
            omniPipeSliceLocalcopyInfo = localcopyTailSliceInfo;
        } else {
            omniPipeSliceInfo = alignSliceInfo;
            omniPipeSliceLocalcopyInfo = localcopySliceInfo;
        }

        CHK_PRT_RET(
            omniPipeSliceInfo.dataSliceLevel2.size() == 0,
            HCCL_ERROR(
                "[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop] level-2 slice plan is "
                "empty, rank[%u], loop[%llu].",
                myRank_, loop),
            HCCL_E_PARA);

        u32 level2StepCount = omniPipeSliceInfo.dataSliceLevel2.size();
        u32 level0StepCount = omniPipeSliceInfo.dataSliceLevel0.size() / omniPipeSliceInfo.dataSliceLevel2.size();

        for (int i = 0; i < level2StepCount; i++) {
            if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
                CHK_RET(GenTemplateAlgParamsByDimData(
                    tempAlgParamMap[OMNIPIPE_LEVEL2], omniPipeSliceInfo.dataSliceLevel2[i]));
                // 对称模板按 user output 完整布局寻址：目标分片描述布局，processedDataCount 推进 loop 偏移。
                // 普通路径不会读取这两个字段。
                tempAlgParamMap[OMNIPIPE_LEVEL2].omniReadDstStepSliceInfo
                    = omniPipeSliceLocalcopyInfo.dataSliceLevel2[i];
                tempAlgParamMap[OMNIPIPE_LEVEL2].processedDataCount = processedDataCount;
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxCtrlToTempZ_));
            }
            for (int j = 0; j < level0StepCount; j++) {
                CHK_RET(PreSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxCtrlToTempXY_));
                // 对称路径每一步都直接写入 user output，不使用末步读和 ccl scratch 中转。
                if (omniUbxLastStepRead_ == true && j == level0StepCount - 1 && !param.supportSymmetricMemory) {
                    tempAlgParamMap[OMNIPIPE_LEVEL0].omniLastStepRead_ = true;
                    tempAlgParamMap[OMNIPIPE_LEVEL0].omniReadDstStepSliceInfo
                        = omniPipeSliceLocalcopyInfo.dataSliceLevel0[i * level0StepCount + j];
                    tempAlgParamMap[OMNIPIPE_LEVEL0].processedDataCount = processedDataCount;
                    tempAlgParamMap[OMNIPIPE_LEVEL1].omniLastStepRead_ = true;
                    tempAlgParamMap[OMNIPIPE_LEVEL1].omniReadDstStepSliceInfo
                        = omniPipeSliceLocalcopyInfo.dataSliceLevel1[i * level0StepCount + j];
                    tempAlgParamMap[OMNIPIPE_LEVEL1].processedDataCount = processedDataCount;
                } else {
                    tempAlgParamMap[OMNIPIPE_LEVEL0].omniLastStepRead_ = false;
                    tempAlgParamMap[OMNIPIPE_LEVEL0].omniReadDstStepSliceInfo
                        = omniPipeSliceLocalcopyInfo.dataSliceLevel0[i * level0StepCount + j];
                    tempAlgParamMap[OMNIPIPE_LEVEL0].processedDataCount = processedDataCount;
                    tempAlgParamMap[OMNIPIPE_LEVEL1].omniLastStepRead_ = false;
                    tempAlgParamMap[OMNIPIPE_LEVEL1].omniReadDstStepSliceInfo
                        = omniPipeSliceLocalcopyInfo.dataSliceLevel1[i * level0StepCount + j];
                    tempAlgParamMap[OMNIPIPE_LEVEL1].processedDataCount = processedDataCount;
                }
                if (rankSizeLevel_[OMNIPIPE_LEVEL0] > 1) {
                    CHK_RET(GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[OMNIPIPE_LEVEL0], omniPipeSliceInfo.dataSliceLevel0[i * level0StepCount + j]));
                    CHK_RET(tempMap[OMNIPIPE_LEVEL0]->KernelRun(
                        param, tempAlgParamMap[OMNIPIPE_LEVEL0], tempResMap[OMNIPIPE_LEVEL0]));
                }
                if (rankSizeLevel_[OMNIPIPE_LEVEL1] > 1) {
                    CHK_RET(GenTemplateAlgParamsByDimData(
                        tempAlgParamMap[OMNIPIPE_LEVEL1], omniPipeSliceInfo.dataSliceLevel1[i * level0StepCount + j]));
                    CHK_RET(tempMap[OMNIPIPE_LEVEL1]->KernelRun(
                        param, tempAlgParamMap[OMNIPIPE_LEVEL1], tempResMap[OMNIPIPE_LEVEL1]));
                }
                // UBX 普通内存路径从第二步开始回拷上一步接收的数据，并与当前通信步骤并行执行。
                // 对称路径的上一步结果已经位于 user output，无需补做中间本地拷贝。
                if (omniUbxLastStepRead_ && j != 0 && !param.supportSymmetricMemory) {
                    CHK_RET(UbxLastStepLocalCopy(
                        param, omniPipeSliceInfo, omniPipeSliceLocalcopyInfo, tempAlgParamMap, processedDataCount, j));
                }
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsXY_, ntfIdxTempToCtrlXY_));
            }
            if (rankSizeLevel_[OMNIPIPE_LEVEL2] > 1) {
                CHK_RET(tempMap[OMNIPIPE_LEVEL2]->KernelRun(
                    param, tempAlgParamMap[OMNIPIPE_LEVEL2], tempResMap[OMNIPIPE_LEVEL2]));
                CHK_RET(PostSyncInterThreads(controlThread_, tempMainThreadsZ_, ntfIdxTempToCtrlZ_));
            }
        }
        // 对称路径的数据已经位于 user output 对称窗口，仅普通路径需要执行最终回拷。
        if (!param.supportSymmetricMemory) {
            if (omniUbxLastStepRead_) {
                CHK_RET(UbxLocalCopy(
                    param, omniPipeSliceInfo, omniPipeSliceLocalcopyInfo, tempAlgParamMap, processedDataCount,
                    level0StepCount));
            } else {
                HCCL_INFO("ccl->out_localcopy");
                for (u32 rank = 0; rank < rankSize_; rank++) {
                    DataSlice dst(
                        param.outputPtr, (rank * dataCount_ + processedDataCount) * dataTypeSize_,
                        currDataCount * dataTypeSize_, currDataCount);
                    DataSlice src(
                        resCtx.cclMem.addr, rank * currDataCount * dataTypeSize_, currDataCount * dataTypeSize_,
                        currDataCount);
                    CHK_RET(LocalCopy(controlThread_, src, dst));
                }
            }
        }
        processedDataCount += currDataCount;
    }
    HCCL_INFO("[InsV2AllGatherOmniPipeExecutor][OrchestrateLoop] finish all-gather pipeline loops, rank[%u].", myRank_);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::RestoreChannelMap(
    const AlgResourceCtxSerializable& resCtx,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& rankIdToChannelInfo) const
{
    // 通道归层已在 Orchestrate 中通过 ClassifyOmniPipeChannelsByLevel 完成，正常路径不会调用本函数。
    // 此处仅满足基类虚函数契约；打印告警以防未来新增调用路径时静默落入基类默认归层（对称路径下结果错误）。
    HCCL_WARNING(
        "[InsV2AllGatherOmniPipeExecutor][RestoreChannelMap] unexpected call: channel classification is "
        "already done in Orchestrate, rank[%u].",
        myRank_);
    (void)resCtx;
    rankIdToChannelInfo.resize(OMNIPIPE_LEVEL_NUM);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::UbxLastStepLocalCopy(
    const OpParam& param, const OmniPipeSliceInfo& omniPipeSliceInfo,
    const OmniPipeSliceInfo& omniPipeSliceLocalcopyInfo, std::map<u32, TemplateDataParams>& tempAlgParamMap,
    const u64 processedDataCount, int step) const
{
    HCCL_DEBUG(
        "[InsV2AllGatherOmniPipeExecutor][UbxLastStepLocalCopy] copy the previous UBX step result "
        "to user output in parallel with the current step, step[%d], processedCount[%llu].",
        step, processedDataCount);
    // 做j-1这一步的localcopy 外面是每个rank遍历，里面是每个rank的多片
    for (int k = 0; k < rankSizeLevel_[OMNIPIPE_LEVEL0]; k++) {
        for (int rpt = 0; rpt < omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k].size();
             rpt++) {
            // level0的localcopy
            void* txSrcPtr0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.hcclBuff.addr;
            void* txDstPtr0 = param.outputPtr;
            u64 txBaseOff0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.inBuffBaseOff
                             + omniPipeSliceInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k][rpt];
            u64 txOffset0 = omniPipeSliceInfo.dataSliceLevel0[step - 1].stepInputSliceStride[k] + txBaseOff0;
            u64 txBaseOffDst0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.inBuffBaseOff
                                + omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k][rpt];
            u64 txOffsetDst0
                = omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepInputSliceStride[k] + txBaseOffDst0;
            txBaseOffDst0 = txOffsetDst0 + processedDataCount * dataTypeSize_;
            // src用ccl的
            DataSlice txSrcSlice0 = DataSlice(
                txSrcPtr0, txOffset0, omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepSliceSize[k][rpt],
                omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepCount[k][rpt]);
            // dst用localcopy的
            DataSlice txDstSlice0 = DataSlice(
                txDstPtr0, txBaseOffDst0, omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepSliceSize[k][rpt],
                omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepCount[k][rpt]);
            CHK_RET(LocalCopy(controlThread_, txSrcSlice0, txDstSlice0));
        }
    }
    for (int k = 0; k < rankSizeLevel_[OMNIPIPE_LEVEL1]; k++) {
        for (int rpt = 0; rpt < omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k].size();
             rpt++) {
            // level1的localcopy
            void* txSrcPtr1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.hcclBuff.addr;
            void* txDstPtr1 = param.outputPtr;
            u64 txBaseOff1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.inBuffBaseOff
                             + omniPipeSliceInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k][rpt];
            u64 txOffset1 = omniPipeSliceInfo.dataSliceLevel1[step - 1].stepInputSliceStride[k] + txBaseOff1;
            u64 txBaseOffDst1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.inBuffBaseOff
                                + omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k][rpt];
            u64 txOffsetDst1
                = omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepInputSliceStride[k] + txBaseOffDst1;
            txBaseOffDst1 = txOffsetDst1 + processedDataCount * dataTypeSize_;
            // src用ccl的
            DataSlice txSrcSlice1 = DataSlice(
                txSrcPtr1, txOffset1, omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepSliceSize[k][rpt],
                omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepCount[k][rpt]);
            // dst用localcopy的
            DataSlice txDstSlice1 = DataSlice(
                txDstPtr1, txBaseOffDst1, omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepSliceSize[k][rpt],
                omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepCount[k][rpt]);
            CHK_RET(LocalCopy(controlThread_, txSrcSlice1, txDstSlice1));
        }
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate0, typename InsAlgTemplate1, typename InsAlgTemplate2>
HcclResult
InsV2AllGatherOmniPipeExecutor<AlgTopoMatch, InsAlgTemplate0, InsAlgTemplate1, InsAlgTemplate2>::UbxLocalCopy(
    const OpParam& param, const OmniPipeSliceInfo& omniPipeSliceInfo,
    const OmniPipeSliceInfo& omniPipeSliceLocalcopyInfo, std::map<u32, TemplateDataParams>& tempAlgParamMap,
    const u64 processedDataCount, int step) const
{
    // 处理最后一步发的数据，做本地拷贝
    int k = rankIdxLevel_[OMNIPIPE_LEVEL0];
    for (int rpt = 0; rpt < omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k].size();
         rpt++) {
        // level0的localcopy
        void* txSrcPtr0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.hcclBuff.addr;
        void* txDstPtr0 = param.outputPtr;
        u64 txBaseOff0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.inBuffBaseOff
                         + omniPipeSliceInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k][rpt];
        u64 txOffset0 = omniPipeSliceInfo.dataSliceLevel0[step - 1].stepInputSliceStride[k] + txBaseOff0;
        u64 txBaseOffDst0 = tempAlgParamMap[OMNIPIPE_LEVEL0].buffInfo.inBuffBaseOff
                            + omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].inputOmniPipeSliceStride[k][rpt];
        u64 txOffsetDst0 = omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepInputSliceStride[k] + txBaseOffDst0;
        txBaseOffDst0 = txOffsetDst0 + processedDataCount * dataTypeSize_;
        // src用ccl的
        DataSlice txSrcSlice0 = DataSlice(
            txSrcPtr0, txOffset0, omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepSliceSize[k][rpt],
            omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepCount[k][rpt]);
        // dst用localcopy的
        DataSlice txDstSlice0 = DataSlice(
            txDstPtr0, txBaseOffDst0, omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepSliceSize[k][rpt],
            omniPipeSliceLocalcopyInfo.dataSliceLevel0[step - 1].stepCount[k][rpt]);
        CHK_RET(LocalCopy(controlThread_, txSrcSlice0, txDstSlice0));
    }

    k = rankIdxLevel_[OMNIPIPE_LEVEL1];
    for (int rpt = 0; rpt < omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k].size();
         rpt++) {
        // level1的localcopy
        void* txSrcPtr1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.hcclBuff.addr;
        void* txDstPtr1 = param.outputPtr;
        u64 txBaseOff1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.inBuffBaseOff
                         + omniPipeSliceInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k][rpt];
        u64 txOffset1 = omniPipeSliceInfo.dataSliceLevel1[step - 1].stepInputSliceStride[k] + txBaseOff1;
        u64 txBaseOffDst1 = tempAlgParamMap[OMNIPIPE_LEVEL1].buffInfo.inBuffBaseOff
                            + omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].inputOmniPipeSliceStride[k][rpt];
        u64 txOffsetDst1 = omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepInputSliceStride[k] + txBaseOffDst1;
        txBaseOffDst1 = txOffsetDst1 + processedDataCount * dataTypeSize_;
        // src用ccl的
        DataSlice txSrcSlice1 = DataSlice(
            txSrcPtr1, txOffset1, omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepSliceSize[k][rpt],
            omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepCount[k][rpt]);
        // dst用localcopy的
        DataSlice txDstSlice1 = DataSlice(
            txDstPtr1, txBaseOffDst1, omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepSliceSize[k][rpt],
            omniPipeSliceLocalcopyInfo.dataSliceLevel1[step - 1].stepCount[k][rpt]);
        CHK_RET(LocalCopy(controlThread_, txSrcSlice1, txDstSlice1));
    }
    return HCCL_SUCCESS;
}

// 2级算法: TopoMatchTwoLevel 产出 2 级 infos，3 个模板中 L2 模板不执行（subCommRanks2 退化为单卡）
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherPipeLineMeshNHR, InsV2AllGatherOmniPipeExecutor, TopoMatchTwoLevel,
    InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);
REGISTER_ALG_ATTRS(
    AicpuAllGatherPipeLineMeshNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
    topo.isSupportLevel0PcieMix = true; topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        bool isEqual = false;
        bool isMultiple = false;
        AutoSelectorBase::CheckMeshNumEqualToClosNum(topo, isEqual);
        AutoSelectorBase::CheckClosNumMultipleOfMeshNum(topo, isMultiple);
        return (topo->level0PcieMix
                && !AutoSelectorBase::IsLayerAllConnetedWithTopo(topo, 0, CommTopo::COMM_TOPO_1DMESH))
               || (!(isEqual && topo->userRankSize <= MAX_RANK_NUM_FOR_CONCURRENT_ALGO) && isMultiple);
    });
// 3级算法: TopoMatchThreeLevel 产出 3 级 infos，3 个模板全部执行
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_ALLGATHER, AicpuAllGatherPipeLineMeshNHRNHR, InsV2AllGatherOmniPipeExecutor,
    TopoMatchThreeLevel, InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHR);
REGISTER_ALG_ATTRS(
    AicpuAllGatherPipeLineMeshNHRNHR, topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return topo->topLevelUboe && topo->level0Symmetric && topo->level1Symmetric
               && topo->deviceNumPerModule == DEVICE_NUM_PER_MODULE_8;
    });
// 3级算法: HostDPU 场景，L2 使用 DPU 专用模板
REGISTER_EXEC_V2_MULTI(
    HcclCMDType::HCCL_CMD_ALLGATHER, DpuAllGatherPipeLineMeshNHRNHR, InsV2AllGatherOmniPipeExecutor,
    TopoMatchThreeLevel, InsTempAllGatherOmniPipeMesh1D, InsTempAllGatherOmniPipeNHR, InsTempAllGatherOmniPipeNHRDPU);
REGISTER_ALG_ATTRS(
    DpuAllGatherPipeLineMeshNHRNHR, topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D_CLOS;
    topo.topoPriorityCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !topo->level0PcieMix;
    });
} // namespace ops_hccl
