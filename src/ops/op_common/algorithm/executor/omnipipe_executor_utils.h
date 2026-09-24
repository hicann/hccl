/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OMNIPIPE_EXECUTOR_UTILS_H
#define OMNIPIPE_EXECUTOR_UTILS_H

#include "executor_v2_base.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {
/* omnipipe流水线执行器公共工具: 原先在 all_reduce/all_gather/reduce_scatter 三个
 * omnipipe executor 的匿名命名空间中逐字重复的helper, 收敛至此, 行为不变。 */
constexpr u32 MIN_NET_LAYER_NUM = 2;
constexpr double OMNIPIPE_FIXED_UB_UTILIZATION = 0.85;
constexpr double GBPS_TO_BYTES_PER_SECOND = 1000.0 * 1000.0 * 1000.0;

struct OmniPipeCostAxes {
    u64 mesh = 1;
    u64 clos = 1;
    u64 third = 1;
};

// 解析mesh/clos/third三维层级; 拓扑形态不满足整除约束时返回false
inline bool CalcOmniPipeCostAxes(const TopoInfoWithNetLayerDetails* topoInfo, OmniPipeCostAxes& axes)
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
            if (localSizes.size() < MIN_NET_LAYER_NUM || localSizes[1] < axes.mesh || localSizes[1] % axes.mesh != 0) {
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

/* 纯2D(CLOS)形态的层级解析: 与三维版CalcOmniPipeCostAxes不同, 本版本仅走
 * topoInstDetailsOfLayer路径且要求mesh*clos==userRankSize(无third维)。原先在
 * reduce/all_reduce_2d/all_gather_2d/reduce_scatter_2d/scatter_2d/broadcast_2d
 * 六个omnipipe executor的匿名命名空间中逐字重复, 收敛至此, 行为不变。 */
inline bool CalcOmniPipe2dCostAxes(const TopoInfoWithNetLayerDetails* topoInfo, u64& meshRankSize, u64& closRankSize)
{
    if (topoInfo == nullptr || topoInfo->topoInstDetailsOfLayer.empty()) {
        return false;
    }
    const auto& rankNumForTopoType = topoInfo->topoInstDetailsOfLayer[0].rankNumForTopoType;
    auto meshIt = rankNumForTopoType.find(CommTopo::COMM_TOPO_1DMESH);
    auto closIt = rankNumForTopoType.find(CommTopo::COMM_TOPO_CLOS);
    if (meshIt == rankNumForTopoType.end() || meshIt->second.empty() || closIt == rankNumForTopoType.end()
        || closIt->second.empty() || meshIt->second[0] == 0 || closIt->second[0] % meshIt->second[0] != 0) {
        return false;
    }
    meshRankSize = meshIt->second[0];
    closRankSize = closIt->second[0] / meshRankSize;
    return closRankSize > 0 && meshRankSize * closRankSize == topoInfo->userRankSize;
}

// 统一的二维步数计算: 慢链路在前, isReduceScatter选择RS/AG步数模型
inline u64 CalcStepNumByAxes(
    double firstBandwidth, double secondBandwidth, u64 firstRankSize, u64 secondRankSize, u64 maxStepNum,
    bool isReduceScatter)
{
    const bool firstIsSlow = firstBandwidth <= secondBandwidth;
    const double slowBandwidth = firstIsSlow ? firstBandwidth : secondBandwidth;
    const double fastBandwidth = firstIsSlow ? secondBandwidth : firstBandwidth;
    const u64 slowRankSize = firstIsSlow ? firstRankSize : secondRankSize;
    const u64 fastRankSize = firstIsSlow ? secondRankSize : firstRankSize;
    return isReduceScatter ?
               CalcReducescatterStepNum2D(slowBandwidth, fastBandwidth, slowRankSize, fastRankSize, maxStepNum) :
               CalcAllgatherStepNum2D(slowBandwidth, fastBandwidth, slowRankSize, fastRankSize, maxStepNum);
}

inline float CalcTemplateLatency(u32 taskNum, EngineType engine)
{
    float latency = 0.0f;
    CostModelManager::Global()->CalcLatencyParams(taskNum, engine, latency);
    return latency;
}

inline float CalcDpuTemplateLatency(int stepNum, int syncNum, int channelNum, int sndRcvnum)
{
    float latency = 0.0f;
    CostModelManager::Global()->CalcDpuLatencyParams(stepNum, syncNum, channelNum, sndRcvnum, latency);
    return latency;
}

/* omnipipe流水线执行器公共基类: 收敛 all_reduce_omnipipe / reduce_omnipipe_3d 等执行器
 * 完全相同的层级与线程管理成员, 派生类成员访问名保持不变, .cc实现无需改动。 */
class InsV2OmniPipeExecutorBase : public InsCollAlgBase {
public:
    InsV2OmniPipeExecutorBase() = default;
    ~InsV2OmniPipeExecutorBase() override = default;

protected:
    enum OmnipipeARLevel {
        OMNIPIPE_RS_LEVEL0 = 0,
        OMNIPIPE_RS_LEVEL1 = 1,
        OMNIPIPE_RS_LEVEL2 = 2,
        OMNIPIPE_AG_LEVEL0 = 3,
        OMNIPIPE_AG_LEVEL1 = 4,
        OMNIPIPE_AG_LEVEL2 = 5,
        OMNIPIPE_AR_LEVEL_NUM = 6
    };

    uint64_t rankSizeLevel0_{0};
    uint64_t rankSizeLevel1_{0};
    uint64_t rankSizeLevel2_{0};

    uint64_t rankIdxLevel0_{0};
    uint64_t rankIdxLevel1_{0};
    uint64_t rankIdxLevel2_{0};

    AlgHierarchyInfoForAllLevel algHierarchyInfo_;
    std::vector<std::map<u32, std::vector<ChannelInfo>>> remoteRankToChannelInfo_;
    std::vector<ThreadHandle> threads_; // 相当于之前的std::vector<InsQuePtr> tempInsQue_;

    ThreadHandle controlThread_ = 0;

    std::vector<ThreadHandle> tempMainThreadsLevel01RS_;
    std::vector<u32> ntfIdxCtrlToTempLevel01RS_;
    std::vector<u32> ntfIdxTempToCtrlLevel01RS_;

    std::vector<ThreadHandle> tempMainThreadsLevel2RS_;
    std::vector<u32> ntfIdxCtrlToTempLevel2RS_;
    std::vector<u32> ntfIdxTempToCtrlLevel2RS_;

    std::vector<std::vector<ThreadHandle>> levelThreadsRS_;
    std::vector<std::vector<ThreadHandle>> levelThreadsAG_;

    std::vector<ThreadHandle> tempMainThreadsLevel01AG_;
    std::vector<u32> ntfIdxCtrlToTempLevel01AG_;
    std::vector<u32> ntfIdxTempToCtrlLevel01AG_;
    std::vector<ThreadHandle> tempMainThreadsLevel2AG_;
    std::vector<u32> ntfIdxCtrlToTempLevel2AG_;
    std::vector<u32> ntfIdxTempToCtrlLevel2AG_;
    OmniNeedSetStepNum omniNeedSetStepNum_ = OmniNeedSetStepNum::OMNIPIPE_DEFAULT;

    std::vector<std::vector<u32>> subCommRanks0_;
    std::vector<std::vector<u32>> subCommRanks1_;
    std::vector<std::vector<u32>> subCommRanks2_;
};
} // namespace ops_hccl

#endif // OMNIPIPE_EXECUTOR_UTILS_H
