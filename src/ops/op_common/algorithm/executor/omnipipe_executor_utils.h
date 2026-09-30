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
// OmniPipe 流水线执行器公共工具。
constexpr double OMNIPIPE_FIXED_UB_UTILIZATION = 0.85;
constexpr double GBPS_TO_BYTES_PER_SECOND = 1000.0 * 1000.0 * 1000.0;

// 缺层时追加本rank；第三层还要求首个子通信域非空。是否清空输出由调用方决定。
inline void FillOmniPipeSubCommRanks(
    const AlgHierarchyInfoForAllLevel& hierarchy, u32 myRank, std::vector<std::vector<u32>>& subCommRanks0,
    std::vector<std::vector<u32>>& subCommRanks1, std::vector<std::vector<u32>>& subCommRanks2)
{
    constexpr u32 ALG_HIERARCHY_NUM2 = 2;
    constexpr u32 ALG_HIERARCHY_NUM3 = 3;
    if (hierarchy.infos.size() >= 1 && !hierarchy.infos[0].empty()) {
        subCommRanks0 = hierarchy.infos[0];
    } else {
        subCommRanks0.emplace_back(std::vector<u32>{myRank});
    }
    if (hierarchy.infos.size() >= ALG_HIERARCHY_NUM2 && !hierarchy.infos[1].empty()) {
        subCommRanks1 = hierarchy.infos[1];
    } else {
        subCommRanks1.emplace_back(std::vector<u32>{myRank});
    }
    if (hierarchy.infos.size() >= ALG_HIERARCHY_NUM3 && !hierarchy.infos[2].empty() && !hierarchy.infos[2][0].empty()) {
        subCommRanks2 = hierarchy.infos[2];
    } else {
        subCommRanks2.emplace_back(std::vector<u32>{myRank});
    }
}

struct OmniPipeCostAxes {
    u64 mesh = 1;
    u64 clos = 1;
    u64 third = 1;
};

// Cost 维度与执行路径使用同一组 TopoMatch 逻辑通信组，避免非对称物理实例在各 rank 上产生不同结果。
inline OmniPipeCostAxes CalcOmniPipeCostAxes(const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    const auto& infos = algHierarchyInfo.infos;
    OmniPipeCostAxes axes;
    axes.mesh = infos[0][0].size();
    axes.clos = infos[1][0].size();
    axes.third = infos.size() == 3 ? infos[2][0].size() : 1;
    return axes;
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
