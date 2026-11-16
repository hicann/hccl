/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rhd_scatter_step.h"
#include "utils.h"

namespace ops_hccl {

bool RhdScatterParamValid(u32 rankSize, u32 rootAlgRank) { return rankSize != 0 && rootAlgRank < rankSize; }

bool RhdTailSizeValid(u64 tailSize, u32 dataTypeSize)
{
    if (tailSize == 0) {
        return true; // 无尾约定：末 unit 全尺寸
    }
    return dataTypeSize != 0 && tailSize % dataTypeSize == 0;
}

HcclResult
GenRhdScatterSteps(u32 rankSize, u32 rootAlgRank, u32 myAlgRank, std::vector<RhdScatterStep>& steps, bool skipPost)
{
    steps.clear();
    if (!RhdScatterParamValid(rankSize, rootAlgRank)) {
        HCCL_ERROR("[RHD][GenRhdScatterSteps] invalid params: rankSize[%u] rootAlgRank[%u].", rankSize, rootAlgRank);
        return HcclResult::HCCL_E_PARA;
    }

    // part1 划分/代表跟随 root/主体有效序：单一事实来源经 CalcRhdPart1Info（含 myAlgRank 越界校验）
    RhdPart1Info info;
    CHK_RET(CalcRhdPart1Info(rankSize, myAlgRank, rootAlgRank, info));

    RhdPart1Info rootInfo;
    CHK_RET(CalcRhdPart1Info(rankSize, rootAlgRank, rootAlgRank, rootInfo));
    const u32 vRoot = rootInfo.virtualRank;

    const u32 r = info.r;
    const u32 effRankSize = info.effectiveRankSize;

    // 非代表 part1 成员不参与主体（代表承载 pair 的 2 份 unit 跑折半链，被合并方只出后处理）
    if (!info.inPart1 || info.isRepresentative) {
        const u32 vMe = info.virtualRank;
        const u32 nSteps = GetRhdStepNum(effRankSize);
        steps.reserve(nSteps + 1);
        for (u32 k = 0; k < nSteps; k++) {
            const u32 dist = effRankSize >> (k + 1);     // 折半序：P/2, P/4, …, 1
            const u32 w = vMe & ~(2 * dist - 1);         // 窗口 [w, w+2·dist)
            const u32 holder = w + (vRoot % (2 * dist)); // 本窗口的数据持有者（静态闭式）
            const u32 peerV = holder ^ dist;             // 伙伴 = XOR 距离
            bool isSend = false;
            u32 segStart = 0;
            if (vMe == holder || vMe == peerV) {
                // 发送"不含 holder 的对齐半区"（单向，holder 保留含自己的半区）；
                // 接收方 peer 与 holder 互补对侧，区段与发送方完全相同
                isSend = (vMe == holder);
                segStart = (holder < w + dist) ? (w + dist) : w;
            } else {
                continue; // 空转步不入表
            }
            RhdScatterStep step;
            step.phase = RhdScatterStep::Phase::MAIN;
            step.step = k;
            step.isSend = isSend;
            step.segStartUnit = RhdUnitStartOfVirtual(r, segStart);
            step.segEndUnit = RhdUnitEndOfVirtual(r, segStart + dist);
            step.peerAlgRank = RhdAlgRankOfVirtual(rankSize, rootAlgRank, isSend ? peerV : holder);
            steps.push_back(step);
        }
    }

    // 非 2^N 后处理：代表将不属于自己的一份（被合并方份额段）单段发送给伙伴（SHARE 模式）。
    // Scatter 前处理恒为数据面空操作（0 条传输任务），不入表。
    // skipPost：Scatter→AG 融合场景跳过 POST 步——代表的 σ 双份布局直接留在 ccl 里，
    // 由相邻 AG（置 fusedWithNextAgPhase）从同偏移续跑。
    if (!info.isPowerOfTwo && info.inPart1 && !skipPost) {
        RhdStepInfo pre;
        RhdStepInfo post;
        CHK_RET(GenRhdPrePostSteps(info, pre, post, RhdSegMode::SHARE, RhdSegMode::SHARE));
        RhdScatterStep step;
        step.phase = RhdScatterStep::Phase::POST;
        step.step = 0;
        step.isSend = info.isRepresentative;
        step.peerAlgRank = post.peer;
        step.segStartUnit = info.isRepresentative ? post.txSegStart : post.rxSegStart;
        step.segEndUnit = info.isRepresentative ? post.txSegEnd : post.rxSegEnd;
        steps.push_back(step);
    }

    HCCL_DEBUG(
        "[RHD][GenRhdScatterSteps] rankSize[%u] rootAlgRank[%u] myAlgRank[%u] r[%u] effRankSize[%u] vRoot[%u] "
        "vMe[%u] steps[%zu].",
        rankSize, rootAlgRank, myAlgRank, r, effRankSize, vRoot, info.virtualRank, steps.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult RhdCalcConnectRanks(u32 rankSize, u32 myAlgRank, std::set<u32>& connectRanks)
{
    connectRanks.clear();
    if (rankSize == 0 || myAlgRank >= rankSize) {
        HCCL_ERROR("[RHD][RhdCalcConnectRanks] invalid params: rankSize[%u] myAlgRank[%u].", rankSize, myAlgRank);
        return HcclResult::HCCL_E_PARA;
    }
    // 对端集合依赖 root 的两个因子：代表侧（pair 槽的物理映射）与 vRoot（步活跃性）。
    // root ∈ [0,p) 恰好覆盖全部组合，全 root 并集按构造覆盖运行时全部可能对端（不能只取
    // 少数代表根——vRoot≠0 的活跃步会产生代表根之外的对端）。每表 O(log p)、总 O(p·log p)，
    // 建链期一次性，可忽略
    for (u32 rootAlgRank = 0; rootAlgRank < rankSize; rootAlgRank++) {
        std::vector<RhdScatterStep> steps;
        CHK_RET(GenRhdScatterSteps(rankSize, rootAlgRank, myAlgRank, steps));
        for (const auto& step : steps) {
            connectRanks.insert(step.peerAlgRank);
        }
    }
    HCCL_DEBUG(
        "[RHD][RhdCalcConnectRanks] rankSize[%u] myAlgRank[%u] connectRanks[%zu].", rankSize, myAlgRank,
        connectRanks.size());
    return HcclResult::HCCL_SUCCESS;
}

} // namespace ops_hccl
