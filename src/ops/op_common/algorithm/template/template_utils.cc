/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "template_utils.h"
#include "utils.h"

#include <limits>
constexpr u32 DIE_NUM_1 = 1;
constexpr u32 DIE_NUM_2 = 2;
namespace ops_hccl {
constexpr double POD_PORT_GROUP_DIVISOR = 2.0; // Pod场景下端口组对半分
constexpr u32 MIN_PORT_NUM_LAYERS = 2;
constexpr double FIXED_PARALLEL_SPLIT_RATIO = 0.5;

HcclResult GetAlgRank(const u32 virtRank, const std::vector<u32>& rankIds, u32& algRank)
{
    std::vector<u32>::const_iterator topoVecIter = std::find(rankIds.begin(), rankIds.end(), virtRank);
    CHK_PRT_RET(
        topoVecIter == rankIds.end(), HCCL_ERROR("[GetAlgRank] Invalid virtual Rank!"), HcclResult::HCCL_E_PARA);
    algRank = distance(rankIds.begin(), topoVecIter);

    return HcclResult::HCCL_SUCCESS;
}

u32 GetNHRStepNum(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[NHRBase][GetStepNumInterServer] rankSize[%u] nSteps[%u]", rankSize, nSteps);

    return nSteps;
}

u32 GetRhdStepNum(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize; tmp > 1; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[RHD][GetRhdStepNum] rankSize[%u] nSteps[%u]", rankSize, nSteps);

    return nSteps;
}

// r = p − 2^⌊log₂p⌋：part1 配对数
static u32 CalcRhdPairNum(u32 rankSize)
{
    if (rankSize == 0) {
        return 0;
    }
    u32 highPow = 1;
    while (highPow <= rankSize / 2) {
        highPow *= 2;
    }
    return rankSize - highPow;
}

static inline bool RhdRepIsLow(u32 r, u32 root)
{
    return (root == INVALID_VALUE_RANKID || root >= 2 * r) ? true : (root < r);
}

static HcclResult CheckRhdInfoInvariant(const RhdPart1Info& info, const char* tag)
{
    if ((info.effectiveRankSize & (info.effectiveRankSize - 1)) != 0 || info.r >= info.effectiveRankSize) {
        HCCL_ERROR(
            "[RHD][%s] invalid info: r[%u] effectiveRankSize[%u] (P must be power of 2 and r < P).", tag, info.r,
            info.effectiveRankSize);
        return HcclResult::HCCL_E_PARA;
    }
    return HcclResult::HCCL_SUCCESS;
}

// σ：algRank → unit 起点
static inline u32 RhdUnitStartOfRankImpl(u32 r, u32 algRank)
{
    if (algRank < r) {
        return 2 * algRank;
    }
    if (algRank < 2 * r) {
        return 2 * (algRank - r) + 1;
    }
    return algRank;
}

u32 RhdUnitStartOfRank(u32 rankSize, u32 algRank)
{
    if (rankSize == 0 || algRank >= rankSize) {
        return INVALID_VALUE_RANKID;
    }
    return RhdUnitStartOfRankImpl(CalcRhdPairNum(rankSize), algRank);
}

u32 RhdAlgRankOfVirtual(u32 physRankSize, u32 root, u32 virtualRank)
{
    if (physRankSize == 0) {
        return INVALID_VALUE_RANKID;
    }
    const u32 r = CalcRhdPairNum(physRankSize);
    const u32 P = physRankSize - r;
    if (virtualRank >= P) {
        return INVALID_VALUE_RANKID;
    }
    if (virtualRank >= r) {
        return virtualRank + r; // block：恒为 v + r（2^N 时 r=0，即恒等）
    }
    return RhdRepIsLow(r, root) ? virtualRank : virtualRank + r; // 配对：取代表侧
}

HcclResult CalcRhdPart1Info(u32 rankSize, u32 myAlgRank, u32 root, RhdPart1Info& info)
{
    if (rankSize == 0 || myAlgRank >= rankSize) {
        HCCL_ERROR("[RHD][CalcRhdPart1Info] invalid params: rankSize[%u] myAlgRank[%u].", rankSize, myAlgRank);
        return HcclResult::HCCL_E_PARA;
    }
    if (root != INVALID_VALUE_RANKID && root >= rankSize) {
        HCCL_ERROR("[RHD][CalcRhdPart1Info] root[%u] is out of range, rankSize[%u].", root, rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    info = RhdPart1Info{};
    u32 r = CalcRhdPairNum(rankSize);
    info.r = r;
    info.isPowerOfTwo = (r == 0);
    info.effectiveRankSize = rankSize - r;
    info.virtualRank = (myAlgRank < 2 * r) ? (myAlgRank % r) : (myAlgRank - r);
    info.inPart1 = (myAlgRank < 2 * r);

    if (info.inPart1) {
        // 配对 (t ↔ t+r)，t = myAlgRank mod r
        u32 pairIdx = myAlgRank % r;
        bool lowSide = myAlgRank < r;
        info.pairedAlgRank = lowSide ? (pairIdx + r) : pairIdx;
        bool repIsLow = RhdRepIsLow(r, root);
        info.isRepresentative = repIsLow ? lowSide : !lowSide;
        if (info.isRepresentative) {
            info.unitStart = 2 * pairIdx; // 代表持有配对整块 [2t, 2t+2)
            info.unitCount = 2;
        } else {
            info.unitStart = RhdUnitStartOfRank(rankSize, myAlgRank);
            info.unitCount = 1;
        }
    } else {
        info.pairedAlgRank = INVALID_VALUE_RANKID;
        info.isRepresentative = false;
        info.unitStart = myAlgRank;
        info.unitCount = 1;
    }
    HCCL_DEBUG(
        "[RHD][CalcRhdPart1Info] rankSize[%u] myAlgRank[%u] root[%u] r[%u] P[%u] v[%u] inPart1[%u] "
        "isRep[%u] pair[%u] unitStart[%u] unitCount[%u].",
        rankSize, myAlgRank, root, info.r, info.effectiveRankSize, info.virtualRank, static_cast<u32>(info.inPart1),
        static_cast<u32>(info.isRepresentative), info.pairedAlgRank, info.unitStart, info.unitCount);

    return HcclResult::HCCL_SUCCESS;
}

// 虚拟对齐区段 [a, b) → unit 空间：起 = (a<r)?2a:a+r，止 = (b<=r)?2b:b+r
static inline u32 RhdUnitStartOfVirtual(u32 r, u32 a) { return (a < r) ? 2 * a : a + r; }

static inline u32 RhdUnitEndOfVirtual(u32 r, u32 b) { return (b <= r) ? 2 * b : b + r; }

HcclResult GenRhdReduceScatterSteps(const RhdPart1Info& info, std::vector<RhdStepInfo>& steps)
{
    steps.clear();
    if (info.effectiveRankSize == 0 || info.virtualRank >= info.effectiveRankSize) {
        HCCL_ERROR(
            "[RHD][GenRhdReduceScatterSteps] invalid params: virtualRank[%u] effectiveRankSize[%u].", info.virtualRank,
            info.effectiveRankSize);
        return HcclResult::HCCL_E_PARA;
    }
    CHK_RET(CheckRhdInfoInvariant(info, "GenRhdReduceScatterSteps"));

    const u32 myVirtualRank = info.virtualRank; // 单一事实来源：主体有效序取自 info
    const u32 P = info.effectiveRankSize;
    const u32 r = info.r;
    const u32 nSteps = GetRhdStepNum(P);
    steps.reserve(nSteps);
    for (u32 k = 0; k < nSteps; k++) {
        // 折半序：dist = P>>(k+1)，对端与本序同处 2·dist 窗口、互为对侧半区
        const u32 dist = P >> (k + 1);
        const u32 peer = myVirtualRank ^ dist;
        const u32 peerBase = peer & ~(dist - 1);
        const u32 myBase = myVirtualRank & ~(dist - 1);
        RhdStepInfo step;
        step.step = k;
        step.phase = RhdPhase::MAIN_RS;
        step.peer = peer; // 对端主体有效序
        step.nSlices = dist;
        // tx = 对端对齐半区（送出），rx = 己方对齐半区（保留并规约）
        step.txSegStart = RhdUnitStartOfVirtual(r, peerBase);
        step.txSegEnd = RhdUnitEndOfVirtual(r, peerBase + dist);
        step.rxSegStart = RhdUnitStartOfVirtual(r, myBase);
        step.rxSegEnd = RhdUnitEndOfVirtual(r, myBase + dist);
        steps.push_back(step);
    }
    HCCL_DEBUG(
        "[RHD][GenRhdReduceScatterSteps] myVirtualRank[%u] P[%u] r[%u] nSteps[%u].", myVirtualRank, P, r,
        static_cast<u32>(steps.size()));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GenRhdAllGatherSteps(const RhdPart1Info& info, std::vector<RhdStepInfo>& steps)
{
    steps.clear();
    if (info.effectiveRankSize == 0 || info.virtualRank >= info.effectiveRankSize) {
        HCCL_ERROR(
            "[RHD][GenRhdAllGatherSteps] invalid params: virtualRank[%u] effectiveRankSize[%u].", info.virtualRank,
            info.effectiveRankSize);
        return HcclResult::HCCL_E_PARA;
    }
    CHK_RET(CheckRhdInfoInvariant(info, "GenRhdAllGatherSteps"));

    const u32 myVirtualRank = info.virtualRank; // 单一事实来源：主体有效序取自 info
    const u32 P = info.effectiveRankSize;
    const u32 r = info.r;
    const u32 nSteps = GetRhdStepNum(P);
    steps.reserve(nSteps);
    for (u32 k = 0; k < nSteps; k++) {
        // 倍增序：dist = 1<<k，整段发送当前持有区段、接收补半区段
        const u32 dist = 1u << k;
        const u32 peer = myVirtualRank ^ dist;
        const u32 heldStart = myVirtualRank & ~(dist - 1);
        const u32 rxStart = heldStart ^ dist;
        RhdStepInfo step;
        step.step = k;
        step.phase = RhdPhase::MAIN_AG;
        step.peer = peer; // 对端主体有效序
        step.nSlices = dist;
        step.txSegStart = RhdUnitStartOfVirtual(r, heldStart);
        step.txSegEnd = RhdUnitEndOfVirtual(r, heldStart + dist);
        step.rxSegStart = RhdUnitStartOfVirtual(r, rxStart);
        step.rxSegEnd = RhdUnitEndOfVirtual(r, rxStart + dist);
        steps.push_back(step);
    }
    HCCL_DEBUG(
        "[RHD][GenRhdAllGatherSteps] myVirtualRank[%u] P[%u] r[%u] nSteps[%u].", myVirtualRank, P, r,
        static_cast<u32>(steps.size()));

    return HcclResult::HCCL_SUCCESS;
}

HcclResult GenRhdPrePostSteps(
    const RhdPart1Info& info, RhdStepInfo& pre, RhdStepInfo& post, RhdSegMode preSegMode, RhdSegMode postSegMode)
{
    pre = RhdStepInfo{};
    post = RhdStepInfo{};
    CHK_RET(CheckRhdInfoInvariant(info, "GenRhdPrePostSteps"));
    pre.phase = RhdPhase::PRE;
    post.phase = RhdPhase::POST;

    if (!info.inPart1) {
        pre.peer = INVALID_VALUE_RANKID;
        post.peer = INVALID_VALUE_RANKID;
        return HcclResult::HCCL_SUCCESS;
    }

    const u32 p = info.effectiveRankSize + info.r;
    if (info.pairedAlgRank >= p || (!info.isRepresentative && info.unitStart >= p)) {
        HCCL_ERROR(
            "[RHD][GenRhdPrePostSteps] invalid info: unitStart[%u] pairedAlgRank[%u] p[%u].", info.unitStart,
            info.pairedAlgRank, p);
        return HcclResult::HCCL_E_PARA;
    }
    const u32 segStart = info.isRepresentative ? RhdUnitStartOfRank(p, info.pairedAlgRank) : info.unitStart;
    const u32 segEnd = segStart + 1;
    // 区段：SHARE = 被合并方份额段；FULL = [0, p) 全量段（RS 族 pre 须全量：代表须含被合并方全量贡献）
    const u32 preStart = (preSegMode == RhdSegMode::FULL) ? 0 : segStart;
    const u32 preEnd = (preSegMode == RhdSegMode::FULL) ? p : segEnd;
    const u32 preSlices = (preSegMode == RhdSegMode::FULL) ? p : 1;
    const u32 postStart = (postSegMode == RhdSegMode::FULL) ? 0 : segStart;
    const u32 postEnd = (postSegMode == RhdSegMode::FULL) ? p : segEnd;
    const u32 postSlices = (postSegMode == RhdSegMode::FULL) ? p : 1;

    pre.peer = info.pairedAlgRank;
    post.peer = info.pairedAlgRank;
    if (info.isRepresentative) {
        // 代表：pre 接收被合并方区段（接收镜像），post 发送
        pre.nSlices = preSlices;
        pre.rxSegStart = preStart;
        pre.rxSegEnd = preEnd;
        post.nSlices = postSlices;
        post.txSegStart = postStart;
        post.txSegEnd = postEnd;
    } else {
        // 被合并方：pre 发送，post 反向接收
        pre.nSlices = preSlices;
        pre.txSegStart = preStart;
        pre.txSegEnd = preEnd;
        post.nSlices = postSlices;
        post.rxSegStart = postStart;
        post.rxSegEnd = postEnd;
    }
    HCCL_DEBUG(
        "[RHD][GenRhdPrePostSteps] r[%u] P[%u] myV[%u] isRep[%u] pair[%u] preSeg[%u,%u) postSeg[%u,%u).", info.r,
        info.effectiveRankSize, info.virtualRank, static_cast<u32>(info.isRepresentative), info.pairedAlgRank, preStart,
        preEnd, postStart, postEnd);

    return HcclResult::HCCL_SUCCESS;
}

RhdSegBytes RhdSegToBytesUniform(u32 unitStart, u32 unitEnd, u64 sliceSize, u64 tailSize, u32 tailUnit)
{
    RhdSegBytes bytes;
    if (unitStart >= unitEnd) {
        return bytes; // 空区段（2^N PRE/POST 步骤为常态），正常返回 {0, 0}
    }
    bytes.offset = static_cast<u64>(unitStart) * sliceSize;
    // 末 unit（tailUnit，恒为布局最后一个）长 tailSize，其余长 sliceSize；
    // tailSize == 0 视为无尾块（对齐 TemplateDataParams/HD 模板 tailSize > 0 判尾约定，直传安全）
    const u64 tailLen = (tailSize == 0) ? sliceSize : tailSize;
    const bool includesTail = unitEnd > tailUnit;
    const u32 normalUnits = unitEnd - unitStart - (includesTail ? 1 : 0);
    bytes.len = static_cast<u64>(normalUnits) * sliceSize + (includesTail ? tailLen : 0);
    return bytes;
}

HcclResult RhdSegToBytesAllRank(u32 unitStart, u32 unitEnd, const std::vector<u64>& unitSizes, RhdSegBytes& bytes)
{
    bytes = RhdSegBytes{};
    if (unitStart >= unitEnd) {
        return HcclResult::HCCL_SUCCESS; // 空区段（2^N PRE/POST 常态），{0, 0}
    }
    if (unitSizes.size() < unitEnd) {
        HCCL_ERROR(
            "[RHD][RhdSegToBytesAllRank] invalid params: unitStart[%u] unitEnd[%u] unitSizesNum[%zu].", unitStart,
            unitEnd, unitSizes.size());
        return HcclResult::HCCL_E_PARA;
    }
    for (u32 u = 0; u < unitStart; u++) {
        bytes.offset += unitSizes[u];
    }
    for (u32 u = unitStart; u < unitEnd; u++) {
        bytes.len += unitSizes[u];
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult RhdBuildUnitSizes(u32 rankSize, const std::vector<u64>& allRankSliceSize, std::vector<u64>& unitSizes)
{
    unitSizes.clear();
    if (allRankSliceSize.size() != rankSize) {
        HCCL_ERROR(
            "[RHD][RhdBuildUnitSizes] size mismatch: rankSize[%u] allRankSliceSize[%zu].", rankSize,
            allRankSliceSize.size());
        return HcclResult::HCCL_E_PARA;
    }
    unitSizes.resize(rankSize, 0);
    const u32 r = CalcRhdPairNum(rankSize); // 配对数提升到循环外（O(p) 而非 O(p·log p)）
    for (u32 j = 0; j < rankSize; j++) {
        unitSizes[RhdUnitStartOfRankImpl(r, j)] = allRankSliceSize[j]; // σ 反查：algRank 索引 → unit 槽位
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcDataSplitByPortGroupCommon(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset, const u32 channelsPerRank)
{
    elemCountOut.clear();
    sizeOut.clear();
    elemOffset.clear();

    std::vector<u32> portGroups;
    u32 totalPorts = 0;
    u32 taskCount
        = (static_cast<int>(channels.size()) > channelsPerRank) ? channelsPerRank : static_cast<int>(channels.size());
    for (u32 i = 0; i < taskCount; i++) {
        const auto& ch = channels[i];
        portGroups.push_back(ch.portGroupSize);
        totalPorts += ch.portGroupSize;
        HCCL_INFO(
            "[CalcDataSplitByPortGroup] ch.portGroupSize[%u], totalPorts[%u], channelsPerRank[%u]", ch.portGroupSize,
            totalPorts, channelsPerRank);
    }

    u32 channelsize = portGroups.size();
    u64 accumCount = 0;
    u64 offset = 0;
    for (u32 channelIdx = 0; channelIdx < channelsize; channelIdx++) {
        u64 elemCount = 0;
        u64 elemSize = 0;
        if (channelIdx == channelsize - 1) {
            elemCount = totalDataCount - accumCount;
        } else {
            CHK_PRT_RET(
                totalPorts == 0, HCCL_ERROR("[CalcDataSplitByPortGroup] totalPorts [%u] is 0.", totalPorts),
                HcclResult::HCCL_E_INTERNAL);
            elemCount = static_cast<u64>((totalDataCount * portGroups[channelIdx]) / totalPorts);
        }
        elemOffset.push_back(offset);
        elemCountOut.push_back(elemCount);
        elemSize = elemCount * dataTypeSize;
        sizeOut.push_back(elemSize);
        offset += elemSize;
        accumCount += elemCount;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CalcDataSplitByPortGroupZAxisDetour(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset,
    const u32 level0ChannelNumPerRank, const u32 level1ChannelNumPerRank, const float level0DataRatio)
{
    elemCountOut.clear();
    sizeOut.clear();
    elemOffset.clear();

    CHK_PRT_RET(
        level0DataRatio < 0.0f || level0DataRatio > 1.0f,
        HCCL_ERROR("[CalcDataSplitByPortGroupZAxisDetour] level0DataRatio[%f] is invalid.", level0DataRatio),
        HcclResult::HCCL_E_PARA);

    u64 level0DataCount;
    if (level1ChannelNumPerRank == 0) {
        level0DataCount = totalDataCount;
    } else {
        level0DataCount = static_cast<u64>(static_cast<double>(totalDataCount) * level0DataRatio);
        level0DataCount = std::min(level0DataCount, totalDataCount);
    }
    u64 level1DataCount = totalDataCount - level0DataCount;

    std::vector<ChannelInfo> level0Chs(channels.begin(), channels.begin() + level0ChannelNumPerRank);
    std::vector<u64> l0ElemCount, l0Size, l0Offset;
    CHK_RET(CalcDataSplitByPortGroupCommon(
        level0DataCount, dataTypeSize, level0Chs, l0ElemCount, l0Size, l0Offset, level0ChannelNumPerRank));

    std::vector<ChannelInfo> level1Chs(channels.begin() + level0ChannelNumPerRank, channels.end());
    std::vector<u64> l1ElemCount, l1Size, l1Offset;

    CHK_RET(CalcDataSplitByPortGroupCommon(
        level1DataCount, dataTypeSize, level1Chs, l1ElemCount, l1Size, l1Offset, level1ChannelNumPerRank));
    u64 level0TotalSize = 0;
    for (auto sz : l0Size) {
        level0TotalSize += sz;
    }
    for (auto& off : l1Offset) {
        off += level0TotalSize;
    }

    elemCountOut = l0ElemCount;
    elemCountOut.insert(elemCountOut.end(), l1ElemCount.begin(), l1ElemCount.end());
    sizeOut = l0Size;
    sizeOut.insert(sizeOut.end(), l1Size.begin(), l1Size.end());
    elemOffset = l0Offset;
    elemOffset.insert(elemOffset.end(), l1Offset.begin(), l1Offset.end());

    HCCL_INFO(
        "[CalcDataSplitByPortGroupZAxisDetour] totalDataCount[%llu], level0DataCount[%llu], "
        "level1DataCount[%llu], level0ChannelNumPerRank[%u], level1ChannelNumPerRank[%u], "
        "level0DataRatio[%f], elemCountOut.size[%zu]",
        totalDataCount, level0DataCount, level1DataCount, level0ChannelNumPerRank, level1ChannelNumPerRank,
        level0DataRatio, elemCountOut.size());

    return HcclResult::HCCL_SUCCESS;
}

// The weights fit in u32, so splitting the quotient and remainder avoids count * weight overflow.
static u64 CalcDetourSplitCount(u64 count, u64 dataTypeSize, u32 weight, u32 totalWeight)
{
    u64 splitCount = count / totalWeight * weight + count % totalWeight * weight / totalWeight;
    if (AICPU_ALIGN_SIZE % dataTypeSize == 0) {
        const u64 alignCount = AICPU_ALIGN_SIZE / dataTypeSize;
        const u64 alignedCount = splitCount / alignCount * alignCount;
        // Preserve small nonempty slices when rounding down would remove their data entirely.
        if (alignedCount != 0) {
            splitCount = alignedCount;
        }
    }
    return splitCount;
}

HcclResult CalcDataSplitByBandwidthZAxisDetour(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset,
    const u32 level0ChannelNumPerRank, const u32 level1ChannelNumPerRank, const u32 serverRankSize)
{
    elemCountOut.clear();
    sizeOut.clear();
    elemOffset.clear();

    CHK_PRT_RET(
        dataTypeSize == 0 || totalDataCount > std::numeric_limits<u64>::max() / dataTypeSize || serverRankSize == 0,
        HCCL_ERROR("[CalcDataSplitByBandwidthZAxisDetour] Invalid count, data type size or server rank size."),
        HCCL_E_PARA);
    // Detour channels are ordered as one intra-server channel followed by the inter-server channels.
    CHK_PRT_RET(
        level0ChannelNumPerRank != MESH_CHANNELS_NUM
            || static_cast<u64>(level0ChannelNumPerRank) + level1ChannelNumPerRank != channels.size(),
        HCCL_ERROR("[CalcDataSplitByBandwidthZAxisDetour] Invalid channel layout."), HCCL_E_PARA);

    // Only remote ranks in the local server contribute level0 bandwidth.
    const u32 level0Weight = serverRankSize - 1;
    u64 level1Bandwidth = 0;
    for (u32 channelIdx = level0ChannelNumPerRank; channelIdx < channels.size(); ++channelIdx) {
        const u32 ports = channels[channelIdx].portGroupSize;
        CHK_PRT_RET(
            ports == 0 || level1Bandwidth + ports > std::numeric_limits<u32>::max() - level0Weight,
            HCCL_ERROR("[CalcDataSplitByBandwidthZAxisDetour] Invalid level1 port weight."), HCCL_E_PARA);
        level1Bandwidth += ports;
    }

    u64 level0DataCount = totalDataCount;
    if (level1ChannelNumPerRank != 0) {
        const u32 totalWeight = level0Weight + static_cast<u32>(level1Bandwidth);
        level0DataCount = CalcDetourSplitCount(totalDataCount, dataTypeSize, level0Weight, totalWeight);
    }
    elemCountOut.push_back(level0DataCount);

    const u64 level1DataCount = totalDataCount - level0DataCount;
    u64 remainingCount = level1DataCount;
    for (u32 channelIdx = level0ChannelNumPerRank; channelIdx < channels.size(); ++channelIdx) {
        const u64 count = channelIdx == channels.size() - 1 ?
                              remainingCount :
                              CalcDetourSplitCount(
                                  level1DataCount, dataTypeSize, channels[channelIdx].portGroupSize,
                                  static_cast<u32>(level1Bandwidth));
        elemCountOut.push_back(count);
        remainingCount -= count;
    }

    u64 offset = 0;
    for (const u64 count : elemCountOut) {
        const u64 size = count * dataTypeSize;
        sizeOut.push_back(size);
        elemOffset.push_back(offset);
        offset += size;
    }
    HCCL_INFO(
        "[CalcDataSplitByBandwidthZAxisDetour] totalDataCount[%llu], level0DataCount[%llu], "
        "level1DataCount[%llu], serverRankSize[%u], level1Bandwidth[%llu]",
        totalDataCount, level0DataCount, level1DataCount, serverRankSize, level1Bandwidth);
    return HCCL_SUCCESS;
}

bool IsAllConnetedWithTopo(const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType)
{
    CHK_PRT_RET(
        topoInfo->netLayerDetails.localNetInsSizeOfLayer.size() <= netLayer,
        HCCL_WARNING(
            "[BaseSelector][IsLayerAllConnetedWithTopo] localNetInsSizeOfLayer size[%zu] <= netLayer[%u]",
            topoInfo->netLayerDetails.localNetInsSizeOfLayer.size(), netLayer),
        false);
    u32 localRankSize = topoInfo->netLayerDetails.localNetInsSizeOfLayer[netLayer];

    CHK_PRT_RET(
        topoInfo->topoInstDetailsOfLayer.size() <= netLayer,
        HCCL_WARNING(
            "[BaseSelector][IsLayerAllConnetedWithTopo] topoInstDetailsOfLayer size[%zu] <= netLayer[%u]",
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

bool GetPortGroupSize(const std::map<u32, std::vector<ChannelInfo>>& channels, uint64_t& portGroupSize)
{
    portGroupSize = 0;
    for (const auto& entry : channels) {
        const auto& channelGroup = entry.second;
        if (!channelGroup.empty()) {
            for (const auto& ch : channelGroup) {
                portGroupSize += ch.portGroupSize;
            }
            return true;
        }
    }
    return false;
}

// 首个非空inter Channel组恰好包含两条有效跨Die链路时，判定为POD机型。
static bool IsPodInterChannelGroup(const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    constexpr size_t podChannelNum = 2;
    for (const auto& entry : channels) {
        const auto& channelGroup = entry.second;
        if (channelGroup.empty()) {
            continue;
        }
        if (channelGroup.size() != podChannelNum) {
            return false;
        }
        const u32 firstDieId = channelGroup[0].dieId;
        const u32 secondDieId = channelGroup[1].dieId;
        return firstDieId != INVALID_VALUE_RANKID && secondDieId != INVALID_VALUE_RANKID && firstDieId != secondDieId;
    }
    return false;
}

// 内置公式所需的端口信息。interPortGroupSize保存原始端口和，
// effectiveInterPortGroupSize保存按算子模型完成POD收敛修正后的有效端口规模。
struct ParallelPortInfo {
    uint64_t intraPortGroupSize = 0;          // 机内端口和乘以(intraRankSize - 1)后的值
    uint64_t interPortGroupSize = 0;          // Server间首个非空Channel组的原始端口和
    double effectiveInterPortGroupSize = 0.0; // 按算子模型修正后的Server间有效端口规模
    bool isPod = false;                       // Server间是否为双Channel、跨Die的POD链路
};

// 两片数据在各自主导阶段处理单位数据所需的时间系数。
// mesh对应"先Mesh后Clos"的数据片，clos对应"先Clos后Mesh"的数据片；
// 数据逐级收缩的算子(ReduceScatter/Scatter类)主导阶段为第一阶段，
// 数据逐级放大的算子(AllGather)主导阶段为第二阶段，此时系数对应的物理链路与命名相反。
// Broadcast沿用Scatter的分片命名，AllGather第二阶段的物理链路与命名一致。
struct ParallelTimeCoeff {
    double mesh = 0.0; // 先Mesh后Clos数据片的主导阶段时间系数
    double clos = 0.0; // 先Clos后Mesh数据片的主导阶段时间系数
};

// 将异常回退值限制到[0, 1]；非有限值统一回退到0.5。
static double NormalizeParallelFallbackRatio(double fallbackRatio)
{
    return std::isfinite(fallbackRatio) ? std::max(0.0, std::min(fallbackRatio, 1.0)) : 0.5;
}

// 从执行侧的ChannelInfo Map中提取原始端口信息，供AICPU/DPU等资源上下文携带ChannelInfo的模式使用。
// 返回false时failureReason指向静态错误描述。
static bool ExtractParallelPortInfoFromChannels(
    const std::map<u32, std::vector<ChannelInfo>>& intraChannels,
    const std::map<u32, std::vector<ChannelInfo>>& interChannels, ParallelChannelPortInfo& rawPortInfo,
    const char*& failureReason)
{
    if (intraChannels.empty()) {
        failureReason = "intraChannels is empty";
        return false;
    }
    if (interChannels.empty()) {
        failureReason = "interChannels is empty";
        return false;
    }
    if (!GetPortGroupSize(intraChannels, rawPortInfo.intraPortGroupSize)) {
        failureReason = "no non-empty channel group in intraChannels";
        return false;
    }
    if (!GetPortGroupSize(interChannels, rawPortInfo.interPortGroupSize)) {
        failureReason = "no non-empty channel group in interChannels";
        return false;
    }
    rawPortInfo.isInterPod = IsPodInterChannelGroup(interChannels);
    rawPortInfo.isValid = true;
    return true;
}

// 校验原始端口信息，完成机内Rank扩展，并按算子模型应用POD收敛修正。
// 返回false时failureReason指向静态错误描述，portInfo保留已计算出的诊断信息。
static bool PrepareParallelPortInfo(
    uint64_t intraRankSize, uint64_t interRankSize, const ParallelChannelPortInfo& rawPortInfo,
    ParallelDataSplitType splitType, ParallelPortInfo& portInfo, const char*& failureReason)
{
    if (intraRankSize == 0) {
        failureReason = "intraRankSize is 0";
        return false;
    }
    if (interRankSize == 0) {
        failureReason = "interRankSize is 0";
        return false;
    }
    portInfo.intraPortGroupSize = rawPortInfo.intraPortGroupSize;
    portInfo.interPortGroupSize = rawPortInfo.interPortGroupSize;
    if (portInfo.intraPortGroupSize == 0) {
        failureReason = "intraPortGroupSize is 0";
        return false;
    }
    if (portInfo.interPortGroupSize == 0) {
        failureReason = "interPortGroupSize is 0";
        return false;
    }
    if (intraRankSize - 1 > std::numeric_limits<uint64_t>::max() / portInfo.intraPortGroupSize) {
        failureReason = "intraPortGroupSize scaling overflow";
        return false;
    }

    portInfo.intraPortGroupSize *= intraRankSize - 1;
    portInfo.isPod = rawPortInfo.isInterPod;
    // Scatter首阶段不做POD收敛修正；Broadcast按AllGather末阶段建模，保留2:1收敛修正。
    const bool needPodConvergence = portInfo.isPod && splitType != ParallelDataSplitType::SCATTER;
    portInfo.effectiveInterPortGroupSize
        = static_cast<double>(portInfo.interPortGroupSize) / (needPodConvergence ? POD_PORT_GROUP_DIVISOR : 1.0);
    if (portInfo.intraPortGroupSize == 0) {
        failureReason = "scaled intraPortGroupSize is 0";
        return false;
    }
    if (IsDoubleEqual(portInfo.effectiveInterPortGroupSize, 0.0)
        || !std::isfinite(portInfo.effectiveInterPortGroupSize)) {
        failureReason = "effectiveInterPortGroupSize is 0 or not finite";
        return false;
    }
    return true;
}

// 根据算子通信模型计算Mesh和Clos两侧的单位数据时间系数。
// splitType不受支持时返回false，由调用方统一执行回退和日志记录。
static bool CalcParallelTimeCoeff(
    uint64_t intraRankSize, uint64_t interRankSize, const ParallelPortInfo& portInfo, ParallelDataSplitType splitType,
    ParallelTimeCoeff& timeCoeff)
{
    switch (splitType) {
        case ParallelDataSplitType::REDUCE_SCATTER_WITH_LOCAL_REDUCE:
            timeCoeff.mesh = 21.0 * static_cast<double>(intraRankSize - 1)
                             / (20.0 * static_cast<double>(intraRankSize) * portInfo.intraPortGroupSize);
            timeCoeff.clos = static_cast<double>(interRankSize - 1)
                             / (static_cast<double>(interRankSize) * portInfo.effectiveInterPortGroupSize);
            return true;
        case ParallelDataSplitType::SCATTER:
        case ParallelDataSplitType::BROADCAST:
            // Broadcast的先Mesh数据片在AllGather第二阶段走Mesh，先Clos数据片走Clos。
            // 每rank通信量分别为r*D*(M-1)/M和(1-r)*D*(N-1)/N，与Scatter首阶段同形；
            // 两者的POD有效带宽由PrepareParallelPortInfo按各自模型修正。
            timeCoeff.mesh = static_cast<double>(intraRankSize - 1)
                             / (static_cast<double>(intraRankSize) * portInfo.intraPortGroupSize);
            timeCoeff.clos = static_cast<double>(interRankSize - 1)
                             / (static_cast<double>(interRankSize) * portInfo.effectiveInterPortGroupSize);
            return true;
        case ParallelDataSplitType::ALL_GATHER:
            // AllGather的数据逐级放大，第二阶段的通信量分别是第一阶段的intraRankSize和interRankSize倍，
            // 第二阶段为耗时主导项，因此按第二阶段配平：
            // 先Mesh后Clos的数据片在第二阶段走Server间链路，通信量放大intraRankSize倍；
            // 先Clos后Mesh的数据片在第二阶段走机内Mesh链路，通信量放大interRankSize倍。
            timeCoeff.mesh = static_cast<double>(intraRankSize) * static_cast<double>(interRankSize - 1)
                             / portInfo.effectiveInterPortGroupSize;
            timeCoeff.clos = static_cast<double>(interRankSize) * static_cast<double>(intraRankSize - 1)
                             / static_cast<double>(portInfo.intraPortGroupSize);
            return true;
        default:
            return false;
    }
}

// 根据两侧时间系数计算未量化比例，并校验分母及结果的有效性。
static bool CalcRawParallelDataSplitRatio(const ParallelTimeCoeff& timeCoeff, double& ratio, const char*& failureReason)
{
    const double denominator = timeCoeff.clos + timeCoeff.mesh;
    if (IsDoubleEqual(denominator, 0.0) || !std::isfinite(denominator)) {
        failureReason = "denominator is 0 or not finite";
        return false;
    }
    ratio = timeCoeff.clos / denominator;
    if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
        failureReason = "ratio is not finite or out of range[0,1]";
        return false;
    }
    return true;
}

// 将内置公式结果按就近原则量化到[1/8, 7/8]中的八分位候选值。
static double QuantizeParallelDataSplitRatio(double ratio)
{
    constexpr double ratioStep = 1.0 / 8.0;
    constexpr double minRatioIndex = 1.0;
    constexpr double maxRatioIndex = 7.0;
    const double nearestRatioIndex = std::round(ratio / ratioStep);
    const double clampedRatioIndex = std::max(minRatioIndex, std::min(nearestRatioIndex, maxRatioIndex));
    return clampedRatioIndex * ratioStep;
}

const char* ParallelDataSplitTypeToStr(ParallelDataSplitType splitType)
{
    switch (splitType) {
        case ParallelDataSplitType::REDUCE_SCATTER_WITH_LOCAL_REDUCE:
            return "REDUCE_SCATTER_WITH_LOCAL_REDUCE";
        case ParallelDataSplitType::SCATTER:
            return "SCATTER";
        case ParallelDataSplitType::BROADCAST:
            return "BROADCAST";
        case ParallelDataSplitType::ALL_GATHER:
            return "ALL_GATHER";
        default:
            return "UNKNOWN";
    }
}

// 统一记录公式回退原因及已提取的拓扑参数，返回规范化后的回退比例。
static double ReturnParallelDataSplitFallback(
    const char* failureReason, uint64_t intraRankSize, uint64_t interRankSize,
    const ParallelChannelPortInfo& rawPortInfo, ParallelDataSplitType splitType, double fallbackRatio)
{
    HCCL_WARNING(
        "[CalcParallelDataSplitRatio] fallback due to: %s, "
        "intraRankSize[%llu], interRankSize[%llu], "
        "intraPortGroupSize[%llu], interPortGroupSize[%llu], isInterPod[%d], "
        "splitType[%s], fallbackRatio[%f]",
        failureReason, intraRankSize, interRankSize, rawPortInfo.intraPortGroupSize, rawPortInfo.interPortGroupSize,
        rawPortInfo.isInterPod, ParallelDataSplitTypeToStr(splitType), fallbackRatio);
    return fallbackRatio;
}

// 新TopoMatch路径直接使用已归一化的ParallelPortInfo，转换为统一的日志格式后执行回退。
static double ReturnParallelDataSplitFallback(
    const char* failureReason, uint64_t intraRankSize, uint64_t interRankSize, const ParallelPortInfo& portInfo,
    ParallelDataSplitType splitType, double fallbackRatio)
{
    ParallelChannelPortInfo rawPortInfo;
    rawPortInfo.intraPortGroupSize = portInfo.intraPortGroupSize;
    rawPortInfo.interPortGroupSize = portInfo.interPortGroupSize;
    rawPortInfo.isInterPod = portInfo.isPod;
    return ReturnParallelDataSplitFallback(
        failureReason, intraRankSize, interRankSize, rawPortInfo, splitType, fallbackRatio);
}

// 统一记录公式原始结果、POD修正信息及最终量化结果。
static void LogParallelDataSplitRatio(
    uint64_t intraRankSize, uint64_t interRankSize, const ParallelPortInfo& portInfo, ParallelDataSplitType splitType,
    double rawRatio, double quantizedRatio)
{
    HCCL_INFO(
        "[CalcParallelDataSplitRatio] intraRankSize[%llu], interRankSize[%llu], "
        "intraPortGroupSize[%llu], interPortGroupSize[%llu], effectiveInterPortGroupSize[%f], isPod[%d], "
        "splitType[%s], rawRatio[%f], quantizedRatio[%f]",
        intraRankSize, interRankSize, portInfo.intraPortGroupSize, portInfo.interPortGroupSize,
        portInfo.effectiveInterPortGroupSize, portInfo.isPod, ParallelDataSplitTypeToStr(splitType), rawRatio,
        quantizedRatio);
}

// 映射值是 physicalLevels 的数组下标，不是 netLayer 编号。折叠或跳过物理层时，
// 同一算法层可能同时包含不同端口数的链路，不能以本地首个对端的端口数自适应切分。
static bool SupportsParallelAdaptiveSplit(
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    const auto& physicalIndices = algHierarchyInfo.physicalIdxForAlgoLevels;
    return topoInfo != nullptr && topoInfo->physicalLevels.size() >= TOPO_LEVEL_NUM_2
           && physicalIndices.size() == TOPO_LEVEL_NUM_2 && physicalIndices[0].size() == 1
           && physicalIndices[1].size() == 1 && static_cast<u32>(physicalIndices[0][0]) == 0
           && static_cast<u32>(physicalIndices[1][0]) == 1;
}

double CalcParallelDataSplitRatio(
    uint64_t intraRankSize, uint64_t interRankSize, const std::map<u32, std::vector<ChannelInfo>>& intraChannels,
    const std::map<u32, std::vector<ChannelInfo>>& interChannels, const ParallelChannelPortInfo& resPortInfo,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    ParallelDataSplitType splitType, double fallbackRatio)
{
    if (!SupportsParallelAdaptiveSplit(topoInfo, algHierarchyInfo)) {
        HCCL_INFO("[CalcParallelDataSplitRatio] physical mapping is not {0},{1}, uses fixed ratio[0.5]");
        return FIXED_PARALLEL_SPLIT_RATIO;
    }
    // 主流程仅负责编排，各类校验、公式和日志细节由独立辅助函数处理。
    const double validFallback = NormalizeParallelFallbackRatio(fallbackRatio);
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        HCCL_INFO("[CalcParallelDataSplitRatio] MESH_1D_CLOS uses fixed ratio[%f]", validFallback);
        return validFallback;
    }
    // CCU模式下资源上下文中没有ChannelInfo，优先使用资源阶段采集的端口信息。
    ParallelChannelPortInfo rawPortInfo = resPortInfo;
    const char* failureReason = nullptr;
    if (!rawPortInfo.isValid
        && !ExtractParallelPortInfoFromChannels(intraChannels, interChannels, rawPortInfo, failureReason)) {
        return ReturnParallelDataSplitFallback(
            failureReason, intraRankSize, interRankSize, rawPortInfo, splitType, validFallback);
    }

    ParallelPortInfo portInfo;
    if (!PrepareParallelPortInfo(intraRankSize, interRankSize, rawPortInfo, splitType, portInfo, failureReason)) {
        return ReturnParallelDataSplitFallback(
            failureReason, intraRankSize, interRankSize, rawPortInfo, splitType, validFallback);
    }

    ParallelTimeCoeff timeCoeff;
    if (!CalcParallelTimeCoeff(intraRankSize, interRankSize, portInfo, splitType, timeCoeff)) {
        return ReturnParallelDataSplitFallback(
            "unknown splitType", intraRankSize, interRankSize, rawPortInfo, splitType, validFallback);
    }

    double ratio = 0.0;
    if (!CalcRawParallelDataSplitRatio(timeCoeff, ratio, failureReason)) {
        return ReturnParallelDataSplitFallback(
            failureReason, intraRankSize, interRankSize, rawPortInfo, splitType, validFallback);
    }

    const double quantizedRatio = QuantizeParallelDataSplitRatio(ratio);
    LogParallelDataSplitRatio(intraRankSize, interRankSize, portInfo, splitType, ratio, quantizedRatio);
    return quantizedRatio;
}

double CalcParallelDataSplitRatio(
    uint64_t intraRankSize, uint64_t interRankSize, const std::vector<u32>& portNum,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    ParallelDataSplitType splitType, double fallbackRatio)
{
    if (!SupportsParallelAdaptiveSplit(topoInfo, algHierarchyInfo)) {
        HCCL_INFO("[CalcParallelDataSplitRatio] physical mapping is not {0},{1}, uses fixed ratio[0.5]");
        return FIXED_PARALLEL_SPLIT_RATIO;
    }
    // 主流程仅负责编排，各类校验、公式和日志细节由独立辅助函数处理。
    const double validFallback = NormalizeParallelFallbackRatio(fallbackRatio);
    if (topoInfo != nullptr && topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        HCCL_INFO("[CalcParallelDataSplitRatio] MESH_1D_CLOS uses fixed ratio[%f]", validFallback);
        return validFallback;
    }
    ParallelPortInfo portInfo;
    const char* failureReason = nullptr;

    // 从 portNum 提取端口和（与 PrepareParallelPortInfo 中 GetPortGroupSize 语义对齐）
    if (portNum.empty()) {
        return ReturnParallelDataSplitFallback(
            "portNum is empty", intraRankSize, interRankSize, portInfo, splitType, validFallback);
    }
    portInfo.intraPortGroupSize = 1;
    portInfo.interPortGroupSize = (portNum.size() >= MIN_PORT_NUM_LAYERS) ? portNum[0] + portNum[1] : portNum[0];

    // 校验端口和非零（与第一个重载对齐）
    if (portInfo.intraPortGroupSize == 0) {
        return ReturnParallelDataSplitFallback(
            "intraPortGroupSize is 0", intraRankSize, interRankSize, portInfo, splitType, validFallback);
    }
    if (portInfo.interPortGroupSize == 0) {
        return ReturnParallelDataSplitFallback(
            "interPortGroupSize is 0", intraRankSize, interRankSize, portInfo, splitType, validFallback);
    }
    // 溢出检查（与第一个重载对齐）
    if (intraRankSize - 1 > std::numeric_limits<uint64_t>::max() / portInfo.intraPortGroupSize) {
        return ReturnParallelDataSplitFallback(
            "intraPortGroupSize scaling overflow", intraRankSize, interRankSize, portInfo, splitType, validFallback);
    }

    // 缩放 + POD 修正（与第一个重载对齐）
    portInfo.intraPortGroupSize *= intraRankSize - 1;
    portInfo.isPod = topoInfo->isPod; // 无 channel map，只能信 topoInfo->isPod
    portInfo.effectiveInterPortGroupSize
        = static_cast<double>(portInfo.interPortGroupSize) / (portInfo.isPod ? POD_PORT_GROUP_DIVISOR : 1.0);

    // 缩放后零值/有限性校验（与第一个重载对齐）
    if (portInfo.intraPortGroupSize == 0) {
        return ReturnParallelDataSplitFallback(
            "scaled intraPortGroupSize is 0", intraRankSize, interRankSize, portInfo, splitType, validFallback);
    }
    if (IsDoubleEqual(portInfo.effectiveInterPortGroupSize, 0.0)
        || !std::isfinite(portInfo.effectiveInterPortGroupSize)) {
        return ReturnParallelDataSplitFallback(
            "effectiveInterPortGroupSize is 0 or not finite", intraRankSize, interRankSize, portInfo, splitType,
            validFallback);
    }

    ParallelTimeCoeff timeCoeff;
    if (!CalcParallelTimeCoeff(intraRankSize, interRankSize, portInfo, splitType, timeCoeff)) {
        return ReturnParallelDataSplitFallback(
            "unknown splitType", intraRankSize, interRankSize, portInfo, splitType, validFallback);
    }

    double ratio = 0.0;
    if (!CalcRawParallelDataSplitRatio(timeCoeff, ratio, failureReason)) {
        return ReturnParallelDataSplitFallback(
            failureReason, intraRankSize, interRankSize, portInfo, splitType, validFallback);
    }

    const double quantizedRatio = QuantizeParallelDataSplitRatio(ratio);
    LogParallelDataSplitRatio(intraRankSize, interRankSize, portInfo, splitType, ratio, quantizedRatio);
    return quantizedRatio;
}

HcclResult FillChannelSymWinPeerAddrs(
    void* inputSymWindow, u64 inputOffset, void* outputSymWindow, u64 outputOffset, ChannelInfo& channel)
{
    // 单侧无数据时（如AllToAllVC只收不发/只发不收）该侧window为空，跳过该侧远端地址获取
    if (inputSymWindow != nullptr) {
        HcclResult ret
            = GetSymWinRemoteMem(inputSymWindow, inputOffset, channel.remoteRank, &channel.remoteInputGraphMode.addr);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS || channel.remoteInputGraphMode.addr == nullptr,
            HCCL_ERROR(
                "[InsCollAlgBase][FillChannelSymWinPeerAddrs] GetSymWinRemoteMem failed, "
                "remoteRank[%u] inputRet[%d] in[%p]",
                channel.remoteRank, ret, channel.remoteInputGraphMode.addr),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_INFO(
            "[InsCollAlgBase][FillChannelSymWinPeerAddrs] GetSymWinRemoteMem success, "
            "remoteRank[%u] in[%p]",
            channel.remoteRank, channel.remoteInputGraphMode.addr);
    }

    if (outputSymWindow != nullptr) {
        HcclResult ret = GetSymWinRemoteMem(
            outputSymWindow, outputOffset, channel.remoteRank, &channel.remoteOutputGraphMode.addr);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS || channel.remoteOutputGraphMode.addr == nullptr,
            HCCL_ERROR(
                "[InsCollAlgBase][FillChannelSymWinPeerAddrs] GetSymWinRemoteMem failed, "
                "remoteRank[%u] outputRet[%d] out[%p]",
                channel.remoteRank, ret, channel.remoteOutputGraphMode.addr),
            HcclResult::HCCL_E_INTERNAL);
        HCCL_INFO(
            "[InsCollAlgBase][FillChannelSymWinPeerAddrs] GetSymWinRemoteMem success, "
            "remoteRank[%u] out[%p]",
            channel.remoteRank, channel.remoteOutputGraphMode.addr);
    }

    return HcclResult::HCCL_SUCCESS;
}

#ifndef AICPU_COMPILE
// 并行模板的各远端rank使用同构链路组，只取首个非空组可避免按rank重复累加，
// 并与AICPU/DPU路径的GetPortGroupSize取组规则保持一致。
static const std::vector<HcclChannelDesc>*
GetFirstNonEmptyChannelDescGroup(const std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    for (const auto& entry : rankIdToChannelDesc) {
        if (!entry.second.empty()) {
            return &entry.second;
        }
    }
    return nullptr;
}

// 按对端rank聚合各kernel的建链请求。CCU下同一对端的多条链路可能分布在不同kernel（不同die）上，
// 聚合后与AICPU侧ChannelInfo Map的分组含义保持一致。
static void GatherCcuChannelDescByRemoteRank(
    const std::vector<CcuKernelInfo>& kernelInfos, std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc)
{
    for (const auto& kernelInfo : kernelInfos) {
        for (const auto& channelDesc : kernelInfo.channels) {
            rankIdToChannelDesc[channelDesc.remoteRank].push_back(channelDesc);
        }
    }
}

// 累加Channel组内各链路的带宽系数作为端口组大小，任一链路查询失败或结果为0均视为采集失败。
static bool SumCcuChannelGroupBwCoeff(
    HcclComm comm, u32 userRank, const std::vector<HcclChannelDesc>& channelGroup, uint64_t& portGroupSize)
{
    portGroupSize = 0;
    for (const auto& channelDesc : channelGroup) {
        EndpointAttrBwCoeff bwCoeff = 0;
        HcclResult ret = HcclRankGraphGetEndpointInfo(
            comm, userRank, &channelDesc.localEndpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(EndpointAttrBwCoeff),
            static_cast<void*>(&bwCoeff));
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CollectParallelPortInfoFromCcuKernels] failed to get bwCoeff for userRank[%u], remoteRank[%u], "
                "ret[0x%016llx].",
                userRank, channelDesc.remoteRank, HCCL_ERROR_CODE(ret));
            return false;
        }
        portGroupSize += bwCoeff;
    }
    return portGroupSize > 0;
}

// 首个非空Channel组恰好包含两条跨Die链路时判定为POD机型，与IsPodInterChannelGroup的判定规则保持一致。
// 返回值表示属性查询是否成功，POD判定通过isPod返回，避免将查询失败误判为非POD。
static bool GetCcuInterChannelGroupPodState(
    HcclComm comm, u32 userRank, const std::vector<HcclChannelDesc>& channelGroup, bool& isPod)
{
    constexpr size_t podChannelNum = 2;
    isPod = false;
    if (channelGroup.size() != podChannelNum) {
        return true;
    }
    EndpointAttrDieId dieIds[podChannelNum] = {INVALID_VALUE_RANKID, INVALID_VALUE_RANKID};
    for (size_t idx = 0; idx < podChannelNum; idx++) {
        HcclResult ret = HcclRankGraphGetEndpointInfo(
            comm, userRank, &channelGroup[idx].localEndpoint, ENDPOINT_ATTR_DIE_ID, sizeof(EndpointAttrDieId),
            static_cast<void*>(&dieIds[idx]));
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CollectParallelPortInfoFromCcuKernels] failed to get dieId for userRank[%u], remoteRank[%u], "
                "ret[0x%016llx]. POD convergence adjustment will not be used.",
                userRank, channelGroup[idx].remoteRank, HCCL_ERROR_CODE(ret));
            return false;
        }
    }
    if (dieIds[0] == INVALID_VALUE_RANKID || dieIds[1] == INVALID_VALUE_RANKID) {
        HCCL_WARNING(
            "[CollectParallelPortInfoFromCcuKernels] invalid dieId for userRank[%u], dieIds[%u, %u].", userRank,
            dieIds[0], dieIds[1]);
        return false;
    }
    isPod = dieIds[0] != dieIds[1];
    return true;
}

HcclResult CollectParallelPortInfoFromCcuKernels(
    HcclComm comm, u32 userRank, const std::vector<CcuKernelInfo>& intraKernelInfos,
    const std::vector<CcuKernelInfo>& interKernelInfos, ParallelChannelPortInfo& portInfo)
{
    portInfo = ParallelChannelPortInfo();
    std::map<u32, std::vector<HcclChannelDesc>> intraChannelDesc;
    std::map<u32, std::vector<HcclChannelDesc>> interChannelDesc;
    GatherCcuChannelDescByRemoteRank(intraKernelInfos, intraChannelDesc);
    GatherCcuChannelDescByRemoteRank(interKernelInfos, interChannelDesc);

    const std::vector<HcclChannelDesc>* intraGroup = GetFirstNonEmptyChannelDescGroup(intraChannelDesc);
    const std::vector<HcclChannelDesc>* interGroup = GetFirstNonEmptyChannelDescGroup(interChannelDesc);
    // 端口信息仅用于数据切分比例寻优，采集失败时保持isValid为false，执行侧按回退比例运行，不阻断资源申请。
    if (intraGroup == nullptr || interGroup == nullptr) {
        HCCL_WARNING(
            "[CollectParallelPortInfoFromCcuKernels] userRank[%u] no non-empty channel group, "
            "intraKernelNum[%zu], interKernelNum[%zu].",
            userRank, intraKernelInfos.size(), interKernelInfos.size());
        return HCCL_SUCCESS;
    }
    if (!SumCcuChannelGroupBwCoeff(comm, userRank, *intraGroup, portInfo.intraPortGroupSize)
        || !SumCcuChannelGroupBwCoeff(comm, userRank, *interGroup, portInfo.interPortGroupSize)) {
        portInfo = ParallelChannelPortInfo();
        return HCCL_SUCCESS;
    }
    if (!GetCcuInterChannelGroupPodState(comm, userRank, *interGroup, portInfo.isInterPod)) {
        portInfo = ParallelChannelPortInfo();
        return HCCL_SUCCESS;
    }
    portInfo.isValid = true;
    HCCL_INFO(
        "[CollectParallelPortInfoFromCcuKernels] userRank[%u] intraPortGroupSize[%llu], interPortGroupSize[%llu], "
        "isInterPod[%d]",
        userRank, portInfo.intraPortGroupSize, portInfo.interPortGroupSize, portInfo.isInterPod);
    return HCCL_SUCCESS;
}
#endif
} // namespace ops_hccl
