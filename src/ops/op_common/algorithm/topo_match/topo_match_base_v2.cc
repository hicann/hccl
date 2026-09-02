/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_base_v2.h"
#include <algorithm>

namespace ops_hccl {

TopoMatchBaseV2::TopoMatchBaseV2() {}
TopoMatchBaseV2::~TopoMatchBaseV2() {}

u32 CalcGcdByPair(u32 a, u32 b)
{
    if (a == 0 || b == 0) {
        return 1;
    }
    while (b != 0) {
        u32 r = a % b;
        a = b;
        b = r;
    }
    HCCL_DEBUG("[CalcGcdByPair] a[%u] b[%u], gcd[%u]", a, b, a);
    return a;
}

u32 CalcGcd(const std::vector<u32>& nums)
{
    if (nums.empty()) {
        return 1;
    }
    u32 result = nums[0];
    for (size_t i = 1; i < nums.size(); i++) {
        result = CalcGcdByPair(result, nums[i]);
        if (result == 1) {
            return 1;
        }
    }
    HCCL_DEBUG("[CalcGcd] size[%u], gcd[%u]", static_cast<u32>(nums.size()), result);
    return result;
}

int32_t FindHighestEffectiveLevel(const std::vector<PhysicalLevelInfo>& physicalLevels)
{
    for (int32_t i = static_cast<int32_t>(physicalLevels.size()) - 1; i >= 0; i--) {
        if (physicalLevels[i].hasTopoInst) {
            return i;
        }
    }
    return INVALID_PHYSICAL_LEVEL_IDX;
}

bool IsInstListSymmetric(const std::vector<uint32_t>& instList)
{
    if (instList.empty()) {
        HCCL_WARNING("[TopoMatchBase] instList is empty!");
        return true;
    }
    for (size_t i = 0; i < instList.size(); i++) {
        if (instList[i] != instList[0]) {
            return false;
        }
    }
    return true;
}

std::vector<u32> BuildRepresentativeGroup(u32 step, u32 count, u32 offset)
{
    std::vector<u32> group;
    group.reserve(count);
    for (u32 i = 0; i < count; i++) {
        group.push_back(offset + i * step);
    }
    return group;
}

HcclResult ValidateGroup(const std::vector<u32>& group, u32 dim, u32 myRank, const std::string& levelName)
{
    if (group.size() != dim) {
        HCCL_ERROR(
            "[TopoMatchBase] Rank [%u], %s group size[%zu] != dim[%u].", myRank, levelName.c_str(), group.size(), dim);
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (std::find(group.begin(), group.end(), myRank) == group.end()) {
        HCCL_ERROR("[TopoMatchBase] Rank [%u], %s group does not contain myRank.", myRank, levelName.c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

// 判断 protocols 是否含 UBG 链路（AIV 引擎需排除这种层）
static bool HasUbgLink(const std::vector<CommProtocol>& protocols)
{
#if CANN_VERSION_NUM >= CANN_VERSION(9, 2, 0)
    for (CommProtocol p : protocols) {
        if (p == COMM_PROTOCOL_UBG) {
            return true;
        }
    }
#endif
    return false;
}

// 按引擎过滤收集 hasTopoInst 的物理层序号：非 hostdpu 排除 HOST 层，AIV 排除含 UBG 链路的层
std::vector<u32> CollectEffectiveIndices(const std::vector<PhysicalLevelInfo>& physicalLevels, OpExecuteConfig engine)
{
    bool isHostdpu = (engine == OpExecuteConfig::HOSTCPU);
    bool isAiv = (engine == OpExecuteConfig::AIV);
    std::vector<u32> effIdx;
    for (u32 i = 0; i < physicalLevels.size(); i++) {
        if (!physicalLevels[i].hasTopoInst) {
            HCCL_INFO("[CollectEffectiveIndices] skip level[%u]: no TopoInstance.", i);
            continue;
        }
        if (!isHostdpu && physicalLevels[i].locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
            HCCL_INFO("[CollectEffectiveIndices] skip level[%u]: HOST locType (non-hostdpu excludes HOST).", i);
            continue;
        }
        if (isAiv && HasUbgLink(physicalLevels[i].protocols)) {
            HCCL_INFO("[CollectEffectiveIndices] skip level[%u]: UBG protocol (AIV excludes UBG).", i);
            continue;
        }
        effIdx.push_back(i);
    }
    return effIdx;
}

// 判断算法是否属于 Mesh 类
bool IsMeshAlgo(AlgoType algo) { return MESH_ALGO_TYPES.count(algo) > 0; }

// 判断算法是否属于 MeshConcur 类（触发 CLOS 双层规则）
bool IsMeshConcurAlgo(AlgoType algo) { return MESH_CONCUR_ALGO_TYPES.count(algo) > 0; }

// 段内匹配：算法 [algoLow..algoHigh] ↔ 物理 [physLow..physHigh]，低层一一 + 最高层压缩多余
static void MatchLayerIdxBySegment(u32 algoLow, u32 algoHigh, u32 physLow, u32 physHigh, std::vector<u32>& pIndices)
{
    if (algoLow > algoHigh) {
        return;
    }
    u32 algoCount = algoHigh - algoLow + 1;
    for (u32 k = 0; k + 1 < algoCount; k++) {
        pIndices[algoLow + k] = physLow + k;
    }
    pIndices[algoHigh] = physHigh;
}

// hostdpu 强约束：最高算法层锚定 HOST 且 localRanks==userRankSize 的物理层（从高到低找首个），
// 并校验 HOST 锚点以下物理层数 >= 剩余待匹配算法层数；找不到或不满足则 not support
static HcclResult AnchorHostDpu(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx, u32 userRankSize, u32 topAlgo,
    u32& topPhysPos, std::set<u32>& anchoredPhys, std::map<u32, u32>& anchors)
{
    bool found = false;
    for (int32_t k = static_cast<int32_t>(effIdx.size()) - 1; k >= 0; k--) {
        const PhysicalLevelInfo& lvl = physicalLevels[effIdx[k]];
        if (lvl.locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST && lvl.localRanks.size() == userRankSize) {
            anchors[topAlgo] = static_cast<u32>(k);
            anchoredPhys.insert(static_cast<u32>(k));
            topPhysPos = static_cast<u32>(k);
            HCCL_INFO(
                "[FindAnchors] hostdpu: algo level[%u] anchored to phys[%u] (HOST, localRankSize==%u).", topAlgo,
                effIdx[k], userRankSize);
            found = true;
            break;
        }
    }
    if (!found) {
        HCCL_INFO(
            "[FindAnchors] hostdpu but no HOST layer with localRanks==userRankSize[%u], not support.", userRankSize);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    // HOST 锚点以下的物理层数（= topPhysPos）须 >= 剩余待匹配的算法层数（= topAlgo），否则低层无足够物理层
    if (topPhysPos < topAlgo) {
        HCCL_INFO(
            "[FindAnchors] hostdpu phys layers below host[%u] < remaining algo levels[%u], not support.", topPhysPos,
            topAlgo);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

// Mesh 锚点：算法层从低到高遍历，优先匹配 COMM_TOPO_1DMESH 物理层，不可重复锚定；
// hostdpu 已锚定的最高层跳过；MeshConcur 未匹配到则 not support
static HcclResult AnchorMeshLevels(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx,
    const std::vector<AlgoType>& algoTypes, u32 topAlgo, u32 topPhysPos, std::set<u32>& anchoredPhys,
    std::map<u32, u32>& anchors)
{
    for (u32 i = 0; i < algoTypes.size(); i++) {
        if (anchors.count(i) > 0) {
            continue;
        }
        if (!IsMeshAlgo(algoTypes[i])) {
            continue;
        }
        // 为上层算法层（i+1..topAlgo）留足物理位：candidateHigh = topPhysPos - (topAlgo - i)
        u32 candidateHigh = topPhysPos - (topAlgo - i);
        bool found = false;
        for (u32 k = i; k <= candidateHigh; k++) {
            if (anchoredPhys.count(k) > 0) {
                continue;
            }
            if (physicalLevels[effIdx[k]].topoType == COMM_TOPO_1DMESH) {
                anchors[i] = k;
                anchoredPhys.insert(k);
                HCCL_INFO("[FindAnchors] mesh: algo level[%u] anchored to phys[%u] (1DMESH).", i, effIdx[k]);
                found = true;
                break;
            }
        }
        if (!found && IsMeshConcurAlgo(algoTypes[i])) {
            HCCL_INFO("[FindAnchors] algo[%u] MeshConcur but no Mesh layer, not support.", i);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// 锚点匹配：hostdpu 强约束最高层选 HOST（优先级高于 Mesh）+ Mesh 层优先匹配 COMM_TOPO_1DMESH
HcclResult FindAnchors(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx,
    const std::vector<AlgoType>& algoTypes, OpExecuteConfig engine, u32 userRankSize, std::map<u32, u32>& anchors)
{
    std::set<u32> anchoredPhys;
    u32 topAlgo = static_cast<u32>(algoTypes.size()) - 1;
    // 最高算法层对应的物理 effIdx 位置：非 hostdpu 由尾段取 effIdx.back()；hostdpu 取 HOST 锚点
    u32 topPhysPos = static_cast<u32>(effIdx.size()) - 1;
    if (engine == OpExecuteConfig::HOSTCPU) {
        CHK_RET(AnchorHostDpu(physicalLevels, effIdx, userRankSize, topAlgo, topPhysPos, anchoredPhys, anchors));
    }
    CHK_RET(AnchorMeshLevels(physicalLevels, effIdx, algoTypes, topAlgo, topPhysPos, anchoredPhys, anchors));
    return HcclResult::HCCL_SUCCESS;
}

// 分段压缩：按锚点将算法层与物理层分段，每段低层一一 + 最高层压缩多余物理层
HcclResult ResolveSegmentMapping(
    const std::vector<u32>& effIdx, const std::vector<AlgoType>& algoTypes, const std::map<u32, u32>& anchors,
    std::vector<u32>& pIndices)
{
    pIndices.resize(algoTypes.size(), INVALID_UINT);
    u32 algoStart = 0;
    u32 physStart = 0;
    for (const auto& [anchorAlgo, anchorPhys] : anchors) {
        // 前段存在当且仅当锚点之前同时有算法层与物理层；anchorAlgo==algoStart 时 anchorAlgo-1 会 u32 下溢，须跳过
        if (anchorAlgo > algoStart && anchorPhys > physStart) {
            MatchLayerIdxBySegment(algoStart, anchorAlgo - 1, physStart, anchorPhys - 1, pIndices);
        }
        pIndices[anchorAlgo] = anchorPhys;
        algoStart = anchorAlgo + 1;
        physStart = anchorPhys + 1;
    }
    MatchLayerIdxBySegment(algoStart, algoTypes.size() - 1, physStart, effIdx.size() - 1, pIndices);
    return HcclResult::HCCL_SUCCESS;
}

// 在 meshEffPos 之上（更高 index）找首个 localRanks 包含 mesh 层 localRanks 的物理层；找不到返回
// INVALID_PHYSICAL_LEVEL_IDX
int32_t FindUpperEncompassingLevel(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx, u32 meshEffPos)
{
    const auto& meshRanks = physicalLevels[effIdx[meshEffPos]].localRanks;
    for (u32 k = meshEffPos + 1; k < effIdx.size(); k++) {
        const auto& upperRanks = physicalLevels[effIdx[k]].localRanks;
        if (std::includes(upperRanks.begin(), upperRanks.end(), meshRanks.begin(), meshRanks.end())) {
            return static_cast<int32_t>(k);
        }
    }
    return INVALID_PHYSICAL_LEVEL_IDX;
}

// 引擎过滤 + 锚点匹配 + 分段，得 effIdx 与 pIndices；校验最高层 localRanks==userRankSize
HcclResult ResolveMapping(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const AlgAttrs& profile, u32 userRankSize,
    std::vector<u32>& effIdx, std::vector<u32>& pIndices)
{
    effIdx = CollectEffectiveIndices(physicalLevels, profile.engine);
    u32 algoLevelNum = profile.algoTypes.size();
    if (effIdx.size() < algoLevelNum) {
        HCCL_INFO("[ResolveMapping] valid level num[%zu] < algoLevelNum[%u].", effIdx.size(), algoLevelNum);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    std::map<u32, u32> anchors;
    // 锚点匹配：含 MeshConcur 的 1DMESH 校验与 hostdpu 强约束，须无条件执行（1:1 时也需校验底层 1DMESH）
    CHK_RET(FindAnchors(physicalLevels, effIdx, profile.algoTypes, profile.engine, userRankSize, anchors));
    CHK_RET(ResolveSegmentMapping(effIdx, profile.algoTypes, anchors, pIndices));
    // 最高算法层 localRanks 必须等于 userRankSize
    u32 topPhys = effIdx[pIndices[algoLevelNum - 1]];
    if (physicalLevels[topPhys].localRanks.size() != userRankSize) {
        HCCL_INFO(
            "[ResolveMapping] top layer localRanks[%zu] != userRankSize[%u].",
            physicalLevels[topPhys].localRanks.size(), userRankSize);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

// 填充 physicalIdxForAlgoLevels（二级）：MeshConcur 层记 {Mesh层, 上层超集层}，普通层记 {该层}
HcclResult FillPhysicalIdxForAlgoLevels(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx,
    const std::vector<u32>& pIndices, const std::vector<AlgoType>& algoTypes,
    std::vector<std::vector<PhysicalLevelIndex>>& physicalIdxForAlgoLevels)
{
    physicalIdxForAlgoLevels.resize(algoTypes.size());
    for (u32 i = 0; i < algoTypes.size(); i++) {
        u32 physIdx = effIdx[pIndices[i]];
        if (IsMeshConcurAlgo(algoTypes[i])) {
            int32_t upperPos = FindUpperEncompassingLevel(physicalLevels, effIdx, pIndices[i]);
            if (upperPos == INVALID_PHYSICAL_LEVEL_IDX) {
                HCCL_INFO("[FillPhysicalIdx] level[%u] MeshConcur no upper encompassing layer, not support.", i);
                return HcclResult::HCCL_E_NOT_SUPPORT;
            }
            physicalIdxForAlgoLevels[i]
                = {static_cast<PhysicalLevelIndex>(physIdx), static_cast<PhysicalLevelIndex>(effIdx[upperPos])};
        } else {
            physicalIdxForAlgoLevels[i] = {static_cast<PhysicalLevelIndex>(physIdx)};
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

std::string FormatPhysicalIdxForAlgoLevels(const std::vector<std::vector<PhysicalLevelIndex>>& physicalIdxForAlgoLevels)
{
    std::string idxStr;
    for (size_t i = 0; i < physicalIdxForAlgoLevels.size(); i++) {
        idxStr += "{";
        for (size_t j = 0; j < physicalIdxForAlgoLevels[i].size(); j++) {
            idxStr += std::to_string(static_cast<u32>(physicalIdxForAlgoLevels[i][j]));
            if (j + 1 < physicalIdxForAlgoLevels[i].size()) {
                idxStr += ",";
            }
        }
        idxStr += "} ";
    }
    return idxStr;
}

} // namespace ops_hccl
