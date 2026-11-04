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
#include <map>
#include <numeric>

namespace ops_hccl {

TopoMatchBaseV2::TopoMatchBaseV2() {}
TopoMatchBaseV2::~TopoMatchBaseV2() {}

bool IsPowerOfTwo(u32 value) { return value != 0 && (value & (value - 1)) == 0; }

u32 CalcGcdByPair(u32 a, u32 b)
{
    if (a == 0 || b == 0) {
        return 1;
    }
    HCCL_DEBUG("[CalcGcdByPair] a[%u] b[%u]", a, b);
    while (b != 0) {
        u32 r = a % b;
        a = b;
        b = r;
    }
    HCCL_DEBUG("[CalcGcdByPair] gcd[%u]", a);
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
        HCCL_WARNING(
            "[TopoMatchBase] Rank [%u], %s group size[%zu] != dim[%u].", myRank, levelName.c_str(), group.size(), dim);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    if (std::find(group.begin(), group.end(), myRank) == group.end()) {
        HCCL_WARNING("[TopoMatchBase] Rank [%u], %s group does not contain myRank.", myRank, levelName.c_str());
        return HcclResult::HCCL_E_NOT_SUPPORT;
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
    bool isAivOrCcu
        = (engine == OpExecuteConfig::AIV || engine == OpExecuteConfig::CCU_MS || engine == OpExecuteConfig::CCU_SCHED);
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
        if (isAivOrCcu && HasUbgLink(physicalLevels[i].protocols)) {
            HCCL_INFO("[CollectEffectiveIndices] skip level[%u]: UBG protocol (AIV excludes UBG).", i);
            continue;
        }
        effIdx.push_back(i);
    }
    return effIdx;
}

// 返回当前引擎可用的最高全域DEVICE来源层号; effective已经完成TopoInstance和协议过滤。
static u32 FindGlobalDeviceLayer(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effective, u32 userRankSize)
{
    u32 globalLayer = INVALID_UINT;
    for (u32 idx : effective) {
        const auto& level = physicalLevels[idx];
        if (level.ref.netLayer != INVALID_UINT && level.view == PhysicalLevelView::GLOBAL
            && level.instSizeListByLayer.size() == 1 && level.instSizeListByLayer.front() == userRankSize
            && level.localRanks.size() == userRankSize && level.locType == ENDPOINT_LOC_TYPE_DEVICE) {
            if (globalLayer == INVALID_UINT || level.ref.netLayer > globalLayer) {
                globalLayer = level.ref.netLayer;
            }
        }
    }
    return globalLayer;
}

bool ShouldUseRailHierarchy(
    const std::vector<PhysicalLevelInfo>& physicalLevels, OpExecuteConfig engine, u32 userRankSize)
{
    if (engine == OpExecuteConfig::HOSTCPU || userRankSize == 0 || !HasLayer2Partition(physicalLevels)) {
        return false;
    }
    // 复用引擎过滤: Host全域来源以及AIV/CCU不能使用的UBG来源不替代设备分轨路径。
    // 全域来源也可能位于Server/Pod层, 不以局部匹配成功与否切换算法布局。
    const auto effective = CollectEffectiveIndices(physicalLevels, engine);
    return FindGlobalDeviceLayer(physicalLevels, effective, userRankSize) == INVALID_UINT;
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
    u32& topPhysPos, std::map<u32, u32>& anchors)
{
    bool found = false;
    for (int32_t k = static_cast<int32_t>(effIdx.size()) - 1; k >= 0; k--) {
        const PhysicalLevelInfo& lvl = physicalLevels[effIdx[k]];
        if (lvl.locType == EndpointLocType::ENDPOINT_LOC_TYPE_HOST && lvl.localRanks.size() == userRankSize) {
            anchors[topAlgo] = static_cast<u32>(k);
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
            "[FindAnchors] hostdpu but no HOST layer with localRanks==userRankSize[%u], unsupported topology.",
            userRankSize);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    // HOST 锚点以下的物理层数（= topPhysPos）须 >= 剩余待匹配的算法层数（= topAlgo），否则低层无足够物理层
    if (topPhysPos < topAlgo) {
        HCCL_INFO(
            "[FindAnchors] hostdpu phys layers below host[%u] < remaining algo levels[%u], unsupported topology.",
            topPhysPos, topAlgo);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

// Mesh 锚点：算法层从低到高遍历，优先匹配 COMM_TOPO_1DMESH 物理层，不可重复锚定；
// hostdpu 已锚定的最高层跳过；MeshConcur 未匹配到则 not support
static HcclResult AnchorMeshLevels(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx,
    const std::vector<AlgoType>& algoTypes, u32 topAlgo, u32 topPhysPos, std::map<u32, u32>& anchors)
{
    bool hasPrevAnchor = false;
    u32 prevAlgo = 0;
    u32 prevPhys = 0;
    for (u32 i = 0; i < algoTypes.size(); i++) {
        if (anchors.count(i) > 0) {
            continue;
        }
        if (!IsMeshAlgo(algoTypes[i])) {
            continue;
        }
        // 为上层算法层（i+1..topAlgo）留足物理位：candidateHigh = topPhysPos - (topAlgo - i)
        u32 candidateHigh = topPhysPos - (topAlgo - i);
        // 前驱锚点之后留足中间算法层的物理位：candidateLow = prevPhys + (i - prevAlgo)
        u32 candidateLow = hasPrevAnchor ? prevPhys + (i - prevAlgo) : i;
        bool found = false;
        for (u32 k = candidateLow; k <= candidateHigh; k++) {
            if (physicalLevels[effIdx[k]].topoType == COMM_TOPO_1DMESH) {
                anchors[i] = k;
                HCCL_INFO("[FindAnchors] mesh: algo level[%u] anchored to phys[%u] (1DMESH).", i, effIdx[k]);
                found = true;
                hasPrevAnchor = true;
                prevAlgo = i;
                prevPhys = k;
                break;
            }
        }
        if (!found && IsMeshConcurAlgo(algoTypes[i])) {
            HCCL_INFO("[FindAnchors] algo[%u] MeshConcur but no Mesh layer, unsupported topology.", i);
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
    u32 topAlgo = static_cast<u32>(algoTypes.size()) - 1;
    // 最高算法层对应的物理 effIdx 位置：非 hostdpu 由尾段取 effIdx.back()；hostdpu 取 HOST 锚点
    u32 topPhysPos = static_cast<u32>(effIdx.size()) - 1;
    HcclResult ret = HCCL_SUCCESS;
    if (engine == OpExecuteConfig::HOSTCPU) {
        ret = AnchorHostDpu(physicalLevels, effIdx, userRankSize, topAlgo, topPhysPos, anchors);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_INFO("[FindAnchors] AnchorHostDpu failed: hcclRet -> %d", ret),
            HcclResult::HCCL_E_NOT_SUPPORT);
    }
    // NDA 变体算法（最高层 NHR_NDA/MESH_NDA，host 直驱网卡）：L0 不做 mesh 锚点，保留 HOST
    // 锚点后交给段内压缩，映射到 HOST 锚点以下的最高物理层（server 内最大域）。
    // 若走 mesh 锚点会被钉到最低 1DMESH（device fullmesh）层，L1 代表组将超出
    // NDA 同号卡直连范围，导致建链失败。
    if (engine == OpExecuteConfig::HOSTCPU && NDA_VARIANT_ALGO_TYPES.count(algoTypes[topAlgo]) > 0) {
        HCCL_INFO("[FindAnchors] NDA variant algo (top level), skip mesh anchor, L0 falls to segment mapping.");
        return HcclResult::HCCL_SUCCESS;
    }
    ret = AnchorMeshLevels(physicalLevels, effIdx, algoTypes, topAlgo, topPhysPos, anchors);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_INFO("[FindAnchors] AnchorMeshLevels failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);
    return HcclResult::HCCL_SUCCESS;
}

// 最后一段匹配：低层一一 + 最高层从 physLow+algoCount-1 起向上找首个 localRanks.size()==userRankSize 的物理层
static HcclResult MatchLastSegment(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx, u32 userRankSize, u32 algoLow,
    u32 algoHigh, u32 physLow, u32 physHigh, std::vector<u32>& pIndices)
{
    if (algoLow > algoHigh) {
        return HcclResult::HCCL_SUCCESS;
    }
    u32 algoCount = algoHigh - algoLow + 1;
    for (u32 k = 0; k + 1 < algoCount; k++) {
        pIndices[algoLow + k] = physLow + k;
    }
    u32 searchStart = physLow + algoCount - 1;
    bool found = false;
    for (u32 k = searchStart; k <= physHigh; k++) {
        if (physicalLevels[effIdx[k]].localRanks.size() == userRankSize) {
            pIndices[algoHigh] = k;
            found = true;
            break;
        }
    }
    if (!found) {
        HCCL_INFO(
            "[MatchLastSegment] no physical level with localRanks==userRankSize[%u] in range [%u, %u], unsupported "
            "topology.",
            userRankSize, searchStart, physHigh);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

// 分段压缩：按锚点将算法层与物理层分段，每段低层一一 + 最后一段最高层按 localRanks==userRankSize 匹配
HcclResult ResolveSegmentMapping(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx,
    const std::vector<AlgoType>& algoTypes, const std::map<u32, u32>& anchors, u32 userRankSize,
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
    HcclResult ret = MatchLastSegment(
        physicalLevels, effIdx, userRankSize, algoStart, algoTypes.size() - 1, physStart, effIdx.size() - 1, pIndices);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_INFO("[ResolveSegmentMapping] MatchLastSegment failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);
    // 兜底校验：所有算法层都应已映射到有效的 effIdx position
    for (size_t i = 0; i < pIndices.size(); i++) {
        if (pIndices[i] >= effIdx.size()) {
            HCCL_INFO("[ResolveSegmentMapping] pIndices[%zu] invalid value[%u], unsupported topology.", i, pIndices[i]);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
    }
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

// 引擎过滤 + 锚点匹配 + 分段，得 effIdx 与 pIndices
HcclResult ResolveMapping(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const AlgAttrs& algAttrs, u32 userRankSize,
    std::vector<u32>& effIdx, std::vector<u32>& pIndices)
{
    effIdx = CollectEffectiveIndices(physicalLevels, algAttrs.engine);
    if (algAttrs.engine != OpExecuteConfig::HOSTCPU && HasLayer2Partition(physicalLevels)) {
        const u32 globalLayer = FindGlobalDeviceLayer(physicalLevels, effIdx, userRankSize);
        if (globalLayer != INVALID_UINT && globalLayer < RAIL_NET_LAYER) {
            // 全域来源仅在Server/Pod层时, layer2可能是其内层。普通匹配按范围大小恢复层次,
            // 否则ThreeLevel会把全域Pod用作中间层, 并在更后方找不到全域来源。
            // 只重排有效索引, 保留物理来源目录; 存在更高层全域来源时维持既有匹配次序。
            std::stable_sort(effIdx.begin(), effIdx.end(), [&physicalLevels](u32 lhs, u32 rhs) {
                return physicalLevels[lhs].localRanks.size() < physicalLevels[rhs].localRanks.size();
            });
        }
    }
    if (algAttrs.engine == OpExecuteConfig::HOSTCPU && HasLayer2Partition(physicalLevels)) {
        // HOSTCPU使用Host全域来源; 设备引擎有全域来源时保留layer2参与普通匹配。
        // 排除非HOST轨层后校验包含链, 保留同层HOST来源供全域锚点匹配。
        // 后续FindAnchors还须校验存在覆盖全域的HOST来源, 失败时不切换到设备分轨路径。
        effIdx.erase(
            std::remove_if(
                effIdx.begin(), effIdx.end(),
                [&physicalLevels](u32 idx) {
                    const auto& level = physicalLevels[idx];
                    return level.ref.netLayer == RAIL_NET_LAYER && level.locType != ENDPOINT_LOC_TYPE_HOST;
                }),
            effIdx.end());
        for (size_t i = 1; i < effIdx.size(); i++) {
            const auto& lower = physicalLevels[effIdx[i - 1]].localRanks;
            const auto& upper = physicalLevels[effIdx[i]].localRanks;
            if (!std::includes(upper.begin(), upper.end(), lower.begin(), lower.end())) {
                return HCCL_E_NOT_SUPPORT;
            }
        }
    }
    u32 algoLevelNum = algAttrs.algoTypes.size();
    if (effIdx.size() < algoLevelNum) {
        HCCL_INFO("[ResolveMapping] valid level num[%zu] < algoLevelNum[%u].", effIdx.size(), algoLevelNum);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    std::map<u32, u32> anchors;
    // 锚点匹配：含 MeshConcur 的 1DMESH 校验与 hostdpu 强约束，须无条件执行（1:1 时也需校验底层 1DMESH）
    HcclResult ret = HCCL_SUCCESS;
    ret = FindAnchors(physicalLevels, effIdx, algAttrs.algoTypes, algAttrs.engine, userRankSize, anchors);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_INFO("[ResolveMapping] FindAnchors failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);
    ret = ResolveSegmentMapping(physicalLevels, effIdx, algAttrs.algoTypes, anchors, userRankSize, pIndices);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_INFO("[ResolveMapping] ResolveSegmentMapping failed: hcclRet -> %d", ret),
        HcclResult::HCCL_E_NOT_SUPPORT);
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
                HCCL_INFO(
                    "[FillPhysicalIdx] level[%u] MeshConcur no upper encompassing layer, unsupported topology.", i);
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

// 分轨公共匹配与普通匹配共享此编译单元, host/device及独立LLT目标均包含它。
namespace {

    const PhysicalLevelInfo* FindNetLevel(const std::vector<PhysicalLevelInfo>& levels, u32 netLayer)
    {
        for (const auto& level : levels) {
            if (level.ref.netLayer == netLayer && level.view == PhysicalLevelView::GLOBAL) {
                return &level;
            }
        }
        return nullptr;
    }

    bool ContainsGroup(const std::vector<u32>& ranks, const std::vector<u32>& group)
    {
        return std::includes(ranks.begin(), ranks.end(), group.begin(), group.end());
    }

    bool IsValidPartition(const PhysicalLevelInfo* level, u32 myRank, u32 rankSize)
    {
        if (level == nullptr) {
            return false;
        }
        const auto& ranks = level->localRanks;
        const auto& sizes = level->instSizeListByLayer;
        return !ranks.empty() && std::is_sorted(ranks.begin(), ranks.end()) && ranks.back() < rankSize
               && std::adjacent_find(ranks.begin(), ranks.end()) == ranks.end()
               && std::binary_search(ranks.begin(), ranks.end(), myRank) && !sizes.empty()
               && std::find(sizes.begin(), sizes.end(), 0U) == sizes.end()
               && std::accumulate(sizes.begin(), sizes.end(), u64{0}) == rankSize
               && std::find(sizes.begin(), sizes.end(), static_cast<u32>(ranks.size())) != sizes.end();
    }

    // 既有执行器以rank编号计算切片坐标, 物理超节点必须由连续且对齐的虚拟块组成。
    bool IsAlignedRange(const std::vector<u32>& ranks, u32 blockSize)
    {
        if (blockSize == 0 || ranks.empty() || ranks.front() % blockSize != 0 || ranks.size() % blockSize != 0) {
            return false;
        }
        for (size_t i = 0; i < ranks.size(); i++) {
            if (static_cast<u64>(ranks.front()) + i != ranks[i]) {
                return false;
            }
        }
        return true;
    }

    bool HasCompleteRailColumns(const PhysicalLevelInfo& rail, u32 podSize, u32 rankSize)
    {
        const u32 columnSize = rankSize / podSize;
        // 每条轨必须能承载若干完整的算法顶层组。此条件属于算法布局, 不是parse连通性判据。
        if (std::any_of(rail.instSizeListByLayer.begin(), rail.instSizeListByLayer.end(), [columnSize](u32 size) {
                return size % columnSize != 0;
            })) {
            return false;
        }
        std::map<u32, u32> columnCounts;
        for (u32 rank : rail.localRanks) {
            columnCounts[rank % podSize]++;
        }
        // 成员已验证唯一且在[0,N)内; 同一余数出现N/P次, 即该编号列完整落在真实轨内。
        return std::all_of(columnCounts.begin(), columnCounts.end(), [columnSize](const auto& column) {
            return column.second == columnSize;
        });
    }

    HcclResult BuildRailGroups(
        const TopoInfoWithNetLayerDetails& topoInfo, size_t levelNum, AlgHierarchyInfoForAllLevel& result,
        std::vector<u32>& netLayers)
    {
        const u32 rankSize = topoInfo.userRankSize;
        const u32 myRank = topoInfo.userRank;
        const auto* pod = FindNetLevel(topoInfo.physicalLevels, POD_NET_LAYER);
        const auto* rail = FindNetLevel(topoInfo.physicalLevels, RAIL_NET_LAYER);
        if (!IsValidPartition(pod, myRank, rankSize) || !IsValidPartition(rail, myRank, rankSize)) {
            return HCCL_E_NOT_SUPPORT;
        }
        // P为算法虚拟块大小, 可以小于真实超节点。infos决定执行器的切片维度,
        // physicalIdx仅记录能承载该组的物理来源, 不要求组等于整个物理范围。
        const u32 podSize = CalcGcd(pod->instSizeListByLayer);
        if (podSize <= 1 || podSize >= rankSize || rankSize % podSize != 0 || !IsAlignedRange(pod->localRanks, podSize)
            || !HasCompleteRailColumns(*rail, podSize, rankSize)) {
            HCCL_INFO("[TopoMatchRail] rank[%u], pod/rail do not support regular blocks of size[%u]", myRank, podSize);
            return HCCL_E_NOT_SUPPORT;
        }

        result.infos.resize(levelNum, std::vector<std::vector<u32>>(1));
        const u32 podBase = (myRank / podSize) * podSize;
        if (levelNum == ALGO_LEVEL_NUM_TWO) {
            result.infos[0][0] = BuildRepresentativeGroup(1, podSize, podBase);
            netLayers = {POD_NET_LAYER, RAIL_NET_LAYER};
            CHK_RET(ValidateGroup(result.infos[0][0], podSize, myRank, "rail pod"));
        } else {
            const auto* server = FindNetLevel(topoInfo.physicalLevels, SERVER_NET_LAYER);
            if (!IsValidPartition(server, myRank, rankSize) || !IsInstListSymmetric(server->instSizeListByLayer)) {
                return HCCL_E_NOT_SUPPORT;
            }
            const u32 serverSize = server->instSizeListByLayer.front();
            if (podSize % serverSize != 0 || !IsAlignedRange(server->localRanks, serverSize)
                || !ContainsGroup(pod->localRanks, server->localRanks)) {
                return HCCL_E_NOT_SUPPORT;
            }
            result.infos[0][0] = server->localRanks;
            result.infos[1][0]
                = BuildRepresentativeGroup(serverSize, podSize / serverSize, podBase + myRank % serverSize);
            netLayers = {SERVER_NET_LAYER, POD_NET_LAYER, RAIL_NET_LAYER};
            CHK_RET(ValidateGroup(result.infos[0][0], serverSize, myRank, "rail server"));
            CHK_RET(ValidateGroup(result.infos[1][0], podSize / serverSize, myRank, "rail pod representatives"));
        }
        result.infos.back()[0] = BuildRepresentativeGroup(podSize, rankSize / podSize, myRank % podSize);
        return ValidateGroup(result.infos.back()[0], rankSize / podSize, myRank, "rail column");
    }

    bool PickSource(
        const std::vector<PhysicalLevelInfo>& levels, const std::vector<u32>& effective, const std::vector<u32>& group,
        u32 netLayer, AlgoType algo, std::vector<PhysicalLevelIndex>& sources)
    {
        // Mesh优先选择能承载整个组的1DMESH, 普通Mesh仍沿用现有CLOS兼容规则。
        // MeshConcur要求双来源, 不能把多个Server拼成一个不存在的Mesh。
        const u32 passNum = IsMeshAlgo(algo) && !IsMeshConcurAlgo(algo) ? 2 : 1;
        for (u32 pass = 0; pass < passNum; pass++) {
            for (size_t pos = 0; pos < effective.size(); pos++) {
                const u32 idx = effective[pos];
                const auto& level = levels[idx];
                if (level.ref.netLayer != netLayer || level.locType != ENDPOINT_LOC_TYPE_DEVICE
                    || !ContainsGroup(level.localRanks, group)
                    || (IsMeshAlgo(algo) && pass == 0 && level.topoType != COMM_TOPO_1DMESH)
                    || (IsMeshConcurAlgo(algo) && level.topoType != COMM_TOPO_1DMESH)) {
                    continue;
                }
                if (!IsMeshConcurAlgo(algo)) {
                    sources = {static_cast<PhysicalLevelIndex>(idx)};
                    return true;
                }
                for (size_t upper = pos + 1; upper < effective.size(); upper++) {
                    const u32 upperIdx = effective[upper];
                    const auto& candidate = levels[upperIdx];
                    // Server可借用超节点来源; 超节点/轨的MeshConcur须有同层双来源,
                    // 不用额外HostDPU等层替代轨内能力。单个CLOS来源不满足MeshConcur。
                    const u32 maxLayer = netLayer == SERVER_NET_LAYER ? POD_NET_LAYER : netLayer;
                    if (candidate.ref.netLayer < netLayer || candidate.ref.netLayer > maxLayer
                        || candidate.locType != ENDPOINT_LOC_TYPE_DEVICE
                        || !ContainsGroup(candidate.localRanks, level.localRanks)) {
                        continue;
                    }
                    sources = {static_cast<PhysicalLevelIndex>(idx), static_cast<PhysicalLevelIndex>(upperIdx)};
                    return true;
                }
            }
        }
        return false;
    }

} // namespace

HcclResult MatchRailHierarchy(
    const TopoInfoWithNetLayerDetails& topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    const AlgAttrs& algAttrs)
{
    const size_t levelNum = algAttrs.algoTypes.size();
    if (topoInfo.userRankSize == 0 || topoInfo.userRank >= topoInfo.userRankSize
        || (levelNum != ALGO_LEVEL_NUM_TWO && levelNum != ALGO_LEVEL_NUM_THREE)
        || !ShouldUseRailHierarchy(topoInfo.physicalLevels, algAttrs.engine, topoInfo.userRankSize)) {
        return HCCL_E_NOT_SUPPORT;
    }
    // 在局部结果上构建, 不让不支持的候选留下部分组或来源索引。
    AlgHierarchyInfoForAllLevel result;
    std::vector<u32> netLayers;
    const HcclResult ret = BuildRailGroups(topoInfo, levelNum, result, netLayers);
    if (ret != HCCL_SUCCESS) {
        HCCL_INFO(
            "[TopoMatchRail] rank[%u], regular %zu-level rail layout is not supported", topoInfo.userRank, levelNum);
        return ret;
    }
    const auto effective = CollectEffectiveIndices(topoInfo.physicalLevels, algAttrs.engine);
    result.physicalIdxForAlgoLevels.resize(levelNum);
    for (size_t i = 0; i < levelNum; i++) {
        if (!PickSource(
                topoInfo.physicalLevels, effective, result.infos[i][0], netLayers[i], algAttrs.algoTypes[i],
                result.physicalIdxForAlgoLevels[i])) {
            HCCL_INFO(
                "[TopoMatchRail] rank[%u], algo level[%zu] has no compatible physical source at netLayer[%u]",
                topoInfo.userRank, i, netLayers[i]);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    HCCL_INFO(
        "[TopoMatchRail] rank[%u], matched [%zu] algorithm levels, physicalIdxForAlgoLevels: [%s]", topoInfo.userRank,
        levelNum, FormatPhysicalIdxForAlgoLevels(result.physicalIdxForAlgoLevels).c_str());
    algHierarchyInfo = std::move(result);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
