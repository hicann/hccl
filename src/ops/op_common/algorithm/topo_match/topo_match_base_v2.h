/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_BASE_V2
#define TOPO_MATCH_BASE_V2

#include "topo_match_base.h"
#include "alg_parse.h"
#include "alg_attrs.h"
#include <set>
#include <map>
#include <vector>
#include <string>
#include <algorithm>

namespace ops_hccl {

// topo match 相关常量
constexpr int32_t INVALID_PHYSICAL_LEVEL_IDX = -1;
constexpr u32 ALGO_LEVEL_NUM_TWO = 2;
constexpr u32 ALGO_LEVEL_NUM_THREE = 3;
constexpr u32 CONCURRENT_SUBGROUP_NUM = 2;

// 辗转相除求两数最大公约数；a 或 b 为 0 时返回 1 避免退化
u32 CalcGcdByPair(u32 a, u32 b);

// 对一组数逐对归约求最大公约数，result==1 时早停
u32 CalcGcd(const std::vector<u32>& nums);

// 从高到低找首个 hasTopoInst 的物理层序号；不存在返回 INVALID_PHYSICAL_LEVEL_IDX
int32_t FindHighestEffectiveLevel(const std::vector<PhysicalLevelInfo>& physicalLevels);

// instList 各元素是否全等（对称判定）
bool IsInstListSymmetric(const std::vector<uint32_t>& instList);

// 构造跨层代表 rank：count 个，从 offset 起、按 step 步长（offset 取 myRank 在本层的偏移，保证 myRank 命中）
std::vector<u32> BuildRepresentativeGroup(u32 step, u32 count, u32 offset);

// 校验单个 group：规模等于 dim 且包含 myRank；失败打 ERROR 并返回 HCCL_E_INTERNAL
HcclResult ValidateGroup(const std::vector<u32>& group, u32 dim, u32 myRank, const std::string& levelName);

// 引擎过滤：非 hostdpu 排除 HOST 层，AIV 排除含 UBG 链路的层
std::vector<u32> CollectEffectiveIndices(const std::vector<PhysicalLevelInfo>& physicalLevels, OpExecuteConfig engine);

// 判断算法是否属于 Mesh 类
bool IsMeshAlgo(AlgoType algo);

// 判断算法是否属于 MeshConcur 类（触发 CLOS 双层规则）
bool IsMeshConcurAlgo(AlgoType algo);

// 锚点匹配：hostdpu 强约束最高算法层锚定 HOST 且 localRanks==userRankSize 的物理层；Mesh 算法优先匹配 COMM_TOPO_1DMESH
// 物理层（不可重复锚定）
HcclResult FindAnchors(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx,
    const std::vector<AlgoType>& algoTypes, OpExecuteConfig engine, u32 userRankSize, std::map<u32, u32>& anchors);

// 分段压缩得各算法层对应的物理层 effIdx position
HcclResult ResolveSegmentMapping(
    const std::vector<u32>& effIdx, const std::vector<AlgoType>& algoTypes, const std::map<u32, u32>& anchors,
    std::vector<u32>& pIndices);

// 引擎过滤 + 锚点匹配 + 分段，得 effIdx 与 pIndices；校验最高层 localRanks==userRankSize
HcclResult ResolveMapping(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const AlgAttrs& profile, u32 userRankSize,
    std::vector<u32>& effIdx, std::vector<u32>& pIndices);

// 在 meshEffPos 之上找首个 localRanks 包含 mesh 层 localRanks 的物理层
int32_t FindUpperEncompassingLevel(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx, u32 meshEffPos);

// 填充 physicalIdxForAlgoLevels（二级）：MeshConcur 层记 {Mesh层, 上层超集层}，普通层记 {该层}
HcclResult FillPhysicalIdxForAlgoLevels(
    const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx,
    const std::vector<u32>& pIndices, const std::vector<AlgoType>& algoTypes,
    std::vector<std::vector<PhysicalLevelIndex>>& physicalIdxForAlgoLevels);

// 将 physicalIdxForAlgoLevels 拼成日志字符串，每层用 {} 包裹，层内多值逗号分隔，如 "{0,1} {2} {3}"
std::string
FormatPhysicalIdxForAlgoLevels(const std::vector<std::vector<PhysicalLevelIndex>>& physicalIdxForAlgoLevels);

// V2 基类：MatchTopo 增加 AlgAttrs 参数
class TopoMatchBaseV2 {
public:
    explicit TopoMatchBaseV2();
    virtual ~TopoMatchBaseV2();

    virtual std::string Describe() const = 0;

    virtual HcclResult MatchTopo(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& profile)
        = 0;
};

} // namespace ops_hccl

#endif // !TOPO_MATCH_BASE_V2
