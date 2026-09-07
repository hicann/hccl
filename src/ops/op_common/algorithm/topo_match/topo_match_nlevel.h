/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_NLEVEL
#define TOPO_MATCH_NLEVEL

#include "topo_match_base.h"

namespace ops_hccl {

constexpr u32 MAX_TOPO_LEVEL_NUM = 4;

class TopoMatchNLevel : public TopoMatchBase {
public:
    explicit TopoMatchNLevel();
    ~TopoMatchNLevel() override;

    std::string Describe() const override { return "Topo Match for N-level hierarchical Algorithm."; }

    HcclResult MatchTopo(
        const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

protected:
    HcclResult TopoForLayer0(
        const HcclComm comm, uint32_t& layer0Size, const uint32_t myRank,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) const;
    HcclResult TopoForLayerGeneric(
        const HcclComm comm, uint32_t netLayer, uint32_t baseModSize, const uint32_t myRank,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo, uint32_t targetLayerIdx) const;
    bool CheckVecElementAllSame(const uint32_t* instSizeList, uint32_t listSize) const;

    template <typename T>
    std::string PrintCArray(const T* values, const u32 valueNum) const
    {
        std::ostringstream oss;
        for (u32 i = 0; i < valueNum; i++) {
            oss << values[i] << " ";
        }
        return oss.str();
    }
};
} // namespace ops_hccl

#endif // !TOPO_MATCH_NLEVEL
