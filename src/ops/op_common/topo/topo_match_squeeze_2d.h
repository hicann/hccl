 /**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_SQUEEZE_2D
#define TOPO_MATCH_SQUEEZE_2D

#include "topo_match_base.h"

namespace ops_hccl {
class TopoMatchSqueeze2D : public TopoMatchBase {
public:
    explicit TopoMatchSqueeze2D();
    ~TopoMatchSqueeze2D() override;

    std::string Describe() const override
    {
        return "Topo Match for Squeeze2d Algorithm: squeeze layer0+layer1 into algHierarchyInfo[0], layer2 into algHierarchyInfo[1].";
    }
    HcclResult MatchTopo(const HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;

private:
    HcclResult TopoForLayer0(const HcclComm comm, uint32_t& combinedSize, const uint32_t myRank,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) const;
    HcclResult TopoForLayer1(const HcclComm comm, uint32_t baseModSize, const uint32_t myRank,
        AlgHierarchyInfoForAllLevel& algHierarchyInfo) const;
    bool CheckVecElementAllSame(const uint32_t* instSizeList, uint32_t listSize) const;

    template<typename T>
    std::string PrintCArray(const T* values, const u32 valueNum) const
    {
        std::ostringstream oss;
        for (u32 i = 0; i < valueNum; i++) {
            oss << values[i] << " ";
        }
        return oss.str();
    }
};
}  // namespace ops_hccl

#endif  // !TOPO_MATCH_SQUEEZE_2D