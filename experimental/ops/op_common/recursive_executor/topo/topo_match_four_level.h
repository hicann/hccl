/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_FOUR_LEVEL_H
#define TOPO_MATCH_FOUR_LEVEL_H

#include "topo_match_base_v2.h"

namespace ops_hccl {

constexpr u32 TOPO_LEVEL_NUM_4 = 4;

class TopoMatchFourLevel : public TopoMatchBaseV2 {
public:
    explicit TopoMatchFourLevel();
    ~TopoMatchFourLevel() override;

    std::string Describe() const override
    {
        return "TopoMatchFourLevel: 4-level topo match (layer 0 Mesh, layer 1/2/3 NHR).";
    }

    HcclResult MatchTopo(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& profile) override;
};

} // namespace ops_hccl

#endif // !TOPO_MATCH_FOUR_LEVEL_H
