/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TOPO_MATCH_THREE_LEVEL_H
#define HCCLV2_TOPO_MATCH_THREE_LEVEL_H

#include "topo_match_base_v2.h"

namespace ops_hccl {

class TopoMatchThreeLevel : public TopoMatchBaseV2 {
public:
    explicit TopoMatchThreeLevel();
    ~TopoMatchThreeLevel() override;

    std::string Describe() const override { return "Topo Match for Three Level Algorithm."; }

    HcclResult MatchTopo(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& profile) override;
};

} // namespace ops_hccl

#endif // !HCCLV2_TOPO_MATCH_THREE_LEVEL_H
