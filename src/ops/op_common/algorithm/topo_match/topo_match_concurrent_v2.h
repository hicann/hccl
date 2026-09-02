/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_CONCURRENT_V2
#define TOPO_MATCH_CONCURRENT_V2

#include "topo_match_base_v2.h"

namespace ops_hccl {

class TopoMatchConcurrentV2 : public TopoMatchBaseV2 {
public:
    explicit TopoMatchConcurrentV2();
    ~TopoMatchConcurrentV2() override;

    std::string Describe() const override
    {
        return "Topo Match for Concurrent Algorithm V2 (supports 950/960 out-place devices).";
    }

    HcclResult MatchTopo(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo,
        const AlgAttrs& profile) override;
};

} // namespace ops_hccl

#endif // !TOPO_MATCH_CONCURRENT_V2
