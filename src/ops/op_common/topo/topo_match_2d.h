/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_CONCURR_MESH
#define TOPO_MATCH_CONCURR_MESH

#include "topo_match_base.h"

namespace ops_hccl {

class TopoMatch2D : public TopoMatchBase {
public:
    explicit TopoMatch2D();
    ~TopoMatch2D() override;

    std::string Describe() const override
    {
        return "Topo Match for Multi-dimensional Concurrent Mesh Algorithm (CURRENTLY only 2-D Concurr Mesh is "
               "supported).";
    }
    HcclResult MatchTopo(const HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo) override;
};
} // namespace ops_hccl

#endif // !TOPO_MATCH_CONCURR_MESH