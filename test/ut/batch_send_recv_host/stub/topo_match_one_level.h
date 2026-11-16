/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_TOPO_MATCH_ONE_LEVEL_H
#define BSR_HOST_TEST_TOPO_MATCH_ONE_LEVEL_H

// 拓扑匹配桩：把用例预置的算法分层信息原样交给 executor，使 CalcRes / Orchestrate 拿到可控的
// subCommRanks（= infos[0]）与 myRank_ / rankSize_。

#include <string>

#include "alg_param.h"

namespace ops_hccl {
namespace bsr_host_test {
    inline AlgHierarchyInfoForAllLevel matchedHierarchyInfo;
    inline HcclResult topoMatchResult = HcclResult::HCCL_SUCCESS;
} // namespace bsr_host_test

class TopoMatchOneLevel {
public:
    TopoMatchOneLevel() = default;
    ~TopoMatchOneLevel() = default;

    std::string Describe() const { return "Topo Match stub."; }

    HcclResult MatchTopo(
        TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
    {
        (void)topoInfo;
        (void)algAttrs;
        algHierarchyInfo = bsr_host_test::matchedHierarchyInfo;
        return bsr_host_test::topoMatchResult;
    }
};

} // namespace ops_hccl

#endif // BSR_HOST_TEST_TOPO_MATCH_ONE_LEVEL_H
