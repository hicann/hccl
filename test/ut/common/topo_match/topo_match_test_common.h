/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_TEST_COMMON_H
#define TOPO_MATCH_TEST_COMMON_H

#include <vector>
#include <string>
#include "alg_param.h"

namespace ops_hccl {
namespace ut_helper {

    inline std::vector<u32> Range(u32 n)
    {
        std::vector<u32> v;
        v.reserve(n);
        for (u32 i = 0; i < n; i++) {
            v.push_back(i);
        }
        return v;
    }

    // 从 base 开始的 count 个连续整数：{base, base+1, ..., base+count-1}
    // 用于构造 myRank 所在 instance 的 localRanks（localRanks 必须含 myRank）
    inline std::vector<u32> RangeFrom(u32 base, u32 count)
    {
        std::vector<u32> v;
        v.reserve(count);
        for (u32 i = 0; i < count; i++) {
            v.push_back(base + i);
        }
        return v;
    }

    // 构造一个物理层。ranks=localRanks(升序去重)；GLOBAL 层填 instSizeListByLayer，LOCAL 层恒空。
    inline PhysicalLevelInfo MakeLevel(
        std::vector<u32> ranks, PhysicalLevelView view, std::vector<u32> instList = {}, bool hasTopoInst = true,
        CommTopo topoType = COMM_TOPO_CLOS, EndpointLocType locType = ENDPOINT_LOC_TYPE_DEVICE,
        std::vector<CommProtocol> protocols = {})
    {
        PhysicalLevelInfo lvl;
        lvl.localRanks = std::move(ranks);
        lvl.view = view;
        lvl.instSizeListByLayer = std::move(instList);
        lvl.hasTopoInst = hasTopoInst;
        lvl.topoType = topoType;
        lvl.locType = locType;
        lvl.protocols = std::move(protocols);
        return lvl;
    }

    inline AlgAttrs MakeProfile(std::vector<AlgoType> algoTypes, OpExecuteConfig engine = OpExecuteConfig::AICPU_TS)
    {
        AlgAttrs p;
        p.name = "ut";
        p.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        p.engine = engine;
        p.algoTypes = std::move(algoTypes);
        return p;
    }

    inline TopoInfoWithNetLayerDetails MakeTopoInfo(u32 myRank, u32 userRankSize, std::vector<PhysicalLevelInfo> levels)
    {
        TopoInfoWithNetLayerDetails t;
        t.userRank = myRank;
        t.userRankSize = userRankSize;
        t.physicalLevels = std::move(levels);
        t.physicalLevelNum = static_cast<u32>(t.physicalLevels.size());
        return t;
    }

} // namespace ut_helper
} // namespace ops_hccl

#endif // TOPO_MATCH_TEST_COMMON_H
