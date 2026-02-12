/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_MATCH_BASE
#define TOPO_MATCH_BASE

#include <unordered_set>
#include <algorithm>

#include "log.h"
#include "alg_param.h"

namespace ops_hccl {
constexpr int RANK_SIZE_TWO   = 2;
constexpr int RANK_SIZE_THREE = 3;
constexpr int RANK_SIZE_FOUR  = 4;
constexpr int RANK_SIZE_EIGHT = 8;

constexpr int COMM_LAYER_SIZE_0 = 0;
constexpr int COMM_LAYER_SIZE_1 = 1;
constexpr int COMM_LAYER_SIZE_2 = 2;
constexpr int COMM_LAYER_SIZE_3 = 3;

constexpr int NET_INST_NUM_0 = 0;
constexpr int NET_INST_NUM_1 = 1;
constexpr int NET_INST_NUM_2 = 2;

const std::vector<std::vector<u32>> SERVER_910A_4_RING_SEQUENCE
    = {{0, 1, 2, 6, 5, 4, 7, 3}, {0, 3, 7, 4, 5, 6, 2, 1}, {0, 2, 3, 1, 5, 7, 6, 4}, {0, 4, 6, 7, 5, 1, 3, 2}};

struct Hccl910AServerValid4PRanksVectorHashFuc {
    std::size_t operator()(const std::vector<s32> key) const
    {
        size_t ret = 0;
        for (auto it : key) {
            ret ^= static_cast<u32>(it);
        }
        return ret;
    }
};

const std::unordered_set<std::vector<s32>, Hccl910AServerValid4PRanksVectorHashFuc> SERVER_910A_VALID_4P_RANKS
    = {{0, 1, 4, 5}, {0, 2, 4, 6}, {0, 3, 4, 7}, {1, 2, 5, 6}, {1, 3, 5, 7}, {2, 3, 6, 7}, {0, 1, 2, 3}, {4, 5, 6, 7}};

const std::vector<u32> SERVER_910A_4P_SEQUENCE = {0, 1, 3, 2};

class TopoMatchBase {
public:
    explicit TopoMatchBase();
    virtual ~TopoMatchBase();

    virtual std::string Describe() const = 0;

    virtual HcclResult MatchTopo(const HcclComm comm, TopoInfo* topoInfo,
                                 AlgHierarchyInfoForAllLevel& algHierarchyInfo);
};

} // namespace ops_hccl

#endif // !HCCLV2_TOPO_MATCH_BASE