/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_TOPO_MODEL_H
#define SIM_TOPO_MODEL_H

#include "hccl_sim_pub.h"
#include "hccl_common.h"
#include "hccl_rank_graph.h"
#include "hccl_rankgraph.h"

namespace HcclSim {
class TopoModel {
public:
    TopoModel() = delete;
    TopoModel(std::vector<GraphRankInfo> rankGraphs) : rankGraphs_(rankGraphs) {}
    ~TopoModel() = default;
    uint32_t GetRankSize() const;

public:
    std::vector<GraphRankInfo> rankGraphs_;
}; // TopoModel

};

#endif  // SIM_TOPO_MODEL_H