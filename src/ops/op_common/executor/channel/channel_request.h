/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_CHANNEL_REQUEST
#define OPS_HCCL_SRC_OPS_CHANNEL_REQUEST

#include <set>
#include "hccl/base.h"

namespace ops_hccl {

constexpr u32 HCCL_RANK_SIZE_EQ_ONE = 1;

HcclResult CalcRingChannelConnect(u32 rank, u32 rankSize, u32 root, std::set<u32> &connectRanks);
HcclResult CalcMeshChannelConnect(u32 rank, u32 rankSize, u32 root, std::set<u32> &connectRanks);
HcclResult CalcNHRChannelConnect(u32 rank, u32 rankSize, u32 root, std::set<u32> &connectRanks);
HcclResult CalcNBChannelConnect(u32 rank, u32 rankSize, u32 root, std::set<u32> &connectRanks);
HcclResult CalcMesh2DChannelConnect(u32 myRank, const std::vector<std::vector<u32>> &tempVTopo, std::set<u32> &connectRanks);
}


#endif