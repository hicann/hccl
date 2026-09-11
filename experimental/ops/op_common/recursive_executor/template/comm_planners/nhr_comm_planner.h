/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RE_NHR_COMM_PLANNER_H
#define RE_NHR_COMM_PLANNER_H

#include <vector>
#include "alg_param.h"
#include "data_types.h"

namespace ops_hccl {

struct DataParams;

struct NhrAllGatherSlicePair {
    void* srcPtr;
    void* dstPtr;
    std::vector<DataSlice>& srcSlices;
    std::vector<DataSlice>& dstSlices;
};

// lastStepRxRanks 输出末步 rx 收到的 rankIds（末步才到位，PostCopy 不能提前搬）
HcclResult RunNhrAllGather(
    const DataParams& tempAlgParams, const std::vector<u32>& ranks, u32 myRank, std::vector<u32>& ranksForOutputData,
    std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>* lastStepRxRanks = nullptr);

bool CanReadLastStepToOutput(const DataParams& tempAlgParams);

} // namespace ops_hccl

#endif
