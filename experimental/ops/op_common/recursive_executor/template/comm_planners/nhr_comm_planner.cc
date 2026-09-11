/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nhr_comm_planner.h"

namespace ops_hccl {

HcclResult RunNhrAllGather(
    const DataParams& tempAlgParams, const std::vector<u32>& ranks, u32 myRank, std::vector<u32>& ranksForOutputData,
    std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>* lastStepRxRanks)
{
    return HCCL_SUCCESS;
}

bool CanReadLastStepToOutput(const DataParams& tempAlgParams) { return false; }

} // namespace ops_hccl
