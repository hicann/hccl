/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_KERNEL_ALL_TO_ALL_MESH_2DIE_H_
#define HCCLV2_CCU_KERNEL_ALL_TO_ALL_MESH_2DIE_H_

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;

struct CcuKernelArgAllToAllMesh2Die : CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    OpParam opParam;
    std::vector<std::vector<RankId>> subCommRanks;
    bool withMyRank;
    std::vector<RankId> rankGroup;
};

struct AllToAllMesh2DieContext : CcuKernelCtxBase {
    const CcuKernelArgAllToAllMesh2Die *arg;

    uint16_t virRankSize{0};
    uint64_t logicRankSize{0};
    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable sliceSize;
    ccu::Variable inputSliceStride;
    ccu::Variable outputoffset;
    GroupOpSizeVars groupOpSize;

    ccu::Event event;
};

CcuResult CcuAllToAllMesh2DieKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // HCCLV2_CCU_KERNEL_ALL_TO_ALL_MESH_2DIE_H_
