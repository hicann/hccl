/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_2Die_H_
#define HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_2Die_H_

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;

struct CcuKernelArgReduceScatterMesh2Die : public CcuKernelArgBase {
    uint64_t rankSize{0};
    uint32_t rankId{0};
    OpParam opParam;
    std::vector<std::vector<RankId>> subCommRanks;
    bool rmtReduceWithMyRank{true};
};

struct ReduceScatterMesh2DieContext : public CcuKernelCtxBase {
    const CcuKernelArgReduceScatterMesh2Die *arg;

    bool rmtReduceWithMyRank{true};
    uint32_t myRankId{0};
    uint32_t rankSize{0};
    uint32_t rmtReduceRankNum{0};
    uint32_t rmtSyncMyBit{0};
    uint32_t rmtSyncWaitBit{0};

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    ccu::Variable myInput;
    ccu::Variable myOutput;
    ccu::Variable myScratch;
    ccu::Variable myToken;
    std::vector<ccu::Variable> peerInput;
    std::vector<ccu::Variable> peerToken;

    ccu::Variable sliceSize;
    ccu::Variable rmtReduceSliceOffset;
    GroupOpSizeVars rmtReduceGoSize;
};

CcuResult CcuReduceScatterMesh2DieKernel(CcuKernelArg arg);

}// namespace ops_hccl

#endif // HCCLV2_CCU_KERNEL_REDUCE_SCATTER_MESH_2Die_H_