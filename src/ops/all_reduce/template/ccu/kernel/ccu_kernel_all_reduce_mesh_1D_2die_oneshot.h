/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_2DIE_ONE_SHOT_H
#define HCCL_CCU_KERNEL_ALLREDUCE_MESH_1D_2DIE_ONE_SHOT_H

#include <vector>
#include <ios>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

struct CcuKernelArgAllreduceMesh1D2DieOneShot : CcuKernelArgBase {
    uint64_t rankSize{0};
    uint32_t rankId{0};
    OpParam opParam;
    std::vector<std::vector<uint32_t>> subCommRanks;
    bool rmtReduceWithMyRank{false};
};

struct AllreduceMesh1D2DieOneShotContext : CcuKernelCtxBase {
    const CcuKernelArgAllreduceMesh1D2DieOneShot *arg;

    bool rmtReduceWithMyRank{false};
    uint32_t rankId{0};
    uint32_t rankSize{0};

    uint32_t rmtReduceRankNum{0};
    uint32_t rmtSyncMyBit{0};
    uint32_t rmtSyncWaitBit{0};

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    uint32_t missionSyncMybit{0};
    uint32_t missionSyncWaitBit{0};

    uint16_t selfBit{0};
    uint16_t allBit{0};

    ccu::Variable myInput;
    ccu::Variable myOutput;
    ccu::Variable myScratch;
    ccu::Variable myToken;

    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> remoteToken;

    ccu::Variable scratchBaseOffset0;
    ccu::Variable scratchBaseOffset1;

    ccu::Variable localReduceSliceOffset0;
    ccu::Variable localReduceSliceOffset1;

    GroupOpSizeVars rmtReduceGoSize;
    GroupOpSizeVars localReduceGoSize0;
    GroupOpSizeVars localReduceGoSize1;
};

CcuResult CcuAllreduceMesh1D2DieOneShotKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCLV2_CCU_KERNEL_ALLREDUCE_MESH_1D_2DIE_ONE_SHOT_H
