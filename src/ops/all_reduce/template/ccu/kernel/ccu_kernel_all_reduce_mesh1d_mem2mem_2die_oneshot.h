/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT
#define HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT
 
#include <vector>
#include <ios>
#include <string>
#include "utils.h"
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"
 
namespace ops_hccl {

struct CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot : CcuKernelArgBase {
    uint64_t rankSize{0};
    uint32_t rankId{0};
    OpParam opParam;
    std::vector<uint32_t> kernelRanks;
    std::vector<std::vector<uint32_t>> subCommRanks;
    bool rmtReduceWithMyRank{false};
};
 
struct AllReduceMesh1DMem2Mem2DieOneShotContext : CcuKernelCtxBase {
    const CcuKernelArgAllReduceMesh1DMem2Mem2DieOneShot *arg;

    uint64_t rankSize{0};
    uint32_t rankId{0};

    bool rmtReduceWithMyRank{false};
    uint32_t rmtReduceRankNum{0};

    uint32_t missionSyncMybit{0};
    uint32_t missionSyncWaitBit{0};

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    ccu::Variable myInput;
    ccu::Variable myOutput;
    ccu::Variable myScratch;
    ccu::Variable myToken;
    std::vector<ccu::Variable> peerInput;
    std::vector<ccu::Variable> peerToken;

    ccu::Variable scratchBaseOffset0;
    ccu::Variable scratchBaseOffset1;
    ccu::Variable normalSliceSize;
    ccu::Variable localReduceSliceOffset0;
    ccu::Variable localReduceSliceOffset1;

    GroupOpSizeVars localReduceGoSize;
    GroupOpSizeVars localReduceGoSize0;
    GroupOpSizeVars localReduceGoSize1;

    ccu::Event event;
};

CcuResult CcuAllReduceMesh1DMem2Mem2DieOneShotKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_ALL_REDUCE_MESH_1D_MEM2MEM_2DIE_ONESHOT