/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_DUO_MEM2MEM_H
#define HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_DUO_MEM2MEM_H

#include <vector>
#include <ios>
#include "ccu_kernel_utils.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

constexpr uint64_t REDUCE_MS_CNT = 8;
constexpr uint16_t REDUCE_SCATTER_2DIE_GROUP_REDUCE_MAX_PIECE_CNT = 8;

constexpr int INPUT_XN_ID        = 0;
constexpr int SCRATCH_XN_ID      = 1;
constexpr int TOKEN_XN_ID        = 2;
constexpr int POST_SYNC_ID       = 3;
constexpr int CKE_IDX_0          = 0;

struct CcuKernelArgReduceScatterMesh1D2DieMem2Mem : CcuKernelArgBase {
    uint32_t                                gRankSize;
    uint32_t                                rankSize;
    uint32_t                                rankId;
    bool                                    isReduceToOutput;
    OpParam                                 opParam;
    std::vector<uint32_t>                   subRankGroup;
    std::vector<std::vector<uint32_t>>      subCommRanks;
};

struct ReduceScatterMesh1D2DieMem2MemContext : CcuKernelCtxBase {
    const CcuKernelArgReduceScatterMesh1D2DieMem2Mem *arg;

    HcclDataType dataType;
    HcclDataType outputDataType;
    HcclReduceOp reduceOp;

    uint64_t rankSize;
    uint64_t gRankSize;
    uint32_t rankId;
    std::vector<uint32_t> subRankGroup;
    bool isReduceToOutput;

    std::vector<ccu::Variable> input;
    ccu::Variable output;
    std::vector<ccu::Variable> scratch;
    std::vector<ccu::Variable> token;
    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable sliceSize;
    ccu::Variable repeatNum;
    ccu::Variable flag;
    GroupOpSizeVars sliceGoSize;

    ccu::LocalAddr myInput;
    std::vector<ccu::RemoteAddr> remoteInput;
    std::vector<ccu::LocalAddr> scratchMem;
    ccu::LocalAddr outputTmp;
    ccu::Event event;

    uint32_t myRankIdx;

    // Loop机制相关变量
    std::array<std::vector<ccu::LocalAddr>, 2> loopScratch;
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopDst[2];
    ccu::Variable  loopLen[2];
    ccu::Variable  loopLenExp[2];
};

CcuResult CcuReduceScatterMesh1D2DieMem2MemKernel(CcuKernelArg arg);

} // namespace ops_hccl
#endif // HCCL_CCU_KERNEL_REDUCE_SCATTER_MESH_1D_DUO_MEM2MEM_H